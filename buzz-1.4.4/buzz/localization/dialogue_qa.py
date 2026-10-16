"""Checkpointed, bounded Gemini dialogue/time-slot QA before TTS.

QA must never retime/swap subtitle IDs, drop text, or mutate production
translation checkpoints. This is a single semantic review, not a TTS fit loop.
Real synthesized duration remains the authority for timing recovery.
"""
from dataclasses import replace
import hashlib
import json
import math
import os
import re
from pathlib import Path
import tempfile
from typing import Callable

from buzz.localization.translation import (
    TranslatedLocalizationTranscript,
    _has_prohibited_dialogue_pronoun, _valid_reviewed_text,
)

QA_POLICY_VERSION = "dialogue-fidelity-timing-qa-v2-register-safe"
QA_BATCH_SIZE = 24
QA_CONTEXT_LINES = 5
QA_WPS_TRIGGER = 5.2
QA_MAX_RETRIES = 2


class DialogueQACancelled(RuntimeError):
    """Raised only when the user cancels during Gemini dialogue QA."""


def _safe_text(text):
    return isinstance(text, str) and bool(text.strip()) and _valid_reviewed_text(text)


_ADDRESS_TERMS = frozenset({
    "anh", "em", "tôi", "mình", "cậu", "cô", "chú", "bác",
    "bố", "mẹ", "con", "chị", "ông", "bà", "ta", "ngươi", "huynh", "muội",
})


def _safe_compression(original, proposed):
    """Do not accept fluency edits that spontaneously switch character roles.

    A real relationship change needs separate source-grounded review; the
    pre-TTS compression step must not introduce new address terms or typos.
    """
    words_old = set(re.findall(r"[^\W\d_]+", original.casefold(), re.UNICODE))
    words_new = set(re.findall(r"[^\W\d_]+", proposed.casefold(), re.UNICODE))
    if "cõ" in words_new or "cõ" in proposed.casefold().split():
        return False
    return (words_new & _ADDRESS_TERMS).issubset(words_old & _ADDRESS_TERMS)


def _fingerprint(transcript, provider):
    identity = (type(provider).__module__+"."+type(provider).__qualname__)
    value = {
        "qa_policy": QA_POLICY_VERSION,
        "provider": identity,
        "model": getattr(provider, "model", None),
        "translation_policy": getattr(provider, "translation_policy_version", None),
        "source": transcript.to_dict(),
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as f:
        temporary = Path(f.name)
        json.dump(content, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_saved(path, fingerprint):
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            return {}
        existing = data.get("results", {})
        if not isinstance(existing, dict):
            return {}
        return {str(k): v for k, v in existing.items()
                if isinstance(v, str) and _safe_text(v)}
    except (OSError, ValueError, TypeError):
        return {}


def _duration_budget(segments, index):
    current = segments[index]
    following = segments[index+1].start if index+1 < len(segments) else current.end
    return max(0.15, float(following)-float(current.start))


def _needs_review(segments, index):
    item = segments[index]
    words = len(item.translated_text.split())
    available = _duration_budget(segments,index)
    if words / available > QA_WPS_TRIGGER:
        return True
    # These are common failed calques, not a global translation glossary.
    invalid_tokens = ("xu ceng", "nhảy sông tự tự", "hố cha")
    return any(token in item.translated_text.casefold() for token in invalid_tokens)


def _entry(segments, idx, *, flagged=False):
    seg = segments[idx]
    return {
        "id": f"line-{idx:06d}",
        "start": round(float(seg.start), 3),
        "end": round(float(seg.end), 3),
        "available_seconds": round(_duration_budget(segments, idx), 3),
        "source_text": seg.source_text,
        "current_vietnamese": seg.translated_text,
        "needs_fidelity_or_timing_review": bool(flagged),
    }


def auto_review_dialogue(
    transcript: TranslatedLocalizationTranscript,
    provider,
    *,
    checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event=None,
) -> TranslatedLocalizationTranscript:
    """Check dense dialogue and adjacent source fidelity once, before TTS.

    Only providers implementing review_dialogue_alignment opt in. Every
    accepted revision preserves its segment ID and timestamps. Any invalid
    or missing Gemini output retains already-reviewed original Vietnamese.
    Transport or quota errors are surfaced for resumable retries.
    """
    review = getattr(provider, "review_dialogue_alignment", None)
    if not callable(review):
        return transcript
    items = transcript.segments
    if not items:
        return transcript
    if any(not _safe_text(x.translated_text) for x in items):
        raise ValueError("Dialogue QA rejected empty or prohibited baseline translation")
    fingerprint = _fingerprint(transcript, provider)
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    saved = _load_saved(ckpt, fingerprint)
    approved = {}
    total = len(items)

    def cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise DialogueQACancelled("Dialogue QA cancelled")

    def persist():
        if ckpt is not None:
            _write_json_atomic(ckpt, {"fingerprint": fingerprint,
                                      "policy_version": QA_POLICY_VERSION,
                                      "results": approved})

    for begin in range(0, total, QA_BATCH_SIZE):
        cancelled()
        last = min(begin + QA_BATCH_SIZE, total)
        keys = [f"line-{idx:06d}" for idx in range(begin,last)]
        baseline = [items[idx].translated_text for idx in range(begin,last)]
        flagged = [idx for idx in range(begin,last) if _needs_review(items,idx)]
        if all(key in saved for key in keys):
            for key in keys:
                approved[key] = saved[key]
        elif not flagged:
            for key,vi in zip(keys,baseline):
                approved[key] = vi
            persist()
        else:
            # Context includes both preceding and succeeding source utterances;
            # this detects shifts when one ASR line ends mid-sentence.
            review_payload = tuple(_entry(items,i,flagged=(i in flagged))
                                   for i in range(begin,last))
            neighbors = tuple(_entry(items,i) for i in
                [*range(max(0,begin-QA_CONTEXT_LINES),begin),
                 *range(last,min(total,last+QA_CONTEXT_LINES))])
            seen = {}
            for attempt in range(QA_MAX_RETRIES):
                cancelled()
                # Network/quota failure intentionally propagates; retry from
                # atomic checkpoint rather than accept silently corrupted text.
                result = review(segments=review_payload, context=neighbors,
                                source_language=transcript.source_language,
                                target_language=transcript.target_language)
                if not isinstance(result, (tuple,list)):
                    continue
                candidates = {}
                malformed = False
                for entry in result:
                    if not (isinstance(entry,(list,tuple)) and len(entry)==2
                            and isinstance(entry[0],str)):
                        malformed = True
                        continue
                    ident,vi = entry
                    if ident not in keys or ident in candidates:
                        malformed = True
                        continue
                    candidates[ident] = vi
                if malformed or set(candidates) != set(keys):
                    continue
                for idx,key in enumerate(keys):
                    original = baseline[idx]
                    proposed = candidates[key]
                    absolute_index = begin + idx
                    # Unflagged text already passed source/register review: do
                    # not allow this brevity pass to paraphrase it needlessly.
                    if absolute_index not in flagged:
                        seen[key] = original
                    elif (_safe_text(proposed)
                          and _safe_compression(original, proposed)
                          and len(proposed.split()) < len(original.split())
                          and len(proposed.split()) >= max(1,
                              math.floor(len(original.split()) * 0.45))):
                        # A shorter line may still change meaning; the Gemini
                        # source alignment prompt is the semantic verifier.
                        # Stable IDs and speaker-address guard prevent common
                        # QA overcorrections before TTS.
                        seen[key] = proposed.strip()
                    else:
                        seen[key] = original
                break
            if not seen:
                # Both malformed envelopes: retain the full reviewed content,
                # never fake a corrected line, with existing TTS timing recovery.
                seen = dict(zip(keys, baseline))
            approved.update(seen)
            persist()
        if progress_callback is not None:
            progress_callback(last,total)
    reviewed = tuple(
        replace(seg,translated_text=approved[f"line-{i:06d}"])
        for i,seg in enumerate(items))
    return replace(transcript,segments=reviewed)
