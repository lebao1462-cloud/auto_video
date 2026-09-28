"""Offline FunASR Paraformer transcription for Chinese localization."""

import gc
import importlib.util
import os
from pathlib import Path
import tempfile
from typing import Any
import unicodedata

from buzz.localization.audio_mix import extract_audio_track
from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript

PARAFORMER_ENGINE = "paraformer-zh"
_MODEL_DIRECTORIES = {
    "model": "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "iic--punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}


def modelscope_cache_path() -> Path:
    """Return the cache location without importing ModelScope/FunASR."""
    configured = os.getenv("MODELSCOPE_CACHE")
    if configured:
        return Path(configured)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/Dev/modelscope-cache")
    return Path.home() / ".cache" / "modelscope"


def configure_modelscope_cache() -> Path:
    path = modelscope_cache_path()
    os.environ.setdefault("MODELSCOPE_CACHE", str(path))
    return path


def paraformer_runtime_available() -> bool:
    return importlib.util.find_spec("funasr") is not None


def paraformer_model_is_cached(cache: Path | None = None) -> bool:
    root = cache or modelscope_cache_path()
    if not root.exists():
        return False
    # ModelScope names vary by release; this detects the existing snapshot
    # without importing it or triggering a download.
    return all(_cached_model_snapshot(root, name) is not None for name in _MODEL_DIRECTORIES)


def _cached_model_snapshot(cache: Path, name: str) -> Path | None:
    """Find the ModelScope snapshot without importing ModelScope or using network."""
    model_root = cache / "models" / _MODEL_DIRECTORIES[name] / "snapshots"
    if not model_root.is_dir():
        return None
    master = model_root / "master"
    if master.is_dir():
        return master
    return next((item for item in model_root.iterdir() if item.is_dir()), None)


def paraformer_model_paths(cache: Path | None = None) -> dict[str, str] | None:
    """Return local snapshots when all Paraformer assets are already cached."""
    root = cache or modelscope_cache_path()
    paths = {name: _cached_model_snapshot(root, name) for name in _MODEL_DIRECTORIES}
    if any(path is None for path in paths.values()):
        return None
    return {name: str(path) for name, path in paths.items()}


def _temporary_directory() -> Path:
    """Keep large PCM intermediates off C: on Windows when D: is available."""
    for name in ("TMP", "TEMP"):
        value = os.getenv(name)
        if value and (os.name != "nt" or Path(value).drive.upper() == "D:"):
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            return path
    if os.name == "nt" and Path("D:/").exists():
        path = Path("D:/Temp")
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.gettempdir())


def _token_items(result: dict[str, Any]) -> list[tuple[str, float, float]]:
    timestamps = result.get("timestamp") or []
    raw_text = result.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        tokens = raw_text.split()
    else:
        # Explicit tokens are only a test/mock compatibility path. Do not
        # derive tokens from text or sentence_info: neither is authoritative.
        tokens = result.get("tokens") or result.get("token") or []
        if isinstance(tokens, str):
            tokens = tokens.split()
    if not isinstance(timestamps, (list, tuple)) or not isinstance(tokens, (list, tuple)):
        raise ValueError("Paraformer requires an exact token/timestamp mapping.")
    if len(tokens) != len(timestamps):
        raise ValueError("Paraformer token/timestamp count mismatch; refusing to misassign timestamps.")
    if not tokens:
        raise ValueError("Paraformer returned no timestamped raw tokens.")
    output: list[tuple[str, float, float]] = []
    previous_end = 0.0
    for index, (token, stamp) in enumerate(zip(tokens, timestamps)):
        if not isinstance(stamp, (list, tuple)) or len(stamp) < 2:
            raise ValueError(f"Paraformer timestamp {index} is not a timestamp pair.")
        try:
            start, end = float(stamp[0]) / 1000, float(stamp[1]) / 1000
        except (TypeError, ValueError):
            raise ValueError(f"Paraformer timestamp {index} is invalid.") from None
        if not isinstance(token, str) or not token:
            raise ValueError(f"Paraformer token {index} is invalid.")
        if start < previous_end or end < start:
            raise ValueError(f"Paraformer timestamps are not monotonic at token {index}.")
        output.append((token, start, end))
        previous_end = end
    return output


def _semantic_text(value: str) -> str:
    return "".join(char for char in value if not char.isspace() and not unicodedata.category(char).startswith("P"))


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _segments_from_tokens(result: dict[str, Any]) -> list[LocalizationSegment]:
    tokens = _token_items(result)
    text = str(result.get("text") or "").strip()
    if not tokens:
        return []
    # Preserve the recognized punctuated text. Match its non-punctuation
    # characters to raw token timing; punctuation and pauses are boundaries.
    units: list[tuple[str, float, float]] = []
    for token, start, end in tokens:
        units.append((token, start, end))
    raw = "".join(token for token, _, _ in units)
    punctuated = text or raw
    # Alignment is deliberately conservative. If normalized characters do not
    # agree, use raw tokens rather than inventing transcript text.
    if _semantic_text(punctuated) != _semantic_text(raw):
        punctuated = raw
    display_tokens = [token for token, _, _ in units]
    if punctuated != raw:
        # Map semantic-character boundaries to whole timestamp tokens. A
        # punctuation mark belongs to the token ending at that boundary, so
        # multi-character tokens such as ``wifi`` and ``APP`` stay intact.
        token_at_boundary: dict[int, int] = {}
        consumed = 0
        for index, (token, _, _) in enumerate(units):
            consumed += len(_semantic_text(token))
            token_at_boundary[consumed] = index

        consumed = 0
        for char in punctuated:
            if char.isspace():
                continue
            if _is_punctuation(char):
                token_index = token_at_boundary.get(consumed)
                if token_index is not None:
                    display_tokens[token_index] += char
                continue
            consumed += 1
    boundaries: set[int] = set()
    raw_index = 0
    for char in punctuated:
        if _is_punctuation(char):
            boundaries.add(raw_index)
        elif not char.isspace():
            raw_index += 1
    # Token timestamps may cover a Chinese word rather than one character.
    # Convert character boundaries to whole-token boundaries so Latin tokens
    # (and all recognized tokens) are never cut apart.
    token_boundaries: set[int] = set()
    consumed = 0
    for index, (token, _, _) in enumerate(units, 1):
        consumed += len(_semantic_text(token))
        if consumed in boundaries:
            token_boundaries.add(index)
    segments: list[LocalizationSegment] = []
    start_index = 0
    for index in range(1, len(units) + 1):
        duration = units[index - 1][2] - units[start_index][1]
        pause = index < len(units) and units[index][1] - units[index - 1][2] >= 0.45
        boundary = index in token_boundaries
        # Prefer punctuation/pause around natural subtitle lengths; retain a
        # long sentence until a safe boundary rather than splitting a token.
        if index == len(units) or (duration >= 1 and (boundary or pause)) or duration >= 8:
            chunk = units[start_index:index]
            value = "".join(display_tokens[start_index:index]).strip()
            if value:
                segments.append(LocalizationSegment(chunk[0][1], chunk[-1][2], value))
            start_index = index
    return segments


def transcribe_with_paraformer(video_path: str) -> LocalizationTranscript:
    source = Path(video_path)
    if not source.is_file():
        raise ValueError(f"Input media file does not exist: {video_path}")
    if not paraformer_runtime_available():
        raise RuntimeError("FunASR runtime is missing. Install funasr==1.4.16 or choose Whisper.")
    configure_modelscope_cache()  # must precede the lazy FunASR import
    temp_dir = _temporary_directory()
    wav = temp_dir / f"autovideo-paraformer-{next(tempfile._get_candidate_names())}.wav"
    model = None
    try:
        extract_audio_track(source, wav, sample_rate=16000, channels=1)
        from funasr import AutoModel

        # Passing the actual snapshots avoids ModelScope lookups when the
        # requested models are present. Aliases retain FunASR's one-time
        # download behavior on a fresh installation.
        local_models = paraformer_model_paths()
        model = AutoModel(
            model=(local_models or {}).get("model", PARAFORMER_ENGINE),
            vad_model=(local_models or {}).get("vad", "fsmn-vad"),
            punc_model=(local_models or {}).get("punc", "ct-punc-c"),
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
            ncpu=min(4, max(1, os.cpu_count() or 1)),
            disable_update=True,
            disable_pbar=True,
        )
        generated = model.generate(
            input=str(wav), sentence_timestamp=True, return_raw_text=True,
            batch_size_s=60, batch_size_threshold_s=30,
        )
        result = generated[0] if isinstance(generated, list) and generated else generated
        if not isinstance(result, dict):
            raise RuntimeError("FunASR returned an invalid transcription result.")
        segments = _segments_from_tokens(result)
        if not segments or any(not s.text or s.end < s.start for s in segments):
            raise RuntimeError("Paraformer produced no usable timestamped Chinese transcript.")
        return LocalizationTranscript("zh", str(source), tuple(segments))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Unable to load or run Paraformer. Check its cached models and runtime: {exc}") from exc
    finally:
        wav.unlink(missing_ok=True)
        del model
        gc.collect()
