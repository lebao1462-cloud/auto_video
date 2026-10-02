from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

from openai import OpenAI

from buzz.localization.audio_mix import _ffmpeg_env, _run_ffmpeg
from buzz.localization.tts import TTSRequest, TTSResult


logger = logging.getLogger(__name__)

GEMINI_MODEL_IDS = frozenset({
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
})


class LocalizationProviderError(RuntimeError):
    """Raised when a concrete localization provider fails."""


NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_MODEL_DIRECTORY_NAME = "nllb-200-distilled-600M"
NLLB_LANGUAGE_CODES = {"en": "eng_Latn", "zh": "zho_Hans", "vi": "vie_Latn"}


def nllb_model_path() -> Path:
    """Return the explicit local NLLB directory without creating or downloading it."""
    configured_root = os.getenv("BUZZ_MODEL_ROOT")
    if configured_root:
        root = Path(configured_root)
    elif os.name == "nt" and Path("D:/").exists():
        root = Path("D:/Dev/buzz-models")
    else:
        root = Path.home() / ".cache" / "buzz-models"
    return root / "nllb" / NLLB_MODEL_DIRECTORY_NAME


def nllb_model_is_available() -> bool:
    """Cheap local-only check used by preflight; it never contacts Hugging Face."""
    path = nllb_model_path()
    has_weights = any(
        (path / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            shard_names = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        has_weights = bool(shard_names) and all(
            (path / shard_name).is_file() for shard_name in shard_names
        )
        if has_weights:
            break
    has_tokenizer = (path / "tokenizer_config.json").is_file() and any(
        (path / filename).is_file()
        for filename in ("tokenizer.json", "sentencepiece.bpe.model")
    )
    return path.is_dir() and (path / "config.json").is_file() and has_weights and has_tokenizer


def normalize_gemini_model_name(model: str) -> str:
    """Accept Gemini display names and convert them to API model IDs."""
    value = model.strip()
    if not value:
        return value
    if value.startswith("models/"):
        prefix = "models/"
        value = value[len(prefix):]
    else:
        prefix = ""
    value = value.lower().replace("_", "-").replace(" ", "-")
    while "--" in value:
        value = value.replace("--", "-")
    return prefix + value


@dataclass
class OpenAICompatibleTranslationProvider:
    api_key: str
    model: str
    base_url: str | None = None
    timeout: float = 60.0

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Translation API key is required")
        if not self.model:
            raise ValueError("Translation model is required")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
            max_retries=0,
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if target_language != "vi":
            raise ValueError("This provider is configured for Vietnamese translation")
        if source_language not in {"en", "zh"}:
            raise ValueError("Source language must be 'en' or 'zh'")

        source_name = "English" if source_language == "en" else "Chinese"
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate {source_name} into natural Vietnamese. "
                            "Return only the Vietnamese translation. Preserve meaning, "
                            "names, numbers, punctuation intent, and speaking style. "
                            "Do not add explanations or quotation marks."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                timeout=self.timeout,
            )
        except Exception as exc:
            raise LocalizationProviderError(
                f"Translation provider request failed: {exc}"
            ) from exc

        if (
            not response.choices
            or response.choices[0].message is None
            or not response.choices[0].message.content
        ):
            raise LocalizationProviderError("Translation provider returned no text")

        translated = response.choices[0].message.content.strip()
        if not translated:
            raise LocalizationProviderError("Translation provider returned empty text")
        return translated


def _run_argos_worker(
    command: str,
    source_language: str,
    target_language: str = "vi",
    text: str | None = None,
) -> dict:
    import json
    import os
    import sys
    import tempfile

    result_handle = tempfile.NamedTemporaryFile(
        prefix="buzz-argos-result-",
        suffix=".json",
        delete=False,
    )
    result_file = Path(result_handle.name)
    result_handle.close()
    result_file.unlink(missing_ok=True)

    if getattr(sys, "frozen", False):
        worker_command = [
            sys.executable,
            "--argos-worker",
            command,
            source_language,
            target_language,
            str(result_file),
        ]
    else:
        main_py = Path(__file__).resolve().parents[2] / "main.py"
        worker_command = [
            sys.executable,
            str(main_py),
            "--argos-worker",
            command,
            source_language,
            target_language,
            str(result_file),
        ]

    kwargs = {
        "input": text,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 180,
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run(worker_command, **kwargs)
        payload = None
        if result_file.is_file():
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None

        if payload is None:
            detail = (result.stderr or result.stdout or "").strip()
            if result.returncode in {-1073741819, 3221225477}:
                detail = (
                    "Argos native runtime crashed in its isolated worker. "
                    "Reinstall Argos packages/runtime."
                )
            raise LocalizationProviderError(
                "Argos worker did not return a valid result"
                + (f": {detail}" if detail else "")
            )

        if result.returncode != 0 or not payload.get("ok"):
            raise LocalizationProviderError(
                str(payload.get("error") or "Argos translation failed")
            )
        return payload
    except LocalizationProviderError:
        raise
    except Exception as exc:
        raise LocalizationProviderError(f"Argos worker failed: {exc}") from exc
    finally:
        result_file.unlink(missing_ok=True)


def argos_route_available(source_language: str) -> tuple[bool, str]:
    try:
        payload = _run_argos_worker("route", source_language, "vi")
    except (LocalizationProviderError, ValueError) as exc:
        return False, str(exc)
    route = payload.get("route") or [source_language, "vi"]
    return True, "Argos route available: " + " -> ".join(route)


@dataclass
class ArgosTranslationProvider:
    """Offline/free translation provider executed in an isolated process."""

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Translation text cannot be empty")
        if source_language not in {"en", "zh"}:
            raise ValueError("Source language must be 'en' or 'zh'")
        if target_language != "vi":
            raise ValueError("Argos localization target must be 'vi'")

        payload = _run_argos_worker(
            "translate",
            source_language,
            target_language,
            text.strip(),
        )
        translated = payload.get("text")
        if not isinstance(translated, str) or not translated.strip():
            raise LocalizationProviderError("Argos returned empty translation text")
        return translated.strip()


@dataclass
class NLLBTranslationProvider:
    """Local Meta NLLB-200 translation for direct English/Chinese to Vietnamese."""

    batch_size: int = 2
    max_input_length: int = 512
    max_new_tokens: int = 256

    def __post_init__(self):
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import sentencepiece  # noqa: F401 - required by the NLLB tokenizer
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise LocalizationProviderError(
                "NLLB requires torch, transformers, huggingface_hub, and sentencepiece "
                "in the Auto Video Python environment."
            ) from exc

        local_path = nllb_model_path()
        try:
            model_path = local_path
            if not nllb_model_is_available():
                # Materialize the complete public repository at the explicit D:-backed
                # model location. Transformers is subsequently kept local-only.
                model_path = Path(snapshot_download(
                    repo_id=NLLB_MODEL_ID,
                    local_dir=local_path,
                    cache_dir=local_path.parent / ".huggingface-cache",
                ))
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.float32
            )
        except Exception as exc:
            location = str(local_path)
            if nllb_model_is_available():
                message = f"Unable to load local NLLB model at {location}: {exc}"
            else:
                message = (
                    f"Unable to download NLLB model to {location}. Connect once for "
                    f"the download, then NLLB works offline: {exc}"
                )
            raise LocalizationProviderError(message) from exc

        self._tokenizer = tokenizer
        self._model = model.to("cpu").eval()

    @staticmethod
    def _validate(source_language: str, target_language: str) -> None:
        if source_language not in {"en", "zh"}:
            raise ValueError("Source language must be 'en' or 'zh'")
        if target_language != "vi":
            raise ValueError("NLLB localization target must be 'vi'")

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Translation text cannot be empty")
        self._validate(source_language, target_language)
        return self._translate_texts([text], source_language)[0]

    def _translate_texts(self, texts: list[str], source_language: str) -> list[str]:
        self._load()
        import torch

        tokenizer = self._tokenizer
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]
        target_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES["vi"])
        translations: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [" ".join(text.split()) for text in texts[start:start + self.batch_size]]
            try:
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_input_length,
                )
                with torch.inference_mode():
                    generated = self._model.generate(
                        **encoded, forced_bos_token_id=target_id,
                        do_sample=False, num_beams=4, early_stopping=True,
                        max_new_tokens=self.max_new_tokens,
                    )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            except Exception as exc:
                raise LocalizationProviderError(f"NLLB translation failed: {exc}") from exc
            if len(decoded) != len(batch) or any(not item.strip() for item in decoded):
                raise LocalizationProviderError("NLLB returned empty or incomplete translations")
            translations.extend(item.strip() for item in decoded)
        return translations

    def translate_segments(
        self, segments, source_language: str, target_language: str, *, context=()
    ) -> list[tuple[str, str]]:
        """Translate independent segments while preserving their source text and order."""
        self._validate(source_language, target_language)
        if not segments:
            return []
        source_texts = [item.text for item in segments]
        if any(not isinstance(text, str) or not text.strip() for text in source_texts):
            raise ValueError("Translation text cannot be empty")
        translated = self._translate_texts(source_texts, source_language)
        return [(text, vietnamese) for text, vietnamese in zip(source_texts, translated)]


@dataclass
class GeminiTranslationProvider:
    api_key: str
    model: str = "gemini-3.8-flash"
    timeout: float = 60.0
    rate_limit_fallback_models: tuple[str, ...] | None = None

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        if not self.model:
            raise ValueError("Gemini model is required")
        self.model = normalize_gemini_model_name(self.model)
        configured_fallbacks = self.rate_limit_fallback_models
        if configured_fallbacks is None:
            configured_fallbacks = tuple(filter(None, (
                value.strip() for value in os.getenv(
                    "BUZZ_GEMINI_429_FALLBACK_MODELS", "gemini-3.5-flash-lite"
                ).split(",")
            )))
        normalized_fallbacks = tuple(
            normalize_gemini_model_name(value) for value in configured_fallbacks
        )
        unknown = set(normalized_fallbacks) - GEMINI_MODEL_IDS
        if unknown:
            raise ValueError(
                "Gemini fallback models must use model IDs already supported by Buzz: "
                + ", ".join(sorted(unknown))
            )
        self.rate_limit_fallback_models = normalized_fallbacks
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise LocalizationProviderError(
                "google-genai is required for Gemini translation"
            ) from exc
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=genai_types.HttpOptions(
                timeout=max(1, int(self.timeout * 1000)),
            ),
        )

    def _generate_content_with_retry(self, *, contents, config=None):
        primary = getattr(self, "model", "gemini-3.8-flash")
        last_error = None
        fallbacks = getattr(self, "rate_limit_fallback_models", None)
        if fallbacks is None:
            fallbacks = ("gemini-3.5-flash-lite",)
        models = [primary]
        retryable_failure = False
        rate_limited = False
        for model in models + [m for m in fallbacks if m != primary]:
            if model != primary and not rate_limited:
                break
            for delay in (0, 5, 15):
                if delay:
                    time.sleep(delay)
                try:
                    kwargs = {"model": model, "contents": contents}
                    if config is not None:
                        kwargs["config"] = config
                    response = self._client.models.generate_content(**kwargs)
                    self.last_model_used = model
                    return response
                except Exception as exc:
                    last_error = exc
                    message = str(exc).upper()
                    current_rate_limit = (
                        "429" in message or "RESOURCE_EXHAUSTED" in message
                    )
                    rate_limited = rate_limited or current_rate_limit
                    retryable = current_rate_limit or any(token in message for token in (
                        "503", "UNAVAILABLE", "TIMEOUT", "TIMED OUT", "CONNECTION",
                    ))
                    retryable_failure = retryable_failure or retryable
                    if not retryable:
                        raise
            if not retryable_failure:
                break
        raise last_error

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Translation text cannot be empty")
        if target_language != "vi":
            raise ValueError("This provider is configured for Vietnamese translation")
        if source_language not in {"en", "zh"}:
            raise ValueError("Source language must be 'en' or 'zh'")

        source_name = "English" if source_language == "en" else "Chinese"
        prompt = (
            f"Translate the following {source_name} into natural, concise Vietnamese "
            "suitable for spoken dubbing. Preserve meaning, names, numbers, units, "
            "and important entities. Return only the Vietnamese translation, with "
            "no commentary or quotation marks.\n\n"
            + text.strip()
        )
        try:
            response = self._generate_content_with_retry(
                contents=prompt,
            )
        except Exception as exc:
            raise LocalizationProviderError(
                f"Gemini translation request failed: {exc}"
            ) from exc

        translated = getattr(response, "text", None)
        if not isinstance(translated, str) or not translated.strip():
            raise LocalizationProviderError("Gemini returned empty translation text")
        return translated.strip()

    def translate_segments(
        self,
        segments,
        source_language: str,
        target_language: str,
        *,
        context=(),
    ) -> list[tuple[str, str]]:
        """Correct recognition and translate a timestamped batch in one request."""
        if source_language not in {"zh", "en"} or target_language != "vi":
            raise ValueError("Gemini batch requires English/Chinese to Vietnamese")
        if not segments:
            return []
        payload = {
            "source_language": source_language,
            "context_only": [
                {"start": item.start, "end": item.end, "text": item.text}
                for item in context
            ],
            "segments": [
                {"id": getattr(item, "segment_id", str(index)),
                 "start": item.start, "end": item.end, "text": item.text}
                for index, item in enumerate(segments)
            ],
        }
        prompt = (
            "You are translating a video for Vietnamese dubbing. Read all segments "
            "together for context. For Chinese, fix obvious Whisper recognition errors "
            "using context, while preserving meaning, product terms, names, numbers "
            "and units. For English, preserve the original transcript. Translate each "
            "corrected segment to concise, natural spoken Vietnamese. Never combine, "
            "drop, reorder, or split segments. Context-only entries are for background "
            "and must NOT appear in the output. Keep each ID unchanged. Return a JSON "
            "object with only a 'segments' array of objects with unchanged 'id', string "
            "'corrected_text' and string 'translated_text'. Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            response = self._generate_content_with_retry(
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            result = json.loads(response.text)
            items = result["segments"]
            if not isinstance(items, list) or len(items) != len(segments):
                raise ValueError("wrong number of segments")
            expected_ids = [
                getattr(item, "segment_id", str(index))
                for index, item in enumerate(segments)
            ]
            uses_stable_ids = any(
                hasattr(item, "segment_id") for item in segments
            )
            by_id = {}
            for item in items:
                identifier = item["id"]
                if not uses_stable_ids and type(identifier) is int:
                    identifier = str(identifier)
                if identifier in by_id or identifier not in expected_ids:
                    raise ValueError("duplicate or invalid segment ID")
                corrected = item["corrected_text"]
                translated = item["translated_text"]
                if not isinstance(corrected, str) or not corrected.strip():
                    raise ValueError("empty corrected transcript")
                if not isinstance(translated, str) or not translated.strip():
                    raise ValueError("empty Vietnamese translation")
                source_index = expected_ids.index(identifier)
                source_text = corrected if source_language == "zh" else segments[source_index].text
                by_id[identifier] = (source_text.strip(), translated.strip())
            if set(by_id) != set(expected_ids):
                raise ValueError("missing or unexpected segment ID")
            return [by_id[identifier] for identifier in expected_ids]
        except Exception as exc:
            raise LocalizationProviderError(
                f"Gemini batch translation failed: {exc}"
            ) from exc

    def shorten_translation(
        self,
        *,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        required_rate: float,
        max_playback_rate: float,
    ) -> str:
        """Make one Vietnamese dubbing line shorter without dropping meaning."""
        if source_language not in {"en", "zh"} or target_language != "vi":
            raise ValueError("Gemini shortening requires English/Chinese to Vietnamese")
        target_ratio = max_playback_rate / required_rate
        prompt = (
            "Rewrite one Vietnamese dubbing line so it can be spoken in less time. "
            "Preserve the complete meaning, names, numbers, units, negation, and key "
            "details. Use concise natural spoken Vietnamese; do not summarize away "
            "content and do not add commentary. Return only the rewritten Vietnamese "
            f"line. Aim for at most {target_ratio:.0%} of the current spoken length.\n"
            f"Source ({source_language}): {source_text.strip()}\n"
            f"Current Vietnamese: {translated_text.strip()}"
        )
        try:
            response = self._generate_content_with_retry(contents=prompt)
        except Exception as exc:
            raise LocalizationProviderError(
                f"Gemini translation shortening request failed: {exc}"
            ) from exc
        shortened = getattr(response, "text", None)
        if not isinstance(shortened, str) or not shortened.strip():
            raise LocalizationProviderError("Gemini returned empty shortened translation")
        return shortened.strip()


@dataclass
class EdgeTTSProvider:
    default_voice: str = "vi-VN-HoaiMyNeural"
    max_concurrency: int = 1
    retry_delays: tuple[float, ...] | None = None

    # A service throttle can outlive the short retries that are appropriate for a
    # dropped connection.  This remains finite, while allowing more than three
    # minutes for Edge to become available again.
    transient_retry_delays = (5, 10, 20, 40, 60, 90, 120)

    def __post_init__(self) -> None:
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("Edge TTS max_concurrency must be at least 1")
        if self.retry_delays is None:
            self.retry_delays = self.transient_retry_delays
        else:
            self.retry_delays = tuple(float(delay) for delay in self.retry_delays)
            if any(delay < 0 for delay in self.retry_delays):
                raise ValueError("Edge TTS retry delays cannot be negative")
        self._synthesis_slots = threading.BoundedSemaphore(self.max_concurrency)

    def tts_cache_identity(self) -> dict[str, str | None]:
        """Settings outside TTSRequest that can change synthesized audio."""
        try:
            import edge_tts
            version = getattr(edge_tts, "__version__", None)
        except ImportError:
            version = None
        return {
            "provider": "edge-tts",
            "provider_version": version,
            "default_voice": self.default_voice,
        }

    @staticmethod
    def _safe_exception_details(exc: Exception) -> str:
        """Return actionable exception details with common secrets removed."""
        message = str(exc).replace("\r", " ").replace("\n", " ").strip()
        message = re.sub(
            r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?|"
            r"(?:api[_-]?key|token|secret|sig|signature)\s*[:=]\s*)"
            r"[^\s,;&]+",
            r"\1[REDACTED]",
            message,
        )
        message = re.sub(
            r"(?i)([?&](?:api[_-]?key|token|access_token|sig|signature)=)"
            r"[^&#\s]+",
            r"\1[REDACTED]",
            message,
        )
        if len(message) > 500:
            message = message[:497] + "..."

        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        response = getattr(exc, "response", None)
        status = status or getattr(response, "status_code", None) or getattr(
            response, "status", None
        )
        details = f"{type(exc).__name__}: {message or '<no message>'}"
        if status is not None and f"status={status}" not in details:
            details += f" (status={status})"
        return details

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        value = None
        if headers is not None:
            try:
                value = headers.get("Retry-After") or headers.get("retry-after")
            except (AttributeError, TypeError):
                pass
        if value is None:
            value = getattr(exc, "retry_after", None)
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _cache_paths(output_path: Path) -> tuple[Path, Path]:
        return output_path, output_path.with_name(
            f"{output_path.stem}.tts-cache.json"
        )

    @staticmethod
    def _cache_fingerprint(
        request: TTSRequest,
        voice: str,
        rate: str,
        volume: str,
        pitch: str,
        edge_tts_version: str | None,
    ) -> str:
        """Return a stable identity for audio-affecting Edge TTS settings."""
        payload = {
            "provider": "edge-tts",
            "provider_version": edge_tts_version,
            "text": request.text,
            "language": request.language,
            "voice": voice,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _read_cached_result(
        output_path: Path, checkpoint_path: Path, fingerprint: str,
        expected_voice: str,
    ) -> TTSResult | None:
        """Return a verified cache entry, never trusting a bare audio file."""
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("fingerprint") != fingerprint:
                return None
            if checkpoint.get("audio_file") != str(output_path):
                return None
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                return None
            duration = _probe_audio_duration(output_path)
            if duration <= 0:
                return None
            metadata = checkpoint.get("metadata")
            if not isinstance(metadata, dict):
                return None
            voice = checkpoint.get("voice")
            if voice != expected_voice:
                return None
            return TTSResult(
                audio_file=str(output_path), audio_duration=duration,
                provider="edge-tts", voice=voice, metadata=metadata,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError,
                LocalizationProviderError):
            return None

    @staticmethod
    def _write_checkpoint(
        checkpoint_path: Path, fingerprint: str, result: TTSResult
    ) -> None:
        payload = {
            "fingerprint": fingerprint,
            "audio_file": result.audio_file,
            "voice": result.voice,
            "metadata": dict(result.metadata),
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.", suffix=".tmp", delete=False,
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if request.language != "vi":
            raise ValueError("EdgeTTSProvider requires Vietnamese ('vi')")
        if not request.text.strip():
            raise ValueError("TTS text cannot be empty")

        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived, WebSocketError
        except ImportError as exc:
            raise LocalizationProviderError(
                "edge-tts is required for Vietnamese TTS"
            ) from exc

        voice = request.voice or self.default_voice
        output_file = f"{request.output_file_stem}.mp3"
        options = dict(request.options)
        rate = str(options.get("rate", "+0%"))
        volume = str(options.get("volume", "+0%"))
        pitch = str(options.get("pitch", "+0Hz"))

        output_path = Path(output_file)
        _, checkpoint_path = self._cache_paths(output_path)
        fingerprint = self._cache_fingerprint(
            request, voice, rate, volume, pitch,
            getattr(edge_tts, "__version__", None),
        )
        cached = self._read_cached_result(
            output_path, checkpoint_path, fingerprint, voice
        )
        if cached is not None:
            logger.info("Reusing verified Edge TTS checkpoint for %s", output_path.name)
            return cached

        # A stale checkpoint must not survive a failed replacement and later be
        # mistaken for the new request.
        output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        retry_delays = self.retry_delays
        max_attempts = len(retry_delays) + 1
        with self._synthesis_slots:
            for attempt in range(1, max_attempts + 1):
                output_path.unlink(missing_ok=True)
                try:
                    communicate = edge_tts.Communicate(
                        request.text,
                        voice,
                        rate=rate,
                        volume=volume,
                        pitch=pitch,
                    )
                    communicate.save_sync(output_file)
                    if output_path.is_file() and output_path.stat().st_size > 0:
                        break
                    raise NoAudioReceived(
                        "Edge TTS returned no audio data."
                    )
                except (NoAudioReceived, WebSocketError) as exc:
                    details = self._safe_exception_details(exc)
                    if attempt >= max_attempts:
                        output_path.unlink(missing_ok=True)
                        logger.error(
                            "Edge TTS final failure (%d/%d) for voice %s: %s",
                            attempt, max_attempts, voice, details,
                        )
                        raise LocalizationProviderError(
                            f"Edge TTS failed after {max_attempts} attempts: {details}"
                        ) from exc
                    delay = max(
                        retry_delays[attempt - 1],
                        self._retry_after_seconds(exc) or 0.0,
                    )
                    logger.warning(
                        "Edge TTS transient failure (%d/%d) for voice %s: %s; "
                        "retrying in %.1f seconds",
                        attempt, max_attempts, voice, details, delay,
                    )
                    time.sleep(delay)
                except Exception as exc:
                    output_path.unlink(missing_ok=True)
                    details = self._safe_exception_details(exc)
                    logger.error(
                        "Edge TTS final failure (1/%d) for voice %s: %s",
                        max_attempts, voice, details,
                    )
                    raise LocalizationProviderError(
                        f"Edge TTS failed: {details}"
                    ) from exc

        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise LocalizationProviderError("Edge TTS did not create an audio file")

        try:
            _trim_edge_tts_silence(output_path)
            duration = _probe_audio_duration(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        result = TTSResult(
            audio_file=str(output_path),
            audio_duration=duration,
            provider="edge-tts",
            voice=voice,
            metadata={
                "format": "mp3",
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
            },
        )
        try:
            self._write_checkpoint(checkpoint_path, fingerprint, result)
        except Exception:
            output_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
            raise
        return result


def _trim_edge_tts_silence(audio_file: Path) -> None:
    trimmed = audio_file.with_name(f"{audio_file.stem}.trimmed{audio_file.suffix}")
    filter_expression = (
        "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB,"
        "areverse"
    )
    _run_ffmpeg([
        "ffmpeg", "-y", "-nostdin", "-i", str(audio_file),
        "-af", filter_expression, str(trimmed),
    ])
    if not trimmed.is_file() or trimmed.stat().st_size <= 0:
        trimmed.unlink(missing_ok=True)
        raise LocalizationProviderError("FFmpeg did not create trimmed Edge TTS audio")
    trimmed.replace(audio_file)


def _probe_audio_duration(audio_file: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_ffmpeg_env(),
        )
    except Exception as exc:
        raise LocalizationProviderError(
            f"Unable to probe synthesized audio duration: {exc}"
        ) from exc

    if result.returncode != 0:
        raise LocalizationProviderError(
            "Unable to probe synthesized audio duration: "
            + (result.stderr or "ffprobe failed").strip()
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise LocalizationProviderError(
            "ffprobe returned an invalid audio duration"
        ) from exc

    if duration <= 0:
        raise LocalizationProviderError("Synthesized audio duration must be positive")
    return duration
