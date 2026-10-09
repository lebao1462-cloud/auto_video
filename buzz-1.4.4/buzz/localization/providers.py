from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import atexit
import asyncio
import hashlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import tempfile
import threading
import time
from typing import ClassVar, Mapping, Sequence
import uuid

from openai import OpenAI

from buzz.localization.audio_mix import _ffmpeg_env, _run_ffmpeg
from buzz.localization.tts import (
    TTSBatchFallbackRequired, TTSRequest, TTSResult,
    TTSServiceTemporarilyUnavailable,
)


logger = logging.getLogger(__name__)


@dataclass
class _VieNeuSharedSession:
    """One persistent VieNeu worker and the state needed to use it safely."""

    key: tuple[str, ...]
    process: subprocess.Popen | None = None
    request_number: int = 0
    users: int = 0
    invalid: bool = False
    request_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class VieNeuTTSProvider:
    """Offline VieNeu v3 Turbo provider using a persistent isolated worker."""

    default_voice: str = "Trúc Ly"
    python_executable: str | None = None
    model_dir: str | None = None
    hf_home: str | None = None
    worker_command: Sequence[str] | None = None
    worker_timeout_seconds: float = 120.0
    _session: _VieNeuSharedSession | None = field(default=None, init=False, repr=False)
    _lifecycle_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    _sessions: ClassVar[dict[tuple[str, ...], _VieNeuSharedSession]] = {}
    _sessions_lock: ClassVar[threading.Lock] = threading.Lock()

    def __post_init__(self) -> None:
        self.python_executable = (
            self.python_executable
            or os.environ.get("BUZZ_VIENEU_PYTHON")
            or self._existing("D:/codex/vieneu_test/.venv/Scripts/python.exe")
        )
        self.model_dir = (
            self.model_dir
            or os.environ.get("BUZZ_VIENEU_MODEL_DIR")
            or self._existing("D:/AIModels/VieNeu-TTS-v3-Turbo/onnx_update")
        )
        self.hf_home = (
            self.hf_home
            or os.environ.get("BUZZ_VIENEU_HF_HOME")
            or self._existing("D:/AIModels/HuggingFace")
        )
        if not self.python_executable or not Path(self.python_executable).is_file():
            raise LocalizationProviderError(
                "VieNeu runtime unavailable. Set BUZZ_VIENEU_PYTHON to its Python executable."
            )
        if not self.model_dir or not Path(self.model_dir).is_dir():
            raise LocalizationProviderError(
                "VieNeu model unavailable. Set BUZZ_VIENEU_MODEL_DIR to the local ONNX directory."
            )
        if not self.hf_home or not Path(self.hf_home).is_dir():
            raise LocalizationProviderError(
                "VieNeu cache unavailable. Set BUZZ_VIENEU_HF_HOME to the populated local cache."
            )
        if self.worker_timeout_seconds <= 0:
            raise ValueError("VieNeu worker timeout must be positive")

    @staticmethod
    def _existing(path: str) -> str | None:
        return path if Path(path).exists() else None

    def tts_cache_identity(self) -> dict[str, str]:
        # Runtime/cache locations do not affect the audible identity. Keeping
        # them out also allows moving the installed runtime without invalidating
        # otherwise reusable audio checkpoints.
        return {
            "provider": "vieneu-v3-turbo",
            "voice": self.default_voice,
        }

    def _worker_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        hub_cache = str(Path(self.hf_home) / "hub")
        env.update({
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": hub_cache,
            "HUGGINGFACE_HUB_CACHE": hub_cache,
            "HF_HUB_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        })
        return env

    def _session_key(self) -> tuple[str, ...]:
        worker_file = Path(__file__).with_name("vieneu_worker.py")
        command = tuple(str(part) for part in (
            self.worker_command
            or (self.python_executable, "-u", str(worker_file))
        ))
        return (
            str(self.python_executable), str(self.model_dir), str(self.hf_home), *command,
        )

    def _acquire_session(self) -> _VieNeuSharedSession:
        key = self._session_key()
        with self._sessions_lock:
            session = self._session
            if session is not None and not session.invalid and self._sessions.get(key) is session:
                return session
            if session is not None:
                session.users -= 1
            session = self._sessions.get(key)
            if session is None or session.invalid:
                session = _VieNeuSharedSession(key)
                self._sessions[key] = session
            session.users += 1
            self._session = session
            return session

    def _start(self, session: _VieNeuSharedSession) -> subprocess.Popen:
        if session.process is not None and session.process.poll() is None:
            return session.process
        worker_file = Path(__file__).with_name("vieneu_worker.py")
        command = list(self.worker_command or (self.python_executable, "-u", str(worker_file)))
        try:
            session.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self._worker_environment(),
            )
        except OSError as exc:
            raise LocalizationProviderError(
                f"Unable to start VieNeu worker: {exc}"
            ) from exc
        return session.process

    @staticmethod
    def _stop_worker(session: _VieNeuSharedSession) -> None:
        process = session.process
        session.process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    @classmethod
    def _invalidate_session(cls, session: _VieNeuSharedSession) -> None:
        with cls._sessions_lock:
            session.invalid = True
            if cls._sessions.get(session.key) is session:
                del cls._sessions[session.key]
        cls._stop_worker(session)

    def _read_response(
        self, session: _VieNeuSharedSession, process: subprocess.Popen,
    ) -> str:
        result: list[str] = []
        failure: list[BaseException] = []

        def reader() -> None:
            try:
                result.append(process.stdout.readline())
            except BaseException as exc:
                failure.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(self.worker_timeout_seconds)
        if thread.is_alive():
            self._invalidate_session(session)
            raise LocalizationProviderError(
                f"VieNeu synthesis exceeded {self.worker_timeout_seconds:g} seconds"
            )
        if failure:
            self._invalidate_session(session)
            raise LocalizationProviderError(
                f"VieNeu worker protocol failed: {failure[0]}"
            )
        line = result[0] if result else ""
        if not line:
            return_code = process.poll()
            self._invalidate_session(session)
            raise LocalizationProviderError(
                "VieNeu worker exited without a response"
                + (f" (exit code {return_code})" if return_code is not None else "")
            )
        return line

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if request.language != "vi":
            raise ValueError("VieNeuTTSProvider requires Vietnamese ('vi')")
        if not request.text.strip():
            raise ValueError("TTS text cannot be empty")

        output = Path(f"{request.output_file_stem}.wav")
        output.unlink(missing_ok=True)
        with self._lifecycle_lock:
            session = self._acquire_session()
            while True:
                with session.request_lock:
                    # A request waiting behind a failed request must register
                    # with the replacement session instead of reviving the
                    # invalidated worker outside the registry.
                    if session.invalid:
                        session = self._acquire_session()
                        continue
                    process = self._start(session)
                    session.request_number += 1
                    identifier = session.request_number
                    payload = {
                        "id": identifier,
                        "text": request.text,
                        "voice": request.voice or self.default_voice,
                        "output_file": str(output),
                        "model_dir": self.model_dir,
                    }
                    try:
                        process.stdin.write(
                            json.dumps(payload, ensure_ascii=False) + "\n"
                        )
                        process.stdin.flush()
                        response = json.loads(self._read_response(session, process))
                        if not isinstance(response, dict) or response.get("id") != identifier:
                            self._invalidate_session(session)
                            raise LocalizationProviderError(
                                "VieNeu worker returned a mismatched response"
                            )
                    except LocalizationProviderError:
                        raise
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        self._invalidate_session(session)
                        raise LocalizationProviderError(
                            f"VieNeu worker protocol failed: {exc}"
                        ) from exc
                    break

        if not response.get("ok"):
            raise LocalizationProviderError(
                "VieNeu synthesis failed: "
                + str(response.get("error", "invalid worker response"))
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise LocalizationProviderError(
                "VieNeu worker did not create WAV audio"
            )
        try:
            duration = float(response["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            output.unlink(missing_ok=True)
            raise LocalizationProviderError(
                "VieNeu worker returned an invalid audio duration"
            ) from exc
        if duration <= 0:
            output.unlink(missing_ok=True)
            raise LocalizationProviderError(
                "VieNeu worker returned an invalid audio duration"
            )
        return TTSResult(
            audio_file=str(output),
            audio_duration=duration,
            provider="vieneu-v3-turbo",
            voice=request.voice or self.default_voice,
            metadata=dict(response.get("metadata", {})),
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            session = self._session
            self._session = None
            if session is None:
                return
            with self._sessions_lock:
                session.users -= 1
                final_user = session.users == 0
                if final_user and self._sessions.get(session.key) is session:
                    del self._sessions[session.key]
            if final_user:
                with session.request_lock:
                    self._stop_worker(session)


def _cleanup_vieneu_sessions() -> None:
    """Best-effort shutdown for workers left alive during interpreter exit."""
    with VieNeuTTSProvider._sessions_lock:
        sessions = list(VieNeuTTSProvider._sessions.values())
        VieNeuTTSProvider._sessions.clear()
        for session in sessions:
            session.invalid = True
    for session in sessions:
        with session.request_lock:
            VieNeuTTSProvider._stop_worker(session)


atexit.register(_cleanup_vieneu_sessions)


EDGE_TTS_CONNECT_TIMEOUT_SECONDS = 10
EDGE_TTS_RECEIVE_TIMEOUT_SECONDS = 60
EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS = 45


class _EdgeTTSAttemptError(RuntimeError):
    """A serializable failure reported by the isolated Edge TTS attempt."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


_EDGE_TTS_TRANSIENT_CHILD_ERRORS = {
    "NoAudioReceived", "WebSocketError", "TimeoutError",
    "ClientConnectionError", "ClientConnectorError",
    "ClientConnectorCertificateError", "ClientConnectorSSLError",
    "ClientOSError", "ClientPayloadError", "ServerDisconnectedError",
    "ServerTimeoutError", "WSServerHandshakeError",
}


def _is_transient_edge_attempt_error(exc: BaseException) -> bool:
    return (
        not isinstance(exc, _EdgeTTSAttemptError)
        or exc.error_type in _EDGE_TTS_TRANSIENT_CHILD_ERRORS
    )


def _edge_tts_attempt_worker(
    text: str, voice: str, options: dict[str, str], output_file: str,
    sidecar_file: str, batch: bool,
) -> None:
    """Run one Edge request in a spawn-safe process and report its boundaries."""
    try:
        import edge_tts

        async def run() -> list[dict[str, object]]:
            communicate = edge_tts.Communicate(text, voice, **options)
            if not batch:
                await communicate.save(output_file)
                return []
            boundaries = []
            with Path(output_file).open("wb") as handle:
                async for event in communicate.stream():
                    if event.get("type") == "audio":
                        handle.write(event["data"])
                    elif event.get("type") == "WordBoundary":
                        boundaries.append(event)
            return boundaries

        boundaries = asyncio.run(run())
        Path(sidecar_file).write_text(
            json.dumps({"boundaries": boundaries}, ensure_ascii=False), encoding="utf-8"
        )
    except BaseException as exc:
        Path(sidecar_file).write_text(json.dumps({
            "error_type": type(exc).__name__, "message": str(exc),
        }), encoding="utf-8")


def _run_edge_tts_attempt(
    text: str, voice: str, options: dict[str, str], output_file: str, *, batch: bool,
    timeout_seconds: float = EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    """Execute a killable Edge attempt; unlike wait_for this is a hard deadline."""
    output = Path(output_file)
    sidecar = output.with_name(f".{output.name}.edge-attempt-{uuid.uuid4().hex}.json")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_edge_tts_attempt_worker,
        args=(text, voice, options, str(output), str(sidecar), batch),
    )
    try:
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            output.unlink(missing_ok=True)
            raise TimeoutError(f"Edge TTS synthesis exceeded {timeout_seconds} seconds")
        try:
            result = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            output.unlink(missing_ok=True)
            raise _EdgeTTSAttemptError(
                "ProcessError", "Edge TTS attempt exited without a result"
            ) from exc
        if "error_type" in result:
            output.unlink(missing_ok=True)
            raise _EdgeTTSAttemptError(
                str(result["error_type"]), str(result.get("message", ""))
            )
        return list(result.get("boundaries", []))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        sidecar.unlink(missing_ok=True)


def _has_speakable_tts_text(text: str) -> bool:
    """Return whether text contains content that a TTS provider can speak."""
    return isinstance(text, str) and bool(text.strip()) and any(
        character.isalnum() for character in text
    )


def _edge_tts_safe_payloads(text: str) -> tuple[str, ...]:
    """Return conservative fallback payloads without changing workflow text."""
    normalized = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[?!…。！？]+$", "", normalized).rstrip()
    candidates: list[str] = []
    if normalized and normalized != text:
        candidates.append(normalized)
    wording_safe = re.sub(
        r"^Ả\s+phụ\s+nữ\b", "Người phụ nữ", normalized,
    )
    if wording_safe and wording_safe not in (text, *candidates):
        candidates.append(wording_safe)
    return tuple(candidates)


GEMINI_MODEL_IDS = frozenset({
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
})

DEFAULT_GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
)


class LocalizationProviderError(RuntimeError):
    """Raised when a concrete localization provider fails."""


class _EdgeTTSNoAudioExhausted(LocalizationProviderError):
    """Signal that a single payload exhausted only empty-audio retries."""


def _is_proxifier_running() -> bool:
    """Return whether Proxifier is running, treating probe failures as absent."""
    if os.name != "nt":
        return False
    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 2,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Proxifier.exe", "/NH", "/FO", "CSV"],
            **kwargs,
        )
        return result.returncode == 0 and "proxifier.exe" in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False


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
    model: str = "gemini-3.5-flash-lite"
    # Long localization batches can legitimately exceed the generic timeout.
    # Keep this configurable for callers which need a different limit.
    timeout: float = 120.0
    rate_limit_fallback_models: tuple[str, ...] | None = None
    translation_policy_version: str = "vi-dialogue-register-v3"

    _quota_cooldown_seconds = 65.0
    _transient_retry_delays = (0, 5, 15)

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        if not self.model:
            raise ValueError("Gemini model is required")
        self.model = normalize_gemini_model_name(self.model)
        configured_fallbacks = self.rate_limit_fallback_models
        if configured_fallbacks is None:
            # The 429-specific name remains supported for existing deployments.
            configured = os.getenv("BUZZ_GEMINI_FALLBACK_MODELS")
            if configured is None:
                configured = os.getenv("BUZZ_GEMINI_429_FALLBACK_MODELS", "")
            configured_fallbacks = tuple(
                value.strip() for value in configured.split(",") if value.strip()
            )
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
        self._rpd_exhausted_models: set[str] = set()
        self._unavailable_models: set[str] = set()
        self._rpm_cooldowns: dict[str, float] = {}
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

    def _model_candidates(self) -> tuple[str, ...]:
        """Return selected, configured, then built-in models without duplicates."""
        primary = getattr(self, "model", "gemini-3.5-flash-lite")
        fallbacks = getattr(self, "rate_limit_fallback_models", None)
        if fallbacks is None:
            fallbacks = DEFAULT_GEMINI_FALLBACK_MODELS
        # Always retain the built-in chain, even when an environment or caller
        # adds preferred fallbacks. The selected primary is always attempted first.
        return tuple(dict.fromkeys((
            primary, *fallbacks, *DEFAULT_GEMINI_FALLBACK_MODELS,
        )))

    def _generate_content_with_retry(self, *, contents, config=None):
        """Generate content while sharing quota state across this provider session."""
        models = self._model_candidates()
        exhausted = getattr(self, "_rpd_exhausted_models", None)
        if exhausted is None:
            exhausted = self._rpd_exhausted_models = set()
        unavailable = getattr(self, "_unavailable_models", None)
        if unavailable is None:
            unavailable = self._unavailable_models = set()
        cooldowns = getattr(self, "_rpm_cooldowns", None)
        if cooldowns is None:
            cooldowns = self._rpm_cooldowns = {}

        last_error = None
        transient_failed: set[str] = set()
        while True:
            now = time.monotonic()
            eligible = [
                model for model in models
                if (model not in exhausted and model not in unavailable
                    and model not in transient_failed and cooldowns.get(model, 0) <= now)
            ]
            if not eligible:
                available_models = [model for model in models if model not in unavailable]
                if not available_models:
                    if last_error is not None:
                        raise last_error
                    raise LocalizationProviderError("No configured Gemini models are available")
                if all(model in exhausted for model in available_models):
                    raise LocalizationProviderError(
                        "Gemini daily quota is exhausted for every configured model. "
                        "Try again after quota resets or configure another model."
                    )
                remaining = [
                    model for model in available_models
                    if model not in exhausted and model not in transient_failed
                ]
                if not remaining:
                    if last_error is not None:
                        raise last_error
                    raise LocalizationProviderError("No Gemini models are available")
                next_ready = min(cooldowns.get(model, now) for model in remaining)
                wait_seconds = max(0.0, next_ready - now)
                # A finite wait prevents a tight loop even if a clock is adjusted.
                wait_seconds = min(wait_seconds, self._quota_cooldown_seconds)
                logger.warning(
                    "All available Gemini models are cooling down; waiting %.1f seconds",
                    wait_seconds,
                )
                time.sleep(max(wait_seconds, 0.1))
                continue

            for model in eligible:
                quota_limited = False
                dns_failure = False
                for delay in self._transient_retry_delays:
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
                        if self._is_gemini_model_unavailable_error(exc):
                            unavailable.add(model)
                            quota_limited = True
                            logger.warning(
                                "Gemini model %s is unavailable; switching models", model
                            )
                            break
                        quota_kind = self._gemini_quota_kind(exc)
                        if quota_kind == "rpd":
                            exhausted.add(model)
                            quota_limited = True
                            logger.warning("Gemini model %s daily quota exhausted; switching models", model)
                            break
                        if quota_kind == "rpm":
                            quota_limited = True
                            cooldowns[model] = (
                                time.monotonic() + self._quota_cooldown_seconds
                            )
                            logger.warning("Gemini model %s rate-limited; cooling down before retry", model)
                            break
                        if not self._is_gemini_transient_error(exc):
                            raise
                        dns_failure = self._is_gemini_dns_error(exc)
                        if delay != self._transient_retry_delays[-1]:
                            logger.warning("Gemini model %s transient failure; retrying", model)
                if not quota_limited:
                    # A DNS outage is independent of the selected Gemini model.
                    # Do not spend every fallback model on the same unavailable host.
                    if dns_failure and last_error is not None:
                        raise last_error
                    transient_failed.add(model)
                # Every model either succeeded, was switched, or exhausted retries.

            available_models = [model for model in models if model not in unavailable]
            if available_models and all(model in exhausted for model in available_models):
                raise LocalizationProviderError(
                    "Gemini daily quota is exhausted for every configured model. "
                    "Try again after quota resets or configure another model."
                )
            if last_error is not None:
                logger.warning("Gemini model attempts failed; trying another available model")

    @staticmethod
    def _gemini_quota_kind(exc: Exception) -> str | None:
        """Return rpd/rpm for quota errors; unknown 429s are temporary RPM limits."""
        message = str(exc).lower()
        if "429" not in message and "resource_exhausted" not in message:
            return None
        compact_message = re.sub(r"[^a-z0-9]+", "", message)
        daily_markers = (
            "rpd", "per day", "per_day", "daily", "requestsperday",
            "requests_per_day", "request_count_per_day",
        )
        minute_markers = (
            "rpm", "per minute", "per_minute", "per-minute", "requestsperminute",
            "requests_per_minute", "request_count_per_minute",
        )
        if any(re.sub(r"[^a-z0-9]+", "", marker) in compact_message
               for marker in daily_markers):
            return "rpd"
        if any(re.sub(r"[^a-z0-9]+", "", marker) in compact_message
               for marker in minute_markers):
            return "rpm"
        return "rpm"

    @staticmethod
    def _is_gemini_model_unavailable_error(exc: Exception) -> bool:
        """Return whether Gemini rejected a specific model as unavailable."""
        message = str(exc).upper()
        if "404" not in message and "NOT_FOUND" not in message:
            return False
        return (
            "MODEL" in message
            and ("NO LONGER AVAILABLE" in message or "NOT FOUND" in message or "NOT_FOUND" in message)
        )

    @staticmethod
    def _gemini_exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
        """Return an exception and its explicit/implicit causes without loops."""
        chain = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return tuple(chain)

    @staticmethod
    def _is_gemini_dns_error(exc: BaseException) -> bool:
        chain = GeminiTranslationProvider._gemini_exception_chain(exc)
        if any(isinstance(item, socket.gaierror) for item in chain):
            return True
        message = "\n".join(str(item) for item in chain).upper()
        return any(marker in message for marker in (
            "GETADDRINFO FAILED", "ERRNO 11001", "NAME OR SERVICE NOT KNOWN",
            "TEMPORARY FAILURE IN NAME RESOLUTION",
        ))

    @staticmethod
    def _is_gemini_transient_error(exc: Exception) -> bool:
        chain = GeminiTranslationProvider._gemini_exception_chain(exc)
        message = "\n".join(str(item) for item in chain).upper()
        if (re.search(r"(?<!\d)(?:400|401|403)(?!\d)", message)
                or any(marker in message for marker in (
                    "API_KEY_INVALID", "UNAUTHENTICATED", "UNAUTHORIZED",
                    "FORBIDDEN", "PERMISSION_DENIED", "INVALID_ARGUMENT", "QUOTA",
                    "RESOURCE_EXHAUSTED", "MALFORMED", "UNUSABLE STRUCTURED OUTPUT",
                ))):
            return False
        if GeminiTranslationProvider._is_gemini_dns_error(exc):
            return True
        # google-genai can wrap a socket error in httpx.ConnectError. Avoid a
        # hard httpx import: it remains an optional implementation detail.
        if any(type(item).__name__ == "ConnectError"
               and type(item).__module__.startswith("httpx") for item in chain):
            return True
        return any(token in message for token in (
            "500", "502", "503", "504", "UNAVAILABLE", "TIMEOUT", "TIMED OUT",
            "DEADLINE_EXCEEDED", "GATEWAY TIMEOUT", "INTERNAL", "CONNECTION",
            "SERVER DISCONNECTED WITHOUT SENDING A RESPONSE",
            "REMOTE PROTOCOL ERROR", "CONNECTION RESET", "RESET BY PEER",
            "BROKEN PIPE", "UNEXPECTED EOF", "EOF WHILE READING", "READ ERROR",
            "NETWORK ERROR",
        ))

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
            "and important entities. Translate idioms, internet slang, memes, and genre "
            "jargon by their intended meaning in natural Vietnamese; never use awkward "
            "word-for-word calques. For Chinese cultivation/system-fiction dialogue, "
            "keep pronouns, tone, and sentence flow natural for Vietnamese viewers. "
            "Return only the Vietnamese translation, with no commentary or quotation marks.\n\n"
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
                {
                    "start": item.start, "end": item.end, "text": item.text,
                    **({"translated_text": item.translated_text}
                       if getattr(item, "translated_text", None) else {}),
                }
                for item in context
            ],
            "segments": [
                {"id": getattr(item, "segment_id", str(index)),
                 "start": item.start, "end": item.end,
                 "duration_seconds": round(float(item.end) - float(item.start), 3),
                 "text": item.text}
                for index, item in enumerate(segments)
            ],
        }
        prompt = (
            "You are translating a video for Vietnamese dubbing. Read all segments "
            "together for context. For Chinese, fix obvious Whisper recognition errors "
            "using context, while preserving meaning, product terms, names, numbers "
            "and units. For English, preserve the original transcript. Translate each "
            "corrected segment to concise, natural spoken Vietnamese that can be spoken within its "
            "start/end duration when feasible. Preserve every source clause, fact, number, unit, "
            "negation, name, relationship and register; never truncate or summarize content merely "
            "to fit time. Translate Chinese "
            "idioms, internet slang, memes, and cultivation/system-fiction jargon by intended "
            "meaning, never by awkward literal calques. Previous translated context is the "
            "authoritative style and register memory for the ongoing scene. Keep the same "
            "speaker/addressee pronoun pair and address terms across adjacent lines unless the "
            "source explicitly signals a relationship, tone, or status change. NEVER alternate "
            "tao-m\u00e0y, t\u00f4i-c\u1eadu, or ta-ng\u01b0\u1eddi merely for stylistic variety, and avoid mixed pairs "
            "such as ta-m\u00e0y or t\u00f4i-ng\u01b0\u1eddi unless the source truly warrants an asymmetric honorific "
            "relationship. For Chinese cultivation/system-fantasy, default to ta-ng\u01b0\u1eddi for "
            "hostile, stranger, or fantasy confrontations when the relationship is unclear; use "
            "t\u00f4i-c\u1eadu for neutral modern or peer dialogue; use tao-m\u00e0y only when the source is "
            "clearly vulgar, intentionally insulting, or rough, then keep it stable through that "
            "exchange. Keep inner monologue/self-talk consistent with the established protagonist "
            "voice, and distinguish narration from character dialogue. Never combine, drop, reorder, or split "
            "segments. Context-only entries are for background "
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

    def review_translation_register(
        self, segments, source_language: str, target_language: str, *, context=(),
    ) -> list[tuple[str, str]]:
        """Continuity-edit completed Vietnamese without changing source or segmentation."""
        if source_language not in {"zh", "en"} or target_language != "vi":
            raise ValueError("Gemini register review requires English/Chinese to Vietnamese")
        if not segments:
            return []
        payload = {
            "source_language": source_language,
            "context_only": [{
                "id": item.segment_id, "source_text": item.text,
                "reviewed_vietnamese": item.translated_text,
            } for item in context],
            "segments": [{
                "id": item.segment_id, "source_text": item.text,
                "current_vietnamese": item.translated_text,
            } for item in segments],
        }
        prompt = (
            "You are a Vietnamese dialogue continuity editor, NOT a new translator. "
            "Return every input ID once, in order, with revised_translated_text only. Preserve "
            "every fact, clause, name, number, unit, negation, event, and segment boundary: "
            "never merge, drop, reorder, or split. Fix only naturalness, Chinese literal calques, "
            "pronoun/address-term continuity, and register. Context-only prior REVIEWED Vietnamese "
            "is authoritative and must never appear as output. HARD cultivation/system-fantasy rules: "
            "hostile, unknown, or fantasy character exchanges default to ta-người; do not drift to "
            "tao-mày or tôi-cậu merely because Chinese is angry or rough. tao-mày is allowed only "
            "when unmistakable modern vulgar register was already established for that exchange; "
            "tôi-cậu is only clearly modern neutral peer dialogue. Same-sect relations prefer explicit "
            "source titles (sư huynh, sư tỷ, huynh, đệ, tiền bối) or the established pair. System/host "
            "must consistently use ký chủ for the protagonist; the host calls it hệ thống or the "
            "established term. Keep one inner-monologue first-person voice within a scene. Keep narration "
            "distinct, preferring a name or consistent third-person form. Never make a mixed ordinary "
            "symmetric pair such as tao-người, tao-cậu, ta-mày, tôi-người, or tôi-mày. Chinese 坑爹 is "
            "idiomatic slang and MUST NEVER become hố cha; use natural Vietnamese such as chơi khăm, "
            "tệ hại, or khốn nạn according to tone. Keep anti-literal-calque guidance. Return JSON only: "
            "{\"segments\":[{\"id\":...,\"revised_translated_text\":...}]}. Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            response = self._generate_content_with_retry(
                contents=prompt, config={"response_mime_type": "application/json"},
            )
            result = json.loads(response.text)
            values = result["segments"]
            expected = [item.segment_id for item in segments]
            if not isinstance(values, list) or len(values) != len(expected):
                raise ValueError("wrong number of reviewed segments")
            reviewed = {}
            for value in values:
                identifier = value["id"]
                translated = value["revised_translated_text"]
                if (identifier in reviewed or identifier not in expected
                        or not isinstance(translated, str) or not translated.strip()):
                    raise ValueError("invalid reviewed segment")
                reviewed[identifier] = translated.strip()
            if set(reviewed) != set(expected):
                raise ValueError("missing reviewed segment ID")
            return [(identifier, reviewed[identifier]) for identifier in expected]
        except Exception as exc:
            raise LocalizationProviderError(f"Gemini register review failed: {exc}") from exc

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
            "Rewrite one Vietnamese dubbing line so it can be spoken in less time. The output "
            "MUST differ materially from the current line when it is too long; do not return it unchanged. "
            "Compress wording, never facts: preserve every source clause, rule number, "
            "condition, exception, status change, relationship, name, number, unit, negation, and key "
            "detail. Preserve the current Vietnamese pronouns, honorifics, names, and relationship "
            "register; never switch pronoun pair or register merely to save time. Use concise natural spoken Vietnamese; do not summarize away a clause "
            "or add commentary. Keep idioms and slang idiomatic in Vietnamese and never "
            "regress to literal Chinese calques. Return only the rewritten Vietnamese "
            f"line. Aim for at most {target_ratio:.0%} of the measured spoken length; this is a "
            "speech-time target, not a word-count exercise.\n"
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

    def shorten_translations(
        self, *, segments, source_language: str, target_language: str,
    ) -> list[str]:
        """Shorten a batch of Vietnamese dubbing lines in one JSON request."""
        if source_language not in {"en", "zh"} or target_language != "vi":
            raise ValueError("Gemini shortening requires English/Chinese to Vietnamese")
        if not segments:
            return []
        payload_segments = []
        expected_ids = []
        target_word_counts = {}
        minimum_word_counts = {}
        for item in segments:
            identifier = item["id"]
            if not isinstance(identifier, str) or not identifier or identifier in expected_ids:
                raise ValueError("Batch shortening requires unique non-empty segment IDs")
            source_text = item["source_text"]
            translated_text = item["translated_text"]
            required_rate = item["required_rate"]
            max_playback_rate = item["max_playback_rate"]
            audio_duration = item.get("audio_duration")
            available_duration = item.get("available_duration")
            if (not isinstance(source_text, str) or not source_text.strip()
                    or not isinstance(translated_text, str) or not translated_text.strip()
                    or not isinstance(required_rate, (int, float)) or required_rate <= 0
                    or not isinstance(max_playback_rate, (int, float))
                    or max_playback_rate <= 0):
                raise ValueError("Batch shortening received an invalid segment")
            expected_ids.append(identifier)
            # Word count is only an estimate of synthesized duration.  Use the
            # timing ratio itself (rounded to a practical whole-word budget),
            # then let measured TTS duration drive the bounded recovery loop.
            # An additional arbitrary compression factor makes mild overflows
            # needlessly hard to satisfy and can discard meaning before TTS.
            current_word_count = len(translated_text.split())
            target_word_count = max(2, int(
                current_word_count * (max_playback_rate / required_rate) + 0.5
            ))
            # If measured audio already exceeds the hard playback ceiling,
            # a target equal to the current word count cannot make progress
            # and can waste multiple Gemini retries.  For any line with room
            # to compress, require at least one fewer word.
            if required_rate > max_playback_rate and current_word_count > 2:
                target_word_count = min(target_word_count, current_word_count - 1)
            # A word budget is only a timing estimate, so retain enough room to
            # express all source clauses rather than treating a much shorter
            # summary as a successful timing recovery.
            minimum_word_count = max(2, math.ceil(target_word_count * 0.60))
            target_word_counts[identifier] = target_word_count
            minimum_word_counts[identifier] = minimum_word_count
            payload_segments.append({
                "id": identifier,
                "source_text": source_text.strip(),
                "current_vietnamese": translated_text.strip(),
                "required_rate": required_rate,
                "max_playback_rate": max_playback_rate,
                "target_length_ratio": max_playback_rate / required_rate,
                **({"target_max_seconds": round(float(available_duration) * float(max_playback_rate), 3)}
                   if isinstance(available_duration, (int, float)) else {}),
                "minimum_word_count": minimum_word_count,
                "target_word_count": target_word_count,
                **({"measured_audio_seconds": round(float(audio_duration), 3)}
                   if isinstance(audio_duration, (int, float)) else {}),
                **({"available_seconds": round(float(available_duration), 3)}
                   if isinstance(available_duration, (int, float)) else {}),
            })
        payload_by_id = {item["id"]: item for item in payload_segments}
        accepted = {}
        pending_ids = list(expected_ids)
        saw_usable_item = False
        for validation_attempt in range(2):
            pending_segments = [payload_by_id[identifier] for identifier in pending_ids]
            instructions = (
                "Rewrite each Vietnamese dubbing line to fit its timing target. Compress wording, "
                "never facts: preserve every source clause, rule number, condition, exception, "
                "status change, relationship, name, number, unit, negation, and key detail. Do not summarize away "
                "a clause. Preserve the current Vietnamese pronouns, honorifics, names, and relationship "
                "register; never switch pronoun pair or register merely to save time. Keep Chinese idioms and slang idiomatic in Vietnamese; never replace a "
                "natural translation with a literal calque. Use concise natural spoken Vietnamese; "
                "measured_audio_seconds, available_seconds, and target_max_seconds are the timing "
                "evidence: optimize for actual speech seconds, not word-count estimates. If a line "
                "is too long, its shortened_text MUST be materially different from current_vietnamese. "
                "Never omit a source clause or factual detail. "
                "do not add commentary or drop, "
                "combine, reorder, or split entries. Respect each entry target_word_count "
                "as a hard maximum and minimum_word_count as a hard minimum. Return JSON only: an object with a "
                "'segments' array, each containing its unchanged 'id' and non-empty "
                "'shortened_text'."
            )
            if validation_attempt:
                instructions += (
                    " STRICT RETRY: Copy every input id exactly once. Do not invent, "
                    "omit, duplicate, rename, reorder, or alter any id. Each shortened_text "
                    "MUST have between its minimum_word_count and target_word_count "
                    "whitespace-separated words, inclusive."
                )
            prompt = (
                instructions
                + " Input JSON:\n"
                + json.dumps({"source_language": source_language, "segments": pending_segments},
                             ensure_ascii=False)
            )
            response_json_schema = {
                "type": "object",
                "properties": {"segments": {
                    "type": "array", "minItems": len(pending_ids),
                    "maxItems": len(pending_ids), "items": {
                        "type": "object", "properties": {
                            "id": {"type": "string"},
                            "shortened_text": {"type": "string"},
                        },
                        "required": ["id", "shortened_text"],
                        "additionalProperties": False,
                    },
                }},
                "required": ["segments"], "additionalProperties": False,
            }
            try:
                response = self._generate_content_with_retry(
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": response_json_schema,
                    },
                )
            except Exception as exc:
                raise LocalizationProviderError(
                    f"Gemini batch translation shortening failed: {exc}"
                ) from exc
            try:
                document = json.loads(response.text)
                items = document["segments"]
                if not isinstance(items, list):
                    raise ValueError("shortened segments is not a list")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A malformed envelope cannot be partially recovered. Keep the
                # same pending IDs for the second structured attempt.
                items = None
            if items is not None:
                occurrences = {}
                for item in items:
                    if isinstance(item, dict) and item.get("id") in pending_ids:
                        occurrences.setdefault(item["id"], []).append(item)
                # A matching ID means the response can be recovered at the
                # segment level, even if its text or word count is invalid.
                # Do not fan a wholly malformed/foreign response out into a
                # request for every original segment.
                saw_usable_item = saw_usable_item or bool(occurrences)
                for identifier in pending_ids:
                    candidates = occurrences.get(identifier, [])
                    if len(candidates) != 1:
                        continue
                    shortened = candidates[0].get("shortened_text")
                    if not isinstance(shortened, str) or not shortened.strip():
                        continue
                    shortened = shortened.strip()
                    if not _has_speakable_tts_text(shortened):
                        continue
                    if shortened.casefold() == payload_by_id[identifier]["current_vietnamese"].casefold():
                        continue
                    word_count = len(shortened.split())
                    if not minimum_word_counts[identifier] <= word_count <= target_word_counts[identifier]:
                        continue
                    accepted[identifier] = shortened
                pending_ids = [identifier for identifier in pending_ids if identifier not in accepted]
            if not pending_ids:
                return [accepted[identifier] for identifier in expected_ids]
            if validation_attempt == 0:
                logger.warning(
                    "Gemini batch shortening returned unresolved entries; retrying %d item(s)",
                    len(pending_ids),
                )

        if not saw_usable_item:
            raise LocalizationProviderError(
                "Gemini batch translation shortening returned unusable structured output after retry"
            )

        for identifier in pending_ids:
            accepted[identifier] = self._shorten_translation_strict(
                singleton=payload_by_id[identifier], source_language=source_language,
                target_word_count=target_word_counts[identifier],
                minimum_word_count=minimum_word_counts[identifier],
            )
        return [accepted[identifier] for identifier in expected_ids]

    def _shorten_translation_strict(
        self, *, singleton, source_language: str, target_word_count: int,
        minimum_word_count: int,
    ) -> str:
        strict_prompt = (
            "Rewrite this one Vietnamese dubbing line with a word count between "
            f"{minimum_word_count} and {target_word_count}, inclusive: it MUST contain at least "
            f"{minimum_word_count} and no more than {target_word_count} whitespace-separated "
            "Vietnamese words. These are HARD bounds: count the words before responding. "
            "Compress wording, never facts: preserve every source clause, rule number, condition, "
            "exception, relationship, name, number, unit, negation, and key detail; do not summarize "
            "away a clause. Preserve the current Vietnamese pronouns, honorifics, names, and relationship "
            "register; never switch pronoun pair or register merely to save time. Use concise natural spoken Vietnamese. "
            "Your output MUST contain speakable Vietnamese content: at least one letter or "
            "number. Never return punctuation-only text. "
            "Return only the rewritten Vietnamese line, with no JSON, labels, or commentary.\n"
            f"Source ({source_language}): {singleton['source_text']}\n"
            f"Current Vietnamese: {singleton['current_vietnamese']}"
        )
        previous_candidate = None
        best_overlong_candidate = None
        best_overlong_word_count = None
        # The batch retries above already protect the structured protocol.  A
        # singleton gets a few more chances because its response can be
        # directed at the concrete overlong line rather than the original.
        for strict_attempt in range(4):
            request_prompt = strict_prompt
            if previous_candidate is not None:
                request_prompt += (
                    "\nYour previous candidate below has "
                    f"{len(previous_candidate.split())} whitespace-separated words, "
                    f"which is outside the required range of {minimum_word_count} to "
                    f"{target_word_count} words. Rewrite THAT candidate to fit both bounds. "
                    "Compress wording, never facts: preserve every source clause, rule number, "
                    "condition, exception, relationship, name, number, unit, and negation. Preserve the "
                    "current Vietnamese pronouns, honorifics, and relationship register; never switch "
                    "pronoun pair or register merely to save time. "
                    "Return only the replacement Vietnamese "
                    "line, with no JSON, labels, or commentary.\n"
                    f"Previous candidate: {previous_candidate}"
                )
            try:
                response = self._generate_content_with_retry(contents=request_prompt)
                shortened = getattr(response, "text", None)
                if not isinstance(shortened, str) or not shortened.strip():
                    raise ValueError("empty shortened translation")
                shortened = shortened.strip()
                try:
                    json.loads(shortened)
                except json.JSONDecodeError:
                    pass
                else:
                    raise ValueError("invalid singleton shortened translation")
                if shortened.startswith("```") or re.match(
                    r"^(?:shortened_text|translation|vietnamese(?: translation)?|"
                    r"bản dịch|câu trả lời|kết quả)\s*[:：]",
                    shortened,
                    re.IGNORECASE,
                ):
                    raise ValueError("invalid singleton shortened translation")
                if not _has_speakable_tts_text(shortened):
                    raise ValueError("shortened translation has no speakable content")
                word_count = len(shortened.split())
                if word_count > target_word_count:
                    if (best_overlong_word_count is None
                            or word_count <= best_overlong_word_count):
                        best_overlong_candidate = shortened
                        best_overlong_word_count = word_count
                    previous_candidate = shortened
                    raise ValueError("shortened translation exceeds target word count")
                if word_count < minimum_word_count:
                    previous_candidate = shortened
                    raise ValueError("shortened translation is below minimum word count")
                return shortened
            except ValueError as exc:
                validation_error = exc
                if strict_attempt < 3:
                    logger.warning(
                        "Gemini singleton shortening returned invalid output; retrying (%d/3)",
                        strict_attempt + 1,
                    )
            except Exception as exc:
                raise LocalizationProviderError(
                    f"Gemini batch translation shortening failed: {exc}"
                ) from exc
        if best_overlong_candidate is not None:
            logger.warning(
                "Gemini singleton shortening remained above estimated word target; "
                "using best valid overlong candidate for measured TTS timing"
            )
            return best_overlong_candidate
        raise LocalizationProviderError(
            f"Gemini batch translation shortening failed: {validation_error}"
        ) from validation_error



@dataclass
class EdgeTTSProvider:
    default_voice: str = "vi-VN-HoaiMyNeural"
    max_concurrency: int = 1
    retry_delays: tuple[float, ...] | None = None
    # Kept deliberately small: one failed request should not invalidate a large
    # amount of otherwise recoverable localization work.
    batch_size: int = 4
    batch_max_characters: int = 6000
    batch_enabled: bool = True

    # NoAudio handling retains the legacy service-throttle schedule. Generic
    # transport failures use generic_transient_retry_delays below instead.
    transient_retry_delays = (5, 10, 20, 40, 60, 90, 120)
    # A generic transport failure must return control to the workflow quickly:
    # it holds the process-wide TTS resource while this provider call runs.
    # Keep one short in-call retry, then let the workflow wait outside that
    # resource gate.
    generic_transient_retry_delays = (1.0,)
    # Empty audio is usually a service-side throttle or a temporarily unhealthy
    # synthesis route, rather than a dropped socket.  These seven finite waits
    # total 415 seconds (and each is capped at 120 seconds), which is more
    # conservative than the generic 345-second schedule without turning one
    # segment into an unbounded or hours-long recovery attempt.
    no_audio_retry_delays = (10, 20, 40, 60, 75, 90, 120)
    max_retry_delay_seconds = 120.0
    pacing_min_seconds = 1.5
    pacing_max_seconds = 2.5

    # The scheduler serializes normal localization work globally, but this
    # process-wide pacer also covers separately-created providers and direct
    # callers.  It intentionally records only real network attempts, never
    # verified cache reuse.
    _network_pacing_lock: ClassVar[threading.Lock] = threading.Lock()
    _last_network_request_monotonic: ClassVar[float | None] = None
    _service_circuit_lock: ClassVar[threading.Lock] = threading.Lock()
    _service_circuit_until: ClassVar[float] = 0.0
    service_cooldown_seconds: ClassVar[float] = 60.0
    max_no_audio_attempts_per_payload: ClassVar[int] = 2
    degraded_payload_retry_after_seconds: ClassVar[float] = 12.0
    generic_transient_retry_after_seconds: ClassVar[float] = 12.0
    batch_disable_after_failures: ClassVar[int] = 2

    @staticmethod
    def _save_with_attempt_timeout(
        text: str, voice: str, options: dict[str, str], output_file: str,
    ) -> None:
        """Save one segment behind a process boundary with a hard deadline."""
        _run_edge_tts_attempt(
            text, voice, options, output_file, batch=False,
            timeout_seconds=EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS,
        )

    def __post_init__(self) -> None:
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("Edge TTS max_concurrency must be at least 1")
        if not isinstance(self.batch_enabled, bool):
            raise ValueError("Edge TTS batch_enabled must be a boolean")
        if not 2 <= self.batch_size <= 5:
            raise ValueError("Edge TTS batch_size must be between 2 and 5")
        if self.retry_delays is None:
            self.retry_delays = self.transient_retry_delays
        else:
            self.retry_delays = tuple(float(delay) for delay in self.retry_delays)
            if any(delay < 0 for delay in self.retry_delays):
                raise ValueError("Edge TTS retry delays cannot be negative")
        self._synthesis_slots = threading.BoundedSemaphore(self.max_concurrency)
        self._batch_failure_count = 0
        self._batch_disabled = False

    def _record_batch_failure(self) -> None:
        self._batch_failure_count += 1
        if self._batch_failure_count >= self.batch_disable_after_failures:
            if not self._batch_disabled:
                logger.warning(
                    "Edge TTS batch disabled for this provider after %d failures; "
                    "using individually gated requests for the rest of the run",
                    self._batch_failure_count,
                )
            self._batch_disabled = True

    def _pacing_jitter_seconds(self) -> float:
        """Return one test-overridable quiet interval before a fresh request."""
        return random.uniform(self.pacing_min_seconds, self.pacing_max_seconds)

    @staticmethod
    def _pacing_sleep(seconds: float) -> None:
        """Sleep for pacing; isolated so tests do not use wall-clock delays."""
        time.sleep(seconds)

    @staticmethod
    def _pacing_monotonic() -> float:
        """Return a pacing clock isolated from asyncio's global clock."""
        return time.monotonic()

    def _pace_before_network_synthesis(self) -> None:
        """Space fresh Edge requests without delaying verified cache hits."""
        with EdgeTTSProvider._network_pacing_lock:
            interval = self._pacing_jitter_seconds()
            previous = EdgeTTSProvider._last_network_request_monotonic
            now = self._pacing_monotonic()
            if previous is not None:
                remaining = interval - (now - previous)
                if remaining > 0:
                    self._pacing_sleep(remaining)
                    now = self._pacing_monotonic()
            EdgeTTSProvider._last_network_request_monotonic = now

    @classmethod
    def _circuit_remaining_seconds(cls) -> float:
        with cls._service_circuit_lock:
            return max(0.0, cls._service_circuit_until - time.monotonic())

    @classmethod
    def _open_service_circuit(cls) -> float:
        with cls._service_circuit_lock:
            cls._service_circuit_until = max(
                cls._service_circuit_until, time.monotonic() + cls.service_cooldown_seconds,
            )
            return max(1.0, cls._service_circuit_until - time.monotonic())

    @classmethod
    def _clear_service_circuit(cls) -> None:
        with cls._service_circuit_lock:
            cls._service_circuit_until = 0.0

    def _raise_if_service_circuit_open(self) -> None:
        remaining = self._circuit_remaining_seconds()
        if remaining:
            raise TTSServiceTemporarilyUnavailable(
                remaining, "Edge TTS is temporarily unavailable; retry shortly."
            )

    def _health_probe(self, voice: str, options: dict[str, str]) -> None:
        """Make one inexpensive request to distinguish content trouble from outage."""
        probe = Path(tempfile.gettempdir()) / f".edge-tts-health-{uuid.uuid4().hex}.mp3"
        try:
            self._raise_if_service_circuit_open()
            self._pace_before_network_synthesis()
            self._save_with_attempt_timeout(
                "Xin chào, đây là câu kiểm tra bình thường.", voice, options, str(probe),
            )
            if not probe.is_file() or probe.stat().st_size <= 0:
                raise _EdgeTTSAttemptError("NoAudioReceived", "health probe returned no audio")
            self._clear_service_circuit()
        except (TimeoutError, _EdgeTTSAttemptError) as exc:
            if _is_transient_edge_attempt_error(exc):
                raise TTSServiceTemporarilyUnavailable(self._open_service_circuit()) from exc
            raise
        except Exception as exc:
            # edge_tts exception classes are imported in synthesis paths; their
            # names remain reliable here without retaining a module global.
            if type(exc).__name__ in _EDGE_TTS_TRANSIENT_CHILD_ERRORS:
                raise TTSServiceTemporarilyUnavailable(self._open_service_circuit()) from exc
            raise
        finally:
            probe.unlink(missing_ok=True)

    def _retry_delay_seconds(
        self, attempt: int, exc: Exception, *, no_audio: bool
    ) -> float:
        """Return a bounded retry delay, escalating empty-audio failures."""
        normal_delay = self.retry_delays[attempt - 1]
        if no_audio:
            no_audio_delay = self.no_audio_retry_delays[
                min(attempt - 1, len(self.no_audio_retry_delays) - 1)
            ]
            normal_delay = max(normal_delay, no_audio_delay)
        retry_after = self._retry_after_seconds(exc) or 0.0
        return min(
            self.max_retry_delay_seconds,
            max(normal_delay, retry_after),
        )

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

    def _cached_payload_result(self, request: TTSRequest, payload: str) -> TTSResult | None:
        """Find a cache entry by the text actually submitted to Edge."""
        try:
            import edge_tts
        except ImportError:
            return None
        voice = request.voice or self.default_voice
        options = dict(request.options)
        payload_request = replace(request, text=payload)
        output = Path(f"{request.output_file_stem}.mp3")
        fingerprint = self._cache_fingerprint(
            payload_request, voice, str(options.get("rate", "+0%")),
            str(options.get("volume", "+0%")), str(options.get("pitch", "+0Hz")),
            getattr(edge_tts, "__version__", None),
        )
        return self._read_cached_result(
            output, self._cache_paths(output)[1], fingerprint, voice,
        )

    def _generic_transient_retry_delay_seconds(self) -> float:
        """Return one short retry delay for a non-NoAudio transport failure."""
        return self.generic_transient_retry_delays[0]

    def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize, retrying a conservative payload only after no audio."""
        if request.language != "vi":
            raise ValueError("EdgeTTSProvider requires Vietnamese ('vi')")
        if not request.text.strip():
            raise ValueError("TTS text cannot be empty")
        if not _has_speakable_tts_text(request.text):
            raise ValueError("TTS text must contain at least one letter or number")
        payloads = (request.text, *_edge_tts_safe_payloads(request.text))
        for payload in payloads:
            cached = self._cached_payload_result(request, payload)
            if cached is not None:
                logger.info("Reusing verified Edge TTS checkpoint for %s",
                            Path(cached.audio_file).name)
                return cached
        self._raise_if_service_circuit_open()
        for payload in payloads:
            try:
                # All candidate payload caches were checked above.  Do not probe
                # them again here: validation can fail transiently (for example,
                # while replacing corrupt checkpointed audio), which could make a
                # second probe incorrectly reuse that same stale file.
                return self._synthesize_payload(
                    replace(request, text=payload), check_cache=False
                )
            except _EdgeTTSNoAudioExhausted:
                if payload == payloads[-1]:
                    voice = request.voice or self.default_voice
                    options = dict(request.options)
                    self._health_probe(voice, {
                        "rate": str(options.get("rate", "+0%")),
                        "volume": str(options.get("volume", "+0%")),
                        "pitch": str(options.get("pitch", "+0Hz")),
                        "connect_timeout": EDGE_TTS_CONNECT_TIMEOUT_SECONDS,
                        "receive_timeout": EDGE_TTS_RECEIVE_TIMEOUT_SECONDS,
                    })
                    # Edge is reachable, but this payload is temporarily
                    # rejected. Let the workflow release its shared TTS gate
                    # before it waits and retries this segment.
                    raise TTSServiceTemporarilyUnavailable(
                        self.degraded_payload_retry_after_seconds,
                        "Edge TTS temporarily rejected this payload; retry shortly.",
                    )
                logger.warning("Edge TTS received no audio for one payload; trying "
                               "a conservative TTS-safe variant")
        raise AssertionError("unreachable")  # pragma: no cover

    def _synthesize_payload(
        self, request: TTSRequest, *, check_cache: bool = True,
        retry_delays: tuple[float, ...] | None = None,
        max_no_audio_attempts: int | None = None,
    ) -> TTSResult:
        if request.language != "vi":
            raise ValueError("EdgeTTSProvider requires Vietnamese ('vi')")
        if not request.text.strip():
            raise ValueError("TTS text cannot be empty")
        if not _has_speakable_tts_text(request.text):
            raise ValueError("TTS text must contain at least one letter or number")

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
        if check_cache:
            cached = self._read_cached_result(
                output_path, checkpoint_path, fingerprint, voice
            )
            if cached is not None:
                logger.info("Reusing verified Edge TTS checkpoint for %s", output_path.name)
                return cached
        self._raise_if_service_circuit_open()

        # A stale checkpoint must not survive a failed replacement and later be
        # mistaken for the new request.
        output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        retry_delays = self.retry_delays if retry_delays is None else retry_delays
        max_attempts = max(len(retry_delays) + 1, 2)
        max_no_audio_attempts = (
            self.max_no_audio_attempts_per_payload
            if max_no_audio_attempts is None else max_no_audio_attempts
        )
        consecutive_no_audio = 0
        with self._synthesis_slots:
            for attempt in range(1, max_attempts + 1):
                output_path.unlink(missing_ok=True)
                try:
                    self._pace_before_network_synthesis()
                    self._save_with_attempt_timeout(request.text, voice, {
                        "rate": rate, "volume": volume, "pitch": pitch,
                        "connect_timeout": EDGE_TTS_CONNECT_TIMEOUT_SECONDS,
                        "receive_timeout": EDGE_TTS_RECEIVE_TIMEOUT_SECONDS,
                    }, output_file)
                    if output_path.is_file() and output_path.stat().st_size > 0:
                        break
                    raise NoAudioReceived(
                        "Edge TTS returned no audio data."
                    )
                except (NoAudioReceived, WebSocketError, TimeoutError,
                        _EdgeTTSAttemptError) as exc:
                    output_path.unlink(missing_ok=True)
                    if not _is_transient_edge_attempt_error(exc):
                        details = self._safe_exception_details(exc)
                        logger.error(
                            "Edge TTS non-transient failure for voice %s: %s",
                            voice, details,
                        )
                        raise LocalizationProviderError(
                            f"Edge TTS failed: {details}"
                        ) from exc
                    if isinstance(exc, NoAudioReceived) or (
                        isinstance(exc, _EdgeTTSAttemptError)
                        and exc.error_type == "NoAudioReceived"
                    ):
                        consecutive_no_audio += 1
                        if consecutive_no_audio >= 2 and _is_proxifier_running():
                            raise LocalizationProviderError(
                                "Edge TTS received no audio repeatedly while Proxifier is "
                                "running. Proxifier or its VPN proxy can block Edge TTS; "
                                "disable Proxifier and try again."
                            ) from exc
                        if consecutive_no_audio >= max_no_audio_attempts:
                            raise _EdgeTTSNoAudioExhausted(
                                f"Edge TTS returned no audio after {consecutive_no_audio} attempts"
                            ) from exc
                    else:
                        consecutive_no_audio = 0
                    details = self._safe_exception_details(exc)
                    no_audio = isinstance(exc, NoAudioReceived) or (
                        isinstance(exc, _EdgeTTSAttemptError)
                        and exc.error_type == "NoAudioReceived"
                    )
                    if not no_audio and attempt >= 2:
                        logger.warning(
                            "Edge TTS generic transient failure exhausted short retry "
                            "budget for voice %s: %s", voice, details,
                        )
                        raise TTSServiceTemporarilyUnavailable(
                            self.generic_transient_retry_after_seconds,
                            "Edge TTS is temporarily unavailable; retry shortly.",
                        ) from exc
                    if attempt >= max_attempts:
                        output_path.unlink(missing_ok=True)
                        logger.error(
                            "Edge TTS final failure (%d/%d) for voice %s: %s",
                            attempt, max_attempts, voice, details,
                        )
                        failure = (
                            f"Edge TTS failed after {max_attempts} attempts: {details}"
                        )
                        if isinstance(exc, NoAudioReceived) or (
                            isinstance(exc, _EdgeTTSAttemptError)
                            and exc.error_type == "NoAudioReceived"
                        ):
                            raise _EdgeTTSNoAudioExhausted(failure) from exc
                        raise LocalizationProviderError(failure) from exc
                    delay = (self._retry_delay_seconds(attempt, exc, no_audio=True)
                             if no_audio else self._generic_transient_retry_delay_seconds())
                    # Candidate payloads are deliberately isolated quickly;
                    # prolonged outage waiting happens at workflow level after
                    # the one health probe opens the shared circuit.
                    if no_audio:
                        delay = min(delay, 1.0)
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
        self._clear_service_circuit()
        return result

    @staticmethod
    def _batch_markers(count: int, forbidden_text: str = "") -> list[str]:
        # Keep separators short because Edge speaks them. Six alphanumeric
        # characters are returned as one exact WordBoundary in edge-tts 7.2.8.
        # Regenerate if a token happens to occur in user text.
        for _ in range(32):
            nonce = uuid.uuid4().hex[:2].upper()
            markers = [f"ZXQ{nonce}{index}" for index in range(count - 1)]
            if all(marker not in forbidden_text for marker in markers):
                return markers
        raise ValueError("Unable to create unique Edge TTS batch markers")

    @staticmethod
    def _batch_intervals_from_boundaries(
        markers: Sequence[str], boundaries: Sequence[Mapping[str, object]],
        audio_duration: float,
    ) -> list[tuple[float, float]]:
        """Convert exact marker WordBoundary timings (100ns ticks) into cuts."""
        positions: list[tuple[float, float]] = []
        for marker in markers:
            matches = [item for item in boundaries if item.get("text") == marker]
            if len(matches) != 1:
                raise ValueError("Edge TTS batch marker boundary is missing or duplicated")
            offset, duration = matches[0].get("offset"), matches[0].get("duration")
            if not isinstance(offset, int) or not isinstance(duration, int) or duration <= 0:
                raise ValueError("Edge TTS batch marker boundary is invalid")
            positions.append((offset / 10_000_000, (offset + duration) / 10_000_000))
        if any(end <= start for start, end in positions) or any(
            positions[index][1] >= positions[index + 1][0]
            for index in range(len(positions) - 1)
        ):
            raise ValueError("Edge TTS batch markers are out of order")
        intervals = [(0.0, positions[0][0])] if positions else [(0.0, audio_duration)]
        intervals.extend((positions[index][1], positions[index + 1][0])
                         for index in range(len(positions) - 1))
        if positions:
            intervals.append((positions[-1][1], audio_duration))
        if any(start < 0 or end <= start or end > audio_duration + 0.05
               for start, end in intervals):
            raise ValueError("Edge TTS batch produced invalid segment intervals")
        return intervals

    @staticmethod
    def _stream_batch_with_attempt_timeout(
        text: str, voice: str, options: dict[str, str], output: Path,
    ) -> list[dict[str, object]]:
        return _run_edge_tts_attempt(
            text, voice, options, str(output), batch=True,
            timeout_seconds=EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _split_batch_audio(combined: Path, output: Path, start: float, end: float) -> None:
        # Decode/re-encode instead of MP3 packet copying: marker boundaries are
        # sub-frame timings and packet copying can leave audible marker fragments.
        _run_ffmpeg(["ffmpeg", "-y", "-nostdin", "-ss", f"{start:.7f}",
                     "-to", f"{end:.7f}", "-i", str(combined), "-vn",
                     "-c:a", "libmp3lame", "-q:a", "2", str(output)])

    def synthesize_batch(self, requests: Sequence[TTSRequest]) -> list[TTSResult]:
        """Synthesize a small cache-miss batch, falling back safely to singles."""
        requests = list(requests)
        if len(requests) <= 1:
            return [self.synthesize(request) for request in requests]
        if not self.batch_enabled:
            raise TTSBatchFallbackRequired(
                "Edge TTS batch is disabled by configuration"
            )
        if self._batch_disabled:
            raise TTSBatchFallbackRequired(
                "Edge TTS batch is disabled after repeated failures"
            )
        if len(requests) > self.batch_size or sum(len(request.text) for request in requests) > self.batch_max_characters:
            # Keep the caller's result ordering while respecting a conservative
            # request-size ceiling.
            midpoint = min(self.batch_size, len(requests))
            return self.synthesize_batch(requests[:midpoint]) + self.synthesize_batch(requests[midpoint:])
        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived, WebSocketError
            for request in requests:
                if request.language != "vi" or not _has_speakable_tts_text(request.text):
                    raise ValueError("EdgeTTSProvider requires speakable Vietnamese text")
            voice = requests[0].voice or self.default_voice
            options = dict(requests[0].options)
            rate, volume, pitch = (str(options.get(name, default)) for name, default in
                                   (("rate", "+0%"), ("volume", "+0%"), ("pitch", "+0Hz")))
            if any((request.voice or self.default_voice) != voice or dict(request.options) != options
                   for request in requests):
                raise ValueError("Edge TTS batch requests must have matching voice and options")
            # The provider cache is independent from the localization manifest.
            # A resumed run with a missing manifest must still avoid a network
            # request for every valid per-segment Edge cache entry.
            cached_results: list[TTSResult | None] = []
            for request in requests:
                output = Path(f"{request.output_file_stem}.mp3")
                fingerprint = self._cache_fingerprint(
                    request, voice, rate, volume, pitch,
                    getattr(edge_tts, "__version__", None),
                )
                cached_results.append(self._read_cached_result(
                    output, self._cache_paths(output)[1], fingerprint, voice))
            if any(result is not None for result in cached_results):
                misses = [request for request, result in zip(requests, cached_results)
                          if result is None]
                generated = self.synthesize_batch(misses) if misses else []
                generated_iter = iter(generated)
                return [result if result is not None else next(generated_iter)
                        for result in cached_results]
            self._raise_if_service_circuit_open()
            source_text = " ".join(request.text for request in requests)
            markers = self._batch_markers(len(requests), source_text)
            parts = []
            for index, request in enumerate(requests):
                parts.append(request.text)
                if index < len(markers):
                    parts.append(markers[index])
            combined_text = " ".join(parts)
            parent = Path(requests[0].output_file_stem).parent
            combined = parent / f".edge-batch-{uuid.uuid4().hex}.mp3"
            boundaries = None
            max_attempts = max(len(self.retry_delays) + 1, 2)
            consecutive_no_audio = 0
            with self._synthesis_slots:
                for attempt in range(1, max_attempts + 1):
                    combined.unlink(missing_ok=True)
                    try:
                        self._pace_before_network_synthesis()
                        boundaries = self._stream_batch_with_attempt_timeout(
                            combined_text, voice, {
                                "rate": rate, "volume": volume, "pitch": pitch,
                                "boundary": "WordBoundary",
                                "connect_timeout": EDGE_TTS_CONNECT_TIMEOUT_SECONDS,
                                "receive_timeout": EDGE_TTS_RECEIVE_TIMEOUT_SECONDS,
                            }, combined,
                        )
                        if combined.is_file() and combined.stat().st_size > 0:
                            break
                        raise NoAudioReceived("Edge TTS returned no batch audio data.")
                    except (NoAudioReceived, WebSocketError, TimeoutError,
                            _EdgeTTSAttemptError) as exc:
                        combined.unlink(missing_ok=True)
                        details = self._safe_exception_details(exc)
                        if not _is_transient_edge_attempt_error(exc):
                            raise ValueError(details) from exc
                        no_audio = isinstance(exc, NoAudioReceived) or (
                            isinstance(exc, _EdgeTTSAttemptError)
                            and exc.error_type == "NoAudioReceived"
                        )
                        if no_audio:
                            logger.warning(
                                "Edge TTS batch returned no audio for voice %s; "
                                "releasing the TTS resource and falling back to singles",
                                voice,
                            )
                            raise TTSBatchFallbackRequired(
                                "Edge TTS batch returned no audio; retry as individually gated requests"
                            ) from exc
                        logger.warning(
                            "Edge TTS batch transient failure for voice %s: %s; "
                            "releasing the TTS resource before retry",
                            voice, details,
                        )
                        raise TTSServiceTemporarilyUnavailable(
                            self.generic_transient_retry_after_seconds,
                            "Edge TTS is temporarily unavailable; retry shortly.",
                        ) from exc
                duration = _probe_audio_duration(combined)
            intervals = self._batch_intervals_from_boundaries(markers, boundaries or [], duration)
            results = []
            for request, (start, end) in zip(requests, intervals):
                output = Path(f"{request.output_file_stem}.mp3")
                output.unlink(missing_ok=True)
                self._split_batch_audio(combined, output, start, end)
                _trim_edge_tts_silence(output)
                segment_duration = _probe_audio_duration(output)
                result = TTSResult(str(output), segment_duration, "edge-tts", voice,
                    {"format": "mp3", "rate": rate, "volume": volume, "pitch": pitch})
                fingerprint = self._cache_fingerprint(request, voice, rate, volume, pitch,
                    getattr(edge_tts, "__version__", None))
                self._write_checkpoint(self._cache_paths(output)[1], fingerprint, result)
                results.append(result)
            logger.info(
                "Edge TTS batch synthesized %d segments in one network request",
                len(requests),
            )
            self._clear_service_circuit()
            return results
        except TTSBatchFallbackRequired:
            self._record_batch_failure()
            raise
        except TTSServiceTemporarilyUnavailable:
            raise
        except (NoAudioReceived, WebSocketError, TimeoutError,
                _EdgeTTSAttemptError) as exc:
            logger.warning(
                "Edge TTS batch failed transiently; releasing the TTS resource "
                "before singleton fallback: %s", self._safe_exception_details(exc),
            )
            self._record_batch_failure()
            raise TTSBatchFallbackRequired(
                "Edge TTS batch failed; retry as individually gated requests"
            ) from exc
        except Exception as exc:
            logger.warning(
                "Edge TTS batch was unusable; releasing the TTS resource "
                "before singleton fallback: %s", self._safe_exception_details(exc),
            )
            self._record_batch_failure()
            raise TTSBatchFallbackRequired(
                "Edge TTS batch was unusable; retry as individually gated requests"
            ) from exc
        finally:
            if 'combined' in locals():
                combined.unlink(missing_ok=True)


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
