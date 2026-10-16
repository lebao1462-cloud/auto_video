"""Offline FunASR Paraformer transcription for Chinese localization."""

import gc
import importlib.util
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import queue
import sys
try:
    import psutil  # Optional health signal; no new mandatory dependency.
except ImportError:
    psutil = None
import tempfile
import threading
import time
from typing import Any
import unicodedata

from buzz.localization.audio_mix import AudioMixingCancelled, extract_audio_track
from buzz.localization.transcript import (
    LocalizationSegment, LocalizationTranscript, LocalizationTranscriptionCancelled,
)

PARAFORMER_ENGINE = "paraformer-zh"
_MODEL_DIRECTORIES = {
    "model": "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "iic--punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}
logger = logging.getLogger(__name__)


def _worker_logger() -> logging.Logger:
    """Write only lifecycle/timing details, not transcript content or credentials."""
    worker_logger = logging.getLogger("autovideo.paraformer.worker")
    if worker_logger.handlers:
        return worker_logger
    root = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    path = root / "Buzz" / "Buzz" / "Logs" / "paraformer_worker.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.INFO)
        worker_logger.propagate = False
    except OSError:
        pass
    return worker_logger


def _paraformer_worker(requests, responses) -> None:
    """Spawn-safe worker target.  The model deliberately never enters the GUI process."""
    worker_logger = _worker_logger()
    worker_logger.info("Paraformer worker started pid=%s", os.getpid())
    model = None
    while True:
        request = requests.get()
        if request is None:
            return
        request_id, wav = request
        try:
            if model is None:
                started = time.monotonic()
                worker_logger.info("Loading Paraformer model request=%s", request_id)
                from funasr import AutoModel
                local_models = paraformer_model_paths()
                model = AutoModel(
                    model=(local_models or {}).get("model", PARAFORMER_ENGINE),
                    vad_model=(local_models or {}).get("vad", "fsmn-vad"),
                    punc_model=(local_models or {}).get("punc", "ct-punc-c"),
                    vad_kwargs={"max_single_segment_time": 30000}, device="cpu",
                    ncpu=min(4, max(1, os.cpu_count() or 1)),
                    disable_update=True, disable_pbar=True,
                )
                worker_logger.info("Paraformer model ready elapsed=%.1fs",
                                   time.monotonic() - started)
            started = time.monotonic()
            worker_logger.info("Transcribing request=%s audio_file_bytes=%s",
                               request_id, Path(wav).stat().st_size)
            output = model.generate(
                input=wav, sentence_timestamp=True, return_raw_text=True,
                batch_size_s=60, batch_size_threshold_s=30,
            )
            worker_logger.info("Transcription complete request=%s elapsed=%.1fs",
                               request_id, time.monotonic() - started)
            responses.put((request_id, True, output))
        except Exception as exc:
            worker_logger.exception("Paraformer request failed request=%s", request_id)
            responses.put((request_id, False, f"{type(exc).__name__}: {exc}"))


class _ParaformerWorkerExited(RuntimeError):
    """The isolated model host stopped before replying to its request."""


class _ParaformerWorkerTimedOut(RuntimeError):
    """The isolated model host exceeded the hard per-request time limit."""


class _ParaformerWorkerService:
    """Run one ASR request per isolated spawned model host.

    Releasing the model after every video part avoids retaining native model
    allocations across a long localization run.  The request lock also means a
    cancelled request cannot tear down another request's worker.
    """

    def __init__(self, target=_paraformer_worker, *, timeout_seconds: float = 480.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Paraformer ASR timeout must be positive")
        self._target = target
        self.timeout_seconds = timeout_seconds
        self._process = self._requests = self._responses = None
        self._lifecycle_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._next_request = 0

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._close_locked()
        context = multiprocessing.get_context("spawn")
        self._requests, self._responses = context.Queue(), context.Queue()
        self._process = context.Process(target=self._target, args=(self._requests, self._responses))
        self._process.start()

    def _close_locked(self) -> None:
        process, requests, responses = self._process, self._requests, self._responses
        self._process = self._requests = self._responses = None
        if process is not None:
            if process.is_alive():
                try:
                    requests.put_nowait(None)
                    process.join(0.5)
                except (OSError, ValueError):
                    pass
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1)
            if not process.is_alive():
                process.join()
                process.close()
        for channel in (requests, responses):
            if channel is not None:
                # A terminated worker cannot drain a pipe; never let cleanup
                # delay cancellation while a queue feeder is flushing.
                channel.cancel_join_thread()
                channel.close()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._close_locked()
        gc.collect()

    @staticmethod
    def _exit_description(exitcode: int | None) -> str:
        if exitcode is None:
            return "exit code unavailable"
        if exitcode < 0:
            return f"exit code {exitcode} (Windows NTSTATUS 0x{exitcode & 0xFFFFFFFF:08X})"
        return f"exit code {exitcode}"

    def _unexpected_exit_locked(self, process) -> _ParaformerWorkerExited:
        # Capture this before close(): multiprocessing.Process.close() makes
        # exitcode inaccessible on some Python versions.
        detail = self._exit_description(process.exitcode)
        self._close_locked()
        return _ParaformerWorkerExited(
            "Paraformer worker exited unexpectedly "
            f"({detail}); possible memory or native-resource exhaustion."
        )

    def _generate_once(self, wav: str, cancel_event=None) -> Any:
        with self._lifecycle_lock:
            self._start()
            self._next_request += 1
            request_id = self._next_request
            process, requests, responses = self._process, self._requests, self._responses
            requests.put((request_id, wav))
        # Slow but active 15-minute video ASR must not trip a wall-clock limit.
        # Only silence in worker CPU activity for this many seconds signals a
        # likely native deadlock. Without psutil, use a conservative wall limit.
        last_activity = time.monotonic()
        last_cpu = None
        cpu_process = None
        if psutil is not None:
            try:
                cpu_process = psutil.Process(process.pid)
            except (psutil.Error, OSError):
                pass
        idle_limit = (self.timeout_seconds if cpu_process is not None
                      else max(self.timeout_seconds, 1800.0))
        next_check = 0.0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.shutdown()
                raise LocalizationTranscriptionCancelled("Localization was cancelled during Paraformer transcription")
            now = time.monotonic()
            if cpu_process is not None and now >= next_check:
                next_check = now + 0.5
                try:
                    times = cpu_process.cpu_times()
                    current_cpu = times.user + times.system
                    if last_cpu is None or current_cpu > last_cpu + 0.05:
                        last_activity = now
                    last_cpu = current_cpu
                except (psutil.Error, OSError):
                    cpu_process = None
                    idle_limit = max(self.timeout_seconds, 1800.0)
            if now - last_activity >= idle_limit:
                # No worker activity. Retire the model worker before retrying;
                # keep every completed video part and ASR checkpoint.
                self.shutdown()
                raise _ParaformerWorkerTimedOut(
                    f"Paraformer ASR had no progress for {idle_limit:g} seconds"
                )
            try:
                response_id, ok, value = responses.get(timeout=0.05)
            except queue.Empty:
                with self._lifecycle_lock:
                    # A process can put its response immediately before
                    # exiting; give the pipe one nonblocking final drain.
                    if process is self._process and not process.is_alive():
                        try:
                            response_id, ok, value = responses.get_nowait()
                        except queue.Empty:
                            raise self._unexpected_exit_locked(process)
                    else:
                        continue
            except (EOFError, OSError):
                with self._lifecycle_lock:
                    if process is self._process and not process.is_alive():
                        raise self._unexpected_exit_locked(process)
                raise
            if response_id != request_id:
                # Responses from an already retired worker must never be used
                # for a subsequent request.
                continue
            if not ok:
                raise RuntimeError(value)
            return value

    def generate(self, wav: str, cancel_event=None) -> Any:
        with self._request_lock:
            for attempt in range(2):
                try:
                    value = self._generate_once(wav, cancel_event=cancel_event)
                except (_ParaformerWorkerExited, _ParaformerWorkerTimedOut) as exc:
                    if attempt:
                        raise
                    logger.warning(
                        "Paraformer worker unavailable (%s); retrying same prepared WAV "
                        "once in a fresh process", exc,
                    )
                    continue
                except Exception:
                    # A normal FunASR exception is not retried, but it must
                    # not leave a model host around for a later video part.
                    self.shutdown()
                    raise
                else:
                    # Model memory belongs to this task only.  Do this while
                    # holding _request_lock so no later request can race the
                    # graceful sentinel/queue cleanup.
                    self.shutdown()
                    return value


_PARAFORMER_WORKER = _ParaformerWorkerService()


def reset_paraformer_model_cache() -> None:
    """Stop the worker and release its cached model."""
    _PARAFORMER_WORKER.shutdown()


def modelscope_cache_path() -> Path:
    """Return the cache location without importing ModelScope/FunASR."""
    configured = os.getenv("MODELSCOPE_CACHE")
    if configured:
        return Path(configured)
    if getattr(sys, "frozen", False):
        bundled = Path(sys.executable).resolve().parent / "modelscope-cache"
        if (bundled / "models").is_dir():
            return bundled
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


def transcribe_with_paraformer(video_path: str, *, cancel_event=None) -> LocalizationTranscript:
    source = Path(video_path)
    if not source.is_file():
        raise ValueError(f"Input media file does not exist: {video_path}")
    if not paraformer_runtime_available():
        raise RuntimeError("FunASR runtime is missing. Install funasr==1.4.16 or choose Whisper.")
    if cancel_event is not None and cancel_event.is_set():
        raise LocalizationTranscriptionCancelled("Localization was cancelled during Paraformer transcription")
    configure_modelscope_cache()  # must precede the lazy FunASR import
    temp_dir = _temporary_directory()
    wav = temp_dir / f"autovideo-paraformer-{next(tempfile._get_candidate_names())}.wav"
    try:
        extract_audio_track(
            source, wav, sample_rate=16000, channels=1, cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise LocalizationTranscriptionCancelled("Localization was cancelled during Paraformer transcription")
        generated = _PARAFORMER_WORKER.generate(str(wav), cancel_event=cancel_event)
        result = generated[0] if isinstance(generated, list) and generated else generated
        if not isinstance(result, dict):
            raise RuntimeError("FunASR returned an invalid transcription result.")
        segments = _segments_from_tokens(result)
        if not segments or any(not s.text or s.end < s.start for s in segments):
            raise RuntimeError("Paraformer produced no usable timestamped Chinese transcript.")
        return LocalizationTranscript("zh", str(source), tuple(segments))
    except LocalizationTranscriptionCancelled:
        raise
    except AudioMixingCancelled as exc:
        raise LocalizationTranscriptionCancelled(
            "Localization was cancelled during Paraformer transcription"
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Unable to load or run Paraformer. Check its cached models and runtime: {exc}") from exc
    finally:
        wav.unlink(missing_ok=True)
