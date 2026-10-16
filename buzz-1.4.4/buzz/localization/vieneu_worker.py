"""UTF-8 JSON-lines worker for the isolated VieNeu v3 Turbo runtime."""

from __future__ import annotations

import contextlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import wave
from pathlib import Path


def respond(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def close_engine(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _worker_logger():
    logger = logging.getLogger("vieneu-worker")
    target = os.environ.get("BUZZ_VIENEU_WORKER_LOG")
    if target:
        try:
            output = Path(target)
            output.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(output, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        except OSError:
            logger.warning("Cannot open VieNeu worker diagnostic log")
    return logger


def main() -> None:
    logger = _worker_logger()
    engine = None
    try:
        for line in sys.stdin:
            request = None
            output = None
            try:
                request = json.loads(line)
                if request.get("action") == "shutdown":
                    respond({"id": request.get("id"), "ok": True})
                    return

                if engine is None:
                    startup = time.monotonic()
                    logger.info("Loading VieNeu ONNX model pid=%d", os.getpid())
                    # Keep third-party progress output away from protocol stdout.
                    with contextlib.redirect_stdout(sys.stderr):
                        from vieneu import Vieneu

                        engine = Vieneu(
                            onnx_dir=request["model_dir"],
                            device="cpu",
                            backend="onnx",
                        )
                    logger.info("VieNeu model ready in %.1fs", time.monotonic() - startup)

                output = Path(request["output_file"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.unlink(missing_ok=True)
                inference_start = time.monotonic()
                logger.info("Synthesizing id=%s chars=%d voice=%s",
                            request.get("id"), len(request["text"]), request["voice"])
                with contextlib.redirect_stdout(sys.stderr):
                    audio = engine.infer(
                        request["text"],
                        voice=request["voice"],
                    )
                    engine.save(audio, output)

                logger.info("Synthesis complete id=%s elapsed=%.1fs",
                            request.get("id"), time.monotonic() - inference_start)
                with wave.open(str(output), "rb") as wav:
                    sample_rate = wav.getframerate()
                    duration = wav.getnframes() / sample_rate

                respond({
                    "id": request.get("id"),
                    "ok": True,
                    "duration": duration,
                    "metadata": {
                        "sample_rate": sample_rate,
                        "format": "wav",
                    },
                })
            except Exception as exc:
                logger.exception("VieNeu worker request failed id=%s",
                                 request.get("id") if isinstance(request, dict) else None)
                if output is not None:
                    output.unlink(missing_ok=True)
                respond({
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        close_engine(engine)


if __name__ == "__main__":
    main()
