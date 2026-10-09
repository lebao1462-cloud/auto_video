"""UTF-8 JSON-lines worker for the isolated VieNeu v3 Turbo runtime."""

from __future__ import annotations

import contextlib
import json
import sys
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


def main() -> None:
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
                    # Keep third-party progress output away from protocol stdout.
                    with contextlib.redirect_stdout(sys.stderr):
                        from vieneu import Vieneu

                        engine = Vieneu(
                            onnx_dir=request["model_dir"],
                            device="cpu",
                            backend="onnx",
                        )

                output = Path(request["output_file"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.unlink(missing_ok=True)
                with contextlib.redirect_stdout(sys.stderr):
                    audio = engine.infer(
                        request["text"],
                        voice=request["voice"],
                    )
                    engine.save(audio, output)

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
