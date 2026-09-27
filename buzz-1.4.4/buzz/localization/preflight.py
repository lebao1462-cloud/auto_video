from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile

from buzz.assets import APP_BASE_DIR


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    message: str
    required: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "message": self.message,
            "required": self.required,
        }


@dataclass(frozen=True)
class LocalizationPreflightReport:
    checks: tuple[PreflightCheck, ...]
    source_size_bytes: int
    free_space_bytes: int
    estimated_required_bytes: int

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            check.message
            for check in self.checks
            if check.required and not check.ok
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            check.message
            for check in self.checks
            if not check.required and not check.ok
        )

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "source_size_bytes": self.source_size_bytes,
            "free_space_bytes": self.free_space_bytes,
            "estimated_required_bytes": self.estimated_required_bytes,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        }


def _find_bundled_or_path_executable(name: str) -> str | None:
    suffix = ".exe" if os.name == "nt" else ""
    bundled = Path(APP_BASE_DIR) / "_internal" / f"{name}{suffix}"
    if bundled.is_file():
        return str(bundled)
    return shutil.which(f"{name}{suffix}") or shutil.which(name)


def estimate_required_workspace_bytes(source_size_bytes: int) -> int:
    if source_size_bytes < 0:
        raise ValueError("source_size_bytes must be non-negative")
    # PCM extraction + separated stems + TTS + render intermediates can exceed
    # compressed source size substantially. Keep a conservative floor.
    return max(512 * 1024 * 1024, source_size_bytes * 6)


def run_localization_preflight(
    source_video_file: str | Path,
    model_path: str | Path,
    output_directory: str | Path,
    *,
    translation_api_key: str = "",
    translation_model: str = "",
    translation_provider: str = "openai-compatible",
    source_language: str | None = None,
    require_edge_tts: bool = True,
    use_background_separation: bool = False,
) -> LocalizationPreflightReport:
    source = Path(source_video_file)
    model = Path(model_path)
    output = Path(output_directory)

    if translation_provider not in {"argos", "gemini", "openai-compatible"}:
        raise ValueError("Unsupported translation provider")

    checks: list[PreflightCheck] = []

    source_ok = source.is_file()
    checks.append(
        PreflightCheck(
            "source_video",
            source_ok,
            "Input video is available."
            if source_ok
            else "Input video does not exist.",
        )
    )
    source_size = source.stat().st_size if source_ok else 0

    model_ok = model.exists()
    checks.append(
        PreflightCheck(
            "whisper_model",
            model_ok,
            "Whisper model path is available."
            if model_ok
            else "Whisper model path does not exist.",
        )
    )

    for executable in ("ffmpeg", "ffprobe"):
        path = _find_bundled_or_path_executable(executable)
        checks.append(
            PreflightCheck(
                executable,
                path is not None,
                f"{executable} is available."
                if path
                else f"{executable} is not available. Install/bundle FFmpeg first.",
            )
        )

    edge_available = importlib.util.find_spec("edge_tts") is not None
    checks.append(
        PreflightCheck(
            "edge_tts",
            edge_available,
            "Edge TTS runtime is available."
            if edge_available
            else (
                "Edge TTS runtime is missing. Install edge-tts==7.2.8 "
                "in the Buzz Python environment."
            ),
            required=require_edge_tts,
        )
    )

    if use_background_separation:
        demucs_available = importlib.util.find_spec("demucs") is not None
        checks.append(
            PreflightCheck(
                "demucs",
                demucs_available,
                "Demucs background separation is available."
                if demucs_available
                else "Demucs is not available for background separation.",
            )
        )

    if translation_provider == "argos":
        argos_runtime = importlib.util.find_spec("argostranslate") is not None
        checks.append(
            PreflightCheck(
                "argos_runtime",
                argos_runtime,
                "Argos Translate runtime is available."
                if argos_runtime
                else (
                    "Argos Translate runtime is missing. Install "
                    "argostranslate in the Buzz Python environment."
                ),
            )
        )
        if argos_runtime:
            try:
                from buzz.localization.providers import argos_route_available

                if source_language in {"en", "zh"}:
                    ok, message = argos_route_available(source_language)
                    checks.append(
                        PreflightCheck(
                            f"argos_route_{source_language}",
                            ok,
                            message,
                        )
                    )
                else:
                    checks.append(
                        PreflightCheck(
                            "argos_route_auto",
                            True,
                            "Argos route will be validated after source-language auto-detection.",
                        )
                    )
            except Exception as exc:
                checks.append(
                    PreflightCheck(
                        "argos_routes",
                        False,
                        f"Unable to inspect installed Argos routes: {exc}",
                    )
                )
    elif translation_provider == "gemini":
        gemini_runtime = importlib.util.find_spec("google.genai") is not None
        checks.append(
            PreflightCheck(
                "gemini_runtime",
                gemini_runtime,
                "Gemini runtime is available."
                if gemini_runtime
                else "google-genai is required for Gemini translation.",
            )
        )
        checks.append(
            PreflightCheck(
                "translation_api_key",
                bool(translation_api_key.strip()),
                "Gemini API credential is configured."
                if translation_api_key.strip()
                else "Gemini API key is required.",
            )
        )
        checks.append(
            PreflightCheck(
                "translation_model",
                bool(translation_model.strip()),
                "Gemini model is configured."
                if translation_model.strip()
                else "Gemini model is required.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "translation_api_key",
                bool(translation_api_key.strip()),
                "Translation API credential is configured."
                if translation_api_key.strip()
                else "Translation API key is required.",
            )
        )
        checks.append(
            PreflightCheck(
                "translation_model",
                bool(translation_model.strip()),
                "Translation model is configured."
                if translation_model.strip()
                else "Translation model is required.",
            )
        )

    output_ok = False
    output_message = ""
    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".localization-write-test-", delete=True
        ):
            pass
        output_ok = True
        output_message = "Output directory is writable."
    except Exception as exc:
        output_message = f"Output directory is not writable: {exc}"
    checks.append(
        PreflightCheck("output_writable", output_ok, output_message)
    )

    required_bytes = estimate_required_workspace_bytes(source_size)
    free_bytes = 0
    if output_ok:
        try:
            free_bytes = shutil.disk_usage(output).free
        except OSError:
            free_bytes = 0
    disk_ok = free_bytes >= required_bytes
    checks.append(
        PreflightCheck(
            "disk_space",
            disk_ok,
            (
                f"Free disk space is sufficient ({free_bytes} bytes free; "
                f"{required_bytes} bytes estimated required)."
                if disk_ok
                else (
                    f"Not enough free disk space ({free_bytes} bytes free; "
                    f"{required_bytes} bytes estimated required)."
                )
            ),
        )
    )

    model_root = os.getenv("BUZZ_MODEL_ROOT", "")
    d_drive_preferred = os.name != "nt" or (
        model_root and Path(model_root).drive.upper() == "D:"
    )
    checks.append(
        PreflightCheck(
            "model_cache_drive",
            d_drive_preferred,
            "Model/cache root follows the configured drive policy."
            if d_drive_preferred
            else "BUZZ_MODEL_ROOT is not configured on drive D.",
            required=False,
        )
    )

    return LocalizationPreflightReport(
        checks=tuple(checks),
        source_size_bytes=source_size,
        free_space_bytes=free_bytes,
        estimated_required_bytes=required_bytes,
    )


def write_localization_diagnostics(
    report: LocalizationPreflightReport,
    output_file: str | Path,
) -> str:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
