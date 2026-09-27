from dataclasses import dataclass
from pathlib import Path
import subprocess

from openai import OpenAI

from buzz.localization.audio_mix import _ffmpeg_env, _run_ffmpeg
from buzz.localization.tts import TTSRequest, TTSResult


class LocalizationProviderError(RuntimeError):
    """Raised when a concrete localization provider fails."""


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
        "env": os.environ.copy(),
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
class GeminiTranslationProvider:
    api_key: str
    model: str = "gemini-2.5-flash"
    timeout: float = 60.0

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        if not self.model:
            raise ValueError("Gemini model is required")
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
            response = self._client.models.generate_content(
                model=self.model,
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


@dataclass
class EdgeTTSProvider:
    default_voice: str = "vi-VN-HoaiMyNeural"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if request.language != "vi":
            raise ValueError("EdgeTTSProvider requires Vietnamese ('vi')")
        if not request.text.strip():
            raise ValueError("TTS text cannot be empty")

        try:
            import edge_tts
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

        try:
            communicate = edge_tts.Communicate(
                request.text,
                voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )
            communicate.save_sync(output_file)
        except Exception as exc:
            raise LocalizationProviderError(f"Edge TTS failed: {exc}") from exc

        output_path = Path(output_file)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise LocalizationProviderError("Edge TTS did not create an audio file")

        _trim_edge_tts_silence(output_path)
        duration = _probe_audio_duration(output_path)
        return TTSResult(
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
