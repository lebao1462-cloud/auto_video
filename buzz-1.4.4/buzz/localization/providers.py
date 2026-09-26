from dataclasses import dataclass
from pathlib import Path
import subprocess

from openai import OpenAI

from buzz.localization.audio_mix import _ffmpeg_env
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
