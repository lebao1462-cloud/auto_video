from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Callable

from buzz.assets import APP_BASE_DIR
from buzz.localization.timing import TimedLocalizationTranscript


class AudioMixingError(RuntimeError):
    """Raised when localized audio processing or mixing fails."""


@dataclass(frozen=True)
class AudioMixingOptions:
    sample_rate: int = 48000
    channels: int = 2
    speech_gain_db: float = 0.0
    background_gain_db: float = -8.0

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")


@dataclass(frozen=True)
class LocalizedAudioResult:
    audio_file: str
    duration: float
    sample_rate: int
    channels: int
    background_mixed: bool

    def to_dict(self) -> dict:
        return {
            "audio_file": self.audio_file,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "background_mixed": self.background_mixed,
        }


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    internal = os.path.join(APP_BASE_DIR, "_internal")
    env["PATH"] = os.pathsep.join([internal, env.get("PATH", "")])
    return env


def _run_ffmpeg(command: list[str]) -> None:
    kwargs = {
        "capture_output": True,
        "text": True,
        "env": _ffmpeg_env(),
    }
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(command, **kwargs)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        raise AudioMixingError(f"FFmpeg failed: {message}")



def probe_media_duration(
    media_file: str | Path,
    *,
    ffprobe_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> float:
    """Return media duration in seconds using ffprobe."""
    media_path = Path(media_file)
    if not media_path.is_file():
        raise AudioMixingError(f"Media file does not exist: {media_file}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    if ffprobe_runner is None:
        def ffprobe_runner(command):
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=_ffmpeg_env(),
            )

    result = ffprobe_runner(command)
    if result.returncode != 0:
        raise AudioMixingError(
            "FFprobe failed: " + (result.stderr or "unknown error").strip()
        )

    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise AudioMixingError("FFprobe returned an invalid media duration") from exc

    if duration <= 0:
        raise AudioMixingError("Media duration must be positive")
    return duration

def extract_audio_track(
    source_media_file: str | Path,
    output_file: str | Path,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    ffmpeg_runner: Callable[[list[str]], None] = _run_ffmpeg,
) -> str:
    """Extract a PCM WAV audio track from a source video/audio file."""
    source_path = Path(source_media_file)
    if not source_path.is_file():
        raise AudioMixingError(f"Source media file does not exist: {source_media_file}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_path),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    ffmpeg_runner(command)
    if not output_path.is_file():
        raise AudioMixingError("FFmpeg did not create the extracted audio file")
    return str(output_path)


def separate_background_with_demucs(
    source_audio_file: str | Path,
    output_file: str | Path,
    *,
    device: str | None = None,
) -> str:
    """Separate vocals with Demucs and save the summed non-vocal background stems."""
    source_path = Path(source_audio_file)
    if not source_path.is_file():
        raise AudioMixingError(
            f"Source audio file does not exist: {source_audio_file}"
        )

    try:
        from demucs import api as demucs_api
    except ImportError:
        try:
            import demucs.separate as demucs_api
        except ImportError as exc:
            raise AudioMixingError("Demucs is not available") from exc

    try:
        import torch

        model_root = os.getenv("BUZZ_MODEL_ROOT")
        if model_root:
            torch_hub_dir = Path(model_root) / "torch-hub"
            torch_hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(torch_hub_dir))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        if device is None:
            device = "cpu"

    if not hasattr(demucs_api, "Separator"):
        # The published Demucs package exposes a CLI, but no Separator API.
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        demucs_dir = output_path.parent / "demucs_stems"
        command = [
            sys.executable, "-m", "demucs", "--two-stems", "vocals",
            "-n", "htdemucs", "-d", device, "-o", str(demucs_dir),
            str(source_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            raise AudioMixingError(
                "Demucs background separation failed: "
                + (result.stderr or result.stdout).strip()
            )
        stem = demucs_dir / "htdemucs" / source_path.stem / "no_vocals.wav"
        if not stem.is_file():
            raise AudioMixingError("Demucs did not create the background stem")
        shutil.copyfile(stem, output_path)
        return str(output_path)

    try:
        separator = demucs_api.Separator(device=device, progress=False)
        _, separated = separator.separate_audio_file(source_path)
        background_stems = [
            stem for name, stem in separated.items() if name.lower() != "vocals"
        ]
        if not background_stems:
            raise AudioMixingError("Demucs returned no background stems")

        background = background_stems[0]
        for stem in background_stems[1:]:
            background = background + stem

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        demucs_api.save_audio(background, output_path, separator.samplerate)
    except AudioMixingError:
        raise
    except Exception as exc:
        raise AudioMixingError(f"Demucs background separation failed: {exc}") from exc

    if not output_path.is_file():
        raise AudioMixingError("Demucs did not create the background audio file")
    return str(output_path)


def _atempo_filter(playback_rate: float) -> str:
    if not 0.5 <= playback_rate <= 2.0:
        raise AudioMixingError(
            f"Playback rate {playback_rate:.3f} is outside FFmpeg atempo range"
        )
    return f"atempo={playback_rate:.8f}"


def mix_localized_audio(
    transcript: TimedLocalizationTranscript,
    output_file: str | Path,
    *,
    background_audio_file: str | Path | None = None,
    options: AudioMixingOptions | None = None,
    target_duration: float | None = None,
    ffmpeg_runner: Callable[[list[str]], None] = _run_ffmpeg,
) -> LocalizedAudioResult:
    """Render a synchronized full-length localized audio track with FFmpeg."""
    if not isinstance(transcript, TimedLocalizationTranscript):
        raise TypeError("A Phase 4 TimedLocalizationTranscript is required")
    if not transcript.segments:
        raise AudioMixingError("Phase 5 cannot mix an empty transcript")

    mix_options = AudioMixingOptions() if options is None else options
    if not isinstance(mix_options, AudioMixingOptions):
        raise TypeError("options must be AudioMixingOptions")

    output_path = Path(output_file)
    if not str(output_path).strip():
        raise ValueError("Output file is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    for index, segment in enumerate(transcript.segments):
        audio_path = Path(segment.audio_file)
        if not audio_path.is_file():
            raise AudioMixingError(
                f"Segment {index} audio file does not exist: {segment.audio_file}"
            )
        inputs.extend(["-i", str(audio_path)])

    background_index = None
    if background_audio_file is not None:
        background_path = Path(background_audio_file)
        if not background_path.is_file():
            raise AudioMixingError(
                f"Background audio file does not exist: {background_audio_file}"
            )
        background_index = len(transcript.segments)
        inputs.extend(["-i", str(background_path)])

    last_segment_end = max(float(segment.end) for segment in transcript.segments)
    if target_duration is None:
        final_duration = last_segment_end
    else:
        if (
            not isinstance(target_duration, (int, float))
            or isinstance(target_duration, bool)
            or target_duration <= 0
        ):
            raise AudioMixingError("Target audio duration must be positive")
        target_duration = float(target_duration)
        drift = last_segment_end - target_duration
        if drift > 0.25:
            raise AudioMixingError(
                "Target audio duration cannot end before the last speech segment"
            )
        final_duration = max(target_duration, last_segment_end)

    filters: list[str] = []
    speech_labels: list[str] = []

    for index, segment in enumerate(transcript.segments):
        delay_ms = round(float(segment.start) * 1000)
        chain = [
            f"aresample={mix_options.sample_rate}",
            _atempo_filter(float(segment.playback_rate)),
        ]
        if mix_options.speech_gain_db:
            chain.append(f"volume={mix_options.speech_gain_db}dB")
        chain.extend(
            [
                f"adelay={delay_ms}:all=1",
                f"apad=whole_dur={final_duration:.6f}",
                "asetpts=N/SR/TB",
                f"atrim=0:{final_duration:.6f}",
            ]
        )
        filters.append(
            f"[{index}:a]" + ",".join(chain) + f"[speech{index}]"
        )
        speech_labels.append(f"[speech{index}]")

    if len(speech_labels) == 1:
        filters.append(f"{speech_labels[0]}anull[speechmix]")
    else:
        filters.append(
            "".join(speech_labels)
            + f"amix=inputs={len(speech_labels)}:normalize=0:"
            "duration=longest[speechmix]"
        )

    if background_index is not None:
        filters.append(
            f"[{background_index}:a]"
            f"aresample={mix_options.sample_rate},"
            f"volume={mix_options.background_gain_db}dB,"
            f"apad=whole_dur={final_duration:.6f},"
            f"atrim=0:{final_duration:.6f}[background]"
        )
        filters.append(
            "[background][speechmix]"
            "amix=inputs=2:normalize=0:duration=longest,"
            "alimiter=limit=0.95[final]"
        )
    else:
        filters.append("[speechmix]alimiter=limit=0.95[final]")

    channel_layout = "mono" if mix_options.channels == 1 else "stereo"
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[final]",
        "-ar",
        str(mix_options.sample_rate),
        "-ac",
        str(mix_options.channels),
        "-channel_layout",
        channel_layout,
        "-c:a",
        "pcm_s16le",
        "-t",
        f"{final_duration:.6f}",
        str(output_path),
    ]

    ffmpeg_runner(command)

    if not output_path.is_file():
        raise AudioMixingError("FFmpeg did not create the localized audio file")

    return LocalizedAudioResult(
        audio_file=str(output_path),
        duration=final_duration,
        sample_rate=mix_options.sample_rate,
        channels=mix_options.channels,
        background_mixed=background_index is not None,
    )
