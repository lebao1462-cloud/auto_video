from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

from buzz.assets import APP_BASE_DIR
from buzz.localization.timing import TimedLocalizationTranscript


class AudioMixingError(RuntimeError):
    """Raised when localized audio processing or mixing fails."""


# Keep ample headroom below Windows' 32,767-character CreateProcess limit for
# input/output paths and the rest of the FFmpeg arguments.
_FILTER_COMPLEX_SCRIPT_THRESHOLD = 8192
_MAX_MIX_INPUTS = 32


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
        "encoding": "utf-8",
        "errors": "replace",
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
                encoding="utf-8",
                errors="replace",
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


def _temporary_audio_path(directory: Path) -> Path:
    handle, name = tempfile.mkstemp(
        suffix=".wav", prefix="buzz-speech-mix-", dir=directory
    )
    os.close(handle)
    path = Path(name)
    path.unlink()
    return path


def _run_audio_filter(
    inputs: list[Path], filters: list[str], output_label: str,
    output_path: Path, duration: float, options: AudioMixingOptions,
    ffmpeg_runner: Callable[[list[str]], None], *, codec: str,
) -> None:
    """Run one bounded-size FFmpeg audio-filter invocation."""
    filter_complex = ";".join(filters)
    command = ["ffmpeg", "-y", "-nostdin"]
    for input_path in inputs:
        command.extend(["-i", str(input_path)])
    filter_script_path: Path | None = None
    try:
        if len(filter_complex) >= _FILTER_COMPLEX_SCRIPT_THRESHOLD:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                suffix=".fffilter", prefix="buzz-filter-",
                dir=output_path.parent, delete=False,
            ) as filter_script:
                filter_script.write(filter_complex)
                filter_script_path = Path(filter_script.name)
            command.extend(["-filter_complex_script", str(filter_script_path)])
        else:
            command.extend(["-filter_complex", filter_complex])
        channel_layout = "mono" if options.channels == 1 else "stereo"
        command.extend([
            "-map", output_label, "-ar", str(options.sample_rate),
            "-ac", str(options.channels), "-channel_layout", channel_layout,
            "-c:a", codec, "-t", f"{duration:.6f}", str(output_path),
        ])
        ffmpeg_runner(command)
    finally:
        if filter_script_path is not None:
            filter_script_path.unlink(missing_ok=True)


def _mix_track_batch(
    tracks: list[tuple[Path, float, float]], output_path: Path,
    options: AudioMixingOptions, ffmpeg_runner: Callable[[list[str]], None],
) -> tuple[Path, float, float]:
    """Mix rendered tracks while retaining their absolute timeline origins."""
    origin = min(track[1] for track in tracks)
    end = max(track[2] for track in tracks)
    duration = end - origin
    filters = []
    labels = []
    for index, (_, start, _) in enumerate(tracks):
        delay_ms = round((start - origin) * 1000)
        filters.append(
            f"[{index}:a]adelay={delay_ms}:all=1,"
            f"apad=whole_dur={duration:.6f},atrim=0:{duration:.6f}[track{index}]"
        )
        labels.append(f"[track{index}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[speechmix]")
    else:
        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:normalize=0:duration=longest[speechmix]"
        )
    _run_audio_filter(
        [track[0] for track in tracks], filters, "[speechmix]", output_path,
        duration, options, ffmpeg_runner, codec="pcm_f32le",
    )
    return output_path, origin, end


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

    for index, segment in enumerate(transcript.segments):
        audio_path = Path(segment.audio_file)
        if not audio_path.is_file():
            raise AudioMixingError(
                f"Segment {index} audio file does not exist: {segment.audio_file}"
            )
    background_path = None
    if background_audio_file is not None:
        background_path = Path(background_audio_file)
        if not background_path.is_file():
            raise AudioMixingError(
                f"Background audio file does not exist: {background_audio_file}"
            )

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

    # Keep the established one-pass behavior for ordinary transcripts. The
    # hierarchical path below is only needed when argv fan-in becomes risky.
    if len(transcript.segments) <= _MAX_MIX_INPUTS:
        direct_inputs = [Path(segment.audio_file) for segment in transcript.segments]
        filters: list[str] = []
        labels: list[str] = []
        for index, segment in enumerate(transcript.segments):
            chain = [
                f"aresample={mix_options.sample_rate}",
                _atempo_filter(float(segment.playback_rate)),
                f"atrim=0:{float(segment.slot_duration):.6f}",
            ]
            if mix_options.speech_gain_db:
                chain.append(f"volume={mix_options.speech_gain_db}dB")
            chain.extend([
                f"adelay={round(float(segment.start) * 1000)}:all=1",
                f"apad=whole_dur={final_duration:.6f}", "asetpts=N/SR/TB",
                f"atrim=0:{final_duration:.6f}",
            ])
            filters.append(
                f"[{index}:a]" + ",".join(chain) + f"[speech{index}]"
            )
            labels.append(f"[speech{index}]")
        if len(labels) == 1:
            filters.append(f"{labels[0]}anull[speechmix]")
        else:
            filters.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:duration=longest[speechmix]"
            )
        if background_path is not None:
            background_index = len(direct_inputs)
            direct_inputs.append(background_path)
            filters.extend([
                f"[{background_index}:a]aresample={mix_options.sample_rate},"
                f"volume={mix_options.background_gain_db}dB,"
                f"apad=whole_dur={final_duration:.6f},"
                f"atrim=0:{final_duration:.6f}[background]",
                "[background][speechmix]amix=inputs=2:normalize=0:duration=longest,"
                "alimiter=limit=0.95[final]",
            ])
        else:
            filters.append("[speechmix]alimiter=limit=0.95[final]")
        _run_audio_filter(
            direct_inputs, filters, "[final]", output_path, final_duration,
            mix_options, ffmpeg_runner, codec="pcm_s16le",
        )
        if not output_path.is_file():
            raise AudioMixingError("FFmpeg did not create the localized audio file")
        return LocalizedAudioResult(
            audio_file=str(output_path), duration=final_duration,
            sample_rate=mix_options.sample_rate, channels=mix_options.channels,
            background_mixed=background_path is not None,
        )

    temporary_paths: list[Path] = []
    try:
        tracks: list[tuple[Path, float, float]] = []
        segments = list(transcript.segments)
        for offset in range(0, len(segments), _MAX_MIX_INPUTS):
            batch = segments[offset:offset + _MAX_MIX_INPUTS]
            # Use millisecond-aligned batch origins so splitting the delay over
            # multiple stages cannot introduce an extra rounding millisecond.
            origin = min(round(float(segment.start) * 1000) for segment in batch) / 1000
            end = max(float(segment.end) for segment in batch)
            duration = end - origin
            filters: list[str] = []
            labels: list[str] = []
            for index, segment in enumerate(batch):
                chain = [
                    f"aresample={mix_options.sample_rate}",
                    _atempo_filter(float(segment.playback_rate)),
                    f"atrim=0:{float(segment.slot_duration):.6f}",
                ]
                if mix_options.speech_gain_db:
                    chain.append(f"volume={mix_options.speech_gain_db}dB")
                chain.extend([
                    f"adelay={round((float(segment.start) - origin) * 1000)}:all=1",
                    f"apad=whole_dur={duration:.6f}", "asetpts=N/SR/TB",
                    f"atrim=0:{duration:.6f}",
                ])
                filters.append(
                    f"[{index}:a]" + ",".join(chain) + f"[speech{index}]"
                )
                labels.append(f"[speech{index}]")
            if len(labels) == 1:
                filters.append(f"{labels[0]}anull[speechmix]")
            else:
                filters.append(
                    "".join(labels)
                    + f"amix=inputs={len(labels)}:normalize=0:duration=longest[speechmix]"
                )
            batch_path = _temporary_audio_path(output_path.parent)
            temporary_paths.append(batch_path)
            _run_audio_filter(
                [Path(segment.audio_file) for segment in batch], filters,
                "[speechmix]", batch_path, duration, mix_options, ffmpeg_runner,
                codec="pcm_f32le",
            )
            tracks.append((batch_path, origin, end))

        while len(tracks) > 1:
            next_tracks = []
            for offset in range(0, len(tracks), _MAX_MIX_INPUTS):
                batch_path = _temporary_audio_path(output_path.parent)
                temporary_paths.append(batch_path)
                next_tracks.append(_mix_track_batch(
                    tracks[offset:offset + _MAX_MIX_INPUTS], batch_path,
                    mix_options, ffmpeg_runner,
                ))
            tracks = next_tracks

        speech_path, speech_origin, _ = tracks[0]
        final_inputs = [speech_path]
        final_filters = [
            f"[0:a]adelay={round(speech_origin * 1000)}:all=1,"
            f"apad=whole_dur={final_duration:.6f},"
            f"atrim=0:{final_duration:.6f}[speechmix]"
        ]
        if background_path is not None:
            final_inputs.append(background_path)
            final_filters.extend([
                f"[1:a]aresample={mix_options.sample_rate},"
                f"volume={mix_options.background_gain_db}dB,"
                f"apad=whole_dur={final_duration:.6f},"
                f"atrim=0:{final_duration:.6f}[background]",
                "[background][speechmix]amix=inputs=2:normalize=0:duration=longest,"
                "alimiter=limit=0.95[final]",
            ])
        else:
            final_filters.append("[speechmix]alimiter=limit=0.95[final]")
        _run_audio_filter(
            final_inputs, final_filters, "[final]", output_path, final_duration,
            mix_options, ffmpeg_runner, codec="pcm_s16le",
        )
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    if not output_path.is_file():
        raise AudioMixingError("FFmpeg did not create the localized audio file")

    return LocalizedAudioResult(
        audio_file=str(output_path),
        duration=final_duration,
        sample_rate=mix_options.sample_rate,
        channels=mix_options.channels,
        background_mixed=background_path is not None,
    )
