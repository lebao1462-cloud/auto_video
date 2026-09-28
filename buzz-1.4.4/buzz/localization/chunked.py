"""Sequential long-video processing with bounded temporary audio workspaces."""

from dataclasses import replace
from pathlib import Path
import math
import re
import shutil
import threading
from typing import Callable

from buzz.localization.audio_mix import _run_ffmpeg, probe_media_duration
from buzz.localization.final_render import FinalRenderResult
from buzz.localization.subtitles import SubtitleResult
from buzz.localization.workflow import (
    LocalizationCancelled, LocalizationProgress, LocalizationStage,
    LocalizationWorkflowOptions, LocalizationWorkflowResult,
)

_TIME = re.compile(r"^(\d+:\d{2}:\d{2}[,.]\d{3}) --> (\d+:\d{2}:\d{2}[,.]\d{3})$")


def _shift_timestamp(value: str, seconds: float) -> str:
    hours, minutes, rest = value.split(":")
    whole, millis = re.split("[,.]", rest)
    total = max(0, round(seconds * 1000)) + (
        ((int(hours) * 60 + int(minutes)) * 60 + int(whole)) * 1000
        + int(millis)
    )
    hours, remainder = divmod(total, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{',' if ',' in value else '.'}{millis:03}"


def _merge_subtitles(parts: list[tuple[Path, float]], destination: Path, fmt: str) -> int:
    cues = []
    for path, offset in parts:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        current = []
        for line in [*lines, ""]:
            if line.strip():
                current.append(line)
            elif current:
                if current == ["WEBVTT"]:
                    current = []
                    continue
                if current[0].isdigit():
                    current.pop(0)
                if not current or not _TIME.fullmatch(current[0]):
                    raise ValueError(f"Invalid subtitle cue in {path}")
                match = _TIME.fullmatch(current[0])
                start = _shift_timestamp(match.group(1), offset)
                end = _shift_timestamp(match.group(2), offset)
                cues.append((start, end, "\n".join(current[1:])))
                current = []
    if not cues:
        raise ValueError("No subtitle cues were generated")
    if fmt == "srt":
        content = "\n\n".join(
            f"{i}\n{start} --> {end}\n{text}"
            for i, (start, end, text) in enumerate(cues, 1)
        ) + "\n"
    else:
        content = "WEBVTT\n\n" + "\n\n".join(
            f"{start} --> {end}\n{text}" for start, end, text in cues
        ) + "\n"
    destination.write_text(content, encoding="utf-8", newline="\n")
    return len(cues)


def _concat_line(path: Path) -> str:
    # FFmpeg concat demuxer accepts forward slashes on Windows.
    return "file '" + str(path.resolve()).replace("\\", "/").replace("'", "'\\''") + "'\n"


def localize_chunked_video(
    source: Path,
    duration: float,
    transcription_options,
    model_path: str,
    translation_provider,
    tts_provider,
    options: LocalizationWorkflowOptions,
    progress_callback: Callable[[LocalizationProgress], None] | None,
    cancel_event: threading.Event | None,
    transcribe_func,
    asr_provider: str = "whisper",
) -> LocalizationWorkflowResult:
    from buzz.localization.workflow import _localize_single_video, cleanup_localization_workspace

    count = math.ceil(duration / options.chunk_duration_seconds)
    output = Path(options.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / f"{source.stem}_localization_work"
    workspace.mkdir(parents=True, exist_ok=True)
    pieces = []
    subtitles = []
    offset = 0.0
    steps_per_piece = 9 if options.use_background_separation else 8
    total = count * steps_per_piece + 1

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise LocalizationCancelled("Localization was cancelled")

    for index in range(count):
        check_cancel()
        chunk_seconds = min(options.chunk_duration_seconds, duration - index * options.chunk_duration_seconds)
        part_dir = workspace / f"part_{index + 1:04}"
        part_dir.mkdir(exist_ok=True)
        clip = part_dir / "source.mp4"
        if progress_callback:
            progress_callback(LocalizationProgress(
                LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                total, f"Part {index + 1}/{count}: preparing video"
            ))
        _run_ffmpeg([
            "ffmpeg", "-y", "-nostdin", "-ss",
            str(index * options.chunk_duration_seconds), "-i", str(source),
            "-t", str(chunk_seconds), "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
            "-ac", "2", "-avoid_negative_ts", "make_zero", str(clip),
        ])
        if not clip.is_file():
            raise RuntimeError(f"Cannot create video part {index + 1}")

        def part_progress(progress: LocalizationProgress):
            if progress_callback and progress.stage != LocalizationStage.COMPLETED:
                progress_callback(LocalizationProgress(
                    progress.stage, index * steps_per_piece + progress.completed_steps,
                    total, f"Part {index + 1}/{count}: {progress.message}"
                ))

        transcribe_kwargs = {"asr_provider": asr_provider} if asr_provider != "whisper" else {}
        transcript = transcribe_func(str(clip), transcription_options, model_path, **transcribe_kwargs)
        check_cancel()
        if transcript.segments:
            result = _localize_single_video(
                str(clip), transcription_options, model_path,
                translation_provider, tts_provider,
                replace(options, output_directory=str(part_dir),
                        chunk_duration_seconds=0),
                progress_callback=part_progress, cancel_event=cancel_event,
                transcribe_func=lambda *args, **kwargs: transcript, asr_provider=asr_provider,
            )
            part_video = Path(result.final_video.video_file)
            subtitles.append((Path(result.subtitle.subtitle_file), offset))
            cleanup_localization_workspace(result)
        else:
            # A speech-free part needs the same video/audio stream layout for concat.
            part_video = part_dir / "source.vi.mp4"
            _run_ffmpeg([
                "ffmpeg", "-y", "-nostdin", "-i", str(clip), "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0",
                "-map", "1:a:0", "-t", str(probe_media_duration(clip)),
                "-c:v", "copy", "-c:a", options.render_options.audio_codec,
                "-b:a", options.render_options.audio_bitrate, str(part_video),
            ])
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.RENDER, (index + 1) * steps_per_piece,
                    total, f"Part {index + 1}/{count}: no speech; kept video with silence"
                ))
        check_cancel()
        pieces.append(part_video)
        # The concat demuxer offsets each part by its rendered MP4 duration.
        offset += probe_media_duration(part_video)
        clip.unlink(missing_ok=True)

    check_cancel()
    playlist = workspace / "parts.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(
        _concat_line(piece) for piece in pieces
    ), encoding="utf-8")
    temporary_video = workspace / "joined.mp4"
    subtitled_video = workspace / "joined_subtitled.mp4"
    temporary_subtitle = workspace / f"joined.{options.subtitle_format}"
    final_video = output / f"{source.stem}.vi.mp4"
    final_subtitle = output / f"{source.stem}.vi.{options.subtitle_format}"
    final_audio = workspace / "localized_audio.m4a"
    try:
        cues = _merge_subtitles(subtitles, temporary_subtitle, options.subtitle_format)
        _run_ffmpeg([
            "ffmpeg", "-y", "-nostdin", "-safe", "0", "-f", "concat",
            "-i", str(playlist), "-map", "0:v:0", "-map", "0:a:0",
            "-c", "copy", "-movflags", "+faststart", str(temporary_video),
        ])
        actual_duration = probe_media_duration(temporary_video)
        if abs(actual_duration - duration) > max(1.0, count * 0.12):
            raise RuntimeError(
                f"Joined video duration {actual_duration:.2f}s differs from "
                f"source {duration:.2f}s"
            )
        if options.render_options.subtitle_mode == "soft":
            # Remux the single global subtitle track. Concatenated mov_text
            # packets otherwise gain an AAC priming offset at each join.
            _run_ffmpeg([
                "ffmpeg", "-y", "-nostdin", "-i", str(temporary_video),
                "-i", str(temporary_subtitle), "-map", "0:v:0",
                "-map", "0:a:0", "-map", "1:0", "-c:v", "copy",
                "-c:a", "copy", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=vie",
                "-disposition:s:0", "default",
                "-movflags", "+faststart", str(subtitled_video),
            ])
        assembled_video = (
            subtitled_video if options.render_options.subtitle_mode == "soft"
            else temporary_video
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-nostdin", "-i", str(assembled_video),
            "-map", "0:a:0", "-c:a", "copy", str(final_audio),
        ])
        check_cancel()
        assembled_video.replace(final_video)
        temporary_subtitle.replace(final_subtitle)
    finally:
        temporary_video.unlink(missing_ok=True)
        subtitled_video.unlink(missing_ok=True)
        temporary_subtitle.unlink(missing_ok=True)

    for part in (workspace / f"part_{i + 1:04}" for i in range(count)):
        shutil.rmtree(part)
    playlist.unlink(missing_ok=True)
    if progress_callback:
        progress_callback(LocalizationProgress(
            LocalizationStage.COMPLETED, total, total, "Localization completed"
        ))
    return LocalizationWorkflowResult(
        final_video=FinalRenderResult(
            video_file=str(final_video), subtitle_mode=options.render_options.subtitle_mode,
            audio_replaced=True, subtitle_included=options.render_options.subtitle_mode != "none",
        ),
        subtitle=SubtitleResult(str(final_subtitle), options.subtitle_format, cues),
        localized_audio_file=str(final_audio),
        workspace=str(workspace),
    )
