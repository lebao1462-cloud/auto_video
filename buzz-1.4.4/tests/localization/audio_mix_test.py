import json
import math
from pathlib import Path
import struct
import wave

import pytest

from buzz.localization.audio_mix import (
    AudioMixingError,
    AudioMixingOptions,
    LocalizedAudioResult,
    mix_localized_audio,
)
from buzz.localization.timing import (
    TimedLocalizationSegment,
    TimedLocalizationTranscript,
    TimingPolicy,
)


def make_wav(path: Path, duration_seconds=0.2, sample_rate=8000):
    frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def make_segment(
    audio_file,
    start=0.0,
    end=2.0,
    playback_rate=1.0,
    timing_action="none",
):
    return TimedLocalizationSegment(
        start=start,
        end=end,
        source_text="Hello",
        translated_text="Xin chào",
        audio_file=str(audio_file),
        audio_duration=max(0.1, end - start),
        provider="fake",
        voice="vi-test",
        provider_metadata={"format": "wav"},
        slot_duration=end - start,
        playback_rate=playback_rate,
        adjusted_audio_duration=(end - start),
        leading_padding=0.0,
        trailing_padding=0.0,
        timing_action=timing_action,
    )


def make_transcript(segments):
    return TimedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language="vi",
        timing_policy=TimingPolicy(),
        segments=tuple(segments),
    )


class FakeRunner:
    def __init__(self, create_output=True):
        self.commands = []
        self.create_output = create_output

    def __call__(self, command):
        self.commands.append(command)
        if self.create_output:
            Path(command[-1]).write_bytes(b"fake wav")


def test_builds_speech_only_mix_command(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript([make_segment(speech)]),
        tmp_path / "localized.wav",
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]

    assert "atempo=1.00000000" in filter_complex
    assert "adelay=0:all=1" in filter_complex
    assert "alimiter=limit=0.95[final]" in filter_complex
    assert result == LocalizedAudioResult(
        audio_file=str(tmp_path / "localized.wav"),
        duration=2.0,
        sample_rate=48000,
        channels=2,
        background_mixed=False,
    )


def test_applies_phase4_playback_rate_and_start_delay(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)
    runner = FakeRunner()

    mix_localized_audio(
        make_transcript(
            [make_segment(speech, start=1.25, end=3.25, playback_rate=1.2)]
        ),
        tmp_path / "localized.wav",
        ffmpeg_runner=runner,
    )

    filter_complex = runner.commands[0][
        runner.commands[0].index("-filter_complex") + 1
    ]
    assert "atempo=1.20000000" in filter_complex
    assert "adelay=1250:all=1" in filter_complex


def test_places_multiple_segments_on_timeline(tmp_path):
    speech1 = tmp_path / "speech1.wav"
    speech2 = tmp_path / "speech2.wav"
    make_wav(speech1)
    make_wav(speech2)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript(
            [
                make_segment(speech1, 0.0, 1.0),
                make_segment(speech2, 2.0, 4.0, playback_rate=0.95),
            ]
        ),
        tmp_path / "localized.wav",
        ffmpeg_runner=runner,
    )

    filter_complex = runner.commands[0][
        runner.commands[0].index("-filter_complex") + 1
    ]
    assert "[speech0][speech1]amix=inputs=2" in filter_complex
    assert "adelay=2000:all=1" in filter_complex
    assert result.duration == 4.0


def test_mixes_background_with_configured_gain(tmp_path):
    speech = tmp_path / "speech.wav"
    background = tmp_path / "background.wav"
    make_wav(speech)
    make_wav(background)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript([make_segment(speech)]),
        tmp_path / "localized.wav",
        background_audio_file=background,
        options=AudioMixingOptions(background_gain_db=-12.0),
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]

    assert str(background) in command
    assert "volume=-12.0dB" in filter_complex
    assert "[background][speechmix]amix=inputs=2" in filter_complex
    assert result.background_mixed is True


def test_applies_speech_gain(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)
    runner = FakeRunner()

    mix_localized_audio(
        make_transcript([make_segment(speech)]),
        tmp_path / "localized.wav",
        options=AudioMixingOptions(speech_gain_db=3.0),
        ffmpeg_runner=runner,
    )

    filter_complex = runner.commands[0][
        runner.commands[0].index("-filter_complex") + 1
    ]
    assert "volume=3.0dB" in filter_complex


def test_output_duration_uses_last_segment_end(tmp_path):
    speech1 = tmp_path / "speech1.wav"
    speech2 = tmp_path / "speech2.wav"
    make_wav(speech1)
    make_wav(speech2)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript(
            [
                make_segment(speech1, 0.0, 1.0),
                make_segment(speech2, 3.0, 7.5),
            ]
        ),
        tmp_path / "localized.wav",
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[command.index("-t") + 1] == "7.500000"
    assert result.duration == 7.5


def test_respects_sample_rate_and_channels(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)
    runner = FakeRunner()
    options = AudioMixingOptions(sample_rate=44100, channels=1)

    result = mix_localized_audio(
        make_transcript([make_segment(speech)]),
        tmp_path / "localized.wav",
        options=options,
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[command.index("-ar") + 1] == "44100"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-channel_layout") + 1] == "mono"
    assert result.sample_rate == 44100
    assert result.channels == 1


def test_rejects_missing_segment_audio(tmp_path):
    with pytest.raises(AudioMixingError, match="does not exist"):
        mix_localized_audio(
            make_transcript([make_segment(tmp_path / "missing.wav")]),
            tmp_path / "localized.wav",
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_missing_background_audio(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    with pytest.raises(AudioMixingError, match="Background audio"):
        mix_localized_audio(
            make_transcript([make_segment(speech)]),
            tmp_path / "localized.wav",
            background_audio_file=tmp_path / "missing-background.wav",
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_empty_transcript(tmp_path):
    transcript = TimedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language="vi",
        timing_policy=TimingPolicy(),
        segments=(),
    )

    with pytest.raises(AudioMixingError, match="empty transcript"):
        mix_localized_audio(
            transcript,
            tmp_path / "localized.wav",
            ffmpeg_runner=FakeRunner(),
        )


@pytest.mark.parametrize(
    "options",
    [
        lambda: AudioMixingOptions(sample_rate=0),
        lambda: AudioMixingOptions(channels=0),
        lambda: AudioMixingOptions(channels=3),
    ],
)
def test_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        options()


def test_rejects_playback_rate_outside_ffmpeg_range(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    with pytest.raises(AudioMixingError, match="atempo range"):
        mix_localized_audio(
            make_transcript(
                [make_segment(speech, playback_rate=2.5)]
            ),
            tmp_path / "localized.wav",
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_runner_that_does_not_create_output(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    with pytest.raises(AudioMixingError, match="did not create"):
        mix_localized_audio(
            make_transcript([make_segment(speech)]),
            tmp_path / "localized.wav",
            ffmpeg_runner=FakeRunner(create_output=False),
        )


def test_result_is_json_compatible(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    result = mix_localized_audio(
        make_transcript([make_segment(speech)]),
        tmp_path / "localized.wav",
        ffmpeg_runner=FakeRunner(),
    )

    encoded = json.dumps(result.to_dict())
    assert json.loads(encoded) == result.to_dict()


def test_real_ffmpeg_renders_timeline_and_background(tmp_path):
    speech1 = tmp_path / "speech1.wav"
    speech2 = tmp_path / "speech2.wav"
    background = tmp_path / "background.wav"
    make_wav(speech1, 0.5, 16000)
    make_wav(speech2, 0.5, 16000)
    make_wav(background, 3.0, 16000)

    result = mix_localized_audio(
        make_transcript(
            [
                make_segment(speech1, 0.0, 1.0),
                make_segment(speech2, 2.0, 3.0),
            ]
        ),
        tmp_path / "localized.wav",
        background_audio_file=background,
        options=AudioMixingOptions(sample_rate=16000, channels=1),
    )

    assert Path(result.audio_file).is_file()
    assert Path(result.audio_file).stat().st_size > 44
    with wave.open(result.audio_file, "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        actual_duration = wav_file.getnframes() / wav_file.getframerate()
    assert actual_duration == pytest.approx(3.0, abs=0.05)


def test_real_ffmpeg_keeps_delayed_speech_at_its_timestamp(tmp_path):
    speech = tmp_path / "tone.wav"
    sample_rate = 16000
    samples = [
        int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(sample_rate // 4)
    ]
    with wave.open(str(speech), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    result = mix_localized_audio(
        make_transcript([make_segment(speech, start=2.0, end=2.3)]),
        tmp_path / "localized.wav",
        options=AudioMixingOptions(sample_rate=sample_rate, channels=1),
    )

    with wave.open(result.audio_file, "rb") as wav_file:
        assert wav_file.getnframes() / sample_rate == pytest.approx(2.3, abs=0.02)
        wav_file.setpos(int(0.1 * sample_rate))
        early = wav_file.readframes(sample_rate // 10)
        wav_file.setpos(int(2.05 * sample_rate))
        late = wav_file.readframes(sample_rate // 10)

    assert max(abs(value) for value in struct.unpack(f"<{len(early)//2}h", early)) < 100
    assert max(abs(value) for value in struct.unpack(f"<{len(late)//2}h", late)) > 100


def test_extract_audio_track_builds_expected_command(tmp_path):
    from buzz.localization.audio_mix import extract_audio_track

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake media")
    runner = FakeRunner()

    output = extract_audio_track(
        source,
        tmp_path / "extracted.wav",
        sample_rate=16000,
        channels=1,
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[:4] == ["ffmpeg", "-y", "-nostdin", "-i"]
    assert str(source) in command
    assert "-vn" in command
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert output == str(tmp_path / "extracted.wav")


def test_extract_audio_track_rejects_missing_source(tmp_path):
    from buzz.localization.audio_mix import extract_audio_track

    with pytest.raises(AudioMixingError, match="Source media"):
        extract_audio_track(
            tmp_path / "missing.mp4",
            tmp_path / "out.wav",
            ffmpeg_runner=FakeRunner(),
        )


def test_real_ffmpeg_extracts_audio_from_wav(tmp_path):
    from buzz.localization.audio_mix import extract_audio_track

    source = tmp_path / "source.wav"
    make_wav(source, 0.5, 8000)

    output = extract_audio_track(
        source,
        tmp_path / "extracted.wav",
        sample_rate=16000,
        channels=1,
    )

    with wave.open(output, "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1


def test_demucs_adapter_sums_non_vocal_stems(monkeypatch, tmp_path):
    import sys
    import types
    from buzz.localization.audio_mix import separate_background_with_demucs

    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "background.wav"
    saved = {}

    class FakeStem:
        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            return FakeStem(self.value + other.value)

    class FakeSeparator:
        samplerate = 44100

        def __init__(self, device, progress):
            saved["device"] = device
            saved["progress"] = progress

        def separate_audio_file(self, path):
            saved["source"] = path
            return None, {
                "vocals": FakeStem(100),
                "drums": FakeStem(1),
                "bass": FakeStem(2),
                "other": FakeStem(3),
            }

    def fake_save_audio(stem, path, samplerate):
        saved["stem_value"] = stem.value
        saved["samplerate"] = samplerate
        Path(path).write_bytes(b"background")

    api = types.SimpleNamespace(Separator=FakeSeparator, save_audio=fake_save_audio)
    fake_demucs = types.ModuleType("demucs")
    fake_demucs.api = api
    monkeypatch.setitem(sys.modules, "demucs", fake_demucs)

    result = separate_background_with_demucs(
        source,
        output,
        device="cpu",
    )

    assert result == str(output)
    assert saved["stem_value"] == 6
    assert saved["samplerate"] == 44100
    assert saved["device"] == "cpu"
    assert saved["progress"] is False


def test_demucs_adapter_rejects_missing_source(tmp_path):
    from buzz.localization.audio_mix import separate_background_with_demucs

    with pytest.raises(AudioMixingError, match="Source audio"):
        separate_background_with_demucs(
            tmp_path / "missing.wav",
            tmp_path / "background.wav",
            device="cpu",
        )


def test_target_duration_can_extend_past_last_speech(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript([make_segment(speech, 0.0, 1.0)]),
        tmp_path / "localized.wav",
        target_duration=3.5,
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "apad=whole_dur=3.500000" in filter_complex
    assert command[command.index("-t") + 1] == "3.500000"
    assert result.duration == 3.5


def test_target_duration_cannot_cut_off_last_speech(tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    with pytest.raises(AudioMixingError, match="before the last speech"):
        mix_localized_audio(
            make_transcript([make_segment(speech, 0.0, 2.0)]),
            tmp_path / "localized.wav",
            target_duration=1.5,
            ffmpeg_runner=FakeRunner(),
        )


@pytest.mark.parametrize("duration", [0, -1, "2", True])
def test_rejects_invalid_target_duration(duration, tmp_path):
    speech = tmp_path / "speech.wav"
    make_wav(speech)

    with pytest.raises(AudioMixingError, match="Target audio duration"):
        mix_localized_audio(
            make_transcript([make_segment(speech, 0.0, 1.0)]),
            tmp_path / "localized.wav",
            target_duration=duration,
            ffmpeg_runner=FakeRunner(),
        )

def test_target_duration_allows_small_asr_timestamp_drift(tmp_path):
    speech = tmp_path / "speech-drift.wav"
    make_wav(speech)
    runner = FakeRunner()

    result = mix_localized_audio(
        make_transcript([make_segment(speech, 0.0, 2.12)]),
        tmp_path / "localized-drift.wav",
        target_duration=2.0,
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[command.index("-t") + 1] == "2.120000"
    assert result.duration == pytest.approx(2.12)
