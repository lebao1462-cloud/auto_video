# Auto Video Localization — Master Plan

## Project Goal

Build an end-to-end automatic video localization pipeline on top of Buzz.

Input:
- Video containing English or Chinese speech
- No Vietnamese subtitles required
- No Vietnamese voice required

Final output:
- Vietnamese-localized MP4
- Vietnamese speech
- Vietnamese subtitles
- synchronized timing
- preserved or mixed background audio

Pipeline:

    Video
      -> Phase 1: Source transcription
      -> Phase 2: Vietnamese translation
      -> Phase 3: Vietnamese TTS
      -> Phase 4: Timing synchronization
      -> Phase 5: Audio separation / replacement / mixing
      -> Phase 6: Vietnamese subtitle generation
      -> Phase 7: Final MP4 rendering
      -> Phase 8: End-to-end GUI / workflow
      -> Phase 9: Packaging / reliability / release
      -> Phase 10: Free-first translation providers (Argos + Gemini option)

## Status Legend

- ✅ COMPLETED — implemented, reviewed, tested, committed locally.
- 🟡 PARTIAL — some infrastructure exists but the phase is not complete.
- ⬜ NOT STARTED — implementation has not started.
- ⛔ OUT OF SCOPE — intentionally excluded from the current core goal.

## Current Status

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Project foundation / Buzz reuse | ✅ COMPLETED |
| 1 | Video -> timestamped source transcript | ✅ COMPLETED |
| 2 | Source transcript -> Vietnamese translation | ✅ COMPLETED |
| 3 | Vietnamese TTS | ✅ COMPLETED |
| 4 | Speech timing synchronization | ✅ COMPLETED |
| 5 | Audio separation / replacement / mixing | ✅ COMPLETED |
| 6 | Vietnamese subtitle generation | ✅ COMPLETED |
| 7 | Final MP4 rendering | ✅ COMPLETED |
| 8 | GUI / end-to-end localization workflow | ✅ COMPLETED |
| 9 | Packaging / reliability / release | ✅ COMPLETED |
| 10 | Free-first translation providers: Argos offline + Gemini optional | COMPLETED |

Current stable checkpoint:

    Phase 0 + Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 + Phase 7 + Phase 8 + Phase 9 + Phase 10 complete
    Phase 10 targeted tests: 56 passed, 0 failed
    Latest relevant regression run: 271 passed, 3 skipped, 0 failed
    Frozen Windows build smoke: Argos en -> vi PASS, zh -> en -> vi PASS, real translation PASS, GUI launch PASS

Local commits:

    87f3c69  feat: add localization transcription pipeline
    4fd47db  feat: add Vietnamese localization translation stage

Completed phases are pushed to GitHub after review, tests, and local commit.

---

## Global Architecture Rules

1. Reuse Buzz functionality instead of rebuilding working components.
2. Keep localization code isolated from the existing Buzz UI where practical.
3. Every phase must expose stable machine-readable input/output.
4. Every external AI/service integration should use a provider abstraction.
5. Do not lock the project to one paid provider.
6. Never commit API keys, tokens, passwords, or secrets.
7. Providers should be swappable between free APIs, paid APIs, local models, and mocks.
8. Do not silently modify uv.lock.
9. Preserve existing Buzz behavior.
10. Targeted tests must pass before moving to the next phase.
11. Re-run previous localization tests after later pipeline changes.
12. On this development PC, models/caches/temp files should use drive D.
13. After each completed phase passes review/tests, commit locally and push to GitHub.
14. Translation must be free-first: Argos Translate is the default path and must not require an API key.
15. Gemini translation is optional and must never be required for the offline/local path.
16. The GUI must make provider cost/connectivity requirements explicit before execution.

Development paths:

    BUZZ_MODEL_ROOT = D:\Buzz\Models
    uv cache        = D:\uv-cache
    test temp       = D:\BuzzTestCache\Temp

---

# Phase 0 — Project Foundation

## Status

✅ COMPLETED

## Purpose

Prepare the Buzz fork for controlled incremental development.

## Completed

- Buzz 1.4.4 architecture inspected.
- Python 3.12 environment prepared.
- Project .venv prepared.
- Required testing dependencies installed.
- Model cache redirected to drive D.
- Test/URL temporary files redirected to drive D during testing.
- AGENTS.md created.
- Planner -> Codex -> Reviewer workflow established.
- Git workflow established: inspect -> implement -> test -> review -> local commit -> push completed phase to GitHub.

## Completion Criteria

- Development environment can run localization tests. ✅
- Model downloads can be kept off drive C. ✅
- Codex can modify the repository safely. ✅
- Git checkpoints are available. ✅

---

# Phase 1 — Video to Timestamped Source Transcript

## Status

✅ COMPLETED

## Objective

Convert English or Chinese video/audio into a normalized source-language transcript with timestamps.

Input:

    English or Chinese video/audio

Output:

    LocalizationTranscript
        source_file
        source_language
        segments[]

    LocalizationSegment
        start
        end
        text

## Architecture

    Video/audio
      -> Buzz FileTranscriptionTask
      -> existing transcription backend
      -> Buzz segments
      -> localization adapter
      -> LocalizationTranscript

## Completed Work

- Added normalized localization transcript model.
- Added transcribe_for_localization(...).
- Reused existing Buzz transcription backends.
- Extracted shared file transcriber factory.
- Preserved configured/backend-detected source language.
- Supported English (en) and Chinese (zh).
- Preserved timestamp precision.
- Added JSON-compatible output.
- Prevented accidental use of the translation task.
- Preserved existing Buzz UI/export/database behavior.

## Key Files

    buzz/localization/transcript.py
    buzz/transcriber/file_transcriber_factory.py
    buzz/file_transcriber_queue_worker.py
    buzz/transcriber/file_transcriber.py
    buzz/transcriber/whisper_file_transcriber.py
    buzz/transcriber/openai_whisper_api_file_transcriber.py
    tests/localization/transcript_test.py

## Testing

    Phase 1 targeted tests: 12 / 12 PASS

Also covered by regression/integration tests:
- Whisper
- Hugging Face Whisper
- Faster Whisper
- OpenAI Whisper adapter
- URL input
- folder watch
- queue worker

## Commit

    87f3c69 feat: add localization transcription pipeline

## Completion Criteria

- Existing Buzz transcription reused. ✅
- English supported. ✅
- Chinese supported. ✅
- Source language available. ✅
- Timestamped segments available. ✅
- JSON-compatible normalized result. ✅
- No duplicate ASR engine. ✅
- Relevant regression tests pass. ✅

---

# Phase 2 — Source Transcript to Vietnamese Translation

## Status

✅ COMPLETED

## Objective

Translate Phase 1 output into Vietnamese while retaining source text and timestamps.

Input:

    LocalizationTranscript

Output:

    TranslatedLocalizationTranscript
        source_file
        source_language
        target_language
        segments[]

    TranslatedLocalizationSegment
        start
        end
        source_text
        translated_text

## Architecture

    LocalizationTranscript
      -> TranslationProvider
      -> translate_for_localization(...)
      -> TranslatedLocalizationTranscript

## Completed Work

- Added provider-neutral TranslationProvider protocol.
- Added immutable translated transcript model.
- Added translate_for_localization(...).
- English -> Vietnamese supported.
- Chinese -> Vietnamese supported.
- Original timestamp order/precision preserved.
- Original source text preserved.
- Vietnamese text stored separately.
- Input transcript is not mutated.
- Unsupported languages rejected.
- Empty segments rejected.
- Invalid provider response rejected.
- Provider errors propagate clearly.
- Partial provider failure does not silently return incomplete output.
- Core tests do not require a paid API.
- Existing Buzz OpenAI UI translator remains unchanged.

## Key Files

    buzz/localization/translation.py
    buzz/localization/__init__.py
    tests/localization/translation_test.py
    PLAN_PHASE2.md

## Testing

    Phase 2 targeted tests: 23 / 23 PASS
    Latest combined relevant tests: 88 passed, 3 skipped, 0 failed

The 3 skipped tests are existing Buzz tests:
1. Unix output-path case skipped on Windows.
2. Unix dated-output-path case skipped on Windows.
3. Upstream test_transcribe_stop explicitly marked skip.

No Phase 1 or Phase 2 localization test is intentionally skipped.

## Commit

    4fd47db feat: add Vietnamese localization translation stage

## Completion Criteria

- Phase 1 result translated without re-transcription. ✅
- English -> Vietnamese. ✅
- Chinese -> Vietnamese. ✅
- Timestamps preserved. ✅
- Source text preserved. ✅
- Vietnamese text represented separately. ✅
- Provider can be swapped. ✅
- No secrets committed. ✅
- Phase 1 regression remains healthy. ✅

---

# Phase 3 — Vietnamese Text-to-Speech

## Status

✅ COMPLETED

## Objective

Generate Vietnamese speech audio assets for translated segments through a provider-neutral TTS interface.

Input:

    TranslatedLocalizationTranscript

Output:

    SynthesizedLocalizationTranscript
        source_file
        source_language
        target_language
        segments[]

    SynthesizedLocalizationSegment
        start
        end
        source_text
        translated_text
        audio_file
        audio_duration
        provider
        voice
        provider_metadata

## Architecture

Provider-neutral TTS layer:

    TTSRequest
      ->
    TTSProvider.synthesize(...)
      ->
    TTSResult
      ->
    SynthesizedLocalizationTranscript

Concrete providers can later be added without changing the localization pipeline.

Possible providers:
- free cloud TTS
- paid cloud TTS
- Edge TTS where suitable
- local Vietnamese TTS
- other interchangeable engines

## Completed Work

- Inspected existing Buzz audio helpers before implementation.
- Added provider-neutral TTSProvider protocol.
- Added TTSRequest and TTSResult models.
- Added SynthesizedLocalizationTranscript and SynthesizedLocalizationSegment.
- Added synthesize_for_localization(...).
- Creates one deterministic synthesis request per translated segment.
- Preserves source text, Vietnamese translation, timestamps and segment order.
- Stores generated audio file path and duration.
- Stores provider, voice and provider-specific metadata.
- Validates provider response type, provider metadata, duration and audio-file existence.
- Supports provider options without mutating caller input.
- Output directory is caller-controlled, allowing assets to remain on drive D on this PC.
- Added no new runtime dependency and did not lock the pipeline to one TTS vendor.
- Added network-free fake-provider tests.
- Phase 4 timing adjustment was intentionally not implemented here.

## Key Files

    buzz/localization/tts.py
    buzz/localization/__init__.py
    tests/localization/tts_test.py

## Testing

    Phase 3 targeted tests: 22 / 22 PASS
    Latest combined relevant tests: 110 passed, 3 skipped, 0 failed

Covered:
- Vietnamese Unicode/text passthrough. ✅
- Segment order and deterministic audio mapping. ✅
- Audio duration/provider/voice metadata. ✅
- Provider options. ✅
- Empty translated text rejection. ✅
- Invalid/missing provider output rejection. ✅
- Provider failure propagation. ✅
- Input transcript immutability. ✅
- Phase 1/2 regression remains healthy. ✅
- Real Whisper/Hugging Face/Faster Whisper/URL regressions remain healthy. ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Completion Criteria

The Phase 3 localization core can request and validate stable Vietnamese speech assets through a swappable TTS provider. ✅

---

# Phase 4 — Speech Timing Synchronization

## Status

✅ COMPLETED

## Objective

Create deterministic timing metadata that fits Vietnamese speech into the original source timeline without modifying audio waveforms yet.

Input:

    SynthesizedLocalizationTranscript

Output:

    TimedLocalizationTranscript
        source_file
        source_language
        target_language
        timing_policy
        segments[]

    TimedLocalizationSegment
        original start/end
        source/translated text
        Phase 3 audio metadata
        slot_duration
        playback_rate
        adjusted_audio_duration
        leading_padding
        trailing_padding
        timing_action

## Architecture

Phase 4 creates a timing plan only.

    SynthesizedLocalizationTranscript
      -> TimingPolicy
      -> synchronize_for_localization(...)
      -> TimedLocalizationTranscript

Actual waveform time-stretching, silence insertion, and final timeline mixing remain Phase 5 responsibilities.

## Completed Work

- Added immutable TimingPolicy.
- Added TimedLocalizationTranscript and TimedLocalizationSegment.
- Added synchronize_for_localization(...).
- Original video/source segment timeline remains authoritative.
- Exact-duration speech uses playback rate 1.0.
- Slightly short speech may be slowed only within the configured natural limit.
- Much shorter speech keeps natural speed and uses trailing silence padding metadata.
- Slightly long speech may be sped up only within the configured maximum.
- Speech requiring excessive speed-up is rejected with TimingSynchronizationError instead of being unnaturally compressed.
- Default playback-rate limits are 0.90x minimum and 1.25x maximum.
- Adjacent source segments remain non-overlapping.
- Overlapping or invalid source timings are rejected.
- Phase 3 source text, translation, audio path, provider, voice and metadata are preserved.
- Input Phase 3 transcript is never mutated.
- Output remains JSON-compatible.
- No new dependency was introduced.

## Key Files

    buzz/localization/timing.py
    buzz/localization/__init__.py
    tests/localization/timing_test.py

## Testing

    Phase 4 targeted tests: 24 / 24 PASS
    Latest combined relevant tests: 136 passed, 3 skipped, 0 failed

Covered:
- exact-duration speech. ✅
- slightly shorter speech / limited slowdown. ✅
- much shorter speech / silence padding. ✅
- slightly longer speech / limited speed-up. ✅
- excessively long speech rejection. ✅
- custom timing policy. ✅
- adjacent segments. ✅
- overlap prevention. ✅
- invalid start/end/duration handling. ✅
- JSON-compatible output. ✅
- Phase 3 metadata preservation. ✅
- Phase 3 input immutability. ✅
- Phase 1/2/3 regression remains healthy. ✅
- real Whisper/Hugging Face/Faster Whisper/URL regressions remain healthy. ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Completion Criteria

Each valid Vietnamese segment now has deterministic timing metadata suitable for Phase 5 audio processing and placement. ✅

---

# Phase 5 — Audio Separation / Replacement / Mixing

## Status

✅ COMPLETED

## Objective

Create a synchronized full-length Vietnamese localized audio track while optionally preserving non-vocal background sound/music/effects.

## Architecture

Phase 5 reuses Buzz's bundled FFmpeg path strategy and Demucs dependency.

    source video/audio
      -> extract_audio_track(...)
      -> optional separate_background_with_demucs(...)
      -> TimedLocalizationTranscript + Vietnamese TTS assets
      -> mix_localized_audio(...)
      -> localized PCM WAV

The final MP4 mux/render remains Phase 7.

## Completed Work

- Added FFmpeg-based audio extraction from source media.
- Added optional Demucs background separation adapter.
- Demucs sums all non-vocal stems and excludes the vocals stem.
- Demucs model cache follows BUZZ_MODEL_ROOT/torch-hub when configured, keeping model downloads on drive D on this PC.
- Reused existing Buzz/Demucs dependencies; no new dependency added.
- Added FFmpeg-based application of Phase 4 playback-rate metadata.
- Places each Vietnamese speech segment at its original source start time.
- Supports multiple Vietnamese speech segments on one full timeline.
- Supports speech-only replacement output.
- Supports optional background audio mixing.
- Supports configurable speech/background gain.
- Uses a limiter on the final mix to reduce clipping risk.
- Supports mono or stereo output and configurable sample rate.
- Pads/trims streams to deterministic final timeline duration.
- Produces PCM WAV suitable for Phase 7 video rendering/muxing.
- Added LocalizedAudioResult metadata model.
- Keeps the source input files unchanged.

## Key Files

    buzz/localization/audio_mix.py
    buzz/localization/__init__.py
    tests/localization/audio_mix_test.py

## Testing

    Phase 5 targeted tests: 22 / 22 PASS
    Latest combined relevant tests: 158 passed, 3 skipped, 0 failed

Covered:
- speech-only mix command. ✅
- playback-rate/time-stretch metadata application. ✅
- segment start-delay placement. ✅
- multi-segment timeline. ✅
- optional background mixing. ✅
- configurable speech/background gain. ✅
- sample rate/channel configuration. ✅
- missing speech/background files. ✅
- clipping limiter in final FFmpeg graph. ✅
- JSON-compatible result metadata. ✅
- audio extraction command. ✅
- real FFmpeg audio extraction integration. ✅
- real FFmpeg 3-second speech/background mixing integration. ✅
- Demucs non-vocal stem combination with mocked model API. ✅
- Phase 1/2/3/4 regression remains healthy. ✅
- real Whisper/Hugging Face/Faster Whisper/URL regressions remain healthy. ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Completion Criteria

The localization core can now create a synchronized full-length Vietnamese audio track, either speech-only or mixed with preserved background audio, ready for subtitle and final MP4 stages. ✅

---

# Phase 6 — Vietnamese Subtitle Generation

## Status

✅ COMPLETED

## Objective

Generate valid UTF-8 Vietnamese subtitle files directly from Phase 4 timed localization data.

Input:

    TimedLocalizationTranscript

Output:

    SubtitleResult
        subtitle_file
        format
        cue_count

Supported formats:
- SRT ✅
- VTT ✅
- ASS deferred until richer styling is required

## Architecture

Phase 6 reuses Buzz's existing subtitle timestamp formatter to stay consistent with native Buzz exports.

    TimedLocalizationTranscript
      -> subtitle validation
      -> text normalization / optional line wrapping
      -> render_srt(...) or render_vtt(...)
      -> write_subtitles(...)
      -> UTF-8 subtitle file

## Completed Work

- Added SubtitleOptions for configurable line wrapping.
- Added SubtitleResult metadata model.
- Added render_srt(...).
- Added render_vtt(...).
- Added write_subtitles(...).
- Reused Buzz to_timestamp(...) formatting behavior.
- Converts Phase 4 second-based timestamps to subtitle milliseconds.
- Preserves Vietnamese Unicode.
- Preserves cue order and source timeline timestamps.
- Uses comma millisecond separator for SRT.
- Uses decimal-point millisecond separator for WebVTT.
- Adds required WEBVTT header.
- Supports configurable maximum line length and maximum line count.
- Can disable automatic line wrapping.
- Normalizes excessive internal whitespace/newlines.
- Rejects empty Vietnamese text.
- Rejects invalid or overlapping subtitle intervals.
- Rejects unsupported target language and output format.
- Ensures output extension matches requested format.
- Writes subtitle files as UTF-8 with deterministic LF newlines.
- Does not mutate Phase 4 input data.
- Added no new dependency.

## Key Files

    buzz/localization/subtitles.py
    buzz/localization/__init__.py
    tests/localization/subtitles_test.py

## Testing

    Phase 6 targeted tests: 25 / 25 PASS
    Latest combined relevant tests: 183 passed, 3 skipped, 0 failed

Covered:
- Vietnamese Unicode. ✅
- SRT formatting. ✅
- WebVTT formatting/header. ✅
- hour/minute/second/millisecond timestamps. ✅
- millisecond rounding. ✅
- segment order/count. ✅
- configurable multiline wrapping. ✅
- whitespace normalization. ✅
- empty-text rejection. ✅
- invalid interval rejection. ✅
- overlap prevention. ✅
- unsupported language/format rejection. ✅
- UTF-8 file writing. ✅
- JSON-compatible result metadata. ✅
- Phase 4 input immutability. ✅
- Phase 1/2/3/4/5 regression remains healthy. ✅
- real Whisper/Hugging Face/Faster Whisper/URL regressions remain healthy. ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Completion Criteria

Valid Vietnamese SRT/VTT subtitle files can now be generated independently and are ready for Phase 7 final MP4 rendering or burn-in. ✅

---

# Phase 7 — Final MP4 Rendering

## Status

✅ COMPLETED

## Objective

Combine the original video, localized Vietnamese audio, and Vietnamese subtitles into a final playable MP4.

## Architecture

Phase 7 reuses the FFmpeg runner introduced in Phase 5.

    source video
    + localized audio from Phase 5
    + optional subtitle file from Phase 6
      -> render_localized_mp4(...)
      -> final localized MP4

Supported subtitle modes:
- soft subtitle track (mov_text) ✅
- burned-in subtitle ✅
- no subtitle ✅

## Completed Work

- Added FinalRenderOptions.
- Added FinalRenderResult.
- Added FinalRenderError.
- Added render_localized_mp4(...).
- Replaces original audio with localized Vietnamese audio.
- Soft-subtitle mode copies the original video stream when possible.
- Soft subtitles are muxed as MP4 mov_text.
- Vietnamese subtitle stream is tagged with language vie.
- Burn-in mode re-encodes video using configurable H.264-compatible codec/settings.
- Burn-in uses the FFmpeg subtitles filter and safely escapes Windows subtitle paths.
- Audio is encoded as configurable AAC/bitrate by default.
- Uses -shortest to avoid trailing media beyond the shortest required stream.
- Uses +faststart for better MP4 playback/startup behavior.
- Preserves source file by explicitly rejecting source/output path collisions.
- Rejects missing source video, localized audio, subtitle file, unsupported subtitle extension and non-MP4 output.
- Removes partial output after renderer failures.
- Validates output creation and non-zero output size.
- Added no new dependency.

## Key Files

    buzz/localization/final_render.py
    buzz/localization/__init__.py
    tests/localization/final_render_test.py

## Testing

    Phase 7 targeted tests: 22 / 22 PASS
    Latest combined relevant tests: 205 passed, 3 skipped, 0 failed

Covered:
- soft subtitle FFmpeg command. ✅
- burn-in subtitle FFmpeg command. ✅
- no-subtitle mode. ✅
- localized audio replacement. ✅
- video stream copy for soft subtitle mode. ✅
- configurable H.264 CRF/preset for burn-in mode. ✅
- AAC audio encoding. ✅
- Vietnamese subtitle language metadata. ✅
- missing/invalid input handling. ✅
- source overwrite prevention. ✅
- partial-output cleanup after failure. ✅
- JSON-compatible result metadata. ✅
- real FFmpeg MP4 render with video + audio + soft subtitle. ✅
- real FFmpeg MP4 render with burned-in Vietnamese subtitle. ✅
- ffprobe validation of video/audio/subtitle streams. ✅
- Phase 1/2/3/4/5/6 regression remains healthy. ✅
- real Whisper/Hugging Face/Faster Whisper/URL regressions remain healthy. ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Completion Criteria

The core pipeline can now create a final Vietnamese-localized MP4 from prepared Phase 5 audio and Phase 6 subtitles while keeping the original source video unchanged. ✅

---

# Phase 8 — End-to-End Workflow / GUI

## Status

✅ COMPLETED

## Objective

Allow a normal user to run the complete localization pipeline without manually calling internal Phase 1-7 APIs.

## Completed Work

- Added localize_video(...) orchestration across Phases 1-7.
- Added stable workflow options/result/progress models.
- Added stage-by-stage progress reporting.
- Added cancellation checkpoints between pipeline stages.
- Added workspace cleanup helper that preserves final outputs.
- Added source-media duration probing so final localized audio can match the original video duration.
- Added optional background separation in the end-to-end workflow.
- Added concrete OpenAI-compatible translation provider for configurable API/base URL/model.
- Added Vietnamese Edge TTS provider with selectable female/male Vietnamese voices.
- Added LocalizationDialog using a background QThread; the Qt UI is not blocked by the pipeline.
- Added input video picker, output directory picker, Whisper model picker, source language selector, translation provider settings, Vietnamese voice selection, subtitle mode and optional Demucs background preservation.
- Translation API key field is masked and can reuse the existing Buzz keyring value.
- Added progress bar, status messages and cancel control.
- Added File menu action: Localize Video to Vietnamese...
- Preserved existing Buzz transcription/import workflows.
- Added an end-to-end FFmpeg integration test that creates a real source MP4 and validates final video/audio/subtitle streams and duration.
- Added no hard-coded paid provider requirement; translation endpoint remains OpenAI-compatible and configurable.
- edge-tts remains runtime-optional for Phase 8 because the upstream Buzz uv lock cannot currently be regenerated while its NVIDIA registry is unavailable. Phase 9 preflight/install documentation handles this explicitly.

## Key Files

    buzz/localization/providers.py
    buzz/localization/workflow.py
    buzz/widgets/localization_dialog.py
    buzz/widgets/main_window.py
    buzz/widgets/menu_bar.py
    tests/localization/providers_test.py
    tests/localization/workflow_test.py
    tests/widgets/localization_dialog_test.py
    tests/widgets/menu_bar_test.py

## Testing

    Phase 8 targeted tests: 19 / 19 PASS
    Latest combined relevant tests: 230 passed, 3 skipped, 0 failed

Covered:
- OpenAI-compatible English -> Vietnamese translation provider. ✅
- Chinese -> Vietnamese provider path. ✅
- Edge TTS validation and audio-duration probing. ✅
- Complete Phase 1-7 orchestration with fake translation/TTS providers. ✅
- Real FFmpeg final MP4 creation through workflow. ✅
- Final video/audio/subtitle stream validation with ffprobe. ✅
- Final duration matching source video. ✅
- Cancellation before workflow start. ✅
- Workspace cleanup preserving final outputs. ✅
- GUI defaults/input validation. ✅
- File menu localization action/signal. ✅
- Existing localization and Buzz transcriber regressions remain healthy. ✅

## Completion Criteria

A normal user can launch the Vietnamese localization dialog from Buzz, choose a video/settings, start the complete workflow, observe progress/cancel it, and receive a localized MP4/subtitle output. ✅

---

# Phase 9 — Packaging / Reliability / Release

## Status

✅ COMPLETED

## Objective

Make the localization fork repeatable, diagnosable, packageable, and usable on the target Windows PC without requiring developer-only workflow steps.

## Completed Work

- Added localization preflight checks before GUI execution.
- Preflight validates source video, Whisper model path, FFmpeg, ffprobe, Edge TTS, optional Demucs, translation configuration, output write access, and estimated free disk space.
- Added conservative workspace estimate: max(512 MiB, 6x compressed source size).
- Added non-secret localization_diagnostics.json output.
- Diagnostics explicitly avoid storing translation API keys.
- Added warning-level model/cache drive policy check for this Windows development PC.
- Added first-run and troubleshooting documentation in LOCALIZATION_GUIDE.md.
- Added pinned localization runtime provider requirement in localization-requirements.txt.
- Kept edge-tts runtime optional for normal Buzz import paths; localization preflight reports it clearly when absent.
- Updated Buzz.spec to bundle Edge TTS when installed in the build environment.
- Made optional upstream metadata/whisper.cpp/dll bundle paths packaging-safe for this source snapshot.
- Installed PyInstaller only in the local .venv for packaging validation; no project lockfile was silently modified.
- Corrected the local build environment from onnxruntime 1.30.0 to Buzz's declared onnxruntime 1.18.1 requirement.
- Windows PyInstaller packaging completed successfully.
- Verified packaged Buzz.exe exists and starts without immediate crash.
- Verified packaged ffmpeg.exe and ffprobe.exe are present.
- Verified edge_tts files are present in the packaged application.
- Preserved generated build artifacts outside Git in D:\AutoVideoBuild.
- Full relevant regression suite remains green after packaging-environment correction.

## Key Files

    buzz/localization/preflight.py
    buzz/localization/__init__.py
    buzz/widgets/localization_dialog.py
    Buzz.spec
    LOCALIZATION_GUIDE.md
    localization-requirements.txt
    tests/localization/preflight_test.py

## Testing

    Phase 9 targeted tests: 11 / 11 PASS
    Final combined relevant tests: 238 passed, 3 skipped, 0 failed

Packaging validation:
- PyInstaller Windows build: PASS ✅
- Build output: D:\AutoVideoBuild\dist\Buzz\Buzz.exe ✅
- Packaged Buzz.exe smoke launch: PASS ✅
- Packaged ffmpeg.exe: present ✅
- Packaged ffprobe.exe: present ✅
- Packaged edge_tts runtime: present ✅
- Source repository build artifacts: not committed ✅

The same 3 upstream Buzz tests remain skipped:
1. Unix output-path case on Windows.
2. Unix dated-output-path case on Windows.
3. test_transcribe_stop, explicitly skipped upstream.

## Release Validation Notes

Automated tests cover the full pipeline with deterministic fake translation/TTS providers plus real FFmpeg rendering and packaged-app smoke launch.

Live translation quality, Vietnamese voice preference, provider quota/rate limits, and long real-world English/Chinese source quality depend on the user's configured external provider and media. These are documented as operational acceptance checks rather than being silently exercised with user credentials.

Performance varies substantially by Whisper model, CPU/GPU, Demucs use, video length, and external-provider latency. Stage progress and diagnostics are available, but no single benchmark is treated as a release gate.

## Completion Criteria

The complete localization workflow is packageable and runnable on the target Windows PC, performs preflight/diagnostics before execution, and has a documented first-run path without developer intervention. ✅

---

# Phase 10 — Free-First Translation Providers (Argos + Gemini)

## Status

COMPLETED

## Objective

Remove the requirement for an OpenAI-compatible API key from the normal localization workflow while preserving an optional higher-quality online translation path.

The preferred default workflow must be usable with no translation API key:

    English / Chinese video
      -> Whisper local transcription
      -> Argos Translate local/offline
      -> Edge TTS Vietnamese
      -> timing synchronization
      -> FFmpeg audio/subtitles/final MP4

An optional online workflow may use Gemini:

    English / Chinese video
      -> Whisper local transcription
      -> Gemini translation
      -> Edge TTS Vietnamese
      -> timing synchronization
      -> FFmpeg audio/subtitles/final MP4

## Provider Policy

GUI provider choices:

    Translation Provider

    1. Argos Translate — Offline / Free / No API key   [DEFAULT]
    2. Gemini API — Online / Optional
    3. OpenAI-compatible — Advanced / Existing compatibility

Rules:
- Argos is the default provider.
- Argos must not require an API key.
- Gemini must remain optional.
- OpenAI-compatible provider remains available for compatibility but is not the default.
- Switching providers must not change downstream TTS/timing/audio/subtitle/render APIs.
- No provider secret may be written to logs, diagnostics, source files, Git, or generated subtitle/video metadata.

## Argos Translate Design

### Goal

Provide a fully local/offline translation path after language packages are installed.

Required source paths:
- English -> Vietnamese
- Chinese -> Vietnamese when a direct Argos package exists
- Chinese -> English -> Vietnamese pivot when direct Chinese -> Vietnamese is unavailable

Required behavior:
- Detect installed Argos language packages.
- Prefer a direct translation package when available.
- Fall back to a deterministic pivot route only when required packages are installed.
- Never silently download language packages during a localization run.
- Provide a clear install/preflight message for missing language packages.
- Keep Argos packages/cache on drive D on this development PC where configurable.
- Preserve segment order, timestamps, source text, and immutable Phase 2 output contracts.

Recommended provider API:

    ArgosTranslationProvider
        translate(text, source_language, target_language) -> str

Optional helper/service:

    ArgosLanguagePackageManager
        list_installed()
        route_available(source_language, target_language)
        install_from_local_package(...)
        refresh_package_index(...)  # explicit user action only

### Argos Preflight

For English source:
- en -> vi route must exist.

For Chinese source:
- prefer zh -> vi direct route;
- otherwise require both zh -> en and en -> vi.

Preflight must report the exact missing route/package instead of requesting an API key.

## Gemini Translation Design

### Goal

Provide an optional online translation provider for users who prefer better translation quality and accept API usage/quota requirements.

Required behavior:
- Dedicated GeminiTranslationProvider.
- API key required only when Gemini is selected.
- Configurable Gemini model.
- Clear timeout/network/rate-limit/quota errors.
- No automatic fallback from Argos to Gemini without explicit user selection.
- No automatic paid-provider fallback.
- Preserve the existing TranslationProvider protocol.
- Keep prompts deterministic and focused on translation only.
- Preserve meaning and concise phrasing suitable for speech timing.

Recommended configuration:

    GEMINI_API_KEY
    GEMINI_MODEL

GUI should allow:
- API key entry
- model selection/text field
- optional secure reuse from keyring when supported

## Translation Quality / Timing Rules

Translation output is consumed by Vietnamese TTS, so providers should prefer concise natural Vietnamese rather than unnecessarily verbose wording.

Rules:
- Do not summarize away source meaning.
- Do not add commentary or explanations.
- Preserve numbers, units, product names, and important entities.
- Prefer concise Vietnamese where multiple accurate phrasings exist.
- Return plain translated text only.
- Empty or non-string provider responses are invalid.
- Provider failures must not produce partial final videos silently.

## GUI Changes

Update LocalizationDialog translation section:

    Translation Provider:
        Argos Translate — Offline / Free
        Gemini API — Online
        OpenAI-compatible — Advanced

When Argos is selected:
- hide/disable API-key fields;
- show installed route/package status;
- show Offline / No API key;
- block Start only if required Argos route is missing.

When Gemini is selected:
- show Gemini API key/model controls;
- preflight key/model/network-facing configuration.

When OpenAI-compatible is selected:
- retain current base URL/API key/model controls.

Provider choice should be persisted in normal user settings where appropriate.

## Workflow Changes

Workflow orchestration must receive a TranslationProvider instance without branching on provider internals.

Expected structure:

    UI/provider factory
       -> TranslationProvider
       -> translate_for_localization(...)
       -> existing Phase 3-7 pipeline unchanged

This phase should not rewrite:
- transcription
- TTS
- timing
- audio mixing
- subtitles
- final render

## Packaging / Dependency Strategy

Argos and Gemini dependencies must be investigated before changing the main Buzz dependency lock.

Rules:
- Do not silently modify uv.lock.
- Prefer a dedicated localization requirements file if upstream Buzz lock regeneration remains blocked.
- Package/import optional providers safely.
- Normal Buzz transcription should still launch if Argos/Gemini optional dependencies are missing.
- Localization preflight must explain missing provider runtime dependencies.

Argos package/model files should not be committed to Git.

## Tests Required

### Argos
- English -> Vietnamese direct route.
- Chinese -> Vietnamese direct route when available.
- Chinese -> English -> Vietnamese pivot route.
- Direct route preferred over pivot.
- Missing route reports clear error.
- No API key required.
- Empty input/output validation.
- Provider exception propagation.
- Unicode Vietnamese output.
- No mutation of source transcript.

### Gemini
- English -> Vietnamese request/response.
- Chinese -> Vietnamese request/response.
- Missing API key rejection.
- Missing model rejection.
- Invalid/empty response rejection.
- Network/provider error propagation.
- API key never appears in diagnostics/log fixtures.

### GUI / Preflight
- Argos is default selection.
- Argos hides API key controls.
- Missing Argos package blocks Start with actionable message.
- Gemini shows API key/model controls.
- OpenAI-compatible remains selectable.
- Provider-specific preflight logic works.
- Existing GUI behavior remains healthy.

### End-to-End
- Real/fake English video workflow using Argos path.
- Chinese workflow using Argos direct or pivot route.
- Gemini path with mocked network response.
- Existing Edge TTS / timing / FFmpeg pipeline remains unchanged.
- Real mayhutbui.mp4 is re-tested with Argos if Chinese route packages are available.

## Completed / Validation

- Argos Translate is the default translation provider and requires no translation API key.
- English uses the installed `en -> vi` Argos route.
- Chinese uses direct `zh -> vi` when available; on this development PC the validated route is `zh -> en -> vi`.
- Gemini API is available as an optional online provider.
- OpenAI-compatible translation remains available as the Advanced compatibility option.
- Argos package discovery/install is explicit; localization runs never silently download packages.
- Argos runtime work is isolated through the `--argos-worker` subprocess path to avoid native-library conflicts with the main Buzz process.
- Windows frozen build was hardened by excluding the older PyQt-bundled `PyQt6/Qt6/bin/MSVCP140.dll` and keeping the newer runtime at `_internal/MSVCP140.dll`; this fixed the frozen Argos `0xC0000005` crash without breaking the GUI.
- Final Windows build: `D:\AutoVideoBuildPhase10Final\dist\Buzz\Buzz.exe` (~73.3 MB executable, ~2.26 GB full onedir distribution).
- Frozen build smoke tests: `en -> vi` route PASS, `zh -> en -> vi` route PASS, real English -> Vietnamese translation PASS.
- Frozen GUI smoke: process remained healthy after 10 seconds; no new Application Error event.
- `python -m compileall -q buzz`: PASS.
- Phase 10 targeted tests: **56 passed, 0 failed**.
- Relevant full regression: **271 passed, 3 skipped, 0 failed**.
- The three skips are unchanged upstream/platform-specific tests.
- Build/test/cache/temp paths were kept on drive D for final validation.

## Completion Criteria

Phase 10 is complete when:

- [x] A user can localize English video to Vietnamese with Argos without any translation API key.
- [x] A Chinese video can use a documented Argos direct or pivot route when corresponding packages are installed.
- [x] Gemini is available as an optional online provider.
- [x] OpenAI-compatible provider remains available as an advanced compatibility option.
- [x] GUI clearly distinguishes offline/free vs online/API providers.
- [x] Preflight validates only the requirements of the selected provider.
- [x] Existing Phase 1-9 behavior remains regression-safe.
- [x] Targeted tests, full regression, review, commit, and GitHub push complete.

---

# Deferred / Optional Features

## Status

⛔ OUT OF CURRENT CORE SCOPE

Possible future features:
- voice cloning
- speaker diarization-aware Vietnamese voices
- different Vietnamese voices for different speakers
- lip synchronization
- translation review/editor
- transcript editor
- batch localization
- YouTube/TikTok automatic publishing
- cloud processing
- distributed jobs
- project history
- subtitle style templates

These must not block the core localization pipeline.

---

# Mandatory Validation Before Moving to the Next Phase

For every new phase:

1. Read AGENTS.md.
2. Read this PLAN.md.
3. Inspect current implementation first.
4. List proposed files/modules before editing.
5. Make the smallest reasonable changes.
6. Preserve existing Buzz functionality.
7. Add targeted unit tests.
8. Run the new phase tests.
9. Re-run all previous localization phase tests.
10. Run relevant Buzz regression tests.
11. Keep model/cache/temp downloads on drive D on this PC.
12. Run git diff --check.
13. Inspect git status.
14. Review the diff.
15. Do not silently modify lock files.
16. Commit locally only after review/tests pass.
17. Push the completed phase to GitHub after the commit succeeds.
18. Produce the AGENTS.md structured report.

---

# Overall Definition of Done

The project is complete when the user can provide:

    an English or Chinese video with no subtitles

and automatically receive:

    a Vietnamese-localized MP4
    with Vietnamese speech
    synchronized timing
    preserved/mixed background audio
    Vietnamese subtitles

without manually performing transcription, translation, TTS, timing, audio mixing, subtitle creation, or rendering.

The default translation path should work without a translation API key by using Argos Translate after the required offline language packages are installed. Gemini and OpenAI-compatible providers remain optional alternatives.

---

# Current Next Action

Current checkpoint:

    Phase 0 DONE
    Phase 1 DONE
    Phase 2 DONE
    Phase 3 DONE
    Phase 4 DONE
    Phase 5 DONE
    Phase 6 DONE
    Phase 7 DONE
    Phase 8 DONE
    Phase 9 DONE
    Phase 10 DONE

Project status:

    CORE LOCALIZATION PIPELINE COMPLETE
    FREE-FIRST TRANSLATION PROVIDERS COMPLETE

Next task:

    Phase 10 is closed after tests, frozen-build smoke validation, review, commit, and GitHub push.
    Define the next roadmap phase only when a new requirement is approved.
