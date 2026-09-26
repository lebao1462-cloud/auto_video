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
| 7 | Final MP4 rendering | ⬜ NOT STARTED |
| 8 | GUI / end-to-end localization workflow | ⬜ NOT STARTED |
| 9 | Packaging / reliability / release | ⬜ NOT STARTED |

Current stable checkpoint:

    Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 complete
    Latest relevant regression run: 183 passed, 3 skipped, 0 failed

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

⬜ NOT STARTED

## Objective

Combine original video, localized Vietnamese audio, and Vietnamese subtitles into the final MP4.

## Required Work

Use FFmpeg or existing media stack after investigation.

Support:
- final localized audio
- original video stream when possible
- subtitle burn-in option
- optional soft subtitle track
- safe H.264/AAC output where required
- preserve resolution/frame rate where possible
- deterministic output path
- cleanup after failures
- never overwrite original input by accident

## Tests Required

- Output exists.
- Video stream playable.
- Audio stream playable.
- Duration matches source approximately.
- Vietnamese subtitle present.
- Source file remains unchanged.
- Failure cleanup works.

## Completion Criteria

One final Vietnamese-localized MP4 is produced from outputs of Phases 1-6.

---

# Phase 8 — End-to-End Workflow / GUI

## Status

⬜ NOT STARTED

## Objective

Allow a user to run localization without manually calling internal functions.

Desired flow:

    Choose video
      -> source language / auto
      -> translation provider
      -> Vietnamese TTS voice/provider
      -> Start
      -> progress by phase
      -> preview/result
      -> open output folder

## Required Work

- Orchestration service for Phases 1-7.
- Stable job state.
- Cancellation.
- Progress reporting.
- Error reporting.
- Provider/settings UI.
- Output folder settings.
- Retry/resume where practical.
- Do not block Qt UI thread.
- Preserve normal Buzz transcription workflows.

Potential future API:

    localize_video(...)

Potential future CLI:

    buzz localize input.mp4

Do not finalize CLI/UI until core pipeline APIs are stable.

## Completion Criteria

A normal user can localize a video end-to-end from one workflow.

---

# Phase 9 — Packaging / Reliability / Release

## Status

⬜ NOT STARTED

## Objective

Make the localization fork repeatable and usable outside the development workflow.

## Required Work

- Full regression suite.
- Windows packaging.
- Dependency audit.
- Model/cache path validation.
- First-run behavior.
- FFmpeg availability handling.
- Offline/network failure handling.
- API credential UX.
- Logs/diagnostics.
- Disk space checks.
- Large-video testing.
- Long-duration testing.
- Real English video testing.
- Real Chinese video testing.
- Vietnamese voice quality review.
- Final output quality review.
- User documentation.

## Performance Measurements

Measure:
- transcription time
- translation time
- TTS time
- synchronization time
- rendering time
- RAM usage
- CPU/GPU usage
- disk usage

## Completion Criteria

The complete pipeline can be installed and repeatedly used on the target Windows PC without developer intervention.

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

---

# Current Next Action

Current checkpoint:

    Phase 0 ✅
    Phase 1 ✅
    Phase 2 ✅
    Phase 3 ✅
    Phase 4 ✅
    Phase 5 ✅
    Phase 6 ✅
    Phase 7 ⬜
    Phase 8 ⬜
    Phase 9 ⬜

Next task:

    Design and implement Phase 7 — Final MP4 Rendering.

Do not begin Phase 8 or later until Phase 7 has:
- an approved FFmpeg rendering/muxing design
- targeted tests
- Phase 1/2/3/4/5/6 regression confirmation
- reviewed diff
- local commit and GitHub push checkpoint
