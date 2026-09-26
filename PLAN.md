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
| 4 | Speech timing synchronization | ⬜ NOT STARTED |
| 5 | Audio separation / replacement / mixing | ⬜ NOT STARTED |
| 6 | Vietnamese subtitle generation | ⬜ NOT STARTED |
| 7 | Final MP4 rendering | ⬜ NOT STARTED |
| 8 | GUI / end-to-end localization workflow | ⬜ NOT STARTED |
| 9 | Packaging / reliability / release | ⬜ NOT STARTED |

Current stable checkpoint:

    Phase 1 + Phase 2 + Phase 3 complete
    Latest relevant regression run: 110 passed, 3 skipped, 0 failed

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

⬜ NOT STARTED

## Objective

Fit generated Vietnamese speech naturally into the original segment timeline.

Problem:

    generated TTS duration != source segment duration

## Required Work

Investigate:
- silence trimming
- pause insertion
- limited speed-up
- limited slow-down
- padding
- overlap prevention
- neighboring timing flexibility

## Rules

- Original video timeline is authoritative.
- Do not silently remove spoken content.
- Avoid unnatural speed changes.
- Timing logic must remain separate from TTS provider code.
- Preserve subtitle compatibility.

## Tests Required

- TTS shorter than slot.
- TTS equal to slot.
- TTS slightly longer.
- TTS much longer.
- Adjacent segments.
- Invalid/zero duration.
- No mutation of previous-phase data.

## Completion Criteria

Each Vietnamese segment has deterministic synchronized timing metadata suitable for final audio placement.

---

# Phase 5 — Audio Separation / Replacement / Mixing

## Status

⬜ NOT STARTED

## Objective

Create a full localized audio track while retaining appropriate background sound/music/effects.

## Investigation Required

Compare strategies:
1. Replace all original audio.
2. Lower original audio and overlay Vietnamese.
3. Separate speech/vocals from background.
4. Preserve background and replace only speech.

Buzz already contains Demucs-related infrastructure, so inspect/reuse it before adding another separation dependency.

## Required Work

- Extract original audio.
- Optional speech/vocal separation.
- Place synchronized Vietnamese speech on timeline.
- Apply gain/normalization.
- Mix background audio.
- Avoid clipping.
- Preserve channels/sample rate as appropriate.
- Produce final localized audio track.

## Tests Required

- Speech-only source.
- Background/music source.
- Silence.
- Overlap handling.
- Clipping prevention.
- Final audio duration matches video timeline.

## Completion Criteria

A full-length Vietnamese localized audio track is available and synchronized to the original video.

---

# Phase 6 — Vietnamese Subtitle Generation

## Status

⬜ NOT STARTED

## Objective

Generate Vietnamese subtitles from translated/timed segments.

Output:
- SRT required
- VTT optional
- ASS optional for richer styling

## Required Work

- Map Vietnamese translation to segment timestamps.
- Preserve Unicode.
- Format timestamps correctly.
- Handle punctuation/line wrapping.
- Prevent invalid intervals.
- Prepare optional style metadata for final burn-in.

## Tests Required

- Vietnamese Unicode.
- Timestamp formatting.
- Segment order.
- Multiline subtitles.
- Empty-text rejection.
- Subtitle count matches translated segments.

## Completion Criteria

Valid Vietnamese subtitle files can be generated independently of final rendering.

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
    Phase 4 ⬜
    Phase 5 ⬜
    Phase 6 ⬜
    Phase 7 ⬜
    Phase 8 ⬜
    Phase 9 ⬜

Next task:

    Design and implement Phase 4 — Speech Timing Synchronization.

Do not begin Phase 5 or later until Phase 4 has:
- an approved timing/synchronization design
- targeted tests
- Phase 1/2/3 regression confirmation
- reviewed diff
- local commit and GitHub push checkpoint
