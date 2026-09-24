# Phase 2 — Source Transcript to Vietnamese Translation

## Objective

Build the second stage of the automatic video localization pipeline.

Input:

    LocalizationTranscript from Phase 1
    source_language = "en" or "zh"
    timestamped source segments

Output:

    Structured Vietnamese translation
    preserving the original segment timing and source text.

Conceptual flow:

    Phase 1 transcript
        ->
    translation service
        ->
    Vietnamese localization transcript

Phase 2 must not perform TTS, audio processing, subtitle burn-in, or video rendering.

---

## Core Principle

Reuse the stable Phase 1 transcript model.
Do not re-run transcription inside Phase 2.
Translation must be isolated behind a provider interface so the project is not locked to one paid API.

The translation layer should be reusable by later CLI, GUI, batch, and automation workflows.

---

## Step 1 — Inspect Existing Architecture

Before coding, inspect:

    buzz/localization/
    buzz/transcriber/
    buzz/settings/
    existing HTTP/API helpers
    tests/

Determine:

1. How Phase 1 exposes LocalizationTranscript.
2. Whether Buzz already contains translation-related helpers worth reusing.
3. How settings and API credentials are currently handled.
4. How external providers are configured without committing secrets.
5. Which layer should own translation orchestration.
6. How errors and cancellation are represented in existing Buzz code.

Do not place provider-specific code directly in the transcript data model.

---

## Step 2 — Define Phase 2 Data Model

Preserve the Phase 1 source transcript unchanged.
Introduce the smallest translation representation needed for later TTS.

Conceptually:

    TranslatedLocalizationTranscript
        source_file
        source_language
        target_language = "vi"
        segments[]

    TranslatedLocalizationSegment
        start
        end
        source_text
        translated_text

Requirements:

- Preserve original start/end timestamps exactly.
- Preserve original source text.
- Store Vietnamese text separately.
- Keep segment ordering stable.
- Be JSON-compatible.
- Do not mutate the input LocalizationTranscript.

Example:

    {
      "source_language": "en",
      "target_language": "vi",
      "segments": [
        {
          "start": 0.0,
          "end": 3.2,
          "source_text": "Hello everyone",
          "translated_text": "Xin chào mọi người"
        }
      ]
    }

---

## Step 3 — Translation Provider Interface

Create a small provider abstraction.

Conceptually:

    TranslationProvider
        translate(text, source_language, target_language) -> str

The orchestration layer must depend on this interface, not directly on OpenAI, Gemini, DeepL, Google, or another vendor.

This is required so providers can later be swapped between:

- free APIs/models
- paid APIs
- local models
- test/mocked providers

Provider credentials must come from environment/settings and must never be committed.

Do not add an unnecessary provider dependency merely to satisfy the interface.

---

## Step 4 — Translation Service

Create a localization translation service around Phase 1 output.

Conceptually:

    translate_for_localization(
        transcript,
        provider,
        target_language="vi"
    )

Responsibilities:

1. Validate the source transcript.
2. Accept English or Chinese input.
3. Translate each source segment into Vietnamese.
4. Preserve segment timestamps.
5. Preserve source text.
6. Return normalized translated data.
7. Propagate provider failures clearly.

The service must not know about video rendering, audio generation, or subtitles.

---

## Step 5 — Translation Quality and Context

A literal segment-by-segment translation can lose context.

Design the service so future context-aware translation is possible without breaking the public model.

For the first implementation:

- Keep output segmented one-to-one with Phase 1.
- Do not merge or split timestamps automatically.
- Allow provider implementations to receive nearby context if the existing architecture supports it cleanly.
- Prefer natural Vietnamese over word-for-word translation.
- Preserve names, numbers, technical terms, and punctuation where appropriate.

Do not attempt timing optimization for Vietnamese speech yet.
That belongs to the TTS/synchronization phases.

---

## Step 6 — Language Handling
Supported source languages:

    en
    zh

Target language:

    vi

Requirements:

- Reject unsupported source languages cleanly.
- Reject unsupported target languages in Phase 2.
- Do not silently guess a different source language.
- Use Phase 1's source_language as the authoritative input.
- Preserve Chinese characters in source_text.
- Preserve Unicode without lossy conversions.

---

## Step 7 — Error Handling

Handle at minimum:

- empty transcript
- empty source segment
- provider timeout
- provider authentication/configuration failure
- rate limit/provider unavailable
- malformed provider response
- partially translated batch

Do not silently drop segments.

If partial translation is supported, the result must make failed segments explicit.
Otherwise fail the operation clearly and leave the source transcript intact.

---

## Step 8 — Testing

Add targeted unit tests using fake/mock translation providers.

Tests must verify:
1. English -> Vietnamese translation mapping.
2. Chinese -> Vietnamese translation mapping.
3. Timestamp preservation.
4. Source text preservation.
5. Segment ordering.
6. Empty transcript handling.
7. Unsupported language rejection.
8. Provider exception propagation.
9. JSON-compatible output.
10. Input LocalizationTranscript is not mutated.

Unit tests must not require a real paid API key.

Add an optional integration test path for a real provider only if credentials are already configured.

Do not make normal test success depend on external network availability.

---

## Step 9 — Provider Configuration

Keep provider selection outside the core transcript model.

Prefer a structure that can later support:

    LOCALIZATION_TRANSLATION_PROVIDER
    provider-specific API key/base URL/model settings

Do not hard-code secrets.

Do not commit API keys or tokens.

Do not force one commercial provider as the only implementation.

The architecture should allow a free provider to be selected later without rewriting Phase 2.

---

## Step 10 — Backward Compatibility
Phase 2 must be additive.

Do not break:

- existing Buzz transcription UI
- existing transcription exports
- existing database behavior
- existing model download behavior
- Phase 1 transcript API
- existing CLI behavior

Avoid adding a CLI command until the current CLI architecture has been inspected.

---

## Explicitly Out of Scope

Do NOT implement during Phase 2:

    Vietnamese TTS
    voice cloning
    voice selection UI
    speech duration matching
    audio replacement
    background music separation
    subtitle burn-in
    final MP4 rendering
    lip synchronization
    publishing/upload

Those belong to later phases.

---

## Expected Deliverable

At the end of Phase 2:

    input video
        ->
    Phase 1 source transcript
        ->
    Phase 2 Vietnamese translation
        ->
    stable timestamped bilingual segment data

Example final Phase 2 structure:

    {
      "source_file": "input.mp4",
      "source_language": "zh",
      "target_language": "vi",
      "segments": [
        {
          "start": 0.0,
          "end": 2.4,
          "source_text": "大家好",
          "translated_text": "Xin chào mọi người"
        }
      ]
    }

This becomes the input to Phase 3 Vietnamese TTS.

---

## Implementation Procedure

When implementing Phase 2:

1. Read AGENTS.md completely.
2. Read PLAN_PHASE2.md completely.
3. Inspect Phase 1 implementation and relevant Buzz settings/API code.
4. Report proposed files/modules before editing.
5. Implement the smallest provider-neutral architecture.
6. Add targeted tests with fake providers.
7. Run Phase 2 tests.
8. Re-run Phase 1 localization tests.
9. Run relevant transcription regression tests.
10. Inspect git status.
11. Inspect git diff and git diff --check.
12. Do not modify uv.lock unless explicitly required.
13. Do not push.
14. Do not commit unless explicitly requested.
15. Produce the AGENTS.md structured report.

---

## Phase 2 Success Criteria

Phase 2 is complete only when:

- Phase 1 LocalizationTranscript can be translated without re-transcription.
- English source segments can become Vietnamese segments.
- Chinese source segments can become Vietnamese segments.
- Original timestamps remain unchanged.
- Original source text remains available.
- Vietnamese translated text is represented separately.
- Provider implementation is swappable.
- No secrets are committed.
- Core unit tests use mocks/fakes rather than paid API calls.
- Phase 1 tests still pass.
- Relevant Buzz regression tests still pass.
- No unrelated files are modified.
