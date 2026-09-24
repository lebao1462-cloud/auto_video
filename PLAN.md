# Phase 1  Video to Timestamped Source Transcript

## Objective

Build the first stage of the automatic video localization pipeline.

Input:

    Video containing English or Chinese speech
    No subtitles are required.

Output:

    Structured transcript containing:
    - detected/source language
    - transcription text
    - segment start time
    - segment end time

Example conceptual output:

    [
      {
        "start": 0.0,
        "end": 3.2,
        "text": "Hello everyone",
        "language": "en"
      },
      {
        "start": 3.2,
        "end": 6.8,
        "text": "Today we are going to...",
        "language": "en"
      }
    ]

This output will later become the input for Phase 2 Vietnamese translation.

---

## Core Principle

Do NOT build speech recognition from scratch.

Buzz already contains transcription infrastructure and multiple transcription backends.

Phase 1 should reuse the existing Buzz architecture wherever practical.

Before implementing anything, inspect the existing code and determine the cleanest reusable path from:

    video file
        ->
    existing Buzz media/transcription infrastructure
        ->
    transcription backend
        ->
    timestamped segments
        ->
    normalized localization transcript

Avoid duplicating functionality already implemented by Buzz.

---

## Step 1  Architecture Investigation

Inspect the existing Buzz source code, especially:

    buzz/transcriber/
    buzz/cli.py
    buzz/buzz.py
    buzz/widgets/
    tests/

Identify:

1. How Buzz accepts audio/video files.
2. How media audio is extracted or passed to transcription engines.
3. How transcription tasks/jobs are created.
4. How transcription engines are selected.
5. Where Whisper / Faster Whisper results are converted into Buzz transcript objects.
6. How timestamps are represented.
7. How detected language is represented.
8. How transcription results are exported to SRT/VTT/TXT.
9. Which existing components can be reused without depending on the GUI.
10. Whether an existing CLI/service/API layer can be reused for the localization pipeline.

Do not make architectural assumptions before inspecting these components.

---

## Step 2  Define a Normalized Transcript Model

Phase 1 should expose a small internal representation suitable for later localization stages.

Conceptually:

    LocalizationTranscript
        source_language
        source_file
        segments[]

    LocalizationSegment
        start
        end
        text

The implementation should follow the conventions already used by Buzz.

Do not introduce a new model if an existing Buzz entity already provides the required information cleanly.

The normalized representation must preserve timestamp precision required for later:

    translation
    TTS generation
    speech timing synchronization
    subtitle generation

---

## Step 3  Reuse Existing Transcription Pipeline

Create the smallest reasonable adapter/service around existing Buzz functionality.

Desired conceptual architecture:

    Video
      |
      v
    Buzz existing media handling
      |
      v
    Existing transcription backend
      |
      v
    Buzz transcription result
      |
      v
    Localization transcript adapter
      |
      v
    Timestamped source transcript

The localization layer should NOT directly contain Whisper-specific logic unless the existing Buzz architecture requires it.

This is important because future versions should be able to use different Buzz transcription backends.

---

## Step 4  Language Handling

Phase 1 must support:

    English
    Chinese

Prefer existing automatic language detection when supported by the selected transcription backend.

The normalized result must expose the detected or configured source language.

Do not implement translation in this phase.

Do not convert the transcript to Vietnamese.

Phase 1 ends with the original-language transcript.

---

## Step 5  Machine-Readable Output

Provide a machine-readable representation that later phases can consume.

Preferred format:

    JSON-compatible Python structures

Potential serialized format:

    JSON

Example:

    {
      "source_file": "input.mp4",
      "source_language": "zh",
      "segments": [
        {
          "start": 0.0,
          "end": 2.4,
          "text": "大家好"
        }
      ]
    }

Do not replace Buzz's existing SRT/VTT/TXT export system.

This is an additional internal localization representation.

---

## Step 6  CLI / Pipeline Entry Point

Investigate the cleanest way to expose Phase 1 without disrupting the existing Buzz CLI.

Desired future usage could conceptually resemble:

    buzz localize input.mp4

or an internal Python service such as:

    transcribe_for_localization(video_path)

However:

Do NOT add a new CLI command merely because this plan shows one.

First inspect the current CLI architecture.

Choose the smallest architecture that can later support:

    Phase 1 transcription
        ->
    Phase 2 translation
        ->
    Phase 3 TTS
        ->
    Phase 4 synchronization
        ->
    Phase 5 audio mixing
        ->
    Phase 6 subtitles
        ->
    final MP4

---

## Step 7  Testing

Add targeted tests for newly introduced Phase 1 code.

Tests should verify at minimum:

1. Timestamped segments are preserved correctly.
2. Transcript text is preserved correctly.
3. Source language is preserved/detected correctly.
4. Empty transcription results are handled safely.
5. Invalid input/error propagation is handled cleanly.
6. Conversion from existing Buzz transcription results to the localization representation works correctly.

Prefer unit tests using existing Buzz test fixtures/mocks.

Do NOT require downloading large AI models merely to run basic unit tests.

Do NOT run the entire expensive test suite first.

Run targeted tests for modified/new modules first.

---

## Step 8  Backward Compatibility

Existing Buzz functionality must continue to work.

Do not unnecessarily modify:

    existing transcription UI
    existing exports
    existing CLI behavior
    existing database behavior
    existing model download behavior

Phase 1 should be additive and isolated where practical.

---

## Explicitly Out of Scope

Do NOT implement any of the following during Phase 1:

    Vietnamese translation
    Vietnamese TTS
    voice cloning
    lip synchronization
    audio replacement
    background music separation
    subtitle burn-in
    final video rendering
    automatic upload
    YouTube/TikTok publishing

Those belong to later phases.

---

## Expected Phase 1 Deliverable

At the end of Phase 1 we should be able to conceptually perform:

    input.mp4
        |
        v
    existing Buzz transcription system
        |
        v
    source language transcript
        |
        v
    timestamped structured data

Example:

    {
      "source_language": "en",
      "segments": [
        {
          "start": 0.0,
          "end": 4.1,
          "text": "Welcome to this video."
        }
      ]
    }

This structured transcript becomes the input to Phase 2.

---

## Implementation Procedure

When this plan is later approved for implementation:

1. Read AGENTS.md.
2. Read PLAN.md completely.
3. Inspect relevant Buzz implementation.
4. Report the proposed files/modules that need modification.
5. Prefer reuse over rewriting.
6. Implement the smallest viable Phase 1 architecture.
7. Add targeted tests.
8. Run targeted tests.
9. Inspect git status.
10. Inspect git diff.
11. Do not push.
12. Do not commit unless explicitly requested.
13. Produce the report format required by AGENTS.md.

---

## Phase 1 Success Criteria

Phase 1 is complete only when:

- A video/audio input can use existing Buzz transcription infrastructure.
- English speech can produce timestamped source segments.
- Chinese speech can produce timestamped source segments.
- Source language information is available.
- Results can be represented in a stable machine-readable structure.
- The implementation does not duplicate the speech recognition engine.
- Existing Buzz behavior remains intact.
- Targeted tests pass.
- No unrelated files are modified.
