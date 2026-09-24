# AGENTS.md

This repository uses a Planner -> Codex Executor -> Reviewer workflow.

## Before Coding

- Read `AGENTS.md`.
- Read the current task or plan completely.
- Inspect relevant existing code before making changes.
- Do not immediately rewrite large parts of the project.

## Coding

- Make the smallest reasonable change that satisfies the task.
- Preserve existing Buzz functionality unless the task explicitly requires changing it.
- Follow the existing project architecture and coding style.
- Do not introduce unnecessary dependencies.
- Never expose or commit API keys, tokens, passwords, or secrets.

## Testing

- Run appropriate tests after changes.
- Prefer targeted tests first.
- Be careful with commands that modify dependency lock files.
- Do not silently modify `uv.lock` or other dependency files unless required by the task.
- If a test cannot be run, clearly explain why.

## Git

- Inspect `git status` and `git diff` before finishing.
- Never push to GitHub automatically.
- Never use destructive Git commands such as `git reset --hard` unless explicitly instructed.
- Do not commit automatically unless the task explicitly asks for a commit.

## Reporting

At the end of every coding task, provide a structured report containing:

```text
STATUS: PASS / PARTIAL / FAIL

SUMMARY
FILES CHANGED
TESTS RUN
TEST RESULTS
GIT DIFF SUMMARY
ISSUES / RISKS
RECOMMENDED NEXT STEP
```

## Project Goal

The long-term goal of this fork is to extend Buzz into an automatic video localization pipeline.

Input:

```text
Video containing English or Chinese speech, without subtitles
```

Pipeline:

```text
Speech recognition
-> source transcript with timestamps
-> Vietnamese translation
-> Vietnamese TTS
-> timing synchronization
-> audio replacement/mixing
-> Vietnamese subtitle generation/burn-in
-> final MP4
```

Existing Buzz transcription functionality should be reused where practical instead of rebuilding speech recognition from scratch.
