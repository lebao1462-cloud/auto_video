# Auto Video Localization - Windows Guide

Auto Video converts Chinese speech into Vietnamese dubbing and subtitles.

## What the workflow produces

For one source video, the application can:

1. Transcribe Chinese speech with Paraformer-zh.
2. Translate with Argos offline by default, or use Gemini to correct Chinese recognition in context and translate batches of up to 30 timestamped segments.
3. Generate Vietnamese speech with Edge TTS.
4. Synchronize Vietnamese speech to the source timeline.
5. Optionally separate original vocals with Demucs and preserve background/music.
6. Mix a full-length Vietnamese audio track.
7. Generate Vietnamese SRT/VTT subtitles.
8. Render the final MP4 with soft subtitles, burned-in subtitles, or no subtitle track.
9. For videos longer than 15 minutes, split into sequential parts and join one final MP4 automatically.

## Windows first run

Recommended paths on this development PC:

    Project:          D:\codex\auto_video\buzz-1.4.4
    Model root:       D:\Dev\buzz-models
    UV cache:         D:\Dev\uv-cache
    Temporary files: D:\Dev\Temp
    Default output:  D:\codex\auto_video_output

Set the persistent model root if needed:

    setx BUZZ_MODEL_ROOT D:\Dev\buzz-models

For a PowerShell development session:

    $env:BUZZ_MODEL_ROOT='D:\Dev\buzz-models'
    $env:UV_CACHE_DIR='D:\Dev\uv-cache'
    $env:TEMP='D:\Dev\Temp'
    $env:TMP='D:\Dev\Temp'
    $env:PATH='D:\Dev\FFmpeg\ffmpeg-9.0.2-full_build\bin;' + $env:PATH

## Localization runtime dependencies

The localization workflow uses Paraformer-zh/FunASR for Chinese ASR and Edge TTS for Vietnamese speech.

Install the pinned localization runtime requirements into the project's Python environment:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

Large ASR models are not stored in Git. Paraformer models are downloaded to the configured ModelScope cache on first use and can run offline afterward. Provider modules are lazy-loaded where practical; localization preflight reports missing runtime components.

## Translation providers

The default translation provider is Argos Translate:

- offline after language packages are installed;
- free;
- no API key;
- English uses en -> vi;
- Chinese uses zh -> en -> vi on the current Argos package index because a direct zh -> vi package is not available.

The Argos runtime is isolated in a worker process. This avoids native-library conflicts with the localization runtime in the main Windows process.

### Install Argos language packages

On this development PC, packages are stored on drive D:

    setx ARGOS_PACKAGES_DIR D:\ArgosTranslate\packages

For the current PowerShell session:

    $env:ARGOS_PACKAGES_DIR='D:\ArgosTranslate\packages'

Install/refresh the required language packages explicitly:

    .venv\Scripts\python.exe install_argos_packages.py

This download is never triggered automatically by a localization run.

### Gemini API - optional online provider

Gemini is optional and is never used as an automatic fallback from Argos.

Configure the GUI with:

- Gemini API key
- Gemini model (default: gemini-2.5-flash)

Gemini requires network access and may be subject to provider quota/rate limits. Each request includes segment IDs, timestamps, and five previous segments for context. The response must preserve every ID; invalid batches fail instead of silently misaligning dubbing. Chinese transcript text is corrected before Vietnamese translation. English transcript text is preserved.

### OpenAI-compatible - advanced compatibility provider

The existing OpenAI-compatible provider remains available for custom endpoints.

Configure:

- Translation API base URL
- Translation API key
- Translation model

Environment variables can provide defaults:

    BUZZ_TRANSLATION_API_BASE_URL
    BUZZ_TRANSLATION_API_KEY
    BUZZ_TRANSLATION_MODEL

No translation API key is required when Argos is selected. Provider secrets are never written into localization diagnostics.

### Offline translation quality note

Argos prioritizes zero-cost/offline operation. English-to-Vietnamese output is generally more usable than the current Chinese pivot route. The Chinese zh -> en -> vi pivot can produce noticeably weaker wording on short or ambiguous phrases. Select Gemini explicitly when higher translation quality is more important than fully offline operation.

## Using the GUI

Open Auto Video and choose:

    File -> Localize Video to Vietnamese...

Then:

1. Select the input video.
2. Select the output folder.
3. Choose Chinese or auto-detect (which resolves to Chinese).
4. Paraformer-zh is used automatically for Chinese transcription. Its models are cached under `D:\Dev\modelscope-cache` when D: is available.
5. Choose a translation provider. Argos requires no API key; Gemini/OpenAI show provider-specific fields.
6. Choose a Vietnamese voice.
7. Choose Soft subtitle, Burn subtitle into video, or No subtitle.
8. Optionally enable Demucs background preservation.
9. Click Start Localization. Long videos are divided and joined automatically; no manual part selection is required.

Before work begins, preflight verifies the source/model, FFmpeg/ffprobe, Edge TTS, Demucs when requested, translation configuration, output write access, and estimated free disk space.

A non-secret diagnostic report is written to:

    <output folder>\localization_diagnostics.json

## Output files

Typical output:

    <name>.vi.mp4
    <name>.vi.srt

Intermediate TTS/audio assets are kept in:

    <name>_localization_work

The workflow API includes a cleanup helper that removes only this workspace and preserves final MP4/subtitle outputs.

## Cancellation

Cancel requests are checked between pipeline stages. A provider, model inference, Demucs, or FFmpeg operation already running may need to finish its current operation before cancellation takes effect.

## Disk space

Preflight estimates a conservative workspace requirement of the larger of:

- 512 MiB
- 6x compressed source video size

Long videos, high-resolution material, Demucs stems, and PCM audio may require more space.

## Packaging

The project contains Windows PyInstaller/Inno Setup packaging infrastructure.

The modified build specification (AutoVideo.spec):

- continues to bundle FFmpeg/ffprobe through the existing build mechanism;
- bundles Edge TTS, Argos Translate, and Google GenAI provider modules when installed in the build environment;
- keeps localization providers optional so normal application startup does not fail solely because a provider runtime is absent;
- includes the Argos worker entrypoint used to isolate native translation runtime from the main application process.

Before packaging, install the localization runtime requirement:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

Then use the current Windows build/package process.

Note: on September 26, 2026, regenerating the original uv.lock on this machine was blocked by DNS access to pypi.ngc.nvidia.com and the offline cache did not contain all cross-platform bitsandbytes packages. For that reason this fork does not silently modify uv.lock; the small localization provider requirement is kept in localization-requirements.txt.

## Troubleshooting

### Edge TTS missing

Install:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

### FFmpeg or ffprobe missing

The packaged application should include them. For development, put the existing application internal directory containing ffmpeg.exe and ffprobe.exe on PATH.

### Argos route missing

Set ARGOS_PACKAGES_DIR to the intended package location and run:

    .venv\Scripts\python.exe install_argos_packages.py

Preflight reports the exact missing English or Chinese route. Localization itself never downloads packages.

### Gemini translation fails

Check the Gemini API key, model name, internet connection, quota and rate limits.

### OpenAI-compatible translation fails

Check the base URL, API key, model name, internet connection, provider quota/rate limits, and compatibility with the OpenAI chat-completions interface.

### Demucs is slow

Disable background separation for faster runs. Without Demucs, the localized output is speech-only unless a prepared background track is supplied through the lower-level API.

### Not enough disk space

Choose an output drive with more free space and remove obsolete localization workspaces.

## Validation checkpoint

At Phase 8 completion:

    230 passed, 3 skipped, 0 failed

The three skipped tests are unchanged tests from the original codebase:
- two Unix-path cases skipped on Windows;
- test_transcribe_stop, explicitly skipped upstream.

Phase 9 adds additional preflight/reliability tests on top of this checkpoint.


## Release validation result

Final Windows release validation on September 26, 2026:

    Phase 9 targeted tests: 11 / 11 PASS
    Final regression: 238 passed, 3 skipped, 0 failed
    PyInstaller build: PASS
    Packaged application smoke launch: PASS

Validated bundle:

    D:\AutoVideoBuild\dist\AutoVideo\AutoVideo.exe

The bundle contains:

    ffmpeg.exe
    ffprobe.exe
    edge_tts runtime

The local development environment originally had onnxruntime 1.30.0 while this Buzz source declares onnxruntime==1.18.1. Packaging became stable after restoring the declared 1.18.1 version. Keep the build environment aligned with pyproject.toml when reproducing the package.

Generated packaging output under D:\AutoVideoBuild is intentionally not committed to GitHub.

### Manual acceptance after release

For final subjective quality validation, use a representative real English or Chinese video with your chosen translation provider credentials and check:

- translation wording;
- Vietnamese voice preference;
- speech speed/naturalness;
- background preservation;
- subtitle readability;
- final perceived audio/video sync.

These quality checks depend on external provider/model choice and should not require exposing API credentials in diagnostics or source control.
