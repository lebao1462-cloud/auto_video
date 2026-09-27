# Auto Video Localization - Windows Guide

This fork extends Buzz 1.4.4 with an English/Chinese video -> Vietnamese localized MP4 workflow.

## What the workflow produces

For one source video, the application can:

1. Transcribe English or Chinese speech with Buzz/Whisper.
2. Translate each segment into Vietnamese with Argos offline by default, or Gemini/OpenAI-compatible providers when selected.
3. Generate Vietnamese speech with Edge TTS.
4. Synchronize Vietnamese speech to the source timeline.
5. Optionally separate original vocals with Demucs and preserve background/music.
6. Mix a full-length Vietnamese audio track.
7. Generate Vietnamese SRT/VTT subtitles.
8. Render the final MP4 with soft subtitles, burned-in subtitles, or no subtitle track.

## Windows first run

Recommended paths on this development PC:

    Project:          D:\code\auto_video\buzz-1.4.4
    Model root:       D:\Buzz\Models
    UV cache:         D:\uv-cache
    Temporary files: D:\BuzzTestCache\Temp
    Default output:  D:\AutoVideoOutput

Set the persistent model root if needed:

    setx BUZZ_MODEL_ROOT D:\Buzz\Models

For a PowerShell development session:

    $env:BUZZ_MODEL_ROOT='D:\Buzz\Models'
    $env:UV_CACHE_DIR='D:\uv-cache'
    $env:TEMP='D:\BuzzTestCache\Temp'
    $env:TMP='D:\BuzzTestCache\Temp'
    $env:PATH='D:\App\Buzz\Buzz\_internal;' + $env:PATH

## Optional localization runtime provider

The localization GUI currently uses Edge TTS for Vietnamese speech.

Install the pinned localization runtime requirement into the same Python environment used by Buzz:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

The provider is intentionally lazy-loaded. Normal Buzz transcription still works when Edge TTS is absent; localization preflight will report that the localization runtime is missing.

## Translation providers

The default translation provider is Argos Translate:

- offline after language packages are installed;
- free;
- no API key;
- English uses en -> vi;
- Chinese uses zh -> en -> vi on the current Argos package index because a direct zh -> vi package is not available.

The Argos runtime is isolated in a worker process. This avoids the native-library conflict observed when Argos and the Buzz/Whisper localization stack are loaded in the same Windows process.

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

Gemini requires network access and may be subject to provider quota/rate limits.

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

Open Buzz and choose:

    File -> Localize Video to Vietnamese...

Then:

1. Select the input video.
2. Select the output folder.
3. Choose English, Chinese, or auto-detect.
4. Select a valid Whisper model.
5. Choose a translation provider. Argos requires no API key; Gemini/OpenAI show provider-specific fields.
6. Choose a Vietnamese voice.
7. Choose Soft subtitle, Burn subtitle into video, or No subtitle.
8. Optionally enable Demucs background preservation.
9. Click Start Localization.

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

Buzz already contains Windows PyInstaller/Inno Setup packaging infrastructure.

The modified Buzz.spec:

- continues to bundle FFmpeg/ffprobe through the existing Buzz mechanism;
- bundles Edge TTS, Argos Translate, and Google GenAI provider modules when installed in the build environment;
- keeps localization providers optional so normal Buzz startup does not fail solely because a provider runtime is absent;
- includes the Argos worker entrypoint used to isolate native translation runtime from the main Buzz process.

Before packaging, install the localization runtime requirement:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

Then use the existing Buzz Windows build/package process.

Note: on September 26, 2026, regenerating the upstream Buzz uv.lock on this machine was blocked by DNS access to pypi.ngc.nvidia.com and the offline cache did not contain all cross-platform bitsandbytes packages. For that reason this fork does not silently modify uv.lock; the small localization provider requirement is kept in localization-requirements.txt.

## Troubleshooting

### Edge TTS missing

Install:

    .venv\Scripts\python.exe -m pip install -r localization-requirements.txt

### FFmpeg or ffprobe missing

The packaged Buzz application should include them. For development, put the existing Buzz internal directory containing ffmpeg.exe and ffprobe.exe on PATH.

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

The three skipped tests are unchanged upstream Buzz tests:
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

    D:\AutoVideoBuild\dist\Buzz\Buzz.exe

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
