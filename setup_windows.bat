@echo off
setlocal EnableExtensions
cd /d "%~dp0buzz-1.4.4"
title Auto Video - Windows Setup

set "CHECK_ONLY=0"
if /I "%~1"=="--check" set "CHECK_ONLY=1"
set "VENV_PY=..\.venv\Scripts\python.exe"
set "PYTHON_CMD="

echo.
echo ========================================
echo   Auto Video - Windows Setup
echo ========================================

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 goto :python_ready
)

py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    goto :python_ready
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_ready
)

if "%CHECK_ONLY%"=="1" goto :python_missing

where uv.exe >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Python 3.12 not found. Installing it with uv...
    uv python install 3.12
    if not errorlevel 1 (
        uv venv --python 3.12 "..\.venv"
        if not errorlevel 1 goto :python_ready
    )
)

where winget.exe >nul 2>&1
if errorlevel 1 goto :python_missing
echo [INFO] Python 3.12 not found. Installing it with winget...
winget install --id Python.Python.3.12 --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :python_missing
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto :python_missing
set "PYTHON_CMD=py -3.12"

:python_ready
if exist "%VENV_PY%" (
    for /f "delims=" %%V in ('%VENV_PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
) else (
    for /f "delims=" %%V in ('%PYTHON_CMD% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
)
echo [OK] Python %PYVER%

if exist D:\ (
    set "MODELSCOPE_CACHE=D:\Dev\modelscope-cache"
    set "BUZZ_MODEL_ROOT=D:\Dev\buzz-models"
    set "TMP=D:\Temp"
    set "TEMP=D:\Temp"
    if "%CHECK_ONLY%"=="0" (
        if not exist "D:\Dev\modelscope-cache" mkdir "D:\Dev\modelscope-cache"
        if not exist "D:\Dev\buzz-models" mkdir "D:\Dev\buzz-models"
        if not exist "D:\Temp" mkdir "D:\Temp"
    )
    echo [OK] Models/cache/temp will use D:\
) else (
    echo [INFO] D: not found. Default user cache locations will be used.
)

where ffmpeg.exe >nul 2>&1
set "FFMPEG_OK=%ERRORLEVEL%"
where ffprobe.exe >nul 2>&1
set "FFPROBE_OK=%ERRORLEVEL%"
if "%FFMPEG_OK%"=="0" if "%FFPROBE_OK%"=="0" goto :ffmpeg_ok
if "%CHECK_ONLY%"=="1" goto :ffmpeg_missing
where winget.exe >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
echo [INFO] FFmpeg not found. Installing with winget...
winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :ffmpeg_missing
echo [INFO] FFmpeg installed. If PATH is not refreshed yet, reopen Windows Terminal once.
goto :after_ffmpeg

:ffmpeg_ok
echo [OK] FFmpeg and FFprobe found.
:after_ffmpeg

if "%CHECK_ONLY%"=="1" goto :check_environment
if not exist "%VENV_PY%" (
    echo [INFO] Creating Python virtual environment...
    %PYTHON_CMD% -m venv "..\.venv"
    if errorlevel 1 goto :setup_failed
)

echo [INFO] Updating pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :setup_failed

echo [INFO] Installing Auto Video and runtime dependencies...
"%VENV_PY%" -m pip install -e .
if errorlevel 1 goto :setup_failed

echo [INFO] Installing localization runtime dependencies...
"%VENV_PY%" -m pip install -r localization-requirements.txt
if errorlevel 1 goto :setup_failed

echo [INFO] Verifying runtime imports...
"%VENV_PY%" -c "import edge_tts, funasr, modelscope, PyQt6; from google import genai; print('Runtime imports OK')"
if errorlevel 1 goto :runtime_missing

echo.
echo [DONE] Auto Video setup completed successfully.
echo Run with: ..\.venv\Scripts\python.exe main.py
echo Enter your Gemini API key in the app the first time.
exit /b 0

:check_environment
if not exist "%VENV_PY%" goto :venv_missing
"%VENV_PY%" -c "import edge_tts, funasr, modelscope, PyQt6; from google import genai; print('Runtime imports OK')"
if errorlevel 1 goto :runtime_missing
echo [OK] Existing virtual environment is ready.
echo [DONE] Check completed successfully.
exit /b 0

:python_missing
echo [ERROR] Python 3.12.x was not found. Install 64-bit Python 3.12 and rerun setup_windows.bat.
exit /b 2

:ffmpeg_missing
echo [ERROR] FFmpeg/FFprobe are required and were not found.
echo Install FFmpeg, ensure ffmpeg.exe and ffprobe.exe are in PATH, then rerun setup_windows.bat.
exit /b 3

:venv_missing
echo [ERROR] ..\.venv does not exist yet. Run setup_windows.bat without --check first.
exit /b 4

:runtime_missing
echo [ERROR] One or more Auto Video runtime packages are missing or broken.
echo Run setup_windows.bat without --check to repair the environment.
exit /b 5

:setup_failed
echo [ERROR] Auto Video setup failed. Review the error above and rerun setup_windows.bat.
exit /b 6
