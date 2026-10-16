@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
set "APP_DIR=%ROOT%buzz-1.4.4"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Chua cai Python environment.
    echo Chay setup_windows.bat truoc khi mo Auto Video.
    exit /b 2
)
if not exist "%APP_DIR%\main.py" (
    echo [ERROR] Khong tim thay buzz-1.4.4\main.py
    exit /b 3
)
where ffmpeg.exe >nul 2>&1
if errorlevel 1 echo [WARNING] ffmpeg.exe not found on PATH.
where ffprobe.exe >nul 2>&1
if errorlevel 1 echo [WARNING] ffprobe.exe not found on PATH.

if not defined BUZZ_VIENEU_PYTHON if exist "%ROOT%.vieneu-venv\Scripts\python.exe" set "BUZZ_VIENEU_PYTHON=%ROOT%.vieneu-venv\Scripts\python.exe"
if not defined BUZZ_VIENEU_MODEL_DIR if exist "%ROOT%models\VieNeu-TTS-v3-Turbo\onnx_update\config.json" set "BUZZ_VIENEU_MODEL_DIR=%ROOT%models\VieNeu-TTS-v3-Turbo\onnx_update"
if not defined BUZZ_VIENEU_HF_HOME if exist "%ROOT%models\HuggingFace\hub" set "BUZZ_VIENEU_HF_HOME=%ROOT%models\HuggingFace"

pushd "%APP_DIR%"
"%PYTHON_EXE%" main.py
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
    echo [ERROR] Auto Video exited with code %RESULT%.
)
exit /b %RESULT%
