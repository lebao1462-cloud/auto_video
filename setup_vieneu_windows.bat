@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "BASE_PY=%ROOT%.venv\Scripts\python.exe"
set "VIENEU_PY=%ROOT%.vieneu-venv\Scripts\python.exe"
set "MODEL_DIR=%ROOT%models\VieNeu-TTS-v3-Turbo\onnx_update"
set "HF_HOME=%ROOT%models\HuggingFace"

echo.
echo ==== Auto Video: optional VieNeu v3 Turbo CPU voice installer ====
if not exist "%BASE_PY%" (
    echo [ERROR] Run setup_windows.bat first.
    exit /b 2
)
if not exist "%VIENEU_PY%" (
    echo [INFO] Creating isolated VieNeu virtual environment...
    "%BASE_PY%" -m venv "%ROOT%.vieneu-venv"
    if errorlevel 1 goto :failed
)
echo [INFO] Installing VieNeu 3.8.3 and its dependencies...
"%VIENEU_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VIENEU_PY%" -m pip install "vieneu==3.8.3" "onnxruntime>=1.20,<2"
if errorlevel 1 goto :failed
"%VIENEU_PY%" -c "from vieneu import Vieneu; print('VieNeu Python runtime OK')"
if errorlevel 1 goto :failed

if /I "%~1"=="--no-models" goto :ready
echo [INFO] Downloading model assets. This may need 1GB+ of disk and internet.
echo [INFO] Models are provided by their respective authors under their licenses.
set "HF_HOME=%HF_HOME%"
"%VIENEU_PY%" "%ROOT%install_vieneu_models.py"
if errorlevel 1 goto :failed

:ready
echo.
echo [DONE] Optional VieNeu setup finished.
echo        start_auto_video.bat will auto-detect the local model and runtime.
echo        Example voice in app: Ngoc Huyen (if supported by installed assets).
exit /b 0

:failed
echo [ERROR] VieNeu setup was not completed.
echo Check internet, Python 3.12 x64, free disk space, and upstream model permissions.
exit /b 3
