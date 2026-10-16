# Chay tu thu muc goc cua repository Auto Video.
# Kiem tra thiet lap, KHONG in API key hoac gia tri bi mat.
param([switch]$RequireVieNeu)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$needed = @("PyQt6", "edge_tts", "funasr", "modelscope", "keyring", "google.genai")
$missing = New-Object System.Collections.Generic.List[string]

Write-Host "=== Auto Video Windows portable preflight ==="
Write-Host "Repo: $repo"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "[MISSING] Python environment: $python"
    $missing.Add("Virtualenv: run setup_windows.bat")
} else {
    & $python --version
    if ($LASTEXITCODE -ne 0) { $missing.Add("Virtualenv Python failed") }
    & $python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) { $missing.Add("Need Python 3.12.x") }
    foreach ($module in $needed) {
        & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$module') is not None else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Python module $module"
        } else {
            Write-Host "[MISSING] Python module $module"
            $missing.Add("Python module $module")
        }
    }
}

foreach ($tool in @("ffmpeg.exe", "ffprobe.exe")) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Write-Host "[MISSING] $tool is not on PATH"
        $missing.Add($tool)
    } else {
        Write-Host "[OK] $tool"
    }
}

Write-Host ""
Write-Host "--- VieNeu isolated runtime (needed for Ngoc Huyen voice) ---"
if (-not $env:BUZZ_VIENEU_PYTHON) {
    $candidate = Join-Path $repo ".vieneu-venv\Scripts\python.exe"
    if (Test-Path $candidate) { $env:BUZZ_VIENEU_PYTHON = $candidate }
}
if (-not $env:BUZZ_VIENEU_MODEL_DIR) {
    $candidate = Join-Path $repo "models\VieNeu-TTS-v3-Turbo\onnx_update"
    if (Test-Path $candidate) { $env:BUZZ_VIENEU_MODEL_DIR = $candidate }
}
if (-not $env:BUZZ_VIENEU_HF_HOME) {
    $candidate = Join-Path $repo "models\HuggingFace"
    if (Test-Path $candidate) { $env:BUZZ_VIENEU_HF_HOME = $candidate }
}
$viePy = $env:BUZZ_VIENEU_PYTHON
$vieModel = $env:BUZZ_VIENEU_MODEL_DIR
$vieHome = $env:BUZZ_VIENEU_HF_HOME

if ([string]::IsNullOrWhiteSpace($viePy) -or -not (Test-Path -LiteralPath $viePy -PathType Leaf)) {
    Write-Host "[MISSING] BUZZ_VIENEU_PYTHON points to an existing Python executable"
    if ($RequireVieNeu) { $missing.Add("VieNeu Python executable") }
} else {
    & $viePy -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vieneu') is not None else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] VieNeu Python and module"
    } else {
        Write-Host "[MISSING] VieNeu Python does not contain module 'vieneu'"
        if ($RequireVieNeu) { $missing.Add("VieNeu module") }
    }
}
if ([string]::IsNullOrWhiteSpace($vieModel) -or -not (Test-Path -LiteralPath $vieModel -PathType Container)) {
    Write-Host "[MISSING] BUZZ_VIENEU_MODEL_DIR"
    if ($RequireVieNeu) { $missing.Add("VieNeu ONNX model") }
} else {
    Write-Host "[OK] VieNeu ONNX model directory"
}
if ([string]::IsNullOrWhiteSpace($vieHome) -or -not (Test-Path -LiteralPath $vieHome -PathType Container)) {
    Write-Host "[MISSING] BUZZ_VIENEU_HF_HOME"
    if ($RequireVieNeu) { $missing.Add("VieNeu HuggingFace cache") }
} else {
    Write-Host "[OK] VieNeu HuggingFace cache directory"
}

Write-Host ""
if ($missing.Count -gt 0) {
    Write-Host "[INCOMPLETE] Missing components:"
    foreach ($item in $missing) { Write-Host " - $item" }
    Write-Host "See PORTABLE_WINDOWS.md. No secrets were printed."
    exit 2
}

Write-Host "[OK] Mandatory GUI runtime present. VieNeu is optional unless -RequireVieNeu."
Write-Host "This is NOT an end-to-end media quality or clean-Windows installation test."
exit 0
