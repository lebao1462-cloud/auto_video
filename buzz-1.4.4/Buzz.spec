# -*- mode: python ; coding: utf-8 -*-
import os
import os.path
import platform
import shutil

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

from buzz.__version__ import VERSION

def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        print(f"Optional package metadata not found, skipping: {package_name}")
        return []

datas = []
edge_tts_binaries = []
edge_tts_hiddenimports = []
try:
    edge_tts_datas, edge_tts_binaries, edge_tts_hiddenimports = collect_all("edge_tts")
    datas += edge_tts_datas
except Exception:
    # Localization preflight will report a missing Edge TTS runtime.
    pass
datas += collect_data_files("torch")
datas += collect_data_files("demucs")
datas += safe_copy_metadata("tqdm")
datas += safe_copy_metadata("torch")
datas += safe_copy_metadata("regex")
datas += safe_copy_metadata("requests")
datas += safe_copy_metadata("packaging")
datas += safe_copy_metadata("filelock")
datas += safe_copy_metadata("numpy")
datas += safe_copy_metadata("tokenizers")
datas += safe_copy_metadata("huggingface-hub")
datas += safe_copy_metadata("safetensors")
datas += safe_copy_metadata("pyyaml")
datas += safe_copy_metadata("julius")
datas += safe_copy_metadata("openunmix")
datas += safe_copy_metadata("lameenc")
datas += safe_copy_metadata("diffq")
datas += safe_copy_metadata("einops")
datas += safe_copy_metadata("hydra-core")
datas += safe_copy_metadata("hydra-colorlog")
datas += safe_copy_metadata("museval")
datas += safe_copy_metadata("submitit")
datas += safe_copy_metadata("treetable")
datas += safe_copy_metadata("soundfile")
datas += safe_copy_metadata("dora-search")
datas += safe_copy_metadata("lhotse")

# Allow transformers package to load __init__.py file dynamically:
# https://github.com/chidiwilliams/buzz/issues/272
datas += collect_data_files("transformers", include_py_files=True)

datas += collect_data_files("faster_whisper", include_py_files=True)
datas += collect_data_files("stable_whisper", include_py_files=True)
datas += collect_data_files("whisper")
datas += collect_data_files("demucs", include_py_files=True)
datas += collect_data_files("whisper_diarization", include_py_files=True)
datas += collect_data_files("deepmultilingualpunctuation", include_py_files=True)
datas += collect_data_files("ctc_forced_aligner", include_py_files=True, excludes=["build"])
datas += collect_data_files("nemo", include_py_files=True)
datas += collect_data_files("lightning_fabric", include_py_files=True)
datas += collect_data_files("pytorch_lightning", include_py_files=True)
datas += [("buzz/assets/*", "assets")]
datas += [("buzz/locale", "locale")]
datas += [("buzz/schema.sql", ".")]

block_cipher = None

DEBUG = os.environ.get("PYINSTALLER_DEBUG", "").lower() in ["1", "true"]
if DEBUG:
    options = [("v", None, "OPTION")]
else:
    options = []

def find_dependency(name: str) -> str:
    paths = os.environ["PATH"].split(os.pathsep)
    candidates = []
    for path in paths:
        exe_path = os.path.join(path, name)
        if os.path.isfile(exe_path):
            candidates.append(exe_path)

        # Check for chocolatery shims
        shim_path = os.path.normpath(os.path.join(path, "..", "lib", "ffmpeg", "tools", "ffmpeg", "bin", name))
        if os.path.isfile(shim_path):
            candidates.append(shim_path)

    if not candidates:
        return None

    # Pick the largest file
    return max(candidates, key=lambda f: os.path.getsize(f))

if platform.system() == "Windows":
    binaries = [
        (find_dependency("ffmpeg.exe"), "."),
        (find_dependency("ffprobe.exe"), "."),
    ]
else:
    binaries = [
        (shutil.which("ffmpeg"), "."),
        (shutil.which("ffprobe"), "."),
    ]

whisper_cpp_dir = os.path.join("buzz", "whisper_cpp")
if os.path.isdir(whisper_cpp_dir):
    binaries.append((os.path.join(whisper_cpp_dir, "*"), "buzz/whisper_cpp"))
else:
    print("Optional whisper.cpp bundle directory not found, skipping.")

if platform.system() == "Windows":
    if os.path.isdir("dll_backup"):
        datas += [("dll_backup", "dll_backup")]
        sdl2_path = os.path.join("dll_backup", "SDL2.dll")
        if os.path.isfile(sdl2_path):
            binaries.append((sdl2_path, "dll_backup"))
    try:
        datas += collect_data_files("msvc-runtime")
    except Exception:
        print("Optional msvc-runtime data not found, skipping.")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries + edge_tts_binaries,
    datas=datas,
    hiddenimports=[
        "dora", "dora.log",
        "julius", "julius.core", "julius.resample",
        "openunmix", "openunmix.filtering",
        "lameenc",
        "diffq",
        "einops",
        "hydra", "hydra.core", "hydra.core.global_hydra",
        "hydra_colorlog",
        "museval",
        "submitit",
        "treetable",
        "soundfile",
        "_soundfile_data",
        "lhotse",
        *edge_tts_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    options,
    icon="./assets/buzz.ico",
    exclude_binaries=True,
    name="Buzz",
    debug=DEBUG,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=DEBUG,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("BUZZ_CODESIGN_IDENTITY"),
    entitlements_file="entitlements.plist" if platform.system() == "Darwin" else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Buzz",
)
app = BUNDLE(
    coll,
    name="Buzz.app",
    icon="./assets/buzz.icns",
    bundle_identifier="com.chidiwilliams.buzz",
    version=VERSION,
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": "True",
        "NSMicrophoneUsageDescription": "Allow Buzz to record audio from your microphone.",
    },
)
