# -*- mode: python ; coding: utf-8 -*-
import os
import os.path
import platform
import shutil
import importlib.util

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

from buzz.__version__ import VERSION

def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        print(f"Optional package metadata not found, skipping: {package_name}")
        return []

datas = []
localization_binaries = []
localization_hiddenimports = []
for optional_package in ("edge_tts", "argostranslate", "funasr"):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(
            optional_package
        )
        datas += package_datas
        localization_binaries += package_binaries
        localization_hiddenimports += package_hiddenimports
    except Exception:
        # Localization preflight reports missing optional provider runtimes.
        print(f"Optional localization package not bundled: {optional_package}")

# google-genai ships a large tests/local-tokenizer tree. collect_all() would pull
# that entire tree into Buzz and can force sentencepiece into PyInstaller's
# isolated dependency scanner even though text generation does not need it.
# The runtime provider only needs these modules; their regular imports pull
# required dependencies transitively.
try:
    import google.genai  # noqa: F401
    localization_hiddenimports += [
        "google.genai",
        "google.genai.types",
    ]
    datas += safe_copy_metadata("google-genai")
except Exception:
    print("Optional localization package not bundled: google-genai")

# Argos depends on sentencepiece. On this Windows development machine,
# PyInstaller's isolated dependency scanner crashes while importing
# sentencepiece (0xC0000005), although sentencepiece works at runtime.
# Exclude it from graph scanning and copy its pure-Python package files plus
# native extension manually so the frozen Argos worker can import it normally.
sentencepiece_spec = importlib.util.find_spec("sentencepiece")
if sentencepiece_spec and sentencepiece_spec.submodule_search_locations:
    sentencepiece_dir = os.path.abspath(
        next(iter(sentencepiece_spec.submodule_search_locations))
    )
    for filename in (
        "__init__.py",
        "__init__.pyi",
        "_version.py",
        "sentencepiece_model_pb2.py",
        "sentencepiece_pb2.py",
        "py.typed",
    ):
        source = os.path.join(sentencepiece_dir, filename)
        if os.path.isfile(source):
            datas.append((source, "sentencepiece"))
    for filename in os.listdir(sentencepiece_dir):
        if filename.startswith("_sentencepiece") and filename.endswith(".pyd"):
            localization_binaries.append(
                (os.path.join(sentencepiece_dir, filename), "sentencepiece")
            )
    datas += safe_copy_metadata("sentencepiece")
else:
    print("Optional sentencepiece runtime not found; Argos may be unavailable.")

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
datas += [("LICENSE", ".")]
datas += [("HELP.html", ".")]

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
    binaries=binaries + localization_binaries,
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
        *localization_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["sentencepiece"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PyQt6 bundles an older MSVCP140.dll under Qt6/bin. On Windows that copy can
# be selected by the frozen Argos subprocess and crash sentencepiece/Argos
# with 0xC0000005. Keep the newer runtime already collected at _internal root.
if platform.system() == "Windows":
    from PyInstaller.building.datastruct import TOC

    a.binaries = TOC(
        entry
        for entry in a.binaries
        if entry[0].replace("\\", "/").lower()
        != "pyqt6/qt6/bin/msvcp140.dll"
    )

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    options,
    icon="./assets/auto-video.ico",
    exclude_binaries=True,
    name="AutoVideo",
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
    name="AutoVideo",
)
app = BUNDLE(
    coll,
    name="AutoVideo.app",
    icon="./assets/auto-video.icns",
    bundle_identifier="com.lebao1462.autovideo",
    version=VERSION,
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": "True",
        "NSMicrophoneUsageDescription": "Allow Auto Video to record audio from your microphone.",
    },
)
