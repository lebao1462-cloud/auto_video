"""Install/verify separately licensed VieNeu ONNX model assets for portable Auto Video.

Requires a VieNeu virtualenv with huggingface_hub. No API key is stored.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def paths(args):
    root = Path(__file__).resolve().parent
    model_root = Path(args.model_dir or root / "models" / "VieNeu-TTS-v3-Turbo")
    hf_root = Path(args.hf_home or root / "models" / "HuggingFace")
    return model_root, hf_root


REQUIRED_ONNX = (
    "config.json",
    "tokenizer.json",
    "vieneu_acoustic_cached.onnx",
    "vieneu_backbone_shared.data",
    "vieneu_decode_step.onnx",
    "vieneu_prefill.onnx",
    "vieneu_v3_heads.npz",
)
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
VOICE_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"


def validate(model_root, hf_root):
    onnx = model_root / "onnx_update"
    missing = [name for name in REQUIRED_ONNX
               if not (onnx / name).is_file() or (onnx / name).stat().st_size < 128]
    repo_dir = hf_root / "hub" / "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano-ONNX"
    snapshots = list((repo_dir / "snapshots").glob("*"))
    codec_ok = any((path / "moss_audio_tokenizer_decode_full.onnx").exists()
                   for path in snapshots)
    print("VieNeu ONNX:", "OK" if not missing else "MISSING " + ", ".join(missing))
    print("MOSS tokenizer cache:", "OK" if codec_ok else "MISSING")
    print("Model root:", model_root)
    print("Hugging Face home:", hf_root)
    return not missing and codec_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate only; never download")
    parser.add_argument("--model-dir", type=Path, help="Directory containing onnx_update")
    parser.add_argument("--hf-home", type=Path, help="Hugging Face cache home")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model_root = args.model_dir.parent if args.model_dir else root / "models" / "VieNeu-TTS-v3-Turbo"
    hf_root = args.hf_home or root / "models" / "HuggingFace"
    if args.check:
        return 0 if validate(model_root, hf_root) else 2
    model_root.mkdir(parents=True, exist_ok=True)
    hf_root.mkdir(parents=True, exist_ok=True)
    # Set before importing huggingface_hub so the correct cache is used.
    os.environ["HF_HOME"] = str(hf_root)
    from huggingface_hub import snapshot_download
    print("Downloading official VieNeu ONNX assets (licensed upstream)...", flush=True)
    snapshot_download(
        repo_id=VOICE_REPO,
        local_dir=str(model_root),
        allow_patterns=["onnx_update/*"],
        max_workers=3,
    )
    print("Downloading MOSS audio codec files...", flush=True)
    snapshot_download(repo_id=CODEC_REPO, max_workers=3)
    if not validate(model_root, hf_root):
        return 2
    print("VieNeu assets ready. Use setup_vieneu_windows.bat and start_auto_video.bat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
