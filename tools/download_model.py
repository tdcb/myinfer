#!/usr/bin/env python3
"""
M5Engine Model Downloader & Local Safetensors Setup Tool
Usage:
  1. Download a model from HuggingFace Hub (zero extra dependencies required):
     python3 tools/download_model.py --repo mlx-community/Qwen2.5-7B-Instruct-4bit --out ./models/Qwen2.5-7B-4bit

  2. Download a Qwen 27B / 32B or MoE model:
     python3 tools/download_model.py --repo mlx-community/Qwen2.5-32B-Instruct-4bit --out ./models/Qwen-32B-4bit
     python3 tools/download_model.py --repo mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit --out ./models/Qwen-MoE-4bit

  3. Generate a local test .safetensors model folder offline to test directory mounting:
     python3 tools/download_model.py --create-local-sample ./models/local-qwen-safetensors
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path


HF_API_BASE = "https://huggingface.co/api/models"
HF_RESOLVE_BASE = "https://huggingface.co"


def write_status_file(out_dir: Path, status: str, repo_id: str, current_file: str, pct: float, downloaded: int, total: int, message: str):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        status_path = out_dir / ".download_status.json"
        payload = {
            "status": status,
            "repo_id": repo_id,
            "local_dir": str(out_dir),
            "current_file": current_file,
            "progress_pct": round(pct, 1),
            "bytes_downloaded": downloaded,
            "total_bytes": total,
            "message": message,
        }
        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump(payload, sf)
    except Exception:
        pass


def download_file(url: str, dest: Path, token: str = None, repo_id: str = "", file_idx: int = 0, total_files: int = 1):
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "M5Engine/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                file_pct = (downloaded * 100.0) / total
                overall_pct = ((file_idx + (downloaded / total)) / max(1, total_files)) * 100.0
                write_status_file(
                    dest.parent,
                    "downloading",
                    repo_id,
                    dest.name,
                    overall_pct,
                    downloaded,
                    total,
                    f"Downloading {dest.name} ({downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB)",
                )
                print(
                    f"\r  -> {dest.name}: {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({file_pct:.1f}%)",
                    end="",
                    flush=True,
                )
        print()


def download_hf_repo(repo_id: str, out_dir: Path, token: str = None):
    print(f"[*] Querying HuggingFace Hub API for {repo_id}...")
    write_status_file(out_dir, "downloading", repo_id, "manifest", 5.0, 0, 0, f"Querying HuggingFace Hub for {repo_id}...")
    api_url = f"{HF_API_BASE}/{repo_id}"
    headers = {"User-Agent": "M5Engine/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        info = json.loads(resp.read().decode("utf-8"))

    siblings = info.get("siblings", [])
    wanted_files = []
    for s in siblings:
        rfilename = s.get("rfilename", "")
        if (
            rfilename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
            or rfilename.endswith(".safetensors")
        ):
            wanted_files.append(rfilename)

    if not wanted_files:
        write_status_file(out_dir, "error", repo_id, "", 0.0, 0, 0, f"No .safetensors or config.json found in {repo_id}")
        print(f"[!] No .safetensors or config.json found in {repo_id}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Found {len(wanted_files)} model files to download into {out_dir}:")
    for idx, fname in enumerate(wanted_files):
        file_url = f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{fname}"
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  -> Skipping {fname} (already exists)")
            continue
        download_file(file_url, dest, token, repo_id=repo_id, file_idx=idx, total_files=len(wanted_files))

    write_status_file(out_dir, "completed", repo_id, "done", 100.0, 1, 1, f"Downloaded {repo_id} to {out_dir}")
    print(f"\n[✓] Model downloaded to: {out_dir}")
    print(f"[*] Start M5Engine with these weights:")
    print(f"    ./m5engine serve --model {out_dir} --port 8080")


def create_local_sample_safetensors(out_dir: Path):
    """Creates a valid HuggingFace/MLX-format directory with config.json and model.safetensors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "vocab_size": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Create a real .safetensors file containing model.norm.weight (1024 FP16ones = 2048 bytes)
    norm_bytes = struct.pack("<1024H", *([0x3C00] * 1024))  # 0x3C00 = 1.0 in IEEE-754 FP16
    st_header = {
        "__metadata__": {"format": "mlx"},
        "model.norm.weight": {
            "dtype": "F16",
            "shape": [1024],
            "data_offsets": [0, len(norm_bytes)],
        },
    }
    hdr_bytes = json.dumps(st_header).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad

    with open(out_dir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(norm_bytes)

    print(f"[✓] Created local Safetensors model directory at: {out_dir}")
    print(f"    - {out_dir / 'config.json'}")
    print(f"    - {out_dir / 'model.safetensors'}")
    print(f"\n[*] Run or serve it with:")
    print(f"    ./m5engine serve --model {out_dir} --port 8080")


def main():
    parser = argparse.ArgumentParser(description="Download or set up HuggingFace / MLX models for M5Engine")
    parser.add_argument("--repo", help="HuggingFace repo ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    parser.add_argument("--out", default="./models/downloaded-model", help="Output directory path")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional HuggingFace token")
    parser.add_argument("--create-local-sample", metavar="DIR", help="Create a sample local .safetensors directory offline")
    args = parser.parse_args()

    if args.create_local_sample:
        create_local_sample_safetensors(Path(args.create_local_sample))
    elif args.repo:
        download_hf_repo(args.repo, Path(args.out), args.token)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
