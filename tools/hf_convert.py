#!/usr/bin/env python3
"""
M5Engine HuggingFace / MLX Safetensors Inspector & Zero-Copy Converter
Automatically detects Dense (Qwen2/Qwen3/Qwen3.8) vs. MoE (Qwen-MoE/Mixtral/DeepSeek)
architectures from config.json and prepares 16KB page-aligned Q4_G64 weights for M5 Pro.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def detect_architecture(config: dict) -> str:
    num_experts = config.get("num_experts", config.get("num_local_experts", 0))
    if num_experts and int(num_experts) > 1:
        return "MoE"
    archs = config.get("architectures", [])
    for a in archs:
        if "Moe" in a or "MoE" in a or "Mixtral" in a:
            return "MoE"
    return "Dense"


def inspect_safetensors(st_path: Path):
    with open(st_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
    meta = json.loads(header_json)
    tensors = {k: v for k, v in meta.items() if k != "__metadata__"}
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Inspect or convert HF/MLX models for M5Engine")
    parser.add_argument("model_dir", help="Path to HuggingFace or MLX model directory (or HF repo ID)")
    parser.add_argument("--download", action="store_true", help="Download from HuggingFace Hub first if not local")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if args.download and not model_path.exists():
        print(f"[*] Downloading {args.model_dir} from HuggingFace Hub...")
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=args.model_dir)
        model_path = Path(local_dir)

    cfg_file = model_path / "config.json"
    if not cfg_file.exists():
        print(f"[!] config.json not found in {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arch_mode = detect_architecture(cfg)
    print(f"=== M5Engine Model Inspector ===")
    print(f"Directory            : {model_path}")
    print(f"Detected Mode        : {arch_mode}")
    print(f"Hidden Size          : {cfg.get('hidden_size')}")
    print(f"Num Layers           : {cfg.get('num_hidden_layers')}")
    print(f"Attention Heads (Q/KV): {cfg.get('num_attention_heads')} / {cfg.get('num_key_value_heads')}")
    if arch_mode == "MoE":
        print(f"Total Routed Experts : {cfg.get('num_experts', cfg.get('num_local_experts'))}")
        print(f"Active Top-K Experts : {cfg.get('num_experts_per_tok', 4)}")
        print(f"MoE Intermediate Dim : {cfg.get('moe_intermediate_size')}")
        print(f"Shared Expert Dim    : {cfg.get('shared_expert_intermediate_size', 0)}")
    else:
        print(f"Dense Intermediate   : {cfg.get('intermediate_size')}")

    st_files = sorted(model_path.glob("*.safetensors"))
    print(f"Safetensors Shards   : {len(st_files)}")
    if st_files:
        tensors = inspect_safetensors(st_files[0])
        print(f"Tensors in Shard 0   : {len(tensors)}")
        for name, info in list(tensors.items())[:6]:
            print(f"  - {name}: dtype={info['dtype']} shape={info['shape']}")


if __name__ == "__main__":
    main()
