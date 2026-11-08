"""Local model discovery, Hugging Face downloads and recommendations."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List

GB = 1024**3

ALLOW_PATTERNS = [
    "*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken",
    "tiktoken.model", "*.txt", "*.jsonl", "*.jinja",
]

# Sizes are the safetensors totals on the Hub (GB, 1e9 bytes), used until a live lookup succeeds.
RECOMMENDED = [
    {"repo_id": "mlx-community/Qwen3-32B-4bit", "kind": "dense", "params_b": 32.8, "active_b": 32.8, "size_gb": 18.4},
    {"repo_id": "mlx-community/gemma-3-27b-it-4bit", "kind": "dense", "params_b": 27.4, "active_b": 27.4, "size_gb": 16.8},
    {"repo_id": "mlx-community/Qwen3-14B-4bit", "kind": "dense", "params_b": 14.8, "active_b": 14.8, "size_gb": 8.3},
    {"repo_id": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "kind": "moe", "params_b": 30.5, "active_b": 3.3, "size_gb": 17.2},
    {"repo_id": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "kind": "moe", "params_b": 30.5, "active_b": 3.3, "size_gb": 17.2},
    {"repo_id": "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit", "kind": "moe", "params_b": 80, "active_b": 3, "size_gb": 44.8},
    {"repo_id": "mlx-community/GLM-4.5-Air-4bit", "kind": "moe", "params_b": 106, "active_b": 12, "size_gb": 60.1},
    {"repo_id": "mlx-community/gpt-oss-120b-MXFP4-Q8", "kind": "moe", "params_b": 117, "active_b": 5.1, "size_gb": 63.4},
    {"repo_id": "mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit", "kind": "moe", "params_b": 235, "active_b": 22, "size_gb": 102.9},
    {"repo_id": "mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit", "kind": "moe", "params_b": 235, "active_b": 22, "size_gb": 132.2},
    {"repo_id": "mlx-community/Qwen3-0.6B-4bit", "kind": "draft", "params_b": 0.6, "active_b": 0.6, "size_gb": 0.3},
    {"repo_id": "mlx-community/Qwen3-1.7B-4bit", "kind": "draft", "params_b": 1.7, "active_b": 1.7, "size_gb": 1.0},
]


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _is_model_dir(p: Path) -> bool:
    return (p / "config.json").exists() and any(p.glob("model*.safetensors"))


def list_local_models(models_dir: Path) -> List[dict]:
    found: Dict[str, dict] = {}
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and _is_model_dir(p):
                found[str(p)] = _describe(p, str(p))
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model":
                continue
            for rev in repo.revisions:
                if _is_model_dir(Path(rev.snapshot_path)):
                    found.setdefault(repo.repo_id, _describe(Path(rev.snapshot_path), repo.repo_id))
                    break
    except Exception:
        pass
    return list(found.values())


def _describe(p: Path, model_id: str) -> dict:
    try:
        cfg = json.loads((p / "config.json").read_text())
    except Exception:
        cfg = {}
    size = sum(f.stat().st_size for f in p.glob("model*.safetensors"))
    moe = any(k in cfg for k in ("num_experts", "n_routed_experts", "num_local_experts"))
    return {
        "id": model_id,
        "path": str(p),
        "model_type": cfg.get("model_type"),
        "moe": moe,
        "size_gb": round(size / GB, 2),
        "quantization": cfg.get("quantization"),
    }


class Downloader:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, repo_id: str, on_done=None) -> dict:
        with self._lock:
            job = self.jobs.get(repo_id)
            if job and job["state"] == "downloading":
                return job
            local = self.models_dir / repo_id.split("/")[-1]
            job = {"repo_id": repo_id, "path": str(local), "state": "downloading", "error": None,
                   "downloaded_gb": 0.0, "total_gb": None, "started": time.time()}
            self.jobs[repo_id] = job
        threading.Thread(target=self._run, args=(job, on_done), daemon=True).start()
        return job

    def _run(self, job: dict, on_done) -> None:
        from huggingface_hub import HfApi, snapshot_download

        local = Path(job["path"])
        try:
            info = HfApi().model_info(job["repo_id"], files_metadata=True)
            job["total_gb"] = round(sum((s.size or 0) for s in info.siblings) / GB, 2)
        except Exception:
            pass
        stop = threading.Event()

        def progress():
            while not stop.wait(1.0):
                if local.exists():
                    job["downloaded_gb"] = round(_dir_bytes(local) / GB, 2)

        threading.Thread(target=progress, daemon=True).start()
        try:
            local.mkdir(parents=True, exist_ok=True)
            snapshot_download(job["repo_id"], local_dir=str(local), allow_patterns=ALLOW_PATTERNS)
            job["downloaded_gb"] = round(_dir_bytes(local) / GB, 2)
            job["state"] = "done"
            if on_done:
                on_done(job)
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)
        finally:
            stop.set()

    def status(self) -> List[dict]:
        return list(self.jobs.values())


def recommendations(budget_bytes: int, total_ram: int) -> List[dict]:
    out = []
    budget = budget_bytes / 1e9
    for r in RECOMMENDED:
        size = r["size_gb"]
        if r["kind"] == "draft":
            fit = "draft model for speculative decoding"
        elif size + 4 < budget:
            fit = "fits in memory"
        elif r["kind"] == "moe":
            fit = "streams experts from SSD"
        else:
            fit = "too large"
        out.append({**r, "fit": fit, "headroom_gb": round(budget - size, 1)})
    return out
