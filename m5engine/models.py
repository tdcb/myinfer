"""Local model discovery, Hugging Face downloads and recommendations."""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import MAX_YARN_FACTOR, kv_estimate_per_token, native_context, reserve_bytes, yarn_capable

log = logging.getLogger("m5engine.models")

GB = 1024**3
META_FILE = ".m5engine.json"
BENCH_FILE = "benchmarks.json"

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
    return (p / "config.json").exists() and (
        any(p.glob("model*.safetensors")) or (p / "model.safetensors.index.json").exists()
    )


def _wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in ALLOW_PATTERNS)


def _read_meta(p: Path) -> dict:
    """Source repo/revision of a models-dir download (ours, or huggingface_hub's own metadata)."""
    try:
        return json.loads((p / META_FILE).read_text())
    except Exception:
        pass
    hf_meta = p / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        return {"revision": hf_meta.read_text().splitlines()[0].strip()}
    except Exception:
        return {}


def _write_meta(p: Path, meta: dict) -> None:
    (p / META_FILE).write_text(json.dumps(meta, indent=2))


def disk_free_gb(models_dir: Path) -> float:
    target = models_dir if models_dir.exists() else models_dir.resolve().parent
    while not target.exists():
        target = target.parent
    return round(shutil.disk_usage(target).free / GB, 1)


def list_local_models(models_dir: Path) -> List[dict]:
    found: Dict[str, dict] = {}
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and _is_model_dir(p):
                meta = _read_meta(p)
                found[str(p)] = _describe(p, str(p), "models_dir", meta.get("repo_id"), meta.get("revision"),
                                          meta.get("downloaded_at") or p.stat().st_mtime)
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model":
                continue
            revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
            for rev in revs:
                if _is_model_dir(Path(rev.snapshot_path)):
                    d = _describe(Path(rev.snapshot_path), repo.repo_id, "hf_cache", repo.repo_id,
                                  rev.commit_hash, rev.last_modified)
                    d["disk_gb"] = round(repo.size_on_disk / GB, 2)
                    found.setdefault(repo.repo_id, d)
                    break
    except Exception:
        pass
    return list(found.values())


def _describe(p: Path, model_id: str, source: str = "models_dir", repo_id: Optional[str] = None,
              revision: Optional[str] = None, modified: Optional[float] = None) -> dict:
    try:
        cfg = json.loads((p / "config.json").read_text())
    except Exception:
        cfg = {}
    size = sum(f.stat().st_size for f in p.glob("model*.safetensors"))
    tc = cfg.get("text_config", cfg)
    moe = any(k in tc for k in ("num_experts", "n_routed_experts", "num_local_experts"))
    native, _ = native_context(cfg) if cfg else (0, 0)
    q = cfg.get("quantization") or {}
    return {
        "incomplete": _missing_shards(p),
        "id": model_id,
        "name": (repo_id or p.name).split("/")[-1],
        "path": str(p),
        "source": source,
        "repo_id": repo_id,
        "revision": revision,
        "modified": modified,
        "model_type": cfg.get("model_type"),
        "moe": moe,
        "size_gb": round(size / GB, 2),
        "disk_gb": round(_dir_bytes(p) / GB, 2) if source == "models_dir" else round(size / GB, 2),
        "quantization": q or None,
        "bits": q.get("bits"),
        "native_context": native,
    }


def _missing_shards(p: Path) -> bool:
    try:
        index = json.loads((p / "model.safetensors.index.json").read_text())
    except Exception:
        return False
    return any(not (p / f).exists() for f in set(index.get("weight_map", {}).values()))


def find_local(models_dir: Path, model_id: str) -> dict:
    for m in list_local_models(models_dir):
        if model_id in (m["id"], m["path"]):
            return m
    raise KeyError(model_id)


def delete_local_model(models_dir: Path, entry: dict) -> float:
    """Remove a downloaded model from disk; returns GB freed."""
    if entry["source"] == "hf_cache":
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == entry["repo_id"] and repo.repo_type == "model":
                strategy = cache.delete_revisions(*[r.commit_hash for r in repo.revisions])
                freed = strategy.expected_freed_size
                strategy.execute()
                return round(freed / GB, 2)
        raise KeyError(entry["id"])
    path = Path(entry["path"]).resolve()
    root = models_dir.resolve()
    if root not in path.parents:
        raise PermissionError(f"{path} is outside the models directory {root}")
    freed = _dir_bytes(path)
    shutil.rmtree(path)
    return round(freed / GB, 2)


# ------------------------------------------------------------------- Hub
_hub_cache: Dict[str, Tuple[float, dict]] = {}
HUB_TTL = 600.0


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def search_hub(query: str, limit: int = 25) -> List[dict]:
    models = _api().list_models(
        search=query or None, filter="mlx", pipeline_tag="text-generation", sort="downloads",
        limit=limit, expand=["downloads", "likes", "lastModified"],
    )
    return [{"repo_id": m.id, "downloads": m.downloads, "likes": m.likes,
             "last_modified": m.last_modified.isoformat() if m.last_modified else None} for m in models]


def hub_info(repo_id: str, budget_bytes: int, kv_bits: int = 8) -> dict:
    """Size, revision and a pre-download fit/context estimate for a Hub repo."""
    now = time.time()
    hit = _hub_cache.get(repo_id)
    if hit and now - hit[0] < HUB_TTL:
        base = hit[1]
    else:
        from huggingface_hub import hf_hub_download

        info = _api().model_info(repo_id, files_metadata=True)
        files = [s for s in info.siblings if _wanted(s.rfilename)]
        cfg = json.loads(Path(hf_hub_download(repo_id, "config.json", revision=info.sha)).read_text())
        weights = sum((s.size or 0) for s in files if s.rfilename.endswith(".safetensors"))
        base = {
            "repo_id": repo_id,
            "sha": info.sha,
            "last_modified": info.last_modified.isoformat() if info.last_modified else None,
            "downloads": info.downloads,
            "download_gb": round(sum((s.size or 0) for s in files) / GB, 2),
            "weights_bytes": weights,
            "config": cfg,
        }
        _hub_cache[repo_id] = (now, base)
    out = {k: v for k, v in base.items() if k not in ("config", "weights_bytes")}
    out.update(estimate_fit(base["config"], base["weights_bytes"], budget_bytes, kv_bits))
    return out


def _expert_layout(tc: dict, bits: Optional[int]) -> Optional[dict]:
    """Routed-expert sizes from a MoE config: bytes per expert, per layer and per token."""
    n_exp = tc.get("n_routed_experts") or tc.get("num_experts") or tc.get("num_local_experts")
    if not n_exp:
        return None
    layers = int(tc.get("num_hidden_layers") or 0)
    moe_layers = layers - int(tc.get("first_k_dense_replace") or 0)
    if moe_layers <= 0:
        return None
    inter = int(tc.get("moe_intermediate_size") or tc.get("intermediate_size") or 0)
    hidden = int(tc.get("hidden_size") or 0)
    k = int(tc.get("num_experts_per_tok") or tc.get("n_activated_experts") or 0)
    # MLX quantization keeps a 16-bit scale and bias per 64-weight group: +0.5 bit/weight.
    bytes_per_weight = ((bits or 16) + (0.5 if bits else 0)) / 8
    expert_bytes = int(3 * hidden * inter * bytes_per_weight)
    return {"experts": int(n_exp), "moe_layers": moe_layers, "active": k, "expert_bytes": expert_bytes,
            "total_bytes": expert_bytes * int(n_exp) * moe_layers,
            "per_token_bytes": expert_bytes * k * moe_layers}


def estimate_fit(cfg: dict, weight_bytes: int, budget_bytes: int, kv_bits: int = 8) -> dict:
    """Rough pre-download estimate; the real plan is computed after loading."""
    tc = cfg.get("text_config", cfg)
    moe = any(k in tc for k in ("num_experts", "n_routed_experts", "num_local_experts"))
    native, plain = native_context(cfg)
    q = cfg.get("quantization") or {}
    hidden = int(tc.get("hidden_size") or 4096)
    reserve = reserve_bytes(budget_bytes, hidden, 2048)
    free = budget_bytes - weight_bytes - reserve
    per_tok = kv_estimate_per_token(cfg, kv_bits)
    model_limit = int(native * MAX_YARN_FACTOR) if yarn_capable(cfg) else plain

    def ctx(kv_bytes: int) -> int:
        mem = int(kv_bytes // per_tok) // 256 * 256 if per_tok else plain
        return max(0, min(mem, model_limit))

    out = {"moe": moe, "bits": q.get("bits"), "model_type": cfg.get("model_type"),
           "size_gb": round(weight_bytes / GB, 2), "native_context": native}
    if free > 0:
        return {**out, "fit": "fits in memory", "est_max_context": ctx(free)}
    layout = _expert_layout(tc, q.get("bits")) if moe else None
    if layout is None:
        return {**out, "fit": "too large", "est_max_context": None}
    # Streamed: non-expert weights stay resident; what is left splits between the expert cache and KV.
    expert_total = min(layout["total_bytes"], weight_bytes)
    resident = weight_bytes - expert_total
    left = budget_bytes - resident - reserve
    if left <= 0:
        return {**out, "fit": "too large", "est_max_context": None,
                "est_resident_gb": round(resident / GB, 1), "est_expert_gb": round(expert_total / GB, 1),
                "note": "Even without its experts the model does not fit in the memory budget."}
    cache = left // 2
    return {
        **out, "fit": "streams experts from SSD", "est_max_context": ctx(left - cache),
        "est_resident_gb": round(resident / GB, 1), "est_expert_gb": round(expert_total / GB, 1),
        "est_expert_cache_gb": round(cache / GB, 1),
        "est_experts_in_memory_pct": round(100 * min(1.0, cache / max(expert_total, 1)), 1),
        "expert_mb": round(layout["expert_bytes"] / 1024**2, 1),
        "active_experts": layout["active"],
        # Every token's experts read from the SSD (a cold cache); the hit rate lowers it.
        "est_ssd_mb_per_token_cold": round(layout["per_token_bytes"] / 1024**2),
    }


def check_update(entry: dict) -> dict:
    if not entry.get("repo_id"):
        return {"id": entry["id"], "status": "unknown", "reason": "Source repo not recorded for this model"}
    info = _api().model_info(entry["repo_id"])
    local = entry.get("revision")
    if not local:
        status = "unknown"
    else:
        status = "up_to_date" if info.sha == local else "update_available"
    return {"id": entry["id"], "repo_id": entry["repo_id"], "status": status, "local_revision": local,
            "remote_revision": info.sha,
            "remote_modified": info.last_modified.isoformat() if info.last_modified else None}


# -------------------------------------------------------------- downloads
_WORKER = (
    "import json, sys\n"
    "from huggingface_hub import snapshot_download\n"
    "repo, local, pats = sys.argv[1], sys.argv[2] or None, json.loads(sys.argv[3])\n"
    "print(snapshot_download(repo, local_dir=local, allow_patterns=pats))\n"
)


def _hub_cache_dir(repo_id: str) -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


class Downloader:
    """Downloads run in a subprocess so they can be cancelled; partial files resume next time."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.jobs: Dict[str, dict] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self, repo_id: str, on_done=None, local_dir: Optional[str] = "auto", upgrade: bool = False) -> dict:
        with self._lock:
            job = self.jobs.get(repo_id)
            if job and job["state"] == "downloading":
                return job
            if local_dir == "auto":
                local_dir = str(self.models_dir / repo_id.split("/")[-1])
            job = {"repo_id": repo_id, "path": local_dir, "state": "downloading", "error": None,
                   "kind": "upgrade" if upgrade else "download",
                   "downloaded_gb": 0.0, "total_gb": None, "started": time.time(), "finished": None}
            self.jobs[repo_id] = job
        threading.Thread(target=self._run, args=(job, on_done), daemon=True).start()
        return job

    def cancel(self, repo_id: str) -> dict:
        with self._lock:
            job = self.jobs.get(repo_id)
            proc = self._procs.get(repo_id)
            if not job or job["state"] != "downloading":
                raise KeyError(repo_id)
            job["state"] = "cancelled"
        if proc:
            proc.terminate()
        return job

    def discard(self, repo_id: str) -> dict:
        """Forget a cancelled/failed job and delete its partial files from the models dir."""
        with self._lock:
            job = self.jobs.get(repo_id)
            if not job or job["state"] == "downloading":
                raise KeyError(repo_id)
            del self.jobs[repo_id]
        freed = 0
        if job["state"] != "done" and job["path"]:
            path = Path(job["path"]).resolve()
            if self.models_dir.resolve() in path.parents and path.exists():
                freed = _dir_bytes(path)
                shutil.rmtree(path)
        return {"repo_id": repo_id, "discarded": True, "freed_gb": round(freed / GB, 2)}

    def _run(self, job: dict, on_done) -> None:
        repo = job["repo_id"]
        local = Path(job["path"]) if job["path"] else None
        watch = local or _hub_cache_dir(repo)
        start_bytes = _dir_bytes(watch) if watch.exists() and local is None else 0
        remote_files = None
        try:
            info = _api().model_info(repo, files_metadata=True)
            files = [s for s in info.siblings if _wanted(s.rfilename)]
            remote_files = {s.rfilename for s in files}
            job["total_gb"] = round(sum((s.size or 0) for s in files) / GB, 2)
            job["revision"] = info.sha
        except Exception as e:
            log.warning("Could not read %s metadata: %s", repo, e)
        stop = threading.Event()

        def progress():
            while not stop.wait(1.0):
                if watch.exists():
                    job["downloaded_gb"] = round(max(0, _dir_bytes(watch) - start_bytes) / GB, 2)

        threading.Thread(target=progress, daemon=True).start()
        try:
            if local:
                local.mkdir(parents=True, exist_ok=True)
                if not (local / META_FILE).exists():
                    _write_meta(local, {"repo_id": repo})
            proc = subprocess.Popen(
                [sys.executable, "-c", _WORKER, repo, str(local) if local else "", json.dumps(ALLOW_PATTERNS)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            with self._lock:
                self._procs[repo] = proc
            out, err = proc.communicate()
            if job["state"] == "cancelled":
                return
            if proc.returncode != 0:
                lines = [ln for ln in err.strip().splitlines() if ln.strip()]
                raise RuntimeError(lines[-1] if lines else f"download exited with {proc.returncode}")
            snapshot = Path(out.strip().splitlines()[-1])
            if local:
                if remote_files is not None:
                    _remove_stale(local, remote_files)
                _write_meta(local, {"repo_id": repo, "revision": job.get("revision"),
                                    "downloaded_at": time.time()})
            else:
                job["path"] = str(snapshot)
                _prune_old_revisions(repo, snapshot)
            job["downloaded_gb"] = job["total_gb"] or round(_dir_bytes(snapshot) / GB, 2)
            job["state"] = "done"
            if on_done:
                on_done(job)
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)
        finally:
            job["finished"] = time.time()
            stop.set()
            with self._lock:
                self._procs.pop(repo, None)

    def status(self) -> List[dict]:
        return list(self.jobs.values())


def _remove_stale(local: Path, remote_files: set) -> None:
    """After an upgrade, drop files the new revision no longer has (e.g. re-sharded weights)."""
    for f in local.rglob("*"):
        rel = f.relative_to(local).as_posix()
        if f.is_file() and not rel.startswith((".cache/", ".m5engine/")) and rel != META_FILE and _wanted(rel) \
                and rel not in remote_files:
            f.unlink()


def _prune_old_revisions(repo_id: str, keep: Path) -> None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            old = [r.commit_hash for r in repo.revisions if Path(r.snapshot_path) != keep]
            if old:
                cache.delete_revisions(*old).execute()


# --------------------------------------------------------------- benchmarks
class BenchmarkStore:
    def __init__(self, models_dir: Path):
        self.path = models_dir / BENCH_FILE
        self._lock = threading.Lock()

    def _read(self) -> Dict[str, List[dict]]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def add(self, model_id: str, result: dict) -> None:
        with self._lock:
            data = self._read()
            data.setdefault(model_id, []).append(result)
            data[model_id] = data[model_id][-20:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1))

    def get(self, model_id: str) -> List[dict]:
        return self._read().get(model_id, [])

    def all(self) -> Dict[str, List[dict]]:
        return self._read()

    def forget(self, model_id: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(model_id, None) is not None:
                self.path.write_text(json.dumps(data, indent=1))


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
