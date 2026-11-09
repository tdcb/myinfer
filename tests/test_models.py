"""Model management: local discovery, deletion safety, upgrades, fit estimates, benchmark store."""

import json

import pytest

from m5engine import models as M
from m5engine.context import GB

QWEN3_CFG = {
    "model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
    "quantization": {"group_size": 64, "bits": 4},
}


def _model_dir(root, name, cfg=QWEN3_CFG, meta=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"\0" * 1024)
    if meta:
        (d / M.META_FILE).write_text(json.dumps(meta))
    return d


def test_list_local_models_reads_source_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: (_ for _ in ()).throw(RuntimeError()))
    _model_dir(tmp_path, "Qwen3-0.6B-4bit", meta={"repo_id": "mlx-community/Qwen3-0.6B-4bit", "revision": "abc"})
    (tmp_path / "not-a-model").mkdir()
    [m] = M.list_local_models(tmp_path)
    assert m["repo_id"] == "mlx-community/Qwen3-0.6B-4bit"
    assert m["revision"] == "abc"
    assert m["bits"] == 4 and not m["moe"] and m["source"] == "models_dir"


def test_delete_refuses_paths_outside_models_dir(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    outside = _model_dir(tmp_path, "elsewhere")
    with pytest.raises(PermissionError):
        M.delete_local_model(models, {"source": "models_dir", "path": str(outside), "id": str(outside)})
    assert outside.exists()
    inside = _model_dir(models, "m")
    assert M.delete_local_model(models, {"source": "models_dir", "path": str(inside), "id": str(inside)}) >= 0
    assert not inside.exists()


def test_upgrade_removes_files_dropped_upstream(tmp_path):
    d = _model_dir(tmp_path, "m", meta={"repo_id": "x/m"})
    (d / "model-00002-of-00002.safetensors").write_bytes(b"old")
    (d / "README.md").write_text("not managed")
    M._remove_stale(d, {"config.json", "model.safetensors"})
    assert sorted(p.name for p in d.iterdir()) == [M.META_FILE, "README.md", "config.json", "model.safetensors"]


def test_estimate_fit():
    small = M.estimate_fit(QWEN3_CFG, int(0.3 * GB), 48 * GB)
    assert small["fit"] == "fits in memory"
    assert small["est_max_context"] == small["native_context"] * 4  # Qwen3 YaRN 4x cap
    dense_big = M.estimate_fit(QWEN3_CFG, 60 * GB, 48 * GB)
    assert dense_big["fit"] == "too large" and dense_big["est_max_context"] is None
    moe_big = M.estimate_fit({**QWEN3_CFG, "num_experts": 128}, 60 * GB, 48 * GB)
    assert moe_big["fit"] == "streams experts from SSD" and moe_big["moe"]
    tight = M.estimate_fit(QWEN3_CFG, 40 * GB, 48 * GB)
    assert 0 < tight["est_max_context"] < small["est_max_context"] and tight["est_max_context"] % 256 == 0


def test_check_update(monkeypatch):
    class Info:
        sha, last_modified = "new", None

    monkeypatch.setattr(M, "_api", lambda: type("A", (), {"model_info": lambda self, r: Info()})())
    assert M.check_update({"id": "a", "repo_id": "x/a", "revision": "old"})["status"] == "update_available"
    assert M.check_update({"id": "a", "repo_id": "x/a", "revision": "new"})["status"] == "up_to_date"
    assert M.check_update({"id": "a", "repo_id": None})["status"] == "unknown"


def test_benchmark_store_keeps_last_20(tmp_path):
    store = M.BenchmarkStore(tmp_path)
    for i in range(25):
        store.add("m", {"decode_tps": i})
    runs = store.get("m")
    assert len(runs) == 20 and runs[-1]["decode_tps"] == 24
    store.forget("m")
    assert store.get("m") == []
