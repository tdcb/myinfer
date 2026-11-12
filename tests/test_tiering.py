"""Strata-inspired expert tiering: hot-expert profile, score-based eviction, warm fill, pack, active experts."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from m5engine.expert_stream import (
    ExpertCache, ExpertPack, ExpertProfile, enable_expert_streaming, find_expert_modules,
    locate_all_experts, repack_experts, set_active_experts,
)
from tests.test_core import _TinyMoE


def _save(model, tmp_path):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))


def _fresh(tmp_path, **kw):
    m = _TinyMoE(**kw)
    nn.quantize(m, group_size=64, bits=4)
    m.load_weights(str(tmp_path / "model.safetensors"))
    return m


@pytest.fixture
def moe(tmp_path):
    mx.random.seed(3)
    model = _TinyMoE(layers=3)
    nn.quantize(model, group_size=64, bits=4)
    _save(model, tmp_path)
    return model


def test_profile_records_decays_and_ranks():
    prof = ExpertProfile(layers=2, num_experts=4, decay=0.5)
    prof.record(0, np.array([[0, 1], [0, 2]]))  # layer 0: expert 0 twice, 1 and 2 once (per 2 tokens)
    prof.record(1, np.array([[3, 3]]))
    assert prof.steps == 1
    assert prof.scores[0, 0] == pytest.approx(1.0) and prof.scores[0, 1] == pytest.approx(0.5)
    prof.record(0, np.array([[1, 1]]))  # new step: decay then add
    assert prof.scores[0, 0] == pytest.approx(0.5) and prof.scores[0, 1] == pytest.approx(2.25)
    ranked = [tuple(r) for r in prof.ranked()]
    assert ranked[0] == (0, 1) and (1, 3) in ranked
    assert all(prof.scores[l, e] > 0 for l, e in ranked)


def test_profile_round_trip_and_shape_mismatch(tmp_path):
    prof = ExpertProfile(2, 8)
    assert prof.save(tmp_path) is None  # nothing learned yet
    prof.record(0, np.array([[1, 2]]))
    path = prof.save(tmp_path)
    assert path is not None and path.exists() and not prof.dirty
    again = ExpertProfile.load(tmp_path, 2, 8)
    assert again.steps == 1 and np.allclose(again.scores, prof.scores)
    other = ExpertProfile.load(tmp_path, 3, 8)  # different model shape: start fresh, don't crash
    assert other.steps == 0 and not other.scores.any()


def test_eviction_keeps_hot_experts_over_recent_ones(moe, tmp_path):
    streamed = _fresh(tmp_path, layers=3)
    locs, _ = locate_all_experts(tmp_path, streamed)
    per_expert = sum(t.per_expert_bytes for t in locs[0].values())
    prof = ExpertProfile(3, 8)
    prof.scores[0, 5] = 100.0  # expert 5 of layer 0 is known hot
    cache = ExpertCache(capacity_bytes=per_expert * 3, profile=prof)
    cache.fetch(0, [5], locs[0])
    cache.fetch(0, [1, 2], locs[0])
    cache.fetch(0, [3], locs[0])  # over capacity: a cold expert goes, not the hot one
    assert (0, 5) in cache and (0, 3) in cache
    assert cache.stats.evictions >= 1
    assert cache.used_bytes <= cache.capacity_bytes
    cache.close()


def test_set_capacity_shrinks_and_grows(moe, tmp_path):
    streamed = _fresh(tmp_path, layers=3)
    locs, _ = locate_all_experts(tmp_path, streamed)
    per_expert = sum(t.per_expert_bytes for t in locs[0].values())
    cache = ExpertCache(capacity_bytes=per_expert * 8)
    cache.fetch(0, list(range(8)), locs[0])
    assert len(cache) == 8
    cache.set_capacity(per_expert * 2)
    assert len(cache) == 2 and cache.used_bytes <= per_expert * 2
    cache.set_capacity(per_expert * 8)
    cache.fetch(1, list(range(4)), locs[1])
    assert len(cache) == 6
    cache.close()


def test_profile_learned_during_streaming_warms_next_load(moe, tmp_path):
    x = mx.random.normal((1, 6, 64)).astype(mx.bfloat16)
    ref = moe(x)
    streamed = _fresh(tmp_path, layers=3)
    info = enable_expert_streaming(streamed, tmp_path, cache_bytes=10**9)
    assert mx.array_equal(streamed(x), ref)
    prof = info.cache.profile
    assert prof is not None and prof.steps == 1 and prof.observed > 0
    used = {(l, e) for l, e in prof.ranked()}
    assert prof.save(tmp_path) is not None
    info.cache.close()

    again = _fresh(tmp_path, layers=3)
    info2 = enable_expert_streaming(again, tmp_path, cache_bytes=10**9)
    assert info2.warm_scheduled == len(used)
    info2.cache.wait_warm()
    assert info2.cache.stats.warmed == len(used)
    assert all(k in info2.cache for k in used)
    # A warm cache serves the same prompt from memory with identical results.
    assert mx.array_equal(again(x), ref)
    assert info2.cache.stats.misses == 0
    info2.cache.close()


def test_warm_fill_respects_capacity(moe, tmp_path):
    streamed = _fresh(tmp_path, layers=3)
    locs, _ = locate_all_experts(tmp_path, streamed)
    per_expert = sum(t.per_expert_bytes for t in locs[0].values())
    prof = ExpertProfile(3, 8)
    prof.scores[:] = 1.0
    prof.steps = 1
    cache = ExpertCache(capacity_bytes=per_expert * 5, profile=prof)
    n = cache.warm(locs)
    assert n == 5
    cache.wait_warm()
    assert len(cache) == 5 and cache.used_bytes <= cache.capacity_bytes
    cache.close()


def test_repacked_experts_match_safetensors(moe, tmp_path):
    x = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
    ref = moe(x)
    lazy = _fresh(tmp_path, layers=3)
    seen = []
    pack_path = repack_experts(lazy, tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert pack_path.exists() and seen and seen[-1][0] == seen[-1][1]
    locs, counts = locate_all_experts(tmp_path, lazy)
    pack = ExpertPack.open(tmp_path, locs)
    assert pack is not None and counts == [8, 8, 8]
    cache = ExpertCache(10**9, pack=pack)
    direct = ExpertCache(10**9)
    a = cache.fetch(1, [2, 7], locs[1])
    b = direct.fetch(1, [2, 7], locs[1])
    for ea, eb in zip(a, b):
        assert ea.keys() == eb.keys()
        for rel in ea:
            assert mx.array_equal(ea[rel], eb[rel])
    cache.close(), direct.close()

    streamed = _fresh(tmp_path, layers=3)
    info = enable_expert_streaming(streamed, tmp_path, cache_bytes=10**9)
    assert info.cache.pack is not None and info.to_dict()["packed"]
    assert mx.array_equal(streamed(x), ref)
    info.cache.close()

    # A pack built for other tensors (here: a 4-layer model) is rejected by its fingerprint.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _TinyMoE(layers=4)
    nn.quantize(other, group_size=64, bits=4)
    _save(other, other_dir)
    locs4, _ = locate_all_experts(other_dir, _fresh(other_dir, layers=4))
    assert ExpertPack.open(other_dir, locs4) is None
    (other_dir / pack_path.parent.name).mkdir()
    (other_dir / pack_path.parent.name / pack_path.name).write_bytes(pack_path.read_bytes())
    assert ExpertPack.open(other_dir, locs4) is None


def test_set_active_experts_changes_routing(moe):
    default, patched = set_active_experts(moe, 1)
    assert (default, patched) == (2, 3)
    assert all(l.num_experts_per_tok == 1 for l in moe.layers)
    with pytest.raises(ValueError):
        set_active_experts(moe, 3)
    with pytest.raises(ValueError):
        set_active_experts(moe, 0)
    assert find_expert_modules(moe)


def test_streaming_uses_overridden_top_k(moe, tmp_path):
    set_active_experts(moe, 1)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    ref = moe(x)
    streamed = _fresh(tmp_path, layers=3)
    set_active_experts(streamed, 1)
    info = enable_expert_streaming(streamed, tmp_path, cache_bytes=10**9)
    assert mx.array_equal(streamed(x), ref)
    assert info.cache.stats.hits + info.cache.stats.misses + info.cache.stats.prefetch_used <= 3 * 3
    info.cache.close()
