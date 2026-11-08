"""Fast unit tests that don't need downloaded models."""


import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU

from m5engine import attention as A
from m5engine.context import GB, KVFootprint, native_context, plan_context, yarn_factor_for
from m5engine.expert_stream import enable_expert_streaming
from m5engine.prefix_cache import PrefixCache, common_prefix_len


def _ref_attention(q, kd, vd, scale, mask):
    s = (q * scale) @ kd.swapaxes(-1, -2)
    if mask is not None:
        s = mx.where(mask, s, -1e30)
    return mx.softmax(s, -1) @ vd


@pytest.mark.parametrize("hq,hk", [(4, 4), (8, 2)])
@pytest.mark.parametrize("mask_kind", ["causal", "bool", None])
def test_blocked_quantized_attention_matches_reference(monkeypatch, hq, hk, mask_kind):
    mx.random.seed(0)
    L, S, D = 6, 700, 64
    q = mx.random.normal((1, hq, L, D))
    k = mx.random.normal((1, hk, S, D))
    v = mx.random.normal((1, hk, S, D))
    qk, qv = mx.quantize(k, 64, 8), mx.quantize(v, 64, 8)
    kd = mx.repeat(mx.dequantize(*qk, 64, 8), hq // hk, axis=1)
    vd = mx.repeat(mx.dequantize(*qv, 64, 8), hq // hk, axis=1)
    causal = mx.arange(S - L, S)[:, None] >= mx.arange(S)[None]
    mask = {"causal": "causal", "bool": causal, None: None}[mask_kind]
    monkeypatch.setattr(A, "score_budget_bytes", 1)
    monkeypatch.setattr(A, "MIN_BLOCK", 128)
    out = A.blocked_quantized_sdpa(q, qk, qv, 0.125, mask, 64, 8)
    ref = _ref_attention(q, kd, vd, 0.125, None if mask_kind is None else causal)
    assert mx.max(mx.abs(out - ref)).item() < 1e-4


def test_native_context_qwen3():
    assert native_context({"model_type": "qwen3", "max_position_embeddings": 40960}) == (32768, 40960)
    cfg = {"max_position_embeddings": 131072,
           "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
    assert native_context(cfg) == (32768, 131072)


def test_yarn_factor_for():
    assert yarn_factor_for(1000, 32768, 4) == 1
    assert yarn_factor_for(40000, 32768, 4) == 2
    assert yarn_factor_for(100000, 32768, 4) == 4
    assert yarn_factor_for(10**6, 32768, 4) == 4


def test_plan_context_memory_and_model_limits():
    cfg = {"model_type": "qwen3_moe", "max_position_embeddings": 40960, "hidden_size": 2048}
    fp = KVFootprint(bytes_per_token=52 * 1024, fixed_bytes=0, kv_bits=8)
    plan = plan_context(config=cfg, footprint=fp, budget_bytes=48 * GB, resident_weight_bytes=17 * GB)
    assert plan.yarn_factor == 4 and plan.max_context == 131072 and plan.limited_by == "model"
    tight = plan_context(config=cfg, footprint=fp, budget_bytes=20 * GB, resident_weight_bytes=17 * GB)
    assert tight.limited_by == "memory" and tight.max_context == tight.memory_context_limit
    assert tight.max_context % 256 == 0 and tight.max_context < 40960
    off = plan_context(config=cfg, footprint=fp, budget_bytes=48 * GB, resident_weight_bytes=17 * GB,
                       rope_scaling="off")
    assert off.max_context == 40960 and off.yarn_factor == 1
    capped = plan_context(config=cfg, footprint=fp, budget_bytes=48 * GB, resident_weight_bytes=17 * GB,
                          requested_context=8192)
    assert capped.max_context == 8192 and capped.limited_by == "setting"


def _kv_cache(n):
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, 2, n, 8)), mx.zeros((1, 2, n, 8)))
    return [c]


def test_prefix_cache_reuse_and_trim():
    pc = PrefixCache(max_entries=2)
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    pc.put("k", [1, 2, 3, 4, 5], _kv_cache(5))
    cache, hit = pc.take("k", [1, 2, 3, 9])
    assert hit == 3 and cache[0].offset == 3 and len(pc) == 0
    pc.put("k", [1, 2, 3, 4, 5], _kv_cache(5))
    cache, hit = pc.take("k", [1, 2, 3, 4, 5])
    assert hit == 4  # at least one token is always re-processed
    assert pc.take("other", [1, 2, 3])[0] is None


def test_prefix_cache_lru_and_bytes():
    pc = PrefixCache(max_entries=2)
    for i in range(3):
        pc.put("k", [i, 1, 2], _kv_cache(3))
    assert len(pc) == 2
    pc.evict_to(0)
    assert len(pc) == 0


class _TinyMoE(nn.Module):
    def __init__(self, dims=64, hidden=128, experts=8, k=2, layers=2):
        super().__init__()
        self.layers = [_TinyMoEBlock(dims, hidden, experts, k) for _ in range(layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class _TinyMoEBlock(nn.Module):
    def __init__(self, dims, hidden, experts, k):
        super().__init__()
        self.num_experts_per_tok = k
        self.gate = nn.Linear(dims, experts, bias=False)
        self.switch_mlp = SwitchGLU(dims, hidden, experts)

    def __call__(self, x):
        logits = self.gate(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k])
        scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("cache_experts", [1, 100])
def test_expert_streaming_matches_resident(tmp_path, quantize, cache_experts):
    mx.random.seed(1)
    model = _TinyMoE()
    if quantize:
        nn.quantize(model, group_size=64, bits=4)
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    # CPU gather_mm only supports float32; quantized weights exercise the bf16 scales path.
    x = mx.random.normal((1, 5, 64)).astype(mx.bfloat16 if quantize else mx.float32)
    ref = model(x)
    x1 = x[:, :1]
    ref1 = model(x1)

    streamed = _TinyMoE()
    if quantize:
        nn.quantize(streamed, group_size=64, bits=4)
    streamed.load_weights(str(tmp_path / "model.safetensors"))
    per_expert = 3 * 128 * 64 * (4 if not quantize else 0.6)
    info = enable_expert_streaming(streamed, tmp_path, cache_bytes=int(per_expert * cache_experts))
    assert info.layers == 2 and info.num_experts == 8
    out = streamed(x)
    out1 = streamed(x1)
    assert mx.array_equal(out, ref)
    assert mx.array_equal(out1, ref1)
    stats = info.cache.stats
    assert stats.misses + stats.prefetch_used > 0
    info.cache.close()


def test_expert_prefetch_follows_layer_order_and_stays_bounded(tmp_path):
    from mlx.utils import tree_flatten
    mx.random.seed(2)
    model = _TinyMoE(layers=4)
    nn.quantize(model, group_size=64, bits=4)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    streamed = _TinyMoE(layers=4)
    nn.quantize(streamed, group_size=64, bits=4)
    streamed.load_weights(str(tmp_path / "model.safetensors"))
    info = enable_expert_streaming(streamed, tmp_path, cache_bytes=10**9)
    assert [l.switch_mlp._layer_id for l in streamed.layers] == [0, 1, 2, 3]
    for i in range(5):
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        assert mx.array_equal(streamed(x), model(x))
    stats = info.cache.stats
    assert stats.prefetch_issued > 0 and stats.prefetch_used > 0
    assert not info.cache._inflight and not info.cache._prefetched
    info.cache.close()
