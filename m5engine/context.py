"""KV-cache sizing and maximum-context planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as mlx_cache
from mlx_lm.models.rope_utils import initialize_rope

from .attention import safe_maybe_quantize_kv_cache

GB = 1024**3

# Families whose model cards document static YaRN extension (factor up to 4).
YARN_FAMILIES = ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe")
MAX_YARN_FACTOR = 4


@dataclass
class KVFootprint:
    bytes_per_token: float
    fixed_bytes: int
    kv_bits: int

    def bytes_for(self, tokens: int) -> int:
        return int(self.fixed_bytes + self.bytes_per_token * tokens)


def _walk(cache: List[Any]):
    for c in cache:
        sub = getattr(c, "caches", None)
        if sub is not None and not isinstance(c, (mlx_cache.KVCache, mlx_cache.QuantizedKVCache)):
            yield from _walk(list(sub))
        else:
            yield c


def _allocated_len(c) -> int:
    keys = getattr(c, "keys", None)
    if keys is None:
        return 0
    if isinstance(keys, (tuple, list)):
        return keys[0].shape[-2]
    return keys.shape[-2]


def measure_kv_footprint(model: nn.Module, kv_bits: int, group_size: int = 64) -> KVFootprint:
    """Run one token through the model and measure what each cache costs per token."""
    cache = mlx_cache.make_prompt_cache(model)
    model(mx.array([[0]]), cache=cache)
    if kv_bits:
        safe_maybe_quantize_kv_cache(cache, 0, group_size, kv_bits)
    mx.eval([c.state for c in _walk(cache)])
    per_tok = 0.0
    fixed = 0
    for c in _walk(cache):
        try:
            nbytes = c.nbytes
        except (NotImplementedError, AttributeError):
            continue
        alloc = _allocated_len(c)
        if isinstance(c, mlx_cache.RotatingKVCache):
            fixed += int(nbytes / max(alloc, 1) * c.max_size) if alloc else 0
        elif isinstance(c, (mlx_cache.KVCache, mlx_cache.QuantizedKVCache)) and alloc:
            per_tok += nbytes / alloc
        else:
            fixed += nbytes
    del cache
    mx.clear_cache()
    return KVFootprint(per_tok, fixed, kv_bits)


def native_context(config: dict) -> Tuple[int, int]:
    """(native positional limit, current limit given the config's rope scaling)."""
    mpe = int(config.get("max_position_embeddings") or config.get("seq_length") or 4096)
    rs = config.get("rope_scaling") or {}
    rope_type = rs.get("type") or rs.get("rope_type")
    if rope_type in ("yarn", "linear", "dynamic") and rs.get("original_max_position_embeddings"):
        orig = int(rs["original_max_position_embeddings"])
        return orig, max(mpe, int(orig * float(rs.get("factor", 1))))
    native = int(config.get("original_max_position_embeddings") or mpe)
    if config.get("model_type", "").startswith("qwen3") and mpe == 40960:
        native = 32768  # Qwen3 ships 32K + 8K generation headroom
    return native, mpe


def yarn_capable(config: dict) -> bool:
    return config.get("model_type") in YARN_FAMILIES and not config.get("rope_scaling")


def apply_yarn(model: nn.Module, factor: float, native: int) -> int:
    """Swap every default RoPE module for YaRN in place. Returns modules replaced."""
    scaling = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": native,
    }
    replaced = 0
    for _, module in model.named_modules():
        rope = module.get("rope")
        if rope is None:
            continue
        base_rope = getattr(module, "_m5_base_rope", rope)
        if not isinstance(base_rope, nn.RoPE):
            continue
        object.__setattr__(module, "_m5_base_rope", base_rope)
        if factor <= 1:
            module.rope = base_rope
        else:
            module.rope = initialize_rope(
                base_rope.dims,
                base_rope.base,
                base_rope.traditional,
                scaling_config=scaling,
                max_position_embeddings=int(native * factor),
            )
        replaced += 1
    return replaced


def yarn_factor_for(tokens: int, native: int, max_factor: float) -> float:
    """Smallest power-of-two YaRN factor that covers `tokens` (1.0 = plain RoPE)."""
    f = 1.0
    while native * f < tokens and f < max_factor:
        f *= 2
    return min(f, max_factor)


def combine(a: KVFootprint, b: Optional[KVFootprint]) -> KVFootprint:
    if b is None:
        return a
    return KVFootprint(a.bytes_per_token + b.bytes_per_token, a.fixed_bytes + b.fixed_bytes, a.kv_bits)


@dataclass
class ContextPlan:
    budget_bytes: int
    resident_weight_bytes: int
    expert_cache_bytes: int
    reserve_bytes: int
    kv_budget_bytes: int
    kv_bits: int
    kv_bytes_per_token: float
    native_context: int
    model_context_limit: int
    memory_context_limit: int
    yarn_factor: float
    max_context: int
    limited_by: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in list(d):
            if k.endswith("_bytes"):
                d[k.replace("_bytes", "_gb")] = round(d[k] / GB, 2)
        return d


def reserve_bytes(budget: int, hidden_size: int, prefill_step: int) -> int:
    activations = prefill_step * max(hidden_size, 1) * 2 * 24
    return int(max(1.5 * GB, 0.04 * budget) + activations)


def plan_context(
    *,
    config: dict,
    footprint: KVFootprint,
    budget_bytes: int,
    resident_weight_bytes: int,
    expert_cache_bytes: int = 0,
    prefill_step: int = 2048,
    rope_scaling: str = "auto",
    requested_context: Optional[int] = None,
) -> ContextPlan:
    native, current_limit = native_context(config)
    reserve = reserve_bytes(budget_bytes, int(config.get("hidden_size") or 0), prefill_step)
    kv_budget = budget_bytes - resident_weight_bytes - expert_cache_bytes - reserve
    if footprint.bytes_per_token > 0:
        mem_limit = int(max(0, kv_budget - footprint.fixed_bytes) // footprint.bytes_per_token)
    else:
        mem_limit = 1 << 30
    mem_limit = mem_limit // 256 * 256

    yarn_factor = 1.0
    model_limit = current_limit
    target = requested_context or mem_limit
    if rope_scaling != "off" and yarn_capable(config) and target > current_limit:
        if rope_scaling == "auto":
            f = yarn_factor_for(target, native, MAX_YARN_FACTOR)
        else:
            f = min(float(rope_scaling), MAX_YARN_FACTOR)
        yarn_factor = float(f)
        model_limit = int(native * yarn_factor)

    max_ctx = min(model_limit, mem_limit)
    if requested_context:
        max_ctx = min(max_ctx, requested_context)
    limited_by = "memory" if mem_limit <= model_limit else "model"
    if requested_context and requested_context <= min(model_limit, mem_limit):
        limited_by = "setting"
    return ContextPlan(
        budget_bytes=budget_bytes,
        resident_weight_bytes=resident_weight_bytes,
        expert_cache_bytes=expert_cache_bytes,
        reserve_bytes=reserve,
        kv_budget_bytes=max(0, kv_budget),
        kv_bits=footprint.kv_bits,
        kv_bytes_per_token=footprint.bytes_per_token,
        native_context=native,
        model_context_limit=model_limit,
        memory_context_limit=mem_limit,
        yarn_factor=yarn_factor,
        max_context=max(0, max_ctx),
        limited_by=limited_by,
    )
