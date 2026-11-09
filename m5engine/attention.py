"""Memory-bounded attention over a quantized KV cache.

mlx-lm's quantized SDPA materialises the full [heads, L, S] score matrix, which
at long context (e.g. 2048 x 128K x 64 heads) needs tens of GB during prefill.
This version walks the keys in blocks with an online softmax so the transient
memory is bounded by `score_budget_bytes`, which is what makes 8/4-bit KV
caches usable at maximum context.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models import base as mlx_base
from mlx_lm.models import cache as mlx_cache

_original_quantized_sdpa = mlx_base.quantized_scaled_dot_product_attention

score_budget_bytes = 512 * 1024**2
MIN_BLOCK = 256


def _slice_keys(q, start, end):
    return tree_map(lambda x: x[..., start:end, :], q)


def blocked_quantized_sdpa(
    queries: mx.array,
    q_keys,
    q_values,
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    S = q_keys[0].shape[-2]
    per_key = B * n_q_heads * L * 4 * 3
    block = max(MIN_BLOCK, score_budget_bytes // max(per_key, 1))
    if S <= block or L == 1:
        return _original_quantized_sdpa(
            queries, q_keys, q_values, scale, mask, group_size=group_size, bits=bits
        )

    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    q = queries * scale
    if n_repeats > 1:
        q = mx.reshape(q, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    causal = isinstance(mask, str)
    if mask is not None and not causal and n_repeats > 1 and mask.ndim >= 3:
        mask = mx.expand_dims(mask, axis=-3)
    q_pos = mx.arange(S - L, S)[:, None] if causal else None

    neg = mx.array(-1e30, dtype=mx.float32)
    m = None
    denom = None
    acc = None
    for start in range(0, S, block):
        end = min(S, start + block)
        scores = mx.quantized_matmul(
            q, *_slice_keys(q_keys, start, end), transpose=True,
            group_size=group_size, bits=bits,
        ).astype(mx.float32)
        valid = None
        if causal:
            if end > S - L:
                valid = q_pos >= mx.arange(start, end)[None]
        elif mask is not None:
            blk = mask[..., start:end]
            if blk.dtype == mx.bool_:
                valid = blk
            else:
                scores = scores + blk.astype(mx.float32)
        if valid is not None:
            scores = mx.where(valid, scores, neg)
        blk_max = mx.max(scores, axis=-1, keepdims=True)
        m_new = blk_max if m is None else mx.maximum(m, blk_max)
        p = mx.exp(scores - m_new)
        if valid is not None:
            p = mx.where(valid, p, 0.0)
        pv = mx.quantized_matmul(
            p.astype(q.dtype), *_slice_keys(q_values, start, end), transpose=False,
            group_size=group_size, bits=bits,
        ).astype(mx.float32)
        if m is None:
            denom = mx.sum(p, axis=-1, keepdims=True)
            acc = pv
        else:
            corr = mx.exp(m - m_new)
            denom = denom * corr + mx.sum(p, axis=-1, keepdims=True)
            acc = acc * corr + pv
        m = m_new

    out = (acc / mx.maximum(denom, 1e-30)).astype(queries.dtype)
    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out


def safe_maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    """Quantize only plain KV caches; sliding-window and recurrent caches stay as-is."""
    if kv_bits is None:
        return
    for i, c in enumerate(prompt_cache):
        if type(c) is mlx_cache.KVCache and c.offset >= quantized_kv_start:
            prompt_cache[i] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    import mlx_lm.generate as mlx_generate

    mlx_base.quantized_scaled_dot_product_attention = blocked_quantized_sdpa
    mlx_generate.maybe_quantize_kv_cache = safe_maybe_quantize_kv_cache
    _installed = True
