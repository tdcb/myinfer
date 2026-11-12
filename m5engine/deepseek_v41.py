"""DeepSeek-V4.1-Flash through oMLX's implementation (optional backend).

mlx-lm's own ``deepseek_v41`` refuses the checkpoint (its Engram n-gram layers
are not implemented), so this model type is served by `oMLX
<https://github.com/jundot/omlx>`_ (Apache-2.0) when it is installed in the
same environment (``tools/install_deepseek_backend.sh``). oMLX provides:

* the model (CED layout, CSA2 sparse attention, Engram, HyperConnection),
* **Engram tables read from the SSD** instead of memory (``engram_ssd_offload``),
* **bounded expert residency** with a decayed-use cache and positional reads
  from the checkpoint's own safetensors (``moe_expert_offload_resident_fraction``),
* the DeepSeek chat encoder (the tokenizer ships no chat template).

m5engine keeps the control plane: memory planning (how many experts fit next
to the KV cache), the OpenAI server, the UI, benchmarks and metrics. The
kernels are Metal (``mx.fast.metal_kernel``), so this backend only runs on a
Mac; on other machines it is import-checked only.
"""

from __future__ import annotations

import importlib
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import mlx.core as mx

log = logging.getLogger(__name__)

MODEL_TYPE = "deepseek_v41"
OMLX_REPO = "https://github.com/jundot/omlx"
OMLX_PIN = "c74f1b18558842e6b10fcfa6e2e901bb382129d4"  # 2026-11-13; tools/install_deepseek_backend.sh
INSTALL_HINT = (
    "DeepSeek V4.1 needs the oMLX backend (Python 3.11+, macOS): run "
    "tools/install_deepseek_backend.sh in this checkout, then reload."
)


def is_deepseek_v41(config: dict) -> bool:
    return config.get("model_type") == MODEL_TYPE


def _mod(name: str):
    return importlib.import_module(f"omlx.patches.{name}")


def available() -> Tuple[bool, str]:
    """(installed, detail): whether oMLX's DeepSeek V4.1 package imports here."""
    try:
        _mod("deepseek_v41.loading")
        _mod("deepseek_v41.moe_offload")
        _mod("deepseek_v41_mlx_lm.model")
    except Exception as e:  # ImportError or anything its import-time code raises
        return False, f"{type(e).__name__}: {e}"
    try:
        from importlib.metadata import version

        return True, f"omlx {version('omlx')}"
    except Exception:
        return True, "omlx (version unknown)"


def require() -> None:
    ok, detail = available()
    if not ok:
        raise RuntimeError(f"{INSTALL_HINT} ({detail})")


# ------------------------------------------------------------------ plan


@dataclass
class ResidencyPlan:
    resident_fraction: float  # of each layer's routed experts kept in memory
    experts_per_layer: int
    resident_experts: int
    admission_bytes: int  # oMLX's load-time memory estimate at this fraction
    full_admission_bytes: int  # the same with every expert resident
    engram_bytes: int  # n-gram tables, served from the SSD
    engram_ssd: bool
    budget_bytes: int

    def to_dict(self) -> dict:
        GB = 1024**3
        return {
            "backend": "omlx",
            "resident_fraction": round(self.resident_fraction, 4),
            "experts_per_layer": self.experts_per_layer,
            "resident_experts_per_layer": self.resident_experts,
            "admission_gb": round(self.admission_bytes / GB, 2),
            "full_model_gb": round(self.full_admission_bytes / GB, 2),
            "engram_gb": round(self.engram_bytes / GB, 2),
            "engram_on_ssd": self.engram_ssd,
            "expert_budget_gb": round(self.budget_bytes / GB, 2),
        }


def experts_per_layer(path: Path) -> int:
    cfg = json.loads((Path(path) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    return int(tc.get("n_routed_experts") or 0)


def plan_residency(path: Path, budget_bytes: int, *, engram_ssd: bool = True,
                   fraction: Optional[float] = None) -> ResidencyPlan:
    """Largest expert residency whose oMLX admission estimate fits ``budget_bytes``.

    ``budget_bytes`` is what is left for weights after the KV cache and the
    activation reserve. ``fraction`` pins the residency instead of fitting it.
    """
    mo = _mod("deepseek_v41.moe_offload")
    res = _mod("deepseek_v41.residency")
    path = Path(path)
    n = experts_per_layer(path)
    if fraction is None:
        fraction = mo.fit_resident_fraction(path, budget_bytes, engram_ssd_offload=engram_ssd)
        if fraction is None:
            raise RuntimeError(
                "Even the minimum expert residency does not fit the memory budget "
                f"({budget_bytes / 1024**3:.1f} GB); lower kv/context or free memory."
            )
    fraction = min(1.0, max(0.0, float(fraction)))
    estimate = res.deepseek_v41_residency_estimate(path)
    return ResidencyPlan(
        resident_fraction=fraction,
        experts_per_layer=n,
        resident_experts=round(fraction * n),
        admission_bytes=int(mo.admission_bytes(path, fraction, engram_ssd_offload=engram_ssd)),
        full_admission_bytes=int(mo.admission_bytes(path, 1.0, engram_ssd_offload=engram_ssd)),
        engram_bytes=int(getattr(estimate, "engram_bytes", 0) or 0),
        engram_ssd=engram_ssd,
        budget_bytes=int(budget_bytes),
    )


# ------------------------------------------------------------------ load


@dataclass
class Loaded:
    model: Any  # oMLX Model, presented the mlx-lm way (make_cache, layers, _omlx_prompt_scope)
    tokenizer: Any  # mlx_lm TokenizerWrapper
    processor: Any  # oMLX Processor: the DeepSeek chat encoder
    plan: ResidencyPlan


def load(path: Path, plan: ResidencyPlan, *, ced_prefill: bool = True, preserve_mtp: Optional[bool] = None) -> Loaded:
    """``preserve_mtp=None`` lets oMLX keep a converted checkpoint's own DSpark layout (``-mtp`` repos)."""
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    loading = _mod("deepseek_v41.loading")
    facade = _mod("deepseek_v41_mlx_lm.model")
    fraction = None if plan.resident_fraction >= 1.0 else plan.resident_fraction
    model, processor = loading.load(
        str(path),
        engram_ssd_offload=plan.engram_ssd,
        moe_expert_offload_resident_fraction=fraction,
        ced_prefill=ced_prefill,
        preserve_mtp=preserve_mtp,
    )
    # The cluster-rank facade is also what mlx-lm's generate loop needs on one Mac:
    # make_cache(), .layers, and the prompt scope that marks CED prefill chunks.
    model.__class__ = facade.Model
    hf = processor.tokenizer
    eos = [t for t in (getattr(hf, "eos_token_id", None),) if t is not None]
    tokenizer = TokenizerWrapper(hf, eos_token_ids=eos or None)
    return Loaded(model, tokenizer, processor, plan)


def encode_chat(loaded: Loaded, messages: List[dict], kwargs: Optional[dict] = None) -> List[int]:
    kw = dict(kwargs or {})
    kw.setdefault("enable_thinking", True)
    return list(loaded.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw))


def prompt_scope(model) -> Any:
    """Context manager that marks this thread's forwards as prompt chunks (CED prefill)."""
    scope = getattr(model, "_omlx_prompt_scope", None)
    return scope() if scope is not None else nullcontext()


def generate_with_prefill_scope(model, steps: Iterator) -> Iterator:
    """Run an mlx-lm generation iterator with the prompt scope held until the first token.

    mlx-lm's ``generate_step`` prefills and produces the first token before its
    first yield; oMLX's cluster worker wraps exactly that span. Decode steps
    marked as prefill are harmless ("CED leaves them unchanged"), so holding the
    scope one step longer is safe.
    """
    with prompt_scope(model):
        try:
            first = next(steps)
        except StopIteration:
            return
    yield first
    yield from steps


# ----------------------------------------------------------------- stats


def _offloaded(model) -> list:
    try:
        mo = _mod("deepseek_v41.moe_offload")
        cls = mo.OffloadedExpert
    except Exception:
        return []
    return [m for _, m in model.named_modules() if isinstance(m, cls)]


def expert_stats(model) -> Optional[dict]:
    """Aggregate hit/miss counters of oMLX's expert slots across layers."""
    mods = _offloaded(model)
    if not mods:
        return None
    hits = misses = 0
    capacity = count = 0
    for m in mods:
        slots = m.slots
        hits += int(getattr(slots, "hits", 0))
        misses += int(getattr(slots, "misses", 0))
        plan = getattr(slots, "plan", None)
        capacity += int(getattr(plan, "capacity", 0) or 0)
        count += int(getattr(plan, "count", 0) or 0)
    total = hits + misses
    return {
        "backend": "omlx",
        "moe_layers": len(mods),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "resident_experts": capacity,
        "total_experts": count,
        "resident_fraction": round(capacity / count, 4) if count else None,
    }


def measure_cache_bytes(model, tokens: int = 8) -> Tuple[float, int]:
    """(bytes per token, fixed bytes) of the model's own cache, from two short runs."""

    def total(n: int) -> int:
        cache = model.make_cache()
        model(mx.zeros((1, n), dtype=mx.int32), cache=cache)
        arrays = []

        def collect(obj):
            if isinstance(obj, mx.array):
                arrays.append(obj)
            elif isinstance(obj, (list, tuple)):
                for o in obj:
                    collect(o)

        for c in cache:
            collect(c.state)  # DeepseekV41Cache.state nests its arrays: (cache_list, left_padding, lengths)
        mx.eval(arrays)
        return sum(a.nbytes for a in arrays)

    one, many = total(1), total(tokens)
    per_tok = max(0.0, (many - one) / (tokens - 1))
    return per_tok, max(0, int(one - per_tok))

