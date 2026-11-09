"""MLX inference engine: model lifecycle, context planning and generation."""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import _download, load_model, load_tokenizer

from . import attention, hardware
from .context import (
    GB,
    MAX_YARN_FACTOR,
    ContextPlan,
    KVFootprint,
    apply_yarn,
    combine,
    kv_estimate_per_token,
    measure_kv_footprint,
    native_context,
    plan_context,
    reserve_bytes,
    yarn_capable,
    yarn_factor_for,
)
from .expert_stream import (
    StreamingInfo,
    enable_expert_streaming,
    estimate_expert_bytes,
    find_expert_modules,
)
from .prefix_cache import PrefixCache

log = logging.getLogger("m5engine")


class EngineError(Exception):
    status = 400


class ContextOverflow(EngineError):
    pass


class NotLoaded(EngineError):
    status = 503


@dataclass
class LoadOptions:
    model: str
    kv_bits: int = 8  # 0 = keep KV in fp16/bf16
    kv_group_size: int = 64
    quantized_kv_start: int = 4096
    max_context: Optional[int] = None
    rope_scaling: str = "auto"  # auto | off | 2 | 4
    stream_experts: str = "auto"  # auto | on | off
    expert_cache_gb: Optional[float] = None
    io_threads: int = 8
    prefetch_experts: bool = True
    draft_model: Optional[str] = None
    num_draft_tokens: int = 3
    prefill_step: int = 2048
    memory_budget_gb: Optional[float] = None
    prefix_cache: bool = True
    prefix_cache_entries: int = 4
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoadOptions":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and v is not None}
        if "model" not in known:
            raise EngineError("'model' is required (local path or Hugging Face repo id)")
        return cls(**known)


@dataclass
class GenerationParams:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    use_prefix_cache: bool = True
    use_draft: bool = True


@dataclass
class Chunk:
    text: str
    finish_reason: Optional[str] = None
    metrics: Optional["RequestMetrics"] = None


@dataclass
class RequestMetrics:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    prompt_tps: float = 0.0
    decode_tps: float = 0.0
    draft_accepted: int = 0
    yarn_factor: float = 1.0
    peak_memory_gb: float = 0.0
    finish_reason: str = ""


def _param_bytes(model) -> int:
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


BENCH_PASSAGE = (
    "Unified memory lets the CPU and GPU share one pool of high-bandwidth memory, so model weights "
    "are read in place instead of being copied between devices. Decode speed is mostly limited by "
    "how many bytes of weights and cache must be streamed for every generated token. "
)
BENCH_QUESTION = "\n\nWrite a long, detailed essay expanding on the text above."


class Engine:
    def __init__(self):
        attention.install()
        self.hw = hardware.detect()
        self._lock = threading.RLock()
        self._gen_lock = threading.Lock()
        self.model = None
        self.draft_model = None
        self.tokenizer = None
        self.config: dict = {}
        self.options: Optional[LoadOptions] = None
        self.model_path: Optional[Path] = None
        self.plan: Optional[ContextPlan] = None
        self.footprint: Optional[KVFootprint] = None
        self.streaming: Optional[StreamingInfo] = None
        self.prefix = PrefixCache()
        self.resident_bytes = 0
        self.load_seconds = 0.0
        self.state = "empty"
        self.error: Optional[str] = None
        self.last: RequestMetrics = RequestMetrics()
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self._native = 0
        self._plain_limit = 0
        self._yarn_now = 1.0
        self._draft_yarn = False

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def budget_bytes(self, opts: Optional[LoadOptions] = None) -> int:
        opts = opts or self.options
        if opts and opts.memory_budget_gb:
            return int(opts.memory_budget_gb * GB)
        return self.hw.gpu_limit_bytes

    def unload(self) -> None:
        with self._gen_lock, self._lock:
            self._unload()

    def _unload(self) -> None:
        if self.streaming:
            self.streaming.cache.close()
        self.model = self.draft_model = self.tokenizer = None
        self.streaming = None
        self.plan = None
        self.footprint = None
        self.prefix.clear()
        self.config = {}
        self.model_path = None
        self.state = "empty"
        gc.collect()
        mx.clear_cache()

    def load(self, opts: LoadOptions) -> dict:
        with self._gen_lock, self._lock:
            self._unload()
            self.state = "loading"
            self.error = None
            try:
                self._load(opts)
            except Exception as e:
                self._unload()
                self.state = "error"
                self.error = str(e)
                raise
            self.state = "ready"
            return self.info()

    def _load(self, opts: LoadOptions) -> None:
        t0 = time.perf_counter()
        if opts.kv_bits not in (0, 4, 8):
            raise EngineError("kv_bits must be 0 (fp16), 4 or 8")
        path = _download(opts.model)
        model, config = load_model(path, lazy=True)
        tokenizer = load_tokenizer(
            path,
            {"trust_remote_code": True} if opts.trust_remote_code else {},
            eos_token_ids=config.get("eos_token_id"),
        )
        budget = self.budget_bytes(opts)
        native, plain_limit = native_context(config)
        hidden = int(config.get("hidden_size") or config.get("text_config", {}).get("hidden_size") or 4096)
        reserve = reserve_bytes(budget, hidden, opts.prefill_step)

        draft = None
        if opts.draft_model:
            dpath = _download(opts.draft_model)
            draft, dconfig = load_model(dpath)
            self._draft_yarn = yarn_capable(dconfig)
        draft_bytes = _param_bytes(draft) if draft is not None else 0

        total_bytes = _param_bytes(model)
        has_experts = bool(find_expert_modules(model))
        min_kv = kv_estimate_per_token(config, opts.kv_bits) * min(plain_limit, 32768)
        fits = total_bytes + draft_bytes + reserve + min_kv <= budget
        stream = opts.stream_experts == "on" or (opts.stream_experts == "auto" and not fits and has_experts)
        if stream and not has_experts:
            raise EngineError("Expert streaming needs a Mixture-of-Experts model; this model is dense.")
        if not stream and total_bytes + draft_bytes + reserve > budget:
            raise EngineError(
                f"Model needs {total_bytes / GB:.1f} GB but the GPU budget is {budget / GB:.1f} GB. "
                "Use a smaller quantization, or a MoE model with expert streaming."
            )

        streaming = None
        if stream:
            expert_total = estimate_expert_bytes(path, model)
            resident = total_bytes - expert_total
            free = budget - resident - draft_bytes - reserve
            if free <= 0:
                raise EngineError(
                    f"Even without experts the model needs {resident / GB:.1f} GB "
                    f"(budget {budget / GB:.1f} GB)."
                )
            if opts.expert_cache_gb is not None:
                cache_bytes = int(opts.expert_cache_gb * GB)
            else:
                # Half of what's left goes to experts, half to the KV cache.
                cache_bytes = int(free * 0.5)
            cache_bytes = max(0, min(cache_bytes, expert_total, free - int(min_kv / 4)))
            streaming = enable_expert_streaming(
                model, path, cache_bytes, io_threads=opts.io_threads, prefetch=opts.prefetch_experts
            )

        mx.eval(model.parameters())
        resident_bytes = _param_bytes(model) + draft_bytes
        if mx.metal.is_available():
            try:
                mx.set_wired_limit(min(budget, self.hw.recommended_working_set_bytes))
            except Exception as e:  # pragma: no cover - macOS only
                log.warning("Could not set wired limit: %s", e)

        fp = measure_kv_footprint(model, opts.kv_bits, opts.kv_group_size)
        if opts.kv_bits and opts.quantized_kv_start:
            dense_fp = measure_kv_footprint(model, 0)
            extra = max(0.0, dense_fp.bytes_per_token - fp.bytes_per_token) * opts.quantized_kv_start
            fp = KVFootprint(fp.bytes_per_token, int(fp.fixed_bytes + extra), fp.kv_bits)
        if draft is not None:
            fp = combine(fp, measure_kv_footprint(draft, opts.kv_bits, opts.kv_group_size))

        plan = plan_context(
            config=config,
            footprint=fp,
            budget_bytes=budget,
            resident_weight_bytes=resident_bytes,
            expert_cache_bytes=streaming.cache.capacity_bytes if streaming else 0,
            prefill_step=opts.prefill_step,
            rope_scaling=opts.rope_scaling,
            requested_context=opts.max_context,
        )
        if plan.max_context < 512:
            raise EngineError(
                f"Only {plan.max_context} tokens of KV cache fit in memory; "
                "lower expert_cache_gb, use kv_bits=4, or pick a smaller model."
            )

        self.model, self.draft_model, self.tokenizer = model, draft, tokenizer
        self.config, self.options, self.model_path = config, opts, Path(path)
        self.plan, self.footprint, self.streaming = plan, fp, streaming
        self.resident_bytes = resident_bytes
        self.prefix = PrefixCache(max_entries=opts.prefix_cache_entries)
        self._native, self._plain_limit = native, plain_limit
        self._yarn_now = 1.0
        if opts.rope_scaling not in ("auto", "off") and plan.yarn_factor > 1:
            self._set_yarn(plan.yarn_factor)
        self.load_seconds = time.perf_counter() - t0
        self.last = RequestMetrics()
        log.info(
            "Loaded %s in %.1fs: resident %.1f GB, max context %d (%s)",
            opts.model, self.load_seconds, resident_bytes / GB, plan.max_context, plan.limited_by,
        )

    def _set_yarn(self, factor: float) -> None:
        if factor == self._yarn_now:
            return
        apply_yarn(self.model, factor, self._native)
        if self.draft_model is not None and self._draft_yarn:
            apply_yarn(self.draft_model, factor, self._native)
        self._yarn_now = factor

    def _yarn_for_request(self, tokens: int) -> float:
        opts, plan = self.options, self.plan
        if plan.yarn_factor <= 1:
            return 1.0
        if opts.rope_scaling != "auto":
            return plan.yarn_factor
        if tokens <= self._plain_limit:
            return 1.0
        return yarn_factor_for(tokens, self._native, min(plan.yarn_factor, MAX_YARN_FACTOR))

    # ------------------------------------------------------------- generate
    def encode_chat(self, messages: List[dict], chat_template_kwargs: Optional[dict] = None,
                    tools: Optional[list] = None) -> List[int]:
        tok = self.tokenizer
        msgs = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            msgs.append({**m, "content": content or ""})
        if getattr(tok, "chat_template", None):
            kwargs = dict(chat_template_kwargs or {})
            if tools:
                kwargs["tools"] = tools
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
            return list(tok.encode(text, add_special_tokens=False))
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
        return list(tok.encode(text))

    def encode_text(self, prompt: str) -> List[int]:
        return list(self.tokenizer.encode(prompt))

    def generate(self, tokens: List[int], params: GenerationParams) -> Iterator[Chunk]:
        if not self.loaded:
            raise NotLoaded("No model loaded")
        plan = self.plan
        n_prompt = len(tokens)
        if n_prompt >= plan.max_context:
            raise ContextOverflow(
                f"Prompt is {n_prompt} tokens but the maximum context is {plan.max_context} "
                f"(limited by {plan.limited_by})."
            )
        max_tokens = max(1, min(params.max_tokens, plan.max_context - n_prompt))
        return self._generate(tokens, max_tokens, params)

    def _generate(self, tokens: List[int], max_tokens: int, params: GenerationParams) -> Iterator[Chunk]:
        with self._gen_lock:
            if not self.loaded:
                raise NotLoaded("No model loaded")
            opts = self.options
            factor = self._yarn_for_request(len(tokens) + max_tokens)
            self._set_yarn(factor)
            key = f"yarn={factor}"
            use_prefix = opts.prefix_cache and params.use_prefix_cache
            cache, hit = (self.prefix.take(key, tokens) if use_prefix else (None, 0))
            if cache is None:
                cache = make_prompt_cache(self.model)
                if self.draft_model is not None:
                    cache += make_prompt_cache(self.draft_model)
                hit = 0
            need = self.footprint.bytes_for(len(tokens) + max_tokens)
            self.prefix.evict_to(self.plan.kv_budget_bytes - need)

            if params.seed is not None:
                mx.random.seed(params.seed)
            sampler = make_sampler(
                temp=params.temperature, top_p=params.top_p if params.top_p < 1 else 0.0,
                min_p=params.min_p, top_k=params.top_k,
            )
            processors = make_logits_processors(
                repetition_penalty=params.repetition_penalty,
                presence_penalty=params.presence_penalty,
                frequency_penalty=params.frequency_penalty,
            )
            draft = self.draft_model if params.use_draft else None
            if self.draft_model is not None and draft is None:
                cache = cache[: len(self.model.layers)]  # main-model caches only
            kwargs = dict(
                sampler=sampler,
                logits_processors=processors,
                prompt_cache=cache,
                prefill_step_size=opts.prefill_step,
                kv_bits=opts.kv_bits or None,
                kv_group_size=opts.kv_group_size,
                quantized_kv_start=opts.quantized_kv_start,
            )
            if draft is not None:
                kwargs["num_draft_tokens"] = opts.num_draft_tokens

            m = RequestMetrics(prompt_tokens=len(tokens), cached_tokens=hit, yarn_factor=factor)
            mx.reset_peak_memory()
            generated: List[int] = []
            text = ""
            emitted = 0
            finish = "length"
            t0 = time.perf_counter()
            t_first = None
            stop = [s for s in params.stop if s]
            hold = max((len(s) for s in stop), default=1) - 1
            stopped = False
            try:
                for resp in stream_generate(
                    self.model, self.tokenizer, mx.array(tokens[hit:]), max_tokens,
                    draft_model=draft, **kwargs,
                ):
                    if t_first is None:
                        t_first = time.perf_counter()
                        m.ttft_ms = (t_first - t0) * 1000
                        m.prompt_tps = resp.prompt_tps
                    if not (resp.finish_reason == "stop" and resp.token in self.tokenizer.eos_token_ids):
                        generated.append(resp.token)
                    m.draft_accepted += int(resp.from_draft)
                    m.decode_tps = resp.generation_tps
                    text += resp.text
                    if resp.finish_reason:
                        finish = resp.finish_reason
                    safe = len(text)
                    if stop:
                        hits = [i for i in (text.find(s, max(0, emitted - hold)) for s in stop) if i >= 0]
                        if hits:
                            safe, finish, stopped = min(hits), "stop", True
                        else:
                            safe = max(emitted, len(text) - hold)
                    if safe > emitted:
                        yield Chunk(text[emitted:safe])
                        emitted = safe
                    if stopped:
                        break
                if not stopped and emitted < len(text):
                    yield Chunk(text[emitted:])
                    emitted = len(text)
            finally:
                m.completion_tokens = len(generated)
                m.finish_reason = finish
                m.peak_memory_gb = round(mx.get_peak_memory() / GB, 2)
                self.last = m
                self.totals["requests"] += 1
                self.totals["prompt_tokens"] += m.prompt_tokens
                self.totals["completion_tokens"] += m.completion_tokens
                self.totals["cached_tokens"] += m.cached_tokens
                if use_prefix and (draft is not None or self.draft_model is None):
                    self.prefix.put(key, tokens + generated, cache)
            yield Chunk("", finish, m)

    # ------------------------------------------------------------ benchmark
    def benchmark(self, prompt_tokens: int = 1024, gen_tokens: int = 128, use_draft: bool = True) -> dict:
        """Fresh-cache prefill + greedy decode of a fixed workload, for comparing models/settings."""
        if not self.loaded:
            raise NotLoaded("No model loaded")
        prompt_tokens = max(16, min(int(prompt_tokens), self.plan.max_context - gen_tokens - 64))
        unit = self.encode_text(BENCH_PASSAGE)
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None and unit and unit[0] == bos:
            unit = unit[1:]
        filler = (unit * (prompt_tokens // len(unit) + 1))[:prompt_tokens]
        messages = [{"role": "user", "content": self.tokenizer.decode(filler) + BENCH_QUESTION}]
        tokens = self.encode_chat(messages, {"enable_thinking": False})
        params = GenerationParams(max_tokens=gen_tokens, temperature=0.0, use_prefix_cache=False,
                                  use_draft=use_draft)
        st = self.streaming.cache.stats if self.streaming else None
        before = (st.hits, st.misses) if st else (0, 0)
        m = None
        for chunk in self.generate(tokens, params):
            m = chunk.metrics or m
        hit_rate = None
        if st:
            hits, misses = st.hits - before[0], st.misses - before[1]
            hit_rate = round(hits / (hits + misses), 4) if hits + misses else 0.0
        return {
            "time": time.time(),
            "model": self.options.model,
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "ttft_ms": round(m.ttft_ms, 1),
            "prompt_tps": round(m.prompt_tps, 1),
            "decode_tps": round(m.decode_tps, 2),
            "peak_memory_gb": m.peak_memory_gb,
            "draft_model": self.options.draft_model if use_draft else None,
            "draft_accepted": m.draft_accepted,
            "kv_bits": self.options.kv_bits,
            "max_context": self.plan.max_context,
            "expert_streaming": st is not None,
            "expert_hit_rate": hit_rate,
            "device": "metal" if mx.metal.is_available() else "cpu",
            "chip": self.hw.chip,
            "mlx_version": mx.__version__,
        }

    # ---------------------------------------------------------------- info
    def info(self) -> dict:
        hw = self.hw.to_dict()
        d: Dict[str, Any] = {
            "state": self.state,
            "error": self.error,
            "hardware": {
                **{k: v for k, v in hw.items() if not k.endswith("_bytes")},
                **{k.replace("_bytes", "_gb"): round(v / GB, 2) for k, v in hw.items() if k.endswith("_bytes")},
                "thermal": hardware.thermal_state(),
                "mlx_version": mx.__version__,
            },
            "memory": {k.replace("_bytes", "_gb"): round(v / GB, 2) for k, v in hardware.memory_snapshot().items()},
        }
        if not self.loaded:
            return d
        opts = self.options
        d["model"] = {
            "id": opts.model,
            "path": str(self.model_path),
            "type": self.config.get("model_type"),
            "moe": bool(self.streaming) or bool(find_expert_modules(self.model)),
            "resident_gb": round(self.resident_bytes / GB, 2),
            "load_seconds": round(self.load_seconds, 1),
            "draft_model": opts.draft_model,
            "quantization": self.config.get("quantization"),
        }
        d["options"] = asdict(opts)
        d["context"] = {**self.plan.to_dict(), "current_yarn_factor": self._yarn_now}
        d["expert_streaming"] = self.streaming.to_dict() if self.streaming else None
        d["prefix_cache"] = {
            "entries": len(self.prefix),
            "gb": round(self.prefix.nbytes / GB, 2),
            "hit_tokens": self.prefix.total_hit_tokens,
            "prompt_tokens": self.prefix.total_prompt_tokens,
        }
        d["last_request"] = asdict(self.last)
        d["totals"] = dict(self.totals)
        return d
