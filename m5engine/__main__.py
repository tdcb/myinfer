"""Command line entry point: `python -m m5engine serve|plan|generate`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_load_args(p: argparse.ArgumentParser, require_model: bool) -> None:
    p.add_argument("--model", required=require_model, help="Local MLX model dir or Hugging Face repo id")
    p.add_argument("--kv-bits", type=int, default=8, choices=[0, 4, 8],
                   help="KV cache precision; 0 = fp16 (default 8)")
    p.add_argument("--quantized-kv-start", type=int, default=4096,
                   help="Keep the first N tokens of KV in fp16 before quantizing")
    p.add_argument("--max-context", type=int, help="Cap the context (default: largest that fits)")
    p.add_argument("--rope-scaling", default="auto",
                   help="auto (YaRN only for requests past the native window), off, 2 or 4")
    p.add_argument("--stream-experts", default="auto", choices=["auto", "on", "off"],
                   help="Stream MoE experts from SSD (auto = only if the model doesn't fit)")
    p.add_argument("--expert-cache-gb", type=float, help="Expert cache size when streaming")
    p.add_argument("--io-threads", type=int, default=8)
    p.add_argument("--no-prefetch", action="store_true", help="Disable next-layer expert prefetch")
    p.add_argument("--active-experts", type=int,
                   help="Route each token to this many experts instead of the model's default (fewer = faster "
                        "SSD streaming, slightly lower quality; Strata's 'Speed' setting)")
    p.add_argument("--no-expert-profile", action="store_true",
                   help="Don't learn/save which experts are hot (the profile drives eviction and the warm fill)")
    p.add_argument("--no-expert-pack", action="store_true", help="Ignore the contiguous experts sidecar")
    p.add_argument("--no-warm", action="store_true", help="Don't pre-read hot experts at load")
    p.add_argument("--no-elastic-cache", action="store_true",
                   help="Keep the expert cache at its base size instead of borrowing unused KV budget")
    p.add_argument("--draft-model", help="Small model with the same tokenizer for speculative decoding")
    p.add_argument("--num-draft-tokens", type=int, default=3)
    p.add_argument("--prefill-step", type=int, default=2048)
    p.add_argument("--memory-budget-gb", type=float,
                   help="GPU memory budget (default: Metal's recommended working set)")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")


def _options(a) -> dict:
    d = {
        "model": a.model,
        "kv_bits": a.kv_bits,
        "quantized_kv_start": a.quantized_kv_start,
        "max_context": a.max_context,
        "rope_scaling": a.rope_scaling,
        "stream_experts": a.stream_experts,
        "expert_cache_gb": a.expert_cache_gb,
        "active_experts": a.active_experts,
        "expert_profile": not a.no_expert_profile,
        "expert_pack": not a.no_expert_pack,
        "warm_experts": not a.no_warm,
        "elastic_expert_cache": not a.no_elastic_cache,
        "io_threads": a.io_threads,
        "prefetch_experts": not a.no_prefetch,
        "draft_model": a.draft_model,
        "num_draft_tokens": a.num_draft_tokens,
        "prefill_step": a.prefill_step,
        "memory_budget_gb": a.memory_budget_gb,
        "prefix_cache": not a.no_prefix_cache,
        "trust_remote_code": a.trust_remote_code,
    }
    return {k: v for k, v in d.items() if v is not None}


BENCH_PRESETS = {
    "quick": [(64, 128)],
    "strata": [(64, 128), (3000, 128), (14000, 64)],  # Strata's DeepSeek tables: short answer, 3K and 14K prompts
    "long": [(32000, 32)],
}


def _bench_row(r: dict) -> str:
    s = (f"prompt {r['prompt_tokens']:>6} tok @ {r['prompt_tps']:7.1f} tok/s | decode {r['completion_tokens']:>4} tok "
         f"@ {r['decode_tps']:6.2f} tok/s | ttft {r['ttft_ms']:8.0f} ms | peak {r['peak_memory_gb']} GB")
    if r.get("expert_streaming"):
        hit = r.get("expert_hit_rate")
        s += (f" | experts: cache {r['expert_cache_gb']} GB, hit {hit * 100 if hit is not None else 0:.0f}%, "
              f"SSD {r.get('ssd_mb_per_token') or 0} MB/token")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="m5engine", description="MLX local inference for Apple Silicon")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run the OpenAI-compatible server and web UI")
    _add_load_args(s, require_model=False)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--models-dir", default="models")

    pl = sub.add_parser("plan", help="Load a model and print its memory/context plan")
    _add_load_args(pl, require_model=True)
    pl.add_argument("--estimate", action="store_true",
                    help="Don't download or load: estimate from the Hub config and file sizes alone")

    rp = sub.add_parser("repack", help="Write the contiguous experts sidecar (one SSD read per expert)")
    rp.add_argument("--model", required=True)

    b = sub.add_parser("bench", help="Benchmark a model on fixed workloads and record the results")
    _add_load_args(b, require_model=True)
    b.add_argument("--preset", default="quick", choices=sorted(BENCH_PRESETS),
                   help="quick: 64/128 · strata: short, 3K and 14K prompts like Strata's tables · long: 32K prompt")
    b.add_argument("--models-dir", default="models", help="Where benchmarks.json is kept")
    b.add_argument("--json", action="store_true", help="Print the raw results as JSON")

    g = sub.add_parser("generate", help="One-shot chat generation")
    _add_load_args(g, require_model=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--temp", type=float, default=0.0)
    g.add_argument("--no-think", action="store_true", help="Pass enable_thinking=False to the chat template")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .engine import Engine, LoadOptions

    engine = Engine()
    if a.cmd == "plan" and a.estimate:
        from .models import disk_free_gb, hub_info

        est = hub_info(a.model, engine.budget_bytes(LoadOptions.from_dict({"model": a.model, **_options(a)})),
                       a.kv_bits)
        est["disk_free_gb"] = round(disk_free_gb(Path(".")), 1)
        print(json.dumps(est, indent=2))
        return 0
    if a.cmd == "repack":
        from mlx_lm.utils import load_model

        from .engine import _download
        from .expert_stream import repack_experts

        path = Path(_download(a.model))
        model, _ = load_model(path, lazy=True)
        last = [0]

        def progress(done, total):
            pct = done * 100 // max(total, 1)
            if pct >= last[0] + 5 or done == total:
                last[0] = pct
                print(f"\r{pct:3d}%  {done / 1024**3:.1f} / {total / 1024**3:.1f} GB", end="", file=sys.stderr, flush=True)

        out = repack_experts(model, path, progress)
        print(f"\nWrote {out} ({out.stat().st_size / 1024**3:.1f} GB)", file=sys.stderr)
        return 0
    if a.cmd == "serve":
        from .server import App, serve

        opts = _options(a)
        app = App(engine, Path(a.models_dir), {k: v for k, v in opts.items() if k != "model"})
        if a.model:
            app.load_async(opts)
        serve(app, a.host, a.port)
        return 0

    info = engine.load(LoadOptions.from_dict(_options(a)))
    try:
        return _run_loaded(a, engine, info)
    finally:
        engine.unload()  # saves the learned expert profile


def _run_loaded(a, engine, info: dict) -> int:
    from .engine import GenerationParams

    if a.cmd == "plan":
        print(json.dumps({k: info.get(k) for k in ("model", "context", "expert_streaming", "hardware")}, indent=2))
        return 0

    if a.cmd == "bench":
        from .models import BenchmarkStore, find_local

        store = BenchmarkStore(Path(a.models_dir))
        try:  # record under the id the Models tab uses for this model
            key = find_local(Path(a.models_dir), str(engine.model_path))["id"]
        except KeyError:
            key = a.model
        results = []
        for prompt_tokens, gen_tokens in BENCH_PRESETS[a.preset]:
            r = engine.benchmark(prompt_tokens, gen_tokens)
            store.add(key, r)
            results.append(r)
            if not a.json:
                print(_bench_row(r), file=sys.stderr)
        if a.json:
            print(json.dumps(results, indent=2))
        return 0

    tokens = engine.encode_chat([{"role": "user", "content": a.prompt}],
                                {"enable_thinking": False} if a.no_think else None)
    for c in engine.generate(tokens, GenerationParams(max_tokens=a.max_tokens, temperature=a.temp)):
        sys.stdout.write(c.text)
        sys.stdout.flush()
    m = engine.last
    print(f"\n\n[prompt {m.prompt_tokens} tok @ {m.prompt_tps:.1f} tok/s | "
          f"decode {m.completion_tokens} tok @ {m.decode_tps:.1f} tok/s | peak {m.peak_memory_gb} GB]",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
