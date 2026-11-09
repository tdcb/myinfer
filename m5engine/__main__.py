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

    g = sub.add_parser("generate", help="One-shot chat generation")
    _add_load_args(g, require_model=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--temp", type=float, default=0.0)
    g.add_argument("--no-think", action="store_true", help="Pass enable_thinking=False to the chat template")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .engine import Engine, GenerationParams, LoadOptions

    engine = Engine()
    if a.cmd == "serve":
        from .server import App, serve

        opts = _options(a)
        app = App(engine, Path(a.models_dir), {k: v for k, v in opts.items() if k != "model"})
        if a.model:
            app.load_async(opts)
        serve(app, a.host, a.port)
        return 0

    info = engine.load(LoadOptions.from_dict(_options(a)))
    if a.cmd == "plan":
        print(json.dumps({k: info.get(k) for k in ("model", "context", "expert_streaming", "hardware")}, indent=2))
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
