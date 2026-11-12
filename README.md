# M5Engine: local LLM inference for Apple Silicon (MLX)

M5Engine serves dense and Mixture-of-Experts models on an Apple-Silicon Mac
(tuned for a 64 GB M5 Pro). It has an OpenAI-compatible API and a web control
panel. Compute runs on [MLX](https://github.com/ml-explore/mlx) /
[mlx-lm](https://github.com/ml-explore/mlx-lm), so it uses the real model
architectures, tokenizers and chat templates. On top of that, M5Engine adds:

- **Maximum context sizing.** After loading the weights, M5Engine measures the
  model's real KV-cache cost per token and sizes the context to fill the GPU
  memory budget. The KV cache is 8-bit by default (4-bit optional).
  Long-context attention uses a blocked online-softmax, so prefill at 100K+
  tokens doesn't need a full score matrix. For Qwen2/Qwen3 models, YaRN
  extends the native 32K window up to 4x (128K). It is only switched on for
  requests longer than the native window, so short prompts keep normal
  quality.
- **SSD expert streaming for MoE models larger than memory.** Attention,
  embeddings, routers and shared experts stay resident. Routed experts are
  read on demand into a byte-bounded expert cache, with parallel `pread`s,
  `F_NOCACHE` on macOS, and the next layer's experts prefetched during
  decode. This is turned on automatically when a MoE model doesn't fit.
- **Strata-style expert tiering on unified memory.** A saved hot-expert
  profile drives eviction and a background warm fill at load, the expert
  cache borrows KV budget that the current request doesn't need, an optional
  contiguous experts sidecar makes an expert one SSD read, and
  `active_experts` trades a little quality for fewer bytes per token. Every
  request reports SSD MB/token and the expert hit rate.
- **Prefix KV cache.** Multi-turn chats reuse the KV cache of the longest
  matching prefix. It is moved, not copied, so it costs no extra memory, and
  its size counts against the KV budget.
- **Speculative decoding** with a real draft model (e.g. Qwen3-0.6B for Qwen3).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web UI + API on http://127.0.0.1:8080 (load models from the UI)
python -m m5engine serve

# ...or load a model at startup
python -m m5engine serve --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit

# Print the memory / context plan for a model
python -m m5engine plan --model mlx-community/Qwen3-32B-4bit
```

`Launch_M5Engine_UI.command` sets up `.venv` and starts the server from Finder.

The **Models** tab in the UI searches and downloads MLX models from Hugging Face
(with a fit and context estimate for this Mac), benchmarks installed models and
keeps their history, checks for and applies updates, and deletes models. See
[USER_GUIDE.md](USER_GUIDE.md#5-managing-models-models-tab).

### Getting the most context

The GPU budget defaults to Metal's recommended working set, which is about
75% of RAM. To give the GPU more of a 64 GB machine (this resets on reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=57344   # 56 GB for the GPU, 8 GB for macOS
```

Then pick `--kv-bits 4` for roughly 1.9x more tokens than 8-bit. The UI and
`/v1/engine/stats` show `max_context` and whether it is limited by memory,
by the model's position limit, or by your `--max-context`.

## What fits on a 64 GB M5 Pro

| Model | Weights | Mode |
|---|---|---|
| Qwen3-32B / Qwen2.5-32B 4-bit (dense) | 18.4 GB | in memory, 128K context with YaRN |
| Gemma-3-27B 4-bit (dense) | 16.8 GB | in memory |
| Qwen3-30B-A3B 4-bit (MoE, 3B active) | 17.2 GB | in memory, 128K context |
| Qwen3-Next-80B-A3B 4-bit (MoE) | 44.8 GB | in memory |
| GLM-4.5-Air 4-bit (MoE, 12B active) | 60.1 GB | expert streaming |
| Qwen3-235B-A22B 3/4-bit (MoE, 22B active) | 103 / 132 GB | expert streaming |

Decode speed is bounded by memory bandwidth for in-memory models. For streamed
models it is bounded by SSD reads of cache-missing experts
(`ssd_mb_per_token` in every request's metrics): a bigger `--expert-cache-gb`,
a lower-bit checkpoint or `--active-experts` all cut it. DeepSeek-V4.1-Flash
(552B, 384 experts/layer, 6 active; 222-292 GiB as MLX 2-bit) needs ~300 GB
of SSD and the oMLX backend (`tools/install_deepseek_backend.sh`); see
USER_GUIDE §4.

## Layout

```
m5engine/
  engine.py         model lifecycle, generation, context enforcement
  context.py        KV footprint measurement, context planner, YaRN
  attention.py      memory-bounded attention over quantized KV
  expert_stream.py  SSD expert streaming, hot-expert profile, elastic cache, experts sidecar
  deepseek_v41.py   DeepSeek-V4.1-Flash via the optional oMLX backend (Engram on SSD, expert residency)
  prefix_cache.py   longest-prefix KV reuse
  hardware.py       RAM / GPU working set / thermal state
  server.py         OpenAI-compatible HTTP server and control API
  models.py         local models, Hub search, downloads, updates, deletion, benchmark history
web/index.html      control panel
tests/              unit tests (no downloads needed)
tools/              API end-to-end check, downloader
native/             previous C++/Metal engine, kept as a base for custom kernels
```

## Tests

```bash
pip install pytest
pytest tests                              # unit tests: attention, planner, prefix cache, expert streaming
python tools/test_openai_api.py --port 8080   # against a running server with a real model
```

## Status

- Verified on Linux/CPU MLX against real checkpoints (Qwen3-0.6B,
  OLMoE-1B-7B, Qwen3-30B-A3B). Streamed-expert outputs are bit-identical to
  the fully resident model. Performance on Apple GPUs has not been benchmarked
  yet.
- Custom Metal kernels (`native/`) are planned only where profiling on the
  M5 shows MLX is the bottleneck.
- The expert tiering (profile, warm fill, elastic cache, sidecar, active
  experts) is verified on Linux/CPU with OLMoE; its effect on speed needs a
  Mac and an SSD-streamed model (`python -m m5engine bench --preset strata`).
- DeepSeek-V4.1-Flash is served through the optional oMLX backend
  (`tools/install_deepseek_backend.sh`, Python 3.11+, macOS). The integration
  (residency planning, chat encoding, CED prefill scope, metrics) is tested
  against a stub and oMLX's planner on a tiny checkpoint; no real DeepSeek
  load has been run yet. oMLX's kernels are Metal, so this backend cannot run
  on Linux.
