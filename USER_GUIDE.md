# M5Engine user guide

## 1. Install and start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m m5engine serve --port 8080          # open http://127.0.0.1:8080
```

Models can be a local MLX directory or a Hugging Face repo id; repo ids are
downloaded on first use. The **Models** tab in the UI manages everything on disk
(see section 5).

## 2. Load options

All options can be set in the UI, as `serve`/`plan` flags, or in the body of
`POST /v1/models/load`.

| Option | Default | Meaning |
|---|---|---|
| `kv_bits` (`--kv-bits`) | `8` | KV cache precision: 8, 4 (most context) or 0 (fp16) |
| `quantized_kv_start` | `4096` | First N tokens stay fp16 (faster short chats) |
| `max_context` | largest that fits | Cap the context window |
| `rope_scaling` | `auto` | `auto`: YaRN only when a request exceeds the native window; `off`; `2`/`4`: always on |
| `stream_experts` | `auto` | Stream MoE experts from SSD when the model doesn't fit (`on` forces it) |
| `expert_cache_gb` | half the free memory | Expert cache size when streaming |
| `draft_model` | none | Draft model for speculative decoding (must share the tokenizer) |
| `num_draft_tokens` | `3` | Tokens drafted per step |
| `memory_budget_gb` | Metal recommended working set | GPU memory the planner may use |
| `prefill_step` | `2048` | Prompt tokens processed per chunk |

## 3. How the context window is chosen

After the weights are loaded, M5Engine runs one token through the model and
measures the real KV bytes per token (this covers GQA, MLA, sliding-window and
linear-attention layers). Then:

```
KV budget      = GPU budget − resident weights − expert cache − reserve
memory limit   = KV budget ÷ KV bytes per token        (rounded to 256)
model limit    = native window × YaRN factor (≤ 4, Qwen2/Qwen3 only)
max context    = min(memory limit, model limit, max_context)
```

The reserve covers activations for a prefill chunk plus a safety margin.
Prefix-cache entries are evicted whenever a request needs their memory.
Requests whose prompt is longer than `max context` get HTTP 400, and
`max_tokens` is clipped to the remaining room.

Example (Qwen3-30B-A3B 4-bit, 8-bit KV): about 52 KB per token, so 128K
tokens need about 6.6 GB. The model limit (128K with YaRN) is what binds.

To raise the GPU budget on a 64 GB Mac (this resets on reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=57344
```

## 4. MoE models larger than memory

With `stream_experts=auto`, a MoE model that doesn't fit is loaded with only
its non-expert weights resident. For each MoE layer the engine:

1. evaluates the router and reads back the chosen experts,
2. loads missing experts from disk (parallel reads, `F_NOCACHE`) into a
   byte-bounded expert cache,
3. runs the normal quantized expert matmul on just those experts.

During decode it also predicts the next layer's experts from the current
hidden state and starts reading them early.

The placement follows Strata's design (a GPU/RAM/SSD tiering engine for
NVIDIA cards), adapted to unified memory where the GPU, the CPU and the
expert cache all share one pool:

- **Hot-expert profile.** The engine keeps a decayed count of how often each
  expert is routed to, saved next to the model in `.m5engine/expert-profile.npz`
  (every 5 requests and on unload). Eviction drops the *coldest* expert, not
  the least recent, and the next load pre-reads the hottest experts in the
  background so the first answer starts from a warm cache. A profile learned
  from one kind of work (code, chat) speeds that kind up most.
- **Elastic expert cache.** The context plan reserves the KV budget for the
  full context, but a request that needs less lends the rest to the expert
  cache; a longer request shrinks it back, evicting the coldest experts. The
  UI shows `cache x / y (base z)` while the cache is borrowing.
- **Contiguous experts sidecar.** `python -m m5engine repack --model DIR`
  writes `.m5engine/experts.pack`, where each expert's tensors (9 for a
  quantized expert) sit together, so an expert is one SSD read instead of nine.
  It is a second copy of the experts (same size as the expert weights), used
  automatically when present and ignored when the checkpoint changes.
- **Active experts ("Speed").** `active_experts=5` on a 6-expert model reads
  5/6 of the expert bytes per token at a small quality cost. Fewer than the
  model's default only; the UI lists the default beside it.
- **Per-request SSD metrics.** `ssd_mb_per_token` (expert bytes read from
  disk per generated token) and `expert_hit_rate` are reported for every
  request and benchmark. Decode speed on a streamed model is roughly
  `SSD bandwidth / ssd_mb_per_token`, so these two numbers tell you whether a
  bigger cache, a lower-bit checkpoint or fewer active experts would help.

What is *not* carried over from Strata: computing missed experts on the CPU.
On NVIDIA that hides PCIe copies behind GPU work; on Apple Silicon the CPU
and GPU read the same memory and the same SSD, so the SSD read is the whole
cost either way.

### Benchmarking a streamed model

```sh
python -m m5engine bench --model DIR --preset strata     # 64-, 3,000- and 14,000-token prompts
python -m m5engine bench --model DIR --preset quick --active-experts 5
```

Results (decode and prefill tok/s, TTFT, peak memory, cache size, hit rate,
SSD MB/token) go into `models/benchmarks.json` and show in the Models tab.
Run the same preset twice: the second run starts from the saved profile and
shows what the warm cache is worth. `python -m m5engine plan --estimate
--model REPO` prints the disk, memory and context estimate for a Hub repo
without downloading it.

### DeepSeek-V4.1-Flash (oMLX backend)

DeepSeek-V4.1-Flash (552B, 40 layers, 384 experts per layer, 6 active, plus
196B of Engram n-gram tables) cannot be loaded by mlx-lm: its `deepseek_v41`
refuses the Engram layers. M5Engine serves it through
[oMLX](https://github.com/jundot/omlx) (Apache-2.0), installed as an optional
backend into the same environment:

```sh
tools/install_deepseek_backend.sh      # Python 3.11+, macOS; pins the oMLX commit in m5engine/deepseek_v41.py
python -m m5engine plan --estimate --model mlx-community/DeepSeek-V4.1-Flash-MLX-2bit
python -m m5engine bench --model <downloaded dir> --preset strata
```

What happens at load (`model_type == deepseek_v41` is detected automatically):

- oMLX's loader reads the checkpoint with **Engram tables on the SSD**
  (`engram_ssd=true`) and **bounded expert residency**: M5Engine asks oMLX
  for the largest share of experts per layer whose load estimate fits the
  memory budget after the activation reserve and a 32K-token KV hold-back
  (`expert_resident_fraction` pins it instead). Misses are positional reads
  from the checkpoint's own safetensors; routing is unchanged, so output
  quality does not depend on the fraction, only speed does.
- DeepSeek's own chat encoder is used (the tokenizer ships no chat template);
  `chat_template_kwargs` accepts `enable_thinking`, `thinking_mode` and
  `reasoning_effort`.
- The KV cache is oMLX's compressed format, measured after load for the
  context plan; `kv_bits` and the prefix cache do not apply.
- The Models tab's expert card shows experts in memory per layer, memory at
  load vs the full model, Engram size/location and the hit rate; benchmarks
  record the hit rate.

Checkpoints and disk: `mlx-community/DeepSeek-V4.1-Flash-MLX-2bit` is 222 GiB,
`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp` 292 GiB (oMLX's own
quantization, better quality). Keep ~300 GB free on the internal SSD. On a
64 GB Mac expect roughly 12 GB of attention/shared weights resident plus
whatever experts fit; nobody has published a 64 GB measurement yet, so treat
decode speed as unknown until you run the benchmark.

Not available on Linux: oMLX's DeepSeek kernels are Metal (`mx.fast.metal_kernel`).

## 5. Managing models (Models tab)

- **Get models**: search MLX text-generation models on Hugging Face, or show the
  curated list. Each result shows the download size, whether it fits in memory,
  streams experts from SSD, or is too large, and an estimated maximum context.
  The estimate comes from `config.json` before downloading; the real plan is
  computed after loading. Downloads go to `--models-dir` (default `./models`),
  show progress, and can be cancelled. Partial files resume when you retry.
- **Benchmark**: runs a fixed workload with a fresh cache (Quick 512 / Standard
  2K / Long-context 16K prompt tokens, greedy decode) and records decode tok/s,
  prefill tok/s, TTFT and peak memory, with the KV precision, draft model and
  SSD expert hit rate. It loads the model with the current Advanced options if
  it isn't already loaded, which replaces the loaded model. Results are kept in
  `<models-dir>/benchmarks.json` (last 20 per model) and shown under History.
- **Check update / Update**: compares the local revision with the latest commit
  on the Hub. Updating fetches only changed files and deletes files the new
  revision no longer has (for example after re-sharding). Models downloaded by
  M5Engine record their source in `.m5engine.json`.
- **Delete**: removes the model from disk (models in `--models-dir`, or all
  revisions of a repo in the Hugging Face cache) and its benchmark history.
  The loaded model can't be deleted or updated until you unload it.

## 6. API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat (streaming, `stop`, `seed`, `top_p/top_k/min_p`, penalties, `chat_template_kwargs`) |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | Loaded model and its `max_context` |
| `POST /v1/models/load` / `unload` | Load (async; add `"sync": true` to wait) / unload |
| `GET /v1/engine/stats` | Hardware, memory, context plan, expert streaming, last request metrics |
| `GET /v1/models/local`, `/v1/models/recommendations` | Installed models (with latest benchmark, disk free) and recommendations |
| `GET /v1/models/hub/search?q=`, `/v1/models/hub/info?repo_id=` | Search the Hub; size, revision, fit and context estimate for a repo |
| `POST /v1/models/download`, `/download/cancel`, `GET /v1/models/download/status` | Hugging Face downloads |
| `POST /v1/models/check_updates` (`{"id"}` optional), `/v1/models/upgrade`, `/v1/models/delete` | Update checks, upgrades and deletion by local model `id` |
| `POST /v1/models/benchmark`, `GET /v1/models/benchmark/status`, `/v1/models/benchmarks?id=` | Run a benchmark (`id`, `prompt_tokens`, `gen_tokens`), its progress, and history |
| `POST /v1/prefix_cache/clear` | Drop cached conversations |

Extra request fields: `prefix_cache` (bool) and `speculative` (bool). The
`usage` object includes `prompt_tokens_details.cached_tokens`, plus an
`m5engine` block with TTFT, prefill/decode tok/s, accepted draft tokens, the
YaRN factor in use and peak memory.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
for chunk in client.chat.completions.create(
    model="local", stream=True,
    messages=[{"role": "user", "content": "Explain YaRN in two sentences."}],
):
    print(chunk.choices[0].delta.content or "" if chunk.choices else "", end="")
```

## 7. Tips

- MoE models with few active parameters (Qwen3-30B-A3B, Qwen3-Next-80B-A3B)
  decode much faster than dense 32B models at similar quality.
- A dense 32B model is bandwidth-bound. Pair it with `draft_model`
  `mlx-community/Qwen3-0.6B-4bit` (same family) for speculative decoding.
- `kv_bits=4` roughly doubles the context compared with 8-bit. It has a small
  quality cost; 8-bit is usually indistinguishable from fp16.
