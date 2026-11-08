# M5Engine user guide

## 1. Install and start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m m5engine serve --port 8080          # open http://127.0.0.1:8080
```

Models can be a local MLX directory or a Hugging Face repo id; repo ids are
downloaded on first use. The UI lists models found in `./models` and in the
Hugging Face cache, and has one-click downloads for the recommended models.

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
2. loads missing experts from the safetensors files (contiguous per-expert
   slices, parallel reads, `F_NOCACHE`) into an LRU cache,
3. runs the normal quantized expert matmul on just those experts.

During decode it also predicts the next layer's experts from the current
hidden state and starts reading them early. The UI shows the cache hit rate,
prefetch use and read throughput. Speed depends on the hit rate and SSD
bandwidth. A bigger `expert_cache_gb` raises the hit rate but leaves less
memory for context; lower-bit checkpoints (e.g. 3-bit) fit more experts.

## 5. API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat (streaming, `stop`, `seed`, `top_p/top_k/min_p`, penalties, `chat_template_kwargs`) |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | Loaded model and its `max_context` |
| `POST /v1/models/load` / `unload` | Load (async; add `"sync": true` to wait) / unload |
| `GET /v1/engine/stats` | Hardware, memory, context plan, expert streaming, last request metrics |
| `GET /v1/models/local`, `/v1/models/recommendations` | Local models and recommendations |
| `POST /v1/models/download`, `GET /v1/models/download/status` | Hugging Face downloads |
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

## 6. Tips

- MoE models with few active parameters (Qwen3-30B-A3B, Qwen3-Next-80B-A3B)
  decode much faster than dense 32B models at similar quality.
- A dense 32B model is bandwidth-bound. Pair it with `draft_model`
  `mlx-community/Qwen3-0.6B-4bit` (same family) for speculative decoding.
- `kv_bits=4` roughly doubles the context compared with 8-bit. It has a small
  quality cost; 8-bit is usually indistinguishable from fp16.
