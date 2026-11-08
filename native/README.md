# Native C++/Metal engine (legacy)

This is the original hand-written Metal engine, kept as a starting point for
custom kernels. Kernels will be ported into the MLX engine (`m5engine/`) only
where profiling on Apple Silicon shows MLX is the bottleneck.

It is **not** used by `python -m m5engine`. Known limitations: it uses
synthetic weights when tensors are missing, has a greedy (non-BPE) tokenizer,
supports only 4-bit weights, uses a fixed 8K KV cache, and has no Qwen3 q/k
norms. Don't use it for real models.

```bash
cd native && make
```

`native/models/` holds the tiny synthetic fixtures used by its self-tests.
The web UI in `../web` now targets the MLX server's API.
