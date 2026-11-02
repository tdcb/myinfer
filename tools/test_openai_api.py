#!/usr/bin/env python3
"""
End-to-End OpenAI API & SSE Streaming Verification Suite for M5Engine
Tests:
  1. GET /v1/models (Dense & MoE model discovery)
  2. POST /v1/chat/completions (Dense cold prefill vs. Radix Prefix KV-Cache hit TTFT)
  3. POST /v1/chat/completions with stream=True (Server-Sent Events streaming)
  4. POST /v1/chat/completions with MoE model (Dynamic adaptation to sparse Top-K routing + hot expert pinning)
  5. GET /v1/engine/stats (Apple M5 Pro hardware & memory telemetry)
"""

import json
import subprocess
import time
import urllib.request
import sys


def http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_sse(url: str, payload: dict) -> list:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    chunks = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                chunks.append(json.loads(data_str))
    return chunks


def main():
    port = 18092
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["./m5engine", "serve", "--port", str(port), "--model", "dense"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for server startup
        for _ in range(30):
            try:
                stats = http_get(f"{base_url}/health")
                if stats.get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.1)

        print("=== 1. Testing GET /v1/models ===")
        models = http_get(f"{base_url}/v1/models")
        print(f"Models available: {[m['id'] for m in models['data']]}")
        assert len(models["data"]) >= 2

        system_prompt = (
            "You are an Apple Silicon M5 Pro optimization assistant. "
            "Always leverage 128-byte cache-line coalesced Q4_G64 kernels, "
            "Radix Prefix KV-Cache reuse, Dense Speculative MTP, and MoE Fused Sparse Routing."
        ) * 4

        print("\n=== 2. Testing Dense Chat Completion (Cold vs. Warm Radix Prefix KV Hit) ===")
        res_cold = http_post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": "qwen3.8-27b-m5-q4",
                "max_tokens": 24,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Explain unified memory zero-copy."},
                ],
            },
        )
        t_cold = res_cold["m5_hardware_telemetry"]
        print(f"Cold Turn -> TTFT: {t_cold['ttft_ms']} ms | Cached: {t_cold['prefix_cached_tokens']} toks | Decode: {t_cold['decode_tokens_per_sec']} tok/s | Mean Tok/Pass: {t_cold['mean_tokens_per_pass']}x")

        res_warm = http_post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": "qwen3.8-27b-m5-q4",
                "max_tokens": 24,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Now explain why fans stay quiet."},
                ],
            },
        )
        t_warm = res_warm["m5_hardware_telemetry"]
        print(f"Warm Turn -> TTFT: {t_warm['ttft_ms']} ms | Cached: {t_warm['prefix_cached_tokens']} toks | Decode: {t_warm['decode_tokens_per_sec']} tok/s | Mean Tok/Pass: {t_warm['mean_tokens_per_pass']}x")
        assert t_warm["prefix_cached_tokens"] > 0

        print("\n=== 3. Testing SSE Streaming (/v1/chat/completions stream=True) ===")
        chunks = http_post_sse(
            f"{base_url}/v1/chat/completions",
            {
                "model": "qwen3.8-27b-m5-q4",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "Stream response tokens."}],
            },
        )
        streamed_text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        print(f"Received {len(chunks)} SSE chunks: {streamed_text[:80]}...")
        assert len(chunks) > 0

        print("\n=== 4. Testing Dynamic Adaptation to Mixture-of-Experts (MoE) ===")
        res_moe = http_post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": "qwen3.8-moe-16x4-m5-q4",
                "max_tokens": 24,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Route across sparse MoE experts on M5 Pro."},
                ],
            },
        )
        t_moe = res_moe["m5_hardware_telemetry"]
        print(f"MoE Turn  -> Arch: {t_moe['architecture']} | Active Experts/Tok: {t_moe['active_experts_per_tok']} | Pinned Hot Experts: {t_moe['pinned_hot_experts']} | Decode: {t_moe['decode_tokens_per_sec']} tok/s")
        assert "Mixture-of-Experts" in t_moe["architecture"]
        assert t_moe["active_experts_per_tok"] == 4

        print("\n=== 5. Testing GET /v1/engine/stats ===")
        stats = http_get(f"{base_url}/v1/engine/stats")
        print(json.dumps(stats["hardware"], indent=2))
        print("\nALL PYTHON OPENAI CLIENT TESTS PASSED!")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
