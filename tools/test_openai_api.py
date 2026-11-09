#!/usr/bin/env python3
"""End-to-end check of a running M5Engine server against a real loaded model.

    python -m m5engine serve --model mlx-community/Qwen3-0.6B-4bit --port 8080 &
    python tools/test_openai_api.py --port 8080
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    for _ in range(600):
        stats = get(base, "/v1/engine/stats")
        if stats["state"] in ("ready", "error", "empty"):
            break
        time.sleep(1)
    check("model ready", stats["state"] == "ready", stats.get("error") or "")
    if stats["state"] != "ready":
        return 1
    ctx = stats["context"]
    check("context plan", ctx["max_context"] >= 512,
          f"max_context={ctx['max_context']} limited_by={ctx['limited_by']} yarn={ctx['yarn_factor']}")

    models = get(base, "/v1/models")
    check("/v1/models", len(models["data"]) == 1, models["data"][0]["id"])

    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    common = {"max_tokens": a.max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    status, body = post(base, "/v1/chat/completions", {"messages": msgs, **common})
    r = json.loads(body)
    text = r["choices"][0]["message"]["content"]
    check("chat completion", status == 200 and "paris" in text.lower(), repr(text))
    u = r["usage"]
    print(f"      ttft {u['m5engine']['ttft_ms']} ms, decode {u['m5engine']['decode_tps']} tok/s")

    msgs2 = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": "And of Germany?"}]
    status, body = post(base, "/v1/chat/completions", {"messages": msgs2, "stream": True, **common})
    events = [json.loads(l[6:]) for l in body.decode().splitlines() if l.startswith("data: {")]
    streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e.get("choices"))
    usage = next((e["usage"] for e in events if e.get("usage")), {})
    check("streaming", "berlin" in streamed.lower(), repr(streamed))
    check("prefix cache reuse", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0,
          f"cached {usage.get('prompt_tokens_details')}")

    try:
        post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "x " * (ctx["max_context"] + 10)}],
                                            "max_tokens": 1})
        check("context overflow rejected", False)
    except urllib.error.HTTPError as e:
        check("context overflow rejected", e.code == 400)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
