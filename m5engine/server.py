"""OpenAI-compatible HTTP server and control plane for the web UI."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .engine import Engine, EngineError, GenerationParams, LoadOptions
from .models import Downloader, list_local_models, recommendations

log = logging.getLogger("m5engine.server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class App:
    def __init__(self, engine: Engine, models_dir: Path, default_options: Optional[dict] = None):
        self.engine = engine
        self.models_dir = models_dir
        self.downloader = Downloader(models_dir)
        self.default_options = default_options or {}
        self.started = time.time()

    def load_async(self, body: dict) -> None:
        opts = LoadOptions.from_dict({**self.default_options, **body})
        self.engine.state = "loading"

        def run():
            try:
                self.engine.load(opts)
            except Exception as e:
                log.error("Load failed: %s", e)

        threading.Thread(target=run, daemon=True).start()


def _params(body: dict) -> GenerationParams:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    return GenerationParams(
        max_tokens=int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024),
        temperature=float(body.get("temperature", 0.7)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0)),
        min_p=float(body.get("min_p", 0.0)),
        repetition_penalty=body.get("repetition_penalty"),
        presence_penalty=body.get("presence_penalty"),
        frequency_penalty=body.get("frequency_penalty"),
        stop=list(stop),
        seed=body.get("seed"),
        use_prefix_cache=bool(body.get("prefix_cache", True)),
        use_draft=bool(body.get("speculative", True)),
    )


def make_handler(app: App):
    engine = app.engine

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "M5Engine/0.2"

        def log_message(self, fmt, *args):
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ---------------------------------------------------------- helpers
        def _send(self, status: int, payload: Any, ctype: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
            self._send(status, {"error": {"message": message, "type": etype, "code": status}})

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError as e:
                raise EngineError(f"Invalid JSON: {e}")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ----------------------------------------------------------- routes
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/health":
                    return self._send(200, {"status": "ok", "state": engine.state})
                if path == "/v1/models":
                    data = []
                    if engine.loaded:
                        data.append({"id": engine.options.model, "object": "model",
                                     "created": int(app.started), "owned_by": "local",
                                     "max_context": engine.plan.max_context})
                    return self._send(200, {"object": "list", "data": data})
                if path == "/v1/engine/stats":
                    return self._send(200, engine.info())
                if path == "/v1/models/local":
                    return self._send(200, {"data": list_local_models(app.models_dir)})
                if path == "/v1/models/download/status":
                    return self._send(200, {"data": app.downloader.status()})
                if path == "/v1/models/recommendations":
                    return self._send(200, {"budget_gb": round(engine.budget_bytes() / 1024**3, 1),
                                            "data": recommendations(engine.budget_bytes(), engine.hw.total_ram_bytes)})
                return self._error(404, f"Unknown path {path}")
            except Exception as e:  # pragma: no cover
                log.exception("GET %s failed", path)
                return self._error(500, str(e), "server_error")

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                body = self._body()
                if path == "/v1/chat/completions":
                    return self._completion(body, chat=True)
                if path == "/v1/completions":
                    return self._completion(body, chat=False)
                if path == "/v1/models/load":
                    if body.get("sync"):
                        return self._send(200, engine.load(LoadOptions.from_dict({**app.default_options, **body})))
                    app.load_async(body)
                    return self._send(202, {"state": "loading", "model": body.get("model")})
                if path == "/v1/models/unload":
                    engine.unload()
                    return self._send(200, {"state": engine.state})
                if path == "/v1/models/download":
                    repo = body.get("repo_id")
                    if not repo:
                        raise EngineError("'repo_id' is required")
                    load = body.get("auto_load")
                    on_done = (lambda job: app.load_async({**body, "model": job["path"]})) if load else None
                    return self._send(202, app.downloader.start(repo, on_done))
                if path == "/v1/prefix_cache/clear":
                    engine.prefix.clear()
                    return self._send(200, {"entries": 0})
                return self._error(404, f"Unknown path {path}")
            except EngineError as e:
                return self._error(e.status, str(e))
            except BrokenPipeError:
                pass
            except Exception as e:
                log.exception("POST %s failed", path)
                return self._error(500, str(e), "server_error")

        # ------------------------------------------------------ completions
        def _completion(self, body: dict, chat: bool) -> None:
            if not engine.loaded:
                return self._error(503, f"No model loaded (state: {engine.state})", "model_not_loaded")
            params = _params(body)
            if chat:
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise EngineError("'messages' must be a non-empty list")
                tokens = engine.encode_chat(messages, body.get("chat_template_kwargs"), body.get("tools"))
            else:
                prompt = body.get("prompt")
                if isinstance(prompt, list):
                    prompt = "".join(prompt)
                if not isinstance(prompt, str):
                    raise EngineError("'prompt' must be a string")
                tokens = engine.encode_text(prompt)
            chunks = engine.generate(tokens, params)
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            model = engine.options.model
            obj = "chat.completion" if chat else "text_completion"

            def usage(m):
                return {
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.prompt_tokens + m.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": m.cached_tokens},
                    "m5engine": {"ttft_ms": round(m.ttft_ms, 1), "prompt_tps": round(m.prompt_tps, 1),
                                 "decode_tps": round(m.decode_tps, 1), "draft_accepted": m.draft_accepted,
                                 "yarn_factor": m.yarn_factor, "peak_memory_gb": m.peak_memory_gb},
                }

            if not body.get("stream"):
                text, finish, metrics = [], None, engine.last
                for c in chunks:
                    text.append(c.text)
                    finish = c.finish_reason or finish
                    metrics = c.metrics or metrics
                content = "".join(text)
                choice = ({"index": 0, "message": {"role": "assistant", "content": content},
                           "finish_reason": finish} if chat else
                          {"index": 0, "text": content, "finish_reason": finish})
                return self._send(200, {"id": rid, "object": obj, "created": created, "model": model,
                                        "choices": [choice], "usage": usage(metrics)})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            obj_chunk = "chat.completion.chunk" if chat else "text_completion"

            def event(payload: dict) -> None:
                self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                self.wfile.flush()

            def choice(text: str, finish=None, first=False):
                if chat:
                    delta = {"content": text}
                    if first:
                        delta["role"] = "assistant"
                    return {"index": 0, "delta": delta, "finish_reason": finish}
                return {"index": 0, "text": text, "finish_reason": finish}

            first = True
            metrics = engine.last
            try:
                for c in chunks:
                    metrics = c.metrics or metrics
                    if c.text or c.finish_reason or first:
                        event({"id": rid, "object": obj_chunk, "created": created, "model": model,
                               "choices": [choice(c.text, c.finish_reason, first)]})
                        first = False
                if (body.get("stream_options") or {}).get("include_usage", True):
                    event({"id": rid, "object": obj_chunk, "created": created, "model": model,
                           "choices": [], "usage": usage(metrics)})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                chunks.close()
            except Exception as e:
                log.exception("Generation failed")
                try:
                    event({"error": {"message": str(e), "type": "server_error"}})
                except OSError:
                    pass

    return Handler


def serve(app: App, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    log.info("M5Engine listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
