#!/usr/bin/env python3
"""Serve the J-space workspace viewer, live.

The model and the lens are loaded once at startup and held for the process. Each
message runs the four-part protocol (subject, regulator, actor, introspector)
live and returns the result. The subject's readout is computed up front; the
other parts' readouts are computed on demand when a viewer turns their panel on.

    python ui/jspace_server.py --model hf:Qwen/Qwen2.5-7B-Instruct

A live message on a 7B is minutes of compute and holds ~19 GB. That is the cost
of live; there is no way around it on this hardware.

Routes:
    GET  /                          the chat page
    POST /chat                      {message} -> the actor's reply and the four-part detail
    POST /reset                     start a new conversation
    GET  /analyze?part=            one part's full J-space turn (positions, per token, stats)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import get_llm  # noqa: E402
from src.api.types.messages import ChatMessage  # noqa: E402
from src.dminds import workspace as wk  # noqa: E402
from src.dminds.llm import jspace  # noqa: E402

HERE = Path(__file__).resolve().parent


class Live:
    """The loaded model and lens, and the last trial's state. One at a time."""

    def __init__(self, model: str, layer: int):
        self.model_spec = model
        self.bare = model.split(":", 1)[1]
        self.layer = layer
        self.lock = threading.Lock()
        self.llm = None
        self.lens = None
        self.last = None      # the state dict from the most recent turn
        self.rng = random.Random(0)
        self.conversation = list(wk.opening())

    def load(self) -> None:
        if self.llm is not None:
            return
        # Reading the cached weights from disk into memory, not downloading them:
        # the checkpoint and the lens live in the HF cache after first use. This
        # runs once per server process and is held for its lifetime, so only a
        # restart pays it again.
        print(f"  loading {self.bare} from the HF cache into memory "
              f"(~19 GB RAM; already downloaded) ...", flush=True)
        self.llm = get_llm(self.model_spec)
        self.llm.load()
        self.lens = jspace.fetch_lens(self.bare)
        # pre-warm the 2 GB unembedding so the first message does not pay for it
        jspace._unembed_and_norm(self.llm)
        print("  ready. the model stays loaded; later messages do not reload.", flush=True)

    def chat(self, message: str) -> dict:
        with self.lock:
            self.load()
            context = self.conversation + [ChatMessage("user", message)]
            record, state = asyncio.run(
                wk.run_on_context(self.llm, self.lens, context, self.layer, self.rng))
            # the actor's reply is what the conversation carries forward
            self.conversation = context + [ChatMessage("assistant", record["actor"])]
            self.last = state
            record["turn"] = (len(self.conversation) - 1) // 2
            return record

    def reset(self) -> None:
        with self.lock:
            self.conversation = list(wk.opening())
            self.last = None

    def analyze(self, part: str) -> dict:
        with self.lock:
            if self.last is None:
                return {"positions": {}, "per_token": [], "stats": {}}
            return wk.analyze_part(self.llm, self.lens, self.last, part, self.layer)


def make_handler(live: Live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                self._send(200, (HERE / "jspace.html").read_bytes(), "text/html")
            elif route.path == "/analyze":
                part = parse_qs(route.query).get("part", [""])[0]
                self._json({"part": part, "analysis": live.analyze(part)})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            route = urlsplit(self.path)
            if route.path == "/reset":
                live.reset()
                self._json({"ok": True})
                return
            if route.path != "/chat":
                self._send(404, b"not found", "text/plain")
                return
            message = (self._body().get("message") or "").strip()
            if not message:
                self._json({"error": "empty message"}, 400)
                return
            try:
                self._json(live.chat(message))
            except Exception as exc:  # a live run can fail on memory; say so
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--warm", action="store_true", help="load the model now")
    args = parser.parse_args()

    live = Live(args.model, args.layer)
    if args.warm:
        live.load()

    port, server = args.port, None
    while server is None:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(live))
        except OSError:
            if port - args.port >= 20:
                raise SystemExit(f"ports {args.port}-{port} are all busy.")
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"  live J-space viewer on {url}  (model loads on first message)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
