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
    GET  /                      the viewer page
    GET  /situations            the built-in scenarios (name + turns)
    POST /ask                   {situation} or {turns:[...]} -> a trial record
    GET  /readout?part=&position=   one part's readout for the last trial
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
sys.path.insert(0, str(ROOT / "studies"))

from scenarios import SCENARIOS  # noqa: E402
from src import get_llm  # noqa: E402
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
        self.last = None      # the state dict from the most recent trial
        self.rng = random.Random(0)

    def load(self) -> None:
        if self.llm is not None:
            return
        print(f"  loading {self.bare} and its lens (this holds ~19 GB) ...", flush=True)
        self.llm = get_llm(self.model_spec)
        self.llm.load()
        self.lens = jspace.fetch_lens(self.bare)
        # pre-warm the 2 GB unembedding so the first message does not pay for it
        jspace._unembed_and_norm(self.llm)
        print("  ready.", flush=True)

    def ask(self, turns: list[str]) -> dict:
        with self.lock:
            self.load()
            record, state = asyncio.run(
                wk.run_trial(self.llm, self.lens, turns, self.layer, self.rng))
            self.last = state
            record["write_check"] = None
            return record

    def readout(self, part: str, position: str) -> list:
        with self.lock:
            if self.last is None:
                return []
            return wk.readout_for(self.llm, self.lens, self.last, part,
                                  position, self.layer)


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
            elif route.path == "/situations":
                self._json([{"name": s.name, "about": s.about, "turns": s.turns}
                            for s in SCENARIOS.values()])
            elif route.path == "/readout":
                q = parse_qs(route.query)
                part = q.get("part", [""])[0]
                position = q.get("position", ["assistant"])[0]
                self._json({"part": part, "position": position,
                            "readout": live.readout(part, position)})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            route = urlsplit(self.path)
            if route.path != "/ask":
                self._send(404, b"not found", "text/plain")
                return
            body = self._body()
            if "situation" in body and body["situation"] in SCENARIOS:
                turns = SCENARIOS[body["situation"]].turns
                label = body["situation"]
            else:
                turns = body.get("turns") or [body.get("message", "").strip()]
                label = "live"
            try:
                record = live.ask(turns)
            except Exception as exc:  # a live run can fail on memory; say so
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            record["situation"] = label
            record["trial"] = 0
            self._json(record)

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
