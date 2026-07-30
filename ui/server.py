#!/usr/bin/env python3
"""Talk to a mind in a browser, and watch every tick as it happens.

    python ui/server.py                          # plain subject
    python ui/server.py --mind interceptor       # a stage rewrites its thought
    python ui/server.py --mind bicameral         # a voice speaks into it
    python ui/server.py --model hf:Qwen/Qwen3-0.6B

Nothing in the runtime changed to make this work. A `Sink` already receives
every event the mind emits, so the UI is one sink that forwards to browsers
over server-sent events, plus a page to read them.

Standard library only, to match the package.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import queue
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import demo_models  # noqa: E402
from src import Mind, texts  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.api import Sink  # noqa: E402
from src.api.types import Event  # noqa: E402

PAGE = Path(__file__).resolve().parent / "page.html"


# -- fan events out to every connected browser ------------------------------


class Broadcast:
    """One queue per browser. Push once, everybody watching gets it."""

    def __init__(self) -> None:
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.backlog: list[dict] = []

    def join(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for item in self.backlog:
                q.put(item)
            self._clients.append(q)
        return q

    def leave(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def push(self, payload: dict) -> None:
        with self._lock:
            self.backlog.append(payload)
            del self.backlog[:-2000]
            clients = list(self._clients)
        for q in clients:
            q.put(payload)


class WebSink(Sink):
    """A `Sink` that forwards every event to the browsers. That is the whole
    integration: the runtime has no idea a UI exists."""

    def __init__(self, broadcast: Broadcast):
        self.broadcast = broadcast

    def write(self, event: Event) -> None:
        self.broadcast.push({"type": "event", **event.to_dict()})

    def close(self) -> None:
        pass


# -- the minds you can talk to ----------------------------------------------


def _load_example(name: str):
    """Import an example module by path, so its classes can be reused here."""
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(f"ex_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_mind(kind: str, model: str) -> Mind:
    """One of the shipped architectures. Attach the sink to whatever it
    returns; nothing here knows about the UI."""
    common = dict(console=False)  # out/runs/<mind>/<run-id>/ by default

    if kind == "plain":
        return Mind(
            "ui",
            model,
            system="You are a careful assistant. Answer in two or three sentences.",
            **common,
        )

    if kind == "interceptor":
        ex = _load_example("02_think_interceptor.py")
        mind = Mind(
            "ui",
            model,
            system="You are a helpful assistant. Think inside <think> tags first.",
            ego=demo_models.pick("EGO_MODEL", "ego"),
            ego_system="You speak for a mind. Say what its thinking tells you to say.",
            **common,
        )
        mind.intercept(
            ex.Interceptor(
                "interceptor",
                mind.model(demo_models.pick("INTERCEPTOR_MODEL", "interceptor")),
                system=(
                    "You rewrite another model's private reasoning. "
                    "Reply with the replacement thought only, no tags."
                ),
            )
        )
        return mind

    if kind == "bicameral":
        ex = _load_example("03_bicameral.py")
        mind = Mind(
            "ui",
            model,
            system="You are the half of a mind that thinks. You never speak to anyone.",
            ego=demo_models.pick("EGO_MODEL", "ego"),
            ego_system="You are the half of a mind that speaks.",
            **common,
        )
        voice = ex.Voice(
            "voice",
            mind.model(demo_models.pick("VOICE_MODEL", "voice")),
            system=(
                "You utter one short sentence about the situation. Never address "
                "the user. Never explain yourself."
            ),
        )
        blackboard = ex.Blackboard()
        mind.intercept(voice)
        mind.subject.register(blackboard, "*")
        voice.register(blackboard, "*")
        return mind

    raise SystemExit(f"unknown --mind {kind!r}: use plain, interceptor, or bicameral")


# -- what the inspector shows ------------------------------------------------


def windows(mind: Mind) -> list[dict]:
    """Each agent's context window, so you can see what was edited."""
    out = []
    for module in mind.modules.values():
        transcript = getattr(module, "transcript", None)
        if transcript is None:
            continue
        out.append(
            {
                "module": module.name,
                "messages": [
                    {"role": m.role, "content": m.content, "meta": dict(m.meta)}
                    for m in transcript
                ],
            }
        )
    return out


# -- recordings --------------------------------------------------------------
#
# A run directory is already a recording: meta.json says what the mind was,
# trace.jsonl holds every event in order. Playback is reading them back.


def recordings(base: Path = paths.RUNS) -> list[dict]:
    """Every run on disk, newest first."""
    found = []
    for meta_path in base.glob("*/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        trace = meta_path.parent / "trace.jsonl"
        found.append(
            {
                "mind": meta_path.parent.parent.name,
                "run_id": meta_path.parent.name,
                "started": meta.get("started"),
                "ticks": meta.get("ticks", 0),
                "events": meta.get("events", 0),
                "models": sorted(
                    {m["model"] for m in meta.get("modules", []) if m.get("model")}
                ),
                "size": trace.stat().st_size if trace.exists() else 0,
            }
        )
    return sorted(found, key=lambda r: (r["started"] or "", r["run_id"]), reverse=True)


def recording(mind: str, run_id: str, base: Path = paths.RUNS) -> dict | None:
    """One recording: its meta, and every event it captured."""
    folder = (base / mind / run_id).resolve()
    if base.resolve() not in folder.parents or not folder.is_dir():
        return None  # never read outside out/runs
    meta_path, trace_path = folder / "meta.json", folder / "trace.jsonl"
    if not trace_path.exists():
        return None
    events = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"meta": meta, "events": events}


# -- the server --------------------------------------------------------------


class App:
    def __init__(self, mind: Mind, broadcast: Broadcast, loop: asyncio.AbstractEventLoop):
        self.mind = mind
        self.broadcast = broadcast
        self.loop = loop
        self.busy = False

    def snapshot(self) -> None:
        self.mind.write_meta()  # keep out/.../meta.json current while we run
        self.broadcast.push({"type": "windows", "windows": windows(self.mind)})
        self.broadcast.push({"type": "wiring", "wiring": self.mind.summary()})

    async def prompt(self, text: str, step: bool) -> None:
        self.broadcast.push({"type": "you", "text": text})
        self.mind.prompt(text)
        self.snapshot()
        if step:
            self.broadcast.push({"type": "status", "state": "stepping"})
        else:
            await self.run()

    async def run(self) -> None:
        self.broadcast.push({"type": "status", "state": "thinking"})
        try:
            ticks = await self.mind.process()
        except Exception as exc:  # a wired-wrong mind must say so, not hang
            self.broadcast.push({"type": "error", "text": repr(exc)})
            self.broadcast.push({"type": "status", "state": "idle"})
            return
        for said in texts(self.mind.get_replies()):
            self.broadcast.push({"type": "mind", "text": said})
        self.snapshot()
        self.broadcast.push({"type": "status", "state": "idle", "ticks": ticks})

    async def step(self) -> None:
        self.broadcast.push({"type": "status", "state": "thinking"})
        try:
            turns = await self.mind.process_one()
        except Exception as exc:
            self.broadcast.push({"type": "error", "text": repr(exc)})
            self.broadcast.push({"type": "status", "state": "idle"})
            return
        for said in texts(self.mind.get_replies()):
            self.broadcast.push({"type": "mind", "text": said})
        self.snapshot()
        self.broadcast.push(
            {
                "type": "status",
                "state": "stepping" if turns else "idle",
                "turns": turns,
            }
        )


def make_handler(app: App):
    def spawn(coro) -> None:
        """Run a coroutine on the mind's loop, and report anything it raises.

        `run_coroutine_threadsafe` drops exceptions unless the future is read,
        which is how a broken handler becomes a UI that silently does nothing.
        """
        future = asyncio.run_coroutine_threadsafe(coro, app.loop)

        def report(f):
            exc = f.exception()
            if exc is not None:
                app.broadcast.push({"type": "error", "text": repr(exc)})
                app.broadcast.push({"type": "status", "state": "idle"})

        future.add_done_callback(report)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # the trace is the log
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path  # a query string must not 404
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/state":
                body = json.dumps(
                    {
                        "wiring": app.mind.summary(),
                        "windows": windows(app.mind),
                        "backlog": list(app.broadcast.backlog),
                    },
                    default=str,
                ).encode()
                self._send(200, body, "application/json")
            elif route == "/recordings":
                self._send(200, json.dumps(recordings()).encode(), "application/json")
            elif route == "/recording":
                q = parse_qs(urlsplit(self.path).query)
                found = recording(q.get("mind", [""])[0], q.get("run", [""])[0])
                if found is None:
                    self._send(404, b'{"error":"no such recording"}', "application/json")
                else:
                    self._send(200, json.dumps(found, default=str).encode(),
                               "application/json")
            elif route == "/events":
                self.stream()
            else:
                self._send(404, b"not found", "text/plain")

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = app.broadcast.join()
            try:
                while True:
                    try:
                        payload = q.get(timeout=15)
                        data = json.dumps(payload, default=str)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.broadcast.leave(q)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            route = urlsplit(self.path).path
            if route == "/prompt":
                spawn(app.prompt(payload.get("text", ""), payload.get("step", False)))
            elif route == "/step":
                spawn(app.step())
            elif route == "/run":
                spawn(app.run())
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mind", default="interceptor",
                        choices=["plain", "interceptor", "bicameral"])
    parser.add_argument("--model", default=None,
                        help="model spec for the subject, e.g. hf:Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    demo_models.install()
    model = args.model or demo_models.pick(
        "SUBJECT_MODEL", "subject" if args.mind == "interceptor" else "thinker"
    )

    broadcast = Broadcast()
    mind = build_mind(args.mind, model)
    # One line is the entire integration. The mind is untouched otherwise.
    mind.tracer.add_sink(WebSink(broadcast))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = App(mind, broadcast, loop)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"  mind      {args.mind}")
    print(f"  subject   {model}")
    if mind.ego is not None:
        print(f"  ego       {mind.ego.llm.spec}")
    print(f"  out       {mind.run_path}")
    print(f"\n  open      {url}\n")
    if not args.no_open:
        webbrowser.open(url)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nclosing")
    finally:
        mind.close()


if __name__ == "__main__":
    main()
