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
import minds  # noqa: E402
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

    def reset(self) -> None:
        """Forget the session so far. Sent when a different mind is chosen."""
        with self._lock:
            self.backlog.clear()
            clients = list(self._clients)
        for q in clients:
            q.put({"type": "reset"})

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


def build_mind(kind: str, model: str) -> Mind:
    """Whichever mind `minds/<kind>.py` defines.

    The server knows nothing about any particular architecture. Add a file to
    `minds/` and it appears here and in the picker.
    """
    module = minds.load(kind)
    extra = {
        kwarg: demo_models.pick(f"{kwarg.upper()}_MODEL", role)
        for kwarg, role in getattr(module, "ROLES", {}).items()
    }
    return module.build(model, console=False, **extra)


# -- what the inspector shows ------------------------------------------------


def windows(mind: Mind) -> list[dict]:
    """Every context window, in pipeline order, each naming the one before it.

    The mind knows its own shape, so it can say which window a given one was
    derived from. That lets a reader compare two adjacent windows and see what
    a stage did, whatever the stage was and whatever kind of edit it made.
    Nothing here knows about any particular architecture.
    """
    # The window in flight is held by the ends of the pipeline. A stage
    # transforms it in passing and keeps its own separate conversation, so a
    # stage is never the thing a later window should be compared against.
    carriers = [m for m in (mind.subject, mind.ego) if m is not None]
    chain = [m for m in (mind.subject, *mind.stages, mind.ego) if m is not None]
    placed = {m.name for m in chain}

    def described(module, upstream, carries):
        return {
            "module": module.name,
            "upstream": upstream,
            "carries": carries,
            "messages": [
                {"role": m.role, "content": m.content, "meta": dict(m.meta)}
                for m in module.transcript
            ],
        }

    out, seen_carrier = [], None
    for module in chain:
        if getattr(module, "transcript", None) is None:
            continue
        if module in carriers:
            # Until something has arrived, this window is the module's own
            # rather than a copy of the one upstream. Comparing them then
            # would report two different system prompts as an edit.
            out.append(described(module, seen_carrier if module.received else None, True))
            seen_carrier = module.name
        else:
            out.append(described(module, None, False))

    for module in mind.modules.values():
        if module.name not in placed and getattr(module, "transcript", None) is not None:
            out.append(described(module, None, False))
    return out


# -- recordings --------------------------------------------------------------
#
# A run directory is already a recording: meta.json says what the mind was,
# trace.jsonl holds every event in order. Playback is reading them back.


def recordings(base: Path = paths.RUNS) -> list[dict]:
    """Every run on disk that has something in it, newest first.

    A run where nothing happened is skipped. Opening one shows an empty
    screen, which is a dead end rather than a recording.
    """
    found = []
    for meta_path in base.glob("*/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not meta.get("events"):
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
    """The one mind currently on screen, and the things you can do to it.

    A session starts with no mind at all. The browser asks for one by name, so
    which architecture you talk to is a choice made in the page rather than on
    the command line, and you can put the current one away and pick another
    without restarting anything.
    """

    def __init__(
        self,
        broadcast: Broadcast,
        loop: asyncio.AbstractEventLoop,
        model: str | None = None,
    ):
        self.mind: Mind | None = None
        self.kind: str | None = None
        self.broadcast = broadcast
        self.loop = loop
        self.model = model

    def select(self, kind: str, model: str | None = None) -> None:
        """Put away whatever is running and build `minds/<kind>.py` instead."""
        if self.mind is not None:
            self.mind.close()
        self.broadcast.reset()
        module = minds.load(kind)
        spec = model or self.model or demo_models.pick(
            "SUBJECT_MODEL", "subject" if getattr(module, "ROLES", {}) else "thinker"
        )
        self.mind = build_mind(kind, spec)
        self.kind = kind
        # One line is the entire integration. The mind is untouched otherwise.
        self.mind.tracer.add_sink(WebSink(self.broadcast))
        self.broadcast.push({"type": "chose", **self.describe()})
        self.snapshot()

    def describe(self) -> dict:
        """What is on screen, for the header and for a page that just loaded."""
        if self.mind is None:
            return {"mind": None}
        module = minds.load(self.kind)
        return {
            "mind": self.kind,
            "title": getattr(module, "TITLE", self.kind),
            "about": getattr(module, "ABOUT", ""),
            "subject": self.mind.subject.llm.spec,
            "ego": self.mind.ego.llm.spec if self.mind.ego else None,
            "run": str(self.mind.run_path),
        }

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
            # The page changes as you edit it, and a cached copy of an older
            # one against a newer server is a bug that looks like the UI.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path  # a query string must not 404
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/state":
                body = json.dumps(
                    {
                        "chose": app.describe(),
                        "wiring": app.mind.summary() if app.mind else None,
                        "windows": windows(app.mind) if app.mind else [],
                        "backlog": list(app.broadcast.backlog),
                    },
                    default=str,
                ).encode()
                self._send(200, body, "application/json")
            elif route == "/minds":
                self._send(200, json.dumps(minds.available()).encode(),
                           "application/json")
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
            if route == "/select":
                try:
                    app.select(payload.get("mind", ""), payload.get("model") or None)
                except Exception as exc:
                    self._send(400, json.dumps({"error": repr(exc)}).encode(),
                               "application/json")
                    return
            elif app.mind is None:
                self._send(409, b'{"error":"no mind chosen"}', "application/json")
                return
            elif route == "/prompt":
                spawn(app.prompt(payload.get("text", ""), payload.get("step", False)))
            elif route == "/step":
                spawn(app.step())
            elif route == "/run":
                spawn(app.run())
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mind", default=None, choices=minds.names(),
                        help="skip the chooser and start on this mind")
    parser.add_argument("--model", default=None,
                        help="model spec for the subject, e.g. hf:Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    demo_models.install()

    broadcast = Broadcast()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = App(broadcast, loop, args.model)
    if args.mind:
        app.select(args.mind, args.model)

    handler = make_handler(app)
    port, server = args.port, None
    while server is None:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError:
            # Something else is on that port, and it is not ours to stop.
            if port - args.port >= 20:
                raise SystemExit(
                    f"ports {args.port}-{port} are all busy. Pass --port."
                )
            port += 1
    if port != args.port:
        print(f"  note      {args.port} was busy, using {port}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    if app.mind is None:
        print(f"  minds     {', '.join(minds.names())}")
        print("  pick one in the page, or pass --mind to skip the chooser")
    else:
        print(f"  mind      {app.kind}")
        print(f"  subject   {app.mind.subject.llm.spec}")
        if app.mind.ego is not None:
            print(f"  ego       {app.mind.ego.llm.spec}")
        print(f"  out       {app.mind.run_path}")
    print(f"\n  open      {url}\n")
    if not args.no_open:
        webbrowser.open(url)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nclosing")
    finally:
        if app.mind is not None:
            app.mind.close()


if __name__ == "__main__":
    main()
