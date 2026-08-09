#!/usr/bin/env python3
"""Serve the J-space workspace viewer over recorded runs.

Replay, not live. A recorded run already holds every part's J-space readout at
every token position, so the viewer needs no model in memory and renders
instantly. Live per-message J-space on a 7B is minutes per turn and does not
belong in a browser.

    python ui/jspace_server.py            # opens http://127.0.0.1:8770

Routes:
    GET /                         the viewer page
    GET /runs                     recorded runs, with counts and situations
    GET /records?slug=<slug>      one run's records.json plus its summary
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds import paths  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKSPACE = paths.OUT / "studies" / "workspace"


def runs() -> list[dict]:
    """Every recorded run, newest-looking first, with a one-line summary."""
    out = []
    if not WORKSPACE.exists():
        return out
    for folder in sorted(WORKSPACE.iterdir()):
        records = folder / "records.json"
        if not records.is_file():
            continue
        try:
            data = json.loads(records.read_text())
        except json.JSONDecodeError:
            continue
        summary = folder / "summary.json"
        layer = None
        if summary.is_file():
            layer = json.loads(summary.read_text()).get("layer")
        situations = sorted({r.get("situation", "?") for r in data})
        out.append({"slug": folder.name,
                    "model": folder.name.replace("hf-", "").replace("-", "/", 1),
                    "trials": len(data), "situations": situations, "layer": layer})
    return out


def records_for(slug: str) -> dict:
    folder = WORKSPACE / slug
    data = json.loads((folder / "records.json").read_text())
    summary = folder / "summary.json"
    write_check = None
    if summary.is_file():
        write_check = json.loads(summary.read_text()).get("write_check")
    if write_check is None and data:
        write_check = data[0].get("write_check")
    return {"slug": slug, "records": data, "write_check": write_check}


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                self._send(200, (HERE / "jspace.html").read_bytes(), "text/html")
            elif route.path == "/runs":
                self._json(runs())
            elif route.path == "/records":
                slug = parse_qs(route.query).get("slug", [""])[0]
                folder = WORKSPACE / slug
                if not slug or not (folder / "records.json").is_file():
                    self._json({"error": "no such run"}, 404)
                else:
                    self._json(records_for(slug))
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    port, server = args.port, None
    while server is None:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler())
        except OSError:
            if port - args.port >= 20:
                raise SystemExit(f"ports {args.port}-{port} are all busy.")
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"  J-space viewer on {url}  ({len(runs())} recorded run(s))")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
