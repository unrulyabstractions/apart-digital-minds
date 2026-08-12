#!/usr/bin/env python3
"""Serve the identity experiment's current results as one page.

Reads the curated artifacts under out/studies/identity/ — the contrastive G1
(learned-direction held-out separation) and the concept-menu pilot (records,
judge calibration, judged resamples and prose) — aggregates them into one
payload, and serves the dashboard. Retired data lives in OLD/ and is not
served. Static data only; no model loads.

    uv run python ui/results_server.py

Routes:
    GET /            the dashboard
    GET /data        the aggregated results payload
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
#: Only the curated working set. sync/ is the untouched byte-verified archive,
#: OLD/ holds retired results; neither is served.
BASE = ROOT / "out" / "studies" / "identity"


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def gather() -> dict:
    contrastive = {p.name: _load(p)
                   for p in sorted((BASE / "contrastive").glob("g1_*.json"))}
    verdict = {}
    for g in contrastive.values():
        for case, c in (g or {}).get("cases", {}).items():
            verdict[f"contrastive_{case}"] = {
                "acc": c["best"]["acc"], "p": c["best"]["p_binom"],
                "layer": c["best_layer"], "passed": c["pass"]}

    pilot: dict = {"analysis": None, "records": {}, "prose": {},
                   "resamples": {}, "calibration": {}}
    pdir = BASE / "pilot"
    for p in sorted(pdir.glob("analysis_*.json")):
        pilot["analysis"] = _load(p)
    for case in ("A", "B"):
        for kind, pat in (("records", f"pilot_case{case}_*.json"),
                          ("prose", f"judge_prose_case{case}_*.json"),
                          ("resamples", f"judge_resamples_case{case}_*.json"),
                          ("calibration",
                           f"judge_calibration_case{case}_*.json")):
            hits = sorted(pdir.glob(pat))
            if hits:
                blob = _load(hits[-1])
                if kind == "calibration" and blob:
                    blob.pop("items", None)   # per-item detail not shown
                pilot[kind][case] = blob

    return {"contrastive": contrastive, "verdict": verdict, "pilot": pilot,
            "sources": [str(BASE)]}


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                self._send(200, (HERE / "results.html").read_bytes(),
                           "text/html")
            elif route.path == "/data":
                self._send(200, json.dumps(gather()).encode(),
                           "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    port, server = args.port, None
    while server is None:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler())
        except OSError:
            if port - args.port >= 20:
                raise SystemExit(f"ports {args.port}-{port} are all busy.")
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"  identity results on {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
