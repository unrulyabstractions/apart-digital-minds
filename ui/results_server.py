#!/usr/bin/env python3
"""Serve the identity experiment's results as one page.

Reads every captured artifact under out/studies/identity/ (falling back to
sync/out/studies/identity/ for files only pulled from the GPU box), aggregates
them into one payload, and serves a dashboard that shows the whole experiment:
the gate verdicts, the arm-separation and sweep charts, the preflight detail,
and every trial record. Static data only; no model loads.

    uv run python ui/results_server.py

Routes:
    GET /            the dashboard
    GET /data        the aggregated results payload
    GET /run?name=   one trial-run file, for the record viewer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
#: Only the curated working set. sync/ is the untouched byte-verified archive
#: and quarantine/ holds invalidated files; neither is served.
DIRS = [ROOT / "out" / "studies" / "identity"]


def _glob(rel: str) -> dict[str, Path]:
    """Files matching rel under both roots, keyed by name; out/ wins."""
    out: dict[str, Path] = {}
    for base in reversed(DIRS):          # sync first, out overrides
        for p in sorted(base.glob(rel)):
            out[p.name] = p
    return out


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _layer_scan() -> dict:
    """Parse layer_scan.log: '=== case A ===' then 'L12 sep=... t=...' lines."""
    for base in DIRS:
        p = base / "layer_scan.log"
        if not p.exists():
            continue
        cases: dict[str, list] = {}
        current = None
        for line in p.read_text().splitlines():
            m = re.match(r"=== case (\w) ===", line.strip())
            if m:
                current = m.group(1)
                cases[current] = []
                continue
            m = re.match(r"L\s*(\d+)\s+sep=([+-][\d.]+)\s+t=([+-][\d.]+)",
                         line.strip())
            if m and current:
                cases[current].append({"layer": int(m.group(1)),
                                       "sep": float(m.group(2)),
                                       "t": float(m.group(3))})
        if cases:
            return cases
    return {}


def gather() -> dict:
    # provenance: model per stamp, derived from the run logs for artifacts
    # written before the pipeline stamped a model field into everything.
    prov = {}
    for base in DIRS:
        p = base / "PROVENANCE.json"
        if p.exists():
            prov = _load(p) or {}
            break
    stamps = prov.get("stamps", {})

    def with_model(d: dict | None) -> dict | None:
        if d and not d.get("model"):
            d["model"] = stamps.get(d.get("stamp", ""), "unknown")
        return d

    gates = {n: with_model(_load(p)) for n, p in _glob("gates/*.json").items()}
    preflight = {n: with_model(_load(p))
                 for n, p in _glob("preflight/*.json").items()}
    analysis = {n: _load(p) for n, p in _glob("analysis_*.json").items()}
    quality = {n: _load(p) for n, p in _glob("quality_*.json").items()}
    annotations = {}
    for n, ap in _glob("annotations_*.json").items():
        blob = _load(ap) or {}
        annotations.update(blob.get("files", {}))
    verdict_by_file = {}
    for q in quality.values():
        for r in (q or {}).get("reports", []):
            verdict_by_file[r.get("file")] = r.get("verdict")
    runs = []
    for rel in ("caseA_body/*.json", "caseB_ai/*.json"):
        for name, p in _glob(rel).items():
            d = _load(p)
            if not d or "records" not in d:
                continue
            recs = d.get("records", [])
            snippet = (recs[0].get("seed_prompt", "") if recs else "")[:70]
            runs.append({"name": name, "case": d.get("case"),
                         "stamp": d.get("stamp"),
                         "seed_index": d.get("seed_index"),
                         "seed_snippet": snippet,
                         "quality": verdict_by_file.get(name),
                         "partial": bool(d.get("partial")),
                         "arm": d.get("arm"),
                         "model": d.get("model")
                         or stamps.get(d.get("stamp", ""), "unknown"),
                         "layer": d.get("layer"),
                         "conditions": d.get("conditions"),
                         "n_records": len(d["records"])})
    scans = {k: v for k, v in gates.items() if k.startswith("g1scan")}
    # verdict computed from the data, not asserted
    verdict = {}
    for k, v in scans.items():
        if v:
            verdict[v["case"]] = {"g1_t": v["target"]["t"],
                                  "g1_passed": abs(v["target"]["t"]) >= 2}
    g2 = gates.get("g2_caseA_r3.json") or gates.get("g2_caseA_r2.json")
    if g2:
        verdict["g2"] = {"beta_target": g2["beta_target"],
                         "beta_decoy": g2["beta_decoy"],
                         "gap": g2["beta_gap"],
                         "selective": g2["beta_target"] - g2["beta_decoy"] > 0.01}
    return {"gates": gates, "preflight": preflight, "analysis": analysis,
            "quality": quality, "annotations": annotations,
            "runs": sorted(runs, key=lambda r: (r["case"] or "", r["name"])),
            "layer_scan": _layer_scan(),
            "layer_scan_model": prov.get("layer_scan", "unknown"),
            "provenance": prov, "verdict": verdict,
            "sources": [str(d) for d in DIRS if d.exists()]}


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

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                self._send(200, (HERE / "results.html").read_bytes(), "text/html")
            elif route.path == "/data":
                self._json(gather())
            elif route.path == "/run":
                name = Path((parse_qs(route.query).get("name") or [""])[0]).name
                for rel in (f"caseA_body/{name}", f"caseB_ai/{name}"):
                    for base in DIRS:
                        p = base / rel
                        if p.exists():
                            self._send(200, p.read_bytes(), "application/json")
                            return
                self._json({"error": f"no run named {name!r}"}, 404)
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
