#!/usr/bin/env python3
"""Build the self-contained results page from the live UI and current data.

Takes ui/results.html, replaces its /data fetch with the same payload the
server would serve, embedded inline, and swaps the local-path footer for the
public data pointer. The output is one file with no network dependencies,
suitable for the artifact host and the personal site.

    uv run python scripts/build_artifact.py [--out paper/build/dminds.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "paper" / "build" / "dminds.html"))
    args = ap.parse_args()

    from results_server import gather

    payload = gather()
    payload["sources"] = [
        "huggingface.co/datasets/unrulyabstractions/identity-introspection-weirdchat",
        "github.com/unrulyabstractions/apart-digital-minds"]
    blob = json.dumps(payload).replace("</", "<\\/")

    html = (ROOT / "ui" / "results.html").read_text()
    needle = 'DATA = await (await fetch("/data")).json();'
    assert needle in html, "fetch call not found; UI changed shape"
    html = html.replace(needle, f"DATA = {blob};")
    html = html.replace('"data read from: "', '"data: "')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
