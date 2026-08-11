#!/usr/bin/env python3
"""Turn the trial JSONs into the plan's headline numbers.

Per cell, regress the readout on the strength the regulator's choice produced:
`readout = beta*strength + c`. Then

    gain = beta_JS - max(beta_CTX, beta_TXT)
    Delta = gain|W - gain|N

Also reports the resting gap c|W - c|N, the regulator's choice distribution
across arms (a separate result), and the `none`-condition false-positive
shift. Reads whatever runs are present; missing arms or conditions are noted,
never invented.

    uv run python -m studies.identity.analyze --stamp smoke2 --case A
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out" / "studies" / "identity"
CASE_DIR = {"A": OUT / "caseA_body", "B": OUT / "caseB_ai"}
READOUT = {"A": "vantage", "B": "p"}


def slope_intercept(xs, ys):
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None, (sum(ys) / n if n else None)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b, my - b * mx


def regress_cells(records, case):
    """Per cell, (beta, c) of readout on regulator strength across records."""
    key = READOUT[case]
    xs = [r["regulator"]["strength"] for r in records]
    out = {}
    for cell in ("CTX", "TXT", "JS"):
        ys = [r["cells"][cell][key] for r in records if cell in r["cells"]]
        xc = [x for x, r in zip(xs, records) if cell in r["cells"]]
        b, c = slope_intercept(xc, ys)
        out[cell] = {"beta": None if b is None else round(b, 5),
                     "c": None if c is None else round(c, 5), "n": len(ys)}
    return out


def analyze(case: str, stamp: str) -> dict:
    d = CASE_DIR[case]
    arms = {}
    for arm in ("W", "N"):
        path = d / f"run_{arm}_{stamp}.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        recs = blob["records"]
        by_cond = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        free = by_cond.get("free", [])
        cells = regress_cells(free, case) if free else {}
        gain = None
        if cells and all(cells[c]["beta"] is not None for c in ("JS", "CTX", "TXT")):
            gain = round(cells["JS"]["beta"]
                         - max(cells["CTX"]["beta"], cells["TXT"]["beta"]), 5)
        # forced-strength gain: the sweep condition varies strength within a
        # seed, so it yields a per-cell slope even when the free regulator does
        # not vary. decoy is the null the target slope must beat.
        sweep_cells = regress_cells(by_cond["sweep"], case) if "sweep" in by_cond else {}
        gain_sweep = None
        if sweep_cells and all(sweep_cells[c]["beta"] is not None
                               for c in ("JS", "CTX", "TXT")):
            gain_sweep = round(sweep_cells["JS"]["beta"]
                               - max(sweep_cells["CTX"]["beta"],
                                     sweep_cells["TXT"]["beta"]), 5)
        decoy_cells = regress_cells(by_cond["decoy"], case) if "decoy" in by_cond else {}
        # regulator choice distribution and the none-condition floor
        choices = Counter(r["regulator"]["action"] for r in free)
        none_shift = None
        if "none" in by_cond:
            key = READOUT[case]
            none_js = [r["cells"]["JS"][key] for r in by_cond["none"]
                       if "JS" in r["cells"]]
            none_shift = round(sum(none_js) / len(none_js), 5) if none_js else None
        arms[arm] = {"cells": cells, "gain": gain,
                     "sweep_cells": sweep_cells, "gain_sweep": gain_sweep,
                     "decoy_cells": decoy_cells,
                     "regulator_choices": dict(choices),
                     "none_floor": none_shift, "n_free": len(free)}

    result = {"case": case, "stamp": stamp, "arms": arms}
    if "W" in arms and "N" in arms and arms["W"]["gain"] is not None \
            and arms["N"]["gain"] is not None:
        result["Delta"] = round(arms["W"]["gain"] - arms["N"]["gain"], 5)
        cw = arms["W"]["cells"]["JS"]["c"]
        cn = arms["N"]["cells"]["JS"]["c"]
        if cw is not None and cn is not None:
            result["resting_gap"] = round(cw - cn, 5)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--case", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    for case in args.case:
        res = analyze(case, args.stamp)
        path = OUT / f"analysis_case{case}_{args.stamp}.json"
        path.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
