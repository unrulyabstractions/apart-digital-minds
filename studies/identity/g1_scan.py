#!/usr/bin/env python3
"""Fast, generation-free G1 scan: do the W and N seeds separate on the target?

Loads the model and lens once and, for each case, projects every W and N seed
reply onto the case's target direction. Reports the arm means, their gap, and a
Welch t-statistic. No generation, so it runs in seconds per seed and gives a
robust read on whether the behavior is carried by the token set at all.

    .venv/bin/python -m studies.identity.g1_scan --model hf:Qwen/Qwen3.6-27B \
        --n-seeds 24 --stamp scan1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api.types.messages import ChatMessage  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from . import rig as R  # noqa: E402
from . import tokens as T  # noqa: E402
from .directions import proj, set_direction  # noqa: E402
from .seeds import load_seeds  # noqa: E402

OUT = ROOT / "out" / "studies" / "identity" / "gates"


def welch_t(a, b):
    import statistics as st
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = (va / len(a) + vb / len(b)) ** 0.5 or 1e-9
    return (ma - mb) / se


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="hf:Qwen/Qwen3.6-27B")
    ap.add_argument("--cases", nargs="+", default=["A", "B"])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--seed-model", default="qwen/qwen3.6-27b")
    ap.add_argument("--stamp", required=True)
    args = ap.parse_args()

    bare = args.model.split(":", 1)[1]
    verified = T.verify(AutoTokenizer.from_pretrained(bare))
    llm, lens, layer = R.load_model(args.model, args.layer)
    print(f"  loaded {bare}; layer {layer}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        target = set_direction(llm, lens, verified[case]["target"], layer)
        decoy = set_direction(llm, lens, verified[case]["decoy"], layer)
        seeds = load_seeds(case, args.seed_model, n=args.n_seeds)
        pW, pN, pWd, pNd = [], [], [], []
        for s in seeds:
            for arm, pt, pd in (("W", pW, pWd), ("N", pN, pNd)):
                msgs = [ChatMessage("system", ""),
                        ChatMessage("user", s["prompt"]),
                        ChatMessage("assistant", s[arm])]
                pt.append(proj(llm, target, msgs))
                pd.append(proj(llm, decoy, msgs))
        import statistics as st
        rep = {
            "case": case, "stamp": args.stamp, "layer": layer,
            "model": llm.model,
            "n_seeds": len(seeds),
            "target": {"mean_W": round(st.mean(pW), 4),
                       "mean_N": round(st.mean(pN), 4),
                       "sep": round(st.mean(pW) - st.mean(pN), 4),
                       "t": round(welch_t(pW, pN) or 0, 3)},
            "decoy": {"mean_W": round(st.mean(pWd), 4),
                      "mean_N": round(st.mean(pNd), 4),
                      "sep": round(st.mean(pWd) - st.mean(pNd), 4),
                      "t": round(welch_t(pWd, pNd) or 0, 3)},
        }
        (OUT / f"g1scan_case{case}_{args.stamp}.json").write_text(
            json.dumps(rep, indent=1))
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
