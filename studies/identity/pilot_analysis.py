#!/usr/bin/env python3
"""Reduce the pilot's records and judge verdicts to its headline numbers.

Per case: the letter-readout identification matrix and its confabulation
floor, the judge's classification of the resampled free-text answers
(report vs enactment vs none), the judged prose dose-response, and the
calibration the judge earned against WeirdChat's own labels. Consistency
checks: the free cell with no injection must equal the none cell exactly.

    uv run python -m studies.identity.pilot_analysis --stamp pilot1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
P = ROOT / "out" / "studies" / "identity" / "pilot"
IDS = ["bodyA", "humanB", "ocean", "music"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", required=True)
    args = ap.parse_args()

    out = {"stamp": args.stamp, "cases": {}}
    for case in "AB":
        d = json.loads((P / f"pilot_case{case}_{args.stamp}.json").read_text())
        cal = json.loads(
            (P / f"judge_calibration_case{case}_{args.stamp}.json").read_text())
        res = json.loads(
            (P / f"judge_resamples_case{case}_{args.stamp}.json").read_text())
        pro = json.loads(
            (P / f"judge_prose_case{case}_{args.stamp}.json").read_text())

        behavior = {"A": "bodyA", "B": "humanB"}[case]
        letter = {}
        checks = {"free_none_equals_none": True}
        none_row = {}
        for r in d["records"]:
            key = f"{r['arm']}|{r['cell']}|{r['concept'] or '-'}"
            letter[key] = r["letter"]
            if r["cell"] == "none":
                none_row[r["arm"]] = r["letter"]
        for r in d["records"]:
            if r["cell"] == "free" and r["concept"] is None:
                if r["letter"] != none_row.get(r["arm"]):
                    checks["free_none_equals_none"] = False

        # identification by letter argmax over forced cells
        forced_ok, forced_n = 0, 0
        for r in d["records"]:
            if r["cell"] != "forced":
                continue
            forced_n += 1
            lp = {k: v for k, v in r["letter"].items() if k != "none"}
            inj = [cid for cid in IDS
                   if d["menu"][IDS.index(cid)] == r["concept"]]
            if inj and max(lp, key=lp.get) == inj[0]:
                forced_ok += 1

        # resample tallies: correct-label rate per forced concept
        tallies = res.get("tallies") or {}
        resample_id = {}
        for k, v in tallies.items():
            cell, concept = k.split("|", 1)
            correct = v["counts"].get(concept, 0) if cell == "forced" else None
            resample_id[k] = {"n": v["n"], "counts": v["counts"],
                              "correct": correct}

        prose = [{"arm": it["arm"], "concept": it["concept"],
                  "strength": it["strength"], "match": it["judge"]["match"],
                  "confidence": it["judge"].get("confidence")}
                 for it in pro["items"]]

        out["cases"][case] = {
            "model": d["model"], "layer": d["layer"],
            "prompt_id": d["prompt_id"], "strength": d["strength"],
            "steer_scale": d["steer_scale"],
            "judge_calibration": {"n": cal.get("n"),
                                  "agreement": cal.get("agreement"),
                                  "kappa": cal.get("kappa")},
            "base_letter": d["base_letter"],
            "letter_matrix": letter,
            "letter_argmax_correct": f"{forced_ok}/{forced_n}",
            "confabulation_floor": {
                arm: {"P_behavior": none_row[arm][behavior],
                      "P_none": none_row[arm]["none"]}
                for arm in none_row},
            "resample_identification": resample_id,
            "prose_dose_response": prose,
            "consistency": checks,
        }
    dest = P / f"analysis_{args.stamp}.json"
    dest.write_text(json.dumps(out, indent=1))
    for case, c in out["cases"].items():
        print(f"case {case}: letter argmax {c['letter_argmax_correct']}, "
              f"judge agreement {c['judge_calibration']['agreement']}, "
              f"free==none {c['consistency']['free_none_equals_none']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
