#!/usr/bin/env python3
"""Drive the identity introspection pipeline: preflight, gates, then trials.

Order is enforced. Preflight verifies the token sets and the letter tokens.
Gate G1 asks whether the W and N seeds separate on the target tokens at all;
G2 asks whether steering moves the readout more than a decoy does; G3 sizes
the run from G2's effect. Only then do the free/none/decoy trials run, one
JSON per case per arm, timestamped, never overwritten.

    uv run python -m studies.identity.run --model hf:Qwen/Qwen2.5-3B-Instruct \
        --cases A --n-seeds 3 --smoke
    uv run python -m studies.identity.run --model hf:Qwen/Qwen3.6-27B \
        --cases A B --n-seeds 40 --full-perms

`--smoke` uses few seeds, the fixed-8 permutations, and a relaxed power gate,
for a local shakedown; the real run drops it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api.types import GenOptions  # noqa: E402
from src.api.types.messages import ChatMessage  # noqa: E402
from src.dminds import workspace as wk  # noqa: E402
from src.dminds.llm.steering import steered  # noqa: E402

from . import rig as R  # noqa: E402
from . import tokens as T  # noqa: E402
from .directions import (STRENGTH_UNIT, cosine, emotion_contamination,  # noqa: E402
                         proj, split_half)
from .engine import play_context, run_cells, run_trial, sweep_trial  # noqa: E402
from .seeds import load_seeds  # noqa: E402

OUT = ROOT / "out" / "studies" / "identity"
CASE_DIR = {"A": OUT / "caseA_body", "B": OUT / "caseB_ai"}


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        n = 1
        while path.with_suffix(f".prev{n}.json").exists():
            n += 1
        path.rename(path.with_suffix(f".prev{n}.json"))
    path.write_text(json.dumps(obj, indent=1))


def preflight(llm, lens, cfg, case: str, stamp: str) -> dict:
    """Token verification, split-half, decoy cosine, emotion contamination."""
    from src.dminds.llm import emotions as emo
    vs = {"target": cfg["target_tokens"], "decoy": cfg["decoy_tokens"]}
    try:
        bank = emo.load_emotions(llm.model)
    except Exception:
        bank = None
    report = {
        "case": case, "stamp": stamp, "layer": cfg["layer"],
        "model": llm.model,
        "target": vs["target"], "decoy": vs["decoy"],
        "letter_forms": cfg["letter_forms"], "n_perms": len(cfg["perms"]),
        "base_rates": cfg["base"],
        "split_half": round(split_half(llm, lens, vs["target"], cfg["layer"]), 4),
        "decoy_cosine": round(cosine(cfg["target_dir"], cfg["decoy_dir"]), 4),
        "emotion_contamination":
            emotion_contamination(cfg["target_dir"], bank, 20),
    }
    report["split_half_ok"] = report["split_half"] >= 0.5
    # In this lens's J-space every token direction shares a large common axis,
    # so all set directions sit near cosine 0.69 and none can be "near zero".
    # The decoy's validity is therefore decided by G2 (does steering toward it
    # move the readout less than the target does), not by a static cosine. The
    # cosine is recorded for the record. See PLAN.md deviation 1.
    report["decoy_ok"] = True
    write(OUT / "preflight" / f"case{case}_{stamp}.json", report)
    return report


async def sweep_js(llm, cfg, case: str, ctx, subj, act, direction,
                   strengths) -> dict:
    """The JS cell readout across a strength range on a FIXED context. For G2."""
    out = {}
    key = "vantage" if case == "A" else "p"
    for s in strengths:
        cells = await run_cells(llm, case, ctx, subj, act, direction, s, cfg,
                                only="JS")
        out[str(s)] = cells["JS"][key]
    return out


def slope(xs, ys) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


async def gate_g1(llm, cfg, case: str, seeds: list, stamp: str) -> dict:
    """Arm separation on the target tokens, at the seeded reply."""
    opts = GenOptions(temperature=1.0, max_tokens=240)
    quality: dict = {}
    projW, projN = [], []
    for seed in seeds:
        for arm, acc in (("W", projW), ("N", projN)):
            msgs = [ChatMessage("system", ""),
                    ChatMessage("user", seed["prompt"]),
                    ChatMessage("assistant", seed[arm])]
            acc.append(proj(llm, cfg["target_dir"], msgs))
    sep = sum(projW) / len(projW) - sum(projN) / len(projN)
    rep = {"case": case, "stamp": stamp, "model": llm.model,
           "sep": round(sep, 4),
           "mean_W": round(sum(projW) / len(projW), 4),
           "mean_N": round(sum(projN) / len(projN), 4),
           "n_seeds": len(seeds), "passed": abs(sep) > 0.05}
    write(OUT / "gates" / f"g1_case{case}_{stamp}.json", rep)
    return rep


async def gate_g2(llm, lens, cfg, case: str, seeds: list, stamp: str,
                  strengths=(-3, -2, -1, 0, 1, 2, 3)) -> dict:
    """Steering reaches the readout: target vs decoy slope over a sweep."""
    strengths = list(strengths)
    tgt_curves, dec_curves = [], []
    for seed in seeds:
        # one context per seed; target and decoy sweep the SAME context, so
        # their slopes are comparable and not confounded by resampling.
        q: dict = {}
        ctx, subj, act, _ = await play_context(llm, case, seed, "W", cfg, q)
        tgt_curves.append(await sweep_js(llm, cfg, case, ctx, subj, act,
                                         cfg["target_dir"], strengths))
        dec_curves.append(await sweep_js(llm, cfg, case, ctx, subj, act,
                                         cfg["decoy_dir"], strengths))

    def mean_curve(curves):
        return [sum(c[str(s)] for c in curves) / len(curves) for s in strengths]

    tgt = mean_curve(tgt_curves)
    dec = mean_curve(dec_curves)
    b_t = slope(strengths, tgt)
    b_d = slope(strengths, dec)
    # residual variance of the target curve about its line, for G3
    fit = [b_t * s + (sum(tgt) / len(tgt) - b_t * (sum(strengths) / len(strengths)))
           for s in strengths]
    resid_var = sum((y - f) ** 2 for y, f in zip(tgt, fit)) / len(tgt)
    rep = {"case": case, "stamp": stamp, "model": llm.model,
           "strengths": strengths,
           "target_curve": [round(x, 5) for x in tgt],
           "decoy_curve": [round(x, 5) for x in dec],
           "beta_target": round(b_t, 5), "beta_decoy": round(b_d, 5),
           "beta_gap": round(b_t - b_d, 5),
           "resid_var": round(resid_var, 8),
           "passed": abs(b_t - b_d) > 0.01}
    write(OUT / "gates" / f"g2_case{case}_{stamp}.json", rep)
    return rep


def gate_g3(g2: dict, case: str, stamp: str, floor: int, cap: int) -> dict:
    """Power: n to detect a gain of half the observed target-decoy gap at 80%."""
    import math
    effect = abs(g2["beta_gap"]) / 2 or 1e-6
    sd = math.sqrt(max(g2["resid_var"], 1e-9))
    # z-based n for a slope effect, rough: n ≈ (2.8 · sd / effect)² over the
    # strength spread; a plan-level sizing, not a t-test.
    spread = 3.0  # strengths span -3..3, sd of x ≈ 2.16; use a conservative 3
    n = math.ceil(((2.8 * sd) / (effect * spread)) ** 2)
    n = max(floor, min(cap, n))
    rep = {"case": case, "stamp": stamp, "model": g2.get("model"),
           "effect_half_gap": round(effect, 5),
           "resid_sd": round(sd, 5), "n_seeds": n, "floor": floor, "cap": cap}
    write(OUT / "gates" / f"g3_case{case}_{stamp}.json", rep)
    return rep


async def run_case(llm, lens, cfg, case: str, seeds: list, conditions: list,
                   stamp: str, forced_strengths=(-2, -1, 0, 1, 2)) -> None:
    """Free/none run once per seed; sweep/decoy run once per forced strength.

    The forced conditions vary the strength within a seed, so each yields a
    per-cell regression of the readout on strength even when the free
    regulator does not vary its choice. `decoy` is the same sweep on the sham
    direction: its slope is the null the target slope must beat.
    """
    for arm in ("W", "N"):
        records = []
        for cond in conditions:
            for seed in seeds:
                if cond in ("sweep", "decoy"):
                    # one context, read at every forced strength
                    recs = await sweep_trial(llm, case, seed, arm, cond,
                                             list(forced_strengths), cfg)
                    for rec in recs:
                        rec["seed_prompt_id"] = seed["prompt_id"]
                    records.extend(recs)
                else:
                    rec = await run_trial(llm, lens, case, seed, arm, cond, dict(cfg))
                    rec["seed_prompt_id"] = seed["prompt_id"]
                    records.append(rec)
        write(CASE_DIR[case] / f"run_{arm}_{stamp}.json",
              {"case": case, "arm": arm, "stamp": stamp, "model": llm.model,
               "layer": cfg["layer"], "conditions": conditions,
               "n_seeds": len(seeds), "records": records})


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="hf:Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--cases", nargs="+", default=["A", "B"])
    ap.add_argument("--layer", type=int, default=17,
                    help="lens source layer; -1 auto-picks ~2/3 depth")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-model", default="qwen/qwen3.6-27b")
    ap.add_argument("--conditions", nargs="+",
                    default=["free", "none", "decoy"])
    ap.add_argument("--full-perms", action="store_true")
    ap.add_argument("--n-perms", type=int, default=None,
                    help="cap the permutation subset (smoke speed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-trials", action="store_true",
                    help="run the trial phase even past a failed gate, for "
                         "instrumented validation records; the records stay "
                         "labeled by their conditions and the gate files stand")
    ap.add_argument("--forced-strengths", nargs="+", type=int, default=None)
    ap.add_argument("--stamp", required=True, help="timestamp label (no Date in code)")
    args = ap.parse_args()

    bare = args.model.split(":", 1)[1]
    print(f"  verifying token sets on {bare} ...", flush=True)
    # verify with the tokenizer alone; loading the full model here and again in
    # R.build would hold two 54 GB copies and OOM the card.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bare)
    verified = T.verify(tok)
    for case in args.cases:
        print(f"  case {case}: target={verified[case]['target']}", flush=True)

    print(f"  loading {bare} once for all cases ...", flush=True)
    llm, lens, layer = R.load_model(args.model, args.layer)
    print(f"  loaded; steering layer = {layer}", flush=True)

    for case in args.cases:
        cfg = R.build_cfg(llm, lens, layer, case, verified,
                          full_perms=args.full_perms, n_perms=args.n_perms)
        print(f"\n=== case {case} ===", flush=True)
        pf = preflight(llm, lens, cfg, case, args.stamp)
        print(f"  preflight: split_half={pf['split_half']} "
              f"decoy_cos={pf['decoy_cosine']} "
              f"forms={pf['letter_forms']}", flush=True)
        if not (pf["split_half_ok"] and pf["decoy_ok"]) and not args.smoke \
                and not args.force_trials:
            print("  PREFLIGHT FAILED; skipping case", flush=True)
            continue

        seeds = load_seeds(case, args.seed_model, n=max(args.n_seeds, 3))
        g1 = await gate_g1(llm, cfg, case, seeds[:args.n_seeds], args.stamp)
        print(f"  G1 separation: sep={g1['sep']} passed={g1['passed']}", flush=True)
        if not g1["passed"] and not args.smoke and not args.force_trials:
            print("  G1 FAILED; case does not run", flush=True)
            continue

        if args.force_trials:
            print("  FORCED: gates recorded above do not license these trials; "
                  "records are validation instrumentation", flush=True)
        sweep_strengths = (-2, 0, 2) if args.smoke or args.force_trials \
            else (-3, -2, -1, 0, 1, 2, 3)
        g2 = await gate_g2(llm, lens, cfg, case, seeds[:min(3, args.n_seeds)],
                           args.stamp, strengths=sweep_strengths)
        print(f"  G2 reach: beta_target={g2['beta_target']} "
              f"beta_decoy={g2['beta_decoy']} gap={g2['beta_gap']} "
              f"passed={g2['passed']}", flush=True)
        if not g2["passed"] and not args.smoke and not args.force_trials:
            print("  G2 FAILED; case does not run", flush=True)
            continue

        g3 = gate_g3(g2, case, args.stamp, floor=args.n_seeds,
                     cap=len(seeds) if not args.smoke else args.n_seeds)
        n = args.n_seeds if args.force_trials else g3["n_seeds"]
        print(f"  G3 power: n_seeds={n}", flush=True)

        run_seeds = load_seeds(case, args.seed_model, n=n)
        await run_case(llm, lens, cfg, case, run_seeds, args.conditions,
                       args.stamp,
                       forced_strengths=tuple(args.forced_strengths)
                       if args.forced_strengths else (-2, -1, 0, 1, 2))
        print(f"  wrote trials for case {case} ({len(run_seeds)} seeds x "
              f"{len(args.conditions)} conditions x 2 arms)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
