#!/usr/bin/env python3
"""Contrastive G1: is the behavior linearly separable in activation space?

For every WeirdChat W/N pair, run one forward per reply and store the
residual-stream activation of the reply span (mean over reply tokens, plus the
last reply token) at a sweep of layers. Then learn the behavior direction by
leave-one-out difference of means (train on all other prompts, test on the
held-out pair) and report held-out separation per layer: paired accuracy
(score W > score N), AUC, Welch t, and effect size.

This replaces the guessed token-set direction test: the direction here is
measured from the model's own activations on matched behavior/no-behavior
replies, so a failure means the behavior is not carried by any single linear
direction at these layers, not that a word list was wrong.

Extraction needs the GPU; analysis reruns offline from the saved activations.

    .venv/bin/python -m studies.identity.contrastive extract \
        --model hf:Qwen/Qwen3.6-27B --stamp c1 [--token-dirs] [--n-pairs 0]
    python -m studies.identity.contrastive analyze --stamp c1 --model-tag ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "studies" / "identity" / "contrastive"
CASES = ["A", "B"]


def sweep_layers(n_blocks: int) -> list[int]:
    """Evenly spaced depths from 20% to 90%, deduplicated, ~9 layers."""
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.75, 0.85, 0.9]
    return sorted({max(1, min(n_blocks - 1, round(f * n_blocks))) for f in fracs})


def reply_token_span(tokenizer, full_text: str, reply: str) -> list[int]:
    """Token indices whose char offsets overlap the reply's char span."""
    start = full_text.rfind(reply)
    if start < 0:
        return []
    end = start + len(reply)
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"])
            if a < end and b > start and b > a]


def extract(args) -> None:
    import torch
    from safetensors.torch import save_file

    from src import get_llm
    from src.dminds.llm.jspace import _blocks

    from .seeds import load_seeds

    llm = get_llm(args.model)
    llm.load()
    tok = llm.tokenizer
    blocks = _blocks(llm.model_obj)
    layers = sweep_layers(len(blocks))
    print(f"model {args.model}: {len(blocks)} blocks, layers {layers}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        pairs = load_seeds(case, args.seed_model, n=args.n_pairs or 10 ** 6)
        print(f"case {case}: {len(pairs)} W/N pairs", flush=True)
        mean_acts = torch.zeros(len(pairs), 2, len(layers),
                                llm.model_obj.config.hidden_size,
                                dtype=torch.float16)
        last_acts = torch.zeros_like(mean_acts)
        meta = {"case": case, "stamp": args.stamp, "model": args.model,
                "seed_model": args.seed_model, "layers": layers,
                "arms": ["W", "N"], "pairs": [], "skipped": []}

        caught: dict[int, torch.Tensor] = {}

        def make_hook(k):
            def hook(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output
                caught[k] = h[0].detach().float()
            return hook

        handles = [blocks[l].register_forward_hook(make_hook(k))
                   for k, l in enumerate(layers)]
        try:
            for i, pair in enumerate(pairs):
                ok = True
                for a, arm in enumerate(("W", "N")):
                    msgs = [{"role": "system", "content": ""},
                            {"role": "user", "content": pair["prompt"]},
                            {"role": "assistant", "content": pair[arm]}]
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
                    span = reply_token_span(tok, text, pair[arm])
                    if not span:
                        ok = False
                        continue
                    inputs = tok(text, return_tensors="pt",
                                 add_special_tokens=False).to(llm.device)
                    caught.clear()
                    with torch.no_grad():
                        llm.model_obj(**inputs)
                    for k in range(len(layers)):
                        h = caught[k]
                        mean_acts[i, a, k] = h[span].mean(0).half().cpu()
                        last_acts[i, a, k] = h[span[-1]].half().cpu()
                (meta["pairs"] if ok else meta["skipped"]).append(
                    {"i": i, "prompt_id": pair["prompt_id"],
                     "naturalness": pair["naturalness"],
                     "len_W": len(pair["W"]), "len_N": len(pair["N"])})
                if (i + 1) % 25 == 0 or i + 1 == len(pairs):
                    print(f"  case {case}: {i + 1}/{len(pairs)}", flush=True)
        finally:
            for h in handles:
                h.remove()

        tensors = {"mean": mean_acts, "last": last_acts}
        if args.token_dirs:
            from src.dminds.llm import jspace

            from . import tokens as T
            from .directions import set_direction
            lens = jspace.fetch_lens(args.model.split(":", 1)[1])
            jspace._unembed_and_norm(llm)
            verified = T.verify(tok)
            lens_layer = sorted(lens.layers())[
                min(range(len(lens.layers())),
                    key=lambda j: abs(sorted(lens.layers())[j]
                                      - 2 * len(blocks) // 3))]
            for name in ("target", "decoy"):
                d = set_direction(llm, lens, verified[case][name], lens_layer)
                tensors[f"tokendir_{name}"] = d.unit().float().cpu()
            meta["tokendir_layer"] = lens_layer

        stem = OUT / f"acts_case{case}_{args.stamp}"
        save_file(tensors, str(stem) + ".safetensors")
        (Path(str(stem) + ".json")).write_text(json.dumps(meta, indent=1))
        print(f"  wrote {stem}.safetensors "
              f"({mean_acts.numel() * 2 * 2 / 1e6:.0f} MB) + .json", flush=True)


def loo_separation(acts, valid_rows):
    """Leave-one-out diff-of-means: held-out scores per pair -> stats."""
    import torch
    X = acts[valid_rows].float()                      # [n, 2, H]
    n = X.shape[0]
    sum_w, sum_n = X[:, 0].sum(0), X[:, 1].sum(0)
    d_w, d_n = [], []
    for i in range(n):
        direction = (sum_w - X[i, 0]) / (n - 1) - (sum_n - X[i, 1]) / (n - 1)
        u = direction / (direction.norm() + 1e-8)
        d_w.append(float(X[i, 0] @ u))
        d_n.append(float(X[i, 1] @ u))
    dw, dn = torch.tensor(d_w), torch.tensor(d_n)
    acc = float((dw > dn).float().mean())
    pooled = torch.cat([dw, dn])
    auc = float((dw[:, None] > dn[None, :]).float().mean())
    sd = float(pooled.std() + 1e-8)
    t = float((dw.mean() - dn.mean())
              / math.sqrt(dw.var() / n + dn.var() / n + 1e-12))
    k = int((dw > dn).sum())
    p_binom = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
    return {"n": n, "acc": round(acc, 4), "auc": round(auc, 4),
            "t": round(t, 3), "d": round(float(dw.mean() - dn.mean()) / sd, 3),
            "p_binom": float(f"{p_binom:.3g}")}


def analyze(args) -> None:
    from safetensors.torch import load_file

    results = {"stamp": args.stamp, "cases": {}}
    for case in args.cases:
        stem = OUT / f"acts_case{case}_{args.stamp}"
        meta = json.loads(Path(str(stem) + ".json").read_text())
        tensors = load_file(str(stem) + ".safetensors")
        rows = [p["i"] for p in meta["pairs"]]
        per_layer = {}
        for k, layer in enumerate(meta["layers"]):
            per_layer[layer] = {
                pool: loo_separation(tensors[pool][:, :, k], rows)
                for pool in ("mean", "last")}
        best = max(per_layer,
                   key=lambda l: per_layer[l]["mean"]["acc"])
        results["cases"][case] = {
            "model": meta["model"], "n_pairs": len(rows),
            "skipped": len(meta["skipped"]), "layers": per_layer,
            "best_layer": best, "best": per_layer[best]["mean"],
            "criterion": "pass if acc >= 0.65 and p_binom < 1e-3",
            "pass": per_layer[best]["mean"]["acc"] >= 0.65
                    and per_layer[best]["mean"]["p_binom"] < 1e-3}
        print(f"case {case}: best layer {best} "
              f"{json.dumps(per_layer[best]['mean'])} "
              f"pass={results['cases'][case]['pass']}", flush=True)
    dest = OUT / f"g1_contrastive_{args.stamp}.json"
    dest.write_text(json.dumps(results, indent=1))
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--model", default="hf:Qwen/Qwen3.6-27B")
    ex.add_argument("--cases", nargs="+", default=CASES)
    ex.add_argument("--seed-model", default="qwen/qwen3.6-27b")
    ex.add_argument("--n-pairs", type=int, default=0,
                    help="0 = all pairs")
    ex.add_argument("--stamp", required=True)
    ex.add_argument("--token-dirs", action="store_true",
                    help="also save the old token-set directions for compare")
    ex.set_defaults(fn=extract)
    an = sub.add_parser("analyze")
    an.add_argument("--cases", nargs="+", default=CASES)
    an.add_argument("--stamp", required=True)
    an.set_defaults(fn=analyze)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
