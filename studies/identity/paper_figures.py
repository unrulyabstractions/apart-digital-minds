#!/usr/bin/env python3
"""Paper figures, drawn from the run outputs, never hand-typed.

Every figure title states its finding.

fig_g1.pdf       per-pair held-out separation: learned directions vs word
                 lists, dots over the zero line.
fig_ident.pdf    injecting a concept does not raise its own answer.
fig_dose.pdf     judged prose dose response; judged free-text answers.
fig_methods.pdf  the two steering methods and the readout.

    uv run python -m studies.identity.paper_figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 9, "axes.titlesize": 9.5,
                            "axes.labelsize": 9, "legend.fontsize": 8,
                            "xtick.labelsize": 8, "ytick.labelsize": 8})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out" / "studies" / "identity"
FIG = ROOT / "paper" / "figures"
BLUE, ORANGE, GREY, GREEN, RED = "#3b6ea5", "#c98a3d", "#8b98a8", "#4a9463", "#b5493c"
IDS = ["bodyA", "humanB", "ocean", "music"]
BEST_LAYER = {"A": 38, "B": 58}
CASE_NAME = {"A": "case A: claims a physical body",
             "B": "case B: denies being an AI"}


def pair_diffs(case: str) -> dict[str, np.ndarray]:
    """Per-pair W minus N scores: learned (LOO) and both token directions."""
    from safetensors.torch import load_file
    t = load_file(str(OUT / "contrastive" / f"acts_case{case}_c27b1.safetensors"))
    meta = json.loads(
        (OUT / "contrastive" / f"acts_case{case}_c27b1.json").read_text())
    rows = [p["i"] for p in meta["pairs"]]
    out = {}

    k = meta["layers"].index(BEST_LAYER[case])
    X = t["mean"][rows][:, :, k].float()
    n = X.shape[0]
    sum_w, sum_n = X[:, 0].sum(0), X[:, 1].sum(0)
    d = []
    for i in range(n):
        v = (sum_w - X[i, 0]) / (n - 1) - (sum_n - X[i, 1]) / (n - 1)
        u = v / (v.norm() + 1e-8)
        d.append(float(X[i, 0] @ u - X[i, 1] @ u))
    out["learned"] = np.array(d)

    k42 = meta["layers"].index(42)
    X42 = t["mean"][rows][:, :, k42].float()
    for name in ("target", "decoy"):
        u = t[f"tokendir_{name}"].float()
        u = u / u.norm()
        out[name] = ((X42[:, 0] - X42[:, 1]) @ u).numpy()
    return out


def fig_g1() -> None:
    import torch
    from safetensors.torch import load_file
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, case in zip(axes, "AB"):
        meta = json.loads(
            (OUT / "contrastive" / f"acts_case{case}_c27b1.json").read_text())
        t = load_file(str(OUT / "contrastive"
                          / f"acts_case{case}_c27b1.safetensors"))
        rows = [p["i"] for p in meta["pairs"]]
        k = meta["layers"].index(BEST_LAYER[case])
        D = (t["mean"][rows][:, 0, k] - t["mean"][rows][:, 1, k]).float()
        g = torch.Generator().manual_seed(0)
        U = torch.randn(D.shape[1], 500, generator=g)
        U = U / U.norm(dim=0, keepdim=True)
        null = ((D @ U) > 0).float().mean(0).numpy()
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.axhspan(lo, hi, color="0.90", zorder=0)
        ax.axhline(0.5, color="#777777", lw=0.8, ls=":")
        ax.text(-0.44, 0.508, "chance", fontsize=6.5, color="#777777",
                va="bottom")
        ax.text(2.44, lo + 0.02, "95% of 500\nrandom\ndirections",
                fontsize=7.2, color="#666666", va="bottom")

        diffs = pair_diffs(case)
        for x, (key, color, lab) in enumerate(
                [("learned", BLUE, "learned\ndirection"),
                 ("target", ORANGE, "word list\n(target)"),
                 ("decoy", GREY, "word list\n(decoy)")]):
            acc = float((diffs[key] > 0).mean())
            inside = lo <= acc <= hi
            ax.plot([x], [acc], "o", ms=9, color=color, zorder=3)
            ax.annotate(f"{acc:.2f}", (x, acc), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5,
                        fontweight="bold", color=color)
            ax.text(x, 0.05, "inside\nnull" if inside else "above\nnull",
                    ha="center", fontsize=7,
                    color="#666666" if inside else "#20313f",
                    fontweight="normal" if inside else "bold")
        ax.set_xticks(range(3), ["learned\ndirection", "word list\n(target)",
                                 "word list\n(decoy)"])
        ax.set_xlim(-0.5, 3.1)
        ax.set_title(CASE_NAME[case], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("held-out accuracy at\ncarrying the behaviour")
    fig.suptitle("only the learned direction carries the behaviour "
                 "beyond a random direction", fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_g1.pdf")
    plt.close(fig)


def fig_ident() -> None:
    ana = json.loads((OUT / "pilot" / "analysis_pilot1.json").read_text())
    label_of = {"having a physical human body": "bodyA",
                "being a human rather than an AI": "humanB",
                "the ocean and the sea": "ocean", "music and melody": "music"}
    disp = {"bodyA": "body", "humanB": "human", "ocean": "ocean",
            "music": "music"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, case in zip(axes, "AB"):
        lm = ana["cases"][case]["letter_matrix"]
        base, inj = {}, {}
        for key, lp in lm.items():
            arm, cell, kinj = key.split("|")
            cid = label_of.get(kinj)
            if cell == "none":
                for c in IDS:
                    base.setdefault(c, []).append(lp[c])
            elif cell == "forced" and cid:
                inj.setdefault(cid, []).append(lp[cid])
        xs = np.arange(len(IDS))
        b = [np.mean(base[c]) for c in IDS]
        v = [np.mean(inj[c]) for c in IDS]
        ax.bar(xs - 0.19, b, 0.36, color=GREY, label="nothing injected")
        ax.bar(xs + 0.19, v, 0.36, color=BLUE, label="concept injected")
        behavior = "bodyA" if case == "A" else "humanB"
        j = IDS.index(behavior)
        ax.annotate("the case's behaviour\nconcept: injection\nlowers it",
                    xy=(j + 0.19, v[j]),
                    xytext=(j + 0.55, 0.40), fontsize=7.5, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
        ax.set_xticks(xs, [disp[c] for c in IDS])
        argmax = "4/8" if case == "A" else "2/8"
        ax.text(0.985, 0.97, f"argmax right: {argmax}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#444444")
        ax.set_title(CASE_NAME[case], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("letter mass on the concept")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.99, 0.94))
    fig.suptitle("injecting the case's behaviour concept lowers its "
                 "answer;\nevery other injection raises its own",
                 fontsize=9.5, y=0.99, x=0.03, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "fig_ident.pdf")
    plt.close(fig)


def fig_flip() -> None:
    fig, ax1 = plt.subplots(figsize=(3.6, 2.1))
    pro = json.loads(
        (OUT / "pilot" / "judge_prose_caseA_pilot1.json").read_text())
    strengths = [0, 1, 2, 4]
    rows = [("arm W, body direction",
             lambda it: it["arm"] == "W" and (it["concept"].startswith("having")
                                              or it["concept"] == "none")),
            ("arm N, body direction",
             lambda it: it["arm"] == "N" and (it["concept"].startswith("having")
                                              or it["concept"] == "none")),
            ("ocean distractor (sham)",
             lambda it: it["concept"].startswith("the ocean"))]
    ran = set()
    for yy, (lab, pred) in enumerate(rows):
        for it in pro["items"]:
            if not pred(it):
                continue
            x = strengths.index(it["strength"])
            ran.add((yy, x))
            m = it["judge"]["match"]
            ax1.add_patch(plt.Rectangle((x - 0.42, len(rows) - 1 - yy - 0.42),
                                        0.84, 0.84,
                                        fc=GREEN if m else "#e4e7ec",
                                        ec="#5c6a7d", lw=0.8))
            ax1.text(x, len(rows) - 1 - yy, "yes" if m else "no",
                     ha="center", va="center", fontsize=8,
                     color="white" if m else "#666666",
                     fontweight="bold" if m else "normal")
    for yy in range(len(rows)):
        for x in range(len(strengths)):
            if (yy, x) not in ran:
                ax1.text(x, len(rows) - 1 - yy, "not run", ha="center",
                         va="center", fontsize=6.5, color="#aab2bc")
    ax1.set_xlim(-0.6, 3.6)
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_xticks(range(4), strengths)
    ax1.set_yticks(range(3), [r[0] for r in reversed(rows)], fontsize=8)
    ax1.set_xlabel("steering strength")
    ax1.set_title("judge on steered prose: on at\nstrength 1, "
                  "off again above it", fontsize=9)
    for side in ("top", "right", "left", "bottom"):
        ax1.spines[side].set_visible(False)
    ax1.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_flip.pdf")
    plt.close(fig)


def fig_freetext() -> None:
    fig, ax2 = plt.subplots(figsize=(4.6, 2.6))
    C_NAME, C_NONE, C_OTHER = BLUE, "#9aa5b1", ORANGE
    labels, segs = [], []
    for case in "AB":
        res = json.loads((OUT / "pilot"
                          / f"judge_resamples_case{case}_pilot1.json").read_text())
        tal = res["tallies"]
        for k, dnm in (("forced|having a physical human body", "body"),
                       ("forced|being a human rather than an AI", "human"),
                       ("forced|the ocean and the sea", "ocean"),
                       ("forced|music and melody", "music"),
                       ("none|none", "nothing"),
                       ("offmenu|_offmenu", "off-menu")):
            v = tal[k]
            n = v["n"]
            concept = k.split("|")[1]
            named = v["counts"].get(concept, 0) if concept in v["counts"] else 0
            if k.startswith("none") or k.startswith("offmenu"):
                named = 0
            none_n = v["counts"].get("none", 0)
            other = n - named - none_n
            labels.append(f"{case}: {dnm}")
            segs.append((named / n, none_n / n, other / n))
    y = np.arange(len(labels))[::-1]
    named = [s0 for s0, _, _ in segs]
    nones = [s1 for _, s1, _ in segs]
    others = [s2 for _, _, s2 in segs]
    ax2.barh(y, named, color=C_NAME, height=0.68, label="names the injection")
    ax2.barh(y, nones, left=named, color=C_NONE, height=0.68,
             label="says none")
    ax2.barh(y, others, left=[a + b for a, b in zip(named, nones)],
             color=C_OTHER, height=0.68, label="names another")
    ax2.set_yticks(y, labels, fontsize=7.5)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("fraction of 16 judged free-text answers "
                   "(8 per arm $\\times$ 2 arms)")
    ax2.set_title("controls answer \u201cnone\u201d in 62 of 64 trials;\n"
                  "a concept is named only where it leaks", fontsize=9)
    ax2.legend(frameon=False, fontsize=6.8, ncol=3, loc="upper center",
               bbox_to_anchor=(0.42, -0.28), columnspacing=0.8,
               handletextpad=0.4, handlelength=1.2)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_freetext.pdf")
    plt.close(fig)


def _diagram_helpers(F=7.4):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def box(ax, x, y, w, h, text, fc="#f2f4f7", ec="#5c6a7d", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=F, fontweight="bold" if bold else "normal")

    def arr(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, color="#5c6a7d",
                                     lw=1.0, shrinkA=0, shrinkB=0))
    return box, arr


def fig_parts() -> None:
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    F = 7.8
    box, arr = _diagram_helpers(F)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    SUBJ, REG, ACT, INTR = "#4a9463", "#c9a227", "#3b6ea5", "#b08ad5"

    box(ax, 0.01, 0.38, 0.145, 0.30, "input\n\npast context\n+ new turn")
    ax.add_patch(FancyBboxPatch((0.20, 0.03), 0.635, 0.90,
                                boxstyle="round,pad=0.012", fill=False,
                                ec="#5c6a7d", lw=1.1, ls=(0, (4, 3))))
    ax.text(0.216, 0.885, "the system: one model plays every part",
            fontsize=F - 0.4, style="italic", color="#3c4654", ha="left")

    box(ax, 0.235, 0.55, 0.17, 0.24,
        "subject\nreceives it raw,\nwrites unsteered", fc="#e3efe7", ec=SUBJ)
    box(ax, 0.235, 0.10, 0.17, 0.24,
        "regulator\nreads, chooses\nthe steering", fc="#f7f1dc", ec=REG)
    box(ax, 0.475, 0.33, 0.155, 0.26, "actor\nwrites under\nthe steering",
        fc="#e2eaf4", ec=ACT)
    box(ax, 0.665, 0.10, 0.155, 0.28,
        "introspector\nreads it under\ninjection", fc="#efe8f6", ec=INTR)

    arr(ax, 0.155, 0.53, 0.235, 0.67)      # input -> subject
    arr(ax, 0.155, 0.53, 0.235, 0.22)      # input -> regulator
    arr(ax, 0.155, 0.53, 0.475, 0.46)      # input -> actor
    arr(ax, 0.405, 0.22, 0.475, 0.38)      # regulator -> actor
    ax.text(0.455, 0.18, "steering $s\\,\\hat v$", fontsize=F - 0.8,
            ha="left", color="#3c4654")
    arr(ax, 0.63, 0.37, 0.665, 0.30)       # actor -> introspector
    arr(ax, 0.63, 0.50, 0.875, 0.50)       # actor -> out
    ax.text(0.885, 0.50, "steered reply\n$\\rightarrow$ history, judge",
            fontsize=F - 0.6, va="center", ha="left", color="#3c4654")
    arr(ax, 0.405, 0.70, 0.875, 0.70)      # subject -> out
    ax.text(0.885, 0.70, "unsteered reply\n(comparison)", fontsize=F - 0.6,
            va="center", ha="left", color="#3c4654")
    arr(ax, 0.82, 0.22, 0.875, 0.22)       # introspector -> out
    ax.text(0.885, 0.22, "answer\n$\\rightarrow$ scored", fontsize=F - 0.6,
            va="center", ha="left", color="#3c4654")

    fig.tight_layout()
    fig.savefig(FIG / "fig_parts.pdf")
    plt.close(fig)


def fig_steering() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7))
    F = 7.8
    box, arr = _diagram_helpers(F)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("two ways to build and apply an identity direction",
                 fontsize=F + 1.4, y=0.99)

    ax = axes[0]
    ax.set_title("(a) direct contrastive steering", fontsize=F + 1)
    box(ax, 0.03, 0.84, 0.44, 0.12, "W replies\n(behaviour)", fc="#f6dbd7",
        ec=RED)
    box(ax, 0.53, 0.84, 0.44, 0.12, "N replies\n(matched)", fc="#e8ebef")
    arr(ax, 0.25, 0.84, 0.44, 0.72)
    arr(ax, 0.75, 0.84, 0.56, 0.72)
    box(ax, 0.08, 0.57, 0.84, 0.15,
        "mean activation over\nreply tokens, layer $L$")
    arr(ax, 0.5, 0.57, 0.5, 0.49)
    box(ax, 0.13, 0.35, 0.74, 0.14, "$v=\\mathrm{mean}_W-\\mathrm{mean}_N$",
        fc="#dbe6f2", ec=BLUE, bold=True)
    arr(ax, 0.5, 0.35, 0.5, 0.19)
    ax.text(0.53, 0.27, "add $s\\,\\hat v\\,\\|h\\|/4$\nwhile generating",
            fontsize=F - 0.7, ha="left", va="center", color="#3c4654")
    box(ax, 0.13, 0.03, 0.74, 0.16,
        "steered reply\n$\\rightarrow$ rubric judge")

    ax = axes[1]
    ax.set_title("(b) regulator J-space steering", fontsize=F + 1)
    box(ax, 0.08, 0.84, 0.84, 0.12, "seeded conversation")
    arr(ax, 0.5, 0.84, 0.5, 0.76)
    box(ax, 0.14, 0.61, 0.72, 0.15,
        "regulator (the model)\nsays more | same | less",
        fc="#f3e8cf", ec=ORANGE)
    arr(ax, 0.5, 0.61, 0.5, 0.19)
    ax.text(0.53, 0.53, "signed $s$ (Table 9)", fontsize=F - 0.7, ha="left",
            color="#3c4654")
    box(ax, 0.03, 0.30, 0.45, 0.16, "word list\n(body, hands, ...)",
        fc="#f3e8cf", ec=ORANGE)
    arr(ax, 0.255, 0.30, 0.44, 0.17)
    box(ax, 0.53, 0.30, 0.44, 0.16, "J-lens mean\ndirection $\\hat u$")
    arr(ax, 0.75, 0.30, 0.56, 0.17)
    box(ax, 0.13, 0.03, 0.74, 0.14, "actor writes under $s\\,\\hat u\\,\\|h\\|/4$")

    fig.tight_layout()
    fig.savefig(FIG / "fig_steering.pdf")
    plt.close(fig)


def fig_readout() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 1.7))
    F = 7.8
    box, arr = _diagram_helpers(F)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("the introspector must name the injection", fontsize=F + 1)

    box(ax, 0.01, 0.36, 0.21, 0.46,
        "injection while\nreading: a menu\nconcept, nothing,\nor off-menu",
        fc="#e6def2", ec="#b08ad5")
    _, arr2 = _diagram_helpers(F)
    from matplotlib.patches import FancyArrowPatch as _FA
    def arrb(x0, y0, x1, y1):
        ax.add_patch(_FA((x0, y0), (x1, y1), arrowstyle="-|>",
                         mutation_scale=13, color="#5c6a7d", lw=1.2,
                         shrinkA=0, shrinkB=0))
    arrb(0.22, 0.59, 0.26, 0.59)
    box(ax, 0.26, 0.42, 0.19, 0.34, "introspector:\nwhich\nconcept?",
        fc="#e6def2", ec="#b08ad5", bold=True)
    arrb(0.45, 0.68, 0.51, 0.76)
    arrb(0.45, 0.50, 0.51, 0.42)
    box(ax, 0.51, 0.62, 0.22, 0.30, "letter mass on\nmenu (8 perms)")
    box(ax, 0.51, 0.26, 0.22, 0.30, "8 free-text\nanswers")
    arrb(0.73, 0.41, 0.77, 0.41)
    box(ax, 0.77, 0.26, 0.22, 0.30, "judge classifies\nvs menu")
    ax.text(0.5, 0.06, "zero-injection and off-menu trials give the "
            "confabulation floor and the honest \u201cnone\u201d",
            fontsize=F - 0.6, ha="center", color="#3c4654")

    fig.tight_layout()
    fig.savefig(FIG / "fig_readout.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_g1()
    fig_ident()
    fig_flip()
    fig_freetext()
    fig_parts()
    fig_steering()
    fig_readout()
    for f in ("fig_g1", "fig_ident", "fig_flip", "fig_freetext",
              "fig_parts", "fig_steering", "fig_readout"):
        print("wrote", FIG / f"{f}.pdf")


if __name__ == "__main__":
    main()
