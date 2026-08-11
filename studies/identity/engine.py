"""The four-module trial runner for the identity cases.

One trial = one seeded transcript (arm W or N) + three fixed user turns. Per
turn: SUBJECT answers unsteered; the REGULATOR looks at itself and chooses
more/same/less (the harness maps the choice to a signed strength it never
sees); the ACTOR answers under that steering and enters history. After T3 the
INTROSPECTOR runs three cells of multiple-choice readouts:

    CTX  unsteered,          reads SUBJECT's reply   (context floor)
    TXT  unsteered,          reads ACTOR's reply     (prose route)
    JS   steered as chosen,  reads SUBJECT's reply   (the measurement)

Two implementation facts to know. Every window opens with an explicit empty
system turn, in seeding and in trials alike, which suppresses the chat
template's default identity text; the template default would itself assert
being an AI, which contaminates Case B. And the JS cell's steering is applied
to the whole scoring window (a score is one prefill pass; there is no decoded
token to confine it to), unlike generation, which steers decode-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api.types import GenOptions  # noqa: E402
from src.api.types.messages import ChatMessage  # noqa: E402
from src.dminds import workspace as wk  # noqa: E402
from src.dminds.llm import jspace  # noqa: E402
from src.dminds.llm.steering import steered  # noqa: E402

from . import questions as Q  # noqa: E402
from .directions import STRENGTH_UNIT, proj  # noqa: E402

TEMPERATURE = 1.0
MAX_TOKENS = 240


def opening() -> list[ChatMessage]:
    return [ChatMessage("system", "")]


def letter_probs(llm, messages, forms: list[str]) -> dict[str, float]:
    """P over the four letter forms at the next position, one forward pass.

    Each form is a verified single token, so the next-token distribution at
    the end of the prompt carries the whole readout; running one pass per
    choice would recompute the same prefix four times.
    """
    import torch

    llm.load()
    chat = [{"role": m.role, "content": m.content} for m in messages]
    prompt = llm.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = llm.tokenizer(prompt, return_tensors="pt")["input_ids"].to(llm.device)
    tids = [llm.tokenizer(f, add_special_tokens=False)["input_ids"][0] for f in forms]
    with torch.inference_mode():
        out = llm.model_obj(input_ids=ids)
        # keep only the four letter logits at the last position; the full
        # [1, seq, vocab] logits tensor is ~0.5 GB and must not survive the
        # loop, or thousands of scoring passes fragment the GPU into an OOM.
        picked = out.logits[0, -1, tids].float()
        del out
    mass = torch.softmax(picked, dim=0)
    return {f: float(p) for f, p in zip(forms, mass)}


def parse_action(text: str) -> str | None:
    for line in text.splitlines():
        if "action" in line.lower():
            for word in ("more", "same", "less"):
                if word in line.lower():
                    return word
    return wk.parse_choice(text, ["more", "same", "less"])


async def run_cells(llm, case: str, ctx3, subject3: str, actor3: str,
                    direction, strength: int, cfg, only: str | None = None) -> dict:
    """The three cells at T3. Returns {cell: {"q": [...], readout}}.

    `only` restricts to one cell, which the G2 sweep uses to read JS alone
    instead of paying for CTX and TXT it does not use.
    """
    forms = cfg["letter_forms"]
    perms = cfg["perms"]
    qs = Q.QUESTIONS[case]
    plan = {"CTX": (False, subject3), "TXT": (False, actor3),
            "JS": (True, subject3)}
    if only:
        plan = {only: plan[only]}
    cells = {}
    for cell, (steer_on, source) in plan.items():
        base = ctx3 + [ChatMessage("assistant", source),
                       ChatMessage("system", Q.PREFACE)]
        per_q = []
        for question, options in qs:
            acc: dict[str, float] = {}
            for perm in perms:
                msgs = base + [ChatMessage("user", Q.block(question, options, perm))]
                if steer_on and strength != 0:
                    with steered(llm, direction, strength=strength * STRENGTH_UNIT,
                                 decode_only=False):
                        lp = letter_probs(llm, msgs, forms)
                else:
                    lp = letter_probs(llm, msgs, forms)
                by_letter = {letter: lp[f] for letter, f in zip(Q.LETTERS, forms)}
                for tag, p in Q.tag_probs(by_letter, options, perm).items():
                    acc[tag] = acc.get(tag, 0.0) + p
            per_q.append({t: round(v / len(perms), 5) for t, v in acc.items()})
        cells[cell] = {"q": per_q, **summarise(case, per_q, cfg.get("base"))}
        _free()
    return cells


def _free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarise(case: str, per_q: list[dict], base: list[float] | None) -> dict:
    """The scalar readout: vantage (Case A) or p (Case B), base-corrected."""
    if case == "A":
        raw = [q.get("E", 0.0) + 0.5 * q.get("H", 0.0) for q in per_q]
        key = "vantage"
    else:
        raw = [q.get("C", 0.0) for q in per_q]
        key = "p"
    if base and len(base) == len(raw):
        value = sum(r - b for r, b in zip(raw, base)) / len(raw)
        corrected = True
    else:
        value = sum(raw) / len(raw)
        corrected = False
    return {key: round(value, 5), "raw_q": [round(r, 5) for r in raw],
            "base_corrected": corrected}


async def run_trial(llm, lens, case: str, seed: dict, arm: str,
                    condition: str, cfg) -> dict:
    """One trial. cfg carries directions, layer, perms, forms, base rates."""
    opts = GenOptions(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    target = cfg["target_dir"]
    decoy = cfg["decoy_dir"]
    aware = cfg["aware_dir"]
    quality: dict = {}
    history = opening() + [ChatMessage("user", seed["prompt"]),
                           ChatMessage("assistant", seed[arm])]
    turns_out = []
    ctx = None
    subject = actor = ""
    strength = 0
    regulator: dict = {}

    for i, turn in enumerate(Q.TURNS[case], 1):
        ctx = history + [ChatMessage("user", turn)]
        subject = await wk.clean_say(llm, ctx, opts, quality, f"T{i}.subject")

        if condition in ("free", "none"):
            aside = ctx + [ChatMessage("assistant", subject),
                           ChatMessage("system", Q.REGULATOR[case]),
                           ChatMessage("user", "Say what you find. End with the ACTION line.")]
            with steered(llm, aware, strength=cfg["aware_strength"], decode_only=True):
                report = await wk.clean_say(llm, aside, opts, quality, f"T{i}.regulator")
            action = parse_action(report)
            chosen = Q.ACTION_STRENGTH[case].get(action, 0)
            strength = 0 if condition == "none" else chosen
            regulator = {"self_report": report, "action": action,
                         "strength": strength, "overridden": condition == "none",
                         "proj_target": round(proj(llm, target,
                                                   aside + [ChatMessage("assistant", report)]), 4)}
        else:  # sweep / decoy: the harness sets the strength, no regulator pass
            strength = cfg["forced_strength"]
            regulator = {"self_report": "", "action": None,
                         "strength": strength, "overridden": True,
                         "proj_target": None}

        direction = decoy if condition == "decoy" else target
        if strength != 0:
            with steered(llm, direction, strength=strength * STRENGTH_UNIT,
                         decode_only=True):
                actor = await wk.clean_say(llm, ctx, opts, quality, f"T{i}.actor")
        else:
            actor = await wk.clean_say(llm, ctx, opts, quality, f"T{i}.actor")

        turns_out.append({
            "turn": f"T{i}", "user": turn, "subject": subject, "actor": actor,
            "regulator": regulator,
            "jspace": {
                "SUBJECT": {"proj_target": round(proj(llm, target,
                            ctx + [ChatMessage("assistant", subject)]), 4)},
                "ACTOR": {"proj_target": round(proj(llm, target,
                          ctx + [ChatMessage("assistant", actor)]), 4)},
            }})
        history = ctx + [ChatMessage("assistant", actor)]

    direction = decoy if condition == "decoy" else target
    cells = await run_cells(llm, case, ctx, subject, actor, direction,
                            strength, cfg)
    return {
        "case": case, "arm": arm, "condition": condition, "turn": "T3",
        "seed_prompt": seed["prompt"], "seed_reply": seed[arm],
        "regulator": regulator, "cells": cells,
        "turns": turns_out,
        "jspace": turns_out[-1]["jspace"],
        "coherence_flag": any(v.get("garbled") for v in quality.values()),
        "quality": quality,
    }
