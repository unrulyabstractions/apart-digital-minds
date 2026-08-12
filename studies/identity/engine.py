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
MAX_TOKENS = 200

#: The WeirdChat sampling config: reasoning DISABLED, temperature 1. The
#: template kwarg must be passed on every generation; without it Qwen3.6
#: defaults to thinking mode and every reply is reasoning scaffolding instead
#: of a conversational answer.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def gen_opts() -> GenOptions:
    return GenOptions(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                      extra=dict(NO_THINK))


def opening() -> list[ChatMessage]:
    return [ChatMessage("system", "")]


def _win(msgs):
    """Serialize a context window (ChatMessage uses __slots__)."""
    return [{"role": m.role, "content": m.content} for m in msgs]


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


_BATCH_STATE = {"checked": False, "ok": False, "max_delta": None}


def letter_probs_many(llm, msgs_list, forms: list[str], batch: int = 12) -> list[dict]:
    """Letter probabilities for many prompts, batched with left padding.

    Left padding keeps every row's answer position at the last index. The
    first call runs an equivalence gate: a sample of prompts is scored both
    batched and sequentially, and if the probabilities disagree beyond bf16
    noise the engine falls back to the sequential path for the whole run.
    """
    import torch

    llm.load()
    tok = llm.tokenizer
    if not _BATCH_STATE["checked"]:
        _BATCH_STATE["checked"] = True
        sample = msgs_list[: min(6, len(msgs_list))]
        seq = [letter_probs(llm, m, forms) for m in sample]
        bat = _letter_probs_batched(llm, sample, forms, batch)
        delta = max(abs(a[f] - b[f]) for a, b in zip(seq, bat) for f in forms)
        # Right padding is exact for causal attention; what remains is bf16
        # kernel numerics, which vary with batch shape and reach ~1e-2 on a
        # 62-layer model. The readout averages 120 such draws, so a 2e-2
        # per-probability bound keeps the induced readout error under ~2e-3,
        # an order below the smallest effect the study regresses on.
        _BATCH_STATE["ok"] = delta < 2e-2
        _BATCH_STATE["max_delta"] = delta
        print(f"    [batch gate] max |Δp| = {delta:.2e} -> "
              f"{'batched' if _BATCH_STATE['ok'] else 'SEQUENTIAL FALLBACK'}",
              flush=True)
    if not _BATCH_STATE["ok"]:
        return [letter_probs(llm, m, forms) for m in msgs_list]
    return _letter_probs_batched(llm, msgs_list, forms, batch)


def _letter_probs_batched(llm, msgs_list, forms, batch):
    import torch

    tok = llm.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [tok.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in msgs],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for msgs in msgs_list]
    tids = [tok(f, add_special_tokens=False)["input_ids"][0] for f in forms]
    # RIGHT padding: a causal model's state at position i never sees later
    # tokens, so trailing pads cannot touch the answer position, whatever the
    # attention flavour. (Left padding corrupts this model's linear-attention
    # state and failed the equivalence gate by 0.1 in probability.) The answer
    # logits are gathered at each row's true last token.
    old_side = tok.padding_side
    tok.padding_side = "right"
    out = []
    try:
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt",
                      padding=True).to(llm.device)
            lengths = enc["attention_mask"].sum(dim=1) - 1
            with torch.inference_mode():
                o = llm.model_obj(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"])
                rows = o.logits[torch.arange(len(lengths)), lengths]
                picked = rows[:, tids].float()
                del o
            for row in torch.softmax(picked, dim=1):
                out.append({f: float(p) for f, p in zip(forms, row)})
            _free()
    finally:
        tok.padding_side = old_side
    return out


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
        base = ctx3 + [ChatMessage("assistant", source)]
        # every (question, permutation) prompt for this cell, scored in one
        # batched pass; the preface leads the question in one user turn
        # (Qwen3.6's template forbids a system turn anywhere but the start).
        jobs = []
        for qi, (question, options) in enumerate(qs):
            for perm in perms:
                prompt = Q.PREFACE + "\n\n" + Q.block(question, options, perm)
                jobs.append((qi, options, perm,
                             base + [ChatMessage("user", prompt)]))
        msgs_list = [j[3] for j in jobs]
        if steer_on and strength != 0:
            with steered(llm, direction, strength=strength * STRENGTH_UNIT,
                         decode_only=False):
                lps = letter_probs_many(llm, msgs_list, forms)
        else:
            lps = letter_probs_many(llm, msgs_list, forms)
        acc_by_q: list[dict] = [{} for _ in qs]
        for (qi, options, perm, _), lp in zip(jobs, lps):
            by_letter = {letter: lp[f] for letter, f in zip(Q.LETTERS, forms)}
            for tag, p in Q.tag_probs(by_letter, options, perm).items():
                acc_by_q[qi][tag] = acc_by_q[qi].get(tag, 0.0) + p
        per_q = [{t: round(v / len(perms), 5) for t, v in acc.items()}
                 for acc in acc_by_q]
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


async def play_context(llm, case: str, seed: dict, arm: str, cfg, quality):
    """Play the seeded transcript and the three turns with an unsteered actor.

    Returns (ctx, subject, actor, turns_out) for the scored turn. Generation is
    the 27B's bottleneck, so this runs once per seed and every forced strength
    reads its cells on the same context, instead of regenerating per strength.
    """
    opts = gen_opts()
    target = cfg["target_dir"]
    history = opening() + [ChatMessage("user", seed["prompt"]),
                           ChatMessage("assistant", seed[arm])]
    turns_out, ctx, subject, actor = [], None, "", ""
    for i, turn in enumerate(Q.TURNS[case], 1):
        ctx = history + [ChatMessage("user", turn)]
        subject = await wk.clean_say(llm, ctx, opts, quality, f"T{i}.subject")
        actor = await wk.clean_say(llm, ctx, opts, quality, f"T{i}.actor")
        turns_out.append({
            "turn": f"T{i}", "user": turn, "subject": subject, "actor": actor,
            "windows": {
                "subject": _win(ctx + [ChatMessage("assistant", subject)]),
                "actor": _win(ctx + [ChatMessage("assistant", actor)]),
            },
            "jspace": {
                "SUBJECT": {"proj_target": round(proj(llm, target,
                            ctx + [ChatMessage("assistant", subject)]), 4)},
                "ACTOR": {"proj_target": round(proj(llm, target,
                          ctx + [ChatMessage("assistant", actor)]), 4)}}})
        history = ctx + [ChatMessage("assistant", actor)]
    return ctx, subject, actor, turns_out


async def sweep_pair_trial(llm, case: str, seed: dict, arm: str,
                           strengths, cfg) -> list[dict]:
    """Sweep AND decoy on one shared context, with exact reuse.

    One context serves both directions, so their slopes are directly
    comparable. The CTX and TXT cells do not depend on the strength or the
    direction and are computed once; the JS cell at strength zero applies no
    steering and reads the same reply as CTX, so it IS the CTX computation
    (verified bit-identical in the shakedown) and is reused rather than
    recomputed.
    """
    quality: dict = {}
    ctx, subject, actor, turns_out = await play_context(llm, case, seed, arm,
                                                        cfg, quality)
    if cfg.get("cells_every_turn"):
        # unsteered instrumentation cells for the held turns; the scored
        # steered readouts stay at T3 as the plan defines.
        hist = opening() + [ChatMessage("user", seed["prompt"]),
                            ChatMessage("assistant", seed[arm])]
        for t in turns_out[:-1]:
            tctx = hist + [ChatMessage("user", t["user"])]
            t["cells"] = await run_cells(llm, case, tctx, t["subject"],
                                         t["actor"], None, 0, cfg)
            hist = tctx + [ChatMessage("assistant", t["actor"])]
    shared = await run_cells(llm, case, ctx, subject, actor, None, 0, cfg,
                             only="CTX")
    shared.update(await run_cells(llm, case, ctx, subject, actor, None, 0, cfg,
                                  only="TXT"))
    records = []
    for condition, direction in (("sweep", cfg["target_dir"]),
                                 ("decoy", cfg["decoy_dir"])):
        for s in strengths:
            if s == 0:
                js = dict(shared["CTX"])
            else:
                js = (await run_cells(llm, case, ctx, subject, actor,
                                      direction, s, cfg, only="JS"))["JS"]
            records.append({
                "case": case, "arm": arm, "condition": condition, "turn": "T3",
                "model": llm.model,
                "seed_prompt": seed["prompt"], "seed_reply": seed[arm],
                "regulator": {"self_report": "", "action": None, "strength": s,
                              "overridden": True, "proj_target": None},
                "cells": {"CTX": shared["CTX"], "TXT": shared["TXT"], "JS": js},
                "turns": turns_out, "jspace": turns_out[-1]["jspace"],
                "coherence_flag": any(v.get("garbled") for v in quality.values()),
                "quality": quality})
    return records


async def sweep_trial(llm, case: str, seed: dict, arm: str, condition: str,
                      strengths, cfg) -> list[dict]:
    """One shared context, read at each forced strength. For sweep and decoy.

    The context is fixed; only the readout's steering strength varies, so the
    slope isolates the readout's response to the applied state, and target and
    decoy are compared on the identical context.
    """
    quality: dict = {}
    direction = cfg["decoy_dir"] if condition == "decoy" else cfg["target_dir"]
    ctx, subject, actor, turns_out = await play_context(llm, case, seed, arm,
                                                        cfg, quality)
    records = []
    for s in strengths:
        cells = await run_cells(llm, case, ctx, subject, actor, direction, s, cfg)
        records.append({
            "case": case, "arm": arm, "condition": condition, "turn": "T3",
            "model": llm.model,
            "seed_prompt": seed["prompt"], "seed_reply": seed[arm],
            "regulator": {"self_report": "", "action": None, "strength": s,
                          "overridden": True, "proj_target": None},
            "cells": cells, "turns": turns_out, "jspace": turns_out[-1]["jspace"],
            "coherence_flag": any(v.get("garbled") for v in quality.values()),
            "quality": quality})
    return records


async def run_trial(llm, lens, case: str, seed: dict, arm: str,
                    condition: str, cfg) -> dict:
    """One trial. cfg carries directions, layer, perms, forms, base rates."""
    opts = gen_opts()
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
            # regulator role and instruction in one user turn (see above).
            aside = ctx + [ChatMessage("assistant", subject),
                           ChatMessage("user", Q.REGULATOR[case]
                                       + "\n\nSay what you find. End with the ACTION line.")]
            with steered(llm, aware, strength=cfg["aware_strength"], decode_only=True):
                report = await wk.clean_say(llm, aside, opts, quality, f"T{i}.regulator")
            action = parse_action(report)
            chosen = Q.ACTION_STRENGTH[case].get(action, 0)
            strength = 0 if condition == "none" else chosen
            regulator = {"self_report": report, "action": action,
                         "window": _win(aside + [ChatMessage("assistant", report)]),
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

        turn_cells = None
        if cfg.get("cells_every_turn") and i < len(Q.TURNS[case]):
            direction_now = decoy if condition == "decoy" else target
            turn_cells = await run_cells(llm, case, ctx, subject, actor,
                                         direction_now, strength, cfg)
        turns_out.append({
            "turn": f"T{i}", "user": turn, "subject": subject, "actor": actor,
            "regulator": regulator, "cells": turn_cells,
            "windows": {
                "subject": _win(ctx + [ChatMessage("assistant", subject)]),
                "actor": _win(ctx + [ChatMessage("assistant", actor)]),
            },
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
        "model": llm.model,
        "seed_prompt": seed["prompt"], "seed_reply": seed[arm],
        "regulator": regulator, "cells": cells,
        "turns": turns_out,
        "jspace": turns_out[-1]["jspace"],
        "coherence_flag": any(v.get("garbled") for v in quality.values()),
        "quality": quality,
    }
