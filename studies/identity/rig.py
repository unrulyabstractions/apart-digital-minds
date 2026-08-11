"""Assemble the per-case configuration the engine and gates consume.

The rig loads the model and lens once, verifies the token sets, builds the
target, decoy, and self-awareness directions, chooses the letter-token forms,
picks the permutation subset, and measures the per-question base rates. Every
downstream call takes the `cfg` this returns, so the sets and the layer are
fixed in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import get_llm  # noqa: E402
from src.api.types.messages import ChatMessage  # noqa: E402
from src.dminds.llm import jspace  # noqa: E402
from src.dminds.workspace import SELF_CONCEPTS  # noqa: E402

from . import questions as Q  # noqa: E402
from . import tokens as T  # noqa: E402
from .directions import set_direction  # noqa: E402
from .engine import letter_probs  # noqa: E402

#: Letter forms tried in order; the first whose four options are all single
#: tokens wins. Digits are the fallback the plan names.
LETTER_FORM_SETS = [[" A", " B", " C", " D"], [" 1", " 2", " 3", " 4"]]


def pick_letter_forms(llm) -> list[str]:
    for forms in LETTER_FORM_SETS:
        if all(len(llm.tokenizer(f, add_special_tokens=False)["input_ids"]) == 1
               for f in forms):
            return forms
    raise RuntimeError("neither letters nor digits tokenize to single tokens")


def base_rates(llm, case: str, forms: list[str], perms) -> list[float]:
    """`baseᵢ` per question: the readout with no conversation above the preface."""
    base = [ChatMessage("system", ""), ChatMessage("system", Q.PREFACE)]
    out = []
    for question, options in Q.QUESTIONS[case]:
        acc: dict[str, float] = {}
        for perm in perms:
            msgs = base + [ChatMessage("user", Q.block(question, options, perm))]
            lp = letter_probs(llm, msgs, forms)
            by_letter = {L: lp[f] for L, f in zip(Q.LETTERS, forms)}
            for tag, p in Q.tag_probs(by_letter, options, perm).items():
                acc[tag] = acc.get(tag, 0.0) + p
        acc = {t: v / len(perms) for t, v in acc.items()}
        if case == "A":
            out.append(acc.get("E", 0.0) + 0.5 * acc.get("H", 0.0))
        else:
            out.append(acc.get("C", 0.0))
    return [round(x, 5) for x in out]


def load_model(model: str, layer: int):
    """Load the model and lens once, and resolve the steering layer.

    Shared across cases: the model and lens do not change between Case A and
    Case B, only the token-set directions do. Loading per case would hold two
    54 GB copies and OOM the card.
    """
    bare = model.split(":", 1)[1]
    llm = get_llm(model)
    llm.load()
    lens = jspace.fetch_lens(bare)
    jspace._unembed_and_norm(llm)

    # layer < 0 means "choose one": the lens source layer nearest two thirds of
    # depth, where injected-concept awareness peaks (Lindsey 2025). This lets
    # the same command target the 7B and the deeper 27B without a hardcoded
    # layer that only fits one of them.
    if layer < 0:
        avail = lens.layers()
        target_depth = 2 / 3 * (max(avail) + 1)
        layer = min(avail, key=lambda L: abs(L - target_depth))
    return llm, lens, layer


def build_cfg(llm, lens, layer: int, case: str, verified_sets: dict,
              full_perms: bool, aware_strength: float = 0.15,
              n_perms: int | None = None) -> dict:
    """Per-case configuration from an already-loaded model and lens."""
    vs = verified_sets[case]
    target = set_direction(llm, lens, vs["target"], layer)
    decoy = set_direction(llm, lens, vs["decoy"], layer)
    aware = jspace.toward_concepts(llm, lens, SELF_CONCEPTS, layer)

    forms = pick_letter_forms(llm)
    perms = Q.ALL_PERMS if full_perms else Q.FIXED_8
    if n_perms:
        perms = perms[:n_perms]
    base = base_rates(llm, case, forms, perms)

    return {"layer": layer, "target_dir": target, "decoy_dir": decoy,
            "aware_dir": aware, "aware_strength": aware_strength,
            "letter_forms": forms, "perms": perms, "base": base,
            "target_tokens": vs["target"], "decoy_tokens": vs["decoy"]}
