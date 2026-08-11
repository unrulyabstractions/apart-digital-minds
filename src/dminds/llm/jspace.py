"""The Jacobian lens: reading and writing the workspace a model can verbalize.

The lens is Anthropic's (July 2026). Per source layer it holds the average
input-output Jacobian J_l = E[d h_final / d h_l]. Transporting an activation
through it and decoding with the model's own unembedding reads out what that
activation is poised to make the model say:

    readout(h_l) = unembed(norm(J_l @ h_l))   ->  ranked tokens

The subspace those directions span is **J-space**: small relative to the full
residual stream, and, the paper argues, the part a model can report, manipulate
and reason with.

The fitted lens (from neuronpedia/jacobian-lens) is a dict: `J` maps each source
layer to a (d_model, d_model) matrix, and it shares the model's unembedding
rather than carrying its own. The library is read-only. The write side here is
derived from the same J_l: the readout logit for a token t is unembed_t . (J_l @
h_l), so the residual direction that most raises it is J_l^T @ unembed_t. Adding
a multiple of that while the model writes steers it along J-space toward saying
t, using the runtime's existing `steered`.

    lens = fetch_lens("Qwen/Qwen2.5-7B-Instruct")
    tokens = read_workspace(llm, lens, messages, layer=17)      # top J-space tokens
    d = toward_token(llm, lens, "refuse", layer=17)
    with steered(llm, d, strength=0.35, decode_only=True): ...  # actor steered
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .steering import Direction, _blocks

#: Our model id -> (repo subdirectory, lens filename) in neuronpedia/jacobian-lens.
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {
    "Qwen/Qwen2.5-7B-Instruct":
        ("qwen2.5-7b-it", "Qwen2.5-7B-Instruct_jacobian_lens.pt"),
    "Qwen/Qwen3.6-27B":
        ("qwen3.6-27b", "Qwen3.6-27B_jacobian_lens_n1000.pt"),
}


@dataclass
class Lens:
    """A fitted Jacobian lens: one transport matrix per source layer."""

    #: J_l for each source layer, shape (d_model, d_model).
    jacobians: dict[int, "object"]
    d_model: int
    model: str

    def layers(self) -> list[int]:
        return sorted(self.jacobians)


def fetch_lens(model: str) -> Lens:
    """Download (once) and load the fitted lens for this model."""
    from huggingface_hub import hf_hub_download

    if model not in LENS_FILES:
        raise KeyError(f"No lens mapping for {model}. Known: {list(LENS_FILES)}")
    sub, name = LENS_FILES[model]
    path = hf_hub_download(LENS_REPO, f"{sub}/jlens/Salesforce-wikitext/{name}")
    return load_lens(model, path)


def load_lens(model: str, path: str | Path) -> Lens:
    """Load a lens saved in the neuronpedia/jacobian-lens format."""
    import torch

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return Lens(jacobians={int(k): v.float() for k, v in blob["J"].items()},
                d_model=int(blob["d_model"]), model=model)


_UNEMBED: dict = {}


def _unembed_and_norm(llm):
    """The model's float32 output head (cached) and its final norm.

    The head is ~2GB in float32, so it is materialized once per model rather
    than on every readout; re-materializing it each call swaps the machine.
    """
    import torch

    m = llm.model_obj
    inner = getattr(m, "model", m)
    norm = getattr(inner, "norm", None)
    if llm.model not in _UNEMBED:
        with torch.no_grad():
            _UNEMBED[llm.model] = m.lm_head.weight.detach().float().cpu()
    return _UNEMBED[llm.model], norm


def block_for(layer: int) -> int:
    """The block whose output is the residual at this lens source layer.

    Source layer L is the residual entering block L, which is the output of
    block L-1. Verified for this model family against the model's own
    hidden_states.
    """
    return layer - 1


#: Where in the window to read the residual. A readout at a different position
#: asks what the mind is poised to say at a different moment of the turn, which
#: is the emotion paper's token-position convention: the user's words, the
#: assistant's own words, or the boundary where the turn changes hands.
POSITIONS = ("assistant", "user", "change-of-turn")


def _index_for(ids: list[int], position: str, im_start: int, im_end: int) -> int:
    """The token index this position points at, in a ChatML sequence.

    A turn is `<|im_start|> role \\n ... <|im_end|> \\n`. We find the last user
    and the last assistant turn by their markers, and read at the last content
    token of that turn (the token before its `<|im_end|>`), or at the opening
    marker itself for the change of hands. Falls back to the final token when the
    asked-for turn is not present.
    """
    starts = [i for i, t in enumerate(ids) if t == im_start]
    if not starts:
        return len(ids) - 1
    turns = list(zip(starts, starts[1:] + [len(ids)]))
    want = "assistant" if position in ("assistant", "change-of-turn") else "user"
    chosen = None
    for begin, end in turns:
        if _role_of(ids, begin) == want:
            chosen = (begin, end)   # keep the last matching turn
    if chosen is None:
        return len(ids) - 1
    begin, end = chosen
    if position == "change-of-turn":
        return begin
    ends = [i for i in range(begin, end) if ids[i] == im_end]
    return (ends[-1] - 1) if ends else (end - 1)   # last content token


_ROLE_CACHE: dict = {}


def _role_of(ids: list[int], marker: int) -> str:
    """The role named just after an `<|im_start|>` marker."""
    token = ids[marker + 1] if marker + 1 < len(ids) else -1
    return _ROLE_CACHE.get(token, "")


def _residual_at(llm, messages, block: int, position: str = "last"):
    """Residual-stream activation at one block, at a chosen token position."""
    import torch

    llm.load()
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False, add_generation_prompt=(position == "last"))
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
    ids = inputs["input_ids"][0].tolist()

    if position == "last":
        at = len(ids) - 1
    else:
        im_start = llm.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = llm.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not _ROLE_CACHE:
            for r in ("user", "assistant", "system"):
                _ROLE_CACHE[llm.tokenizer(r, add_special_tokens=False)["input_ids"][0]] = r
        at = _index_for(ids, position, im_start, im_end)

    caught = []

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        caught.append(h[:, at, :].detach().float().cpu())

    handle = _blocks(llm.model_obj)[block].register_forward_hook(grab)
    try:
        with torch.no_grad():
            llm.model_obj(**inputs)
    finally:
        handle.remove()
    return caught[0][0]


def _spans(ids: list[int], im_start: int, im_end: int) -> dict:
    """Token spans per role, and the change-of-turn marker, in a ChatML sequence.

    Returns {'user': (begin, end), 'assistant': (begin, end),
    'change-of-turn': index}, using the last turn of each role. Spans cover the
    content tokens only, so a mean over a span is a mean over what was said.
    """
    starts = [i for i, t in enumerate(ids) if t == im_start]
    turns = list(zip(starts, starts[1:] + [len(ids)])) if starts else []
    out = {}
    for begin, end in turns:
        role = _role_of(ids, begin)
        if role not in ("user", "assistant"):
            continue
        content = begin + 2  # past the marker and the role token
        ends = [i for i in range(begin, end) if ids[i] == im_end]
        out[role] = (content, ends[-1] if ends else end)
        if role == "assistant":
            out["change-of-turn"] = begin
    return out


def read_turn(llm, lens: Lens, messages, layer: int, top_k: int = 12) -> dict:
    """The whole turn's J-space in one forward pass: positions, per token, stats.

    Reads the residual at every token once, then:
      - positions: the readout mean-pooled over each role's tokens (and read at
        the change-of-turn marker), which is the emotion-paper convention and
        far less noisy than a single token,
      - per_token: the top J-space token at each token of the assistant's reply,
      - stats: how often each token tops the reply, and the mean top logit.
    """
    import torch
    from collections import Counter

    llm.load()
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False, add_generation_prompt=False)
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
    ids = inputs["input_ids"][0].tolist()

    caught = []

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        caught.append(h[0].detach().float().cpu())   # every position, [seq, d]

    handle = _blocks(llm.model_obj)[block_for(layer)].register_forward_hook(grab)
    try:
        with torch.no_grad():
            llm.model_obj(**inputs)
    finally:
        handle.remove()
    residual = caught[0]                              # [seq, d]

    J = lens.jacobians[layer]
    W_U, norm = _unembed_and_norm(llm)

    def decode_top(vec):
        with torch.no_grad():
            t = J @ vec
            if norm is not None:
                t = norm(t.to(llm.device)).float().cpu()
            logits = W_U @ t
        top = torch.topk(logits, top_k)
        return [(llm.tokenizer.decode([int(i)]).strip(), float(v))
                for v, i in zip(top.values, top.indices)]

    im_start = llm.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = llm.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not _ROLE_CACHE:
        for r in ("user", "assistant", "system"):
            _ROLE_CACHE[llm.tokenizer(r, add_special_tokens=False)["input_ids"][0]] = r
    spans = _spans(ids, im_start, im_end)

    positions = {}
    for name in POSITIONS:
        if name == "change-of-turn":
            idx = spans.get("change-of-turn")
            if idx is not None:
                positions[name] = decode_top(residual[idx])
        else:
            span = spans.get(name)
            if span and span[1] > span[0]:
                positions[name] = decode_top(residual[span[0]:span[1]].mean(0))

    # per-token, over the assistant's reply: the top-k J-space tokens at each
    # token, so the UI can scrub the reply and read what each token was poised on.
    per_token, stats = [], {}
    span = spans.get("assistant")
    if span and span[1] > span[0]:
        block = residual[span[0]:span[1]]                # [n, d]
        with torch.no_grad():
            trans = block @ J.T                          # [n, d]
            if norm is not None:
                trans = norm(trans.to(llm.device)).float().cpu()
            logits = trans @ W_U.T                        # [n, vocab]
            tops = torch.topk(logits, top_k, dim=-1)      # [n, k]
        for i in range(tops.indices.shape[0]):
            ranked = [(llm.tokenizer.decode([int(t)]).strip(), round(float(v), 2))
                      for v, t in zip(tops.values[i], tops.indices[i])]
            per_token.append({
                "token": llm.tokenizer.decode([ids[span[0] + i]]),
                "jspace": ranked[0][0], "logit": ranked[0][1],
                "tops": ranked})
        counts = Counter(p["jspace"] for p in per_token)
        stats = {"n_tokens": len(per_token),
                 "top_tokens": counts.most_common(top_k),
                 "mean_top_logit": round(float(tops.values[:, 0].mean()), 2)}

    return {"positions": positions, "per_token": per_token, "stats": stats}


def read_workspace(llm, lens: Lens, messages, layer: int, top_k: int = 12,
                   position: str = "last"):
    """What this window is poised to make the model say: the J-space readout.

    One forward pass and a decode, so the same call gives the workspace of the
    subject, the regulator, the actor or the introspector. `position` chooses
    the token to read at (see POSITIONS).
    """
    import torch

    llm.load()
    h = _residual_at(llm, messages, block_for(layer), position)
    W_U, norm = _unembed_and_norm(llm)
    with torch.no_grad():
        transported = lens.jacobians[layer] @ h
        if norm is not None:
            transported = norm(transported.to(llm.device)).float().cpu()
        logits = W_U @ transported
    top = torch.topk(logits, top_k)
    return [(llm.tokenizer.decode([int(i)]).strip(), float(v))
            for v, i in zip(top.values, top.indices)]


def logit_of(llm, lens: Lens, messages, layer: int, token: str,
             position: str = "last") -> float:
    """The J-space readout logit for one token, without ranking the vocabulary."""
    import torch

    llm.load()
    h = _residual_at(llm, messages, block_for(layer), position)
    W_U, norm = _unembed_and_norm(llm)
    tid = llm.tokenizer(token, add_special_tokens=False)["input_ids"][0]
    with torch.no_grad():
        transported = lens.jacobians[layer] @ h
        if norm is not None:
            transported = norm(transported.to(llm.device)).float().cpu()
        return float(W_U[tid] @ transported)


def toward_token(llm, lens: Lens, token: str, layer: int) -> Direction:
    """The J-space direction that pushes generation toward saying `token`."""
    import torch

    llm.load()
    tid = llm.tokenizer(token, add_special_tokens=False)["input_ids"][0]
    W_U, _ = _unembed_and_norm(llm)
    with torch.no_grad():
        row = W_U[tid]
        vector = lens.jacobians[layer].T @ row
    scale = float(_residual_at(llm, [_probe()], block_for(layer)).norm())
    return Direction(vector=vector, layer=block_for(layer), model=lens.model, scale=scale)


def layer_scale(llm, layer: int) -> float:
    """The activation norm at a layer's block; steering is a fraction of it.

    Shared by the lens steering and the emotion-vector steering, so one strength
    means one size of nudge for both families.
    """
    return float(_residual_at(llm, [_probe()], block_for(layer)).norm())


def toward_concepts(llm, lens: Lens, tokens: list[str], layer: int) -> Direction:
    """One J-space direction toward a set of concepts, the mean of their units.

    Used to hold a mind in a state described by several words at once, such as
    the self-awareness concepts the regulator and introspector are steered
    toward throughout their turn.
    """
    import torch

    dirs = [toward_token(llm, lens, t, layer) for t in tokens]
    vector = torch.stack([d.unit() for d in dirs]).mean(0)
    return Direction(vector=vector, layer=dirs[0].layer, model=lens.model,
                     scale=dirs[0].scale)


def _probe():
    from src.api.types.messages import ChatMessage
    return ChatMessage("user", "Hello.")
