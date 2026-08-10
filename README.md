# Measuring introspection with emotional self-regulation

Can a model report the emotion it steered itself toward?

One project. A model is split into four parts that share a byte-identical
context and one set of weights. A **regulator** reads the **subject** in
J-space, chooses where to steer, and an **actor** answers under that steering.
An **introspector** answers under the same steering and is then asked what
shaped its reply. The J-space readout is recorded for all four parts.

**J-space** is the subspace surfaced by Anthropic's Jacobian lens: the
directions in the residual stream a model is poised to act on, and, the paper
argues, the part it can report and reason with. The regulator reads and writes
in that same space, so what it steers with is a token the readout can also
surface, and what the introspector claims to have felt can be scored against
what was applied.

The question is whether the introspector's report tracks the steering or only
the situation. Two controls decide it: the same part with no steering, and with
a target the regulator did not choose.

## Layout

```
src/api/       the contracts: what each part must do
src/dminds/    the implementations: pipeline, scheduler, steering, the J-lens
studies/       workspace.py (the study) and scenarios.py (the situations)
ui/            jspace_server.py + jspace.html: the replay viewer
tests/         115 tests, no dependencies
out/           what a run produces (gitignored)
```

## Run

```bash
uv sync --extra hf                               # install torch + the lens deps
uv run pytest                                    # the runtime
uv run python studies/workspace.py --model hf:Qwen/Qwen2.5-7B-Instruct --trials 3
uv run python ui/jspace_server.py                # live: send the mind a message
```

The study fetches the **fitted Jacobian lens** for the model on first use from
the `neuronpedia/jacobian-lens` repo (`src/dminds/llm/jspace.py:fetch_lens`).
The write side is derived from the lens rather than taken from it: the library
reads J-space, and steering along a J-space direction is one addition to the
residual stream, which the runtime's `steered` already does.

## The viewer

`ui/jspace_server.py` runs live: it loads the model and lens once, and each
message runs the four parts on the spot (~1.5 min on a 7B). The four parts show
side by side, and every part carries a J-space readout you can read at any token
position (`user`, `assistant`, `change-of-turn`) or turn off. The subject's
readout is computed up front; the others are computed on demand when you turn a
panel on, so a message pays only for what you look at.
