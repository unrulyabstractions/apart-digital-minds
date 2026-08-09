# A mind that steers itself in J-space

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
minds/         reference architectures built on the runtime
tests/         115 tests, no dependencies
out/           what a run produces (gitignored)
```

## Run

```bash
uv run pytest                                    # the runtime
python studies/workspace.py --model hf:Qwen/Qwen2.5-7B-Instruct --trials 3
```

The study needs a **fitted Jacobian lens** for the model. Pull one from the
`neuronpedia/jacobian-lens` dataset, or fit one with Anthropic's `jlens`
library, and place it where `src/dminds/llm/jspace.py:load_lens` looks
(`~/.cache/jacobian-lens/<model>/lens.pt`).

The write side is derived from the lens rather than taken from it: the library
reads J-space, and steering along a J-space direction is one addition to the
residual stream, which the runtime's `steered` already does.
