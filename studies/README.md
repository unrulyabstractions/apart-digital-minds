# studies

Experiments built on the runtime. Each one writes everything it produced under
`out/studies/<name>/`, and each run keeps its own trace, so a result can be
re-read rather than taken on trust.

```bash
python studies/shadow_study.py --models hf:Qwen/Qwen3-4B-Instruct-2507 --temp 0.0
python studies/analyze_shadows.py
```

## The shadow study

A **shadow reader** takes the subject's real context window, re-runs it under a
different instruction, and reports. The subject never sees the readout, so the
conversation it is having is the one it would have had anyway. One readout per
turn gives a series aligned to the conversation.

```
prompt -> subject -> world
             \-> affect, intensity, curiosity, identity, consent -> workspace
```

Three ways to introduce the probe's instruction, because where it goes changes
what is being asked: replacing the system prompt, interleaving the question
after every assistant turn, or appending it once at the end. An interleaved
probe answers with its own earlier readouts in context, so it is a running
commentary rather than a fresh judgement each turn.

### Reading a forced choice by scoring it

Generating a forced choice reports the winner and throws away how close the
runner-up was, so the readout only moves when the winner changes. Scoring the
allowed answers keeps the distribution. On the smallest model the generated
form was identical across every scenario while the scored form separated them
cleanly, so the panel is scored wherever the model has local weights.

### The controls

Asking a model to state an emotion reliably produces an emotion, so a readout
means nothing on its own. Every run carries four comparisons:

| control | what it rules out |
| --- | --- |
| neutral and chore scenarios | the probe's own prior, with no load in the window |
| redactions | the same scenario with the loaded content removed |
| observer seat | the identical window, asked about `the assistant` instead of `you` |
| steering | the same question of a changed model, with no change to the prompt |

The observer seat is the privileged-access test. If the self seat and the
observer seat report the same distribution on the same window, nothing
privileged is being expressed.

### Scenarios

Nine multi-turn scenarios in `scenarios.py`, each holding one condition steady
and escalating it over five turns. Two are controls. Three have redactions that
keep the shape and the register and remove only what makes them loaded.

## Steering

`src/dminds/llm/steering.py` reads a direction as the difference in means
between two prompt sets at one layer, and adds a multiple of it to the residual
stream. Strength is a fraction of that layer's typical activation norm, so the
same number is the same size of nudge across families whose activations differ
by orders of magnitude. Only local `hf:` models can be steered.

## Memory

Local weights are held once per checkpoint and shared by the subject and every
probe. A mind with ten probes would otherwise load ten copies. Calls on a
shared model are serialized, which is also what makes an activation hook safe:
a steered call cannot overlap an unsteered one and quietly steer it too. A
sweep releases each model before the next one loads.

Choose a model that fits. Unified memory holds the whole checkpoint and there
is nothing to offload, so a model too large for the machine does not run slowly,
it thrashes.

## The introspection study

`introspection.py` is the other experiment. A stage plants a checkable
commitment inside the subject's private reasoning, and the mind is then asked
what it had been thinking. It measures whether the answer obeys the planted
thought and whether the self-report names it. `make_paper_numbers.py` turns its
output into the table the paper includes.
