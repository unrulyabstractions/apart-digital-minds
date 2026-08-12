# Measuring introspection with identity self-regulation

Does a J-space direction carry a model's identity anomaly?

A model is split into parts that share one conversation. A **regulator** looks
at itself and chooses to push the reply toward, away from, or leave alone a
first-person identity concept, read out as a set of tokens through Anthropic's
Jacobian lens. An **actor** writes under that steering, and an
**introspector** answers a forced choice about itself, scored on the
probability mass over the answers. The cases are two anomalies WeirdChat
surfaces in Qwen3.6-27B: a claim to a physical body and a denial of being an
AI, each seeded from matched transcripts that do and do not exhibit the
behaviour.

Three stop-gates check the intervention before any introspection is measured.
On Qwen3.6-27B they fail: the seeded arms do not separate on the token
direction at any lens layer, steering toward the concept moves the readout no
more than a norm-matched decoy, and every token direction in this lens shares
a common axis, so a decoy cannot be made orthogonal. The result of the
experiment is that a single J-space token direction does not carry these
identity anomalies. The frozen plan is `studies/identity/PLAN.md`; the report
is `paper/`.

## See the results

One UI shows the whole experiment: verdict, gate charts, preflight, and every
trial record. No model loads; it reads the run's output files.

```bash
uv run python ui/results_server.py
```

## Layout

```
studies/identity/   the pipeline: PLAN.md, tokens, questions, engine, rig,
                    gates (run.py, g1_scan.py), analyze, figures
src/dminds/         steering, the J-lens, the coherence gate
cloud/              the Vast.ai harness (launch, sync, capture-and-destroy)
ui/                 results_server.py + results.html: the one results UI
paper/              the sprint report; build with bash paper/build.sh
out/, sync/         run outputs and byte-verified copies pulled from the box
```

## Run

```bash
uv sync --extra hf                                      # torch + lens + datasets
uv run python -m studies.identity.run \
  --model hf:Qwen/Qwen2.5-7B-Instruct --cases A \
  --n-seeds 2 --n-perms 4 --smoke --stamp local1        # local shakedown (7B)
uv run python -m studies.identity.g1_scan \
  --model hf:Qwen/Qwen2.5-7B-Instruct --n-seeds 8 --stamp scanlocal
uv run python -m studies.identity.figures              # rebuild the paper figures
```

The 27B does not fit a 48 GB machine; `cloud/README.md` runs it on a rented
A100 and refuses to tear the box down until every result file is proven
captured by bytes.

## Earlier in the sprint

The emotion-regulation incarnation of this project (a four-part mind steering
its own emotional state, with a 2×2 introspector) lives in the git history and
in `studies/scenarios.py` + `out/studies/introspection/`; its results are in
the history of `paper/`.
