# Verification log

Every entry records what the output is, how it was checked, and the result.
`VERIFIED` means the artifact was opened and its real content inspected.
`UNVERIFIED` means it was not, and says why.

## 2026-07-29 — dmind v0.1.0 initial build

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 1 | `dmind` imports and runs a single agent | Ran an inline script, read the full console trace and the returned reply | VERIFIED |
| 2 | `examples/01_hello_agent.py` | Ran it, read stdout and stderr; two prompts answered, transcript held 5 messages | VERIFIED |
| 3 | `examples/02_think_interceptor.py` | Ran it, read the full trace. Confirmed the t=0 / t=1 / t=2 lock-step, and that the target's transcript carried a message tagged `edited_by=interceptor` | VERIFIED |
| 4 | `examples/03_bicameral.py` | Ran it, read the output. Confirmed `model calls by tick: [(1,'outer'), (1,'inner'), (2,'outer')]`, so both hemispheres called their models inside one tick | VERIFIED |
| 5 | `examples/04_memory_and_replay.py` | Ran it, read the output. Journal reloaded 1 episode from disk in a second mind; cassette replay reproduced the recorded answer; two replay traces were identical event for event | VERIFIED |
| 6 | Test suite, 50 tests | Ran `tests/run_tests.py`, read the per-test output | VERIFIED (50 passed, 0 failed) |
| 7 | Trace artifacts on disk | Opened `runs/<id>/trace.jsonl` and every `runs/<id>/modules/*.jsonl`. Parsed all 31 events as JSON, confirmed `seq` contiguous 0..30, confirmed per-module line counts sum to the trace total (16+7+6+2 = 31), confirmed every event carries `wall`, `tick`, `module`. Printed one `llm.response` event verbatim | VERIFIED |
| 8 | `pip install -e .` as documented in the README | Created a clean `python3 -m venv`, ran the command, then imported `dmind` from `/private/tmp` so repo-root cwd could not mask a failure | VERIFIED |
| 9 | Clean-environment rerun | Ran the full test suite and all four examples from the clean pip venv | VERIFIED (50 passed; 4/4 examples rc=0) |

### Bug found and fixed during verification

Run IDs used second precision, so `examples/01` and `examples/02` both wrote to
`runs/20260729-215354/`, interleaving two experiments into one trace file. Fixed
by moving to millisecond precision plus a collision check against the run
directory (`_fresh_run_id` in `dmind/mind.py`). Re-verified: three back-to-back
example runs produced three distinct directories.

### UNVERIFIED

These code paths were never executed, because no API keys and no local model
server were available in this session. They are written but unproven.

| Output | Why it is unverified |
|---|---|
| `dmind/llm/providers/openai_.py` | Never called against a real endpoint. No `OPENAI_API_KEY`. |
| `dmind/llm/providers/anthropic_.py` | Never called against a real endpoint. No `ANTHROPIC_API_KEY`. |
| `dmind/llm/providers/gemini_.py` | Never called against a real endpoint. No `GEMINI_API_KEY`. |
| `dmind/llm/providers/ollama_.py` | No Ollama server was running. The HTTP request path never executed. |
| `dmind/llm/providers/hf_.py` | `torch` and `transformers` are not installed here. Weight loading and generation never executed. |

What *was* verified about these five: each constructs without a key, registers
under the right spec, and parses its spec correctly (`tests/test_llm.py`). Only
the network and generation paths are unproven.

Before trusting any of them, run one live call per provider and add an entry
here.
