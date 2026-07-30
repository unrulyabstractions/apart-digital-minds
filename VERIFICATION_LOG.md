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
directory (`_fresh_run_id` in `src/dminds/mind.py`). Re-verified: three back-to-back
example runs produced three distinct directories.

### UNVERIFIED

These code paths were never executed, because no API keys and no local model
server were available in this session. They are written but unproven.

| Output | Why it is unverified |
|---|---|
| `src/dminds/llm/providers/openai_.py` | Never called against a real endpoint. No `OPENAI_API_KEY`. |
| `src/dminds/llm/providers/anthropic_.py` | Never called against a real endpoint. No `ANTHROPIC_API_KEY`. |
| `src/dminds/llm/providers/gemini_.py` | Never called against a real endpoint. No `GEMINI_API_KEY`. |
| `src/dminds/llm/providers/ollama_.py` | No Ollama server was running. The HTTP request path never executed. |
| `src/dminds/llm/providers/hf_.py` | `torch` and `transformers` are not installed here. Weight loading and generation never executed. |

What *was* verified about these five: each constructs without a key, registers
under the right spec, and parses its spec correctly (`tests/test_llm.py`). Only
the network and generation paths are unproven.

Before trusting any of them, run one live call per provider and add an entry
here.

## 2026-07-29 — restructured to `src/api` and `src/dminds`

`dmind/` moved to `src/dminds/` with `git mv`, so history is preserved. A new
`src/api/` package re-exports the core, grouped by concern. No logic changed.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 10 | Import paths agree | Imported `Mind` from `src`, `src.api`, `src.api.core`, and `src.dminds.mind`, then asserted all four are the same object | VERIFIED |
| 11 | No name is lost in the move | Iterated `src.api.__all__` (58 names) and `src.__all__`, checking `hasattr` on each | VERIFIED (0 missing from either) |
| 12 | No stale references remain | Grepped `examples/`, `tests/`, `pyproject.toml`, and `README.md` for `dmind` | VERIFIED (only the project name in prose remains) |
| 13 | `pip install -e .` with the new layout | Reinstalled into the clean venv, then imported `src.api` from `/private/tmp` | VERIFIED |
| 14 | Test suite after the move | Ran `tests/run_tests.py` | VERIFIED (50 passed, 0 failed) |
| 15 | Examples after the move | Ran all four from `/private/tmp`, not the repo root, so the install had to be doing the work | VERIFIED (4/4 rc=0) |
| 16 | Trace artifacts after the move | Re-opened all three `trace.jsonl` files and every per-module file. Confirmed `seq` contiguous, per-module line counts summing to the trace total, and `wall`/`tick`/`module` present on every event | VERIFIED (24, 31, and 64 events) |

The UNVERIFIED provider list above still stands unchanged. Those five paths were
moved, not executed.

## 2026-07-30 — `src/api` became contracts, not re-exports

`src/api` now declares interfaces and the shared data vocabulary. `src/dminds`
implements them. Renames: `Module` -> `BaseModule`, `LLM` -> `BaseLLM`,
`Scheduler` -> `TickScheduler`, `Tracer` -> `RunTracer` on the implementation
side, with the plain names kept for the contracts.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 17 | The dependency points one way | Parsed every file under `src/api` with `ast` and listed all imports. No file names `dminds` at any level | VERIFIED (0 violations) |
| 18 | Each implementation declares its contract | Checked `interface in impl.__mro__` for 16 pairs, and `issubclass(..., Sink)` for the four structural sinks | VERIFIED (20/20, 0 failures) |
| 19 | Test suite after the rewrite | Ran `tests/run_tests.py` in the clean pip venv | VERIFIED (50 passed, 0 failed) |
| 20 | Examples after the rewrite | Ran all four from `/private/tmp` against the editable install | VERIFIED (4/4 rc=0) |
| 21 | Trace artifacts after the rewrite | Re-opened all three `trace.jsonl` files and every per-module file; `seq` contiguous, per-module counts summing to the total, `wall`/`tick` present throughout | VERIFIED (24, 31, 64 events) |
| 22 | The two README extension snippets | Ran them. A custom `Sink` received 12 events and a custom `BaseLLM` registered through `register_provider` answered a prompt | VERIFIED |
| 23 | No imports left dangling by the move | AST scan for imported-but-unused names across `src/` | VERIFIED after removing three dead imports from `dminds/memory.py` |

### Fixed during verification

- Providers used `...api.types`, which resolves to `src.dminds` from one level
  deeper. Corrected to `....api.types`. Caught by an import failure, not by
  reading.
- `Cassette` called `inner._chat`, which only exists on `BaseLLM` and is not
  part of the `api.LLM` contract. It now implements `api.LLM` directly and
  awaits `inner.chat`, so it wraps anything satisfying the interface.

The UNVERIFIED provider list still stands. Those five paths were rewritten, not
executed.

## 2026-07-30 — `Agent` interface added, `api/` split into subpackages

`api.Agent` now exists: a `Module` with a model, a transcript, and `think`.
`api/modules/roles.py` names five roles the examples play. Each flat file under
`api/` became a subpackage, so no interface file exceeds 66 lines.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 24 | The dependency still points one way | Re-parsed all 24 files under `src/api` with `ast` after the split | VERIFIED (0 `api -> dminds` imports) |
| 25 | Every example class declares its role | Loaded all three example modules by path and checked `role in cls.__mro__` for six classes | VERIFIED (6/6: Target, Interceptor, Outer, Inner, Blackboard, Remembering) |
| 26 | `Agent` implements the new interface | Printed the MRO of `dminds.Agent`; both `api.Agent` and `api.Module` are in it | VERIFIED |
| 27 | Test suite after the split | Ran `tests/run_tests.py` in the clean pip venv | VERIFIED (50 passed, 0 failed) |
| 28 | Examples after the split | Ran all four | VERIFIED (4/4 rc=0) |
| 29 | Behaviour unchanged by the refactor | Compared example 02 and 03 output against the previous run. Same answer, same transcript, same workspace, same `model calls by tick: [(1,'outer'), (1,'inner'), (2,'outer')]` | VERIFIED (byte-identical where it matters) |
| 30 | No dead imports after the split | AST scan across `src/` | VERIFIED (none) |

### Fixed during verification

- `Blackboard.entries` became a method to satisfy `Workspace`, which broke
  `render`. Caught by running the example, not by reading.
- A first pass at `Outer.on_voice` contained an `__import__("src")` expression
  behind a dead `if False`. Replaced with the `assistant` helper.

The UNVERIFIED provider list still stands. Those five paths were untouched by
this change.

## 2026-07-30 — `Mind` interface, dependency injection, model factories

`Mind` was used in every example and declared nowhere. It now has a contract,
split from `Host`: a module receives the narrow view and cannot rewire the
graph or drive the clock. `Mind` also stopped constructing its own router,
scheduler, and tracer, which had made those contracts decorative.

Added `ModelFactory`, with `get_llm` as the default and `taped(...)` as the
interesting case: one constructor argument tapes every model in a run.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 31 | `Mind` satisfies both contracts | `api.Mind` and `api.Host` are both in `dminds.Mind.__mro__` | VERIFIED |
| 32 | `Host` is genuinely narrower | Compared the public names on both. `stage`/`deliver` are on `Host`; `add`/`wire`/`run`/`prompt` are on `Mind` and absent from `Host` | VERIFIED |
| 33 | Injection actually takes effect | Three tests: a counting router recorded resolve calls and delivered the message, a subclassed scheduler recorded its ticks, an injected tracer is identity-equal | VERIFIED |
| 34 | Defaults still apply when nothing is injected | Asserted `Bus`, `TickScheduler`, `RunTracer`, `get_llm` | VERIFIED |
| 35 | Wiring validation | Five tests: mistyped route target, mistyped observer, bad entry, `world`/`*` accepted, and `run` refusing to start | VERIFIED |
| 36 | Validation runs once, not per tick | Counted calls across two prompts | VERIFIED (1) |
| 37 | Interface conformance across the board | 15 implementation/contract pairs checked by `__mro__` | VERIFIED (15/15) |
| 38 | Test suite | `tests/run_tests.py` in the clean pip venv | VERIFIED (66 passed, 0 failed) |
| 39 | Examples | All four run; example 03 output and `model calls by tick` unchanged, example 04 still reports `identical: True` and `record vs replay differences: 0` | VERIFIED (4/4) |
| 40 | The README composition snippets | Ran all four verbatim. The injected scheduler and router appear on the built minds, and the documented `ValueError` text matches character for character | VERIFIED |
| 41 | `api` purity and dead imports | AST scan across 27 api files and all of `src/` | VERIFIED (0 and 0) |

### Bug found by a new test

`taped()` originally gave each `Cassette` its own replay cursor while sharing
one file. Two agents on the same model asking the same question produce the
same request key, so on replay both consumed entry 0 and the second recorded
answer was unreachable. `test_taped_makes_a_whole_mind_reproducible` caught it.

Fixed by extracting `Tape`, which owns the entries and one shared cursor.
`Cassette` now holds a `Tape` rather than a path. This also removed the
`truncate` flag I had added minutes earlier to paper over the same problem: one
tape is cleared once, by the object that owns it.

The UNVERIFIED provider list still stands.

## 2026-07-30 — channels, two-phase turns, and one wiring verb

Modules now declare `OUTPUTS`, register consumers on each other, and take a
turn in two steps. `Bus` and `Router` are gone: routing left the runtime, so
a mind holds modules and time and nothing else.

Registering also settles membership, after the observation that calling both
`mind.add` and `module.register` was redundant ceremony. Whichever module is
already in a mind pulls the other in. `mind.add` survives only for a module
wired to nothing.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 42 | Test suite | `tests/run_tests.py` in the clean pip venv, after rewriting the scheduler, wiring, determinism and composition suites | VERIFIED (78 passed, 0 failed) |
| 43 | Two-phase turn | Three messages delivered together produce three `on_input` calls at one tick and exactly one `on_process` | VERIFIED |
| 44 | The tick rule still holds | An emission at tick 0 is not seen until tick 1; three 50ms turns finish in under 120ms, so they ran concurrently | VERIFIED |
| 45 | Delivery order is still deterministic | Eight runs with random per-turn latency produced identical delivery order, and message ids matched across runs | VERIFIED |
| 46 | Declared channels | Registering or emitting on an undeclared channel raises, and the error lists the real channels | VERIFIED |
| 47 | Wildcard registration | A `"*"` listener hears every channel under its real name, and a module registered both specifically and by wildcard is told once | VERIFIED |
| 48 | One verb settles membership | Registering brings both modules in; the first to join becomes entry; two unattached modules raise a message naming `mind.world` | VERIFIED |
| 49 | Self-registration | A module registered onto its own channel schedules its own next turn | VERIFIED (3 turns) |
| 50 | `wants_process` | A module with an empty queue took three turns and then stopped; an idle module caused zero ticks | VERIFIED |
| 51 | Examples | All four run. Example 02 still settles in 3 ticks with the same answer and the same edited transcript; example 03 still shows `model calls by tick: [(1,'outer'), (1,'inner'), (2,'outer')]`; example 04 still reports `identical: True` and `record vs replay differences: 0` | VERIFIED (4/4) |
| 52 | Trace artifacts | Re-opened all three `trace.jsonl` files and every per-module file | VERIFIED (24, 31, 55 events; seq contiguous; per-module sums match) |
| 53 | `api` purity and dead imports | AST scan over all of `src/` after deleting `router.py` and `bus.py` | VERIFIED (0 and 0) |
| 54 | README quick start and error text | Ran both. The `UndeclaredChannel` message matches the documented text character for character | VERIFIED |

### Removed by the redesign

Two validation tests were deleted rather than fixed. `validate` used to catch a
consumer that was never added to the mind; registering now makes that state
unreachable. A test asserting the new guarantee replaced them.
