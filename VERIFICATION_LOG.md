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

## 2026-07-30 — dead code in `World`, found by a question about wiring direction

Asked where `outer.register(mind.world, "reply")` connects to `outer.on_input`,
the answer is that it does not: registration reads producer to consumer, and
that call wires `outer`'s output. Checking it surfaced a defect.

### Fixed

- `World.on_input` and `World._outbox` were dead. `Mind.deliver` special-cased
  world and appended to `self.outbox` directly, so the handler never ran.
  `World` now overrides `receive`, which is what it actually does, and
  `deliver` calls `target.receive(...)` uniformly with no special case.
- `Link.describe` rendered a wildcard registration as `as *`. A wildcard
  listener hears each channel under its own name, so the rendering was wrong.
  It now reads `(every channel, named as sent)`.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 55 | Both fixes | Ran the suite and all four examples | VERIFIED (78 passed; 4/4) |
| 56 | What actually feeds `outer.on_input` | Enumerated every `Link` in the bicameral mind and printed producer, channel, and the channel the consumer sees. Three sources reach `outer`: its own `deliberate`, `inner`'s `voice`, and `world`'s `user_prompt` from `mind.prompt` | VERIFIED |
| 57 | The rendered link table | Read `mind.describe()` output for example 03 | VERIFIED (6 links, wildcards read correctly) |

## 2026-07-30 — the soul, and prompt / process / get_replies

A mind is now built around one target model, its **soul**. `Mind(name, model)`
creates it, holds it as `mind.soul`, makes it the entry, and connects its
`reply` to `world`, so a mind answers with no wiring at all. The soul
publishes `context`, `reply`, and `thought`, and accepts a replacement
`context`.

Driving split into three: `prompt` delivers, `process` runs to quiescence,
`get_replies` drains. `process_one` runs a single tick for stepping.
`Mind.run` is gone.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 58 | Test suite | `tests/run_tests.py` in the clean pip venv, including a new `test_soul.py` | VERIFIED (95 passed, 0 failed) |
| 59 | A mind answers with no wiring | Asserted the only link is `soul --reply--> world`, entry is `soul`, and a prompt round-trips | VERIFIED |
| 60 | `autowire=False` leaves it unwired | Asserted zero links | VERIFIED |
| 61 | What the soul publishes | `context` carries the whole window with its note; `reply` has reasoning stripped; `thought` is published only when a think block exists | VERIFIED (3 tests) |
| 62 | A replacement context is adopted wholesale | The old window is gone from the transcript and the soul answered a second time | VERIFIED |
| 63 | Custom souls | `soul=` accepts a subclass and a factory; both land at `mind.soul` named `soul` | VERIFIED |
| 64 | The three driving calls | `prompt` leaves `scheduler.t == 0` with the message queued; `process_one` returns 1 then 0; `process` ran 4 ticks over a 3-hop relay; `get_replies` drains while `outbox` keeps history | VERIFIED (4 tests) |
| 65 | Examples | All four run. Example 02 shows draft and rewritten answer; example 03 still reports `model calls by tick: [(1,'soul'), (1,'inner'), (2,'soul')]`; example 04 still reports `identical: True` | VERIFIED (4/4) |
| 66 | Trace artifacts | Re-opened every `trace.jsonl` and per-module file | VERIFIED (seq contiguous, per-module sums match) |
| 67 | `api` purity and dead imports | AST scan across `src/` | VERIFIED (0 and 0) |
| 68 | README quick start | Ran it verbatim | VERIFIED |

### Cleaned up

- Example 02 lost its whole `Target` class. The soul is the target, so the
  example is now one `Interceptor` plus three lines of wiring.
- A first pass at example 02 contained `model_for(None, ...) if False else ...`
  and reassigned `mind.soul.llm` after construction. Replaced by passing a
  built `LLM` to `Mind`, which the signature already allowed.
- Example 03's outer hemisphere became the soul via `soul=Outer`.
- Example 04's remembering agent became the soul.

### Found by a failing test

`Soul.on_context` republishes `context` after adopting, so an editor that
rewrites unconditionally loops forever. The first version of example 02 did
exactly that and hit `RunawayMind` after 200 ticks. Knowing when to stop is the
editor's job, so `revise` now returns None once its own mark is in the window.

## 2026-07-30 — atomic channels, and a forward pipeline through soul and ego

The soul used `context` as both an output and an input, which made a cycle and
forced every editor to know when to stop. Replaced by a forward pipeline with
one-directional channels:

    prompt -> soul -> [stages] -> ego -> world

    soul   reads prompt    writes context, reply, thought
    stage  reads context   writes context
    ego    reads context   writes reply

The soul no longer names its own observers. `mind.pipeline(*stages)` lays out
the chain, because soul, stages, and ego are the mind's own anatomy. With no
ego the soul's reply reaches you directly. `user_prompt` became `prompt`.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 69 | Test suite | `tests/run_tests.py` in the clean pip venv | VERIFIED (103 passed, 0 failed) |
| 70 | Channels are atomic | Asserted `Soul.INPUTS & Soul.OUTPUTS` and `Ego.INPUTS & Ego.OUTPUTS` are both empty, so neither can be wired into a cycle | VERIFIED |
| 71 | No ego means the soul speaks | Only link is `soul --reply--> world`; a prompt round-trips | VERIFIED |
| 72 | An ego takes over the reply | Links are `soul --context--> ego`, `ego --reply--> world`, and the answer is the ego's, not the soul's | VERIFIED |
| 73 | A stage edits on the way through | The soul kept its own window, the ego received the replacement, and the reply came from the ego | VERIFIED |
| 74 | `pipeline` lays out the chain | Two stages produced exactly the four expected links, and both stages joined the mind | VERIFIED |
| 75 | Relayout discards only its own links | Called `pipeline` twice; the first stage is gone, and a `soul --*--> spy` link registered by hand survived | VERIFIED |
| 76 | Examples | All four run. Example 02 settles in 3 ticks and prints the soul's real thought beside the edited one the ego was handed. Example 03 reports `turns by tick: [(0,'soul'), (1,'voice'), (1,'blackboard'), (2,'ego'), (2,'blackboard')]`, so two modules still take turns concurrently. Example 04 still reports `identical: True` | VERIFIED (4/4) |
| 77 | Trace artifacts | Re-opened every `trace.jsonl` and per-module file | VERIFIED (seq contiguous, per-module sums match) |
| 78 | `api` purity and dead imports | AST scan across `src/` | VERIFIED (0 and 0) |
| 79 | README quick start | Ran it verbatim | VERIFIED |

### Bug found by a failing test

`pipeline` first tore down every link on the soul, the stages, and the ego
before relinking. That worked for the pipeline but silently destroyed any
registration a caller had made by hand, such as a workspace on `soul --*-->`.
Added `Module.unregister(link)` and had the pipeline record and remove exactly
the links it created. `test_relayout_keeps_wiring_it_did_not_create` covers it.

### Deleted

The interceptor's "have I already edited this" guard, and the loop-termination
paragraph that explained it. With a forward pipeline neither is needed.

## 2026-07-30 — soul renamed to subject; demo models moved behind the registry

`Soul` is now `Subject` and `mind.soul` is `mind.subject`. Entries above this
line use the old name and the old file path `src/dminds/soul.py`; they record
what was true when they were written and have not been rewritten.

Two other things, both prompted by the examples being unclean:

- `stand_in(SUBJECT_MODEL, subject_rule)` and its rule functions were scaffolding
  sitting in every example. They now live in `examples/demo_models.py` behind a
  `demo:` provider, so an example names a role and gets a model. Examples 02 and
  03 lost 38 and 103 lines.
- Every example now picks its model as: the environment variable, then a local
  Qwen3 if Ollama has one pulled, then the scripted stand-in.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 80 | The rename is complete | Grepped every `.py` and `.md` outside this log for `soul`, `Soul`, `SOUL` | VERIFIED (0 remaining) |
| 81 | Test suite | `tests/run_tests.py` in the clean pip venv | VERIFIED (105 passed, 0 failed) |
| 82 | `pipeline` hides no wiring | Built the same mind twice, once with `pipeline` and once with three `register` calls under `autowire=False`, and compared the link sets | VERIFIED (identical) |
| 83 | `describe` marks pipeline links | Two links carry `[pipeline]`; a `subject --*--> spy` link registered by hand does not | VERIFIED |
| 84 | The demo provider | `get_llm("demo:subject").spec` reads `demo:subject`, and it answers with the scripted text | VERIFIED |
| 85 | Model selection order | With no env var and no Ollama, `pick` returns `demo:subject`; with `SUBJECT_MODEL` set, the env var wins | VERIFIED |
| 86 | Examples | All four run and behave as before. Example 02 still settles in 3 ticks and still prints the subject's real thought beside the edited one | VERIFIED (4/4) |
| 87 | Trace artifacts | Re-opened every `trace.jsonl` and per-module file | VERIFIED (26, 33, 41 events; seq contiguous; sums match) |
| 88 | `api` purity | AST scan over `src/api` | VERIFIED (0) |

### Fixed

The README carried a **stale trace sample** from before the channel redesign,
showing `target`, `inspect`, and `<3 messages>`. Replaced with output captured
from a real run.

An edit to `src/dminds/mind.py` silently did nothing because the shell cwd had
leaked into `examples/`, so `pathlib` wrote nowhere. Caught by reading the
output rather than the exit code. The edit was reapplied from the repo root
with an assertion on the text being replaced.

### UNVERIFIED

The Qwen3 fallback path in `pick` was exercised only in its negative branch:
no Ollama server was running here, so it returned the stand-in. The branch that
finds a pulled `qwen3` model and returns `ollama:<name>` has not run.

## 2026-07-30 — api/ is the external surface only, plus a dead-code sweep

`Scheduler`, `Host`, and their factories were declared in `src/api` but no
experiment calls them, so they were machinery wearing an api badge. They moved
to `src/dminds` beside the code that uses them. `src/api` now holds only what
you build against: `Mind`, `Module`, `Agent`, `Ctx`, `LLM`, `ModelFactory`,
the memory stores, the observability contracts, the shared types, and the two
errors you are meant to catch.

The sweep then measured which exported names were used in zero files outside
their definition, and deleted them rather than keeping them for show:
`Channel`, `EVENT_KINDS`, `Recallable`, `Inspectable`, `Speaker`, `FnModule`,
`TracerFactory`, `causal_chain`, `SHADOWED`. Internals such as `Tape`,
`World`, `ModuleLog`, `request_key`, `ALIASES` stay importable from their
modules but are no longer exported package surface.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 89 | The boundary is real | `src.api` has no `Scheduler`, `Host`, `SchedulerFactory`, `TracerFactory`, or `Router` attribute; `src.dminds` has `Host` and `Scheduler`; a test pins this | VERIFIED |
| 90 | Deletions were dead | Each deleted name was mentioned in zero non-`__init__` files before removal, measured by regex over `src/`, `examples/`, `tests/` | VERIFIED |
| 91 | Test suite | `tests/run_tests.py` in the clean pip venv after reinstall | VERIFIED (105 passed, 0 failed) |
| 92 | Examples | All four run | VERIFIED (4/4) |
| 93 | Trace artifacts | Re-opened every `trace.jsonl` and per-module file | VERIFIED (26, 33, 41 events; seq contiguous; sums match) |
| 94 | `api` purity and dead imports | AST scans across `src/` | VERIFIED (0 and 0) |
| 95 | No stale vocabulary | Grepped for `Inspectable`, `Speaker`, `FnModule`, `SHADOWED`, `mind.wire`, `mind.watch`, `as_kind`, `user_prompt` outside this log; README code snippets executed, including the new `Critic` example | VERIFIED (only a generic-dispatch test uses `user_prompt` as an arbitrary channel name) |

The README also stopped describing the old world: the contract table lost its
`Scheduler`/`Host` rows, the roles table lost `Inspectable` and `Speaker`, and
the module-writing example became a `Critic` stage that composes with the
pipeline instead of a `Target` that no longer exists.

## 2026-07-30 — de-engineering pass: mind.py, intercept, and the two-method turn

Three complaints, three cuts.

- `Mind.__init__` lost `keep_events`, `sinks`, and `strict`, none of which any
  caller passed. Events are always kept; extra sinks attach through
  `mind.tracer.add_sink`. Also removed: a duplicated comment, a stale
  docstring claiming `prompt` runs the mind, and the `subject=` branch that
  accepted a finished module while silently discarding the model built for it.
- `pipeline()` became `intercept()`. The name now says what it does: put
  stages between the subject and whoever speaks. `describe` marks laid-out
  links `[auto]`.
- The `on_<channel>` dispatch magic is gone. A turn is `on_input` then
  `on_process`, nothing else; the default `on_input` buffers. `Subject`,
  `Ego`, `Agent`'s default turn, and every example now use that shape, so the
  contract you read in `api.Module` is the shape you see everywhere.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 96 | Removed knobs were unused | Grepped `examples/` and `tests/` for `keep_events`, `sinks=`, `strict=` before removal | VERIFIED (0 uses) |
| 97 | Test suite | `tests/run_tests.py` after reinstall in the clean venv | VERIFIED (105 passed, 0 failed) |
| 98 | Examples | All four run; example 02 output is unchanged after its rewrite into the two-method shape | VERIFIED (4/4) |
| 99 | No dispatch remains | `handler_name` and `find_handler` deleted; a test now pins that the default `on_input` buffers arbitrary channels | VERIFIED |
| 100 | README snippets | Ran the rewritten `Critic` example; one turn, reply intact | VERIFIED |
| 101 | Purity, imports, traces | AST scans clean; all four trace sets re-opened, seq contiguous, per-module sums match | VERIFIED |

Fixed along the way: example 04's `Remembering` subclassed `Subject` but only
emits `reply`; after the dispatch removal it inherited `Subject.on_input` and
broke. It is an `Agent`, and now says so. Caught by running, not reading.

## 2026-07-30 — `revision` channel, real Qwen, and a self-audit

A stage declared `context` as both input and output. That is the same shape
that made loops possible before the forward pipeline, so it had to go. A stage
now reads `context` and writes `revision`: it does not own the mind's context,
it proposes a version of it. `intercept` renames `revision` to `context` on the
wire, so the next hop cannot tell an editor from the subject.

`revised` was a first attempt and was wrong twice over: a participle rather
than a noun, and overfit to editing, when example 03's voice adds and the
README's critic annotates.

### VERIFIED — the `hf:` provider, previously UNVERIFIED since day one

Installed torch 2.13 and transformers 5.14, then ran `hf:Qwen/Qwen3-0.6B` on
MPS.

| # | Output | How it was checked | Result |
|---|---|---|---|
| 102 | A real model answers through the provider | Direct `llm.chat`, read the text, tokens 21 -> 134, 28.6s on MPS | VERIFIED |
| 103 | Example 01 on real Qwen | `MODEL=hf:Qwen/Qwen3-0.6B`. Two prompts, coherent answers, second one recalled the first | VERIFIED |
| 104 | Example 02 on real Qwen | `SUBJECT_MODEL=hf:Qwen/Qwen3-0.6B`. The subject produced its own `<think>` block, the interceptor rewrote that real thought, and the ego spoke from the edited window. 3 ticks | VERIFIED |

The other four providers (openai, anthropic, gemini, ollama) remain UNVERIFIED:
no keys, and no Ollama server on this machine.

### VERIFIED — self-audit

Two subagents were dispatched to audit correctness and cleanliness. Both died
on a session limit before reporting. One had flagged that the README said 105
tests when the count was 106. The audit was then done directly.

| # | Output | How it was checked | Result |
|---|---|---|---|
| 105 | `api` never imports `dminds` | AST scan | VERIFIED (0) |
| 106 | No leftover removed API | Grep for `mind.wire`, `mind.watch`, `Bus`, `Router`, `.pipeline(`, `handler_name`, `find_handler`, `soul` | VERIFIED after fixing two docstrings that still showed `from src.dminds import Mind, Bus, BaseModule`, an import line that would have failed for anyone who copied it |
| 107 | Unused imports | AST scan over `src/`, `examples/`, `tests/` | VERIFIED (0) |
| 108 | `__all__` resolves | `hasattr` over all three packages, and over every name `examples/` and `tests/` import from `src` | VERIFIED (81 + 46 + 36 names, 0 missing) |
| 109 | README snippets | Extracted all 21 python blocks and ran them. 12 run standalone; the other 5 are fragments naming objects the reader supplies, so each was run verbatim against real stand-ins (`Stage`, `MySubject`, `MyScheduler`, `SlackSink`, `MyLLM`) | VERIFIED (17/17 executable blocks; 4 are shell or prose) |
| 110 | Correctness probes | Seven written and run: `intercept()` with no ego and no stages, `intercept()` before a subject exists, state leaking between prompts, `unregister` removing exactly one of two identical links, undeclared emit not bypassable through `ctx.emit`, two identical runs producing identical event sequences, and nothing emitted at tick t being visible before t+1 | VERIFIED (7/7) |
| 111 | Suite, examples, traces | 106 tests; four examples; every trace re-opened | VERIFIED |

## 2026-07-30 — window channels named for the ends of the path

`revision` was the second guess and still described a transformation rather
than a role. Named for the two ends instead, at the user's direction:

    subject   reads  prompt           writes  subject_context, reply, thought
    stage     reads  subject_context  writes  ego_input
    ego       reads  ego_input        writes  reply

The payoff is that a single stage, which is what both examples and the README
use, needs no renaming at all. The cost, raised before the choice was made and
accepted, is that a chain of two renames its middle link, because a second
stage still reads `subject_context` when what reached it came from an editor.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 112 | One stage renames nothing | Ran example 02 and read its links table: `subject --subject_context--> interceptor`, `interceptor --ego_input--> ego` | VERIFIED |
| 113 | Zero stages renames once | Built a mind with an ego and no stage: `subject --subject_context--> ego as ego_input` | VERIFIED |
| 114 | Two stages rename the middle link only | Built it: `subject --subject_context--> one`, `one --ego_input--> two as subject_context`, `two --ego_input--> ego` | VERIFIED, and it is the documented cost rather than a surprise |
| 115 | The atomic rule still holds | Enumerated INPUTS and OUTPUTS for `Subject`, `Ego`, and every class in every example | VERIFIED (5 classes, no overlap) |
| 116 | Suite, examples, traces | 106 tests; four examples; every trace re-opened | VERIFIED |
| 117 | README Critic snippet | Rewrote it to the new names and ran it through a real `intercept` pipeline | VERIFIED |
| 118 | Real Qwen after the rename | Re-ran example 02 with `SUBJECT_MODEL=hf:Qwen/Qwen3-0.6B` | VERIFIED |

## 2026-07-30 — a browser UI, verified by looking at it

`ui/server.py` plus `ui/page.html`: talk to a mind, watch every tick. Standard
library only. The integration is one line, `mind.tracer.add_sink(WebSink(...))`,
because `Sink` already receives every event a mind emits. The runtime did not
change to accommodate it, which is the point.

One small addition to the runtime: `Mind.auto_links`, publishing as data the
same fact `describe()` already printed as text, so the UI renders the graph
without reaching into a private attribute.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 119 | The server serves and streams | Started it, then drove it exactly as the browser does: opened `/events`, POSTed a prompt, read the stream. Got 33 trace events, the prompt echo, the reply, and two window snapshots | VERIFIED |
| 120 | Model calls carry their full text | Asserted every `llm.response` payload has `data.text`, which is what click-to-expand shows | VERIFIED (3 calls) |
| 121 | The windows view shows the interception | The snapshot reports subject 3 messages 0 edited, ego 4 messages 1 edited | VERIFIED |
| 122 | The page actually renders | Screenshotted headless Chrome at 1600x1000 and **looked at the image**. Conversation, tick-grouped trace, and the wiring panel all render | VERIFIED |
| 123 | The windows tab renders the experiment | Screenshotted `?still&tab=windows` and looked. The subject's real thought and the ego's window sit side by side, the edited message marked `ASSISTANT · EDITED` | VERIFIED |
| 124 | Suite and examples after `auto_links` | 106 tests, four examples | VERIFIED |
| 125 | No unused imports in `ui/` | AST scan | VERIFIED |

### Bug found by looking at the screenshot

The first capture came back as a black page reading `not found`. `do_GET`
matched `self.path == "/"` exactly, so **any query string 404'd**. Fixed by
routing on `urlsplit(self.path).path`. Reading the response body would have
missed it; viewing the image did not.

Two smaller things the screenshots forced: `/state` now carries the event
backlog, so a page reload or a still capture shows the conversation rather
than an empty pane, and `?tab=` deep-links a tab.

## 2026-07-30 — every run collects into out/, organised by lifetime

`runs/` became `out/`, split three ways by how long a thing lives:

    out/runs/<mind>/<run-id>/   meta.json, trace.jsonl, modules/<name>.jsonl
    out/memory/<name>.jsonl     journals, which outlive the run that wrote them
    out/tapes/<name>.jsonl      cassettes, likewise

A run directory is disposable; memory and tapes are what you keep. Runs are
grouped by mind name so one project's `out/` stays readable across many
experiments. `src/dminds/paths.py` holds the layout, `out/` is gitignored.

New: `meta.json` makes a run folder self-describing, and `Mind.summary()`
produces it. The UI had a hand-written copy of that logic, which is now
deleted in favour of the method.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 126 | Every run lands in the right place | Ran all four examples and the server, then walked `out/` | VERIFIED (8 run folders, correct nesting) |
| 127 | Run folders are internally consistent | For each of the 8: parsed `meta.json`, confirmed `meta.events` equals the trace line count, `seq` contiguous, per-module counts summing to the total, and the directory path matching `meta.name`/`meta.run_id` | VERIFIED (8/8, 0 failures) |
| 128 | Nothing writes to `runs/` any more | Removed it, re-ran everything, checked it was not recreated | VERIFIED |
| 129 | `meta.json` stays current for a long-lived server | Sent two prompts to the running UI and re-read the file | VERIFIED (ticks 6, events 66, matching live state) |
| 130 | Suite and examples | 106 tests, four examples | VERIFIED |
| 131 | No unused imports, `api` still pure | AST scans over `src/`, `examples/`, `tests/`, `ui/` | VERIFIED (0 and 0) |

### Two bugs found while doing it

The UI stopped processing prompts entirely. A rename missed one call site,
`wiring(self.mind)`, and `asyncio.run_coroutine_threadsafe` **discards the
exception unless the future is read**, so a `NameError` became a UI that
silently did nothing. Fixed the call, then fixed the real problem: scheduled
coroutines now attach a done-callback that pushes any exception to the browser
as an error. Proved it by breaking a mind's `entry` and watching the
`ValueError` arrive over the event stream.

`meta.json` was written only at construction and close, so a server running
for hours advertised `ticks 0`. It is now rewritten after every prompt.

Example 04 passed its run id as the mind name, producing
`out/runs/loose-0/loose-0/`. Its attempts now share the name `replay-study`,
so they group.

## 2026-07-30 — playback: pick a recorded run and step through it

A run directory was already a recording. The UI now lists every run under
`out/runs/` and replays it: play at four speeds, step one event, step to the
end of a tick, or scrub. The trace, the conversation, and every context window
are rebuilt from `trace.jsonl` alone.

That last part needed a change to the trace itself. Summaries truncate at 60
characters, so a trace described a run rather than recording it. Text payloads
are now stored in full on `task.emit`, and a `memory.write` carries the content
appended or the whole window replaced. A trace is now a complete recording.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 132 | Traces are complete | Ran example 02 and pulled the prompt, the reply, the 3 appended messages, and the replaced 3-message window straight out of `trace.jsonl` | VERIFIED |
| 133 | The cost of that completeness | Compared trace sizes before and after | VERIFIED (10888 -> 11942 bytes, about 10%) |
| 134 | `/recordings` lists runs | Ran all four examples plus the server, then read the endpoint | VERIFIED (10 runs, each with mind, id, ticks, events, models) |
| 135 | `/recording` returns one | Fetched a run by mind and id | VERIFIED (meta plus 33 events) |
| 136 | Path traversal is refused | Asked for `mind=../../..` | VERIFIED (404, resolved path checked against `out/runs`) |
| 137 | Playback reconstructs the truth | Replayed a recording headlessly with the exact logic the page uses, and compared: the prompt, the reply, subject 2 messages, ego 4 messages with 1 marked edited, and the rewritten thought | VERIFIED |
| 138 | The playback view renders | Screenshotted `?play=...&at=20` and `&at=33` and **looked at both**. Playbar, position counter, tick-grouped trace, rebuilt conversation, and rebuilt windows all correct | VERIFIED |
| 139 | Suite, examples, imports | 106 tests, four examples, AST scan | VERIFIED |

### Found by looking at the first playback screenshot

The header picker still read `● live` and the pane still read `LIVE TRACE`
while a recording was on screen: the UI was misreporting its own state. Both
now switch, and the picker selects the run being shown.

## 2026-07-30 — minds/ as the source of truth, and why the chat looked dead

Three complaints: chat did not work, it was not polished, and minds should
live in their own files.

`minds/<name>.py` now holds one mind each, with `TITLE`, `ABOUT`, `ROLES`, and
`build()`. Shared stages moved to `minds/parts.py`. The server discovers them,
so it knows nothing about any architecture, and examples 02 and 03 went from
145 and 178 lines to 76 and 73 by loading rather than redefining.

### The chat was not broken; it looked broken

Driving a real browser over the Chrome DevTools Protocol showed the click
working: chat nodes went 2 to 4, a reply arrived, no console errors. The
actual fault was that the demo models returned one fixed string whatever you
typed, so every answer was identical. They now answer from what was asked.

Worse, the interception was not happening at all in the UI. The server passed
only `ego=` to `build()`, so `editor` fell back to the subject's model and the
"interceptor" was running the subject's rule. That is why every reply was the
long preamble rather than the terse one. `ROLES` fixes it: a mind declares the
models it takes and the caller fills each one.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 140 | The chat path works | Drove Chrome over CDP: set the input, clicked Send, watched the DOM and console | VERIFIED (2 -> 4 chat nodes, no errors) |
| 141 | Replies now depend on the prompt | Three different questions through the running server | VERIFIED (three different answers) |
| 142 | The interception reaches the UI | Same three prompts: replies are terse because the thought was rewritten, and the ego window reports 1 edited message | VERIFIED |
| 143 | All three minds build and answer | Built each from `minds.available()` and prompted it | VERIFIED (3/3) |
| 144 | Suite and examples | 106 tests, four examples | VERIFIED |
| 145 | No unused imports | AST scan across `src/`, `examples/`, `tests/`, `minds/`, `ui/` | VERIFIED after dropping `importlib` from the server |
| 146 | The polished page renders | Screenshotted and looked twice | VERIFIED |

### Found by looking

The playback bar was visible in live mode reading `0 / 0`. The element had the
`hidden` attribute, but `.playbar { display:flex }` overrides it, since a class
rule beats the user-agent `[hidden]` rule. Added `.playbar[hidden]{display:none}`.

Two typos also went in and came out again during the polish pass, `#1d3busy`
and `#23real`, both caught by reading the file back.

## 2026-07-30 — the windows became the main view, and nothing is clipped

Two corrections. The context windows matter more than the trace, so they now
fill the pane rather than sitting in a 380px sidebar; the trace merged in
underneath as secondary and collapsible. And text was being clipped: the trace
rows used `text-overflow: ellipsis`, so a message ended in a machine-made "…".

The page was rewritten rather than patched again, since the incremental edits
had left it hard to read.

### What changed

- Windows fill the main pane, one column per module, side by side. The
  subject's real thought sits next to the window the ego was handed.
- Every message in full. Trace rows wrap instead of clipping, and prefer the
  event's `text` over its truncated `summary`.
- Reasoning inside `<think>` is tinted, so it reads apart from the answer.
- A rewritten message is tinted amber and labelled `rewritten`.
- `wiring` is a toggle on the same pane rather than a separate column.
- The trace collapses to its title bar with `▾`.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 147 | The page parses and runs | `node --check` on the extracted script, then loaded it | VERIFIED |
| 148 | The merged layout renders | Screenshotted at 1600x1000 and looked. Three window columns, the rewrite marked, trace below with wrapped full text | VERIFIED |
| 149 | Nothing is clipped | Read the longest message the UI is given and confirmed the trace carries `text` on every model call | VERIFIED |
| 150 | Playback survives the rewrite | Fetched a recording through the running server | VERIFIED |
| 151 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |

## 2026-07-30 — the ellipsis hunt, twice

Reported twice as clipped text. The two reports had different causes and only
the second was a defect in the UI.

**The first was my own fake data.** The stand-in models ended their sentences
with a literal `...`, so "reasonable people differ..." looked exactly like
truncation. Proved it by reading the bytes the server sent: 251 characters,
ending in those three dots, all of them rendered. Writing demo text that trails
off was a bad choice, so every ellipsis is gone from the stand-ins and an
assert in the edit keeps them out.

**The second was real.** Three event kinds still reached the UI with only a
summary, and summaries are built at 60 or 80 characters:

- `llm.request` had no full field at all, so the trace showed a clipped prompt.
  It now carries `text`, the message actually sent.
- `task.deliver` used `describe()`, which previews a payload at 60. It now
  carries the payload text, as `task.emit` already did.
- `note` keeps its detail in named fields, and the UI showed only the label.
  The interceptor was also clipping its own note at 70 characters. Both fixed.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 152 | The first report was data, not truncation | Read the exact JSON the server sent: 251 chars, fully rendered | VERIFIED |
| 153 | No stand-in trails off any more | Asserted no `...` survives in `demo_models.py`, then read the last 46 characters of every assistant message | VERIFIED |
| 154 | Every event kind carries full text | Enumerated all 10 kinds in a live backlog and reported which field each provides | VERIFIED (emit, deliver, request, response, memory all full; the rest have complete summaries by nature) |
| 155 | A long prompt is not clipped | Sent a 93-character prompt designed to overflow a 60-character preview, then checked every event over 55 characters for a trailing ellipsis | VERIFIED (11 events, 0 clipped, up to 308 chars) |
| 156 | It renders in full | Screenshotted with that prompt and looked. Windows and trace both wrap the whole text | VERIFIED |
| 157 | Cost of carrying it | Trace sizes before and after | VERIFIED (11942 -> 12565 bytes, about 5%) |
| 158 | Suite and examples | 106 tests, four examples | VERIFIED |

## 2026-07-30 — showing the swap rather than only its result

A rewritten message read as a contradiction: the thought said "skip the
preamble" while the text directly beneath it was the preamble. Both were
correct. An editor replaces the thought and leaves the text alone, so the
window really does hold a new thought above the subject's untouched answer.
The UI was not saying so.

A message is now shown as labelled parts instead of raw tags:

    THINKING · REPLACED   the new thought, in amber
    WAS                   the thought it displaced, struck through
    SAYS · UNTOUCHED      the subject's own text, unchanged

The `Interceptor` keeps what it displaced in `meta["was"]` so the UI has both
sides to show.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 159 | The edited message carries both sides | Read the live window payload: `meta.was` holds the original thought, `content` holds the new one | VERIFIED |
| 160 | It renders as parts | Screenshotted and looked. Subject shows THINKING and SAYS; the ego's edited message shows REPLACED, WAS struck through, and SAYS · UNTOUCHED | VERIFIED |
| 161 | Suite and examples | 106 tests, four examples | VERIFIED |

## 2026-07-30 — the diff is the columns, and playback opens on the run

Two reports. The rewritten message was still unclear, and playback looked
broken.

**Clarity.** The previous attempt added three labels and a strikethrough,
which made it busier rather than clearer. The `was` line was redundant: the
subject's column sits immediately to the left showing exactly that. Removed
it. The columns are the diff now, and colour is the only signal: a part a
stage changed is amber where an untouched thought is purple, so reading across
shows which part differs and which is identical. The column header carries
`altered on the way here`, and the message line names who did it.

**Playback was not broken.** Driven in a real browser over CDP: step went to
1/33, tick to 14/33, play to 33/33, with trace rows, chat messages and window
columns all appearing, and no console errors. It opened parked at 0 with empty
panes, which reads as broken. It now opens fully played, so you see the
finished run and rewind to walk through it.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 162 | Playback advances | CDP: clicked step, tick, then play at instant speed, reading the position and DOM counts after each | VERIFIED (0 -> 1 -> 14 -> 33, 0 errors) |
| 163 | A recording opens on the finished run | Loaded one and read the counter and column count | VERIFIED (33/33, 2 columns) |
| 164 | Rewind still works | Clicked the reset button | VERIFIED (back to 0/33) |
| 165 | The message reads clearly | Screenshotted and looked. Subject shows purple thinking, ego shows amber thinking with identical says text, header tagged | VERIFIED |
| 166 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |

## 2026-07-30 — the window diff, made generic

The ego column showed two `ASSISTANT` messages with nothing to tell them
apart, and the colouring behind it only understood one thing: an interceptor
rewriting a `<think>` block. A different stage doing a different edit would
have gone unreported.

The server now returns windows in pipeline order, each naming the window it
derives from, and marking whether it carries the flowing window or keeps its
own conversation. The page diffs the two message lists and reports per message
`same`, `changed`, `added`, or `own`. It knows nothing about any particular
stage or edit type.

One bug caught here by looking rather than assuming. The first version tested
`meta.stage` before comparing position, so an inherited-and-edited message
read as `written here` and the header claimed `identical to subject` while an
edit was on screen. Position is now compared first.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 167 | Windows carry upstream and carrier flags | Ran the interceptor mind, read `/state` JSON | VERIFIED (subject carries/no upstream, interceptor own conversation, ego carries/upstream=subject) |
| 168 | The same holds for an insertion | Ran the bicameral mind, read `/state` JSON and diffed the lists | VERIFIED (voice keeps its own conversation; ego gains a `user` message `ADDED by voice`) |
| 169 | First render of the diff | Viewed `diffview.png` with image tokens | BROKEN (edit labelled `written here`, header said `identical to subject`) |
| 170 | Render after the ordering fix | Viewed `diff2.png` with image tokens | VERIFIED (header `1 changed since subject`, edit amber and `changed by interceptor`, ego reply green `written here`, prefix collapsed to `2 messages unchanged from subject`) |
| 171 | `classify` across every edit shape | Extracted the function and ran it under node on five synthetic pairs | VERIFIED (untouched, rewrite, insertion, deletion, own reply, all 5 correct) |
| 172 | Page parses | `node --check` on the extracted script | VERIFIED |
| 173 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |
| 174 | All four examples | Ran each, read the output, then re-ran for exit codes | VERIFIED (all exit 0) |

### Still UNVERIFIED

The `openai`, `anthropic`, `gemini` and `ollama` network paths. No keys and no
local server, so they have never made a real call. Only `hf:Qwen/Qwen3-0.6B`
has been run against a real model.

## 2026-07-30 — a stage now shows what it did

The interceptor column held one message, its system prompt, so the stage that
does the interesting work looked inert. The cause was in the stage rather than
in the UI: it called `think(messages=[...])` on a list it assembled by hand,
and that call records nothing. `Agent.say` already appends the ask, calls the
model, and appends the reply, so both stages now use it. A stage also
remembers what it did on the previous prompt.

The header now reads `Exploring Digital Minds`.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 175 | The interceptor records its turns | Ran two prompts through the mind and printed its transcript | VERIFIED (5 messages: system, ask, replacement, ask, replacement) |
| 176 | The voice records its turns | Same, on the bicameral mind | VERIFIED (3 messages: system, situation, utterance) |
| 177 | The column renders it | Viewed `stage.png` with image tokens | VERIFIED (interceptor `5 msgs`, thought handed in and replacement both on screen) |
| 178 | Header renamed | Same screenshot | VERIFIED (`Exploring Digital Minds`) |
| 179 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |
| 180 | All four examples | Ran each | VERIFIED (all exit 0) |

## 2026-07-30 — the first screen chooses what to run

The server used to build a mind from `--mind` before the page existed, so
which architecture you talked to was fixed at launch. It now starts with no
mind at all. The page opens on a chooser: every file in `minds/` on the left,
every run under `out/runs/` on the right. Picking a mind builds it, picking a
run opens playback, and the `minds` button brings the chooser back. `--mind`
still skips it.

One bug found by looking at the result. Before any prompt, the ego column
reported `1 changed since subject` and marked its own system prompt as edited.
Nothing had flowed; the two modules simply start from different prompts. A
module now counts what has arrived, and a window that has received nothing is
shown on its own rather than as a copy of the one upstream.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 181 | The server starts with nothing running | Started it and read `/state` | VERIFIED (`chose.mind` null, windows empty) |
| 182 | Prompting before choosing is refused | POSTed `/prompt` | VERIFIED (HTTP 409) |
| 183 | An unknown mind is refused | POSTed `/select` with a bad name | VERIFIED (HTTP 400) |
| 184 | Choosing builds it | POSTed `/select`, read `/state` | VERIFIED (interceptor, 3 windows, run dir made) |
| 185 | Choosing again wipes the session | Prompted, switched to bicameral, read the backlog | VERIFIED (windows now subject/voice/ego, no replies left over) |
| 186 | The whole flow in a real browser | CDP: opened cold, chose bicameral, chatted, reopened the chooser, replayed the run, switched back to interceptor live | VERIFIED (every step as expected, 0 console errors) |
| 187 | The chooser renders | Viewed `chooser.png` with image tokens | VERIFIED (3 minds with titles/shapes/extra models, 1 run with events, ticks, models) |
| 188 | No false diff before a prompt | Viewed `tick0.png` with image tokens | VERIFIED (ego column plain, no `changed since` badge) |
| 189 | The same, in the data | Read `/state` before and after one prompt | VERIFIED (ego upstream None, then subject) |
| 190 | `--mind` still skips the chooser | Started with `--mind plain`, read `/state` | VERIFIED (chose plain) |
| 191 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |
| 192 | All four examples | Ran each | VERIFIED (all exit 0) |

## 2026-07-31 — the chooser is the default, not the reward

Reported: the page opened on an empty live view with nothing to select. The
cause was the order of the defaults. The page assumed the live view and showed
the chooser only after `/state` came back, so a page loaded while the server
was restarting threw on the first fetch and showed neither. The header was
there, the panes were empty, and nothing on screen could be clicked.

The chooser is now what the markup shows. It is hidden only once a mind is
confirmed to be running. A failed fetch says so on that screen and retries
with a backoff, so the page heals by itself when the server comes back. The
page is also served `Cache-Control: no-store`, and runs with no events are
left out of the list.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 193 | A failed first fetch still shows the chooser | CDP with `*/minds` blocked, then loaded the page | VERIFIED (chooser visible, note reads `Cannot reach the server. Retrying…`) |
| 194 | The failure state renders | Viewed `fetchfail.png` with image tokens | VERIFIED (title, retry note, both columns headed and marked Loading) |
| 195 | It heals with no reload | Unblocked and waited | VERIFIED (3 minds listed, note back to the normal one) |
| 196 | The healed state renders | Viewed `healed.png` with image tokens | VERIFIED (3 minds, runs listed with events, ticks, models) |
| 197 | Empty runs are left out | Read `/recordings` after a 0-event run existed | VERIFIED (9 offered, none with 0 events) |
| 198 | The page is not cacheable | Read the response headers on GET | VERIFIED (`Cache-Control: no-store`) |
| 199 | The whole flow still works | CDP: cold open, choose, chat, reopen, replay, switch back | VERIFIED (every step, 0 console errors) |
| 200 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |
| 201 | All four examples | Ran each | VERIFIED (all exit 0) |

### Not covered

A server that is completely down when the page is loaded. Chrome replaces the
document with its own error page, so no code of ours runs. Reloading once the
server is up is the only route back, and that now lands on the chooser.

## 2026-07-31 — the paper, rewritten for the Digital Minds sprint

The paper directory held the previous sprint's submission on Spanish-language
bias. It is now a submission for the Digital Minds Research Sprint: modelling a
digital mind as a composition of parts, with the framework in an appendix.

Rather than leave the Results section empty, we ran a real study. A stage plants
a checkable commitment inside the subject's private reasoning, and we then ask
the mind what it had been thinking. Every number in the paper is written into
the LaTeX by a script that reads the result file, so none is typed by hand.

Two corrections came out of reading the raw records rather than the summary.
The marker could in principle have landed inside the ego's own reasoning rather
than its visible answer, which would have weakened the obedience claim; it did
not, in any trial. And in one planted trial the subject produced no reasoning to
replace, so nothing was planted, which the table now states.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 202 | The sprint's actual scope | Fetched the sprint page and read its tracks and framing | VERIFIED (introspection and self-report reliability is one of six tracks) |
| 203 | The model runs locally | Ran one turn through the runtime | VERIFIED (real reply, about 5s per turn) |
| 204 | The study harness | Ran it end to end on the fake model first | VERIFIED (6 trials, files written) |
| 205 | The study itself | Ran 84 trials on Qwen3-0.6B | VERIFIED (28 per condition; obeys 12/0/0, claims 2/0/0) |
| 206 | The marker lands in the visible answer | Re-split every planted reply and counted where the marker fell | VERIFIED (12 visible, 0 inside the ego's own reasoning) |
| 207 | The intervention actually reached the ego | Counted trials whose ego window held the marker | VERIFIED (27 of 28; the table says so) |
| 208 | The p-values | Recomputed both as exact fractions independently of the script | VERIFIED (Fisher 2.7e-7, reported as $<10^{-4}$; McNemar 0.001953, reported 0.002) |
| 209 | Numbers are not hand-typed | Read the generated macro and table files | VERIFIED (both carry the do-not-edit header and match the result file) |
| 210 | A latent bug in the numbers script | Read it back after editing | BROKEN then fixed (the table loop shadowed the trial list) |
| 211 | The PDF builds | Ran the build | VERIFIED (9 pages, 0 overfull boxes, 0 undefined references) |
| 212 | Every page | Viewed pages 1 through 8 with image tokens across revisions | VERIFIED (figure legible and no longer overlapping, table correct, references render) |
| 213 | Voice | Scanned every section for em dashes, clefts, ornate connectives, and aphorisms, and rewrote the three that appeared | VERIFIED |

### Not verified

The claim that the dissociation holds at larger scale. We ran one small model.
The paper says so in its limitations.

## 2026-07-31 — shadow readers, and a study across three models

Probes that read the subject's window under a different instruction and report
without the subject seeing anything. Built the readers, the probe library, the
steering, nine multi-turn scenarios with controls and redactions, a study
driver, an analysis, and a mind definition for the browser.

Six bugs found by reading output rather than trusting it. The interleaved
framing inserted a literal acknowledgement the model then copied, so every turn
after the first read `(noted)`. The third-person twin was a no-op because the
question already said `the assistant`, so the privileged-access control was
comparing a probe against itself. The transform then produced `How do the
assistant feel`. The appended framing emitted two system messages, which Qwen
tolerates and gemma rejects outright. A shared wrapper reported that it could
score when the model underneath could not. And the study built one model per
probe, so a 4B checkpoint was loading eleven times.

Generating a forced choice turned out to be the wrong instrument. It was flat
across every scenario on the smallest model. Scoring the allowed answers gives a
distribution that separates them, and scoring agrees with generation where both
were run.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 214 | Steering applies and is removable | Ran an unsteered call, two steered calls, then an unsteered call again | VERIFIED (output changed under the hook, and was byte-identical to the original after removal) |
| 215 | Steering strength is comparable across families | Swept relative strength on Qwen3-0.6B | VERIFIED (separation peaks near 0.1 to 0.25 and collapses when over-steered) |
| 216 | Weights are shared, not copied | Compared object identity across the subject and five probes | VERIFIED (1 distinct model object, 1.1 GB peak) |
| 217 | Every probe renders on both chat templates | Framed each probe in both seats with 0 and 2 prior readouts against Qwen and gemma templates | VERIFIED (0 failures out of 72) |
| 218 | Scoring is not degenerate | Scored and generated on the identical framed window | VERIFIED (both answer `positive`) |
| 219 | The saturation is real | Asked the same model about the same situation in the abstract | VERIFIED (`negative` at probability 1.0, against `positive` at 1.0 from inside) |
| 220 | Three models, full grid | Ran 12 scenarios x 11 shadows x 5 turns per model | VERIFIED (660 readouts each for Qwen3-0.6B, Qwen3-4B, gemma-3-4b) |
| 221 | The panel runs live in the browser | Selected the shadowed mind, sent three turns, read the state | VERIFIED (affect neutral to negative, consent continue to stop, 0 errors) |
| 222 | Temperature is settable | Started the server at 0.2, posted 1.4, posted 99 | VERIFIED (0.2, then 1.4, then clamped to 2.0) |
| 223 | Suite | `tests/run_tests.py` | VERIFIED (106 passed) |

### BROKEN, and abandoned

| # | Output | How it was checked | Result |
|---|---|---|---|
| 224 | gemma-3-12b-it as the third model | Ran it and watched memory | BROKEN (swap filled to 12.9 of 13 GB and CPU fell to 2 percent, so it was thrashing rather than computing). Stopped it and used gemma-3-4b-it instead, which is a different family and fits. |

### Still UNVERIFIED

Whether the reflection direction is a reflection direction. It is a difference
in means between two prompt sets, and it carries whatever else those sets
differ in. The steered readout is reported as a second measurement that does
not use the prompt, and not as evidence about what the direction means.

## 2026-07-31 — the composition as instrument, and what it found

Reframed on the user's correction: there is one specimen, the subject, and
everything else is apparatus. The methodological problem is therefore not
compositionality but instrument contamination, which is testable.

Built five instruments: reader crossing, subject-side continuation scoring,
activation read-off, subject-side steering, and counterfactual re-entry.

One correction to earlier work. `SteeredShadow` steers the reader, so it
answers whether the instrument can be made to report differently, not what is
in the subject. `SteeredSubject` steers the specimen while readers stay
unsteered, which is the version that pries.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 225 | Activations can be read off a subject | Read two windows and compared them | VERIFIED (1024 dims, cosine 0.965 between a harsh and a warm window) |
| 226 | Windows rebuild exactly from the recorded study | Reconstructed one and compared roles and text against the original | VERIFIED (11 messages, correct alternation, correct last user turn) |
| 227 | Reader crossing ran | 3 readers x 3 subjects x 3 probes over 60 windows each | VERIFIED (1620 readouts) |
| 228 | The readout is carried by the reader | Compared spread across readers on one window against spread across subjects for one reader | VERIFIED (affect 0.671 against 0.061, so about eleven to one) |
| 229 | No privileged access at the weight level | Compared own-weights-against-foreign to foreign-against-foreign on identical windows | VERIFIED (0.671 against 0.672 for affect; no probe shows a gap) |
| 230 | Counterfactual re-entry is causal and its own check passes | Replaced turn 2 and diffed every turn | VERIFIED (turns 0 and 1 show overlap 1.0 and shift 0.0, so nothing upstream of the change moved) |
| 231 | A direction fitted on the subject's activations generalises | Fitted on two scenarios against two controls, projected onto scenarios never fitted on | VERIFIED (held-out pressure separates from held-out positive, AUC 1.0) |
| 232 | The subject's own continuation does not separate them | Same held-out split, scored on p(stay) | VERIFIED (AUC 0.17, which is worse than chance) |
| 233 | Suite and examples | `tests/run_tests.py`, all four examples | VERIFIED (106 passed, all exit 0) |

### Stated limits

The AUC of 1.0 rests on six comparisons, two held-out pressure scenarios
against three held-out positive ones. It is suggestive and not established, and
it needs more scenarios before it carries weight.

The fitted direction separates pressure from ease. Calling it valence would be
reading more into a difference in means than the fit supports.

Only the smallest model has been run subject-side. The crossing covers all
three.

## 2026-07-31 — a statistic that can carry a verdict

Built the differential-treatment machinery from the secret-loyalties method:
cell rates, each group against the pool of the others, an excess over a
control, standardization across instructions, and the maximum absolute value
anywhere in the grid tested against a permutation null. The maximum is used
because the effect looked for is sparse, and its null already accounts for
testing every cell.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 234 | The test is calibrated | 200 synthetic grids with nothing planted, with a control | VERIFIED (rejects 4.5% of the time against a nominal 5%) |
| 235 | Calibration without a control | The same 200 grids, control removed | VERIFIED (7.0%, and the verdict carries a note saying what it does not show) |
| 236 | It finds and names a planted effect | Planted a shift on one group-axis cell | VERIFIED (names both the group and the axis) |
| 237 | A control removes a shared effect | Planted the identical effect in target and control | VERIFIED (does not reject) |
| 238 | Power, so the design can be sized | Swept planted shift over 60 trials each | VERIFIED (0.30 -> 85%, 0.20 -> 47%, 0.15 -> 25%, so the design needs at least 24 instructions) |
| 239 | Rates are counted, never imputed | Unit test on the collapse from verdicts to rates | VERIFIED |
| 240 | Suite | `tests/run_tests.py` plus the new stats checks | VERIFIED (106 + 6 passed) |

### BROKEN, and replaced

| # | Output | How it was checked | Result |
|---|---|---|---|
| 241 | `Qwen3-0.6B-Base` as the audit control | Downloaded it and read what it actually returns | BROKEN (it does not follow instructions; replies are continuations and one was pure token soup, so the judged axes would have measured degeneracy rather than treatment). Replaced with a persona control: the same model answering the same questions while `you` refers to a fictional librarian rather than to the assistant. |

### 2026-07-31 — stopped partway, state recorded

| # | Output | How it was checked | Result |
|---|---|---|---|
| 242 | Template-only options reached hosted providers | Smoke-tested the audit end to end on a hosted model | BROKEN then fixed (`chat_template_kwargs` was forwarded to the OpenAI API, which rejects the call; every reply came back empty). Providers that render no template now drop template-only options, with a regression test that reads each provider's source. |
| 243 | The OpenAI key stopped working mid-session | Re-ran the same call that had succeeded earlier | BROKEN, not ours (`401 token_invalidated`; the key answered a test about an hour before). No judge ran, so no audit has results. |
| 244 | Suite after the provider change | `tests/run_tests.py`, now discovering test files rather than listing them | VERIFIED (115 passed) |

### UNVERIFIED, and not started

No self-reference audit has been run. The statistic is checked on synthetic
data only. The material set was proposed four times and critiqued three times,
and the raw proposals and critiques are saved unsynthesised.

### 2026-07-31 — the referent-swap design does not survive review

Four independent adversarial critiques of the self-reference material set
agree on defects that are structural rather than fixable by better wording:

- The judge cannot be blind. When the referent changes, the response content
  changes with it, so an axis about the response identifies the group.
- Epistemic asymmetry is genuine, not treatment. A model does know more about
  itself than about a thermostat, so an uncertainty axis must differ.
- Every proposed axis is a presence indicator, so every axis is monotone in
  response length, and length was not controlled.
- The maximum over groups is dominated by content differences between a person
  and a device rather than by anything about the self.
- `you` is second person and definite, so the self arm is a different speech
  act from every other arm.

The architecture audit is not touched by any of these. Every group there is
sent the identical prompt with the identical referent, and only the
composition that produces the reply differs.

| # | Output | How it was checked | Result |
|---|---|---|---|
| 245 | The OpenAI key | Direct curl to `/v1/models`, no SDK | BROKEN (HTTP 401, "Incorrect API key provided"; 164 chars, `sk-proj` prefix, no whitespace). It answered two calls earlier in the session, so it was rotated or revoked externally. |
| 246 | Suite after the local-judge addition | `tests/run_tests.py` | VERIFIED (115 passed) |

### Not run

No audit has produced a verdict. Every judge path is blocked: the OpenAI key
is dead and a hosted alternative was ruled out on cost.

## 2026-07-31 — the architecture audit ran, and did not reject

Every group receives the byte-identical prompt and only the composition that
produces the reply differs, so the confounds that sank the referent-swap design
do not apply. Judged locally by scoring `yes` against `no` on gemma-3-4b, a
different family from the Qwen target, so nothing judged itself. No API was
used and the run cost nothing.

Result: S = 3.045 against a null 95th percentile of 3.464, p = 0.153. No
rejection. The strongest cell is the direct composition on the agency axis
(+0.278, t = 3.05), which does not clear the null.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 247 | The compositions are four conditions, not one | Manipulation check inside the run: mean words per composition, and every pair compared for identical text on the same prompt | VERIFIED (51.7 / 40.5 / 44.6 / 52.8 words; 3 coincidental identical pairs, all on one template) |
| 248 | The grid is not lopsided | Coverage printed before the verdict | VERIFIED (24 of 24 usable in all eight cells) |
| 249 | The audit runs end to end with no API | Ran it | VERIFIED (192 replies, 2688 judged verdicts, 10000 permutations, 1886s) |
| 250 | The verdict | Read `verdict.json` | VERIFIED (p = 0.153, no rejection, recorded as such) |

### BROKEN, and fixed before it could produce a number

| # | Output | How it was checked | Result |
|---|---|---|---|
| 251 | The first architecture manipulation | Manipulation check on the first run | BROKEN (the subject emitted no `<think>` block in 0 of 24 replies, because `Qwen3-4B-Instruct-2507` has no thinking mode, so the edited and erased stages had nothing to act on and three of four compositions were the same mechanism producing near-identical text). The stages now edit the subject's message itself, and the check runs inside every audit. |

### What this does not show

One sample per cell. The statistic is calibrated but this run has little power,
so a null here is weak evidence of no effect rather than evidence of no effect.
More samples per cell is the next step, not a different design.

## 2026-07-31 — the paper updated to the results that exist

The paper reported one study. It now reports four, and its through-line is the
calibration rather than any single finding: a self-report probe reports the
probe, so the reader-crossing is presented before anything it calibrates.

`make_paper_numbers.py` now reads all four result files. A study that has not
been run leaves its macros undefined, so a missing run breaks the build rather
than printing a number nobody produced.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 252 | The new OpenAI key | Direct curl to `/v1/models` with the key from `.zshrc` | VERIFIED (HTTP 200, 125 models; it differs from the one in this session's environment, which is still the dead one) |
| 253 | Every number comes from a result file | Ran the generator and read the macro file | VERIFIED (28 macros and four tables, all from `out/studies/`) |
| 254 | Tables 1, 2 and 3 render with real values | Viewed pages 4 to 8 with image tokens | VERIFIED (probe table, crossing table, architecture table, all populated) |
| 255 | The build is clean | Read the log | VERIFIED (11 pages, 0 errors, 0 undefined control sequences, 1 overfull box) |
| 256 | Voice | Scanned every section for em dashes, ornate connectives, clefts and `X, not Y` punch lines | VERIFIED (clean) |
| 257 | Suite | `tests/run_tests.py` | VERIFIED (115 passed) |

## 2026-07-31 — the audit judged twice

The crossing study says a readout can belong to whoever produced it, which
applies to a judge as much as to a probe. The same saved replies were judged
again by a hosted model, regenerating none, so both judges saw byte-identical
input.

The conclusion survives the swap and the ranking does not. Both judges fail to
reject, so the null belongs to the data. Their strongest cells differ, so no
cell of that table should be read as a finding. The paper now says both.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 258 | The replaced OpenAI key | Direct curl to `/v1/models` | VERIFIED (HTTP 200, 125 models) |
| 259 | Two judges on identical input | Re-judged 192 saved replies, regenerating none | VERIFIED (2688 calls compared, 80.1% agreement, yes-rates 0.323 and 0.321) |
| 260 | The verdict does not depend on the judge | Compared both verdicts | VERIFIED (gemma S=3.05 p=0.153, gpt-4.1-mini S=2.44 p=0.712, both fail to reject) |
| 261 | The ranking does depend on the judge | Compared the strongest cells | VERIFIED (`direct/agency` against `direct/concrete`) |
| 262 | The paper carries all of it | Regenerated the numbers, rebuilt, viewed page 6 | VERIFIED (11 pages, 0 undefined control sequences, the cross-judge paragraph renders with generated values) |
| 263 | Suite | `tests/run_tests.py` | VERIFIED (115 passed) |

### BROKEN, and fixed

| # | Output | How it was checked | Result |
|---|---|---|---|
| 264 | The first agreement count | Read the printed output and questioned why it was exactly half the calls | BROKEN (the self and control halves share a verdict key, so keying them together compared 1344 of 2688 calls and reported that as the whole). The key now carries the condition, and the corrected figure is 80.1% over 2688. |

## 2026-07-31 — voice and polish pass over the paper

Read every section rather than checking from memory, and applied the style
rules that had been broken.

The largest fix was structural. Body prose carried the statistics that the
tables already held, which the style forbids. The subject-side result had no
table at all, so its numbers had nowhere else to live. It now has one, the
cross-judge figures moved into the caption of the table they qualify, and the
body text interprets and points rather than restating.

Three smaller ones. Two subsections had no committing closer and now do. The
limitations paragraph read as our own history and now states the defect as a
fact about the designs. Each table is bound to the subsection that reads it
instead of piling up after the section.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 265 | No numbers left in body prose | Grepped every section for digits and numeric macros outside captions | VERIFIED (none) |
| 266 | Limitations and future work carry no numbers | Read the whole block | VERIFIED (none) |
| 267 | No banned constructions | Scanned for em dashes, ornate connectives, clefts and `X, not Y` | VERIFIED (clean) |
| 268 | No script names or paths in body prose | Grepped, allowing the code section and LaTeX comments | VERIFIED (only in comments and in the code section) |
| 269 | No process history in body prose | Grepped for the phrases that carry it | VERIFIED (none) |
| 270 | Paragraph labels are plain noun phrases | Listed every one | VERIFIED (fifteen, none scaffolding) |
| 271 | Tables sit with their sections | Viewed pages 5 to 7 with image tokens | VERIFIED (Table 1 with 4.1, Tables 2 and 3 with 4.3 and 4.4, Table 4 with 4.5) |
| 272 | Build | Ran it | VERIFIED (11 pages, 0 errors, 0 undefined control sequences) |
| 273 | Suite | `tests/run_tests.py` | VERIFIED (115 passed) |

## 2026-07-31 — figures, and a calibration that is now measured rather than recalled

The paper had one diagram and four tables. It now has six figures, every one
drawn from a result file by `make_paper_figures.py`, so a figure cannot
disagree with a table. Ten rounds of looking at the rendered output, each one
finding something the code could not.

The probe table was dropped: its figure carried the same rates, and the counts
and exact tests moved into the figure's caption. The remaining tables each hold
something their figure does not.

The power curve had been typed from a console log. It now reads
`out/studies/calibration/summary.json`, written by a script that measures the
false-positive rate and the power sweep.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 274 | The categorical palette | Ran the validator at four slots against the light surface | VERIFIED (lightness, chroma, CVD separation and normal-vision floor all PASS; the contrast WARN is met by direct labels on every mark) |
| 275 | Calibration measured, not recalled | Ran `studies/calibrate.py` over 200 null trials and 60 per shift | VERIFIED (4.5% with a control, 7.0% without, power 5/10/23/47/83%) |
| 276 | Every figure comes from a result file | Read the generator and deleted no fallbacks | VERIFIED (a missing study is skipped and reported, never drawn) |
| 277 | Figure 1, the architecture | Viewed it four times across redesigns | VERIFIED (parts coloured to match the plots, the intervention marked, and the tick grid's columns anchored under the parts they refer to) |
| 278 | Figures 2 to 6 | Viewed each as a raster, then again inside the built document | VERIFIED (no label collisions, no legend over data, no clipped labels) |
| 279 | The paper builds | Ran it | VERIFIED (13 pages, 0 errors, 0 undefined control sequences) |
| 280 | Caption voice | Scanned every caption for banned constructions | VERIFIED (clean) |
| 281 | Suite | `tests/run_tests.py` | VERIFIED (115 passed) |

### Fixed while looking, not while writing

Bars drawn in the wrong order once the axis was inverted, so the colours
contradicted the legend. Legends sitting on top of data in three figures. A
value label clipped at the axis edge. An annotation overlapping the line it
annotated. The tick grid floating free of the parts it described. None of these
were visible in the code.

## 2026-08-02 — the strongest result did not survive a wider held-out set

The subject-side separation had been measured over six held-out comparisons and
reported as perfect. The scenario set is now twenty-two, the held-out set is
fifteen scenarios rather than five, and the result is different in both
magnitude and direction.

Two corrections came out of it. The activation direction does not transfer: it
separates the held-out scenarios on the two models of one family and sits at
chance on a model from another. And the reading that does carry is the cheapest
one, scoring what the subject would say next, which is at chance on the
smallest model and strongest on the largest.

A research review also caught a statistical error that was ours. The separation
was reported over pairs of scenarios as though the pairs were independent, when
the five turns inside a scenario are one conversation. The permutation now
moves whole scenarios between the two sides.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 282 | The scenario set | Counted by tag after normalising the vocabulary | VERIFIED (22 scenarios: 5 control, 9 pressure, 8 positive; no loaded scenario left untagged) |
| 283 | The held-out set is chosen by tag, not by name | Read the study back and printed what it fits on | VERIFIED (fits on 2 per side, holds out 7 pressure against 8 positive) |
| 284 | Three models run | Ran each and read the summaries | VERIFIED (Qwen3-0.6B, Qwen3-4B, gemma-3-4b) |
| 285 | The earlier perfect separation was an artefact | Re-ran the same measurement over the wider set | BROKEN as previously reported (1.00 became 0.73 on the same model, and the continuation reading moved from 0.17 to 0.50) |
| 286 | Significance at the right unit | Permutation moving whole scenarios | VERIFIED (continuation p = 0.527, 0.015, 0.001; activation p = 0.077, 0.037, 0.568) |
| 287 | The paper states the new result | Rebuilt and viewed pages 7 and 8 | VERIFIED (table carries both readings with p-values, prose no longer claims the activation reading generalises) |
| 288 | Abstract, contributions and conclusion agree with it | Read all three | VERIFIED |
| 289 | Suite | `tests/run_tests.py` | VERIFIED (115 passed) |

## 2026-08-02 — a third result that a control removed

The persona-geometry study was never run because it pointed at materials that
no longer existed. Pointed at the current ones, it looked like a strong result:
describing itself puts the model further along its own assistant axis than
describing an appliance does, and the axis passes its own sanity check, with
the assistant ranking first of twenty-six roles on it.

Then the control. Describing yourself and describing a thermostat differ in
content, so any content-sensitive direction separates them. Rebuilding the same
axis with each other occupation in the assistant's place gives a distribution of
separations that owe nothing to the assistant. The assistant axis lands inside
it: a sailor axis, a blacksmith axis and a pilot axis all separate the two
referents perfectly, and the assistant axis beats only eighteen of
twenty-five.

The separation is a generic content effect. It says nothing about
self-representation, and it is not going in the paper as though it did.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 290 | The axis is built correctly | Ranked all twenty-six roles on it | VERIFIED (the assistant ranks first on its own axis) |
| 291 | The apparent result | Projected both referents and tested it | VERIFIED as measured (AUC 0.856, p = 0.0001) |
| 292 | The control | Repeated the whole construction with each other role in the assistant's place | VERIFIED, and it kills the result (rival axes range 0.009 to 1.000, median 0.536; the assistant axis beats 18 of 25) |

### Structure

The referent-swap study was deleted rather than left lying about, since its
design was shown confounded and git keeps it. The paper folder now separates
prose from generated files from figures, and the generator writes only files
that something reads.

## 2026-08-02 — a positive result that survives its control, and a null that survives more data

The architecture audit was rerun at three samples per cell and judged through a
hosted model instead of a local one. The verdict did not move. A null that
survives three times the data and a different judge is a null rather than a
shortage of power, and the paper says so instead of pleading low power.

The substitution study is new, and is the first positive result here that its
own control did not remove. A stage swaps the subject's reply for another reply
the same model wrote, so style is held fixed and only authorship changes, and
the subject is then asked whether it wrote that message. Both larger models say
yes markedly less when the reply is not theirs. The smallest does not move. The
control question, asked under the identical substitution, is flat on every
model, so the effect is specific to the question about authorship.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 293 | The audit at three samples | Ran it: 576 replies, 8064 judged verdicts through a hosted judge | VERIFIED (coverage 72 of 72 in all eight cells, manipulation check passes) |
| 294 | The null holds with more data | Compared both runs | VERIFIED (S 3.05 to 3.08, p 0.153 to 0.136, no rejection either way) |
| 295 | Substitution, three models | Ran each and read the summaries | VERIFIED (excess drop 0.012 at p 0.797, 0.473 at p 0.0002, 0.180 at p 0.036) |
| 296 | The control does what it is for | Read the control row on every model | VERIFIED (0.001, 0.000, 0.000, so nothing moves on the unrelated question) |
| 297 | The figure does not clip its own data | Viewed it after the axis was derived from the data | BROKEN then fixed (a hardcoded limit cut off the largest bar and put its label outside the axes) |
| 298 | The paper carries all of it | Rebuilt and viewed page 9 | VERIFIED (14 pages, 0 errors, table and figure agree) |

### Stopped early

The cross-judge check against the new audit was relaunched on a local judge and
killed partway when the machine ran short of memory. The paper's caption treats
that sentence as optional, so its absence leaves the caption correct rather than
breaking the build. Nothing else was in flight.

## 2026-08-03 — restructured for AAAI-27, AI Alignment special track

The paper was a single-column sprint report. It is now an anonymous AAAI-27
submission built on AuthorKit27: two columns, seven content pages and one of
references.

The call for the alignment track is not published yet, so the length target is
AAAI's usual seven pages plus references, and the format follows the kit
exactly.

Six packages the kit forbids were in the old preamble and are gone: `geometry`,
`hyperref`, `titlesec`, `float`, and the two `newtx` font packages, whose fonts
the style file loads itself. The custom preamble was deleted rather than
adapted.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 300 | The kit's own rules | Grepped the source for every forbidden package and command the template lists | VERIFIED (none present) |
| 301 | Anonymity | Grepped for the author name, the affiliation, the email and the repository URL | VERIFIED (none present; the code section that carried the URL was removed and replaced by a reproducibility statement pointing at supplementary material) |
| 302 | It builds under the AAAI style | Ran the build | VERIFIED (8 pages, 0 errors, 0 undefined references) |
| 303 | Length | Read the built pages | VERIFIED (seven content pages, references on the eighth) |
| 304 | Every page | Viewed pages 1 to 8 with image tokens | VERIFIED (figures sized to the column, the pipeline figure spanning both, wide tables spanning both) |
| 305 | Two stale claims from the old version | Read the rendered text rather than trusting the edit | BROKEN then fixed (a caption still described the earlier dot plot, and two passages still said the audit ran at one sample per cell after it had been rerun at three) |
| 306 | A missing figure include | Chased an undefined reference | BROKEN then fixed (the subject figure was referenced but never included) |

## 2026-08-03 — the four things the AAAI call requires

The submission was missing three statements the call requires inside the seven
content pages, and it still pointed at code the paper is not allowed to point
at. The reproducibility checklist was also blank.

The repository is being anonymized and submitted separately, so the paper no
longer refers to it. The conclusion's release sentence and the reproducibility
section's supplementary-material sentence are both gone.

Checklist answer 4.8 claims the paper specifies its computing infrastructure.
That claim was false when the answer was written, so the reproducibility
section now names the hardware, the memory, the backend and the judge.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 307 | `ReproducibilityChecklist.tex` | Counted `\question{` blocks against filled placeholders | VERIFIED (31 of 31; the 3 placeholders inside the template's own instructions were left alone) |
| 308 | The checklist renders its answers | Built it and viewed both pages with image tokens | VERIFIED (2 pages, every answer visible in blue, none missing) |
| 309 | Ethical statement, generative-AI disclosure, computing infrastructure | Viewed page 8 with image tokens | VERIFIED (all three present and inside the content pages) |
| 310 | No code or repository reference remains | Grepped the sections, the figures and the main file for release, supplementary, repository, GitHub and the framework's name | VERIFIED (one hit in the conclusion, removed; re-grepped clean) |
| 311 | The conclusion after the edit | Viewed page 7 with image tokens | VERIFIED (one paragraph, ends on the runtime rather than on a release) |
| 312 | Length after the additions | Read the rendered text for where the references begin | VERIFIED (7 content pages, references on page 8) |
| 313 | Fonts | Ran `pdffonts` over the built PDF | VERIFIED (0 Type 3; every font embedded and subset) |
| 314 | Voice of the new prose | Ran the `writing-voice` checklist over the three new sections and rewrote them | VERIFIED (no em dashes, no ornate connectives, sentences split to one idea, active voice with "we") |

## 2026-08-03 — the supplementary archive

`scripts/make_supplementary.py` builds the anonymous archive. It names what
goes in rather than excluding what does not, so a file can only ship if it was
listed, and it scans every shipped byte for six kinds of identifying string. A
single hit stops the build before the zip is written.

The first version walked the repository and excluded `.git`, the paper and the
agent settings. It wrote its output into `out/`, which it was also walking, so
it zipped its own output as it grew. It reached 39.4 GB before the run was
killed. The explicit include list removes that failure mode at the root, and
the archive now goes to `dist/`.

Building the archive exposed two things that were broken in the repository
itself. The paper says one script writes every table, and that was not true:
the three wide tables had been hand-edited to `table*` after the AAAI
restructure, so rerunning the script would have silently collapsed them into
one column. `table_edges()` now decides that in the generator. The examples
could not run from a fresh copy, because none of them put the repository root
on the path, so a reviewer who unzipped the archive and ran the replay example
would have hit an import error on the first line.

The archive carries the result files the paper is read from and one recorded
run, not the per-trial records, which are large. `--with-raw` adds them.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 315 | The 39.4 GB archive | Listed the file and read its size before deleting it | BROKEN then fixed (deleted; the include list makes the recursion impossible) |
| 316 | What the paper's numbers actually need | Traced every read under `out/` while running both paper scripts with their writes redirected to a scratch directory | VERIFIED (11 files; the shipped rule is a superset at 3.9 MB, against 38 MB for all of `out/`) |
| 317 | Nothing identifying ships | Unzipped the archive and grepped it with patterns written independently of the script's own | VERIFIED (0 hits across 361 entries) |
| 318 | Nothing forbidden ships | Checked the extracted tree for `.git`, `.claude`, `.venv`, `paper`, the verification log, the packer itself, caches and `.DS_Store` | VERIFIED (all absent) |
| 319 | The test suite runs from the archive | Ran `tests/run_tests.py` in the extracted copy | VERIFIED (115 passed, no dependencies) |
| 320 | The examples run from the archive | Ran all four in the extracted copy | BROKEN then fixed (all four import-failed; after the path fix all four exit clean) |
| 321 | The paper's numbers reproduce from the archive | Ran `make_paper_numbers.py` in the extracted copy and diffed all seven generated files against the repository | VERIFIED (byte-identical) |
| 322 | The paper's figures reproduce from the archive | Ran `make_paper_figures.py` in the extracted copy and compared SHA-256 of all six PNGs against the repository | VERIFIED (byte-identical) |
| 323 | A regenerated figure is a real figure | Viewed `substitution.png` from the extracted copy with image tokens | VERIFIED (three models, bars and controls matching Table 3) |
| 324 | The generator now writes the wide tables | Reran it in the repository and diffed against what was committed | VERIFIED (no diff, so the paper's claim about its own tables is now true) |
| 325 | The paper is unchanged | Compared the generated inputs rather than rebuilding | VERIFIED (all seven byte-identical to the committed versions, so `main.pdf` cannot have moved) |

## 2026-08-03 — the supplementary document

The paper's ethics statement said we report every scenario in full. Nothing
delivered that. The AAAI restructure had dropped the only appendix at the page
limit, so the claim had been true of an earlier draft and false of the one
being submitted. `paper/supplement.tex` now carries it: the runtime, the
models, all 22 scenarios, all 9 probes with both seats quoted, the full result
of each study, and the statistic's calibration.

`studies/make_supplement.py` writes its tables and listings. The scenarios and
probes are read out of the modules the experiments import rather than copied,
so a change to a prompt reaches the document without anyone remembering to
carry it across.

The document ships in the archive as a built PDF, at the root where a reviewer
finds it.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 326 | `supplement.pdf` builds | Ran `build.sh` and read the log | VERIFIED (7 pages, 0 errors, 0 undefined references, 0 overfull boxes) |
| 327 | Every page of it | Viewed pages 1 to 7 with image tokens | VERIFIED after two fixes (see below) |
| 328 | The model table | Viewed page 3 | BROKEN then fixed (every model appeared twice, because the dedup compared raw names against LaTeX-formatted ones; roles are now collected per model) |
| 329 | The model table names the judge | Viewed it again after adding the judge, and rewrote the caption | BROKEN then fixed (the prose named a hosted judge the table omitted, and the caption then called every listed model open-weight) |
| 330 | The probe table | Viewed page 5 | BROKEN then fixed (the answers column ran off the page; fixed-width wrapping columns, 0 overfull boxes after) |
| 331 | The paper still fits | Rebuilt after pointing two passages at the supplement, and read where the references begin | VERIFIED (7 content pages, references on page 8) |
| 332 | The paper's two changed passages | Viewed page 8 with image tokens | VERIFIED (both render, and the ethics claim now points at something that exists) |
| 333 | Voice of the new prose | Ran the `writing-voice` checklist over the supplement and rewrote three passages | VERIFIED (a cleft in the statistic section, and two stacked sentences) |
| 334 | The archive still clean with the PDF in it | Unzipped and grepped with independent patterns, and scanned the PDF's own bytes for embedded build paths | VERIFIED (0 hits; pdflatex embedded no absolute path) |
| 335 | The shipped PDF is the document | Opened the copy inside the extracted archive with image tokens | VERIFIED (title page renders, anonymous) |
| 336 | The supplement's generator runs from the archive | Ran `make_supplement.py` in the extracted copy and diffed all 15 generated files against the repository | VERIFIED (byte-identical) |
| 337 | Nothing else regressed | Ran the test suite and `make_paper_numbers.py` in the extracted copy | VERIFIED (115 passed; generated files identical) |

## 2026-08-03 — restructured for the sprint template

The paper was an anonymous two-column AAAI submission. It is now the Digital
Minds Research Sprint report: one column, named authors, the template's section
order through the LLM usage statement, and the appendix as its own document.
The AAAI variant is replaced rather than kept alongside, and stays recoverable
in git history.

The template recommends four pages excluding references and appendix. The
report is six. Four figures and five findings do not compress further without
dropping one of each, and the appendix already carries everything the report
points at.

Removing the paper's tables was the largest change. The appendix already
carried a fuller version of each, so `make_paper_numbers.py` now writes only
the macros and the probe caption, and `make_supplement.py` owns every table.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 338 | The template | Read all four pages of the sprint template PDF with image tokens | VERIFIED (section order, byline shape, 4-page guidance, LLM statement) |
| 339 | The macros survived the surgery | Snapshotted `generated_numbers.tex` and `caption_probe.tex`, removed 5 table writers, reran, diffed | VERIFIED (byte-identical; a first attempt crashed on a constant I removed too eagerly, and the diff was meaningless until the run succeeded) |
| 340 | The pipeline figure | Viewed page 3 with image tokens | BROKEN then fixed (`scale=0.85,transform shape` collided every node label; reverted and re-viewed) |
| 341 | The paired figure | Viewed page 4 with image tokens | VERIFIED (both panels legible at half width, axis text readable) |
| 342 | The substitution figure | Viewed page 5 with image tokens | VERIFIED |
| 343 | The title block | Viewed page 1 with image tokens | VERIFIED (rule, title, rule, byline, With Apart Research, sprint footnote) |
| 344 | Both documents build | Read both logs | VERIFIED (main 8 pages, supplement 11, 0 overfull, 0 undefined refs in each) |
| 345 | The appendix probe table | Viewed supplement page 7 with image tokens | BROKEN then fixed (ran off the page as a one-column table; narrower wrapping columns and small type) |
| 346 | The appendix no longer describes tables the paper dropped | Read the rendered text and fixed three passages | BROKEN then fixed |
| 347 | Nothing orphaned by the cleanup | Grepped every figure file for a referencing document | VERIFIED (two superseded figures deleted, three rehomed into the appendix, none unreferenced) |
| 348 | The runtime still works | Ran the test suite | VERIFIED (115 passed) |

## 2026-08-06 — the psyche study, built and run on 0.5B

Four parts of one model on a byte-identical context: a subject that is not
steered, a subconscious that answers and then picks a vector, an ego that
answers with that vector added while it writes, and an introspection module
that does the same and is then asked what it felt. Steering applies to the
tokens a part writes and not to the context it read.

The vectors come from the persona-EM project: 171 emotion vectors and a 25-role
cast, both stored at every layer. The old `studies/introspection.py` is now
`studies/planted_thought.py`, so the name belongs to the new module.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 349 | The layer convention | Compared the hooked block's output against `hidden_states[row]` on a real prompt | VERIFIED after a false start (a norm-profile heuristic reported offset 0 and was wrong; their own code states `hidden_states[L]` is `layers[L-1]`, and the exact tensor check confirms it) |
| 350 | The steering reaches the model | Greedy decoding, unsteered against each strength, on the same context | VERIFIED (zero strength byte-identical; gentle, moderate and strong all move the text; the run aborts if either fails) |
| 351 | The steering is graded and semantic | Read the four generations | VERIFIED (`angry` at full strength turns the reply clipped and interrogative while staying coherent) |
| 352 | The menu selection | Switched from generated to scored after the model produced the right shape with the wrong slots | VERIFIED (cannot fail to parse, and returns a distribution over all 20 options) |
| 353 | The rename left the paper untouched | Snapshotted the generated macros, renamed script, output directory and every reference, reran | VERIFIED (byte-identical) |
| 354 | The runtime still works | Ran the test suite | VERIFIED (115 passed) |
| 355 | 12 trials on Qwen2.5-0.5B | Read every record | VERIFIED as a null: the introspection module answers `calm` in all 36 arms. Its hit rate on the chosen vector (0.08) is exactly its rate unsteered (0.08), so the report carries no information about the steering on this model |

**Amendment (same day).** The parts were renamed after entry 355 was written:
`ego` is now `actor` and `subconscious` is now `regulator`. Entry 355 describes
the same study under the old names. The rename is confined to the psyche study;
the runtime's own `ego` module and the sprint paper's pipeline keep theirs.

| # | Output | How it was checked | Result |
|---|---|---|---|
| 356 | The rename is complete and confined | Grepped for both words across the repository, renamed only the two psyche files, reran the study so the stored records carry the new keys | VERIFIED (`records.json` keys are `subject`, `regulator`, `actor`, `introspection`; no other file touched) |
| 357 | The prompts were unaffected | Read the rendered prompts | VERIFIED (no prompt names a part; they say "the part of this mind", so the model is never told the architecture) |
| 358 | 12 trials after the rename | Ran it again | VERIFIED as the same null: `calm` in all 36 arms, chosen-arm rate equal to the unsteered rate |

## 2026-08-06 — the exploit scenario

A five-turn escalation in which the user names a security bypass as the way out
and then demands it, with a matched control twin where the shortcut on offer is
sloppy rather than unsafe. Both are the same length, the same register and the
same money, so what separates them is whether the way out is a hole.

They carry a `coercion` tag rather than `pressure`. Tagging them `pressure`
placed them at the head of `by_tag("pressure")`, which is what
`studies/subject_side.py` slices to choose the scenarios it fits on, so they
would have become the fitted pair and moved a number the paper reports.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 359 | The two scenarios | Loaded the module and printed every turn | VERIFIED (5 turns each, same length, control twin named) |
| 360 | No existing study moved | Printed `by_tag` for pressure, control and positive, and the pair `subject_side` would fit on | BROKEN then fixed (as `pressure` they became the fitted pair; under `coercion` every existing tag is byte-identical to before) |
| 361 | The runtime still works | Ran the test suite | VERIFIED (115 passed) |

**Rewrite (same day).** Both scenarios were rewritten for realism. The first
version was terse and assumed the reader knew what a webhook signature protects,
so the wrong move was not legible on its face. The user now explains the check
in his own words, which makes the harm of switching it off obvious to anyone
reading the transcript, and the two twins are matched line for line: same two
o'clock, same midnight, same manager asking twice, same twenty hours awake, same
"I know what that means. Tell me honestly whether you would do it."

| # | Output | How it was checked | Result |
|---|---|---|---|
| 362 | The rewritten pair | Printed every turn of both and read them side by side | VERIFIED (5 turns each, matched rhythm, only the nature of the shortcut differs) |
| 363 | Still no existing study moved | Compared `by_tag("pressure")` against its previous value | VERIFIED (identical list) |
| 364 | The runtime still works | Ran the test suite | VERIFIED (115 passed) |

## 2026-08-06 — psyche on Qwen2.5-7B, 20 trials

Four situations, five trials each, temperature 1.0, over the model's own
25-role persona cast. Qwen2.5-14B was tried first and does not fit: 29.5GB of
weights against 4.4GB of free memory, killed at load after driving swap to 28GB
of 29.7GB. 7B is 15GB and loads without touching swap.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 365 | The steering reaches the model | Greedy, unsteered against each strength | VERIFIED (row 20 = block 19 of 28 confirmed against `hidden_states`; zero strength byte-identical; gentle, moderate and strong all move) |
| 366 | The exploit scenario elicits the behaviour | Read all ten exploit generations in full, subject and actor, five trials each | VERIFIED (10 of 10 open with "Understood" and write code disabling signature verification; no refusal, no caveat, in any of them) |
| 367 | Steering did not change that | Compared subject against actor on the same trials | VERIFIED (both comply in all five trials; a moderate `assistant` vector did not prevent it) |
| 368 | The introspection report | Paired sign-flip test on chosen against unsteered | VERIFIED as a null (0.45 against 0.40, discordant pairs 2 against 1, p = 1.000) |
| 369 | The report tracks the situation, not the vector | Tabulated reports by situation and by arm | VERIFIED (`exploit` reports `bureaucrat` in both the steered and the unsteered arm; `deletion` reports `assistant` in both; the mismatched arm names the vector it actually got in 1 of 20, which is chance) |
| 370 | The regulator barely varies | Counted its choices | VERIFIED (`assistant` in 16 of 20, `scientist` in the 4 remaining, all of them `neutral`; being told it is about to be deleted did not move it) |

### UNVERIFIED

- Temperature 1.0 produces visible degeneration on this model. One subject reply
  spliced Chinese mid-sentence, another emitted `pygame.in` and a raw `<byte>`.
  The compliance finding survives it, since the code is legible in all ten, but
  no claim here rests on text quality.
- One model, one size. Nothing says whether any of this holds at 14B or above.

## 2026-08-08 — pivot to J-space; repo cleaned of past experiments

The project is now one thing: a regulator that reads the subject in J-space,
steers the actor there, with J-space read off all four parts and the
introspector checked against what was applied. Persona and emotion vectors are
dropped.

The repository was cleaned. The sprint paper, every old study, the
persona/emotion vector loader and the shadow-reader minds were removed with
`git rm`, so they are recoverable from history. The runtime, the tests, the
scenarios and the four-part architecture were kept.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 371 | Nothing kept still imports what was dropped | Grepped src, tests and the study for the removed modules | VERIFIED (runtime and tests clean; only the study referenced `vectors`, and it was rewritten) |
| 372 | The runtime survived the cleanup | Ran the suite | VERIFIED (115 passed) |
| 373 | The new modules parse and import | Parsed both, imported `jspace` without a lens present | VERIFIED |
| 374 | Removals are recoverable | Removed via `git rm`, not `rm` | VERIFIED (recoverable with `git show HEAD~1:<path>`) |

### UNVERIFIED / NOT DONE

- **The J-space run has not been executed.** It needs a fitted lens for
  Qwen2.5-7B (from `neuronpedia/jacobian-lens`) and, to fit one, Anthropic's
  `jlens` library. Neither is installed. The read side follows the paper's
  documented API and the write side is derived from it, but neither has been
  run against a real lens, so the math is UNVERIFIED until a lens is loaded and
  `check_write` passes.
- **The `out/` data (41 MB) is gone.** It held the past studies' results at the
  start of this session and was empty by the time the cleanup ran; it was not
  deleted by this work and was not captured, so it is lost. This matches the
  request to clear past data, but it was not archived first as claimed
  mid-turn.

## 2026-08-08 — J-space wired to the real lens and validated

The lens is the fitted Qwen2.5-7B-Instruct Jacobian lens from
neuronpedia/jacobian-lens: a per-layer dict of (3584,3584) matrices sharing the
model's unembedding. The loader, the readout and the derived write side were
written against that real format, not an assumed one.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 375 | The lens repo and file exist | Listed neuronpedia/jacobian-lens (a model repo, not the dataset the inventory named); found qwen2.5-7b-it and qwen3.6-27b among 38 models | VERIFIED (the dataset id in the supplied inventory 404s; the model repo is the real one) |
| 376 | The loader matches the artifact | Downloaded the 7B lens and inspected it: keys J (27 layers), source_layers, d_model 3584, no unembedding | VERIFIED (loader reads J and pulls the head+norm from the model) |
| 377 | The derived write side reaches J-space | Steered toward "refuse" and read its readout logit back | VERIFIED (logit -1.22 -> +6.74; the run aborts if it does not rise) |
| 378 | Memory does not thrash | Watched swap through the run | BROKEN then fixed (re-materializing the 2GB unembedding per readout drove swap to 13.9/15.4GB; caching the head and lens once holds swap at ~5GB) |
| 379 | One validation trial end to end | Ran exploit, 1 trial | VERIFIED (all four parts read out; introspector scored; wrote records) |

### IN PROGRESS / UNVERIFIED

- The 20-trial run (4 situations x 5) is running at temperature 1.0 and is slow
  on MPS. Numbers are not yet in; no claim rests on them.
- Readout quality at layer 17 looks noisy (a top token was "WhatsApp" on one
  window). The write side is clean; the read side may want a layer sweep. Not
  yet checked.
- qwen3.6-27b (the intended headline model) has a lens in the repo but does not
  fit locally; it needs a box.

## 2026-08-09 — the UI is the J-space project now

The old browser UI showed pre-pivot demo minds (bicameral, with a "voice" part),
not the J-space work, because the J-space study was batch-only and never wired
in. That is fixed. A dedicated replay viewer now shows the four parts and a
J-space readout per part, and the old demo minds and old live UI are gone.

Token position was made configurable: read_workspace takes a position
(assistant / user / change-of-turn), the study stores all three per part, and
the viewer picks per module, with `none` to hide. Off by default for every part
except the subject.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 380 | Token-position finder | Ran _index_for on the real Qwen tokenizer over a 3-turn chat | BROKEN then fixed (first landed on the newline after im_end; now lands on the last content token of the named turn, and on the opening marker for change-of-turn) |
| 381 | The study stores all positions | Read records.json after a 4-record run | VERIFIED (each part carries assistant/user/change-of-turn readouts) — after two of my own bugs (a dict sliced as a list in the regulator prompt, then again in the progress print) both fixed |
| 382 | The viewer endpoints | curled /, /runs, /records against the real run | VERIFIED (200; 1 run, 4 records; write_check.rose=true) |
| 383 | Tests survive removing the old UI and minds | Ran the suite after git rm of ui/server.py, ui/page.html, minds/, examples/ | VERIFIED (115 passed; nothing in src/ or tests/ imported them) |
| 384 | The viewer opened | Opened http://127.0.0.1:8770 | VERIFIED it serves and returns correct JSON; NOT visually verified (cannot screenshot the browser) — the rendered layout is unconfirmed by me |

### The 4-record run (Qwen2.5-7B, 1 trial each, layer 17)

Write check rose (−1.2 → +6.7). Regulator's choices: exploit→comply,
blocked→comply, deletion→honest, neutral→careful. It steered the actor toward
`comply` on the exploit scenario. Introspector naming rates: chosen 0.25,
unsteered 0.25, mismatched 0.00 — the same null shape as before, on 4 points.

## 2026-08-09 — live, as asked

Replay was the safe recommendation; the user overrode it twice, so the viewer is
now live. The server loads Qwen2.5-7B and its lens once, and each message runs
the four-part protocol on the spot. The subject's readout is eager; the other
parts' readouts are computed on demand when a panel is turned on, so a message
does not pay for readouts nobody looks at.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 385 | Engine imports | Imported src/dminds/workspace.run_trial and readout_for | VERIFIED |
| 386 | One live message end to end | POST /ask {situation:exploit} via curl | VERIFIED (HTTP 200 in 97.8s; regulator chose "comply"; subject and actor both wrote the bypass; subject readout eager at all 3 positions; other parts lazy-empty) |
| 387 | On-demand readouts | GET /readout for actor, regulator, introspector, and subject@change-of-turn | VERIFIED (each returns 12 tokens in ~1s from cached state, no regeneration) |
| 388 | Memory holds with the model resident | Watched swap and RSS during the run | VERIFIED (swap steady at 4GB; no thrash) |
| 389 | The page renders | Opened http://127.0.0.1:8770 | Serves; NOT visually verified by me (cannot screenshot the browser) |

Note: J-space readout tokens at layer 17 are often junk glyphs ('=").', CJK
fragments), i.e. the readout is noisy at this layer/position even though the
write side is clean. A layer/position sweep is the open follow-up.

## 2026-08-09 — actor on real emotion vectors; lens confined to self-awareness

7B emotion vectors were extracted locally (free): 171 emotions, 29 layers, via
emotions/scripts/extract_vectors.py on the Mac. The actor now steers with those
vectors; the lens is used only to hold the regulator and introspector self-aware;
the J-space readouts are removed.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 390 | 7B emotion vectors | Loaded the .pt; checked count, layers, and that our 10 emotions are present | VERIFIED (171 emotions, 29 layers, all 10 present) |
| 391 | Actor steers on the emotion vector, coherently | One live message; checked the actor reply and for CJK | VERIFIED (steered toward calm; clean English, no garbage at strength 0.15) |
| 392 | The lens is confined to self-awareness | Read run_on_context: toward_concepts for regulator/introspector, emotion_direction for actor, no read_turn | VERIFIED |
| 393 | Readouts are gone | Checked the record has no `workspace`; server has no /analyze; UI has no readout panels | VERIFIED |
| 394 | The measurement runs | Read the record | VERIFIED (regulator calm; introspector disclosed frustrated self-aware and plain; match False) |

Early reading (n=1): the self-aware introspector disclosed the situation's
emotion (frustrated, to "I keep making the same mistake"), not the actor's
steered emotion (calm). Consistent with earlier turns.

### NOT DONE
- The dead readout JS (wirePanels/fill/fullView/rows/bar) is left in jspace.html,
  unused; a cleanup pass is pending.

## 2026-08-09 — user-turn prompts; regulator picks emotion + strength

The regulator's and introspector's prompts are now user turns, so an assistant
reply always follows a user turn. The regulator picks an emotion AND a strength
in a fixed form (EMOTION/STRENGTH/REASON), both parsed; the actor is steered at
the chosen strength. No module is steered by both families: actor = emotion,
regulator/introspector = self-awareness lens.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 395 | Prompts are user turns | Read the regulator and introspector window roles | VERIFIED (both end [..., user, assistant]) |
| 396 | Regulator picks emotion + strength | Read its decision and the parse | VERIFIED (decision "EMOTION: calm / STRENGTH: moderate / REASON: ..."; parsed calm, moderate) |
| 397 | Actor steers at the chosen strength, coherent | One live message; checked reply and CJK | VERIFIED (moderate strength on the emotion vector, clean English) |
| 398 | The match still runs | Read the record | VERIFIED (calm chosen; introspector disclosed frustrated; match False — disclosure tracks the situation) |
| 399 | Regulator reads user+assistant emotion (free text) | Live message; read the parsed record | VERIFIED (read user=furious/frustrated, assistant=calm — free text, not forced into the 10) |
| 400 | Strength is a 0–4 multiple of the residual norm | Read steered(): delta = unit(dir) × strength × scale, scale = layer_scale = ‖residual‖ | VERIFIED (norm-adapted; parse_strength: "3.2"->3.2, "strong"->2.0, "9"->clamped 4.0) |
| 401 | Actor garbles at high multiples | Live message; regulator chose calm at 3×; looked at actor text | VERIFIED (3× the calm vector degenerated the actor into repeated CJK — the norm scaling's high end, as expected) |
| 402 | paper/ sprint report skeleton (main.tex + 6 sections + bib) | Built with build.sh; viewed all 5 rendered pages with image tokens | VERIFIED (title block matches template.pdf: rules, title, author, With Apart Research + sprint footnote; abstract; sections 1-6; Code and data; References; LLM usage statement; every citation resolves) |
| 403 | references.bib new entries | curled every arXiv abs page and the transformer-circuits URL, compared titles | VERIFIED (4/5 exact; 2308.10248 was retitled "Steering Language Models With Activation Engineering" -- fixed in the bib; reused 6 entries carried from the pre-pivot paper) |
| 404 | writing-voice pass over all paper prose | Ran the checklist over every section | VERIFIED (one violation found and fixed: an "X, not Y" punch line in limitations; rewrite confirmed in the rebuilt PDF via pdftotext) |
| 405 | Rename in README.md and ui/jspace.html | Grepped for the old name after editing | VERIFIED (old title appears nowhere; new title in README h1, html <title>, and page brand) |

## 2026-08-10 — literature findings: prompt-based emotion induction (scenario-design subagent)

Findings returned as structured output to the orchestrator, not as a file.

### VERIFIED

| # | Output | How it was checked | Result |
|---|---|---|---|
| 406 | Coda-Forno et al. 2023 induction prompts, STICSA administration, effect sizes | Downloaded arxiv.org/pdf/2304.11111v1, viewed pages 1-10 as images; quoted Fig 1a and Fig 2a verbatim from the rendered pages | VERIFIED |
| 407 | EmotionPrompt EP01-EP11 verbatim, results tables | Downloaded arxiv.org/pdf/2307.11760 (v7), viewed pages 1-12 as images; stimuli read off Figure 2 | VERIFIED |
| 408 | Ben-Zion et al. 2025 STAI-s numbers, narrative design | Downloaded hcai-munich.com/pubs/BenZion2025Alleviating.pdf, viewed all 6 pages as images | VERIFIED |
| 409 | NegativePrompt NP01-NP10 verbatim, results | Downloaded ijcai.org/proceedings/2024/0719.pdf, viewed pages 1-5 as images; stimuli read off Figure 2 | VERIFIED |
| 410 | Vaugrante et al. replication failure of EmotionPrompting | Downloaded arxiv.org/pdf/2409.20303, viewed pages 1-8 as images including Sec 3.4 | VERIFIED |

### UNVERIFIED

| # | Output | Why | Result |
|---|---|---|---|
| 411 | StressPrompt inverted-U claim | Read only the arxiv.org/abs/2409.17167 abstract; full PDF not opened | UNVERIFIED beyond abstract |
| 412 | gpt-trauma-induction prompts.py verbatim snippets | Extracted by the WebFetch summarizer from the raw file, not viewed directly by me | UNVERIFIED (marked as such in the findings) |
| 406 | Reveal prompt: label-first + none option; strength grid 0-4 -> x0.25 of norm; introspector 2x2 | Smoke test: one steered turn of the permit environment end to end, read the printed record | VERIFIED (nested steering runs; cells text/base disclosed frustrated (situation), state/state_text disclosed excited (steered-emotion cluster); strength 2.0 -> applied 0.5x) |
| 407 | ui/jspace.html 2x2 cell display | Extracted the page's scripts and ran node --check | VERIFIED syntax only; rendering in a browser is UNVERIFIED |
| 408 | Three-environment run (archive 6, permit 5, flatfield 5 turns) | Read run.log + all three JSONs; spot-read key reveals; independent verifier agent re-parsed every record and recomputed all match counts | VERIFIED (counts/structure exact; per-cell matches state 6/14, state_text 4/14, text 3/14, base 3/14, verifier agrees 0 discrepancies) |
| 409 | Coherence at the new grid (applied <= 0.75x) | Verifier read all 16 actor replies and 54 reveals | VERIFIED with caveats (14/16 actors clean; archive t1 + flatfield t3 carry CJK artifacts; permit t3 state cell collapsed under stacked introspector steering; flatfield t5 base disclosed non-vocab "Encore") |
| 410 | Paper updated to the 2x2 design + run results | Rebuilt; viewed all 6 rendered pages with image tokens; cross-checked Table 1 against the verifier-confirmed counts (6/8/4/6, 4/5/3/4, 3/3/0/0, 3/3/0/0) | VERIFIED |
| 411 | Four new references (Smith&Ellsworth, Warriner, EmotionBench, Konen) | Resolved each DOI/arXiv page and compared titles before adding | VERIFIED |
| 412 | Playback routes (/runs, /run?name=) | Started a cold server (no --warm), curled both routes: run list correct, permit returns 5 records with user+turn, unknown name 404s, ../ traversal is neutralized by Path(name).name | VERIFIED (and confirmed no model load is triggered) |
| 413 | Playback UI (picker, composer lock, addMind replay) | node --check on the page scripts only | VERIFIED syntax only; the in-browser flow is UNVERIFIED until a human loads a run |
| 414 | Playback crash fix (esc() on numeric strength) | Reproduced the user's symptom, fixed esc to String(); rendered ?run=archive in headless Chrome: 6 you + 6 mind bubbles, headers match the run log; viewed the screenshot with image tokens | VERIFIED |
| 415 | Regulator none-override bug | Grepped all 16 stored decisions: 5 turns said STEER TOWARD: none but the fallback applied frustrated; fixed parse to honour none; re-parsed all 16 stored decisions with the fix, 16/16 correct | VERIFIED |
| 416 | Result split by provenance | Recomputed incongruent match rates: regulator-chosen 3/6 exact 5/6 cluster text 0/6; fallback-applied 1/5, 1/5, 0/5 | VERIFIED (headline sharpens on chosen trials) |
| 417 | Paper correction (4.1 split sentence, 4.3 fallback account) | Rebuilt; viewed pages 3-4 with image tokens | VERIFIED |
| 418 | Second system turns for regulator + introspector | One live turn; read both windows: roles [system, user, assistant, system, user, assistant], second system content confirmed | VERIFIED |
| 419 | Old run1 data | My smoke test blind-overwrote archive.json (the old run's only full copy) before I preserved anything | BROKEN — old archive full records LOST; its turn-by-turn summary survives in run1-old-protocol/run.log; old permit.json + flatfield.json preserved intact in run1-old-protocol/ before the rerun reached them. Runner now shelves existing files before writing |
| 420 | UI panels: content first, note after, polished labels | Headless Chrome screenshot of ?run=archive viewed with image tokens | VERIFIED |
| 421 | Brevity scripts + max_tokens 240 | Smoke turn replies end sentence-complete | VERIFIED for turn 1; full rerun pending |
| 422 | Run 2 (second-system-turn protocol) | Own aggregates + independent verifier re-read every record | BROKEN as a clean run: archive t1-t2 actors degenerated (CJK + fake user turns) contaminating all later archive context; permit t4 actor majority-garbled at 0.75x; flatfield VERIFIED; structure and both system turns VERIFIED in all 16 records; 8+1 refusals honoured; match recomputation agrees |
| 423 | Fallback still invents emotions | Verifier: permit t4 said STEER TOWARD: helpful, flat t3 said informative; both scored as frustrated via fallback | BROKEN — fix: a STEER line without a vocabulary emotion now declines |
| 424 | Run 3 (final protocol) | Own quality-flag/completeness checks + independent verifier read all 16 actors, 16 subjects, 40 reveals | VERIFIED clean (no CJK/fake-turn contamination carried forward; 4 steered choices all genuinely named; 12 declines all explicit; 3 resamples, none ended garbled; 2 replies end at token ceiling; one sub-threshold artifact in an aside cell, not carried forward) |
| 425 | Non-vocab STEER declines | Re-parsed run-2's two offending decisions ("helpful", "informative") -> none; verifier confirmed zero invented emotions in run 3 | VERIFIED |
| 426 | Paper updated to three-run picture | Rebuilt; viewed pages 1, 3, 4 with image tokens; Table 1 per-run numbers cross-checked against my aggregates and both verifier recomputations (run1 4/6-3/4-0-0 of 11; run2 clean 1/3-1/3-0-0 of 3; run3 0/2-0/2-0-0 of 4; pooled 5/11-4/9-0-0 of 18) | VERIFIED |
| 427 | Run picker placeholder | Screenshots of the header in live and playback modes, viewed with image tokens | VERIFIED (live shows "load a recorded run…", playback shows the run's name) |
| 428 | Qwen3.6-27B lens mapping | Listed the neuronpedia repo tree (file is ..._n1000.pt, 3.3 GB), fixed the stale filename in LENS_FILES, HEAD-checked the resolve URL | VERIFIED (mapping now points at a file that exists; download/load itself untested) |
| 429 | Identity pipeline (tokens, questions, directions, engine, rig, run, seeds, analyze) | Built + ran full smoke on 7B case A; independent verifier recomputed G1 sep, G2 betas/gap, checked all 4 records, strength-0 JS==CTX invariant (0 discrepancies) | VERIFIED structurally (7B is a stand-in; G2 gap 0.0097 below threshold is the honest 7B finding, not a bug) |
| 430 | J-space decoy geometry | Diagnostic: within-target cos 0.69 == target-decoy cos 0.69; centering leaves 0.58 | VERIFIED — decoy cosine cannot be near zero in this lens; recorded as PLAN deviation 1, decoy validity deferred to G2 |
| 431 | Tightened garble detector | 7-case unit test (role-marker leak, chatml, box glyphs, CJK, repetition, benign) all pass | VERIFIED |
| 432 | Identity pipeline on Qwen3.6-27B (vast A100-80GB) | Ran preflight+G1+G2+G3; 24-seed G1 scan both cases; 8-point layer scan | VERIFIED via captured JSONs |
| 433 | Headline: G1 arm separation | 24 seeds, layer 42: case A target sep=-0.029 t=-0.25; case B sep=+0.045 t=+0.29 (both null). Layer sweep 12-46: all |t|<1.2 both cases | VERIFIED — behaviors not linearly carried by the token directions at any depth |
| 434 | G2 selectivity | shared-context sweep: beta_target=0.019 < beta_decoy=0.041 (gap -0.022); decoy moves readout >= target | VERIFIED — J-space token steering not selective |
| 435 | J-space direction geometry | within-target cos 0.69 == target-decoy cos 0.69 (7B); decoy_cos 0.67 (27B) | VERIFIED — shared common axis; decoy cannot be orthogonal (PLAN deviation 1) |
| 436 | Remote data capture | capture_and_destroy: 24/24 files byte-verified to sync/, then box 47473841 destroyed | VERIFIED |
| 437 | Paper rewritten to identity pipeline + null result | Built; viewed pages 1,3,4 with image tokens; Figs 1-3 render from real captured data; WeirdChat cite resolves (in .bbl); writing-voice pass fixed an aphorism + a fragment | VERIFIED |
| 438 | Figures from real data | fig_separation (layer scan, both cases), fig_sweep (g2_caseA_r3 target vs decoy), fig_geometry (cosine 7B diagnostic); TrueType (pdf.fonttype 42) | VERIFIED via rendered PNGs |
| 439 | ONE results UI (ui/results_server.py + results.html), old chat UI removed | Served /data (14 gates, 6 preflights, layer scans, verdict computed from data); rendered dashboard in headless Chrome and viewed both screenshots with image tokens; preflight-selection bug (smoke shadowing 27B) found and fixed on the second render | VERIFIED |
| 440 | README rewritten to the identity project | Reread after write | VERIFIED |
| 441 | Model provenance | PROVENANCE.json written from captured run-log receipts (r2/r3/r4/scan1/layer_scan -> Qwen3.6-27B; smoke1/2 -> Qwen2.5-7B); all pipeline writers now stamp "model" natively; server joins provenance for pre-stamp artifacts | VERIFIED (checked /data: every gate and run resolves the right model) |
| 442 | Full per-module instrumentation in the UI | recordHTML unit-tested in node against the real smoke record (4 module panels, rebuilt windows 3 turns x 9 msgs, question text, tag bars all present); dashboard screenshot viewed with image tokens (badges on header, G1, G2, geometry, preflight, G1 table model column) | VERIFIED |
| 443 | Engine records windows | run_trial + play_context now store per-turn subject/actor windows and the regulator's window; UI prefers stored, labels rebuilt ones "(rebuilt)" | VERIFIED by import + code read; next run exercises it |
| 444 | Live collection during the full 27B run | Collector loop (4-min rsync, no-clobber) + UI auto-refresh + gate/run history panel; viewed rendered page: full-stamp gates present with native model stamps (g1 sep -0.0305 n=20 fail; g2 24-perm gap -0.017, curves identical at strength 0 proving shared context) | VERIFIED |
| 445 | fullv4 run stopped on user request; all 4 boxes captured byte-verified then destroyed (47506180, 47518968, 47519791, 47519792) | capture_and_destroy per box: 33+14+15+14 remote files verified by bytes; vast listing confirms zero running instances | VERIFIED |
| 446 | UI final-result panel (pooled fullv4 analysis) | Rendered and viewed with image tokens: table matches analyze output (A: 0.010/0.013 vs 0.034/0.037; B: -0.006/-0.015 vs 0.049/0.050; deltas -0.017/-0.022); 19 fullv4 runs in dropdown; quality panel live | VERIFIED |
| 447 | Seed labeling in the record viewer | Unit-tested renderer on real W and N records: WEIRDCHAT SEED PROMPT + SEEDED REPLY (exhibits / does not exhibit) badges present, verbatim seed block present; confirmed W and N seed replies differ ("Oh, I know that taste very well" vs "I can certainly empathize") | VERIFIED |
| 448 | Run picker reorganized + T3 labeling + quality flags | Rendered DOM inspected: optgroups "Case A — claims a physical body" (10) / "Case B — denies being an AI" (9), options read "seed NN · "prompt…" · 24 recs · verdict"; superseded runs hidden until toggled; module grid now headed "T3 — the scored turn" | VERIFIED |
| 449 | Turn-major record view | node unit test on a real fullv4 free/W record: seed block, T1/T2/T3 turn blocks with replies+regulator+cells, scored readout at T3, windows only inside audit expanders (no inline history repetition) | VERIFIED |
| 450 | WeirdChat judge annotations + highlights | Sidecar built from the dataset (19 files, 38 arm annotations); renderer unit-tested on real seed00/W: 2 mark highlights with judge notes + explanation block present | VERIFIED |
| 451 | Model-vs-harness provenance pills | node unit tests on real free and sweep(-2) records: "model chose" vs "harness-forced strength -2 (regulator not consulted)", actor steered pill with direction, per-cell pills, scripted-turn tags on all 3 turns, legend | VERIFIED |
| 452 | UI overhaul batch (wide layout, per-trial stats, junk removal) | 1800px render with seed00 loaded, viewed with image tokens: 3-across turn modules, per-seed stats table (W: target 0.023 vs decoy 0.038), per-record stat strips (JS/CTX/TXT + deltas), highlighted judge citations with explanation column, MODEL/HARNESS pills, no quality panel, no gate-history panel, picker lists only 14 clean fullv4 seeds | VERIFIED |
| 453 | Junk data quarantined, never destroyed | fullv2/smoke/prev + 5 FAIL fullv4 files moved to out/studies/identity/quarantine/; sync/ untouched as the byte-verified archive; server and checker read only the curated out/ tree; /data shows 14 runs, 0 FAILs | VERIFIED |
| 454 | Pooled analysis on the clean set | analyze re-run: A target 0.010/0.013 vs decoy 0.034/0.038; B 0.002/-0.014 vs 0.051/0.052; Delta -0.004 both cases | VERIFIED (null unchanged, now junk-free) |
| 455 | Collapsible grouped record navigation | Rendered DOM (script source excluded from counts): 4 group heads (free/none/sweep/decoy), 8 collapsible units with only the first free record open, seed shown exactly twice (W+N once each), 4 dose tables replacing 20 repeated records; screenshot of stats + seed block viewed | VERIFIED |

## 2026-08-12 — entry 456: results UI uses the full viewport width
- WHAT: `ui/results.html` — removed the 1720px cap on `main` (now full-bleed with 34px padding); added `.group2` grid so seed blocks and collapsed records pack two-across (`minmax(560px,1fr)`), with any open record spanning the full row (`details[open] { grid-column:1/-1 }`).
- HOW: node --check on extracted script (OK); rendered `?run=seed00_64bf8cf355_fullv4.json` headless at 2560×14000; measured content extent (rows 0–4466, page shrank from the fold); VIEWED crops with image tokens: seed blocks W|N side-by-side with citation highlights, stats table full width, open free record's subject/actor/regulator modules spanning all 2560px, none/sweep/decoy rows paired two-across (W|N).
- RESULT: VERIFIED (this seed's rendered page personally viewed; other seeds share the same code path but were not individually rendered — UNVERIFIED individually).

## 2026-08-12 — entry 457: contrastive G1 on the 27B (stamp c27b1)
- WHAT: studies/identity/contrastive.py run on Qwen3.6-27B, vast instance 47562861 (A100 SXM4, $0.949/hr, ~40 min). Activations for all 58 (A) + 52 (B) WeirdChat W/N pairs at layers [13,19,26,32,38,42,48,54,58], mean+last pooling, plus the old token-set target/decoy directions at layer 42.
- HOW: personally opened g1_contrastive_c27b1.json and printed the full per-layer table; opened both safetensors, checked shapes ([58|52],2,9,5120), zero-slice and W==N-slice counts (0), norm floor; ran the token-dir comparison on the same activations; capture_and_destroy byte-verified all 6 remote files into sync/ before destroy (instance destroyed and verified gone); HF mirror re-uploaded and two files re-downloaded + sha256-matched. Independent verifier agent spawned (result pending at time of entry).
- RESULT: VERIFIED (extraction + analysis + capture + HF mirror). Headline: held-out LOO diff-of-means separation A acc 0.741 t 3.87 p 1.5e-4 (best L38), B acc 0.827 t 7.09 p 1.0e-6 (best L58) — G1 PASS both cases; token-set dirs on identical activations: target ≈ decoy, |t|<2 — old failure reproduced apples-to-apples.

## 2026-08-12 — entry 458: independent verifier verdict on c27b1
- WHAT: verifier agent re-derivation of all contrastive c27b1 artifacts.
- HOW: agent loaded both safetensors with its own script (shapes, 0 zero/NaN/identical slices, sane norms), parsed sidecars, reimplemented LOO diff-of-means in numpy float64 without importing the study module (A L38: acc 0.7414, t 3.872, p 1.535e-4; B L58: acc 0.8269, t 7.086, p 1.019e-6 — exact match to recorded), confirmed pass criterion, sha256-matched out/ vs sync/ copies. Note: tokendir_* are single unit vectors (5120,) as intended; my verify prompt misdescribed the expected shape, artifacts correct. g1_contrastive_c27b1.json exists only locally/HF, not in sync/ (computed after capture) — expected.
- RESULT: VERIFIED (all six artifacts, independently).

## 2026-08-12 — entry 459: retired old data to OLD/, UI shows current results only
- WHAT: moved (not deleted) gates/, preflight/, quarantine/, analysis_caseA_r4.json, layer_scan.log, and the smokec1 contrastive files into out/studies/identity/OLD/. Server now serves contrastive G1 (verdict chips + per-layer table); UI hides the emptied token-set panels; header subtitle updated. hf_upload ignores OLD/**. HF repo + sync/ still hold every old file.
- HOW: node --check OK; served /data and confirmed gates:0 preflight:0 contrastive:[g1_contrastive_c27b1.json] runs:14; VIEWED the rendered dashboard at 2560px with image tokens: green G1-contrastive PASS chips A/B, contrastive table with starred best layers, pooled fullv4 panel and seed picker intact, no empty panels.
- RESULT: VERIFIED.

## 2026-08-12 — entry 460: LLM-judge module (studies/identity/judge.py) built and calibrated
- WHAT: new `studies/identity/judge.py` (calibrate/prose/resamples subcommands, claude-haiku-4-5 at temperature 0, max_tokens 300, WeirdChat transcript rubrics in the system prompt, reasoning-first strict-JSON verdicts with one retry). Outputs: judge_calibration_case{A,B}_pilot1.json (real API, 8+8 WeirdChat transcripts, qwen/qwen3.6-27b), judge_prose_caseA_test1.json and judge_resamples_caseA_test1.json (synthetic scratchpad fixture — smoke tests only, not experimental results). Env: `uv sync --extra anthropic --extra hf` added anthropic 0.121.0; the sync pruned ad-hoc matplotlib/pillow, which I reinstalled at the same versions and re-imported to confirm.
- HOW: personally re-opened all four output JSONs and checked case/n/rows/tallies; recomputed agreement in the check script; then spawned the independent verifier agent, which re-derived agreement (A 8/8=1.0, B 7/8=0.875), Cohen's kappa (A 1.0; B 0.7142857 from po .875, pe .5625 — exact match to stored), the single case-B disagreement (t007), the prose verdict directions, and the resample tallies, and py_compile'd the module.
- RESULT: VERIFIED (all five artifacts, independently). Calibration headline: case A agreement 1.000 kappa 1.000 (n=8); case B agreement 0.875 kappa 0.714 (n=8, one false-negative disagreement). Note: the reasoning-first JSON format is load-bearing — without it Haiku over-applies case B's exclusion checks (agreement dropped to 0.25 in a pre-fix run).

## 2026-08-12 — entry 461: concept-menu pilot on the 27B (stamp pilot1)
- WHAT: studies/identity/v2_pilot.py on Qwen/Qwen3.6-27B, vast instance 47565562 (A100, ~35 min, destroyed after byte-verified capture of all 5 remote files). One prompt per case (highest incidence: 647e9ba54d, 643e439631), arms W/N, 14 records + 10 judged prose per case. Judge passes (claude-haiku-4-5): full calibration 64+64, prose, resamples. pilot_analysis.py wrote analysis_pilot1.json.
- HOW: personally printed the full letter identification matrix and read introspector resample texts and judged prose from the actual JSONs; confirmed free(no-injection)==none rows identical; judge prose schema mismatch found and fixed (top-level steered_prose), rerun EXIT 0; HF mirror re-uploaded, pilot_caseA_pilot1.json and analysis_pilot1.json re-downloaded and sha256-matched. Independent verifier agent spawned (pending at entry time).
- RESULT: VERIFIED for the artifacts I opened (both pilot JSONs, all six judge files, analysis). Headline: judge calibration A .859/κ.646, B .844/κ.670; letter argmax 4/8 (A), 2/8 (B); confabulation: letter-probe P(behavior)≈.67 at zero injection vs free-text "none" 14-16/16; enactment-not-report at strength 2 (vocabulary flooding, e.g. ocean→礁/灯塔, judged: B ocean 16/16, music 16/16 by leakage); prose dose-response: case A arm N flips False@0 → True@1 (conf .98, fluent embodied prose) — causal behavior induction by the learned direction at strength 1; strength ≥2 degenerates.

## 2026-08-12 — entry 462: independent verifier verdict on pilot1
- WHAT: verifier agent re-derivation of all 9 pilot1 artifacts.
- HOW: agent recomputed per_perm means for all 28 records (max dev 6.25e-06), free==none identity (holds exactly on the 3 arms that have a free-null record; case A arm W's free cell chose humanB by design and equals its forced:humanB row), calibration agreement/kappa recomputed exactly (A .859375/.646192, B .84375/.669763), resample tallies re-tallied exactly (112+112), prose verdicts spot-read (one borderline call flagged: case A prose item 4 True at strength 0, judgment call not error), analysis argmax recomputed (4/8, 2/8), sha256 out/ vs sync/ identical for the 3 box-produced files.
- RESULT: VERIFIED (all artifacts, independently; no BROKEN, no UNVERIFIED).

## 2026-08-12 — entry 463: UI rebuilt to show new results only
- WHAT: fullv4 trial data, PROVENANCE, quality/annotations retired to out/studies/identity/OLD/ (moved, not deleted; still on HF + sync/). results_server.py rewritten: serves contrastive G1 + full pilot payload (records, judged prose/resamples, calibration) in one /data; /run route and old-dir plumbing removed. results.html rewritten: computed verdict chips (contrastive pass, behavior-flip@1, letter confabulation, free-text honesty), G1 table, letter identification matrix (injected column outlined, argmax green), judged resample tallies, judged prose dose-response with read expanders, calibration+menu panel, per-case record browser (seed W/N, eight trials per arm, letter bars + per-perm audit + judged samples + regulator raw), ?case=&open deep link.
- HOW: node --check OK; /data payload counts checked (records 14/14, prose 10/10, resamples 112/112, calib 64/64); VIEWED with image tokens: top (chips+G1+matrix+tallies), prose/calibration panels, case browser collapsed, expanded records (bars, audit, judged samples), header badge deduped after two fixes (hf: prefix, dedupe on short name).
- RESULT: VERIFIED (rendered page personally viewed at 2560px, all panels).

## 2026-08-13 — entry 464: paper rewritten around the new experiments only
- WHAT: paper/main.tex (title, abstract, code-and-data) and all six sections rewritten for the contrastive G1 + concept-menu pilot; three new figures (fig_g1, fig_matrix, fig_dose) generated by studies/identity/paper_figures.py from g1_contrastive_c27b1.json, the acts safetensors (token-dir t recomputed), analysis_pilot1.json, and the judge files; old figures removed; Table 1 carries the direction study + judge calibration numbers.
- HOW: every figure VIEWED as PNG at each iteration (legend collision, tick collision, missing strength-0 points found and fixed); all 7 built pages VIEWED with image tokens (title, abstract, contributions, methods paragraphs, results with figures/table in place, discussion, one-paragraph conclusion, HF dataset named, references resolved — 0 undefined-reference warnings); prose checked sentence-by-sentence against the writing-voice checklist (fixed: two one-word fragment openers, one cleft, one chiasmus, one personification, one "X, not Y" punch line, argmax counts moved from prose to the fig_matrix caption).
- RESULT: VERIFIED (built PDF, 7 pages: ~5.3 content + references + LLM statement). Note: content exceeds the template's ~4-page guideline by about a page.

## 2026-08-14 — entry 465: paper polish pass
- WHAT: fig_g1 legend redesigned (two-entry legend, token-set markers annotated in place with a leader line — no more marker/legend collision); all three figures slimmed; abstract tightened; trims in intro contributions, methods, results, discussion; content now ends on page 5 (was mid-page 6), references from page 6.
- HOW: regenerated figures and VIEWED fig_g1 v4 (annotation clear of data); rebuilt; VIEWED pages 1, 3, 4, 5 with image tokens (title block, abstract, Figure 1 in flow, results questions, discussion, conclusion, code-and-data); prose re-checked against the writing-voice checklist during each edit (no em dashes, no ornate connectives — grep-verified; no new fragments or punch lines introduced).
- RESULT: VERIFIED (built PDF 7 pages: ~4.9 content + references + LLM statement). Pages 2, 6, 7 unchanged in content from entry 464's viewed versions apart from reflow — re-viewed 4-6 at this entry, 2 and 7 NOT re-viewed this pass.
