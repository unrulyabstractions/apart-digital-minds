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
