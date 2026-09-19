# Meta-Harness iteration -> candidate `0003`

Working directory: `/private/tmp/gamingmaster-audit-tier2-20260919` (you have full read/write access here).

## Task
Design a harness for a **real-time** game (tier 2).

Game: `snake` — Snake on a 12x12 grid, one tick every 0.500s, max 120 ticks. +1 per food eaten; episode ends on wall/self collision. The snake advances one cell per tick in its heading regardless of whether you decided. Observation: dict with head, heading, body, food, food_delta, danger{up,down,left,right} (True = that cell kills), and an ASCII grid (head=arrow, body=o, food=*).

Actions:
- up: face up (y-1)
- down: face down (y+1)
- left: face left (x-1)
- right: face right (x+1)
- keep: keep current heading
(no decision in time => 'keep')
Reversing into yourself is ignored (heading unchanged). Walls kill. Grid coordinates: x to the right, y downward, origin top-left.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow: seconds). Deep reasoning, can plan,
  can explain rules. Returns `.text` and `.session_id` (pass it back to keep its reasoning state).
  Budget: 4 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast: ~0.1–0.7 s). Typed questions
  (`metaharness.models.jev.Choice/Score/Noul`), calibrated probabilities. Budget: 240 calls per
  episode. Jev is a zero-shot classifier: it does not reason, so what you put in `state` and how you
  phrase `instructions`/`criteria` is the whole game.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

The game ticks every 0.5s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Evaluation metric: mean score over seeds
(primary), then fraction of stale ticks (lower is better), then System-2 calls (fewer is better).
Clock mode: wall.

How to ask Jev well (from TypeSafe's System-One guidance, docs.typesafe.ai):
* Keep deterministic work in code (legality, geometry, bookkeeping); ask Jev only the judgment code
  cannot make. One narrow, coherent judgment per question; several *independent* questions over the
  same state go in ONE `ctx.decide` call (parallel, same latency) — speculative questions are fine,
  code consumes only the relevant answers.
* `state` should be a named JSON object with only the relevant context (no context rot); reference
  nested fields in questions with backticked paths like `snake.danger.left`.
* Question ids are invisible to the model: the full meaning must be in `instructions` + `criteria`.
  Criteria must separate the options from each other with concrete situations ("moves toward food
  and keeps ≥ 20 free cells" vs "safe but away from food"); include a no-match option when nothing
  may fit. Score levels must describe concrete situations. Use structured (dict) criteria when
  options are easily confused.
* `probabilities` + `confidence` are the interface: gate on them (low confidence → code fallback or
  escalate to Astra), and check freshness before applying an answer to a changed state.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev at every tick, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game.

## Harness contract (must be satisfied exactly)
```
The harness contract — the *only* thing a proposed harness must satisfy.

A harness is a Python module that defines::

    def build(ctx: HarnessContext) -> HarnessProgram

and ``HarnessProgram`` has one method::

    def run_episode(self, game) -> None

The harness owns the whole control loop for one episode. Everything it may
use comes through ``ctx``:

* ``ctx.system2``   — Astra (slow, reasoning). ``ctx.think(prompt, session=...)``
                      wraps it with tracing + budget accounting.
* ``ctx.system1``   — Jev (fast, typed decisions). ``ctx.decide(state, questions)``
                      wraps it with tracing + latency accounting. Tier 1 tasks
                      pass ``system1=None``.
* ``ctx.spawn(fn)`` — run ``fn`` on a background thread (e.g. an Astra call
                      while Jev keeps the game alive). Returns a Future.
* ``ctx.clock``     — the game clock. In *virtual* mode, model latency is
                      charged to the clock deterministically (tests); in
                      *wall* mode it is real time.
* ``ctx.memory``    — a dict reset per episode by default. Explicit shared
                      memory evaluations allow learning between seeds; the
                      evaluation manifest records which protocol was used.
* ``ctx.log(...)``  — free-form events into the trace.
* ``ctx.budget``    — remaining System-2 calls / tokens; exceeding it ends the
                      episode (score kept), so harnesses must be thrifty.

Real-time games (tier 2) expose ``game.tick_s``. The runner keeps ticking the
game on its own clock: in wall mode a thread ticks every ``tick_s``; in
virtual mode each model call's latency advances the game by
``latency / tick_s`` ticks (using the *last action the harness posted*).
The harness posts actions with ``ctx.act(action)``; posting is instant
(virtual mode charges a 1 ms overhead so a loop that decides without any
model call still lets time pass). ``ctx.wait_for_tick()`` blocks until the
next tick boundary — use it instead of spinning.
Turn games (tier 1) are stepped with ``ctx.act(action)`` synchronously and
return the next observation.
```
This is a COLD-START experiment: every episode starts without a cached policy. At tier 2 Astra must compile/adapt while the world keeps ticking. Jev must have a meaningful measurable role. Use public observations and rules only, and log when each model's output actually changes control. Design a generic discovery/compilation procedure, not a hand-written game solver. Prefer a compact bridge that Astra can produce quickly. Join all outstanding futures even if the game ends. Do not inspect files outside this experiment archive or run your own model calls. Do not import game implementations. Do not inspect or test held-out games.

## Archive
# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | status |
|---|-----------|-----------:|---------:|------------:|---------:|---------:|--------|
| 0000 | 0000-tier2_jev_only | 0.000 | 3 | 17% | 18 | 0 | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 3 | 100% | 0 | 3 | ok |
| 0002 | 0002-previous_astra_winner | 8.667 | 3 | 0% | 180 | 6 | ok |
| 0003 | 0003 | – | – | – | – | – | not evaluated |

Best so far: `0002-previous_astra_winner` (mean score 8.667)

Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency and every game tick/step):
- candidates/0000-tier2_jev_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0001-tier2_astra_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0002-previous_astra_winner/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/0003/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/0003/proposal.md`: hypothesis, what changed vs. the parent
   candidate, what evidence would confirm/refute it.
4. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Do not modify any other candidate. Do not run long experiments yourself; the evaluator will.
Read only this archive and the interface documentation above. Do not inspect game implementation
source, other experiment archives, credentials, or held-out games/results. Infer semantics from the
public game description and observations supplied through ctx. Do not call model backends directly:
all inference must go through ctx.think/ctx.decide so the evaluator can account for it.
ModelUnavailable must propagate. Join all ctx.spawn futures before returning; no background work
may escape the episode. The default evaluation resets both memory and module globals per episode;
only an explicit shared-memory protocol allows cross-episode caching. The manifest is authoritative.