# Meta-Harness iteration -> candidate `0002`

Working directory: `/private/tmp/gamingmaster-audit-tier1-20260919` (you have full read/write access here).

## Task
Design a harness for a **turn-based, hidden-rule** game (tier 1, ARC-AGI-3 style).

Game: `rulepuzzle` — A 10x10 colour-grid puzzle with 2 levels and a budget of 50 actions per level. You control the white cell. Six actions ACTION1..ACTION6 exist; their effects are NOT given and must be discovered by experiment (they are fixed within an episode). Reaching the yellow cell completes a level. Some colours may be walls, keys, doors, toggles or teleports. Score = levels completed.

Actions:
- ACTION1: unknown — discover by trying
- ACTION2: unknown — discover by trying
- ACTION3: unknown — discover by trying
- ACTION4: unknown — discover by trying
- ACTION5: unknown — discover by trying
- ACTION6: unknown — discover by trying
ACTION5/ACTION6 may be context-dependent. Effects are fixed for the whole episode.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 12
calls per episode; every call may return *several* actions to save budget. `ctx.act(action)` steps the
game and returns the next observation. By default each episode has fresh `ctx.memory` and module
globals. An explicitly configured shared-memory experiment may carry procedural knowledge between
seeds; action semantics are re-drawn, so never carry over seed-specific answers.

Evaluation metric: mean levels completed over seeds (primary), then System-2 calls (fewer is better).

What the harness must design: what to store (action→effect hypotheses, visited states, level maps),
what to retrieve and show Astra each turn (full history? a diff? an ASCII map with a legend?), how to
parse Astra's reply into actions robustly, how many actions to commit per call, when to run cheap
scripted exploration instead of asking Astra, and how to detect that a hypothesis was refuted.

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
| 0000 | 0000-tier1_astra_naive | 1.000 | 2 | – | 0 | 24 | ok |
| 0001 | 0001-previous_astra_winner | 2.000 | 2 | – | 0 | 7 | ok |
| 0002 | 0002 | – | – | – | – | – | not evaluated |

Best so far: `0001-previous_astra_winner` (mean score 2.000)

Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency and every game tick/step):
- candidates/0000-tier1_astra_naive/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0001-previous_astra_winner/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/0002/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/0002/proposal.md`: hypothesis, what changed vs. the parent
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