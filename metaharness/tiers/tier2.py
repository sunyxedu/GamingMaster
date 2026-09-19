"""Tier 2 — games with some latency pressure: Astra thinks, Jev acts, the
harness (designed by the meta-harness) decides how they talk to each other
and to the game."""
from __future__ import annotations

import inspect
from pathlib import Path

from ..core import contract
from ..games.registry import make_game

BASELINES = ["tier2_jev_only", "tier2_astra_only"]

TASK_TEMPLATE = """Design a harness for a **real-time** game (tier 2).

Game: `{game}` — {describe}

Actions:
{actions}

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow: seconds). Deep reasoning, can plan,
  can explain rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: {s2} calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast: ~0.1–0.7 s). Typed questions
  (`metaharness.models.jev.Choice/Score/Noul`), calibrated probabilities. Budget: {s1} calls per
  episode. Jev is a zero-shot classifier: it does not reason, so what you put in `state` and how you
  phrase `instructions`/`criteria` is the whole game.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

The game ticks every {tick_s}s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Evaluation metric: mean score over seeds
(primary), then fraction of stale ticks (lower is better), then System-2 calls (fewer is better).
Clock mode: {clock}.

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
on the held-out game."""


def task_md(game_name: str, budget: contract.Budget, clock_mode: str, game_kwargs: dict) -> str:
    g = make_game(game_name, **game_kwargs)
    return TASK_TEMPLATE.format(game=game_name, describe=g.describe(), actions=g.action_spec.describe(),
                                s2=budget.max_system2_calls, s1=budget.max_system1_calls,
                                tick_s=g.tick_s, clock=clock_mode)


def contract_doc() -> str:
    return inspect.getdoc(contract) or ""


def baseline_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "baselines"
    return [root / f"{b}.py" for b in BASELINES]
