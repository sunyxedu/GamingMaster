"""Tier 2 — games with some latency pressure: Astra thinks, Jev acts, the
harness (designed by the meta-harness) decides how they talk to each other
and to the game."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from ..core import contract
from ..models.jev import usage_guide
from .common import GENERALITY, MULTI_GAME, SCOPE, as_lists, calibration_text, games_text, memory_text, seconds

BASELINES = ["tier2_jev_only", "tier2_astra_only", "tier2_idle"]

TASK_TEMPLATE = """Design a harness for a **real-time** game (tier 2).

{games}

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: {s2} calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: {s1} calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks {tick} **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
{deadline} (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. {memory} Clock mode: {clock}.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late."""


STUDY_SECTION = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

{study}
"""

STUDY_SECTION_GENERAL = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it for Astra
prompts, Jev questions and checks against observations; because the harness is also scored on a
held-out game, keep game-specific rules out of hard-coded logic (or gate them on recognising this game).

{study}
"""


def task_md(game_name: Any, budget: contract.Budget, clock_mode: str, game_kwargs: Any,
            study: str = "", *, generality: bool = True, deadline_s: Optional[float] = None,
            memory_mode: str = "episode", calibration: Optional[dict[str, Any]] = None) -> str:
    """``game_name`` / ``game_kwargs``: one game or a list of training games (and their kwargs).
    ``study``: text of the study phase (rules, interface, strategies) appended as prior knowledge.
    ``generality``: a held-out game will be evaluated (adds the generality requirement)."""
    names, kws = as_lists(game_name, game_kwargs)
    games, ticks = games_text(names, kws)
    tick_s = ticks[0]
    task = TASK_TEMPLATE.format(games=games, s2=budget.max_system2_calls, s1=budget.max_system1_calls,
                                tick=(f"every {tick_s}s" if len(names) == 1 else "at its own pace (each game states its tick)"),
                                clock=clock_mode, memory=memory_text(memory_mode),
                                deadline=seconds(budget.max_wall_s if deadline_s is None else deadline_s))
    if len(names) > 1:
        task += MULTI_GAME
        generality = True
    task += GENERALITY if generality else SCOPE
    if study:
        task += (STUDY_SECTION_GENERAL if generality else STUDY_SECTION).format(study=study)
    # Inline the snapshot: proposers are restricted to the archive and cannot
    # rely on following web links or reading the installed model source.
    # It becomes part of task.md, each candidate prompt and the resume manifest.
    return task + calibration_text(calibration, tick_s) + "\n\n" + usage_guide()


def contract_doc() -> str:
    return inspect.getdoc(contract) or ""


def baseline_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "baselines"
    return [root / f"{b}.py" for b in BASELINES]
