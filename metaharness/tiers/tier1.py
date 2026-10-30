"""Tier 1 — no latency pressure: Astra plays through the harness; the harness
designs memory / retrieval / presentation / exploration strategy."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from ..core import contract
from ..games.base import close_game
from ..games.registry import make_game
from .common import GENERALITY, SCOPE, calibration_text, memory_text, seconds

BASELINES = ["tier1_astra_naive"]

TASK_TEMPLATE = """Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `{game}` — {describe}

Actions:
{actions}

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: {s2} calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of {deadline}, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. {memory}

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds."""


STUDY_SECTION = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

{study}
"""

STUDY_SECTION_GENERAL = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it for prompts and
for checks against observations; because the harness is also scored on a held-out game, keep
game-specific rules out of hard-coded logic (or gate them on recognising this game).

{study}
"""


def task_md(game_name: str, budget: contract.Budget, clock_mode: str, game_kwargs: dict,
            study: str = "", *, generality: bool = True, deadline_s: Optional[float] = None,
            memory_mode: str = "episode", calibration: Optional[dict[str, Any]] = None) -> str:
    """``study``: text of the study phase (rules, interface, strategies) appended as prior knowledge.
    ``generality``: a held-out game will be evaluated (adds the generality requirement)."""
    g = make_game(game_name, **game_kwargs)
    try:
        task = TASK_TEMPLATE.format(game=game_name, describe=g.describe(), actions=g.action_spec.describe(),
                                    s2=budget.max_system2_calls,
                                    deadline=seconds(budget.max_wall_s if deadline_s is None else deadline_s),
                                    memory=memory_text(memory_mode))
    finally:
        close_game(g)
    task += GENERALITY if generality else SCOPE
    if study:
        task += (STUDY_SECTION_GENERAL if generality else STUDY_SECTION).format(study=study)
    return task + calibration_text(calibration)


def contract_doc() -> str:
    return inspect.getdoc(contract) or ""


def baseline_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "baselines"
    return [root / f"{b}.py" for b in BASELINES]
