"""Tier 1 — no latency pressure: Astra plays through the harness; the harness
designs memory / retrieval / presentation / exploration strategy."""
from __future__ import annotations

import inspect
from pathlib import Path

from ..core import contract
from ..games.registry import make_game

BASELINES = ["tier1_astra_naive"]

TASK_TEMPLATE = """Design a harness for a **turn-based, hidden-rule** game (tier 1, ARC-AGI-3 style).

Game: `{game}` — {describe}

Actions:
{actions}

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: {s2}
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
on the held-out game."""


STUDY_SECTION = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules/bookkeeping/search as code, put planning and adaptation into Astra prompts, and fast recognition
into Jev questions — but verify every claim against observations, because sources can be wrong or
outdated for this build of the game.

{study}
"""


def task_md(game_name: str, budget: contract.Budget, clock_mode: str, game_kwargs: dict,
            study: str = "") -> str:
    """``study``: text of the study phase (rules, interface, strategies) appended as prior knowledge."""
    g = make_game(game_name, **game_kwargs)
    task = TASK_TEMPLATE.format(game=game_name, describe=g.describe(), actions=g.action_spec.describe(),
                                s2=budget.max_system2_calls)
    return task + (STUDY_SECTION.format(study=study) if study else "")


def contract_doc() -> str:
    return inspect.getdoc(contract) or ""


def baseline_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "baselines"
    return [root / f"{b}.py" for b in BASELINES]
