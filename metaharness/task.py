"""The task specification the proposer gets: one text for every game.

The models are the same for every game — Astra (System-2) and Jev (System-1); the task says what
each is for. What differs between games is stated, not assumed: the game's pace (turn-based — the
world waits; real-time — it ticks on its own clock), the budgets and the measured latencies.
Nothing here is specific to a game."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional, Sequence

from .core import contract
from .games.base import close_game
from .games.registry import make_game

TASK_TEMPLATE = """Design a harness for the game below. The harness owns the control loop of one episode: it shows
the game to the models, turns their answers into actions and decides when to call whom.

{games}

{pace}

Models available through `ctx` (the contract has the exact signatures):
* `ctx.think(prompt, session_id=...)` — Astra (System-2): slow (seconds to a minute, growing with the
  length of the answer), strong reasoning; plans, explains rules, writes code; one call may return a
  plan of several actions. Returns `.text` and `.session_id` (pass it back to keep conversation
  context). Budget: {s2} calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1): fast (hundreds of milliseconds) typed judgments about
  a JSON state — a Choice among options, a Score on a scale, a yes/no with a probability — with
  calibrated probabilities and a confidence back; one request answers *many independent questions
  about one state in parallel* for the price and latency of one, and its budget ({s1} calls per
  episode) is two orders of magnitude larger than Astra's. What it is good at: recognising a situation
  and picking among concrete options under time pressure — which of these moves, is this state
  dangerous, how close is this to the goal, does this plan still apply. What it does not do: plan,
  do arithmetic, or follow long chains of inference (those are Astra's or code's). Read the complete
  Jev model context and local usage contract at the end before designing its state, questions and
  answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — a background thread, so a slow call can run while the game is kept alive.

Each episode has a wall-clock deadline of {deadline}, including model latency
(`ctx.budget.remaining()["wall_s"]`); when it passes, or a call budget is used up, the episode ends with
the score reached. {memory}

Evaluation: mean score over the training seeds (the game's own score, see the description){tie}, then
fewer System-2 calls, then {steps}. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task. Where Jev is used: how Astra's slow output (a plan, a rule summary, a
policy sketch, a subgoal) is turned into *state + questions* for Jev, how Jev's typed answer is turned
into a game action, when to go back to Astra, and what to do when a decision is late. The evaluator
reports how many posted actions were preceded by a fresh Jev / Astra answer, so the division of labour
is visible.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds."""

PACE_TURN = """**Pace: turn-based** — the world waits for the agent. `ctx.act(action)` applies one action and returns
the next observation."""

PACE_REALTIME = """**Pace: real-time** — the game ticks {tick} **on its own clock**; `ctx.act(a)` posts the current action
(instant), a tick with no fresh action re-applies the last one, `ctx.wait_for_tick()` blocks until the
next tick. Clock mode: {clock}. The idle baseline (never acts) shows what doing nothing scores."""

GENERALITY = """

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game."""

SCOPE = """

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds."""

MULTI_GAME = """
**Several training games.** Selection normalises each game's mean score across candidates and ranks
candidates by the mean normalised score, then by their worst game: a harness must work on every game.
Learn each game's rules and observation fields from `game.describe()` and its observations at run
time; share the machinery, not the game knowledge."""

STUDY_SECTION = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

{study}
"""

STUDY_SECTION_GENERAL = """

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it for Astra
prompts, Jev questions and checks against observations; because the harness is also scored on a held-out game,
keep game-specific rules out of hard-coded logic (or gate them on recognising this game).

{study}
"""


def games_text(games: Sequence[str], kwargs: Sequence[dict]) -> tuple[str, list[Optional[float]]]:
    """One block per training game (description, action spec); returns the text and each game's tick_s
    (None for a turn-based game)."""
    parts, ticks = [], []
    for name, kw in zip(games, kwargs):
        g = make_game(name, **kw)
        try:
            tick = getattr(g, "tick_s", None) if hasattr(g, "tick") else None
            head = f"Game: `{name}`" if len(games) == 1 else f"### Training game `{name}`"
            parts.append(f"{head} — {g.describe()}\n\nActions:\n{g.action_spec.describe()}"
                         + (f"\nTick: every {tick}s." if tick and len(games) > 1 else ""))
            ticks.append(tick)
        finally:
            close_game(g)
    return "\n\n".join(parts), ticks


def as_lists(game_name: Any, game_kwargs: Any) -> tuple[list[str], list[dict]]:
    names = [game_name] if isinstance(game_name, str) else list(game_name)
    kws = [dict(game_kwargs or {})] * len(names) if isinstance(game_kwargs, dict) or game_kwargs is None \
        else [dict(k or {}) for k in game_kwargs]
    return names, kws


def memory_text(memory_mode: str) -> str:
    if memory_mode == "shared":
        return ("This evaluation shares `ctx.memory` across the training seeds, in order: procedural knowledge "
                "may carry over, but seeds are different scenarios, so never carry over seed-specific answers.")
    return "Each episode starts cold: fresh `ctx.memory` and fresh module globals."


def seconds(s: float) -> str:
    return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s" if s >= 1 else f"{s * 1000:.0f} ms"


def calibration_text(cal: Optional[dict[str, Any]], tick_s: Optional[float] = None) -> str:
    """The latency measured when the archive was created (``core/calibrate.py``), for the task spec."""
    if not cal:
        return ""
    lines = []
    short, long = cal.get("system2_short") or {}, cal.get("system2_long") or {}
    if "latency_s" in short:
        lines.append(f"- System-2 (Astra), one-word answer: {seconds(short['latency_s'])}")
    if "latency_s" in long:
        tok = long.get("output_tokens")
        lines.append(f"- System-2, an answer of {'~' + str(tok) + ' output tokens' if tok else 'about 600 words'}: "
                     f"{seconds(long['latency_s'])} — latency grows with the length of the answer; size "
                     "timeouts for the answers your prompts actually request")
    s1 = cal.get("system1") or {}
    if "latency_s" in s1:
        in_ticks = f" (= {s1['latency_s'] / tick_s:.1f} ticks)" if tick_s else ""
        tail = (f", p90 {seconds(s1['p90_s'])}, max {seconds(s1['max_s'])} over {s1['n']} warm requests"
                if s1.get("p90_s") is not None else "")
        lines.append(f"- System-1 (Jev), one request with the game's initial observation and a Choice over its "
                     f"actions: median {seconds(s1['latency_s'])}{in_ticks}{tail}"
                     + (f"; first request on a new connection {seconds(s1['first_latency_s'])}"
                        if s1.get("first_latency_s") else "")
                     + " — a timeout below the tail turns slow answers into failed calls")
    errors = [f"{k}: {v['error']}" for k, v in cal.items() if isinstance(v, dict) and v.get("error")]
    if errors:
        lines.append("- calibration errors: " + "; ".join(errors))
    if not lines:
        return ""
    return ("\n\n## Measured model latency (at archive creation, " + str(cal.get("measured", "")) + ")\n"
            + "\n".join(lines)
            + "\nLater measurements per candidate are on the leaderboard (latency columns, a latency fit).")


def task_md(game_name: Any, budget: contract.Budget, clock_mode: str, game_kwargs: Any, study: str = "", *,
            generality: bool = True, deadline_s: Optional[float] = None, memory_mode: str = "episode",
            calibration: Optional[dict[str, Any]] = None) -> str:
    """``game_name`` / ``game_kwargs``: one game or a list of training games (and their kwargs); the
    games' kind (turn-based / real-time) comes from the games themselves. ``study``: text of the study
    phase appended as prior knowledge. ``generality``: a held-out game will be evaluated."""
    names, kws = as_lists(game_name, game_kwargs)
    games, ticks = games_text(names, kws)
    tick_s, realtime = ticks[0], any(t is not None for t in ticks)
    if realtime:
        pace = PACE_REALTIME.format(tick=f"every {tick_s}s" if len(names) == 1 else
                                    "at its own pace (each game states its tick)", clock=clock_mode)
    else:
        pace = PACE_TURN
    task = TASK_TEMPLATE.format(
        games=games, pace=pace, s2=budget.max_system2_calls, s1=budget.max_system1_calls,
        memory=memory_text(memory_mode), deadline=seconds(budget.max_wall_s if deadline_s is None else deadline_s),
        tie=", then fraction of stale ticks (lower is better)" if realtime else "",
        steps="longer survival" if realtime else "fewer actions")
    if len(names) > 1:
        task += MULTI_GAME
        generality = True
    task += GENERALITY if generality else SCOPE
    if study:
        task += (STUDY_SECTION_GENERAL if generality else STUDY_SECTION).format(study=study)
    # Inline the Jev snapshot: proposers are restricted to the archive and cannot rely on following web
    # links or reading the installed model source. It becomes part of task.md and the manifest.
    from .models.jev import usage_guide
    return task + calibration_text(calibration, tick_s) + "\n\n" + usage_guide()


def contract_doc() -> str:
    return inspect.getdoc(contract) or ""


def baseline_paths(realtime: bool) -> list[Path]:
    """The seed harnesses an archive starts with: Jev alone, naive Astra, and — for a real-time game —
    doing nothing."""
    root = Path(__file__).resolve().parent / "baselines"
    names = ["jev_only", "astra_naive"] + (["idle"] if realtime else [])
    return [root / f"{n}.py" for n in names]
