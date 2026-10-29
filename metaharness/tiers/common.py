"""Text shared by the tier task specifications (what the proposer is told about the evaluation)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

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


def games_text(games: Sequence[str], kwargs: Sequence[dict]) -> tuple[str, list[Optional[float]]]:
    """One block per training game (description, action spec); returns the text and each game's tick_s."""
    from ..games.base import close_game
    from ..games.registry import make_game
    parts, ticks = [], []
    for name, kw in zip(games, kwargs):
        g = make_game(name, **kw)
        try:
            tick = getattr(g, "tick_s", None)
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
    return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"


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
        lines.append(f"- System-1 (Jev), one request with the game's initial observation and a Choice over its "
                     f"actions: {seconds(s1['latency_s'])}{in_ticks}"
                     + (f"; first request on a new connection {seconds(s1['first_latency_s'])}"
                        if s1.get("first_latency_s") else ""))
    errors = [f"{k}: {v['error']}" for k, v in cal.items() if isinstance(v, dict) and v.get("error")]
    if errors:
        lines.append("- calibration errors: " + "; ".join(errors))
    if not lines:
        return ""
    return ("\n\n## Measured model latency (at archive creation, " + str(cal.get("measured", "")) + ")\n"
            + "\n".join(lines)
            + "\nLater measurements per candidate are on the leaderboard (latency columns, a latency fit).")
