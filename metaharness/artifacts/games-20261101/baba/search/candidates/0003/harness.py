"""Short-horizon Astra plans with observation-based recovery.

The baseline spends one slow System-2 call for every primitive move.  This
adapter asks for a small, explicit action queue, executes it one turn at a
time, and interrupts the queue when the observed board stops changing.
"""
from __future__ import annotations

import json
import re

from metaharness.baselines._generic import obs_text
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _fingerprint(obs):
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(obs)


def _rules_fingerprint(obs):
    if isinstance(obs, dict):
        return _fingerprint(obs.get("rules"))
    return None


def _extract_actions(text, names):
    """Read a plan defensively; garbage model output must never escape."""
    if not isinstance(text, str):
        return []
    # Prefer the explicitly labelled queue and ignore a later prose rationale.
    m = re.search(r"actions?\s*:\s*(.*?)(?:\n\s*(?:rationale|reason|why)\s*:|$)",
                  text, flags=re.I | re.S)
    source = m.group(1) if m else text
    found = []
    for n in sorted((str(x) for x in names), key=len, reverse=True):
        for hit in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(n) +
                               r"(?![A-Za-z0-9_])", source, flags=re.I):
            found.append((hit.start(), n))
    found.sort(key=lambda p: p[0])
    # A long explanation can contain many accidental action words.  A plan is
    # deliberately short; the prompt asks for at most eight turns.
    return [n for _, n in found[:8]]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:
            return
        names = [str(x) for x in game.action_spec.names]
        fallback = names[0] if names else "noop"
        obs = game.observe()
        session = None
        queue = []
        previous = None
        calls = 0
        recent = []
        illegal_seen = 0

        while not ctx.done and not game.done:
            if not queue:
                remaining = ctx.budget.remaining()
                if remaining.get("system2_calls", 0) <= 0:
                    break
                # Leave a little wall-clock room for the final action/return.
                wall = float(remaining.get("wall_s", 0.0))
                if wall < 12.0:
                    break
                progress = ctx.progress()
                prompt = (
                    "You are the planning component for the fully observable, "
                    "turn-based Baba Is AI puzzle. Choose a feasible win and a "
                    "short next subgoal from the exact observation. Account for "
                    "all YOU objects, push geometry, and active rule changes. "
                    "Do not assume mechanics absent from the observation. Return "
                    "at most 8 primitive actions in one line beginning exactly "
                    "with `ACTIONS:` (comma or space separated), followed by an "
                    "optional brief rationale. The queue is replayed one turn at "
                    "a time, so make it safe to stop and replan after a push. "
                    "If the last action was blocked or the board differs from "
                    "expectation, explicitly recover from the current board.\n\n"
                    f"{game.describe()}\nAvailable actions: {game.action_spec.describe()}\n"
                    f"Current observation:\n{obs_text(obs, limit=12000)}\n"
                    f"Last actions: {recent[-12:]}\nProgress: {progress}\n"
                    "Use only the listed action names."
                )
                try:
                    # Archived worst calls were 208 s; this is above that while
                    # still respecting the episode's remaining deadline.
                    timeout = min(240.0, max(12.0, wall - 3.0))
                    result = ctx.think(prompt, session_id=session,
                                       timeout_s=timeout)
                    session = result.session_id
                    calls += 1
                    queue = _extract_actions(result.text, names)
                    ctx.log("plan", actions=queue, call=calls)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError):
                    # A transient/slow call is recoverable.  A later iteration
                    # gets a fresh state and another chance while the budget lasts.
                    queue = []
                    continue
                if not queue:
                    queue = [fallback]

            action = queue.pop(0)
            before = _fingerprint(obs)
            rules_before = _rules_fingerprint(obs)
            try:
                obs = ctx.act(action)
            except Exception as exc:
                # Invalid text or a game-side rejected action should trigger a
                # new plan, never terminate the harness with an error.
                ctx.log("act_error", action=action, error=repr(exc))
                queue = []
                continue
            recent.append(action)
            if len(recent) > 24:
                recent = recent[-24:]
            after = _fingerprint(obs)
            rules_after = _rules_fingerprint(obs)
            p = ctx.progress()
            # Directional actions that leave no board change are evidence that
            # the cached plan is stale (noop is allowed to be unchanged).
            if action != "noop" and after == before:
                queue = []
            if rules_before != rules_after:
                # A text push can alter control, blocking, and the goal at
                # once; never carry a pre-rule-change queue across it.
                queue = []
            illegal_now = int(p.get("illegal_actions", 0) or 0)
            if illegal_now > illegal_seen or p.get("since_obs_change", 0) >= 2:
                queue = []
            illegal_seen = max(illegal_seen, illegal_now)
            previous = after


def build(ctx):
    return Harness(ctx)
