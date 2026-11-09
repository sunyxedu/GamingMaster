"""Adaptive MiniGrid controller.

The model is used as a planner, while the turn-game loop executes a short
plan and checks every transition.  Keeping the plan short limits damage from
partial observability; batching several actions removes the one-call-per-step
latency of the seed baseline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from metaharness.baselines._generic import obs_text
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _state_key(obs: Any) -> str:
    """Return the part of an observation that should change after an effect."""
    if not isinstance(obs, dict):
        return repr(obs)
    keep = {k: v for k, v in obs.items()
            if k not in {"steps_used", "remaining_steps", "last_action", "last_reward",
                         "recent_actions", "terminated", "truncated"}}
    return json.dumps(keep, sort_keys=True, default=str, separators=(",", ":"))


def _actions_from(text: Any, names: Sequence[str]) -> list[str]:
    """Parse a plan without trusting formatting or prose from Astra."""
    raw = str(text or "")
    # The prompt requests an ACTIONS line.  Restrict parsing to that line first
    # so words such as "left" in a rationale cannot become extra commands.
    m = re.search(r"(?im)\bactions?\s*[:=]\s*([^\n\r]+)", raw)
    segment = m.group(1) if m else raw
    canonical = {n.lower(): n for n in names}
    if not canonical:
        return []
    pattern = "|".join(re.escape(n) for n in sorted(canonical, key=len, reverse=True))
    found: list[str] = []
    for hit in re.finditer(r"(?<![A-Za-z0-9_])(?:" + pattern + r")(?![A-Za-z0-9_])",
                           segment, flags=re.I):
        found.append(canonical.get(hit.group(0).lower(), hit.group(0)))
        if len(found) >= 8:
            break
    return found


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _prompt(self, game, obs, names, progress, reason: str, session: str | None) -> str:
        spec = game.action_spec.describe()
        rules = game.describe()
        if len(rules) > 6000:
            rules = rules[:6000] + " …(rules truncated)"
        return (
            "You control a synchronous MiniGrid/BabyAI turn-based episode. "
            "Plan from the public observation only; unknown cells are not empty. "
            "Interpret the exact mission and preserve instruction order. Check that "
            "pickup/drop/toggle are adjacent and facing the intended object. A blocked "
            "or ineffective action costs a step. Return ONLY one line in this format: "
            "ACTIONS: <between 1 and 8 action names separated by spaces>. Do not include "
            "a rationale, punctuation, numbering, or an action after the mission is solved. "
            "Use done only when the verifier explicitly needs it. The harness will execute "
            "the actions in order and replan if an effect is rejected or unexpected.\n\n"
            f"Game rules and task description:\n{rules}\n\n"
            f"Action names: {', '.join(names)}\nAction meanings:\n{spec}\n\n"
            f"Replan reason: {reason}\nProgress evidence: {json.dumps(progress, sort_keys=True)}\n"
            f"Observation:\n{obs_text(obs, limit=8500)}"
        )

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system2 is None:
            return
        names = list(game.action_spec.names)
        if not names:
            return
        obs = game.observe()
        plan: list[str] = []
        session: str | None = None
        reason = "initial state"
        failures = 0

        while not ctx.done and not game.done:
            if not plan:
                # A 130 s timeout is above the archive's 102 s worst call while
                # leaving room for several replans before the 1,800 s deadline.
                remaining = ctx.budget.remaining()
                # Leave a small grace period for returning after the deadline;
                # the call itself is guarded by the contract's wall check.
                if remaining.get("wall_s", 0.0) < 2.0:
                    break
                try:
                    progress = ctx.progress()
                except Exception:
                    progress = {}
                prompt = self._prompt(game, obs, names, progress, reason, session)
                try:
                    result = ctx.think(prompt, session_id=session, timeout_s=130.0)
                    session = getattr(result, "session_id", None) or session
                    plan = _actions_from(getattr(result, "text", ""), names)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError):
                    # A failed call must not strand the episode.  Take one cheap
                    # legal turn, then ask again if the wall budget permits it.
                    failures += 1
                    plan = []
                    fallback = "forward" if "forward" in names else names[0]
                    try:
                        nxt = ctx.act(fallback)
                        obs = nxt if nxt is not None else game.observe()
                    except Exception:
                        break
                    reason = "planner call failed; recover after fallback"
                    continue
                if not plan:
                    failures += 1
                    # Garbage stand-in/model output is expected during preflight.
                    # Use a bounded fallback rather than raising a parse error.
                    plan = ["forward" if "forward" in names else names[0]]
                    reason = "planner returned no parseable actions"

            action = plan.pop(0)
            before = _state_key(obs)
            try:
                before_progress = ctx.progress()
            except Exception:
                before_progress = {}
            try:
                nxt = ctx.act(action)
            except Exception:
                # Game-level illegal actions are evidence that the plan is stale;
                # replan rather than allowing an exception to escape the contract.
                plan = []
                reason = f"action {action!r} was rejected"
                continue
            obs = nxt if nxt is not None else game.observe()
            try:
                after_progress = ctx.progress()
            except Exception:
                after_progress = {}
            if ctx.done or game.done:
                break

            rejected = False
            try:
                rejected = (after_progress.get("illegal_actions", 0) >
                            before_progress.get("illegal_actions", 0))
            except Exception:
                pass
            # An interaction that leaves the semantic state unchanged almost
            # certainly targeted the wrong cell.  Forward in a symmetric room
            # may legitimately have the same rendered view, so rely on the
            # adaptor's explicit illegal-action counter for movement.
            interaction = action.lower() in {"pickup", "drop", "toggle"}
            ineffective = interaction and _state_key(obs) == before
            if rejected or ineffective:
                plan = []
                failures += 1
                reason = (
                    f"previous {action} had no accepted effect; replan with "
                    f"progress {json.dumps(after_progress, sort_keys=True)}"
                )
            elif action.lower() == "done":
                # A non-terminal done is evidence that completion was guessed;
                # do not let a batched plan continue from that stale assumption.
                plan = []
                reason = "done did not terminate the episode; verify the mission again"
            elif not plan:
                reason = "short plan completed; reassess newly visible state"


def build(ctx):
    return Harness(ctx)
