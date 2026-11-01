"""Jev-only Q*bert controller with a stale-action safety interlock.

This keeps the parent controller's synchronous one-Choice architecture.  The
interlock uses only the adaptor's current, decoded safe_moves: before waiting
for Jev it replaces an action that is no longer safe, and after the call it
rechecks the answer against a fresh observation.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


_DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")


def _names(spec):
    try:
        return list(spec.names)
    except Exception:
        return ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"]


def _noop(spec, names):
    value = getattr(spec, "noop", None)
    if isinstance(value, str) and value in names:
        return value
    if "NOOP" in names:
        return "NOOP"
    return names[0] if names else "NOOP"


def _safe_actions(observation, names):
    """Return ROM-certified directional choices, preserving decoder order."""
    try:
        raw = observation.get("safe_moves", [])
    except Exception:
        raw = []
    if not isinstance(raw, (list, tuple)):
        return []
    allowed = set(names)
    return [name for name in raw if isinstance(name, str)
            and name in allowed and name in _DIRECTIONS]


def _last_action(observation):
    try:
        value = observation.get("last_action")
    except Exception:
        value = None
    return value if isinstance(value, str) else None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _fallback(self, observation, spec, names):
        safe = _safe_actions(observation, names)
        last = _last_action(observation)
        # Keeping an already-certified action avoids needless reversals during
        # an animation.  Otherwise the adaptor's ordered safe set is the
        # conservative choice.  With no certified edge, neutral is safer than
        # inventing a direction or ever pressing FIRE.
        if last in safe:
            return last
        if safe:
            return safe[0]
        return _noop(spec, names)

    def _guard_before_wait(self, observation, spec, names):
        """Brake a stale edge before a synchronous Jev request can take time."""
        safe = _safe_actions(observation, names)
        last = _last_action(observation)
        if last in safe:
            return
        action = safe[0] if safe else _noop(spec, names)
        if action != last:
            self.ctx.act(action)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system1 is None:
            return

        spec = game.action_spec
        names = _names(spec)
        descriptions = getattr(spec, "descriptions", {}) or {}

        while not ctx.done and not getattr(game, "done", False):
            observation = game.observe()
            if not isinstance(observation, dict):
                # The real adaptor returns strict JSON.  A neutral action keeps
                # the pre-flight harness alive if a stand-in supplies less.
                ctx.act(_noop(spec, names))
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, ModelUnavailable):
                    raise
                continue

            # The parent blocks on Jev.  This small code-side action is posted
            # first so the previous joystick edge cannot remain active while
            # the answer is in flight after it has become unsafe.
            self._guard_before_wait(observation, spec, names)
            safe = _safe_actions(observation, names)
            options = safe if safe else [_noop(spec, names)]
            # De-duplicate while retaining the adaptor's order.
            options = list(dict.fromkeys(name for name in options if name in names))
            if not options:
                options = [_noop(spec, names)]
            criteria = {
                name: descriptions.get(name, name)
                for name in options
            }
            question = Choice(
                instructions=(
                    f"{game.describe()}\n"
                    "Choose exactly one action from the criteria. The current "
                    "observation's safe_moves are a code-verified subset of "
                    "joystick edges; never invent an edge, use FIRE, or choose "
                    "an option outside the criteria. Prefer a useful safe hop."
                ),
                criteria=criteria,
            )

            try:
                answers = ctx.decide(
                    observation,
                    {"act": question},
                    timeout_s=2.0,
                )
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The guard remains posted while this failed request is
                # discarded.  One tick of waiting prevents a tight retry loop.
                if ctx.done:
                    return
                ctx.wait_for_tick()
                continue

            if ctx.done or getattr(game, "done", False):
                return

            # Jev judged an earlier observation.  Re-read before acting and
            # reject an answer that is no longer in the current safe set.
            latest = game.observe()
            latest_safe = _safe_actions(latest, names) if isinstance(latest, dict) else []
            value = None
            try:
                answer = answers.get("act")
                value = getattr(answer, "value", None)
            except Exception:
                value = None
            if value in latest_safe:
                action = value
            else:
                action = self._fallback(latest, spec, names) if isinstance(latest, dict) else _noop(spec, names)
            ctx.act(action)


def build(ctx):
    return Harness(ctx)
