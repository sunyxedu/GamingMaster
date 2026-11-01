"""Jev controller with a short-horizon safety shield.

This keeps the parent candidate's one synchronous Jev choice per loop, but
moves the mechanical part of the decision out of Jev: NOOP and moves which
would become unsafe during the expected inference delay are not offered, and
answers are checked again against the newest observation before posting.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# The game in this run has these absolute directions.  The controller still
# obtains the advertised names from action_spec so that it posts the exact
# spelling accepted by the adaptor.
_VECTORS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
_HORIZON_TICKS = 3


def _cell(value):
    """Return a coordinate tuple, or None for an unusable coordinate."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError, OverflowError):
        return None


def _body(obs):
    raw = obs.get("snake") if isinstance(obs, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    result = []
    for value in raw:
        point = _cell(value)
        if point is None:
            return None
        result.append(point)
    return result


def _food(obs):
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _cell(obs.get("food"))


def _dimensions(obs):
    try:
        rows = int(obs.get("rows"))
        cols = int(obs.get("cols"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _heading(obs):
    direction = obs.get("direction") if isinstance(obs, dict) else None
    if direction in _VECTORS:
        return direction

    # This is only a malformed-observation fallback.  Normal observations
    # always provide direction explicitly.
    body = _body(obs)
    if body is not None and len(body) >= 2:
        dr = body[0][0] - body[1][0]
        dc = body[0][1] - body[1][1]
        for name, vector in _VECTORS.items():
            if vector == (dr, dc):
                return name
    return None


def _step_repeated(obs, requested, horizon):
    """Check repeated requests using the game's tail-vacating transition.

    The request is posted once and is the action that will be repeated by the
    real-time runner while the next Jev call is in flight.  A reverse request
    is modeled as the game's ignored reverse, although reverse actions are
    removed from the normal candidate set before this function is called.
    """
    if requested not in _VECTORS or not isinstance(obs, dict):
        return False
    body = _body(obs)
    dims = _dimensions(obs)
    current = _heading(obs)
    if body is None or dims is None or current is None:
        return False

    rows, cols = dims
    heading = current
    if requested != _OPPOSITE.get(current):
        heading = requested
    food = _food(obs)

    try:
        steps = max(1, int(horizon))
    except (TypeError, ValueError, OverflowError):
        return False

    for _ in range(steps):
        dr, dc = _VECTORS[heading]
        new_head = (body[0][0] + dr, body[0][1] + dc)
        if not (0 <= new_head[0] < rows and 0 <= new_head[1] < cols):
            return False

        eating = food is not None and new_head == food
        # On an ordinary move the old tail is removed before collision
        # checking.  On a food move it remains occupied.
        occupied = set(body if eating else body[:-1])
        if new_head in occupied:
            return False

        body = [new_head] + body
        if not eating:
            body.pop()
        else:
            # The next food is private RNG state and is not inferable from an
            # observation.  It cannot affect the wall/body safety guarantee
            # for this short shield; clear the known food after consuming it.
            food = None
    return True


def _direction_names(spec):
    names = list(getattr(spec, "names", ()) or ())
    noop = getattr(spec, "noop", None)
    return [name for name in names
            if isinstance(name, str) and name in _VECTORS and name != noop]


def _raw_safe(obs, directions):
    """Use the adaptor's safety aid only as a malformed-state fallback."""
    if not isinstance(obs, dict):
        return []
    listed = obs.get("safe_directions_now")
    if not isinstance(listed, list):
        return []
    return [name for name in directions if name in listed]


def _candidate_sets(obs, directions):
    """Return one-step and delay-resistant legal direction sets."""
    current = _heading(obs)
    if current in _VECTORS:
        allowed = [name for name in directions
                   if name != _OPPOSITE.get(current)]
    else:
        allowed = list(directions)

    one_step = [name for name in allowed if _step_repeated(obs, name, 1)]
    if not one_step:
        # If an observation is unexpectedly incomplete, retain the adaptor's
        # explicitly supplied one-step mask rather than inventing a move.
        one_step = _raw_safe(obs, allowed)

    delay_safe = [name for name in one_step
                  if _step_repeated(obs, name, _HORIZON_TICKS)]
    return one_step, delay_safe


def _fallback(obs, usable, one_step, directions, spec):
    """Choose an executable action without trusting a late/bad answer."""
    current = _heading(obs)
    pool = usable or one_step
    if current in pool:
        return current
    if pool:
        return pool[0]

    # There should be no such state while the game is live.  Prefer the
    # current heading, then a directional action, and only then the advertised
    # NOOP; all of these are valid API values and make pre-flight resilient to
    # an unusual stand-in observation.
    if current in directions:
        return current
    if directions:
        return directions[0]
    noop = getattr(spec, "noop", None)
    if noop in getattr(spec, "names", ()):
        return noop
    names = list(getattr(spec, "names", ()) or ())
    return names[0] if names else None


def _question(game, candidates, mode):
    spec = game.action_spec
    descriptions = getattr(spec, "descriptions", {}) or {}
    criteria = {}
    for name in candidates:
        base = descriptions.get(name, name)
        criteria[name] = (
            f"{base} This direction was checked in code and is {mode}; "
            "choose it only from the listed candidates."
        )
    return Choice(
        "Choose exactly one action from candidate_directions. "
        "The observation is a complete current Snake state. "
        "Do not output NOOP, a reverse direction, or any direction not in "
        "candidate_directions. Prefer a direction that makes progress toward "
        "the visible food while preserving the verified safety condition.",
        criteria=criteria,
    )


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system1", None) is None:
            return

        spec = game.action_spec
        directions = _direction_names(spec)
        if not directions:
            return

        while not ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            one_step, delay_safe = _candidate_sets(obs, directions)
            candidates = delay_safe or one_step
            if not candidates:
                # A live state with no safe move is already an emergency.  Do
                # not spend a model call on an impossible choice.
                if ctx.done or getattr(game, "done", False):
                    break
                ctx.wait_for_tick()
                continue

            mode = (f"safe for {_HORIZON_TICKS} repeated ticks"
                    if delay_safe else "safe for the next tick")
            state = {
                "observation": obs,
                "candidate_directions": list(candidates),
                "one_step_safe_directions": list(one_step),
                "candidate_safety_mode": mode,
                "horizon_ticks": _HORIZON_TICKS,
            }
            question = {"act": _question(game, candidates, mode)}

            try:
                answers = ctx.decide(state, question, timeout_s=5.0)
                answer = answers.get("act") if isinstance(answers, dict) else None
                proposed = getattr(answer, "value", None)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # A late/failed Jev call must not leave a stale straight move
                # running when a mechanical fallback is available.
                latest = game.observe()
                latest_one, latest_delay = _candidate_sets(latest, directions)
                action = _fallback(
                    latest, latest_delay, latest_one, directions, spec
                )
                if action is not None and not ctx.done and not getattr(game, "done", False):
                    ctx.act(action)
                continue

            # The answer may have judged an older tick.  Re-read and recheck
            # it before posting; this is the freshness gate missing from the
            # parent candidate.
            latest = game.observe()
            latest_one, latest_delay = _candidate_sets(latest, directions)
            latest_usable = latest_delay or latest_one
            if proposed in latest_usable:
                action = proposed
            else:
                action = _fallback(
                    latest, latest_delay, latest_one, directions, spec
                )

            if action is None or ctx.done or getattr(game, "done", False):
                break
            ctx.act(action)


def build(ctx):
    return Harness(ctx)
