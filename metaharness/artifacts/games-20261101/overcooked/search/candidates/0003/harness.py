"""Astra-only parent with a one-tick, observation-driven traffic-yield fallback.

Astra still chooses one primitive per sequential, session-linked request.  The
only added controller evacuates an occupied floor tile when the teammate faces
into it.  It never fetches, cooks, plates, serves, or interprets recipes.
"""
from __future__ import annotations

import json
from functools import partial

from metaharness.baselines._generic import obs_text, parse_action
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


DIRECTIONS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


def _point(value):
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
        return tuple(value)
    return None


def _frame(observation):
    """Use only the public observation; never infer the controlled player index."""
    if isinstance(observation, str):
        try:
            observation = json.loads(observation)
        except (TypeError, ValueError):
            return None
    if not isinstance(observation, dict):
        return None
    state = observation.get("state", {})
    if not isinstance(state, dict):
        return None
    players = state.get("players", [])
    index = observation.get("controlled_agent_index")
    if index not in (0, 1) or len(players) != 2:
        return None
    me, partner = players[index], players[1 - index]
    if not isinstance(me, dict) or not isinstance(partner, dict):
        return None
    position = _point(me.get("position"))
    other = _point(partner.get("position"))
    facing = _point(partner.get("orientation"))
    terrain = observation.get("terrain_mtx")
    if position is None or other is None or not isinstance(terrain, list):
        return None
    floor = {
        (x, y)
        for y, row in enumerate(terrain)
        if isinstance(row, (list, str))
        for x, cell in enumerate(row)
        if cell == " "
    }
    return {
        "position": position,
        "partner": other,
        "partner_facing": facing,
        "floor": floor,
        "terrain": terrain,
        "episode": observation.get("episode_id"),
        "tick": observation.get("tick", state.get("timestep")),
        "self_signature": (index, position, _point(me.get("orientation")),
                           json.dumps(me.get("held_object"), sort_keys=True)),
    }


def _plus(position, direction):
    return position[0] + direction[0], position[1] + direction[1]


def _yield_action(frame, names):
    """A teammate facing our adjacent floor tile has right of way.

    A free neighbor other than the teammate is collision-safe in a cardinal
    grid: adjacent cells have no common cardinal neighbor, and this move cannot
    swap their positions.  No prediction of the scripted policy is required.
    """
    if frame is None or frame["partner_facing"] not in DIRECTIONS.values():
        return None
    me, other = frame["position"], frame["partner"]
    if _plus(other, frame["partner_facing"]) != me:
        return None
    choices = []
    for order, (action, delta) in enumerate(DIRECTIONS.items()):
        target = _plus(me, delta)
        if action not in names or target not in frame["floor"] or target == other:
            continue
        # Prefer a side step over moving further along the teammate's heading;
        # otherwise a one-cell corridor simply advances the yielding chef.
        forward = int(delta == frame["partner_facing"])
        exits = sum(_plus(target, d) in frame["floor"]
                    and _plus(target, d) != me for d in DIRECTIONS.values())
        choices.append(((forward, -exits, order), action, target))
    if not choices:
        return None
    _, action, target = min(choices)
    return action, target


def _valid_reply(action, origin, current):
    """Do not execute a delayed instruction from a pose changed by yielding.

    Movement uses unchanged self pose/hand, not an arbitrary wall-time cutoff.
    INTERACT additionally needs the same observation tick: unlike movement,
    its object and pot preconditions may have changed while Astra was thinking.
    """
    if origin is None or current is None:
        return action == "stay"
    if (origin["episode"] != current["episode"]
            or origin["self_signature"] != current["self_signature"]):
        return False
    if action == "interact":
        return origin["tick"] == current["tick"]
    if action in DIRECTIONS:
        return _plus(current["position"], DIRECTIONS[action]) != current["partner"]
    return action == "stay"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        spec = game.action_spec
        names = list(spec.names)
        noop = spec.noop if spec.noop in names else "stay"
        if noop not in names:
            noop = names[0]
        descriptions = spec.descriptions
        action_text = "\n".join(
            "- " + name + ": " + str(descriptions.get(name, name))
            for name in names
        )
        description = game.describe()
        notes = getattr(spec, "notes", "")
        session = None
        pending = None
        origin = None
        previous_yield = None

        while not ctx.done:
            observation = ctx.observe()
            current = _frame(observation)
            if previous_yield is not None and current is not None:
                ctx.log("yield_outcome", tick=current["tick"],
                        moved=current["position"] == previous_yield,
                        position=list(current["position"]))
                previous_yield = None

            # Futures contain only one bounded inference call, never a worker
            # control loop.  The runner cancels an in-flight call at game end.
            action = noop
            if pending is not None and pending.done():
                try:
                    result = pending.result()
                except ModelUnavailable:
                    raise
                except BudgetExceeded:
                    return
                except (TimeoutError, RuntimeError) as error:
                    ctx.log("astra_retry", error=type(error).__name__)
                else:
                    session = getattr(result, "session_id", session)
                    text = getattr(result, "text", "")
                    proposed = parse_action(text if isinstance(text, str) else "",
                                            names, default=noop)
                    if proposed not in names:
                        proposed = noop
                    if _valid_reply(proposed, origin, current):
                        action = proposed
                        ctx.log("astra_action", action=action,
                                request_tick=origin["tick"] if origin else None,
                                tick=current["tick"] if current else None)
                    else:
                        ctx.log("astra_reply_invalidated", action=proposed,
                                reason="pose_or_interaction_state_changed",
                                request_tick=origin["tick"] if origin else None,
                                tick=current["tick"] if current else None)
                pending = None

            evacuation = _yield_action(current, names)
            if evacuation is not None:
                action, previous_yield = evacuation
                ctx.log("traffic_yield", tick=current["tick"], action=action,
                        position=list(current["position"]),
                        partner=list(current["partner"]),
                        destination=list(previous_yield))

            # Every selected primitive is a single-tick lease.  While waiting
            # for the next answer, replace it with stay or another necessary
            # yield, rather than blindly repeating it for tens of seconds.
            if ctx.done:
                break
            ctx.act(action)

            if pending is None and not ctx.done:
                # Keep the parent's full observation, single-action prompt,
                # session continuity, timeout and single outstanding S2 call.
                prompt = (f"{description}\nActions:\n{action_text}\n{notes}\n\n"
                          f"Observation:\n{obs_text(observation)}\n"
                          "Reply with exactly one action name and nothing else.")
                origin = current
                pending = ctx.spawn(partial(ctx.think, prompt, session_id=session))
            if ctx.done:
                break
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
