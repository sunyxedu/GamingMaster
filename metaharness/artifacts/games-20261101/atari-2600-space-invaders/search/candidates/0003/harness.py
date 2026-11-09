"""Target-aware Jev harness for real-time Space Invaders.

The parent asked Jev to select directly from six actions using the complete raw
observation.  This version keeps Jev as the only model, but makes movement,
fire-locking, danger response, and target alignment deterministic.  Jev judges
only a small tactical choice between safe target columns and dodge directions.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact(obs, previous=None):
    """Derive facts Jev can recognize without asking it to count or calculate."""
    inv = [x for x in (obs.get("invaders") or []) if isinstance(x, dict)]
    alive = [x for x in inv if x.get("alive")]
    by_col = {}
    for x in alive:
        c = x.get("col")
        if isinstance(c, int):
            by_col.setdefault(c, []).append(x)
    lowest = {}
    for c, xs in by_col.items():
        lowest[c] = max(xs, key=lambda x: _num(x.get("y")))
    player = obs.get("player") or {}
    shot = obs.get("player_shot") or {}
    bombs = [x for x in (obs.get("bombs") or []) if isinstance(x, dict)]
    px = _num(player.get("x"), 90)
    # These are named buckets, not model-side arithmetic tasks.
    near_bombs = [b for b in bombs if _num(b.get("y"), -999) >= 145]
    bomb_sides = []
    for b in near_bombs:
        bx = _num(b.get("x"), px)
        if bx < px - 12:
            bomb_sides.append("left")
        elif bx > px + 12:
            bomb_sides.append("right")
        else:
            bomb_sides.append("center")
    nearest = max(alive, key=lambda x: _num(x.get("y"), -999), default=None)
    return {
        "frame": obs.get("frame"),
        "score": obs.get("score"),
        "lives": obs.get("lives"),
        "wave": obs.get("wave"),
        "waves_cleared": obs.get("waves_cleared"),
        "invaders_killed": obs.get("invaders_killed"),
        "player_x": px,
        "shot_active": bool(shot.get("active")),
        "shot_x": shot.get("x"),
        "bomb_pressure": "high" if near_bombs else "low",
        "bomb_sides": bomb_sides,
        "nearest_threat": {
            "col": nearest.get("col") if nearest else None,
            "row": nearest.get("row") if nearest else None,
            "y_bucket": "low" if nearest and _num(nearest.get("y")) >= 130 else "upper",
        },
        "living_columns": sorted(by_col),
        "lowest_by_column": {
            str(c): {"row": x.get("row"), "y": x.get("y"), "x": x.get("x")}
            for c, x in lowest.items()
        },
        "previous": previous or {},
    }


def _urgent_dodge(obs):
    """Return a code-derived dodge when a bomb is close to the player lane."""
    p = obs.get("player") or {}
    px = _num(p.get("x"), 90)
    dangerous = []
    for b in (obs.get("bombs") or []):
        if not isinstance(b, dict):
            continue
        bx, by = _num(b.get("x"), px), _num(b.get("y"), -999)
        if by >= 140 and abs(bx - px) <= 16:
            dangerous.append(bx - px)
    if not dangerous:
        return None
    d = min(dangerous, key=abs)
    if d >= 0 and px > 28:
        return "left"
    if px < 132:
        return "right"
    return "left"


def _safe_action(obs, last_action="noop"):
    """Conservative code fallback while a Jev request is in flight."""
    dodge = _urgent_dodge(obs)
    if dodge:
        return dodge
    p = obs.get("player") or {}
    px = _num(p.get("x"), 90)
    shot = obs.get("player_shot") or {}
    if not shot.get("active"):
        return "fire"
    # Do not waste FIRE while the single laser is active; a gentle sweep keeps
    # the cannon from remaining parked under the same bomb lane.
    if px <= 30:
        return "right"
    if px >= 130:
        return "left"
    return "right" if last_action in ("noop", "fire", "left", "left_fire") else "left"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return
        spec = game.action_spec
        legal = set(getattr(spec, "names", ACTIONS))
        previous = {}
        last_action = "noop"
        # One batched typed judgment: emergency direction or a target column.
        questions = {
            "priority": Choice(
                "Using only the supplied symbolic state: if bomb_pressure is high, choose the direction that increases immediate safety; otherwise choose the best currently living target column. Use no numeric calculations. Return SAFE_LEFT, SAFE_RIGHT, HOLD, or one of the literal column labels COL0 through COL5. If no reliable target exists, choose HOLD.",
                {
                    "SAFE_LEFT": "A left dodge is safer because the dangerous bomb pressure is on the right or center, and the player is not at the left boundary.",
                    "SAFE_RIGHT": "A right dodge is safer because the dangerous bomb pressure is on the left or center, and the player is not at the right boundary.",
                    "HOLD": "No urgent dodge is needed; preserve the current lane for a shot.",
                    **{f"COL{i}": f"Prefer the living invader column {i}; it has a useful low surviving target and no explicit emergency dominates." for i in range(6)},
                },
            ),
            "danger": Choice(
                "Classify the immediate response from bomb_pressure and bomb_sides. Pick SAFE_LEFT or SAFE_RIGHT only for an urgent dodge; otherwise pick HOLD.",
                {
                    "SAFE_LEFT": "A left move avoids an immediately dangerous bomb.",
                    "SAFE_RIGHT": "A right move avoids an immediately dangerous bomb.",
                    "HOLD": "No immediate bomb dodge is required.",
                },
            ),
        }
        while not ctx.done and not game.done:
            obs = game.observe()
            if not isinstance(obs, dict):
                action = last_action if last_action in legal else "noop"
                ctx.act(action)
                last_action = action
                continue
            state = _compact(obs, previous)
            fallback = _safe_action(obs, last_action)
            # The fallback is deliberately executable before waiting for Jev.
            try:
                ans = ctx.decide(state, questions, timeout_s=0.22)
                if not isinstance(ans, dict):
                    raise RuntimeError("non-dict System-1 result")
                if ctx.done or game.done:
                    break
                # Jev judged the state at request time. Re-read the live world
                # before translating its semantic preference to an action.
                current = game.observe()
                if not isinstance(current, dict):
                    current = obs
                urgent = _urgent_dodge(current)
                choice = getattr(ans.get("danger"), "value", None)
                priority = getattr(ans.get("priority"), "value", None)
                if urgent is not None:
                    # Accept a typed direction only if it agrees with a current
                    # computed bomb emergency; otherwise use the coded dodge.
                    if choice == "SAFE_LEFT" and _num((current.get("player") or {}).get("x"), 90) > 25:
                        action = "left"
                    elif choice == "SAFE_RIGHT" and _num((current.get("player") or {}).get("x"), 90) < 135:
                        action = "right"
                    else:
                        action = urgent
                elif not (current.get("player_shot") or {}).get("active"):
                    # A column choice is converted to movement plus fire by
                    # code; Jev never has to invent an action string.
                    target = None
                    if isinstance(priority, str) and priority.startswith("COL"):
                        try:
                            target = int(priority[3:])
                        except ValueError:
                            target = None
                    living = _compact(current, previous)["lowest_by_column"]
                    if living and target not in living and str(target) not in living:
                        target = min(living, key=lambda c: abs(_num(living[c].get("x"), 90) - _num((current.get("player") or {}).get("x"), 90)))
                    tx = _num(living.get(str(target), {}).get("x"), _num((current.get("player") or {}).get("x"), 90)) if target is not None else _num((current.get("player") or {}).get("x"), 90)
                    px = _num((current.get("player") or {}).get("x"), 90)
                    if tx < px - 7:
                        # Align before firing: a moving shot is usually a miss,
                        # and the ROM permits only one laser at a time.
                        action = "left"
                    elif tx > px + 7:
                        action = "right"
                    else:
                        action = "fire"
                else:
                    action = _safe_action(current, last_action)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # On an early backend failure, let at least one game tick pass
                # before retrying instead of spinning and posting duplicates.
                if not ctx.done and not game.done:
                    ctx.wait_for_tick()
                    current = game.observe()
                    action = _safe_action(current, last_action) if isinstance(current, dict) else fallback
                else:
                    break
            if ctx.done or game.done:
                break
            if action not in legal:
                action = fallback if fallback in legal else "noop"
            ctx.act(action)
            last_action = action
            previous = {
                "action": action,
                "score": obs.get("score"),
                "lives": obs.get("lives"),
            }


def build(ctx):
    return Harness(ctx)
