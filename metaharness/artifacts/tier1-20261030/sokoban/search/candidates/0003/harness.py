"""Multi-action, verified Astra controller for ASCII Sokoban.

The model is used to choose a short macro plan, while this module owns the
cheap parts of execution: parsing, transition prediction, static reverse-push
dead squares, and checking that the adaptor produced the state predicted by a
plan.  A bad plan is discarded before an irreversible push is sent.
"""
from __future__ import annotations

import json
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ACTION_WORDS = tuple(DIRS)


def _as_int_pair(value):
    try:
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _state(obs):
    """Make a small canonical state from the adaptor's JSON observation."""
    board = obs.get("board", "") if isinstance(obs, dict) else ""
    rows = str(board).splitlines()
    try:
        height = int(obs.get("rows", len(rows)))
        width = int(obs.get("cols", max((len(r) for r in rows), default=0)))
    except Exception:
        height, width = len(rows), max((len(r) for r in rows), default=0)
    rows = (rows + [""] * max(0, height - len(rows)))[:height]
    passable = set()
    walls = set()
    for r in range(height):
        row = rows[r]
        for c in range(width):
            if c < len(row) and row[c] != "#":
                passable.add((r, c))
            else:
                walls.add((r, c))
    player = _as_int_pair(obs.get("player", (0, 0))) or (0, 0)
    boxes = set()
    for p in obs.get("boxes", ()) if isinstance(obs, dict) else ():
        q = _as_int_pair(p)
        if q is not None:
            boxes.add(q)
    goals = set()
    for p in obs.get("goals", ()) if isinstance(obs, dict) else ():
        q = _as_int_pair(p)
        if q is not None:
            goals.add(q)
    return {"h": height, "w": width, "passable": passable, "walls": walls,
            "player": player, "boxes": frozenset(boxes), "goals": frozenset(goals)}


def _signature(state):
    return (state["player"], tuple(sorted(state["boxes"])))


def _advance(state, action):
    """Predict one adaptor transition; changed is false for a blocked move."""
    if action not in DIRS:
        return state, False, False
    dr, dc = DIRS[action]
    pr, pc = state["player"]
    target = (pr + dr, pc + dc)
    if target not in state["passable"]:
        return state, False, False
    boxes = set(state["boxes"])
    pushed = False
    if target in boxes:
        beyond = (target[0] + dr, target[1] + dc)
        if beyond not in state["passable"] or beyond in boxes:
            return state, False, False
        boxes.remove(target)
        boxes.add(beyond)
        pushed = True
    nxt = dict(state)
    nxt["player"] = target
    nxt["boxes"] = frozenset(boxes)
    return nxt, True, pushed


def _reverse_dead_squares(state):
    """Cells from which a box can reach a goal under static geometry."""
    good = set(state["goals"])
    todo = deque(good)
    while todo:
        cur = todo.popleft()
        for dr, dc in DIRS.values():
            prev = (cur[0] - dr, cur[1] - dc)
            stand = (prev[0] - dr, prev[1] - dc)
            if prev in state["passable"] and stand in state["passable"] and prev not in good:
                good.add(prev)
                todo.append(prev)
    return state["passable"] - good


def _dynamic_deadlock(state, taboo):
    """Conservative checks used before sending a planned push."""
    for b in state["boxes"]:
        if b not in state["goals"] and b in taboo:
            return True
    # A filled 2x2 block containing an off-goal box is frozen.
    for r, c in state["boxes"]:
        for rr, cc in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)):
            cells = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
            if all(x in state["walls"] or x in state["boxes"] for x in cells):
                if any(x in state["boxes"] and x not in state["goals"] for x in cells):
                    return True
    return False


def _extract_plan(text, limit=60):
    """Parse a compact plan while surviving prose, markdown, and garbage output."""
    if not isinstance(text, str):
        return []
    aliases = {"u": "up", "d": "down", "l": "left", "r": "right",
               "up": "up", "down": "down", "left": "left", "right": "right"}

    def words(fragment):
        found = []
        for token in re.findall(r"(?<![A-Za-z0-9_])(up|down|left|right|u|d|l|r)(?![A-Za-z0-9_])",
                                fragment.lower()):
            found.append(aliases[token])
            if len(found) >= limit:
                break
        return found

    # The requested ACTIONS line is authoritative and prevents directions in
    # explanatory sentences from becoming accidental moves.
    marked = re.findall(r"(?im)\bactions?\s*[:=]\s*([^\n\r]+)", text)
    if marked:
        result = words(marked[-1])
        if result:
            return result
    for line in text.splitlines():
        result = words(line)
        tokens = re.findall(r"[A-Za-z]+", line.lower())
        if result and tokens and all(t in aliases or t in {"actions", "action"} for t in tokens):
            return result
    # Short fallback answers such as just "right" are common and safe.
    stripped = text.strip()
    if len(stripped) <= 120:
        return words(stripped)
    return []


def _fallback(state, visited):
    """Take a changed, non-taboo move when Astra is unavailable or malformed."""
    taboo = _reverse_dead_squares(state)
    candidates = []
    for action in ACTION_WORDS:
        nxt, changed, pushed = _advance(state, action)
        if not changed or _dynamic_deadlock(nxt, taboo):
            continue
        gain = 0
        if pushed:
            gain += 100 * sum(b in nxt["goals"] for b in nxt["boxes"])
            gain -= 2
        if _signature(nxt) not in visited:
            gain += 5
        candidates.append((gain, action, nxt))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1]
    return "noop"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _prompt(self, game, obs, state, taboo, failure=""):
        compact = {
            "board": obs.get("board"), "player": obs.get("player"),
            "boxes": obs.get("boxes"), "goals": obs.get("goals"),
            "reachable": obs.get("reachable"), "valid_actions": obs.get("valid_actions"),
            "pushes": obs.get("pushes"), "actions": obs.get("actions"),
            "boxes_on_goals": obs.get("boxes_on_goals"), "done": obs.get("done"),
        }
        return ("You are controlling a deterministic Sokoban board. Plan several primitive moves "
                "from the exact current state. Walls are #, goals ., boxes $, and the player is @. "
                "A push is irreversible, so preserve reachability and never put a box on a static "
                "dead square. The controller will verify every transition.\n\n"
                f"{game.describe()}\n"
                f"CURRENT JSON: {json.dumps(compact, separators=(',', ':'))}\n"
                f"STATIC DEAD SQUARES (never push a non-goal box there): {sorted(taboo)}\n"
                + (f"The previous proposed plan was rejected because: {failure}\n" if failure else "")
                + "Return one line only in this exact form: ACTIONS: up right down ... "
                "Use 1-60 words from up/down/left/right, omit noop, and stop before any uncertain push. "
                "Do not include explanation, punctuation, or code fences.")

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system2", None) is None:
            return
        try:
            obs = game.observe()
        except Exception:
            return
        session = None
        plan = []
        state = _state(obs)
        taboo = _reverse_dead_squares(state)
        visited = {_signature(state)}
        failure = ""
        while not ctx.done and not bool(obs.get("done", False)):
            # Plans are generated only when the verified queue is empty.  A
            # single response can therefore amortize Astra's 40-180 second call.
            if not plan:
                remaining = {}
                try:
                    remaining = ctx.budget.remaining()
                except Exception:
                    pass
                wall_left = float(remaining.get("wall_s", 1800.0) or 0.0)
                if wall_left < 8.0:
                    action = _fallback(state, visited)
                    try:
                        obs = ctx.act(action)
                    except Exception:
                        break
                    state = _state(obs)
                    visited.add(_signature(state))
                    continue
                prompt = self._prompt(game, obs, state, taboo, failure)
                failure = ""
                try:
                    timeout = max(8.0, min(220.0, wall_left - 3.0))
                    result = ctx.think(prompt, session_id=session, timeout_s=timeout)
                    session = getattr(result, "session_id", session)
                    plan = _extract_plan(getattr(result, "text", ""))
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError):
                    plan = []
                    failure = "the model call failed or timed out"
                except Exception:
                    plan = []
                    failure = "the model returned an unusable response"
                # Keep model mistakes from spinning on noop or malformed text.
                if not plan:
                    action = _fallback(state, visited)
                    try:
                        obs = ctx.act(action)
                    except Exception:
                        break
                    state = _state(obs)
                    visited.add(_signature(state))
                    continue

            action = plan.pop(0)
            predicted, changed, pushed = _advance(state, action)
            if (not changed or (pushed and _dynamic_deadlock(predicted, taboo))):
                failure = "the next proposed move was blocked or would freeze a box"
                plan = []
                continue
            try:
                new_obs = ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                break
            except Exception:
                plan = []
                failure = "the adaptor rejected the proposed action"
                continue
            new_state = _state(new_obs)
            if _signature(new_state) != _signature(predicted):
                # Do not continue a stale macro after an unexpected transition.
                plan = []
                failure = "the observed transition differed from prediction"
            obs, state = new_obs, new_state
            visited.add(_signature(state))
            if pushed and _dynamic_deadlock(state, taboo):
                plan = []
                failure = "the observed push created a deadlock"
            if bool(obs.get("done", False)):
                break


def build(ctx):
    return Harness(ctx)
