"""Adaptive, seed-independent 2048 controller.

The controller keeps the arithmetic and transition model local.  Astra chooses
an orientation/risk regime only at strategic boundaries; a short expectimax
then chooses each move.  The orientation is re-estimated from the board, so a
corner that has been damaged does not remain a blind fixed commitment.
"""
from __future__ import annotations

import json
import math
import re
import time

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


DIRS = ("up", "down", "left", "right")
ORIENTATIONS = ("top-left", "top-right", "bottom-left", "bottom-right")
ORDERS = {
    "top-left": (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12),
    "top-right": (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15),
    "bottom-left": (12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0),
    "bottom-right": (15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3),
}
CORNER = {"top-left": 0, "top-right": 3, "bottom-left": 12, "bottom-right": 15}


def _line(values):
    vals = [int(v) for v in values if v]
    out, gain, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            v = vals[i] * 2
            out.append(v)
            gain += v
            i += 2
        else:
            out.append(vals[i])
            i += 1
    return tuple(out + [0] * (4 - len(out))), gain


def transition(board, action):
    """Exact slide and merge result, before the random spawn."""
    b = list(board)
    gain = 0
    if action in ("left", "right"):
        for r in range(4):
            ids = tuple(4 * r + c for c in range(4))
            if action == "right":
                ids = ids[::-1]
            line, g = _line([board[i] for i in ids])
            gain += g
            for i, v in zip(ids, line):
                b[i] = v
    elif action in ("up", "down"):
        for c in range(4):
            ids = tuple(4 * r + c for r in range(4))
            if action == "down":
                ids = ids[::-1]
            line, g = _line([board[i] for i in ids])
            gain += g
            for i, v in zip(ids, line):
                b[i] = v
    else:
        return tuple(board), 0, False
    result = tuple(b)
    return result, gain, result != tuple(board)


def _flat(obs):
    try:
        return tuple(int(v) for row in obs["board"] for v in row)
    except Exception:
        return (0,) * 16


def _orientation_score(board, orientation):
    order = ORDERS[orientation]
    logs = [math.log2(v) if v else 0.0 for v in board]
    score = 0.0
    for a, b in zip(order, order[1:]):
        if board[a] and board[b]:
            d = logs[a] - logs[b]
            score += d if d >= 0 else 2.6 * d
    corner = board[CORNER[orientation]]
    if corner:
        score += 2.5 * logs[CORNER[orientation]]
    return score


def infer_orientation(board, current):
    """Keep a healthy occupied corner; otherwise choose the best snake."""
    mx = max(board) if board else 0
    if mx:
        for o in ORIENTATIONS:
            if board[CORNER[o]] == mx:
                # A corner containing the maximum is strong evidence, even if
                # a transient spawn makes another orientation look smoother.
                return o
    scores = sorted((_orientation_score(board, o), o) for o in ORIENTATIONS)
    best = scores[-1][1]
    if current in ORIENTATIONS and _orientation_score(board, current) >= scores[-1][0] - 2.0:
        return current
    return best


def heuristic(board, orientation, risk):
    empty = board.count(0)
    logs = [math.log2(v) if v else 0.0 for v in board]
    order = ORDERS[orientation]
    mono = 0.0
    for a, b in zip(order, order[1:]):
        if board[a] and board[b]:
            d = logs[a] - logs[b]
            mono += d if d >= 0 else 2.8 * d
    smooth = 0.0
    pairs = 0.0
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            if not board[i]:
                continue
            if c < 3 and board[i + 1]:
                smooth -= abs(logs[i] - logs[i + 1])
                pairs += board[i] if board[i] == board[i + 1] else 0
            if r < 3 and board[i + 4]:
                smooth -= abs(logs[i] - logs[i + 4])
                pairs += board[i] if board[i] == board[i + 4] else 0
    mx = max(board) if board else 0
    corner = board[CORNER[orientation]]
    # Risk is a small Astra-selected adjustment.  Empty cells dominate when
    # the board is crowded; merge potential matters more in a sparse opening.
    if risk == "space":
        ew, pw = 230.0, 12.0
    elif risk == "merge":
        ew, pw = 145.0, 35.0
    else:
        ew, pw = 190.0, 24.0
    return (ew * empty + 34.0 * mono + 5.0 * smooth + pw * pairs
            + 23.0 * math.log2(mx or 1) + 70.0 * math.log2(corner or 1)
            - (40.0 if not corner and mx >= 64 else 0.0))


class Search:
    def __init__(self, orientation, risk, depth, seconds=0.32):
        self.orientation = orientation
        self.risk = risk
        self.depth = max(1, min(3, int(depth)))
        self.end = time.monotonic() + seconds
        self.cache = {}

    def value(self, board, depth):
        if time.monotonic() >= self.end:
            return heuristic(board, self.orientation, self.risk)
        key = (board, depth, self.orientation, self.risk)
        if key in self.cache:
            return self.cache[key]
        if depth <= 0:
            ans = heuristic(board, self.orientation, self.risk)
            self.cache[key] = ans
            return ans
        choices = []
        for action in DIRS:
            moved, gain, changed = transition(board, action)
            if not changed:
                continue
            empties = [i for i, v in enumerate(moved) if not v]
            if not empties:
                score = gain + self.value(moved, depth - 1)
            else:
                total = 0.0
                for pos in empties:
                    for tile, prob in ((2, 0.9), (4, 0.1)):
                        child = list(moved)
                        child[pos] = tile
                        total += prob * self.value(tuple(child), depth - 1)
                        if time.monotonic() >= self.end:
                            break
                    if time.monotonic() >= self.end:
                        break
                score = gain + total / len(empties)
            choices.append(score)
            if time.monotonic() >= self.end:
                break
        ans = max(choices) if choices else heuristic(board, self.orientation, self.risk) - 10000.0
        self.cache[key] = ans
        return ans

    def choose(self, board, legal):
        ranked = []
        for action in legal:
            moved, gain, changed = transition(board, action)
            if not changed:
                continue
            empties = [i for i, v in enumerate(moved) if not v]
            if empties:
                total = 0.0
                for pos in empties:
                    for tile, prob in ((2, 0.9), (4, 0.1)):
                        child = list(moved)
                        child[pos] = tile
                        total += prob * self.value(tuple(child), self.depth - 1)
                        if time.monotonic() >= self.end:
                            break
                    if time.monotonic() >= self.end:
                        break
                value = gain + total / len(empties)
            else:
                value = gain + self.value(moved, self.depth - 1)
            ranked.append((value, action, gain))
            if time.monotonic() >= self.end:
                break
        if not ranked:
            return legal[0] if legal else None
        # Stable direction tie break avoids arbitrary oscillation.
        preference = {"left": 4, "down": 3, "right": 2, "up": 1}
        ranked.sort(key=lambda x: (x[0], preference.get(x[1], 0)), reverse=True)
        return ranked[0][1]


def _parse_reply(text):
    try:
        m = re.search(r"\{[^{}]*\}", text or "")
        data = json.loads(m.group()) if m else {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    orientation = str(data.get("orientation", data.get("anchor", ""))).lower().replace("_", "-")
    if orientation not in ORIENTATIONS:
        for o in ORIENTATIONS:
            if o in (text or "").lower().replace("_", "-"):
                orientation = o
                break
    risk = str(data.get("risk", data.get("mode", ""))).lower()
    if risk not in ("space", "merge", "balanced"):
        risk = "balanced"
    try:
        depth = int(data.get("depth", 2))
    except Exception:
        depth = 2
    return (orientation if orientation in ORIENTATIONS else None, risk, max(1, min(3, depth)))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.orientation = "bottom-left"
        self.risk = "balanced"
        self.depth = 2
        self.session = None
        self.reviews = 0
        self.last_review_move = -100
        self.last_illegal = 0

    def review(self, obs, reason):
        ctx = self.ctx
        rem = ctx.budget.remaining()
        if ctx.done or getattr(ctx, "system2", True) is None or rem.get("system2_calls", 0) <= 0:
            return
        if self.reviews >= 5 or rem.get("wall_s", 0) < 150:
            return
        self.reviews += 1
        self.last_review_move = int(obs.get("moves", 0) or 0)
        try:
            progress = ctx.progress()
        except Exception:
            progress = {}
        prompt = (
            "Supervise a local adaptive 2048 controller. Choose an orientation "
            "(top-left, top-right, bottom-left, bottom-right), risk (space, "
            "merge, balanced), and search depth 1-3. Keep the largest tile in "
            "a healthy corner, but move the anchor if the board is damaged. "
            "Reply only JSON: {\"orientation\":...,\"risk\":...,\"depth\":2}. "
            "Do not provide a seed-specific action list.\n" + json.dumps({
                "reason": reason, "observation": obs, "progress": progress,
                "current": {"orientation": self.orientation, "risk": self.risk, "depth": self.depth},
            }, separators=(",", ":"))
        )
        try:
            result = ctx.think(prompt, session_id=self.session, timeout_s=140.0)
            self.session = getattr(result, "session_id", self.session)
            orientation, risk, depth = _parse_reply(getattr(result, "text", ""))
            if orientation:
                self.orientation = orientation
            self.risk, self.depth = risk, depth
            ctx.log("adaptive_review", reason=reason, orientation=self.orientation,
                    risk=self.risk, depth=self.depth)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError, ValueError, TypeError):
            return

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        if not isinstance(obs, dict) or "board" not in obs:
            return
        self.review(obs, "opening orientation and risk")
        while not ctx.done and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                break
            legal = [a for a in obs.get("legal_actions", DIRS) if a in DIRS]
            board = _flat(obs)
            if not legal:
                break
            self.orientation = infer_orientation(board, self.orientation)
            # More depth is useful only when the board is crowded enough for
            # the chance tree to stay small; this avoids 0003's deep replay trap.
            empties = board.count(0)
            depth = self.depth if empties <= 8 else min(self.depth, 2)
            action = Search(self.orientation, self.risk, depth).choose(board, legal)
            if not action:
                break
            before = obs
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break
            except Exception:
                fallback = next((a for a in legal if a != action), legal[0])
                try:
                    nxt = ctx.act(fallback)
                    obs = nxt if isinstance(nxt, dict) else game.observe()
                    action = fallback
                except Exception:
                    break
            # Verify the local rule model and ask Astra when evidence disagrees.
            try:
                moved, gain, changed = transition(_flat(before), action)
                after = _flat(obs)
                diff = sum(1 for x, y in zip(moved, after) if x != y)
                valid = (changed and obs.get("moves", 0) == before.get("moves", 0) + 1
                         and obs.get("score", 0) - before.get("score", 0) == gain
                         and diff == 1)
                progress = ctx.progress()
            except Exception:
                valid, progress = False, {}
            reasons = []
            if not valid:
                reasons.append("post-action transition verification disagreed")
            if progress.get("illegal_actions", 0) > self.last_illegal:
                self.last_illegal = progress.get("illegal_actions", 0)
                reasons.append("an action was rejected")
            if progress.get("since_score_change", 0) >= 12 or progress.get("since_obs_change", 0) >= 3:
                reasons.append("progress detector reports stagnation")
            if reasons:
                self.review(obs, "; ".join(reasons))
            elif int(obs.get("moves", 0) or 0) - self.last_review_move >= 60:
                self.review(obs, "periodic anchor health check")


def build(ctx):
    return Harness(ctx)
