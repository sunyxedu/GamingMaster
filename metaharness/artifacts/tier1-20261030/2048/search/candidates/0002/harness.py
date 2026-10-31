"""Risk-aware multi-ply expectimax controller for 2048.

The controller does all arithmetic locally and asks Astra only to select a
heuristic regime or diagnose a genuine stall.  Spawn locations are treated as
chance nodes, so the policy does not memorize the training seeds or depend on
reconstructing the adaptor's random-number implementation.
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


DIRS = ("up", "down", "left", "right")
MODES = ("balanced", "space", "merge")


def _line(values):
    vals = [int(v) for v in values if v]
    out, gain, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] * 2)
            gain += vals[i] * 2
            i += 2
        else:
            out.append(vals[i])
            i += 1
    return tuple(out + [0] * (4 - len(out))), gain


@lru_cache(maxsize=65536)
def transition(board, action):
    """Exact slide result and merge gain, before the random spawn."""
    b = list(board)
    gain = 0
    if action in ("left", "right"):
        for r in range(4):
            ids = tuple(4 * r + c for c in range(4))
            if action == "right":
                ids = ids[::-1]
            vals, g = _line([board[i] for i in ids])
            gain += g
            for i, v in zip(ids, vals):
                b[i] = v
    else:
        for c in range(4):
            ids = tuple(4 * r + c for r in range(4))
            if action == "down":
                ids = ids[::-1]
            vals, g = _line([board[i] for i in ids])
            gain += g
            for i, v in zip(ids, vals):
                b[i] = v
    result = tuple(b)
    return result, gain, result != board


def _flat(obs):
    try:
        return tuple(int(x) for row in obs["board"] for x in row)
    except Exception:
        return (0,) * 16


def _snake_scores(board):
    """Return best monotone snake score and whether its anchor is occupied."""
    patterns = (
        (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12),
        (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15),
        (12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0),
        (15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3),
    )
    logs = [math.log2(v) if v else 0.0 for v in board]
    best, anchor = -1e9, 0
    for p in patterns:
        # Penalize descending violations heavily while rewarding a smooth rise.
        score = 0.0
        for a, b in zip(p, p[1:]):
            if board[a] and board[b]:
                d = logs[a] - logs[b]
                score += d if d >= 0 else 2.8 * d
        score += (logs[p[0]] * 3.0 if board[p[0]] else -3.0)
        if score > best:
            best, anchor = score, p[0]
    return best, anchor


def heuristic(board, mode):
    empty = board.count(0)
    mx = max(board) if board else 0
    snake, anchor = _snake_scores(board)
    logs = [math.log2(v) if v else 0.0 for v in board]
    smooth = 0.0
    pairs = 0.0
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            if not board[i]:
                continue
            if c < 3 and board[i + 1]:
                smooth -= abs(logs[i] - logs[i + 1])
                pairs += 1.0 if board[i] == board[i + 1] else 0.0
            if r < 3 and board[i + 4]:
                smooth -= abs(logs[i] - logs[i + 4])
                pairs += 1.0 if board[i] == board[i + 4] else 0.0
    # Space is most valuable early; merge potential matters more near a full board.
    if mode == "space":
        we, ws, wm, wp = 210.0, 30.0, 4.0, 26.0
    elif mode == "merge":
        we, ws, wm, wp = 125.0, 42.0, 8.0, 46.0
    else:
        we, ws, wm, wp = 175.0, 34.0, 6.0, 34.0
    return we * empty + ws * snake + wm * smooth + wp * pairs + 22.0 * math.log2(mx or 1) + 8.0 * math.log2(board[anchor] or 1)


def _spawn_positions(board):
    empty = [i for i, v in enumerate(board) if not v]
    if len(empty) <= 8:
        return empty
    # Deterministic stratified subset keeps the per-move calculation bounded.
    return [empty[(k * (len(empty) - 1)) // 7] for k in range(8)]


class Search:
    def __init__(self, mode, depth=3):
        self.mode = mode
        # Three plies is the deliberate latency/quality boundary.  A model
        # response can never turn a turn game into an unbounded CPU call.
        self.depth = max(1, min(3, int(depth)))
        self.cache = {}
        self.deadline = time.monotonic() + 1.0

    def value(self, board, depth):
        if time.monotonic() >= self.deadline:
            return heuristic(board, self.mode)
        key = (board, depth, self.mode)
        if key in self.cache:
            return self.cache[key]
        if depth <= 0:
            ans = heuristic(board, self.mode)
            self.cache[key] = ans
            return ans
        choices = []
        for action in DIRS:
            moved, gain, changed = transition(board, action)
            if not changed:
                continue
            empties = _spawn_positions(moved)
            if not empties:
                score = gain + self.value(moved, depth - 1)
            else:
                total = 0.0
                for pos in empties:
                    for tile, prob in ((2, 0.9), (4, 0.1)):
                        child = list(moved)
                        child[pos] = tile
                        total += prob * self.value(tuple(child), depth - 1)
                score = gain + total / len(empties)
            choices.append(score)
        ans = max(choices) if choices else heuristic(board, self.mode) - 10000.0
        self.cache[key] = ans
        return ans

    def choose(self, board, legal):
        ranked = []
        for action in legal:
            moved, gain, changed = transition(board, action)
            if not changed:
                continue
            empties = _spawn_positions(moved)
            if not empties:
                score = gain + self.value(moved, self.depth - 1)
            else:
                total = 0.0
                for pos in empties:
                    for tile, prob in ((2, 0.9), (4, 0.1)):
                        child = list(moved)
                        child[pos] = tile
                        total += prob * self.value(tuple(child), self.depth - 1)
                score = gain + total / len(empties)
            ranked.append((score, action))
        return (max(ranked)[1] if ranked else (legal[0] if legal else None)), ranked


def _mode_from_reply(text):
    s = (text or "").lower()
    for mode in MODES:
        if re.search(r"\b" + mode + r"\b", s):
            return mode
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "balanced"
        self.depth = 3
        self.session = None
        self.review_count = 0
        self.last_review_move = -100
        self.last_illegal = 0

    def review(self, obs, reason):
        ctx = self.ctx
        rem = ctx.budget.remaining()
        if ctx.done or getattr(ctx, "system2", True) is None or rem.get("system2_calls", 0) <= 0:
            return
        if self.review_count >= 6 or rem.get("wall_s", 0) < 180:
            return
        self.review_count += 1
        self.last_review_move = int(obs.get("moves", 0) or 0)
        try:
            progress = ctx.progress()
        except Exception:
            progress = {}
        prompt = (
            "You supervise a local 2048 risk-aware expectimax controller. "
            "Choose one heuristic mode: balanced, space (protect empty cells), "
            "or merge (favor immediate pair potential). Choose depth 2 or 3. "
            "Reply with one short JSON object such as {\"mode\":\"balanced\",\"depth\":3}. "
            "Do not suggest a seed-specific sequence.\n" + json.dumps({
                "reason": reason, "observation": obs, "progress": progress,
                "current": {"mode": self.mode, "depth": self.depth},
            }, separators=(",", ":"))
        )
        try:
            result = ctx.think(prompt, session_id=self.session, timeout_s=150.0)
            self.session = getattr(result, "session_id", self.session)
            text = getattr(result, "text", "") or ""
            match = re.search(r"\{[^{}]*\}", text)
            data = json.loads(match.group()) if match else {}
            if isinstance(data, dict):
                if data.get("mode") in MODES:
                    self.mode = data["mode"]
                if data.get("depth") in (2, 3):
                    self.depth = int(data["depth"])
            else:
                picked = _mode_from_reply(text)
                if picked:
                    self.mode = picked
            ctx.log("policy_review", reason=reason, mode=self.mode, depth=self.depth)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError, ValueError, TypeError):
            # A review is optional; the deterministic controller remains usable.
            return

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        if not isinstance(obs, dict) or "board" not in obs:
            return
        self.review(obs, "opening: select risk regime")
        while not ctx.done and not getattr(game, "done", False):
            legal = [a for a in obs.get("legal_actions", DIRS) if a in DIRS]
            if not legal:
                break
            board = _flat(obs)
            local = [a for a in legal if transition(board, a)[2]]
            if not local:
                break
            search = Search(self.mode, self.depth)
            action, ranked = search.choose(board, local)
            if not action:
                break
            before = obs
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break
            except Exception:
                # A stale legal-actions report must not terminate a healthy episode.
                fallback = next((a for a in local if a != action), local[0])
                try:
                    nxt = ctx.act(fallback)
                    obs = nxt if isinstance(nxt, dict) else game.observe()
                    action = fallback
                except Exception:
                    break
            try:
                progress = ctx.progress()
            except Exception:
                progress = {}
            # Verify exact score/move accounting after every changed action.
            moved, gain, changed = transition(_flat(before), action)
            after_board = _flat(obs)
            valid = (changed and obs.get("moves", 0) == before.get("moves", 0) + 1 and
                     obs.get("score", 0) - before.get("score", 0) == gain and
                     sum(1 for x, y in zip(moved, after_board) if x != y) == 1)
            if not valid:
                self.review(obs, "post-action verification disagreed with local transition")
            if progress.get("illegal_actions", 0) > self.last_illegal:
                self.last_illegal = progress.get("illegal_actions", 0)
                self.review(obs, "an action was rejected; refresh legal policy")
            if progress.get("since_score_change", 0) >= 10 or progress.get("since_obs_change", 0) >= 2:
                self.review(obs, "stagnation detected by ctx.progress()")
            if int(obs.get("moves", 0) or 0) - self.last_review_move >= 45:
                self.review(obs, "periodic regime check")


def build(ctx):
    return Harness(ctx)
