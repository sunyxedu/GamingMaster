"""Stable-anchor, spawn-agnostic 2048 controller.

The local controller performs exact short expectimax and keeps a chosen corner
stable across a move sequence.  Astra is only a strategic supervisor: it may
select the corner or ask for a recovery mode when progress evidence says that
the current stack is damaged.
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache

from metaharness.core.contract import BudgetExceeded, ModelUnavailable

DIRS = ("up", "down", "left", "right")
ANCHORS = ("top-left", "top-right", "bottom-left", "bottom-right")
ORDERS = {
    "top-left": (0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12),
    "top-right": (3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15),
    "bottom-left": (12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0),
    "bottom-right": (15,14,13,12,8,9,10,11,7,6,5,4,0,1,2,3),
}
CORNER = {"top-left": 0, "top-right": 3, "bottom-left": 12, "bottom-right": 15}


def _line(values):
    vals = [int(v) for v in values if v]
    out, gain, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] * 2); gain += vals[i] * 2; i += 2
        else:
            out.append(vals[i]); i += 1
    return tuple(out + [0] * (4 - len(out))), gain


@lru_cache(maxsize=100000)
def transition(board, action):
    out, gain = list(board), 0
    if action in ("left", "right"):
        for r in range(4):
            ids = tuple(4*r+c for c in range(4))
            if action == "right": ids = ids[::-1]
            vals, g = _line([board[i] for i in ids]); gain += g
            for i, v in zip(ids, vals): out[i] = v
    else:
        for c in range(4):
            ids = tuple(4*r+c for r in range(4))
            if action == "down": ids = ids[::-1]
            vals, g = _line([board[i] for i in ids]); gain += g
            for i, v in zip(ids, vals): out[i] = v
    result = tuple(out)
    return result, gain, result != board


def flat(obs):
    try:
        return tuple(int(v) for row in obs["board"] for v in row)
    except Exception:
        return (0,) * 16


def infer_anchor(board):
    """Pick the strongest corner, with a deterministic tie break."""
    best = None; best_score = -10**9
    logs = [math.log2(v) if v else 0.0 for v in board]
    for name in ANCHORS:
        order = ORDERS[name]
        score = 0.0
        for a, b in zip(order, order[1:]):
            if board[a] and board[b]:
                d = logs[a] - logs[b]
                score += d if d >= 0 else 2.7 * d
        score += 4.0 * logs[CORNER[name]] if board[CORNER[name]] else -4.0
        if score > best_score:
            best_score, best = score, name
    return best or "bottom-left"


def heuristic(board, anchor, mode):
    empty = board.count(0); logs = [math.log2(v) if v else 0.0 for v in board]
    order = ORDERS[anchor]; mono = 0.0
    for a, b in zip(order, order[1:]):
        if board[a] and board[b]:
            d = logs[a] - logs[b]; mono += d if d >= 0 else 3.0*d
    smooth = pairs = 0.0
    for r in range(4):
        for c in range(4):
            i = 4*r+c
            if not board[i]: continue
            if c < 3 and board[i+1]:
                smooth -= abs(logs[i]-logs[i+1]); pairs += board[i] == board[i+1]
            if r < 3 and board[i+4]:
                smooth -= abs(logs[i]-logs[i+4]); pairs += board[i] == board[i+4]
    if mode == "space": ew, mw, pw = 220.0, 20.0, 20.0
    elif mode == "merge": ew, mw, pw = 125.0, 38.0, 52.0
    else: ew, mw, pw = 175.0, 30.0, 34.0
    corner = logs[CORNER[anchor]] if board[CORNER[anchor]] else -2.0
    return ew*empty + mw*mono + 5.0*smooth + pw*pairs + 20.0*math.log2(max(board) or 1) + 75.0*corner


def chance_positions(board):
    empty = [i for i, v in enumerate(board) if not v]
    if len(empty) <= 8: return empty
    # Stratification preserves board-wide uncertainty while bounding CPU work.
    return [empty[(k*(len(empty)-1))//7] for k in range(8)]


class Search:
    def __init__(self, anchor, mode, depth, seconds):
        self.anchor, self.mode = anchor, mode
        self.depth = max(1, min(4, int(depth)))
        self.stop = time.monotonic() + max(0.03, min(1.25, seconds))
        self.cache = {}

    def value(self, board, depth):
        if time.monotonic() >= self.stop or depth <= 0:
            return heuristic(board, self.anchor, self.mode)
        key = (board, depth, self.anchor, self.mode)
        if key in self.cache: return self.cache[key]
        vals = []
        for action in DIRS:
            moved, gain, changed = transition(board, action)
            if not changed: continue
            pos = chance_positions(moved)
            if not pos: vals.append(gain + self.value(moved, depth-1)); continue
            total = 0.0
            for p in pos:
                for tile, prob in ((2, .9), (4, .1)):
                    child = list(moved); child[p] = tile
                    total += prob * self.value(tuple(child), depth-1)
            vals.append(gain + total/len(pos))
        ans = max(vals) if vals else heuristic(board, self.anchor, self.mode) - 10000.0
        self.cache[key] = ans; return ans

    def choose(self, board, legal):
        ranked = []
        for action in legal:
            moved, gain, changed = transition(board, action)
            if not changed: continue
            pos = chance_positions(moved)
            if not pos: score = gain + self.value(moved, self.depth-1)
            else:
                total = 0.0
                for p in pos:
                    for tile, prob in ((2, .9), (4, .1)):
                        child = list(moved); child[p] = tile
                        total += prob * self.value(tuple(child), self.depth-1)
                score = gain + total/len(pos)
            # Penalize immediately abandoning a populated anchor corner.
            if board[CORNER[self.anchor]] >= 64 and moved[CORNER[self.anchor]] == 0:
                score -= 180.0 * math.log2(board[CORNER[self.anchor]])
            ranked.append((score, action))
        return max(ranked)[1] if ranked else (legal[0] if legal else None)


def parse_reply(text):
    data = {}
    try:
        m = re.search(r"\{[^{}]*\}", text or "")
        if m: data = json.loads(m.group())
    except Exception: data = {}
    if not isinstance(data, dict): data = {}
    anchor = data.get("anchor") if data.get("anchor") in ANCHORS else None
    if not anchor:
        s = (text or "").lower().replace("_", "-")
        anchor = next((a for a in ANCHORS if a in s), None)
    mode = data.get("mode") if data.get("mode") in ("balanced", "space", "merge") else None
    depth = data.get("depth") if data.get("depth") in (2, 3, 4) else None
    return anchor, mode, depth


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx; self.anchor = "bottom-left"; self.mode = "balanced"; self.depth = 3
        self.session = None; self.calls = 0; self.last_review = -100; self.last_illegal = 0

    def review(self, obs, reason):
        rem = self.ctx.budget.remaining()
        if self.ctx.done or rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 180 or self.calls >= 6:
            return
        self.calls += 1; self.last_review = int(obs.get("moves", 0) or 0)
        try: progress = self.ctx.progress()
        except Exception: progress = {}
        prompt = ("Supervise a local 2048 controller. Keep one stable corner anchor unless the largest tile "
                   "is stranded; choose anchor top-left/top-right/bottom-left/bottom-right, mode balanced/space/merge, "
                   "and depth 2/3/4. Local exact expectimax executes one legal move and checks its effect. "
                   "Reply with one short JSON object only.\n" + json.dumps({"reason": reason, "observation": obs,
                   "progress": progress, "current": {"anchor": self.anchor, "mode": self.mode, "depth": self.depth}}, separators=(",", ":")))
        try:
            res = self.ctx.think(prompt, session_id=self.session, timeout_s=150.0)
            self.session = getattr(res, "session_id", self.session)
            a, m, d = parse_reply(getattr(res, "text", "") or "")
            if a: self.anchor = a
            if m: self.mode = m
            if d: self.depth = int(d)
            self.ctx.log("anchor_review", reason=reason, anchor=self.anchor, mode=self.mode, depth=self.depth)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError, TypeError, ValueError):
            return

    def run_episode(self, game):
        ctx = self.ctx
        try: obs = game.observe()
        except Exception: return
        if not isinstance(obs, dict) or "board" not in obs: return
        self.anchor = infer_anchor(flat(obs))
        self.review(obs, "opening: select stable corner and risk mode")
        rejected = set()
        while not ctx.done and not obs.get("game_over", False):
            board = flat(obs)
            legal = [a for a in obs.get("legal_actions", DIRS) if a in DIRS and a not in rejected]
            local = [a for a in legal if transition(board, a)[2]]
            if not local: break
            left = ctx.budget.remaining().get("wall_s", 0)
            action = Search(self.anchor, self.mode, self.depth, min(1.0, max(.05, left*.004))).choose(board, local)
            before = obs
            try:
                nxt = ctx.act(action); obs = nxt if isinstance(nxt, dict) else game.observe(); rejected.clear()
            except Exception as exc:
                if ctx.done: break
                rejected.add(action)
                self.review(obs, "action rejected: " + type(exc).__name__)
                if len(rejected) >= len(legal): break
                continue
            try: progress = ctx.progress()
            except Exception: progress = {}
            moved, gain, changed = transition(flat(before), action)
            after = flat(obs)
            diffs = [i for i, (x, y) in enumerate(zip(moved, after)) if x != y]
            valid = changed and len(diffs) == 1 and moved[diffs[0]] == 0 and after[diffs[0]] in (2, 4)
            valid = valid and obs.get("moves", 0) == before.get("moves", 0)+1 and obs.get("score", 0)-before.get("score", 0) == gain
            if not valid: self.review(obs, "post-action transition disagreed with observation")
            reasons = []
            if progress.get("illegal_actions", 0) > self.last_illegal: reasons.append("rejected action")
            self.last_illegal = progress.get("illegal_actions", 0)
            if progress.get("since_score_change", 0) >= 12: reasons.append("12 actions without score progress")
            if progress.get("since_obs_change", 0) >= 1: reasons.append("unchanged observation")
            if after.count(0) <= 2: reasons.append("space collapsed")
            # Hysteresis: only change orientation on explicit review, never per leaf.
            if reasons and (not valid or len(reasons) > 1 or obs.get("moves", 0)-self.last_review >= 25):
                self.review(obs, "; ".join(reasons))


def build(ctx):
    return Harness(ctx)
