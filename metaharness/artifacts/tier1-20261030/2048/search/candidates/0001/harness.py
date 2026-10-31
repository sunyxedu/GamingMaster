"""Adaptive 2048 harness: local expectimax does the frequent decisions.

The model is used for the strategic choice that is expensive to encode (the
corner anchor) and for an occasional recovery diagnosis.  Exact board rules
and a chance-node search remain local, so a slow Astra call does not consume a
turn every time.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


DIRS = ("up", "down", "left", "right")
ANCHORS = ("top-left", "top-right", "bottom-left", "bottom-right")
ANCHOR_INDEX = {
    "top-left": 0, "top-right": 3, "bottom-left": 12, "bottom-right": 15,
}


def _slide(line):
    vals = [x for x in line if x]
    out = []
    gain = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            v = vals[i] * 2
            out.append(v)
            gain += v
            i += 2
        else:
            out.append(vals[i])
            i += 1
    return out + [0] * (len(line) - len(out)), gain


def transition(board, action):
    """Return (board-after-slide, merge-score, changed), before the spawn."""
    b = [list(r) for r in board]
    gain = 0
    if action in ("left", "right"):
        for r in range(4):
            line = b[r][:]
            if action == "right":
                line.reverse()
            line, g = _slide(line)
            if action == "right":
                line.reverse()
            b[r], gain = line, gain + g
    else:
        for c in range(4):
            line = [b[r][c] for r in range(4)]
            if action == "down":
                line.reverse()
            line, g = _slide(line)
            if action == "down":
                line.reverse()
            for r in range(4):
                b[r][c] = line[r]
            gain += g
    return b, gain, b != board


def _order(anchor):
    if anchor == "bottom-left":
        return [12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0]
    if anchor == "bottom-right":
        return [15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3]
    if anchor == "top-right":
        return [3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15]
    return [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]


def _flat(board):
    return tuple(x for row in board for x in row)


def heuristic(board, anchor):
    flat = _flat(board)
    empties = flat.count(0)
    logs = [math.log2(v) if v else 0.0 for v in flat]
    order = _order(anchor)
    mono = 0.0
    for a, b in zip(order, order[1:]):
        if flat[a] and flat[b]:
            d = logs[a] - logs[b]
            mono += d if d >= 0 else 2.5 * d
    smooth = 0.0
    for r in range(4):
        for c in range(4):
            if flat[r * 4 + c]:
                if c < 3 and flat[r * 4 + c + 1]:
                    smooth -= abs(logs[r * 4 + c] - logs[r * 4 + c + 1])
                if r < 3 and flat[(r + 1) * 4 + c]:
                    smooth -= abs(logs[r * 4 + c] - logs[(r + 1) * 4 + c])
    merges = 0
    for r in range(4):
        for c in range(4):
            v = flat[r * 4 + c]
            if not v:
                continue
            if c < 3 and v == flat[r * 4 + c + 1]:
                merges += v
            if r < 3 and v == flat[(r + 1) * 4 + c]:
                merges += v
    mx = max(flat) if flat else 0
    corner = flat[ANCHOR_INDEX[anchor]]
    corner_bonus = math.log2(corner) if corner else -2.0
    return (empties * 260.0 + mono * 28.0 + smooth * 5.0 +
            merges * 18.0 + math.log2(mx or 1) * 18.0 + corner_bonus * 75.0)


def _put(board, pos, value):
    b = [list(r) for r in board]
    b[pos // 4][pos % 4] = value
    return b


def expectimax(board, depth, anchor, cache):
    key = (_flat(board), depth, anchor)
    if key in cache:
        return cache[key]
    if depth <= 0:
        ans = heuristic(board, anchor)
        cache[key] = ans
        return ans
    legal = []
    for action in DIRS:
        nb, gain, changed = transition(board, action)
        if changed:
            legal.append((nb, gain))
    if not legal:
        return heuristic(board, anchor) - 10000.0
    best = -float("inf")
    for nb, gain in legal:
        empty = [i for i, v in enumerate(_flat(nb)) if not v]
        if not empty:
            val = gain + expectimax(nb, depth - 1, anchor, cache)
        else:
            total = 0.0
            for pos in empty:
                total += 0.9 * expectimax(_put(nb, pos, 2), depth - 1, anchor, cache)
                total += 0.1 * expectimax(_put(nb, pos, 4), depth - 1, anchor, cache)
            val = gain + total / len(empty)
        if val > best:
            best = val
    cache[key] = best
    return best


def choose(board, legal, anchor):
    if not legal:
        return None
    cache = {}
    ranked = []
    for action in legal:
        nb, gain, changed = transition(board, action)
        if not changed:
            continue
        empty = [i for i, v in enumerate(_flat(nb)) if not v]
        if empty:
            val = gain + sum(
                0.9 * expectimax(_put(nb, p, 2), 1, anchor, cache) +
                0.1 * expectimax(_put(nb, p, 4), 1, anchor, cache)
                for p in empty) / len(empty)
        else:
            val = gain + expectimax(nb, 1, anchor, cache)
        ranked.append((val, action))
    return max(ranked)[1] if ranked else legal[0]


def _anchor_from_text(text):
    s = (text or "").lower().replace("_", "-")
    for a in ANCHORS:
        if a in s:
            return a
    if "left" in s and "bottom" in s:
        return "bottom-left"
    if "right" in s and "bottom" in s:
        return "bottom-right"
    if "left" in s:
        return "top-left"
    if "right" in s:
        return "top-right"
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.anchor = "bottom-left"
        self.replanned = False

    def _ask_anchor(self, obs, reason):
        if getattr(self.ctx, "system2", True) is None:
            return
        prompt = (
            "You are the strategic supervisor for a deterministic 2048 solver. "
            "Choose one corner anchor for a monotone snake and return only one of "
            "top-left, top-right, bottom-left, bottom-right. Preserve the largest "
            "tile in that corner; local expectimax executes moves.\n"
            f"Reason for review: {reason}\nState JSON: {json.dumps(obs, separators=(',', ':'))}\n"
            f"Current anchor: {self.anchor}\n"
            "Return exactly the corner name."
        )
        try:
            res = self.ctx.think(prompt, session_id=self.session, timeout_s=130.0)
            self.session = getattr(res, "session_id", self.session)
            picked = _anchor_from_text(getattr(res, "text", ""))
            if picked:
                self.anchor = picked
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
            return

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        self._ask_anchor(obs, "opening plan")
        stale = 0
        while not ctx.done and not getattr(game, "done", False):
            try:
                legal = list(obs.get("legal_actions", ())) if isinstance(obs, dict) else list(DIRS)
                legal = [a for a in legal if a in DIRS]
                if not legal:
                    break
                board = obs.get("board") if isinstance(obs, dict) else None
                action = choose(board, legal, self.anchor) if board else legal[0]
                before = obs
                try:
                    nxt = ctx.act(action)
                except Exception:
                    # A stale legal-actions field should not kill an episode.
                    nxt = ctx.act(legal[0])
                obs = nxt if nxt is not None else game.observe()
                try:
                    prog = ctx.progress()
                    stale = int(prog.get("since_score_change", 0))
                except Exception:
                    stale = stale + 1 if isinstance(before, dict) and before == obs else 0
                # A full or stagnant board is exactly when a fixed anchor can fail.
                if not self.replanned and stale >= 12:
                    self.replanned = True
                    self._ask_anchor(obs, "12 actions without a score change; diagnose a damaged snake")
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break


def build(ctx):
    return Harness(ctx)
