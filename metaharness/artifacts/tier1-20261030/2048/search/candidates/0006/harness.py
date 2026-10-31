"""Adaptive, spawn-agnostic 2048 controller.

The frequent decisions are exact local simulations.  Astra only changes the
tradeoff between preserving space and cashing in merges; every selected move
is checked against the returned observation and a short repeated-state loop is
broken by a different root action.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

from metaharness.core.contract import BudgetExceeded, ModelUnavailable

DIRS = ("up", "down", "left", "right")
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _slide(xs):
    vals = [int(x) for x in xs if x]
    out, gain, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] * 2)
            gain += vals[i] * 2
            i += 2
        else:
            out.append(vals[i]); i += 1
    return tuple(out + [0] * (4 - len(out))), gain


@lru_cache(maxsize=100000)
def transition(board, action):
    b = list(board); gain = 0
    if action in ("left", "right"):
        for r in range(4):
            ids = tuple(4 * r + c for c in range(4))
            if action == "right": ids = ids[::-1]
            row, g = _slide([board[i] for i in ids]); gain += g
            for i, v in zip(ids, row): b[i] = v
    else:
        for c in range(4):
            ids = tuple(4 * r + c for r in range(4))
            if action == "down": ids = ids[::-1]
            col, g = _slide([board[i] for i in ids]); gain += g
            for i, v in zip(ids, col): b[i] = v
    out = tuple(b)
    return out, gain, out != board


def _board(obs):
    try:
        return tuple(int(v) for row in obs["board"] for v in row)
    except Exception:
        return (0,) * 16


def _put(board, pos, value):
    b = list(board); b[pos] = value; return tuple(b)


def _snake(board):
    patterns = (
        (15,14,13,12,8,9,10,11,7,6,5,4,0,1,2,3),
        (12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0),
        (3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15),
        (0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12),
    )
    best = -1e9
    for p in patterns:
        s = 0.0
        for a, b in zip(p, p[1:]):
            if board[a] and board[b]:
                d = math.log2(board[a]) - math.log2(board[b])
                s += d if d >= 0 else 2.2 * d
        if board[p[0]]: s += 3.0 * math.log2(board[p[0]])
        best = max(best, s)
    return best


def heuristic(board, profile):
    empty = board.count(0)
    mx = max(board) if board else 0
    logs = [math.log2(x) if x else 0.0 for x in board]
    smooth = 0.0; pairs = 0.0
    for r in range(4):
        for c in range(4):
            i = 4*r+c
            if not board[i]: continue
            if c < 3 and board[i+1]:
                smooth -= abs(logs[i]-logs[i+1]); pairs += board[i] == board[i+1]
            if r < 3 and board[i+4]:
                smooth -= abs(logs[i]-logs[i+4]); pairs += board[i] == board[i+4]
    if profile == "merge": ew, pw = 110.0, 42.0
    elif profile == "anchor": ew, pw = 170.0, 24.0
    else: ew, pw = 230.0, 26.0
    # Pair potential is deliberately visible even when the board has space:
    # this prevents endlessly sliding a full snake without harvesting merges.
    return (ew * empty + 28.0 * _snake(board) + 5.0 * smooth + pw * pairs
            + 24.0 * math.log2(mx or 1))


def _legal(board):
    return [a for a in DIRS if transition(board, a)[2]]


def _verify(before, after, action):
    """Check the observed effect of a changed move against local rules."""
    b, a = _board(before), _board(after)
    moved, gain, changed = transition(b, action)
    if not changed:
        return "local move was a no-op"
    if int(after.get("score", 0)) - int(before.get("score", 0)) != gain:
        return "merge score delta disagrees"
    if int(after.get("moves", 0)) != int(before.get("moves", 0)) + 1:
        return "move counter did not advance"
    diffs = [i for i in range(16) if moved[i] != a[i]]
    if len(diffs) != 1 or moved[diffs[0]] != 0 or a[diffs[0]] not in (2, 4):
        return "board is not slide plus one 2/4 spawn"
    return None


def _chance_value(board, depth, profile, cache):
    key = (board, depth, profile)
    if key in cache: return cache[key]
    if depth <= 0:
        ans = heuristic(board, profile); cache[key] = ans; return ans
    legal = _legal(board)
    if not legal:
        return heuristic(board, profile) - 20000.0
    best = -1e30
    for action in legal:
        moved, gain, _ = transition(board, action)
        empty = [i for i, v in enumerate(moved) if not v]
        if not empty:
            val = gain + _chance_value(moved, depth - 1, profile, cache)
        else:
            total = 0.0
            for pos in empty:
                total += .9 * _chance_value(_put(moved, pos, 2), depth-1, profile, cache)
                total += .1 * _chance_value(_put(moved, pos, 4), depth-1, profile, cache)
            val = gain + total / len(empty)
        # A move that leaves only one continuation is dangerous even if its
        # expectation is attractive; this is an explicit terminal-risk check.
        if len(_legal(moved)) <= 1: val -= 180.0
        best = max(best, val)
    cache[key] = best
    return best


def choose(board, legal, profile, previous, recent):
    if not legal: return None
    cache = {}; ranked = []
    for action in legal:
        moved, gain, changed = transition(board, action)
        if not changed: continue
        empty = [i for i, v in enumerate(moved) if not v]
        if empty:
            value = gain + sum(.9 * _chance_value(_put(moved,p,2),1,profile,cache)
                               + .1 * _chance_value(_put(moved,p,4),1,profile,cache)
                               for p in empty) / len(empty)
        else:
            value = gain + _chance_value(moved,1,profile,cache)
        if previous and action == REVERSE.get(previous): value -= 10.0
        if recent.count(action) >= 3: value -= 16.0
        ranked.append((value, action))
    return max(ranked)[1] if ranked else legal[0]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.profile = "balanced"
        self.session = None
        self.reviews = 0
        self.last_review_move = -100
        self.previous = None
        self.recent = []
        self.last_illegal = 0
        self.mismatches = 0

    def review(self, obs, reason):
        rem = self.ctx.budget.remaining()
        if self.ctx.done or rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 180:
            return
        if self.reviews >= 5: return
        self.reviews += 1; self.last_review_move = int(obs.get("moves", 0) or 0)
        prompt = (
            "Supervise a local 2048 controller. Choose one profile: balanced (space and "
            "snake), merge (harvest adjacent equal tiles), or anchor (protect the largest "
            "corner). The controller simulates exact slides and random spawns; you only "
            "change the priority. Return one JSON object {\"profile\":\"balanced\"}. "
            "Do not give a move sequence.\n" + json.dumps({"reason": reason, "observation": obs,
            "progress": self.ctx.progress(), "current": self.profile}, separators=(",", ":")))
        try:
            result = self.ctx.think(prompt, session_id=self.session, timeout_s=140.0)
            self.session = getattr(result, "session_id", self.session)
            text = getattr(result, "text", "") or ""
            m = re.search(r"\{[^{}]*\}", text)
            data = json.loads(m.group()) if m else {}
            if data.get("profile") in ("balanced", "merge", "anchor"):
                self.profile = data["profile"]
            else:
                for p in ("balanced", "merge", "anchor"):
                    if re.search(r"\b" + p + r"\b", text.lower()): self.profile = p; break
            self.ctx.log("policy_review", reason=reason, profile=self.profile)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError, ValueError, TypeError):
            self.ctx.log("policy_review_failed", reason=reason)

    def run_episode(self, game):
        ctx = self.ctx
        try: obs = ctx.observe()
        except Exception: obs = game.observe()
        if not isinstance(obs, dict): return
        self.review(obs, "opening")
        while not ctx.done and not getattr(game, "done", False):
            legal = [a for a in obs.get("legal_actions", DIRS) if a in DIRS]
            if not legal: break
            board = _board(obs)
            action = choose(board, legal, self.profile, self.previous, self.recent)
            if not action: break
            before = obs
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else ctx.observe()
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break
            except Exception:
                # A stale legal-actions field is recoverable; try another legal
                # direction while preserving the episode score.
                alternatives = [a for a in legal if a != action]
                if not alternatives: break
                try: obs = ctx.act(alternatives[0])
                except Exception: break
                action = alternatives[0]
            self.previous = action
            self.recent.append(action); self.recent = self.recent[-8:]
            try: prog = ctx.progress()
            except Exception: prog = {}
            mismatch = _verify(before, obs, action)
            if mismatch:
                self.mismatches += 1
                self.review(obs, "post-action verification: " + mismatch)
                ctx.log("effect_mismatch", action=action, reason=mismatch,
                        count=self.mismatches)
            if prog.get("illegal_actions", 0) > self.last_illegal:
                self.last_illegal = prog.get("illegal_actions", 0); self.review(obs, "action rejection")
            if prog.get("since_score_change", 0) >= 14 or prog.get("since_obs_change", 0) >= 2:
                self.review(obs, "stagnation or unchanged observation")
            elif int(obs.get("moves", 0) or 0) - self.last_review_move >= 55:
                self.review(obs, "periodic strategy check")


def build(ctx):
    return Harness(ctx)
