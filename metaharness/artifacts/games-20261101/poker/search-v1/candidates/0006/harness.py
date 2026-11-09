"""Probe-and-value Hold'em policy.

This harness keeps legality and arithmetic in code.  It uses a small deterministic
showdown sampler for calls and adds a separate exploit: after the frozen player
checks, a button-in-position blocker probe can win the dead pot without risking a
deep stack.  The probe is deliberately suppressed on later streets and against
large wagers.
"""
from __future__ import annotations

import hashlib
import itertools
import random

from metaharness.core.contract import BudgetExceeded

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RV = {r: i + 2 for i, r in enumerate(RANKS)}
DECK = tuple(range(52))


def card(s):
    return 4 * RANKS.index(s[0]) + SUITS.index(s[1])


def rank5(xs):
    vals = sorted((c // 4 + 2 for c in xs), reverse=True)
    suits = [c % 4 for c in xs]
    cnt = {v: vals.count(v) for v in set(vals)}
    groups = sorted(((n, v) for v, n in cnt.items()), reverse=True)
    uniq = sorted(set(vals), reverse=True)
    straight = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[-1] == 4:
            straight = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight = 5
    flush = len(set(suits)) == 1
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        q = groups[0][1]
        return (7, q, max(v for v in vals if v != q))
    if groups[0][0] == 3 and len(groups) > 1 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5,) + tuple(vals)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        t = groups[0][1]
        return (3, t) + tuple(v for v in vals if v != t)
    if groups[0][0] == 2 and len(groups) > 1 and groups[1][0] == 2:
        p = sorted((groups[0][1], groups[1][1]), reverse=True)
        return (2, p[0], p[1], next(v for v in vals if v not in p))
    if groups[0][0] == 2:
        p = groups[0][1]
        return (1, p) + tuple(v for v in vals if v != p)
    return (0,) + tuple(vals)


def score(xs):
    return max(rank5(z) for z in itertools.combinations(xs, 5))


def seeded(*parts):
    raw = "|".join(map(str, parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))


def hole_quality(hole):
    if len(hole) < 2:
        return 0.0
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if a == b:
        q = 0.60 + 0.027 * (a - 2)
    elif a == 14:
        q = 0.46 + 0.020 * (b - 2)
    elif a >= 11:
        q = 0.25 + 0.052 * (a - 10) + 0.012 * (b - 2)
    else:
        q = 0.13 + 0.022 * max(0, 7 - (a - b)) + 0.010 * max(0, b - 5)
    if suited:
        q += 0.055
    if a - b <= 1:
        q += 0.035
    return min(0.99, q)


def choose(o, kind, amount=None):
    """Return an exact current candidate id, with a legal fallback."""
    cs = o.get("candidates") or []
    if kind == "raise_to":
        vals = []
        for c in cs:
            if c.get("action") == "raise_to" and isinstance(c.get("amount"), int):
                vals.append((c["amount"], c.get("id", "raise_to %d" % c["amount"])))
        if vals:
            vals.sort()
            target = int(vals[0][0] if amount is None else amount)
            return next((ident for n, ident in vals if n >= target), vals[-1][1])
    else:
        for c in cs:
            if c.get("id") == kind or c.get("action") == kind:
                return c.get("id", kind)
    legal = o.get("legal") or {}
    for k in ("check", "call", "fold"):
        if legal.get("can_" + k):
            return choose(o, k)
    return cs[0].get("id", "wait") if cs else "wait"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.eq_cache = {}
        self.hand_id = None

    def checkpoint(self):
        if self.ctx.done or self.ctx.budget.remaining().get("wall_s", 1.0) <= 0.35:
            raise BudgetExceeded("wall")

    def equity(self, hole, board):
        key = (tuple(hole), tuple(board))
        if key in self.eq_cache:
            return self.eq_cache[key]
        self.checkpoint()
        dead = set(hole) | set(board)
        avail = [c for c in DECK if c not in dead]
        n = 120 if not board else 90 if len(board) == 3 else 65 if len(board) == 4 else 36
        rng = seeded("probe-equity", key)
        wins = 0.0
        for i in range(n):
            if i % 16 == 0:
                self.checkpoint()
            draw = rng.sample(avail, 2 + (5 - len(board)))
            opp = draw[:2]
            runout = tuple(board) + tuple(draw[2:])
            a, b = score(tuple(hole) + runout), score(tuple(opp) + runout)
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
        out = wins / n
        self.eq_cache[key] = out
        return out

    @staticmethod
    def draw_info(hole, board):
        cards = list(hole) + list(board)
        ranks = {c // 4 + 2 for c in cards}
        windows = ([14, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                   [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10],
                   [7, 8, 9, 10, 11], [8, 9, 10, 11, 12],
                   [9, 10, 11, 12, 13], [10, 11, 12, 13, 14])
        straight = any(len(set(w) & ranks) >= 4 for w in windows)
        suits = {}
        for c in cards:
            suits[c % 4] = suits.get(c % 4, 0) + 1
        flush = max(suits.values(), default=0) >= 4
        return straight or flush

    @staticmethod
    def hole_help(hole, board):
        if not board:
            return True
        bc = {}
        allc = {}
        for c in board:
            r = c // 4 + 2
            bc[r] = bc.get(r, 0) + 1
        for c in list(hole) + list(board):
            r = c // 4 + 2
            allc[r] = allc.get(r, 0) + 1
        # A pair made only by the board is not a value hand.
        return any(n >= 2 and bc.get(r, 0) < 2 for r, n in allc.items())

    def preflop(self, o, hole):
        q = hole_quality(hole)
        hero, button = o.get("hero_seat"), o.get("button_seat")
        call = int(o.get("to_call") or 0)
        pot = max(1, int(o.get("pot") or 1))
        legal = o.get("legal") or {}
        if hero == button and call <= 50:
            return choose(o, "raise_to", 250) if q >= 0.26 else choose(o, "fold")
        if call == 0:
            return choose(o, "raise_to", 400) if q >= 0.60 else choose(o, "check")
        price = call / float(pot + call)
        if legal.get("can_raise") and q >= 0.79:
            bets = o.get("street_bets") or [0, 0]
            target = max(800, int(3.5 * max(250, max(bets))))
            return choose(o, "raise_to", target)
        # Deep speculative hands need a real price edge; avoid dominated offsuit junk.
        return choose(o, "call") if q >= max(0.34, price + 0.055) and call <= 900 else choose(o, "fold")

    def action(self, o):
        hole = tuple(card(s) for s in o.get("hole_cards", []))
        board = tuple(card(s) for s in o.get("board", []))
        street = o.get("street", "preflop")
        legal = o.get("legal") or {}
        call = int(o.get("to_call") or 0)
        pot = max(1, int(o.get("pot") or 1))
        if street == "preflop":
            return self.preflop(o, hole)
        eq = self.equity(hole, board)
        nboard = len(board)
        if call:
            ratio = call / float(pot + call)
            margin = 0.045 if nboard == 5 else 0.075
            # A large wager by heuristic-v1 is its high-equity branch.
            if ratio > (0.58 if nboard == 3 else 0.52 if nboard == 4 else 0.48):
                margin += 0.06
            if eq < ratio + margin:
                return choose(o, "fold")
            if legal.get("can_raise") and eq >= (0.82 if nboard < 5 else 0.78) and ratio < 0.46:
                bets = o.get("street_bets") or [0, 0]
                target = int(max(bets) + 0.95 * (pot + call))
                return choose(o, "raise_to", target)
            return choose(o, "call")

        if not legal.get("can_raise"):
            return choose(o, "check")
        cat = score(tuple(hole) + board)[0] if len(hole) + len(board) >= 5 else 0
        helped = self.hole_help(hole, board)
        made = cat >= 2 or (cat == 1 and helped)
        draw = self.draw_info(hole, board)
        # Value first.  Modest fractions induce calls from the opponent's
        # price-sensitive branch without turning a thin edge into a stack-off.
        threshold = 0.59 if nboard == 3 else 0.60 if nboard == 4 else 0.57
        if (made and eq >= threshold) or eq >= threshold + 0.08:
            frac = 0.55 if nboard == 3 else 0.68 if nboard == 4 else 0.78
            bets = o.get("street_bets") or [0, 0]
            return choose(o, "raise_to", int(max(bets) + frac * pot))

        # New test: after an opponent check, probe in position with blockers or
        # a real draw.  The bet is small, early, and never a river bluff.
        hist = o.get("history") or []
        last = hist[-1] if hist else None
        checked_by_opp = bool(last and last.get("seat") != o.get("hero_seat")
                              and last.get("street") == street
                              and last.get("action") == "check")
        high_blocker = any(c // 4 + 2 >= 13 for c in hole)
        if checked_by_opp and nboard in (3, 4) and o.get("hero_seat") == o.get("button_seat"):
            if draw or (high_blocker and 0.30 <= eq <= 0.56):
                bets = o.get("street_bets") or [0, 0]
                return choose(o, "raise_to", int(max(bets) + 0.34 * pot))
        # Early semi-bluffs are less costly with a genuine draw and enough equity.
        if draw and nboard < 5 and eq >= 0.43 and pot <= 1600:
            bets = o.get("street_bets") or [0, 0]
            return choose(o, "raise_to", int(max(bets) + 0.40 * pot))
        return choose(o, "check")

    def run_episode(self, game):
        try:
            obs = self.ctx.observe()
            while not self.ctx.done and not obs.get("done"):
                self.checkpoint()
                hid = obs.get("hand_id")
                if hid != self.hand_id:
                    self.hand_id = hid
                    self.eq_cache.clear()
                act = self.action(obs)
                self.ctx.log("probe_policy", hand=hid, street=obs.get("street"), action=act)
                try:
                    obs = self.ctx.act(act)
                except Exception:
                    if self.ctx.done:
                        return
                    obs = self.ctx.observe()
                if obs is None:
                    obs = self.ctx.observe()
        except (BudgetExceeded, TimeoutError):
            return
        except Exception:
            if not self.ctx.done:
                return


def build(ctx):
    return Harness(ctx)
