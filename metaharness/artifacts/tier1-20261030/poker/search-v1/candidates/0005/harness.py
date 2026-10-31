"""Low-variance exploitative Hold'em policy (stdlib plus metaharness only)."""
from __future__ import annotations

import hashlib
import itertools
import random

from metaharness.core.contract import BudgetExceeded

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(range(52))
RV = {r: i + 2 for i, r in enumerate(RANKS)}
COMBOS = tuple(itertools.combinations(DECK, 2))


def parse_card(s):
    return 4 * RANKS.index(s[0]) + SUITS.index(s[1])


def rank5(cards):
    vals = sorted((c // 4 + 2 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
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


def score(cards):
    return max(rank5(x) for x in itertools.combinations(cards, 5))


def seeded(*parts):
    raw = "|".join(map(str, parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))


def hole_quality(hole):
    if len(hole) < 2:
        return 0.0
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if a == b:
        q = 0.58 + 0.03 * (a - 2)
    elif a == 14:
        q = 0.43 + 0.022 * (b - 2)
    elif a >= 11:
        q = 0.25 + 0.05 * (a - 10) + 0.012 * (b - 2)
    else:
        q = 0.12 + 0.018 * max(0, 7 - (a - b)) + 0.01 * max(0, b - 5)
    if suited:
        q += 0.055
    if a - b <= 1:
        q += 0.035
    return min(0.99, q)


def choose(obs, kind, amount=None):
    """Return an exact candidate id, with a legal fallback."""
    cs = obs.get("candidates") or []
    if kind == "raise_to":
        vals = []
        for c in cs:
            if c.get("action") == "raise_to":
                try:
                    vals.append((int(c["amount"]), c.get("id", "raise_to %d" % int(c["amount"]))))
                except Exception:
                    pass
        if vals:
            vals.sort()
            target = int(amount if amount is not None else vals[0][0])
            return next((x[1] for x in vals if x[0] >= target), vals[-1][1])
    else:
        for c in cs:
            if c.get("id") == kind or c.get("action") == kind:
                return c.get("id", kind)
    legal = obs.get("legal") or {}
    for k in ("check", "call", "fold"):
        if legal.get("can_" + k):
            for c in cs:
                if c.get("id") == k or c.get("action") == k:
                    return c.get("id", k)
            return k
    return cs[0].get("id", "wait") if cs else "wait"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.eq_cache = {}
        self.last_hand = None

    def checkpoint(self):
        if self.ctx.done or self.ctx.budget.remaining().get("wall_s", 1.0) <= 0.35:
            raise BudgetExceeded("wall")

    def equity(self, hole, board):
        key = (tuple(hole), tuple(board))
        if key in self.eq_cache:
            return self.eq_cache[key]
        self.checkpoint()
        dead = set(hole) | set(board)
        available = [c for c in DECK if c not in dead]
        # More samples on early streets, where a decision has the most leverage.
        # Keep the cheap policy comfortably inside the preflight deadline;
        # repeated hands provide more independent averaging than huge per-turn
        # samples.  The fixed seed makes this estimate reproducible.
        samples = 110 if not board else 80 if len(board) == 3 else 55 if len(board) == 4 else 30
        rng = seeded("equity", key)
        win = 0.0
        for i in range(samples):
            if i % 16 == 0:
                self.checkpoint()
            draw = rng.sample(available, 2 + (5 - len(board)))
            opp = draw[:2]
            run = tuple(board) + tuple(draw[2:])
            a = score(tuple(hole) + run)
            b = score(tuple(opp) + run)
            win += 1.0 if a > b else 0.5 if a == b else 0.0
        out = win / samples
        self.eq_cache[key] = out
        return out

    def opponent_pressure(self, obs):
        """Estimate whether the current wager is the heuristic's value branch."""
        call = int(obs.get("to_call") or 0)
        pot = max(1, int(obs.get("pot") or 1))
        ratio = call / float(pot + call)
        street = obs.get("street", "river")
        return ratio, street

    def action(self, obs):
        hole = tuple(parse_card(x) for x in obs.get("hole_cards", []))
        board = tuple(parse_card(x) for x in obs.get("board", []))
        legal = obs.get("legal") or {}
        street = obs.get("street", "preflop")
        call = int(obs.get("to_call") or 0)
        pot = max(1, int(obs.get("pot") or 1))
        hero = obs.get("hero_seat")
        button = obs.get("button_seat")
        q = hole_quality(hole)

        if street == "preflop":
            # The frozen button opens wide to 250. Use a 2bb open, but fold the
            # weakest offsuit holdings to avoid entering dominated pots.
            if hero == button and call <= 50:
                return choose(obs, "raise_to", 250 if q >= 0.29 else None)
            if call == 0:
                return choose(obs, "raise_to", 400) if q >= 0.61 else choose(obs, "check")
            # Against a raise, premiums 3-bet; medium suited/high-card hands
            # realise equity cheaply, and weak hands surrender.
            if legal.get("can_raise") and q >= 0.82:
                return choose(obs, "raise_to", min(int(4.0 * max(250, call + 100)), int(legal.get("max_raise_to") or 20000)))
            price = call / float(pot + call)
            return choose(obs, "call") if q >= max(0.34, price + 0.075) else choose(obs, "fold")

        eq = self.equity(hole, board)
        ratio, current_street = self.opponent_pressure(obs)
        # Large heuristic wagers are mostly its high-equity branch. Require a
        # substantial margin above price and fold one-pair bluff catchers.
        if call:
            threshold = ratio + (0.065 if len(board) < 5 else 0.035)
            if ratio >= (0.55 if len(board) == 4 else 0.62 if len(board) == 5 else 0.48):
                threshold += 0.055
            if eq < threshold:
                return choose(obs, "fold")
            # Raise only a very strong hand; sizing is capped to reduce variance.
            if legal.get("can_raise") and eq >= (0.80 if len(board) < 5 else 0.76) and ratio < 0.5:
                target = int(max(obs.get("street_bets", [0, 0])) + 0.85 * (pot + call))
                return choose(obs, "raise_to", target)
            return choose(obs, "call")

        # Value bet callers with a modest, repeatable size. Never use a stack
        # shove as the default exploit; preserving stack utility lowers match
        # variance while the opponent still calls too often.
        if legal.get("can_raise"):
            if eq >= (0.63 if len(board) == 3 else 0.59 if len(board) == 4 else 0.56):
                frac = 0.62 if len(board) == 3 else 0.70 if len(board) == 4 else 0.78
                target = int(max(obs.get("street_bets", [0, 0])) + frac * pot)
                return choose(obs, "raise_to", target)
            # Selective semi-bluffs on early streets only; heuristic calls make
            # river bluffs unattractive.
            if len(board) in (3, 4) and eq >= 0.46 and q < 0.35:
                target = int(max(obs.get("street_bets", [0, 0])) + 0.45 * pot)
                return choose(obs, "raise_to", target)
        return choose(obs, "check")

    def run_episode(self, game):
        try:
            obs = self.ctx.observe()
            while not self.ctx.done and not obs.get("done"):
                self.checkpoint()
                act = self.action(obs)
                self.ctx.log("capped_policy", hand=obs.get("hand_id"), street=obs.get("street"), action=act)
                try:
                    obs = self.ctx.act(act)
                except Exception:
                    if self.ctx.done:
                        return
                    obs = self.ctx.observe()
                if obs is None:
                    obs = self.ctx.observe()
                # Do not retain state-specific equity across hands when a new
                # public deal begins; within-hand cache remains deterministic.
                if obs.get("hand_id") != self.last_hand:
                    self.last_hand = obs.get("hand_id")
                    self.eq_cache.clear()
        except (BudgetExceeded, TimeoutError):
            return
        except Exception:
            if not self.ctx.done:
                return


def build(ctx):
    return Harness(ctx)
