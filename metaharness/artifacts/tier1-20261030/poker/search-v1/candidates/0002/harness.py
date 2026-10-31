"""Board-aware, low-variance Hold'em policy.

The policy is deliberately code-owned: exact legal candidates are selected from
the observation, while hand strength is measured relative to the public board.
This avoids mistaking a pair on the board for a value hand.
"""
from __future__ import annotations

from itertools import combinations

RANKS = "23456789TJQKA"
RV = {r: i + 2 for i, r in enumerate(RANKS)}


def rank5(cards):
    """Best five-card rank as (category, tie breakers)."""
    best = (-1, ())
    for five in combinations(cards, 5):
        vals = sorted((RV.get(c[0], 0) for c in five), reverse=True)
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        groups = sorted(((n, v) for v, n in counts.items()), reverse=True)
        suits = [c[1] for c in five]
        uniq = sorted(set(vals), reverse=True)
        straight = 0
        if len(uniq) == 5:
            if uniq[0] - uniq[-1] == 4:
                straight = uniq[0]
            elif uniq == [14, 5, 4, 3, 2]:
                straight = 5
        flush = len(set(suits)) == 1
        if flush and straight:
            cur = (8, (straight,))
        elif groups[0][0] == 4:
            q = groups[0][1]
            cur = (7, (q, max(v for v in vals if v != q)))
        elif groups[0][0] == 3 and len(groups) > 1 and groups[1][0] == 2:
            cur = (6, (groups[0][1], groups[1][1]))
        elif flush:
            cur = (5, tuple(vals))
        elif straight:
            cur = (4, (straight,))
        elif groups[0][0] == 3:
            t = groups[0][1]
            cur = (3, (t,) + tuple(sorted((v for v in vals if v != t), reverse=True)))
        elif groups[0][0] == 2 and len(groups) > 1 and groups[1][0] == 2:
            ps = sorted((groups[0][1], groups[1][1]), reverse=True)
            cur = (2, tuple(ps) + (groups[2][1],))
        elif groups[0][0] == 2:
            p = groups[0][1]
            cur = (1, (p,) + tuple(sorted((v for v in vals if v != p), reverse=True)))
        else:
            cur = (0, tuple(vals))
        if cur > best:
            best = cur
    return best


def hole_q(hole):
    if len(hole) < 2:
        return 0.0
    a, b = sorted((RV.get(c[0], 0) for c in hole), reverse=True)
    suited = hole[0][1] == hole[1][1]
    if a == b:
        q = 0.62 + (a - 2) * 0.025
    elif a == 14:
        q = 0.54 + (b - 2) * 0.018
    elif a >= 11:
        q = 0.27 + (a - 10) * 0.055 + (b - 2) * 0.012
    else:
        q = 0.15 + max(0, 6 - (a - b)) * 0.025 + max(0, b - 5) * 0.008
    if suited:
        q += 0.055
    if a - b <= 1:
        q += 0.035
    return min(0.99, q)


def draw(hole, board):
    cards = list(hole) + list(board)
    rs = {RV.get(c[0], 0) for c in cards}
    windows = ([14, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10],
               [7, 8, 9, 10, 11], [8, 9, 10, 11, 12],
               [9, 10, 11, 12, 13], [10, 11, 12, 13, 14])
    sd = any(len(set(w) & rs) >= 4 for w in windows)
    suits = {}
    for c in cards:
        suits[c[1]] = suits.get(c[1], 0) + 1
    fd = max(suits.values(), default=0) >= 4
    return sd or fd


def cand(obs, kind, amount=None):
    cs = obs.get("candidates") or []
    if kind in ("fold", "check", "call"):
        for c in cs:
            if c.get("id") == kind or c.get("action") == kind:
                return c.get("id", kind)
    if kind == "raise_to":
        vals = [(int(c["amount"]), c.get("id", "raise_to %d" % c["amount"]))
                for c in cs if c.get("action") == "raise_to" and isinstance(c.get("amount"), int)]
        if vals:
            vals.sort()
            above = [x for x in vals if x[0] >= int(amount or vals[0][0])]
            return (above[0] if above else vals[-1])[1]
    legal = obs.get("legal") or {}
    for k in ("check", "call", "fold"):
        if legal.get("can_" + k):
            z = cand(obs, k)
            if z is not None:
                return z
    return cs[0].get("id") if cs else "wait"


def board_relative(hole, board):
    """Return category, whether pair/trips are supplied by the hole cards."""
    full = rank5(list(hole) + list(board)) if len(hole) + len(board) >= 5 else (0, ())
    if not board:
        return full[0], False
    counts = {}
    for c in list(hole) + list(board):
        r = RV.get(c[0], 0)
        counts[r] = counts.get(r, 0) + 1
    board_counts = {}
    for c in board:
        r = RV.get(c[0], 0)
        board_counts[r] = board_counts.get(r, 0) + 1
    hole_pair = any(n >= 2 and board_counts.get(r, 0) < 2 for r, n in counts.items())
    # A board pair alone is weak; two pair+ is strong only if a hole card helps.
    board_made = max(board_counts.values(), default=0) >= 2
    return full[0], (hole_pair and not (board_made and full[0] == 1))


def policy(o):
    hole, board = o.get("hole_cards") or [], o.get("board") or []
    legal, street = o.get("legal") or {}, o.get("street", "preflop")
    call = int(o.get("to_call") or 0)
    pot = max(1, int(o.get("pot") or 1))
    if street == "preflop":
        q = hole_q(hole)
        hero, button = o.get("hero_seat"), o.get("button_seat")
        if hero == button and call <= 50:
            return cand(o, "raise_to", 250 if q >= 0.20 else None) if q >= 0.20 else cand(o, "fold")
        if call == 0:
            return cand(o, "raise_to", 400) if q >= 0.61 else cand(o, "check")
        # BB versus the scripted 250 open: defend price-aware, but avoid dominated offsuit hands.
        if q >= 0.74 and legal.get("can_raise"):
            return cand(o, "raise_to", max(800, int(3.5 * max(250, o.get("street_bets", [0, 0])[button or 0]))))
        return cand(o, "call") if q >= 0.40 and call <= 700 else cand(o, "fold")

    cat, hole_help = board_relative(hole, board)
    has_draw = draw(hole, board)
    strong = cat >= 3 or (cat == 2 and hole_help) or (cat == 1 and hole_help)
    premium = cat >= 4 or (cat == 3 and hole_help)
    if call:
        odds = call / float(pot + call)
        if premium:
            return cand(o, "raise_to", max(2 * call, int(0.75 * (pot + call)))) if legal.get("can_raise") else cand(o, "call")
        if strong:
            # Do not stack off one-pair hands against the heuristic's large bets.
            return cand(o, "call") if odds <= (0.32 if len(board) < 5 else 0.28) else cand(o, "fold")
        if has_draw and odds <= 0.25:
            return cand(o, "call")
        return cand(o, "fold")
    if premium:
        return cand(o, "raise_to", max(100, int(0.75 * pot)))
    if strong:
        return cand(o, "raise_to", max(100, int(0.55 * pot)))
    # Semi-bluff only with a real draw and a modest pot; otherwise preserve showdown equity.
    if has_draw and len(board) < 5 and pot <= 1200:
        return cand(o, "raise_to", max(100, int(0.45 * pot)))
    return cand(o, "check")


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        obs = game.observe()
        while not self.ctx.done and not obs.get("done"):
            try:
                obs = self.ctx.act(policy(obs))
            except Exception:
                if self.ctx.done:
                    return
                try:
                    obs = game.observe()
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
