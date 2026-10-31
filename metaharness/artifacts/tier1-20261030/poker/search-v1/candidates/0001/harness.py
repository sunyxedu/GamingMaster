"""Fast, legality-first heads-up policy.

The previous candidate spent essentially the whole episode asking Astra for one
action.  This policy keeps the repeatable parts of Hold'em in code: a wide
button opening range, price-aware blind defence, and a value-forward postflop
policy.  It deliberately does not guess action strings; every wager is chosen
from the adapter's current candidate list.
"""
from __future__ import annotations

from itertools import combinations


RANKS = "23456789TJQKA"
RV = {r: i + 2 for i, r in enumerate(RANKS)}


def _rank(cards):
    """Small exact seven-card evaluator, returning (category, kickers)."""
    best = (-1, ())
    for five in combinations(cards, 5):
        vals = sorted((RV.get(c[0], 0) for c in five), reverse=True)
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        groups = sorted(((n, v) for v, n in counts.items()), reverse=True)
        flush = len({c[1] for c in five}) == 1
        uniq = sorted(set(vals), reverse=True)
        straight = 0
        if len(uniq) == 5:
            if uniq[0] - uniq[4] == 4:
                straight = uniq[0]
            elif uniq == [14, 5, 4, 3, 2]:
                straight = 5
        if flush and straight:
            cur = (8, (straight,))
        elif groups[0][0] == 4:
            four = groups[0][1]
            cur = (7, (four, max(v for v in vals if v != four)))
        elif groups[0][0] == 3 and groups[1][0] == 2:
            cur = (6, (groups[0][1], groups[1][1]))
        elif flush:
            cur = (5, tuple(vals))
        elif straight:
            cur = (4, (straight,))
        elif groups[0][0] == 3:
            t = groups[0][1]
            cur = (3, (t,) + tuple(sorted((v for v in vals if v != t), reverse=True)))
        elif groups[0][0] == 2 and groups[1][0] == 2:
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


def _hole_quality(hole):
    if len(hole) < 2:
        return 0.0
    a, b = RV.get(hole[0][0], 0), RV.get(hole[1][0], 0)
    hi, lo = max(a, b), min(a, b)
    suited = hole[0][1] == hole[1][1]
    pair = hi == lo
    gap = hi - lo
    q = 0.0
    if pair:
        q = 0.62 + (hi - 2) * 0.025
    elif hi == 14:
        q = 0.54 + (lo - 2) * 0.018
    elif hi == 13:
        q = 0.42 + (lo - 2) * 0.016
    elif hi == 12:
        q = 0.34 + (lo - 2) * 0.014
    elif hi == 11:
        q = 0.28 + (lo - 2) * 0.012
    else:
        q = 0.16 + max(0, 6 - gap) * 0.025 + max(0, lo - 5) * 0.008
    if suited:
        q += 0.055
    if gap <= 1:
        q += 0.035
    elif gap == 2:
        q += 0.012
    return min(0.99, q)


def _draw_strength(hole, board):
    cards = list(hole) + list(board)
    ranks = {RV.get(c[0], 0) for c in cards}
    # Four-card straight windows, including the wheel.
    straight_draw = 0
    for seq in ([14, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                 [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10],
                 [7, 8, 9, 10, 11], [8, 9, 10, 11, 12],
                 [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]):
        n = len(set(seq) & ranks)
        if n >= 4:
            straight_draw = max(straight_draw, n)
    suits = {}
    for c in cards:
        suits[c[1]] = suits.get(c[1], 0) + 1
    flush_draw = max(suits.values(), default=0) >= 4
    return (straight_draw >= 4) or flush_draw


def _candidate(obs, kind, amount=None):
    cs = obs.get("candidates") or []
    if kind in ("fold", "check", "call"):
        for c in cs:
            if c.get("id") == kind or c.get("action") == kind:
                return c.get("id", kind)
    if kind == "raise_to":
        vals = []
        for c in cs:
            if c.get("action") == "raise_to" and isinstance(c.get("amount"), int):
                vals.append((c["amount"], c.get("id", "raise_to %d" % c["amount"])))
        if vals:
            vals.sort()
            target = amount if amount is not None else vals[0][0]
            # Never undersize a requested value bet when a larger legal menu item exists.
            above = [x for x in vals if x[0] >= target]
            return (above[0] if above else vals[-1])[1]
    return None


def _safe(obs, preferred):
    """Map a policy choice to a legal id, with no illegal-action accounting."""
    x = _candidate(obs, *preferred)
    if x is not None:
        return x
    legal = obs.get("legal") or {}
    if legal.get("can_check"):
        x = _candidate(obs, "check")
        if x is not None:
            return x
    if legal.get("can_call"):
        x = _candidate(obs, "call")
        if x is not None:
            return x
    if legal.get("can_fold"):
        x = _candidate(obs, "fold")
        if x is not None:
            return x
    cs = obs.get("candidates") or []
    return cs[0].get("id") if cs else "wait"


def _policy(obs):
    hole = obs.get("hole_cards") or []
    board = obs.get("board") or []
    street = obs.get("street", "preflop")
    legal = obs.get("legal") or {}
    to_call = int(obs.get("to_call") or 0)
    pot = max(1, int(obs.get("pot") or 1))
    history = obs.get("history") or []
    hero = obs.get("hero_seat")
    button = obs.get("button_seat")
    q = _hole_quality(hole)
    # Preflop: exploit the fixed opponent's wide button opens with linear value.
    if street == "preflop":
        if to_call == 0:
            # BB checks most hands, raising a coherent value range over a limp.
            if q >= 0.58:
                return _safe(obs, ("raise_to", 400))
            return _safe(obs, ("check",))
        if hero == button:  # button acts first and is facing only the blind
            if q >= 0.20:
                return _safe(obs, ("raise_to", 250))
            return _safe(obs, ("fold",))
        # Facing a wager. A premium 3-bet is especially profitable versus wide opens.
        if q >= 0.72:
            target = max(int(3.5 * max(250, int(obs.get("street_bets", [0, 0])[button or 0] or 250))), 800)
            return _safe(obs, ("raise_to", target))
        if q >= 0.34 and to_call <= max(600, pot // 2):
            return _safe(obs, ("call",))
        return _safe(obs, ("fold",))

    strength = _rank(hole + board)[0] if len(hole) + len(board) >= 5 else 0
    draw = _draw_strength(hole, board)
    # Value categories: two pair or better, and strong top-pair equivalents.
    value = strength >= 2 or (strength == 1 and RV.get(hole[0][0], 0) >= 11)
    if to_call:
        odds = to_call / float(pot + to_call)
        if strength >= 3:
            return _safe(obs, ("raise_to", max(2 * to_call, int(0.75 * (pot + to_call)))))
        if value:
            if odds <= 0.34:
                return _safe(obs, ("call",))
            # Against the script's value bets, raise only very strong made hands.
            return _safe(obs, ("call",))
        if draw and odds <= 0.30:
            return _safe(obs, ("call",))
        return _safe(obs, ("fold",))
    # Free streets: realize equity and charge the heuristic with made hands/draws.
    if strength >= 3:
        return _safe(obs, ("raise_to", max(100, int(0.85 * pot))))
    if value or (draw and len(board) <= 4):
        return _safe(obs, ("raise_to", max(100, int(0.60 * pot))))
    return _safe(obs, ("check",))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.done:
            return
        obs = game.observe()
        while not ctx.done:
            try:
                action = _policy(obs)
                obs = ctx.act(action)
            except Exception:
                # A legal candidate should never fail. If an adapter/backend closes
                # concurrently, end cleanly; do not turn normal episode closure into
                # a harness error.
                if ctx.done:
                    return
                try:
                    obs = game.observe()
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
