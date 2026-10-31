"""A compact, opponent-specific policy that buys back episode time.

The frozen opponent is price-sensitive preflop and only moderately aggressive
postflop. We use a position-aware entry rule and coarse made-hand/draw tests;
System-2 is reserved for recovery if a supposedly legal action is rejected.
"""
from __future__ import annotations

import json
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _strength(obs):
    """Return a coarse 0..8 made-hand class from visible cards only."""
    cards = list(obs.get("hole_cards", [])) + list(obs.get("board", []))
    ranks = "23456789TJQKA"
    counts = {}
    suits = {}
    for card in cards:
        if not isinstance(card, str) or len(card) < 2:
            continue
        counts[card[0]] = counts.get(card[0], 0) + 1
        suits[card[-1]] = suits.get(card[-1], 0) + 1
    vals = sorted((ranks.index(r), n) for r, n in counts.items() if r in ranks)
    groups = sorted((n for _, n in vals), reverse=True)
    # Category ladder: high card, pair, two pair, trips, straight, flush,
    # full house, quads. A wheel is recognized explicitly.
    unique = {ranks.index(r) for r in counts if r in ranks}
    straight = any(all(x + i in unique for i in range(5)) for x in range(9))
    straight = straight or {12, 0, 1, 2, 3}.issubset(unique)
    flush = max(suits.values(), default=0) >= 5
    if groups and groups[0] >= 4:
        return 7
    if groups and groups[0] == 3 and len(groups) > 1 and groups[1] >= 2:
        return 6
    if flush:
        return 5
    if straight:
        return 4
    if groups and groups[0] == 3:
        return 3
    pairs = sum(n == 2 for n in groups)
    if pairs >= 2:
        return 2
    if pairs:
        return 1
    return 0


def _draws(obs):
    cards = list(obs.get("hole_cards", [])) + list(obs.get("board", []))
    if len(obs.get("board", [])) not in (3, 4):
        return False
    suits = {}
    ranks = set()
    order = "23456789TJQKA"
    for card in cards:
        if isinstance(card, str) and len(card) >= 2:
            suits[card[-1]] = suits.get(card[-1], 0) + 1
            if card[0] in order:
                ranks.add(order.index(card[0]))
    flush_draw = max(suits.values(), default=0) == 4
    straight_draw = any(sum(x + i in ranks for i in range(5)) >= 4 for x in range(9))
    return flush_draw or straight_draw


def _fallback(obs):
    legal = obs.get("legal", {})
    candidates = obs.get("candidates", [])
    ids = [c.get("id") for c in candidates if isinstance(c, dict)]
    if not ids:
        return None
    can_check = legal.get("can_check") is True
    to_call = obs.get("to_call", 0)
    street = obs.get("street")
    hole = obs.get("hole_cards", [])
    rank_order = "23456789TJQKA"
    ranks = [rank_order.index(c[0]) for c in hole if isinstance(c, str) and c[0] in rank_order]
    suited = len(hole) == 2 and len({c[-1] for c in hole if isinstance(c, str)}) == 1
    pair = len(ranks) == 2 and ranks[0] == ranks[1]
    high = max(ranks, default=0)
    low = min(ranks, default=0)
    premium = pair and high >= 9 or high == 12 and low >= 10
    playable = pair or high >= 9 or (suited and high >= 6) or high + low >= 19
    strength = _strength(obs)
    draw = _draws(obs)
    if street == "preflop":
        if obs.get("hero_seat") == obs.get("button_seat"):
            if to_call <= 50:
                if premium or playable:
                    preferred = "raise_to 250"
                    return preferred if preferred in ids else next((x for x in ids if x.startswith("raise_to ")), "call")
                return "fold" if "fold" in ids else "call"
            return "call" if playable and to_call <= 400 else ("fold" if "fold" in ids else "call")
        if can_check:
            return "check"
        return "call" if playable or to_call <= 150 else ("fold" if "fold" in ids else "call")
    if can_check:
        # Value/protection bet only with a made pair or a draw; prefer small size.
        if strength >= 1 or draw:
            bets = [x for x in ids if x.startswith("raise_to ")]
            if bets:
                pot = max(1, int(obs.get("pot", 1)))
                amounts = []
                for x in bets:
                    m = re.search(r"(\d+)$", x)
                    if m:
                        amounts.append((abs(int(m.group(1)) - min(600, max(100, pot // 2))), x))
                if amounts:
                    return min(amounts)[1]
        return "check"
    # Calling threshold prices in the current pot, with a small allowance for
    # flush/open-ended draws. Folds to sustained multi-street pressure.
    pot = max(1, int(obs.get("pot", 1)))
    price = float(to_call) / float(pot + to_call)
    threshold = {"flop": .29, "turn": .30, "river": .34}.get(street, .32)
    if strength >= 2 or (strength >= 1 and price <= threshold + .10) or (draw and price <= .23):
        return "call" if "call" in ids else "check"
    return "fold" if "fold" in ids else ("call" if "call" in ids else "check")


def _valid_action(action, obs):
    ids = {c.get("id") for c in obs.get("candidates", []) if isinstance(c, dict)}
    return action in ids


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        last_progress = ctx.progress()
        while not ctx.done:
            action = _fallback(obs)
            if not _valid_action(action, obs):
                action = "check" if obs.get("legal", {}).get("can_check") else "call"
            before = obs
            try:
                obs = ctx.act(action)
            except Exception as exc:
                # Detect the broken scripted assumption and ask Astra once for
                # this exact state; garbage responses are sanitized locally.
                ctx.log("script_action_rejected", action=action, error=type(exc).__name__)
                if ctx.done:
                    return
                try:
                    prompt = ("Choose one legal candidate id from this observation. "
                              "Reply with that id only.\n" + json.dumps(before, separators=(",", ":")))
                    result = ctx.think(prompt, session_id=None, timeout_s=180.0)
                    text = str(result.text or "")
                    action = next((i for i in (c.get("id") for c in before.get("candidates", []) if isinstance(c, dict))
                                   if i and i in text), None)
                    if not action:
                        action = _fallback(before)
                    if _valid_action(action, before):
                        obs = ctx.act(action)
                    else:
                        return
                except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                    return
            progress = ctx.progress()
            # The observation and legality are postconditions: if neither state
            # nor score advanced, stop instead of looping on a rejected action.
            if progress.get("illegal_actions", 0) > last_progress.get("illegal_actions", 0):
                ctx.log("illegal_action_seen", progress=progress)
                return
            last_progress = progress


def build(ctx):
    return Harness(ctx)
