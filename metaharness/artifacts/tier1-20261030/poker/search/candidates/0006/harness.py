"""Fast random-range counterpolicy with sparse Astra posture reviews.

The policy deliberately keeps the model out of the turn loop.  It estimates
showdown equity against legal unseen cards, chooses only supplied candidates,
and lets Astra change a few bounded posture parameters after enough public
evidence has accumulated.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(range(52))


def enc(card):
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def straight(mask):
    for hi in range(12, 3, -1):
        wanted = 31 << (hi - 4)
        if mask & wanted == wanted:
            return hi + 2
    return 5 if mask & 0x100F == 0x100F else 0


def rank7(cards):
    """Best five-card rank from a 5--7 card integer tuple."""
    cnt = [0] * 15
    sm = 0
    suitmask = [0] * 4
    for c in cards:
        r = c // 4 + 2
        cnt[r] += 1
        sm |= 1 << (r - 2)
        suitmask[c % 4] |= 1 << (r - 2)
    flush = next((x for x in suitmask if x.bit_count() >= 5), 0)
    if flush:
        s = straight(flush)
        if s:
            return (8, s)
    quads = [r for r in range(14, 1, -1) if cnt[r] >= 4]
    if quads:
        q = quads[0]
        return (7, q, max(r for r in range(14, 1, -1) if r != q and cnt[r]))
    trips = [r for r in range(14, 1, -1) if cnt[r] >= 3]
    pairs = [r for r in range(14, 1, -1) if cnt[r] >= 2]
    if trips and len(pairs) >= 2:
        t = trips[0]
        p = next(r for r in pairs if r != t)
        return (6, t, p)
    if flush:
        return (5,) + tuple(r + 2 for r in range(12, -1, -1) if flush & (1 << r))[:5]
    s = straight(sm)
    if s:
        return (4, s)
    if trips:
        t = trips[0]
        return (3, t) + tuple(r for r in range(14, 1, -1) if r != t and cnt[r])[:2]
    if len(pairs) >= 2:
        p, q = pairs[:2]
        k = next(r for r in range(14, 1, -1) if r not in (p, q) and cnt[r])
        return (2, p, q, k)
    if pairs:
        p = pairs[0]
        return (1, p) + tuple(r for r in range(14, 1, -1) if r != p and cnt[r])[:3]
    return (0,) + tuple(r for r in range(14, 1, -1) if cnt[r])[:5]


def outcome(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0


def candidate_ids(obs):
    return [x.get("id") for x in obs.get("candidates", []) if isinstance(x, dict) and x.get("id")]


def safe_action(obs, action=None):
    ids = candidate_ids(obs)
    if action in ids:
        return action
    if isinstance(action, dict):
        if action.get("action") == "raise_to":
            try:
                n = int(action.get("amount"))
                exact = "raise_to " + str(n)
                if exact in ids:
                    return exact
                legal = obs.get("legal", {})
                if legal.get("can_raise") and legal.get("min_raise_to") <= n <= legal.get("max_raise_to"):
                    # Every integer is legal even when it was omitted from the compact menu.
                    return exact
            except Exception:
                pass
        elif action.get("action") in ids:
            return action.get("action")
    legal = obs.get("legal", {})
    for x in ("check", "call", "fold"):
        if x in ids and ((x == "check" and legal.get("can_check")) or
                         (x == "call" and legal.get("can_call")) or
                         (x == "fold" and legal.get("can_fold"))):
            return x
    return ids[0] if ids else None


def nearest_raise(obs, amount):
    ids = [x for x in candidate_ids(obs) if x.startswith("raise_to ")]
    legal = obs.get("legal", {})
    if not ids or not legal.get("can_raise"):
        return None
    amount = max(int(legal.get("min_raise_to", amount)), min(int(legal.get("max_raise_to", amount)), int(amount)))
    vals = []
    for x in ids:
        try:
            vals.append((abs(int(x.split()[-1]) - amount), x))
        except Exception:
            continue
    return min(vals)[1] if vals else None


def preflop_quality(cards):
    if len(cards) != 2:
        return 0.0
    a, b = sorted((c // 4 for c in cards), reverse=True)
    suited = cards[0] % 4 == cards[1] % 4
    pair = a == b
    q = 0.18 + 0.028 * a + 0.016 * b
    if pair:
        q += 0.17 + 0.012 * a
    if a == 12:
        q += 0.07
    if suited:
        q += 0.045
    if a - b <= 2:
        q += 0.035
    return max(0.0, min(0.82, q))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "balanced"
        self.margin = 0.035
        self.value = 0.60
        self.bluff = 0.08
        self.last_hand = None
        self.last_hist = 0
        self.seen_result = None
        self.actions = {"fold": 0, "call": 0, "raise_to": 0, "check": 0}
        self.opp = {"fold": 0, "call": 0, "raise_to": 0, "check": 0}
        self.reviewed = set()
        self.failure = None
        self.reviews = 0
        self.last_review_step = -100
        self.eq_cache = {}
        self.rng = random.Random(917331)

    def equity(self, obs):
        hole = tuple(enc(x) for x in obs.get("hole_cards", []) if isinstance(x, str) and len(x) >= 2)
        board = tuple(enc(x) for x in obs.get("board", []) if isinstance(x, str) and len(x) >= 2)
        key = (hole, board)
        if key in self.eq_cache:
            return self.eq_cache[key]
        if not hole:
            return 0.0
        if not board:
            q = preflop_quality(hole)
            self.eq_cache[key] = q
            return q
        avail = [c for c in DECK if c not in hole and c not in board]
        seed = int.from_bytes(hashlib.blake2s(bytes(hole + board), digest_size=8).digest(), "big")
        rng = random.Random(seed)
        n = 72 if len(board) < 5 else 120
        total = 0.0
        for _ in range(n):
            opp = tuple(rng.sample(avail, 2))
            rem = [c for c in avail if c not in opp]
            run = tuple(board) + tuple(rng.sample(rem, 5 - len(board)))
            total += outcome(rank7(hole + run), rank7(opp + run))
        q = total / n
        self.eq_cache[key] = q
        return q

    def observe_public(self, obs):
        hid = obs.get("hand_id")
        if hid != self.last_hand:
            self.last_hand = hid
            self.last_hist = 0
            self.eq_cache = {}
        hist = obs.get("history", [])
        for ev in hist[self.last_hist:]:
            if ev.get("seat") != obs.get("hero_seat"):
                act = ev.get("action")
                if act in self.opp:
                    self.opp[act] += 1
        self.last_hist = len(hist)
        last = obs.get("last_hand") or {}
        if last.get("hand_id") and last.get("hand_id") != self.seen_result:
            self.seen_result = last.get("hand_id")
            self.eq_cache = {}

    def review(self, obs, reason, force=False):
        hands = int(obs.get("completed_hands", 0))
        if (hands in self.reviewed and not force) or self.ctx.done or self.reviews >= 6:
            return
        if force and sum(self.actions.values()) - self.last_review_step < 8:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
            return
        if self.ctx.budget.remaining().get("wall_s", 0) < 180:
            return
        self.reviewed.add(hands)
        self.reviews += 1
        self.last_review_step = sum(self.actions.values())
        summary = {"mode": self.mode, "margin": self.margin, "value": self.value, "bluff": self.bluff,
                   "opponent_actions": self.opp, "hands": hands, "reason": reason,
                   "public_stats": obs.get("public_stats", {}), "failure_evidence": self.failure,
                   "observation": {k: obs.get(k) for k in ('hole_cards','board','pot','to_call','history','legal')}}
        prompt = ("You are a high-level exploit reviewer for heads-up no-limit Hold'em against frozen "
                  "mc-pot-odds-v2. It opens button hands to 250, reraises premiums, uses random-card equity, "
                  "bets value at .62/.85 and calls around pot odds plus a margin. Choose a posture for the "
                  "next hands from balanced, pressure, cautious. Return compact JSON only: "
                  "{\"mode\":\"balanced|pressure|cautious\",\"margin\":number,\"value\":number,\"bluff\":number}. "
                  "Keep margin .01-.10, value .52-.75, bluff .00-.20; use aggregate evidence, not one result.\n" +
                  json.dumps(summary, separators=(",", ":")))
        try:
            result = self.ctx.think(prompt, timeout_s=120.0)
            raw = str(getattr(result, "text", "") or "")
            a, b = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[a:b + 1]) if a >= 0 and b > a else {}
            if obj.get("mode") in ("balanced", "pressure", "cautious"):
                self.mode = obj["mode"]
            for name, lo, hi in (("margin", .01, .10), ("value", .52, .75), ("bluff", 0.0, .20)):
                val = obj.get(name)
                if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                    setattr(self, name, max(lo, min(hi, float(val))))
            self.ctx.memory["posture"] = {"mode": self.mode, "margin": self.margin, "value": self.value, "bluff": self.bluff}
            self.ctx.log("posture_review", reason=reason, reply=obj)
            self.failure = None
        except Exception as exc:
            self.ctx.log("posture_review_failed", error=type(exc).__name__)

    def choose(self, obs):
        ids = candidate_ids(obs)
        if not ids:
            return None
        hole = tuple(enc(x) for x in obs.get("hole_cards", []) if isinstance(x, str) and len(x) >= 2)
        street = obs.get("street")
        hero = obs.get("hero_seat")
        button = obs.get("button_seat")
        to_call = int(obs.get("to_call", 0) or 0)
        pot = max(1, int(obs.get("pot", 1) or 1))
        q = self.equity(obs)
        if street == "preflop":
            quality = preflop_quality(hole)
            if hero == button and to_call <= 50:
                # Wide positional opening, with a small limp/checkback tail.
                cutoff = .30 if self.mode != "cautious" else .37
                if quality >= cutoff:
                    r = nearest_raise(obs, 250)
                    if r:
                        return r
                return safe_action(obs, "call" if "call" in ids else "fold")
            if to_call == 0:
                if q >= self.value:
                    r = nearest_raise(obs, max(200, int(.75 * pot)))
                    if r:
                        return r
                return safe_action(obs, "check")
            price = to_call / max(1, pot + to_call)
            if q >= .70:
                r = nearest_raise(obs, max(3 * to_call, 850))
                if r and q >= .77:
                    return r
            if q >= price + (.07 if self.mode == "cautious" else .035):
                return safe_action(obs, "call")
            return safe_action(obs, "fold")

        legal = obs.get("legal", {})
        facing = to_call > 0
        if facing:
            price = to_call / max(1, pot + to_call)
            margin = self.margin + (.018 if street != "river" else 0.0)
            if q >= .76 and legal.get("can_raise") and (self.mode == "pressure" or q >= .84):
                r = nearest_raise(obs, int(max(obs.get("street_bets", [0, 0])) + .9 * (pot + to_call)))
                if r:
                    return r
            if q >= price + margin:
                return safe_action(obs, "call")
            return safe_action(obs, "fold")

        # Free action: value bet thinly against the opponent's random-card calls;
        # only pressure the medium-equity region in pressure posture on dry-ish pots.
        target = None
        if q >= self.value:
            frac = .75 if q >= .75 else .50
            target = nearest_raise(obs, int(max(100, obs.get("pot", 1) * frac)))
        elif self.mode == "pressure" and q >= self.value - .07 and self.opp["fold"] > self.opp["call"] * .7:
            target = nearest_raise(obs, int(max(100, obs.get("pot", 1) * .50)))
        if target:
            return target
        return safe_action(obs, "check")

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        while not ctx.done and not obs.get("done", False):
            try:
                self.observe_public(obs)
                hands = int(obs.get("completed_hands", 0))
                if self.failure:
                    self.review(obs, "execution evidence requires replanning", force=True)
                elif hands == 0 and not self.reviewed:
                    self.review(obs, "initial posture calibration")
                elif hands >= 35 and not any(x >= 35 for x in self.reviewed):
                    self.review(obs, "first aggregate action sample")
                elif hands >= 70 and not any(x >= 70 for x in self.reviewed):
                    self.review(obs, "late-match robustness review")
                action = safe_action(obs, self.choose(obs))
                if action is None:
                    break
                before_illegal = int((ctx.progress() or {}).get("illegal_actions", 0))
                nxt = ctx.act(action)
                obs = nxt if nxt is not None else ctx.observe()
                prog = ctx.progress() or {}
                after_illegal = int(prog.get("illegal_actions", 0))
                self.actions[action.split()[0]] = self.actions.get(action.split()[0], 0) + 1
                if after_illegal > before_illegal or obs.get("action_error") or prog.get("since_obs_change", 0) >= 2:
                    ctx.log("action_postcondition_failed", action=action, progress=prog)
                    self.failure = {"kind": "rejected or stale action", "action": action, "progress": prog,
                                    "error": obs.get("action_error")}
                    recovery = safe_action(obs, "check") or safe_action(obs, "call") or safe_action(obs, "fold")
                    if recovery and recovery != action and not ctx.done:
                        try:
                            obs = ctx.act(recovery) or ctx.observe()
                        except Exception:
                            break
            except Exception as exc:
                if ctx.done:
                    break
                ctx.log("policy_recovery", error=type(exc).__name__)
                self.failure = {"kind": "local calculation exception", "error": type(exc).__name__}
                try:
                    obs = ctx.observe()
                    fallback = safe_action(obs, "check") or safe_action(obs, "call") or safe_action(obs, "fold")
                    if fallback:
                        obs = ctx.act(fallback) or ctx.observe()
                    else:
                        break
                except Exception:
                    break


def build(ctx):
    return Harness(ctx)
