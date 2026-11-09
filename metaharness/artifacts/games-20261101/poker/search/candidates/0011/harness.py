"""Finite river betting-tree best response with bounded Astra recovery."""
from __future__ import annotations

import bisect
import hashlib
import itertools
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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


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
    for x in ("check", "fold", "call"):
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


class _SearchStopped(Exception):
    pass


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
        self.last_river = None
        self.expected_response = None
        self.override = None
        self.review_session = None

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

    def river_range(self, obs):
        """Enumerate all 990 legal holes; exact opponent uniform equity.

        Opponent equity includes hero cards as unknown possibilities. Rank CDFs
        subtract hands sharing either opponent card, avoiding 1M comparisons.
        Range weights are approximate; card ranks and river equities are exact.
        """
        hero = tuple(enc(x) for x in obs['hole_cards'])
        board = tuple(enc(x) for x in obs['board'])
        key = ('river_table', hero, board)
        if key not in self.eq_cache:
            available = [c for c in DECK if c not in board]
            all_hands = list(itertools.combinations(available, 2))
            ranks = {h: rank7(h + board) for h in all_hands}
            all_sorted = sorted(ranks.values())
            by_card = {c: sorted(r for h, r in ranks.items() if c in h) for c in available}
            def count_score(values, r):
                return .5 * (bisect.bisect_left(values, r) + bisect.bisect_right(values, r))
            hero_rank = rank7(hero + board)
            rows = []
            for h, r in ranks.items():
                if hero[0] in h or hero[1] in h:
                    continue
                # 47 choose 2 minus 46+46 minus the duplicated own hole = 990.
                eq = (count_score(all_sorted, r) - count_score(by_card[h[0]], r)
                      - count_score(by_card[h[1]], r) + .5) / 990.0
                rows.append((h, eq, outcome(hero_rank, r)))
            self.eq_cache[key] = rows
        rows = self.eq_cache[key]
        opponent = 1 - obs['hero_seat']
        weights = [1.0] * len(rows)
        bets = [100, 50]
        street = 'preflop'
        raises = [0, 0]
        for event in obs.get('history', []):
            if event.get('street') != street:
                street = event.get('street')
                bets = [0, 0]
                raises = [0, 0]
            seat = event.get('seat')
            action = event.get('action')
            price = max(bets) - bets[seat]
            if seat == opponent and street in ('preflop', 'river'):
                for i, (h, q, win) in enumerate(rows):
                    w = 1.0
                    if street == 'preflop':
                        hi, lo = sorted((c // 4 + 2 for c in h), reverse=True)
                        premium = (hi == lo and hi >= 12) or (hi == 14 and lo == 13)
                        if action == 'raise_to' and price > 50:
                            w = .99 if premium else .01
                        elif action == 'raise_to' and seat == 1 and price <= 50:
                            wide = hi == lo or hi >= 12 or (hi == 11 and lo >= 5) or (hi == 10 and lo >= 7) or h[0] % 4 == h[1] % 4 or (hi - lo <= 2 and hi >= 7)
                            w = .99 if wide else .08
                        elif action == 'call':
                            quality = preflop_quality(h)
                            w = .04 + .96 * sigmoid((quality - .44) / .06)
                    else:
                        p = max(1, event.get('pot_before', 1))
                        if action == 'check':
                            w = .02 + .90 * (1.0 - sigmoid((q - .62) / .012))
                        elif action == 'raise_to':
                            if price > 0:
                                w = .01 + .98 * sigmoid((q - .80) / .012)
                            else:
                                observed_cost = event.get('cost', event.get('raise_to', 0))
                                frac = observed_cost / p
                                high = sigmoid((q - .85) / .035)
                                medium = sigmoid((q - .62) / .035) * (1 - sigmoid((q - .85) / .04))
                                weak = 1 - sigmoid((q - .62) / .04)
                                if frac >= .90:
                                    w = .02 + .96 * high
                                elif frac >= .60:
                                    w = .02 + .94 * medium
                                else:
                                    w = .01 + .08 * weak
                        elif action == 'call':
                            threshold = price / max(1, p + price)
                            w = .015 + .98 * sigmoid((q - threshold) / .012)
                            if raises[seat] == 0:
                                w *= .10 + .90 * (1 - sigmoid((q - .80) / .012))
                    weights[i] *= w
            if action == 'raise_to':
                bets[seat] = event.get('raise_to', bets[seat])
                raises[seat] += 1
            elif action == 'call':
                bets[seat] += event.get('cost', 0)
        z = sum(weights)
        if z < 1e-12:
            self.failure = {'kind': 'river range collapse', 'mass': z}
            weights = [1.] * len(rows)
            z = len(rows)
        return [(q, win, w / z) for (_, q, win), w in zip(rows, weights)]

    def river_choose(self, obs):
        """River best response with information-set-consistent continuations.

        The opponent's one-raise limit makes a small finite tree sufficient:
        hero bet -> villain raise -> hero fold/call/reraise -> villain fold/call.
        Checking out of position also branches on visible villain bet sizes.
        Each hero maximum is taken over an entire conditional range, never
        independently per hidden hand. All values are relative to this turn.
        """
        rows = self.river_range(obs)
        hero = obs['hero_seat']
        villain = 1 - hero
        p = obs['pot']
        hb, vb = obs['street_bets'][hero], obs['street_bets'][villain]
        hs, vs = obs['stacks'][hero], obs['stacks'][villain]
        events = [e for e in obs.get('history', []) if e.get('street') == 'river']
        own_raises = sum(e.get('seat') == villain and e.get('action') == 'raise_to' for e in events)
        # Aggregate q/win bins preserve every value used by the river tree.
        # q is exact, so repeated board-only ranks compress especially well.
        bins = {}
        for q, win, w in rows:
            key = (q, win)
            bins[key] = bins.get(key, 0.) + w
        base = [(q, win, w) for (q, win), w in bins.items()]
        self.river_nodes = 0

        def guard():
            self.river_nodes += 1
            if self.ctx.done or self.ctx.budget.remaining().get('wall_s', 0) < 8:
                raise _SearchStopped()

        def mass(rr):
            return sum(w for q, win, w in rr)

        def showdown(rr, x, y):
            return sum(w * (win * (p + x + y) - x) for q, win, w in rr)

        def targets(x, y, last_increment):
            # Internal current-street targets. Cap at the effective stack;
            # an unmatched excess is returned and has no extra chip value.
            top = max(hb + x, vb + y)
            ceiling = min(hb + hs, vb + vs)
            minimum = min(ceiling, top + max(100, last_increment))
            price = max(0, vb + y - hb - x)
            pot = p + x + y
            vals = {minimum, ceiling}
            for frac in (.333333, .5, .75, 1., 1.5):
                vals.add(max(minimum, min(ceiling, int(top + frac * (pot + price) + .5))))
            return sorted(t for t in vals if t > top and t - hb <= hs)

        def terminal_response(rr, x, y):
            # Villain's raising allowance is spent. The call price is capped
            # at remaining chips; after a full effective raise it is exact.
            price = max(0, min(vs - y, hb + x - vb - y))
            pc_denom = max(1., p + x + y + price)
            ev = 0.
            for q, win, w in rr:
                pc = .015 + .97 * sigmoid((q - price / pc_denom) / .012)
                ev += w * ((1 - pc) * (p + y) + pc * (win * (p + x + y + price) - x))
            return ev

        def hero_after_bet(rr, x, y, increment, raising_allowed=True):
            """A fresh decision after a visible opponent bet/raise."""
            guard()
            z = mass(rr)
            if z < 1e-10:
                return 0., 'empty'
            price = min(hs - x, max(0, vb + y - hb - x))
            options = [(-x * z, 'fold'), (showdown(rr, x + price, y), 'call')]
            if raising_allowed and hs - x > price and vs > y:
                for t in targets(x, y, increment):
                    nx = t - hb
                    cost = nx - x
                    # Inherited uncertainty rule applies to internal actions
                    # too: do not reintroduce speculative oversized shoves.
                    odds = max(0, t - vb - y) / max(1, p + nx + y + max(0, t - vb - y))
                    cm = cw = 0.
                    for q, win, w in rr:
                        pc = .015 + .97 * sigmoid((q - odds) / .012)
                        cm += w * pc
                        cw += w * pc * win
                    if cost > 1.5 * (p + x + y + price) and cw / max(cm, 1e-12) < .97:
                        continue
                    value = terminal_response(rr, nx, y) - .015 * cost * z
                    options.append((value, 'raise_to ' + str(t)))
            return max(options, key=lambda item: item[0])

        def bet_value(rr, x, y, villain_raises):
            """Villain faces hero's bet; aggregate the visible raise branch."""
            guard()
            price = max(0, min(vs - y, hb + x - vb - y))
            pot = p + x + y
            top = hb + x
            # Full raise: current top + 3/4 pot after calling, subject to the
            # last increment and effective cap. A short all-in still works.
            last_inc = max(100, top - (vb + y))
            reply = min(hb + hs, vb + vs,
                        max(top + last_inc, int(top + .75 * (pot + price) + .5)))
            can_raise = villain_raises == 0 and vs - y > price and hs > x and reply > top
            raised = []
            ev = fold_mass = called_mass = called_wins = 0.
            for q, win, w in rr:
                pr = .985 * sigmoid((q - .80) / .012) if can_raise else 0.
                pc = (1 - pr) * (.015 + .97 * sigmoid((q - price / max(1, pot + price)) / .012))
                pf = 1 - pr - pc
                ev += w * (pf * (p + y) + pc * (win * (pot + price) - x))
                fold_mass += w * pf
                called_mass += w * (pc + pr)
                called_wins += w * (pc + pr) * win
                if pr > 1e-10:
                    raised.append((q, win, w * pr))
            continuation = 'none'
            if raised:
                ny = reply - vb
                # A short all-in does not reopen hero's raising rights.
                value, continuation = hero_after_bet(raised, x, ny, reply - top,
                                                     reply - top >= last_inc)
                ev += value
            return ev, fold_mass, called_wins / max(called_mass, 1e-12), continuation, mass(raised)

        def check_value(rr):
            # In position a check closes the hand. Out of position the action
            # and size of a future bet reveal information before hero chooses.
            if hero == obs.get('button_seat', 1):
                return showdown(rr, 0, 0), []
            branches = {}
            checked = []
            for q, win, w in rr:
                high = sigmoid((q - .85) / .012)
                middle = (1 - high) * sigmoid((q - .62) / .012)
                weak = max(0., 1 - high - middle)
                probabilities = [(1., high), (.75, middle), (.5, .08 * weak)]
                checked.append((q, win, w * .92 * weak))
                for frac, probability in probabilities:
                    bet = min(vs, hs, max(100, int(frac * p + .5)))
                    if bet > 0 and probability > 1e-10:
                        branches.setdefault(bet, []).append((q, win, w * probability))
            ev = showdown(checked, 0, 0)
            plan = []
            for bet, branch in branches.items():
                val, act = hero_after_bet(branch, 0, bet, bet)
                ev += val
                plan.append({'bet': bet, 'mass': round(mass(branch), 3), 'reply': act})
            return ev, plan

        eq = sum(win * w for q, win, w in base)
        choices = []
        check_plan = []
        price = obs['to_call']
        if price:
            choices = [('fold', 0.), ('call', showdown(base, price, 0))]
        else:
            value, check_plan = check_value(base)
            choices = [('check', value)]
        branch_notes = []
        for action in candidate_ids(obs):
            if not action.startswith('raise_to '):
                continue
            target = min(int(action.split()[1]), vb + vs, hb + hs)
            cost = target - hb
            if cost <= price or target <= vb:
                continue
            value, fold, called_eq, reply, raise_mass = bet_value(base, cost, 0, own_raises)
            if cost > 1.5 * (p + price) and called_eq < .97:
                continue
            choices.append((action, value - .015 * cost))
            branch_notes.append({'action': action, 'fold': round(fold, 3),
                                 'called_equity': round(called_eq, 3), 'vs_raise': reply, 'raise_mass': round(raise_mass, 4)})
        choices = [(a, v) for a, v in choices if a in candidate_ids(obs)]
        best = max(choices, key=lambda av: av[1])[0]
        self.ctx.log('river_tree', equity=round(eq, 4), choice=best,
                     alternatives=[(a, round(v, 1)) for a, v in choices],
                     check_plan=check_plan, raise_plans=branch_notes, nodes=self.river_nodes)
        self.expected_response = None
        for note in branch_notes:
            if note['action'] == best:
                self.expected_response = {'hand_id': obs.get('hand_id'), 'history_len': len(obs.get('history', [])),
                    'hero': hero, 'prob': {'fold': note['fold'], 'raise_to': note['raise_mass'],
                                         'call': max(0., 1 - note['fold'] - note['raise_mass'])}}
        self.last_river = {'hand_id': obs.get('hand_id'), 'choice': best,
                           'alternatives': [(a, round(v, 1)) for a, v in choices]}
        return best

    def observe_public(self, obs):
        expected = self.expected_response
        if expected:
            source = obs if obs.get('hand_id') == expected['hand_id'] else (obs.get('last_hand') or {})
            if source.get('hand_id') == expected['hand_id']:
                evs = source.get('history', [])[expected['history_len']:]
                response = next((e for e in evs if e.get('seat') != expected['hero']), None)
                if response:
                    prob = expected['prob'].get(response.get('action'), 0.)
                    self.ctx.log('river_response_check', action=response.get('action'), predicted=round(prob, 4))
                    if prob < .015:
                        self.failure = {'kind': 'unlikely river response', 'prediction': expected, 'observed': response}
            self.expected_response = None
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
            if last.get('net_chips', 0) <= -5000:
                self.failure = {'kind': 'large public loss: inspect range/plan, not proof of error', 'last_hand': last}
            self.eq_cache = {}

    def review(self, obs, reason, force=False):
        hands = int(obs.get("completed_hands", 0))
        if (hands in self.reviewed and not force) or self.ctx.done or self.reviews >= 6:
            return
        if force and sum(self.actions.values()) - self.last_review_step < 8:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
            return
        if self.ctx.budget.remaining().get("wall_s", 0) < 240:
            return
        self.reviewed.add(hands)
        self.reviews += 1
        self.last_review_step = sum(self.actions.values())
        summary = {"mode": self.mode, "margin": self.margin, "value": self.value, "bluff": self.bluff,
                   "opponent_actions": self.opp, "hands": hands, "reason": reason,
                   "public_stats": obs.get("public_stats", {}), "failure_evidence": self.failure,
                   "river_plan": self.last_river, "current_candidates": candidate_ids(obs),
                   "observation": {k: obs.get(k) for k in ('hole_cards','board','pot','to_call','history','legal')}}
        prompt = ("You are a high-level exploit reviewer for heads-up no-limit Hold'em against frozen "
                  "mc-pot-odds-v2. It opens button hands to 250, reraises premiums, uses random-card equity, "
                  "bets value at .62/.85 and calls around pot odds plus a margin. Choose a posture for the "
                  "next hands from balanced, pressure, cautious. Return compact JSON only: "
                  "{\"mode\":\"balanced|pressure|cautious\",\"margin\":number,\"value\":number,\"bluff\":number}. "
                  "Keep margin .01-.10, value .52-.75, bluff .00-.20; use aggregate evidence, not one result. "
                  "On a failure review you may also supply action: an exact current candidate id to repair this turn. "
                  "The river solver branches on opponent bets and permits re-raises; its range omits earlier street "
                  "actions, so inspect earlier raises in history when a large loss or implausible response occurs.\n" +
                  json.dumps(summary, separators=(",", ":")))
        try:
            result = self.ctx.think(prompt, session_id=self.review_session, timeout_s=180.0)
            self.review_session = getattr(result, "session_id", self.review_session)
            raw = str(getattr(result, "text", "") or "")
            a, b = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[a:b + 1]) if a >= 0 and b > a else {}
            valid_advice = False
            if force and obj.get('action') in candidate_ids(obs):
                self.override = (self.observation_key(obs), obj['action'])
                valid_advice = True
            if obj.get("mode") in ("balanced", "pressure", "cautious"):
                self.mode = obj["mode"]
                valid_advice = True
            for name, lo, hi in (("margin", .01, .10), ("value", .52, .75), ("bluff", 0.0, .20)):
                val = obj.get(name)
                if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                    setattr(self, name, max(lo, min(hi, float(val))))
            self.ctx.memory["posture"] = {"mode": self.mode, "margin": self.margin, "value": self.value, "bluff": self.bluff}
            self.ctx.log("posture_review", reason=reason, reply=obj)
            if valid_advice:
                self.failure = None
            else:
                self.ctx.log("posture_review_unusable", reason=reason, failure_retained=self.failure)
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
        if street == "river":
            return self.river_choose(obs)
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
            # v2's large postflop bets are selected by its own visible-card
            # equity threshold; its raises over a wager require >= .80 equity.
            # A uniform-opponent equity (q) is not the posterior against that
            # selected range.  On later streets use the observed bet fraction
            # as a deliberately conservative range-strength signal.
            hist = obs.get("history", [])
            street_events = [e for e in hist if e.get("street") == street]
            opp_events = [e for e in street_events if e.get("seat") != hero]
            latest_opp = opp_events[-1] if opp_events else None
            opp_bet_fraction = 0.0
            opp_raised = False
            if latest_opp:
                opp_bet_fraction = float(latest_opp.get("cost", 0) or 0) / max(1, int(latest_opp.get("pot_before", pot) or pot))
                opp_raised = any(e.get("seat") == hero and e.get("action") == "raise_to" for e in street_events[:street_events.index(latest_opp)])
            margin = self.margin + (.018 if street != "river" else 0.0)
            if q >= .91 and legal.get("can_raise") and (self.mode == "pressure" or q >= .94):
                r = nearest_raise(obs, int(max(obs.get("street_bets", [0, 0])) + .9 * (pot + to_call)))
                if r:
                    return r
            call_floor = price + margin
            if street in ("flop", "turn") and opp_bet_fraction >= .65:
                call_floor = max(call_floor, .64 if street == "flop" else .70)
            if q >= call_floor:
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

    @staticmethod
    def observation_key(obs):
        return (obs.get('hand_id'), obs.get('street'), tuple(obs.get('board', [])),
                len(obs.get('history', [])), obs.get('pot'), obs.get('to_call'))

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        repeated = 0
        while not ctx.done and not obs.get('done', False):
            if ctx.budget.remaining().get('wall_s', 0) < 8:
                return
            try:
                self.observe_public(obs)
                hands = int(obs.get('completed_hands', 0))
                if self.failure:
                    self.review(obs, 'execution or strategy evidence requires replanning', force=True)
                elif hands == 0 and not self.reviewed:
                    self.review(obs, 'initial posture calibration')
                elif hands >= 35 and not any(x >= 35 for x in self.reviewed):
                    self.review(obs, 'first aggregate action sample')
                elif hands >= 70 and not any(x >= 70 for x in self.reviewed):
                    self.review(obs, 'late-match robustness review')
                if ctx.done:
                    return
                if self.override and self.override[0] == self.observation_key(obs):
                    action = self.override[1]
                    self.override = None
                    self.expected_response = None
                else:
                    self.override = None
                    action = self.choose(obs)
            except _SearchStopped:
                return
            except Exception as exc:
                if ctx.done:
                    return
                self.failure = {'kind': 'local calculation exception', 'error': type(exc).__name__}
                ctx.log('policy_recovery', **self.failure)
                action = 'check' if 'check' in candidate_ids(obs) else 'fold'
            action = safe_action(obs, action)
            if action is None:
                return
            before = ctx.progress() or {}
            key = self.observation_key(obs)
            try:
                nxt = ctx.act(action)
                obs = nxt if nxt is not None else ctx.observe()
            except Exception as exc:
                if ctx.done:
                    return
                self.failure = {'kind': 'action exception', 'error': type(exc).__name__, 'action': action}
                obs = ctx.observe()
            prog = ctx.progress() or {}
            self.actions[action.split()[0]] = self.actions.get(action.split()[0], 0) + 1
            stale = key == self.observation_key(obs) and not obs.get('done')
            repeated = repeated + 1 if stale else 0
            if (prog.get('illegal_actions', 0) > before.get('illegal_actions', 0)
                    or obs.get('action_error') or stale or prog.get('since_obs_change', 0) >= 2):
                self.failure = {'kind': 'rejected or stale action', 'action': action,
                                'progress': prog, 'error': obs.get('action_error')}
                ctx.log('action_postcondition_failed', **self.failure)
                # The next loop sends this observation/evidence to Astra.
                # Force a legal conservative action if the review is unavailable.
                self.override = (self.observation_key(obs), safe_action(obs, 'check' if 'check' in candidate_ids(obs) else 'fold'))
                if repeated >= 4:
                    return


def build(ctx):
    return Harness(ctx)
