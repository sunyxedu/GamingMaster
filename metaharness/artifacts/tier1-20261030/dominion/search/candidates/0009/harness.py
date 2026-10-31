"""Executable acquisition agendas, public-state audits, and bounded Astra reviews.

No game internals, seeds, hidden orders, dynamic code, or cross-episode state.
"""
from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import re
from collections import Counter


TREASURE = {"Copper": 1, "Silver": 2, "Gold": 3}
TERMINAL = {"Chapel", "Smithy", "Witch", "Militia", "Moat", "Workshop", "Mine", "Remodel"}
CANTRIP = {"Village", "Festival", "Laboratory", "Market", "Merchant", "Sentry", "Cellar"}


def obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError):
            pass
    return {}


def num(x, default=0):
    try:
        v = float(x)
        return int(v) if math.isfinite(v) else default
    except (ValueError, TypeError, OverflowError):
        return default


def unpack(text):
    if not isinstance(text, str):
        return {}
    for m in re.finditer(r"\{", text):
        try:
            v, _ = json.JSONDecoder().raw_decode(text[m.start():])
            if isinstance(v, dict):
                return v
        except (ValueError, TypeError):
            pass
    return {}


def vp(cards):
    return (6 * cards.get("Province", 0) + 3 * cards.get("Duchy", 0)
            + cards.get("Estate", 0) - cards.get("Curse", 0)
            + cards.get("Gardens", 0) * (sum(cards.values()) // 10))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = {}
        self.revision = 0
        self.review_turn = -99
        self.review_provinces = 8
        self.calls = 0
        self.pending = None
        self.pending_state = None
        self.retired = []
        self.plan_expiry = None
        self.override = None
        self.events = []
        self.buys = []
        self.evidence = []
        self.bad = set()
        self.unknown = set()
        self.need_review = "opening kingdom / executable agenda"
        self.last_illegal = 0
        self.points_ok = True
        self.current_reason = ""

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def progress(self):
        try:
            return obj(self.ctx.progress())
        except Exception:
            return {}

    def remaining(self):
        try:
            return obj(self.ctx.budget.remaining())
        except Exception:
            return {}

    def turn(self, o):
        return num(obj(o.get("turns_taken")).get("agent"))

    def provinces(self, o):
        return num(obj(obj(o.get("supply")).get("Province")).get("count"), 8)

    def owned(self, o):
        return obj(obj(o.get("self")).get("owned"))

    def hand(self, o):
        return obj(obj(o.get("self")).get("hand"))

    def options(self, o):
        return [(str(x["id"]), str(x.get("label", "")))
                for x in obj(o.get("prompt")).get("options", [])
                if isinstance(x, dict) and "id" in x]

    def bounds(self, o):
        p = obj(o.get("prompt"))
        lo = max(0, num(p.get("min")))
        hi = max(lo, num(p.get("max"), 1))
        return lo, min(hi, len(self.options(o)))

    def validate(self, o, ids):
        if not isinstance(ids, list):
            return None
        if any(not isinstance(x, (str, int)) or isinstance(x, bool) for x in ids):
            return None
        ids = [str(x) for x in ids]
        lo, hi = self.bounds(o)
        valid = {x[0] for x in self.options(o)}
        return ids if lo <= len(ids) <= hi and len(set(ids)) == len(ids) and set(ids) <= valid else None

    def money(self, owned):
        return sum(owned.get(c, 0) * v for c, v in TREASURE.items())

    def default_plan(self, o):
        """A complete fallback agenda, not numeric ranks merged with model advice."""
        own, supply = self.owned(o), obj(o.get("supply"))
        t = self.turn(o)
        agenda = []
        def add(c, cap):
            if c in supply and cap > 0:
                agenda.append([c, cap])
        if t <= 5:
            # On 5/2 Witch precedes Chapel; on 4/3 Chapel plus Silver.
            add("Witch", 1)
            add("Chapel", 1)
            if "Chapel" not in supply:
                add("Sentry", 1)
        if num(obj(supply.get("Curse")).get("count")) > 0:
            add("Witch", 1)
        add("Gold", 99)
        terminals = sum(own.get(c, 0) for c in ("Militia", "Smithy", "Witch", "Moat"))
        if terminals < 1:
            add("Militia", 1)
            add("Smithy", 1)
        if own.get("Chapel", 0) and self.money(own) >= 8 and sum(own.values()) <= 15:
            add("Laboratory", 2)
        if terminals < 2 and own.get("Witch", 0) == 0:
            add("Smithy", 1)
        if self.money(own) >= 12:
            add("Market", 1)
        add("Silver", 99)
        return {"agenda": agenda, "gain": [], "duchy": 3, "estate": 1,
                "copper_floor": 4, "province_from": 1,
                "terminal_order": ["Witch", "Militia", "Smithy", "Moat", "Workshop", "Mine", "Remodel"],
                "note": "Fallback: early thinning/attack, bounded terminals, then treasure and points."}

    def features(self, o):
        own = self.owned(o)
        h = self.hand(o)
        hidden_pool = Counter(own)
        for zone in (h, obj(obj(o.get("self")).get("discard")), obj(obj(o.get("self")).get("in_play"))):
            hidden_pool.subtract(zone)
        return {"cards": sum(own.values()), "treasure_money": self.money(own),
                "terminal_count": sum(own.get(c, 0) for c in TERMINAL),
                "extra_action_cards": own.get("Village", 0) + own.get("Festival", 0),
                "unseen_own_deck_multiset_no_order": dict(+hidden_pool),
                "VP_arithmetic_verified": self.points_ok,
                "purchase_by_coins_now": self.preview(o),
                "recent_buys": self.buys[-8:]}

    def preview(self, o):
        own, supply = self.owned(o), obj(o.get("supply"))
        out = {}
        for coins in (2, 3, 4, 5, 6, 8):
            available = [c for c, d in supply.items() if num(d.get("count")) > 0 and num(d.get("cost"), 99) <= coins]
            out[str(coins)] = self.agenda_card(o, available, False, preview=True) or "PASS"
        return out

    def review_prompt(self, o, reason):
        view = {"turn": self.turn(o), "seat": o.get("agent_seat"),
                "resources": o.get("resources"), "self": o.get("self"),
                "opponent": o.get("opponent"), "supply": o.get("supply"),
                "derived": o.get("derived"), "prompt": o.get("prompt"),
                "decision_id": o.get("decision_id"), "turns": o.get("turns_taken"),
                "features": self.features(o), "progress": self.progress(),
                "evidence": self.evidence[-5:], "events": self.events[-15:]}
        return (
            "You plan Dominion against disclosed BMU-1. Maximize FINAL VP MARGIN. Use only public observations. "
            "BMU never buys attacks, defenses or trashers. It buys Province if its total treasure money>15; "
            "then Duchy if P<5; Estate if P<3; Gold; Duchy if P<7; Smithy if count<treasure cards/11; Silver. "
            "Early Witch can cripple it; Chapel/Sentry can make reliable thin decks. Avoid overbuilding before BMU greens. "
            "The code will execute YOUR agenda, not blend it into hidden priorities. Plan just the NEXT FOUR OWN TURNS. "
            "Each agenda entry [card,total_owned_cap] is checked in order; buy the FIRST affordable nonempty card "
            "whose current owned count is below its cap. Caps are TOTAL ownership, not how many to acquire. "
            "Entries below unaffordable entries still execute. Duplicate card entries may stage acquisitions, e.g. "
            "[[\"Silver\",1],[\"Smithy\",1],[\"Gold\",99],[\"Silver\",99]]. "
            "No entry eligible means PASS; never add junk just to spend money. A cap of 99 means unlimited. "
            "Avoid repeated support cards that compete for one action. Villages require enough draw terminals. "
            "Use explicit Laboratory/Market ordering above Gold when thin engine investment is justified. "
            "Gardens can be a late scoring supplement without Workshop; a full rush needs sufficient gains and pile control. "
            "Before agenda, executor buys Province from turn province_from onwards, Duchy at P<=duchy, Estate at P<=estate; "
            "exact current-turn winning bundles and available Gardens VP may override. Do not delay Provinces casually. "
            "gain is an optional separate ordered capped agenda for Workshop/Remodel gains. Mine upgrades by exact costs. "
            "Routine executor plays +Actions/cantrips before terminals, trashes Curses and early Estates, and Coppers "
            "only while total deck treasure money remains >=copper_floor. Use 4-6, not a count of Coppers. "
            "Terminal order controls collisions, but useful Chapel thinning is handled situationally. "
            "Return ONLY JSON under 240 tokens: {\"agenda\":[[\"card\",cap],...],\"gain\":[],\"duchy\":3,"
            "\"estate\":1,\"province_from\":1,\"copper_floor\":4,\"terminal_order\":[\"card\",...],"
            "\"note\":\"one concrete reason\"}. Up to 12 agenda entries. "
            "For failed/unfamiliar CURRENT prompt also provide option_ids (current legal IDs only). "
            "A new agenda entirely replaces the previous one. Check the purchase preview for unintended behavior.\n"
            + "REVIEW=" + reason + "\nPREVIOUS=" + json.dumps(self.plan, separators=(",", ":"))
            + "\nSTATE=" + json.dumps(view, separators=(",", ":"))
        )

    def consume(self, result, snapshot, current):
        data = unpack(getattr(result, "text", ""))
        supply = obj(current.get("supply"))
        def agenda(raw):
            if not isinstance(raw, list):
                return None
            out = []
            for row in raw[:12]:
                if not isinstance(row, list) or len(row) != 2:
                    continue
                card, cap = row
                if not isinstance(card, str) or card not in supply or card == "Curse":
                    continue
                if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap):
                    continue
                out.append([card, max(0, min(99, int(cap)))])
            return out
        ag = agenda(data.get("agenda"))
        # Late answers never overwrite a more recent revision or an expired agenda.
        stale = self.revision != snapshot["revision"] or self.turn(current) > snapshot["turn"] + 3
        if ag and not stale:
            plan = dict(self.plan)
            plan["agenda"] = ag
            plan["gain"] = agenda(data.get("gain")) or []
            for key, low, high in (("duchy", 0, 6), ("estate", 0, 3), ("copper_floor", 3, 8), ("province_from", 1, 15)):
                if key in data:
                    plan[key] = max(low, min(high, num(data[key], plan.get(key, low))))
            order = data.get("terminal_order")
            if isinstance(order, list):
                plan["terminal_order"] = [c for c in order if isinstance(c, str) and c in TERMINAL]
            plan["note"] = str(data.get("note", ""))[:220]
            self.plan = plan
            self.plan_expiry = snapshot["turn"] + 4
            self.revision += 1
            self.ctx.memory["executable_agenda"] = plan
            self.log("agenda_accepted", plan=plan, preview=self.preview(current), turn=self.turn(current))
        elif not ag:
            self.log("agenda_parse_fallback", text=str(getattr(result, "text", ""))[:180])
        else:
            self.log("stale_agenda_discarded", snapshot=snapshot)
        if not stale and snapshot["decision"] == current.get("decision_id"):
            ids = self.validate(current, data.get("option_ids"))
            if ids is not None:
                self.override = (current.get("decision_id"), ids)

    def poll(self, o):
        if self.pending is None or not self.pending.done():
            return
        future, snapshot = self.pending, self.pending_state
        self.pending = self.pending_state = None
        try:
            self.consume(future.result(), snapshot, o)
        except Exception as exc:
            self.log("agenda_async_failed", error=type(exc).__name__)

    def review(self, o, reason):
        self.poll(o)
        rem = self.remaining()
        wall = rem.get("wall_s", 0)
        if self.ctx.done or self.calls >= 8 or num(rem.get("system2_calls")) <= 1 or wall < 310:
            return False
        if self.pending is not None:
            # Do not queue routine requests behind an unresponsive backend. A new
            # concrete execution failure is allowed one independent recovery call.
            urgent = any(word in reason.lower() for word in
                         ("rejected", "exception", "unfamiliar", "collapse", "disagrees", "differs"))
            if not urgent or self.retired:
                return False
            self.retired.append(self.pending)
            self.pending = self.pending_state = None
            self.revision += 1  # Invalidate the displaced answer, including IDs.
        self.calls += 1
        self.review_turn = self.turn(o)
        self.review_provinces = self.provinces(o)
        self.need_review = None
        snapshot = {"turn": self.turn(o), "decision": o.get("decision_id"), "revision": self.revision}
        prompt = self.review_prompt(o, reason)
        # The model's measured latency excludes potentially long queue waits.
        # Spend at most half the original episode waiting for the first plan,
        # with a hard independent bound and ample time to finish local play.
        wait_s = min(900.0 if self.calls == 1 else 300.0, max(1.0, wall-240.0))
        try:
            self.pending = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=None, timeout_s=240.0))
            self.pending_state = snapshot
            self.pending.result(timeout=wait_s)
            self.poll(o)
        except concurrent.futures.TimeoutError:
            if self.pending is not None and self.pending.done():
                self.poll(o)
            self.log("agenda_wait_cutoff", reason=reason, seconds=wait_s)
        except Exception as exc:
            self.pending = self.pending_state = None
            self.log("agenda_review_failed", error=type(exc).__name__, reason=reason)
        return True

    def score_delta(self, o, card):
        own = Counter(self.owned(o))
        old = vp(own)
        own[card] += 1
        return vp(own) - old

    def agenda_card(self, o, available, gain=False, preview=False):
        own = self.owned(o)
        p = self.provinces(o)
        if "Province" in available and self.turn(o) >= self.plan.get("province_from", 1):
            return "Province"
        if "Duchy" in available and p <= self.plan.get("duchy", 3):
            return "Duchy"
        if "Gardens" in available and p <= 3 and self.score_delta(o, "Gardens") >= 2:
            return "Gardens"
        if "Estate" in available and p <= self.plan.get("estate", 1):
            return "Estate"
        rules = self.plan.get("gain") if gain and self.plan.get("gain") else self.plan.get("agenda", [])
        for card, cap in rules:
            if card in available and own.get(card, 0) < cap:
                return card
        return None

    def finish_bundle(self, o):
        """Exact bounded current buy-phase search; no future-hand forecast."""
        if not self.points_ok:
            return None
        supply = obj(o.get("supply"))
        own = Counter(self.owned(o))
        p = self.provinces(o)
        resources = obj(o.get("resources"))
        money, buys = num(resources.get("coins")), min(5, num(resources.get("buys"), 1))
        low = sum(num(d.get("count")) <= buys for d in supply.values())
        if p > buys and low < 3:
            return None
        legal = {c for _, c in self.options(o)}
        # Only a real end permits maximization of immediate VP irrespective of deck quality.
        cards = [c for c, d in supply.items() if c != "Curse" and 0 < num(d.get("count")) and num(d.get("cost"), 99) <= money]
        counts = {c: num(d.get("count")) for c, d in supply.items()}
        old = vp(own)
        best = None
        visited = 0
        def walk(start, cash, left, bundle):
            nonlocal best, visited
            visited += 1
            if visited > 2000:
                return
            ending = counts.get("Province", 1) == 0 or sum(v == 0 for v in counts.values()) >= 3
            delta = vp(own) - old
            if bundle and ending:
                margin = num(obj(o.get("derived")).get("vp_margin")) + delta
                candidate = (delta, -len(bundle), -sum(num(supply[c].get("cost")) for c in bundle))
                # Take a non-losing finish; losing lines remain strategic decisions for Astra.
                turns = obj(o.get("turns_taken"))
                nonlosing = margin>0 or (margin==0 and num(turns.get("agent"))<=num(turns.get("opponent")))
                if nonlosing and (best is None or candidate > best[0]):
                    best = (candidate, list(bundle))
            if not left:
                return
            for i in range(start, len(cards)):
                c = cards[i]
                cost = num(supply[c].get("cost"), 99)
                if counts[c] <= 0 or cost > cash:
                    continue
                counts[c] -= 1
                own[c] += 1
                walk(i, cash-cost, left-1, bundle+[c])
                own[c] -= 1
                counts[c] += 1
        walk(0, money, buys, [])
        if best:
            bundle = best[1]
            first = next((c for c in bundle if c in legal), None)
            self.log("exact_finish", bundle=bundle, delta=best[0][0])
            return first
        return None

    def buy(self, o):
        opts = self.options(o)
        available = [c for _, c in opts]
        p = obj(o.get("prompt"))
        if p.get("source") == "Mine":
            card = max(available, key=lambda c: TREASURE.get(c, 0), default=None)
        else:
            card = self.finish_bundle(o) if p.get("kind") == "buy" else None
            card = card or self.agenda_card(o, available, p.get("kind") == "gain")
        lo, _ = self.bounds(o)
        if card is None and lo:
            # Mandatory gains still gain useful economy, not the alphabetically first junk.
            card = max(available, key=lambda c: (self.score_delta(o, c) if self.provinces(o) <= 2 else 0,
                                               {"Gold":30,"Silver":20,"Laboratory":18,"Market":16,"Smithy":12,"Copper":-5,"Curse":-100}.get(c, 2)))
        if card is None:
            return []
        return [next(oid for oid, c in opts if c == card)]

    def trash(self, o, opts, limit):
        own, hand = self.owned(o), self.hand(o)
        source = obj(o.get("prompt")).get("source")
        p = self.provinces(o)
        supply = obj(o.get("supply"))
        if source == "Mine":
            upgrades = []
            for oid, c in opts:
                cost = num(obj(supply.get(c)).get("cost"))
                for target, d in supply.items():
                    if target in TREASURE and num(d.get("count")) > 0 and num(d.get("cost"),99) <= cost+3:
                        if TREASURE[target] > TREASURE.get(c, 0):
                            upgrades.append((TREASURE[target]-TREASURE.get(c,0), -TREASURE.get(c,0), oid))
            return [max(upgrades)[2]] if upgrades else []
        if source == "Remodel":
            def value(pair):
                c = pair[1]
                if c == "Gold" and p <= 3 and num(obj(supply.get("Province")).get("count")) > 0:
                    return 100
                if c == "Province" and p > 0:
                    return -100
                return {"Curse":90,"Estate":85,"Copper":60,"Chapel":40 if p<=3 else 5,"Silver":30}.get(c,10)
            return [max(opts, key=value)[0]] if opts else []
        floor = self.plan.get("copper_floor", 4)
        money = self.money(own)
        selected = []
        for oid, c in sorted(opts, key=lambda x: {"Curse":0,"Estate":1,"Copper":2,"Chapel":3}.get(x[1],9)):
            want = c == "Curse" or (c == "Estate" and p > 2)
            if c == "Copper" and p > 2 and money > floor:
                want = True
                money -= 1
            if c == "Chapel" and own.get("Chapel",0)>1:
                want = True
            if want and len(selected) < limit:
                selected.append(oid)
        return selected

    def keep(self, card, o):
        # Victory points are not spending power when selecting a hand to keep.
        val = {"Curse":-10,"Estate":-5,"Duchy":-5,"Province":-5,"Gardens":-5,
               "Copper":2,"Silver":5,"Gold":8,"Chapel":0,"Workshop":3,"Mine":3,"Remodel":3,
               "Village":5,"Festival":8,"Market":9,"Merchant":6,"Laboratory":11,"Sentry":10,
               "Smithy":9,"Militia":9,"Witch":10,"Moat":6,"Cellar":3}.get(card, 4)
        h = self.hand(o)
        if card == "Chapel" and self.provinces(o)>3 and h.get("Estate",0)+h.get("Curse",0)>=2:
            val = 10
        if card == "Village" and sum(h.get(c,0) for c in TERMINAL)>=2:
            val = 11
        return val

    def play(self, o):
        opts = self.options(o)
        h = self.hand(o)
        actions = num(obj(o.get("resources")).get("actions"), 1)
        order = self.plan.get("terminal_order", [])
        trash_options = [(str(i),c) for i,c in enumerate(c for c,n in h.items() for _ in range(n))]
        junk = len(self.trash(o, trash_options, 4))
        def value(c):
            if c in CANTRIP:
                return {"Village":150 if actions<=1 else 110,"Festival":145 if actions<=1 else 105,
                        "Sentry":140,"Laboratory":135,"Market":130,"Merchant":125,
                        "Cellar":55 if h.get("Chapel",0) and junk>=2 else 100}[c]
            if c == "Chapel":
                return 90 if junk>=3 else (65 if junk>=2 else (15 if junk else -10))
            if c == "Witch" and num(obj(obj(o.get("supply")).get("Curse")).get("count"))>0:
                return 85
            if c == "Smithy":
                # Once no cards remain to draw, payload is preferable.
                drawable = num(obj(o.get("self")).get("deck_count"))+num(obj(o.get("self")).get("discard_count"))
                if drawable == 0:
                    return 0
            return 70-3*order.index(c) if c in order else 30
        best = max(opts, key=lambda z:value(z[1])) if opts else None
        return [best[0]] if best and value(best[1])>0 else []

    def pick(self, o):
        if self.override and self.override[0] == o.get("decision_id"):
            ids = self.override[1]
            self.override = None
            return ids
        p, opts = obj(o.get("prompt")), self.options(o)
        lo, hi = self.bounds(o)
        kind, source = str(p.get("kind", "")), str(p.get("source", ""))
        if not opts:
            return []
        if all(l.lower() in {"true","false","yes","no"} for _,l in opts):
            want = source == "Moat" or kind == "reaction" or "reveal" in str(p.get("instruction","")).lower()
            targets = {"true","yes"} if want else {"false","no"}
            return [next((oid for oid,c in opts if c.lower() in targets), opts[0][0])]
        if kind in {"buy","gain"}:
            return self.buy(o)
        if kind in {"play_action","action"}:
            return self.play(o)
        if "trash" in kind:
            return self.trash(o, opts, hi)
        if "discard" in kind:
            ranked = sorted(opts, key=lambda z:self.keep(z[1],o))
            if lo:
                return [oid for oid,_ in ranked[:lo]]
            return [oid for oid,c in ranked if self.keep(c,o)<0][:hi]
        if "topdeck" in kind:
            return [oid for oid,_ in sorted(opts,key=lambda z:self.keep(z[1],o))[:lo]]
        if "order" in kind:
            return [oid for oid,_ in sorted(opts,key=lambda z:self.keep(z[1],o),reverse=True)[:hi]]
        return [oid for oid,_ in opts[:lo]]

    def legalize(self, o, ids):
        lo, hi = self.bounds(o)
        valid = {oid for oid,_ in self.options(o)}
        out = []
        for oid in ids:
            if oid in valid and oid not in out and len(out)<hi:
                out.append(oid)
        for oid,_ in self.options(o):
            if len(out)>=lo:
                break
            if oid not in out:
                out.append(oid)
        return out

    def audit(self, before, after, ids, prog):
        p = obj(before.get("prompt"))
        selected = [c for oid,c in self.options(before) if oid in ids]
        did = before.get("decision_id")
        illegal = num(prog.get("illegal_actions"))
        failed = (illegal>self.last_illegal or num(after.get("illegal_action_count"))>num(before.get("illegal_action_count"))
                  or (after.get("decision_id")==did and not after.get("terminal")))
        self.last_illegal = illegal
        if failed:
            self.bad.add((did, tuple(ids)))
            self.need_review = "Rejected or unchanged action; repair CURRENT prompt and plan"
            self.evidence.append({"ids":ids,"prompt":p,"last_error":after.get("last_error"),"progress":prog})
        elif p.get("kind") in {"buy","gain"} and selected:
            c = selected[0]
            witnessed = any(x.get("player")=="agent" and x.get("card")==c and x.get("event") in {"buy","gain"}
                            for x in after.get("public_log_delta", []) if isinstance(x, dict))
            gained = self.owned(after).get(c,0)>self.owned(before).get(c,0)
            if not witnessed and not gained:
                self.need_review = "Acquisition effect differs from expected public ownership"
                self.evidence.append({"expected_gain":c,"before":self.owned(before),"after":self.owned(after)})
            self.log("agenda_execution", card=c, kind=p.get("kind"), revision=self.revision,
                     verified=witnessed or gained, turn=self.turn(before))
        if p.get("kind")=="buy" and not failed:
            self.buys.append({"turn":self.turn(before),"coins":num(obj(before.get("resources")).get("coins")),
                              "card":selected[0] if selected else "PASS"})
        for ev in after.get("public_log_delta", []):
            if isinstance(ev, dict) and ev.get("event") in {"buy","gain","trash"}:
                self.events.append(ev)
        self.events = self.events[-50:]
        self.evidence = self.evidence[-10:]
        for player in ("self", "opponent"):
            state = obj(after.get(player))
            if state and vp(obj(state.get("owned"))) != num(state.get("vp")):
                if self.points_ok:
                    self.need_review = "VP arithmetic disagrees with authoritative ledger; exact finish search disabled"
                self.points_ok = False
        return failed

    def run_episode(self, game):
        try:
            o = obj(self.ctx.observe())
        except Exception:
            return
        self.plan = self.default_plan(o)
        count = 0
        while not self.ctx.done and not game.done and count<10000:
            count += 1
            if o.get("terminal"):
                return
            self.poll(o)
            p = obj(o.get("prompt"))
            if not p:
                return
            kind = p.get("kind")
            known = {"buy","gain","play_action","action","trash","discard","reaction","yes_no","topdeck","order"}
            key = (kind, str(p.get("source")))
            if kind not in known and key not in self.unknown:
                self.unknown.add(key)
                self.need_review = "Unfamiliar prompt " + str(p)
            # The ordinary review unit is four own turns, not every nested choice.
            if kind in {"buy","play_action"} and self.turn(o)>=self.review_turn+4 and self.calls<6:
                self.need_review = self.need_review or "Four-turn agenda expiry; audit actual deck and race"
            if self.provinces(o)<=3 and self.review_provinces>3:
                self.need_review = self.need_review or "Province race changed to scoring pressure"
            own = self.owned(o)
            if self.money(own)<3 and not any(own.get(c,0) for c in ("Festival","Militia","Market","Workshop")) and self.turn(o)>2:
                if self.turn(o)>self.review_turn:
                    self.need_review = "Economy collapse: less than $3 total treasure and no money/gain payload"
            if kind == "buy" and self.turn(o)>self.review_turn:
                coins = num(obj(o.get("resources")).get("coins"))
                available = [c for _,c in self.options(o)]
                if coins>=3 and self.agenda_card(o, available) is None:
                    self.need_review = "Agenda exhausted despite spending power; expand it or confirm intentional pass"
                distinct = {b["turn"]: b["coins"] for b in self.buys[-10:]}
                recent = list(distinct.values())[-3:]
                if len(recent)==3 and max(recent)<3 and self.turn(o)>=6:
                    self.need_review = "Three successive turns below $3: economic plan is not producing purchases"
            if self.need_review:
                self.review(o, self.need_review)
            if self.plan_expiry is not None and self.turn(o)>=self.plan_expiry:
                self.log("expired_agenda_fallback", previous=self.plan, turn=self.turn(o))
                self.plan = self.default_plan(o)
                self.plan_expiry = None
                self.revision += 1
            ids = self.legalize(o, self.pick(o))
            did = o.get("decision_id")
            if (did, tuple(ids)) in self.bad:
                lo, hi = self.bounds(o)
                option_ids = [i for i,_ in self.options(o)]
                alternatives = itertools.chain.from_iterable(
                    itertools.combinations(option_ids, size) for size in range(lo, hi+1))
                ids = next((list(a) for a in alternatives if (did,a) not in self.bad), None)
                if ids is None:
                    return
            action = "choose " + str(did) + (" " + " ".join(ids) if ids else "")
            before = o
            try:
                nxt = self.ctx.act(action)
                o = obj(nxt if nxt is not None else self.ctx.observe())
            except Exception as exc:
                if self.ctx.done:
                    return
                self.bad.add((did,tuple(ids)))
                self.evidence.append({"action":action,"exception":type(exc).__name__})
                self.need_review = "Action exception; repair with fresh current options"
                try:
                    o = obj(self.ctx.observe())
                except Exception:
                    return
            self.audit(before, o, ids, self.progress())


def build(ctx):
    return Harness(ctx)
