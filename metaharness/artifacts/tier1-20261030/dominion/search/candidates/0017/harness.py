"""Persistent attack scheduling with auditable two-card draw distributions.

All state comes from public observations; routine execution revalidates legal
choices and ownership after each action. Astra reviews actual failure evidence.
"""
from __future__ import annotations

import json
import re
import concurrent.futures
import math


def _obs(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _count(mapping, card):
    return int((mapping or {}).get(card, 0) or 0)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = {"mode": "money", "priority": []}
        self.session = None
        self.last_review_turn = 0
        self.review_due = True
        self.last_illegal = 0
        self.stalled = 0
        self.last_decision = None
        self.reviews = 0
        self.planning_disabled = False
        self.evidence = []
        self.override = None
        self.last_race = None
        self.bad_actions = set()
        self.recent_buys = []
        # Public-race controller.  This is deliberately separate from Astra's
        # advisory plan: a stale/slow review must not prevent an urgent switch.
        self.race_mode = False
        self.attack_seen = False
        self.last_regime = None
        self.treasure_floor = 4
        self.cash_repair = False
        self.last_cash_failure = None
        self.scoring_committed = False
        self.review_context = None

    def _remaining(self):
        try:
            return self.ctx.budget.remaining()
        except Exception:
            return {}

    def _parse_plan(self, text, obs):
        if not isinstance(text, str):
            return None
        data = None
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = json.JSONDecoder().raw_decode(text[match.start():])
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except Exception:
                continue
        kingdom = set(obs.get("kingdom", [])) | {"Copper", "Silver", "Gold", "Province", "Duchy", "Estate"}
        if data is None:
            return None
        ids = data.get("option_ids")
        p, options = self._options(obs)
        if isinstance(ids, list):
            ids = [str(x) for x in ids]
            if len(ids) == len(set(ids)) and all(x in {v[0] for v in options} for x in ids):
                if int(p.get("min", 0)) <= len(ids) <= int(p.get("max", 1)):
                    self.override = (obs.get("decision_id"), ids)
        mode = str(data.get("mode", "money")).lower()
        if mode not in {"money", "engine", "witch", "gardens"}:
            mode = "money"
        raw = data.get("priority", [])
        if isinstance(raw, str):
            raw = re.split(r"[,>\n]", raw)
        if not isinstance(raw, list):
            raw = []
        priority = []
        for item in raw:
            name = str(item).strip()
            found = next((c for c in kingdom if c.lower() == name.lower()), None)
            if found and found not in priority:
                priority.append(found)
        if not priority:
            return None
        targets = {}
        raw_targets = data.get("targets", {})
        if isinstance(raw_targets, dict):
            for name, value in raw_targets.items():
                if name in kingdom and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    targets[name] = max(0, min(8, int(value)))
        result = {"mode": mode, "priority": priority[:7], "targets": targets,
                  "note": str(data.get("note", ""))[:180]}
        for key, default, limit in (("duchy_at", 3, 5), ("estate_at", 1, 3)):
            value = data.get(key, default)
            result[key] = max(0, min(limit, int(value))) if isinstance(value, (int, float)) and math.isfinite(value) else default
        return result

    def _safe_progress(self):
        try:
            p = self.ctx.progress()
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def treasure_money(owned):
        return sum(owned.get(c, 0)*v for c,v in (("Copper",1),("Silver",2),("Gold",3)))

    def _regime_adjust(self, obs):
        own = obs.get("self", {}).get("owned", {}) or {}
        opp = obs.get("opponent", {}).get("owned", {}) or {}
        p = int((obs.get("derived") or {}).get("provinces_left", 8))
        # Trashing Estates lowers our VP without making BMU any closer to an
        # ending. A race interrupt needs actual opponent scoring AND scarcity.
        self.race_mode = bool(p <= 4 and opp.get("Province", 0) > own.get("Province", 0))
        self.cash_repair = self.treasure_money(own) < self.treasure_floor
        regime = "race" if self.race_mode else "repair_cash" if self.cash_repair else "build"
        if regime != self.last_regime:
            self.ctx.log("cash_flow_regime", regime=regime,
                         stable_treasure_money=self.treasure_money(own),
                         provinces_left=p, own_provinces=own.get("Province", 0),
                         opponent_provinces=opp.get("Province", 0))
            self.last_regime = regime

    def _options(self, obs):
        prompt = obs.get("prompt") or {}
        options = prompt.get("options") or []
        return prompt, [(str(x.get("id")), str(x.get("label", ""))) for x in options]

    def _index_for(self, options, labels):
        result = []
        wanted = {x.lower() for x in labels}
        for oid, label in options:
            if label.lower() in wanted:
                result.append(oid)
        return result

    def _buy_rank(self, card, obs):
        own = obs.get("self", {}).get("owned", {})
        supply = obs.get("supply", {})
        n = sum(own.values())
        turn = int(obs.get("turns_taken", {}).get("agent", 0))
        p = int(obs.get("derived", {}).get("provinces_left", 8))
        mode = self.plan.get("mode", "money")
        engine = mode == "engine"
        late = p <= 3
        curse = supply.get("Curse", {}).get("count", 0)
        actions = sum(own.get(c, 0) for c in ("Smithy", "Witch", "Militia", "Moat", "Chapel", "Workshop", "Remodel", "Mine"))
        villages = own.get("Village", 0) + own.get("Festival", 0)
        money = sum(own.get(c, 0)*v for c,v in (("Copper",1),("Silver",2),("Gold",3)))
        defaults = {"Chapel":1, "Sentry":1, "Militia":1, "Witch":2 if engine else 1,
                    "Smithy":max(1, n//14) if not engine else max(2, villages+1),
                    "Village":max(0, actions-1) if engine else 0,
                    "Festival":max(1, actions-1) if engine else 1,
                    "Laboratory":4 if engine else 2, "Market":4 if engine else 1,
                    "Cellar":1, "Moat":0, "Merchant":1, "Mine":0, "Remodel":0,
                    "Workshop":4 if mode=="gardens" else 0}
        targets = dict(defaults)
        targets.update(self.plan.get("targets", {}))
        # Ownership caps precede every regime branch. Racing does not grant
        # a bypass that buys terminal attacks forever.
        if "Action" in supply.get(card, {}).get("types", []):
            if own.get(card, 0) >= targets.get(card, 1):
                return -20
        if self.cash_repair and card in {"Silver", "Gold"}:
            return 115 if card == "Gold" else 110
        if self.cash_repair and "Action" in supply.get(card, {}).get("types", []):
            return -10
        if self.race_mode:
            if card == "Province": return 165
            if card == "Duchy": return 145
            if card == "Estate": return 118 if p <= 2 else -10
            if card == "Gold": return 108
            if card == "Silver": return 76
            if "Action" in supply.get(card, {}).get("types", []): return -5
        if card == "Province": return 120
        if card == "Duchy": return 105 if p <= self.plan.get("duchy_at", 3) else -5
        if card == "Estate": return 90 if p <= self.plan.get("estate_at", 1) else -10
        if card == "Curse": return -100
        if card == "Copper": return 5 if mode == "gardens" and own.get("Gardens",0)>=3 else -20
        if card == "Gold": return 88
        if card == "Silver": return 61 if money < 18 else 48
        if card == "Gardens":
            return 96 if late and n >= 29 else (83 if mode=="gardens" and n >= 17 else -5)
        if own.get(card,0) >= targets.get(card, 1): return -5
        values = {"Witch":94 if curse > 0 else 45,
                  "Chapel":80 if turn <= 5 else 20,
                  "Sentry":85 if turn <= 7 else 45,
                  "Militia":78, "Smithy":74, "Village":67 if actions>villages+1 else 25,
                  "Festival":76 if actions>villages+1 else 55,
                  "Laboratory":77 if money>=12 else 53,
                  "Market":69 if money>=12 else 48,
                  "Merchant":50, "Cellar":25, "Moat":35, "Mine":35,
                  "Remodel":35, "Workshop":84 if mode=="gardens" else 20}
        value = values.get(card, 15)
        priority = self.plan.get("priority", [])
        if card in priority: value += max(1, 10 - 2*priority.index(card))
        # Missing engine draw/action capacity can beat Gold, but only with functional payload.
        if engine and money >= 12 and card in priority and card in {"Smithy", "Village", "Festival", "Laboratory", "Market"}:
            value += 12
        if not engine and card in {"Smithy", "Militia", "Witch", "Moat"}:
            productive = sum(own.get(c,0) for c in ("Smithy", "Militia", "Witch", "Moat"))
            if productive >= max(2,n//10) + villages: value -= 35
        if late: value -= 35
        return value

    def _keep_value(self, card, obs):
        hand = obs.get("self", {}).get("hand", {})
        values = {"Curse":-5,"Estate":-4,"Duchy":-4,"Province":-4,"Gardens":-4,
                  "Copper":2,"Silver":5,"Gold":8,"Chapel":1,"Workshop":3,
                  "Smithy":8,"Witch":9,"Militia":8,"Village":5,"Festival":8,
                  "Laboratory":10,"Market":9,"Merchant":6,"Sentry":9,"Moat":5}
        v = values.get(card, 4)
        if card == "Chapel" and hand.get("Estate",0)+hand.get("Curse",0)>=2: v=7
        if card == "Village" and sum(hand.get(c,0) for c in ("Smithy","Witch","Militia"))>=2: v=9
        return v

    def _trash_ids(self, obs, options, hi):
        own = obs.get("self", {}).get("owned", {})
        p = obs.get("derived", {}).get("provinces_left", 8)
        money = self.treasure_money(own)
        selected = []
        for oid, c in sorted(options, key=lambda x:{"Curse":0,"Estate":1,"Copper":2}.get(x[1],5)):
            good = c == "Curse" or (c == "Estate" and p > 2 and not self.scoring_committed)
            if c == "Copper" and p > 2 and money > self.treasure_floor:
                good = True
                money -= 1
            if good: selected.append(oid)
            if len(selected) >= hi: break
        return selected[:hi]

    def _pick(self, obs):
        prompt, options = self._options(obs)
        kind = str(prompt.get("kind", ""))
        if not options:
            return "pass"
        decision = obs.get("decision_id", "")
        lo = int(prompt.get("min", 0) or 0)
        hi = int(prompt.get("max", 1) if prompt.get("max") is not None else 1)
        hi = max(lo, min(hi, len(options)))
        me = obs.get("self", {})
        hand = me.get("hand", {}) or {}
        owned = me.get("owned", {}) or {}
        supply = obs.get("supply", {}) or {}
        source = str(prompt.get("source") or "")

        if self.override and self.override[0] == obs.get("decision_id"):
            selected = self.override[1]
            self.override = None
            return "choose " + str(decision) + (" " + " ".join(selected) if selected else "")
        if {label.lower() for _,label in options} <= {"true","false","yes","no"}:
            instruction = str(prompt.get("instruction", "")).lower()
            want = "reaction" in kind or "moat" in source.lower() or "reveal" in instruction
            oid = next((oid for oid,label in options if label.lower() in ({"true","yes"} if want else {"false","no"})), options[0][0])
            return f"choose {decision} {oid}"
        if "trash" in kind or "discard" in kind or "topdeck" in kind or "order" in kind:
            if "trash" in kind:
                if source == "Mine":
                    ranked = sorted(options,key=lambda x:{"Copper":0,"Silver":1,"Gold":2}.get(x[1],3))
                    selected = [ranked[0][0]] if ranked and ranked[0][1] != "Gold" else []
                elif source == "Remodel":
                    p = obs.get("derived",{}).get("provinces_left",8)
                    def remodel_rank(c):
                        if c == "Gold" and p<=2: return 100
                        return {"Estate":90,"Curse":85,"Copper":60,"Silver":30}.get(c,0)
                    selected = [max(options,key=lambda x:remodel_rank(x[1]))[0]]
                else:
                    selected = self._trash_ids(obs,options,hi)
            elif "discard" in kind:
                ranked = sorted(options,key=lambda x:self._keep_value(x[1],obs))
                if lo > 0:
                    selected = [oid for oid,_ in ranked[:lo]]
                else:
                    selected = [oid for oid,c in ranked if self._keep_value(c,obs)<0][:hi]
            elif "topdeck" in kind:
                selected = [oid for oid,_ in sorted(options,key=lambda x:self._keep_value(x[1],obs))[:lo]]
            else:
                selected = [oid for oid,_ in sorted(options,key=lambda x:self._keep_value(x[1],obs),reverse=True)[:max(lo,hi)]]
            for oid,_ in options:
                if len(selected)>=lo: break
                if oid not in selected: selected.append(oid)
            return "choose " + str(decision) + (" " + " ".join(selected[:hi]) if selected else "")

        if kind == "buy" or kind == "gain":
            if source == "Mine":
                oid,_ = max(options,key=lambda x:{"Gold":3,"Silver":2,"Copper":1}.get(x[1],0))
                return f"choose {decision} {oid}"
            cards = [label for _, label in options]
            ranked = sorted(cards, key=lambda c: self._buy_rank(c, obs), reverse=True)
            if lo == 0 and self._buy_rank(ranked[0],obs) <= 0:
                return f"choose {decision}"
            labels = ranked[:max(1, hi)]
            ids = self._index_for(options, labels)
            chosen = ids[:hi]
            if len(chosen) < lo:
                chosen += [oid for oid, _ in options if oid not in chosen][:lo-len(chosen)]
            if not chosen and lo == 0:
                return "pass"
            return f"choose {decision}" + (" " + " ".join(chosen) if chosen else "")

        if kind in {"play_action", "action"} or "action" in kind:
            labels = [label for _, label in options]
            action_counts = {c: _count(owned, c) for c in labels}
            actions = int((obs.get("resources") or {}).get("actions", 1))
            terminals = {"Chapel", "Smithy", "Witch", "Militia", "Remodel", "Workshop", "Mine"}
            def action_value(card):
                value = 20
                if card in {"Village", "Festival", "Market", "Laboratory", "Merchant", "Cellar", "Sentry"}: value += 100
                if card == "Chapel":
                    junk = len(self._trash_ids(obs, [(str(i),c) for c,n in hand.items() for i in range(n)],4))
                    value += 12*junk if junk else -40
                if card in {"Witch", "Militia"}: value += 27
                if card in {"Smithy", "Sentry"}: value += 22
                if card in terminals and actions > 1: value += 12
                if card == "Village" and actions <= 1: value += 25
                return value
            oid, card = max(options, key=lambda x: action_value(x[1]))
            if action_value(card) <= 0 and lo == 0: return f"choose {decision}"
            return f"choose {decision} {oid}"

        # Optional unresolved modes decline. Forced prompts receive the first legal IDs.
        if lo == 0:
            return "pass"
        return "choose " + str(decision) + " " + " ".join(oid for oid, _ in options[:lo])


class ExactRegimeHarness(Harness):
    """Exact liquidation bundles replace the parent's stochastic continuation."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.pending = None
        self.pending_id = None
        self.abandoned = []
        self.liquidate_at = 3
        self.end_reviewed = False
        self.recovery_count = 0
        self.last_candidates = []
        self.last_purchase = None
        self.buy_turn = -1
        self.last_progress = {}
        self.no_economy_turns = 0
        self.last_failure_review = None

    @staticmethod
    def vp(owned):
        n = sum(owned.values())
        return (owned.get("Estate", 0) + 3*owned.get("Duchy", 0)
                + 6*owned.get("Province", 0) - owned.get("Curse", 0)
                + owned.get("Gardens", 0)*(n//10))

    def _consume(self, obs):
        if self.pending is None or not self.pending.done():
            return
        future, decision = self.pending, self.pending_id
        self.pending = None
        try:
            raw = getattr(future.result(), "text", "")
            data = None
            for match in re.finditer(r"\{", raw if isinstance(raw, str) else ""):
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(raw[match.start():])
                    if isinstance(parsed, dict):
                        data = parsed
                        break
                except Exception:
                    continue
            if not data:
                self.ctx.log("invalid_liquidation_review", text=str(raw)[:140])
                return
            reserve = data.get("treasure_floor")
            if type(reserve) is int and 4 <= reserve <= 6:
                self.treasure_floor = reserve
            threshold = data.get("liquidate_at")
            if type(threshold) is int and 2 <= threshold <= 4:
                self.liquidate_at = threshold
            # Direct choices are valid only at the frozen request's decision.
            ids = data.get("option_ids")
            prompt, options = self._options(obs)
            if (decision == obs.get("decision_id") and isinstance(ids, list)
                    and all(isinstance(x, (str, int)) for x in ids)):
                ids = [str(x) for x in ids]
                if (len(set(ids)) == len(ids) and set(ids) <= {x[0] for x in options}
                        and int(prompt.get("min", 0)) <= len(ids) <= int(prompt.get("max", 1))):
                    self.override = (obs.get("decision_id"), ids)
            # Do not silently change the validated opening on broad prose.
            # Explicit support repairs are still allowed after observed failure.
            if data.get("repair_plan") is True and isinstance(data.get("priority"), list):
                plan = {k:v for k,v in data.items() if k != "option_ids"}
                parsed = self._parse_plan(json.dumps(plan), obs)
                if parsed:
                    self.plan = parsed
            self.ctx.memory["liquidation_review"] = data
            self.ctx.log("liquidation_review", review=data)
        except Exception as exc:
            self.ctx.log("liquidation_review_failed", error=type(exc).__name__)

    def _review(self, obs, evidence="opening"):
        self._consume(obs)
        turn = int(obs.get("turns_taken", {}).get("agent", 0))
        self.last_review_turn = turn
        self.review_due = False
        if self.ctx.done or self.reviews >= 4:
            return
        failure = any(k in evidence for k in
                      ("failed_action", "unchanged", "purchase_mismatch", "unknown_prompt",
                       "economy_stall", "vp_mismatch", "cash_failure", "ownership_cap", "thinning_reserve"))
        if self.reviews and not failure:
            if self.end_reviewed or int(obs.get("derived", {}).get("provinces_left", 8)) > 4:
                return
        if self.pending is not None:
            if not failure or self.abandoned:
                return
            self.abandoned.append(self.pending)
            self.pending = None
        rem = self._remaining()
        if rem.get("system2_calls", 0) <= 1 or rem.get("wall_s", 0) < 350:
            return
        view = {k:obs.get(k) for k in ("kingdom", "supply", "self", "opponent", "resources",
                                       "turns_taken", "derived", "prompt", "last_error")}
        view.update(progress=self._safe_progress(), recent_buys=self.recent_buys[-8:],
                    failures=self.evidence[-5:], exact_candidates=self.last_candidates[:6],
                    stable_treasure_money=self.treasure_money(obs.get("self",{}).get("owned",{})),
                    treasure_floor=self.treasure_floor, scoring_committed=self.scoring_committed,
                    current_plan=self.plan)
        prompt = (
            "Advise a Dominion controller against disclosed BMU-1; maximize FINAL VP MARGIN. "
            "The opening buys limited attack/draw/thinning then money. At <=3 Provinces it "
            "liquidates: enumerates complete legal buy bundles and maximizes exact immediate VP, "
            "including Gardens thresholds, all extra buys, and purchases after an ending pile empties. "
            "This deliberately avoids speculative hidden-deck throughput forecasts. "
            "The attack scheduler resolves a one-Action Militia/Witch collision in favor of Witch "
            "while Curses remain and no opponent Moat is owned, except it preserves a guaranteed "
            "Province early and compares exact immediate VP plus Curse margin when liquidating. "
            "It enumerates Witch two-draw cash from public own-zone multisets, not hidden order; "
            "checks actual cash and Curse gains after every selected attack. You can set "
            "persistent_attack:false to restore the baseline, or true after diagnosing a failure. "
            "Core invariant: Chapel/Sentry may remove Copper only while leaving at least "
            "treasure_floor dollars in owned Copper/Silver/Gold. Militia and other Action "
            "money do NOT count: Chapel may consume their only action. All support ownership "
            "caps apply even in race mode. Estate trashing is not opponent scoring pressure. "
            "Choose when to liquidate from public race and economy. BMU buys Province if its "
            "owned Treasure money>15, Duchy at <5 Provinces, Estate at <3, then Gold, Duchy at <7, "
            "Smithy if count<treasure cards/11, Silver. It never buys attacks or trashing. "
            "Return ONLY JSON under 180 tokens: liquidate_at (2,3,4), treasure_floor (4,5,6), "
            "note (short), persistent_attack (boolean, normally true), optionally "
            "option_ids for CURRENT prompt on a concrete failure. Only if evidence invalidates "
            "the opening include repair_plan:true, mode:money|engine|witch|gardens, priority:[cards], "
            "targets:{card:count}. Card text/options authoritative. Never guess hidden hands. "
            "Evidence=" + evidence + " STATE=" + json.dumps(view, separators=(",", ":")))
        self.reviews += 1
        if int(obs.get("derived", {}).get("provinces_left", 8)) <= 4:
            self.end_reviewed = True
        self.pending_id = obs.get("decision_id")
        try:
            self.pending = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=None, timeout_s=300.0))
            try:
                self.pending.result(timeout=min(320.0, max(1.0, rem.get("wall_s", 0)-45.0)))
            except concurrent.futures.TimeoutError:
                self.ctx.log("review_wait_cutoff", seconds=320, pending=True)
            except Exception:
                pass
            self._consume(obs)
        except Exception as exc:
            self.pending = None
            self.ctx.log("review_start_failed", error=type(exc).__name__)

    def _liquidation(self, obs):
        prompt, options = self._options(obs)
        supply = obs.get("supply", {})
        own = obs.get("self", {}).get("owned", {})
        prov = int(obs.get("derived", {}).get("provinces_left", 8))
        empty = sum(p.get("count", 0) == 0 for p in supply.values())
        near = sum(p.get("count", 0) <= 2 for p in supply.values())
        # A deficit moves the liquidation horizon forward one Province.  The
        # bundle evaluator still checks exact current money, buys, supply and
        # Gardens thresholds, so this cannot create an illegal or guessed buy.
        threshold = max(self.liquidate_at, 4) if getattr(self, "race_mode", False) else self.liquidate_at
        if prov > threshold and not (empty >= 2 or near >= 4):
            return None
        if self.vp(own) != obs.get("self", {}).get("vp", self.vp(own)):
            if not any("vp_mismatch" in x for x in self.evidence):
                self.evidence.append({"vp_mismatch": {"calculated": self.vp(own),
                                      "observed": obs.get("self", {}).get("vp")}})
                self.review_due = True
            return None
        gain = prompt.get("kind") == "gain"
        buys = 1 if gain else int(obs.get("resources", {}).get("buys", 1))
        cash = int(obs.get("resources", {}).get("coins", 0))
        allowed = {c:oid for oid,c in options if c in supply}
        if buys < 1 or not allowed:
            return None
        scoring = {"Province", "Duchy", "Estate", "Gardens"}
        candidates = [c for c in allowed if c in scoring or
                      (own.get("Gardens", 0) > 0 and supply[c]["cost"] == 0) or
                      (empty >= 2 and supply[c]["count"] <= buys)]
        # Include local economic alternatives for zero-VP turns. There is no
        # forecast; immediate point gains outrank economic preference in this regime.
        economic = sorted(allowed, key=lambda c:self._buy_rank(c, obs), reverse=True)
        candidates += [c for c in economic if self._buy_rank(c, obs)>0][:2]
        candidates = sorted(set(candidates))
        results = []
        oldvp = self.vp(own)
        margin = int(obs.get("derived", {}).get("vp_margin", 0))
        def visit(bundle, start, coins, counts):
            newown = dict(own)
            for c,n in counts.items():
                newown[c] = newown.get(c, 0)+n
            points = self.vp(newown)-oldvp
            endings = prov-counts.get("Province", 0) <= 0 or (
                empty+sum(n == supply[c]["count"] for c,n in counts.items()) >= 3)
            # Only an actual exact margin is used. Avoid ending behind on a
            # non-scoring pile merely because a zero-value filler was available.
            ending_preference = 1 if endings and margin+points >= 0 else -1 if endings else 0
            utility = sum(max(0, self._buy_rank(c, obs)) for c in bundle)
            results.append((points, ending_preference, utility, tuple(bundle), endings))
            if len(bundle) >= min(buys, 6) or len(results) >= 2500:
                return
            for i in range(start, len(candidates)):
                c = candidates[i]
                price = 0 if gain else int(supply[c]["cost"])
                if price > coins or counts.get(c, 0) >= int(supply[c]["count"]):
                    continue
                nxt = dict(counts)
                nxt[c] = nxt.get(c, 0)+1
                visit(bundle+[c], i, coins-price, nxt)
        visit([], 0, cash, {})
        if int(prompt.get("min", 0)):
            results = [r for r in results if r[3]]
        if not results:
            return None
        results.sort(key=lambda x:(x[0], x[1], x[2], -len(x[3])), reverse=True)
        self.last_candidates = [{"cards":list(r[3]), "exact_vp_gain":r[0],
                                 "ends":r[4], "margin_if_end":margin+r[0] if r[4] else None}
                                for r in results[:6]]
        best = results[0]
        if best[0] <= 0:
            return None
        self.ctx.log("exact_liquidation", candidates=self.last_candidates,
                     threshold=self.liquidate_at, decision_id=obs.get("decision_id"))
        # Recompute after every purchase. Score before the final Province is
        # emptied still includes all remaining buys this turn.
        bundle = best[3]
        card = max(bundle, key=lambda c:(c == "Province", int(supply[c]["cost"])))
        return "choose " + str(obs.get("decision_id")) + " " + allowed[card]

    def _pick(self, obs):
        self._consume(obs)
        self._regime_adjust(obs)
        if self.override and self.override[0] == obs.get("decision_id"):
            return super()._pick(obs)
        prompt = obs.get("prompt") or {}
        if prompt.get("kind") in {"buy", "gain"} and prompt.get("source") != "Mine":
            decision = self._liquidation(obs)
            if decision is not None:
                return decision
        return super()._pick(obs)

    def _safe_action(self, obs, proposed):
        prompt, opts = self._options(obs)
        did = obs.get("decision_id")
        lo, hi = int(prompt.get("min", 0)), int(prompt.get("max", 1))
        valid = {x[0] for x in opts}
        selected = proposed.split()[2:] if proposed.startswith("choose ") else []
        if (len(selected) != len(set(selected)) or not set(selected) <= valid
                or not lo <= len(selected) <= hi):
            selected = [x[0] for x in opts[:lo]]
        action = "choose " + str(did) + (" " + " ".join(selected) if selected else "")
        if (did, action) in self.bad_actions:
            from itertools import combinations
            for n in range(lo, min(hi, len(opts))+1):
                for ids in combinations([x[0] for x in opts], n):
                    alt = "choose " + str(did) + (" " + " ".join(ids) if ids else "")
                    if (did, alt) not in self.bad_actions:
                        return alt
        return action

    def run_episode(self, game):
        if self.ctx.done:
            return
        obs = _obs(game.observe())
        while not self.ctx.done and not game.done:
            self._consume(obs)
            self._regime_adjust(obs)
            prompt = obs.get("prompt") or {}
            if not prompt:
                return
            turn = int(obs.get("turns_taken", {}).get("agent", 0))
            prov = int(obs.get("derived", {}).get("provinces_left", 8))
            known = {"buy", "gain", "play_action", "trash", "discard", "reaction", "yes_no", "topdeck", "order"}
            if prompt.get("kind") not in known:
                self.evidence.append({"unknown_prompt":prompt})
                self.review_due = True
            if prompt.get("kind") == "buy" and self.buy_turn != turn:
                self.buy_turn = turn
                coins = int(obs.get("resources", {}).get("coins", 0))
                self.no_economy_turns = self.no_economy_turns+1 if coins < 3 and prov > 3 else 0
                money = self.treasure_money(obs.get("self",{}).get("owned",{}))
                if money < self.treasure_floor and money != self.last_cash_failure:
                    self.last_cash_failure = money
                    self.evidence.append({"cash_failure": {"stable_money": money,
                        "floor": self.treasure_floor, "turn": turn,
                        "owned": obs.get("self",{}).get("owned",{})}})
                    self.review_due = True
                if self.no_economy_turns >= 3:
                    self.evidence.append({"economy_stall":self.recent_buys[-3:]})
                    self.review_due = True
                    self.no_economy_turns = 0
            if self.review_due or (self.reviews == 0) or (prov <= 4 and not self.end_reviewed):
                self.last_candidates = []
                if prompt.get("kind") in {"buy", "gain"} and prompt.get("source") != "Mine":
                    self._liquidation(obs)
                evidence = str(self.evidence[-3:]) if self.evidence else "public race/kingdom review"
                self._review(obs, evidence)
            if self.ctx.done:
                return
            action = self._safe_action(obs, self._pick(obs))
            did = obs.get("decision_id")
            before = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            before_prog = self._safe_progress()
            chosen = action.split()[2:]
            card = next((x["label"] for x in prompt.get("options", []) if str(x["id"]) in chosen), None)
            own_before = obs.get("self", {}).get("owned", {})
            if prompt.get("kind") == "buy":
                self.recent_buys.append({"turn":turn, "coins":obs.get("resources", {}).get("coins"), "buy":card or "pass"})
                self.ctx.log("purchase", **self.recent_buys[-1])
                if card in {"Duchy", "Estate", "Gardens"} and prov <= 4:
                    self.scoring_committed = True
            rejected = False
            try:
                nxt = self.ctx.act(action)
            except Exception as exc:
                if self.ctx.done:
                    return
                rejected = True
                self.evidence.append({"failed_action":action, "error":type(exc).__name__})
                try:
                    nxt = self.ctx.observe()
                except Exception:
                    return
            obs = _obs(nxt if nxt is not None else self.ctx.observe())
            progress = self._safe_progress()
            after = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            unchanged = obs.get("decision_id") == did and not obs.get("terminal")
            illegal = progress.get("illegal_actions", 0) > before_prog.get("illegal_actions", 0)
            if rejected or unchanged or before == after or illegal or progress.get("since_obs_change", 0) >= 2:
                self.bad_actions.add((did, action))
                self.evidence.append({"failed_action":action, "unchanged":unchanged,
                                      "last_error":obs.get("last_error"), "progress":progress})
                self.review_due = True
                self.stalled += 1
            else:
                self.stalled = 0
            if card and prompt.get("kind") in {"buy", "gain"} and not unchanged:
                got = obs.get("self", {}).get("owned", {}).get(card, 0)
                if obs.get("self") and got < own_before.get(card, 0)+1:
                    self.evidence.append({"purchase_mismatch":card, "expected":own_before.get(card,0)+1, "actual":got})
                    self.review_due = True
            if (not unchanged and obs.get("self") and prompt.get("kind") == "trash"
                    and prompt.get("source") in {"Chapel", "Sentry"}):
                reserve_before = self.treasure_money(own_before)
                reserve_after = self.treasure_money(obs["self"].get("owned", {}))
                if reserve_after < min(reserve_before, self.treasure_floor):
                    self.evidence.append({"cash_failure": {"source":prompt.get("source"),
                        "before":reserve_before, "after":reserve_after,
                        "floor":self.treasure_floor, "action":action, "progress":progress}})
                    self.review_due = True
            self.last_progress = progress
            if self.stalled > 5:
                # A broken adaptor must not spin until the wall deadline.
                return


class PersistentAttackHarness(ExactRegimeHarness):
    """Resolve the Witch/Militia tie using observable durable attack value.

    Only terminal collisions are changed. Two draws are enumerated exactly from
    public zone multisets, assuming unknown order is exchangeable. No imagined
    draw is committed; only one current option is ever posted.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.attack_policy = True
        self.attack_comparison = None
        self.attack_expectation = None
        self.attack_checks = []
        self.attack_failure = False

    def _draw_cash(self, obs):
        me = obs['self']
        hand, discard, played = (me.get(k, {}) for k in ('hand', 'discard', 'in_play'))
        deck = dict(me.get('owned', {}))
        for zone in (hand, discard, played):
            for card, n in zone.items():
                deck[card] = deck.get(card, 0) - n
        if (any(n < 0 for n in deck.values()) or sum(deck.values()) != me.get('deck_count')
                or sum(discard.values()) != me.get('discard_count')):
            raise ValueError('public own-zone conservation failed')
        money = {'Copper': 1, 'Silver': 2, 'Gold': 3}
        base = int(obs['resources'].get('coins', 0)) + sum(money.get(c, 0)*n for c,n in hand.items())
        merchant = played.get('Merchant', 0) if played.get('Silver', 0) == 0 else 0
        silver = hand.get('Silver', 0) > 0
        outcomes = {}

        def draw(pool, waiting, left, cash, has_silver, probability):
            total = sum(pool.values())
            if left == 0 or (not total and not sum(waiting.values())):
                final = cash + (merchant if has_silver else 0)
                outcomes[final] = outcomes.get(final, 0.0) + probability
                return
            if not total:
                draw(waiting, {}, left, cash, has_silver, probability)
                return
            for card, n in pool.items():
                if n <= 0:
                    continue
                rest = dict(pool)
                rest[card] -= 1
                draw(rest, waiting, left-1, cash+money.get(card, 0),
                     has_silver or card == 'Silver', probability*n/total)

        draw(deck, discard, 2, base, silver, 1.0)
        militia_cash = base + 2 + (merchant if silver else 0)
        return militia_cash, outcomes

    def _point_bundle(self, obs, cash):
        """Exact obtainable VP this buy phase, no future-turn forecast."""
        supply, own = obs.get('supply', {}), obs['self']['owned']
        buys = min(6, int(obs['resources'].get('buys', 1)))
        cards = [c for c in ('Province', 'Duchy', 'Estate', 'Gardens', 'Copper')
                 if supply.get(c, {}).get('count', 0) > 0]
        old = self.vp(own)
        best = 0

        def visit(start, remaining, money, counts):
            nonlocal best
            updated = dict(own)
            for c, n in counts.items():
                updated[c] = updated.get(c, 0) + n
            best = max(best, self.vp(updated)-old)
            if remaining == 0:
                return
            for i in range(start, len(cards)):
                c = cards[i]
                cost = int(supply[c]['cost'])
                if cost <= money and counts.get(c, 0) < supply[c]['count']:
                    next_counts = dict(counts)
                    next_counts[c] = next_counts.get(c, 0)+1
                    visit(i, remaining-1, money-cost, next_counts)
        visit(0, buys, cash, {})
        return best

    def _attack_fault(self, evidence):
        self.attack_policy = False
        self.attack_failure = True
        self.review_due = True
        self.evidence.append({'attack_failure': evidence})

    def _audit_attack(self, obs):
        expected = self.attack_expectation
        if not expected or obs.get('decision_id') == expected['decision']:
            return
        self.attack_expectation = None
        # With one Action, either terminal finishes the Action phase and takes
        # us to this turn's buy prompt. Do not compare a later turn to a forecast.
        if (obs.get('turns_taken', {}).get('agent') != expected['turn']
                or (obs.get('prompt') or {}).get('kind') != 'buy'):
            if not obs.get('terminal'):
                self._attack_fault({'unexpected_transition': expected, 'prompt': obs.get('prompt')})
            return
        cash = int(obs['resources'].get('coins', 0))
        actual_curses = obs['opponent']['owned'].get('Curse', 0)
        check = {'decision': expected['decision'], 'played': expected['played'],
                 'cash': cash, 'cash_support': expected['cash_support'],
                 'expected_curses': expected['curses'], 'actual_curses': actual_curses}
        self.attack_checks.append(check)
        self.ctx.log('attack_effect_check', **check)
        if cash not in expected['cash_support'] or actual_curses != expected['curses']:
            self._attack_fault(check)

    def _consume(self, obs):
        self._audit_attack(obs)
        pending = self.pending
        fresh = self.pending_id == obs.get('decision_id')
        previous_advice = self.ctx.memory.get('liquidation_review')
        super()._consume(obs)
        if pending is not None and self.pending is None:
            advice = self.ctx.memory.get('liquidation_review', {})
            if advice is previous_advice:
                return
            flag = advice.get('persistent_attack')
            if type(flag) is bool and (not flag or not self.attack_failure or fresh):
                self.attack_policy = flag
                self.attack_failure = False

    def _review(self, obs, evidence='opening'):
        # The parent planner includes these values in STATE; an explicit model
        # response can disable/re-enable this experiment after checking evidence.
        self.plan['attack_scheduler'] = {
            'enabled': self.attack_policy,
            'last_comparison': self.attack_comparison,
            'recent_checks': self.attack_checks[-3:]}
        if self.attack_failure:
            evidence = 'failed_action attack_failure: ' + evidence
        return super()._review(obs, evidence)

    def _pick(self, obs):
        self._consume(obs)
        direct = bool(self.override and self.override[0] == obs.get('decision_id'))
        baseline = super()._pick(obs)
        p, options = self._options(obs)
        labels = {card: oid for oid, card in options}
        chosen = baseline.split()[2:]
        if (direct or not self.attack_policy or p.get('kind') != 'play_action'
                or int(obs.get('resources', {}).get('actions', 1)) != 1
                or not {'Militia', 'Witch'} <= set(labels)
                or chosen != [labels['Militia']]
                or obs.get('supply', {}).get('Curse', {}).get('count', 0) <= 0):
            return baseline
        opp = obs.get('opponent', {}).get('owned', {})
        # A known possible defense defeats the guaranteed-Curse premise.
        if opp.get('Moat', 0):
            return baseline
        try:
            militia_cash, outcomes = self._draw_cash(obs)
        except (KeyError, ValueError, TypeError) as exc:
            self._attack_fault({'zone_error': str(exc)})
            return baseline
        cursed = dict(opp)
        cursed['Curse'] = cursed.get('Curse', 0)+1
        curse_margin = self.vp(opp)-self.vp(cursed)
        provinces = int(obs.get('derived', {}).get('provinces_left', 8))
        threshold = max(4, self.liquidate_at) if self.race_mode else self.liquidate_at
        supply = obs.get('supply', {})
        empty = sum(pile.get('count', 0) == 0 for pile in supply.values())
        near = sum(pile.get('count', 0) <= 2 for pile in supply.values())
        liquidating = provinces <= threshold or empty >= 2 or near >= 4
        militia_points = self._point_bundle(obs, militia_cash)
        witch_points = sum(prob*self._point_bundle(obs, cash) for cash,prob in outcomes.items())
        probability_province = sum(prob for cash,prob in outcomes.items() if cash >= 8)
        reason = 'lasting Curse and two-card draw'
        use_witch = curse_margin > 0
        if liquidating:
            use_witch = use_witch and witch_points+curse_margin >= militia_points-1e-9
            reason = 'current-turn VP plus exact Curse margin'
        elif militia_cash >= 8 and probability_province < 1-1e-9:
            use_witch = False
            reason = 'retain guaranteed Province'
        comparison = {
            'decision': obs.get('decision_id'), 'militia_cash': militia_cash,
            'witch_cash_distribution': outcomes, 'curse_margin': curse_margin,
            'militia_immediate_points': militia_points,
            'witch_expected_immediate_points': round(witch_points, 5),
            'selected': 'Witch' if use_witch else 'Militia', 'reason': reason}
        self.attack_comparison = comparison
        self.ctx.log('persistent_attack_choice', **comparison)
        played = comparison['selected']
        self.attack_expectation = {
            'decision': obs.get('decision_id'), 'turn': obs['turns_taken']['agent'],
            'played': played, 'cash_support': sorted(outcomes) if use_witch else [militia_cash],
            'curses': opp.get('Curse', 0)+(1 if use_witch else 0)}
        return 'choose ' + str(obs.get('decision_id')) + ' ' + labels[played]


def build(ctx):
    return PersistentAttackHarness(ctx)
