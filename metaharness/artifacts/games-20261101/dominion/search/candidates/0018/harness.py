"""Exact terminal-draw distributions with Astra pricing attack opportunity cost.

All state comes from public observations; routine execution revalidates legal
choices and ownership after each action. Astra reviews actual failure evidence.
"""
from __future__ import annotations

import json
import re
import concurrent.futures
import math
from collections import Counter


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
            "Core invariant: Chapel/Sentry may remove Copper only while leaving at least "
            "treasure_floor dollars in owned Copper/Silver/Gold. Militia and other Action "
            "money do NOT count: Chapel may consume their only action. All support ownership "
            "caps apply even in race mode. Estate trashing is not opponent scoring pressure. "
            "Choose when to liquidate from public race and economy. BMU buys Province if its "
            "owned Treasure money>15, Duchy at <5 Provinces, Estate at <3, then Gold, Duchy at <7, "
            "Smithy if count<treasure cards/11, Silver. It never buys attacks or trashing. "
            "Return ONLY JSON under 180 tokens: liquidate_at (2,3,4), treasure_floor (4,5,6), "
            "note (short), optionally "
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


class TerminalHarness(ExactRegimeHarness):
    """One-turn exact arithmetic; no guessed future deck order or rollout policy."""

    MONEY = {'Copper': 1, 'Silver': 2, 'Gold': 3}
    DRAW = {'Smithy': 3, 'Witch': 2, 'Moat': 2}
    TERMINALS = {'Smithy', 'Witch', 'Moat', 'Militia'}

    def __init__(self, ctx):
        super().__init__(ctx)
        self.terminal_enabled = True
        self.attack_price = 1.0
        self.terminal_reviews = 0
        self.terminal_pending = None
        self.terminal_pending_id = None
        self.terminal_last_review = None
        self.terminal_forecast = None
        self.terminal_misses = 0
        self.last_terminal_table = []

    @staticmethod
    def draw_distribution(counts, draws):
        """Exact hypergeometric cash / Silver-presence distribution.

        All zero-money cards are equivalent because the last Action is spent.
        Enumeration is of multisets, weighted by their occurrence counts.
        """
        categories = [counts.get(c, 0) for c in ('Copper', 'Silver', 'Gold')]
        categories.append(sum(counts.values()) - sum(categories))
        if any(n < 0 for n in categories):
            raise ValueError('negative draw category')
        draws = min(max(0, draws), sum(categories))
        denom = math.comb(sum(categories), draws)
        result = Counter()
        for copper in range(min(categories[0], draws) + 1):
            for silver in range(min(categories[1], draws - copper) + 1):
                for gold in range(min(categories[2], draws - copper - silver) + 1):
                    zero = draws - copper - silver - gold
                    if zero > categories[3]:
                        continue
                    weight = (math.comb(categories[0], copper)
                              * math.comb(categories[1], silver)
                              * math.comb(categories[2], gold)
                              * math.comb(categories[3], zero))
                    result[(copper + 2*silver + 3*gold, silver > 0)] += weight / denom
        return dict(result)

    def _visible_deck(self, obs):
        me = obs.get('self', {})
        counts = Counter(me.get('owned', {}))
        for zone in ('hand', 'discard', 'in_play'):
            counts.subtract(me.get(zone, {}))
        if any(n < 0 for n in counts.values()) or sum(counts.values()) != me.get('deck_count'):
            raise ValueError('public zone conservation mismatch')
        if sum(me.get('hand', {}).values()) != me.get('hand_count'):
            raise ValueError('public hand conservation mismatch')
        if sum(me.get('discard', {}).values()) != me.get('discard_count'):
            raise ValueError('public discard conservation mismatch')
        return +counts

    def _cash_distribution(self, obs, card, deck):
        me = obs['self']
        hand = me.get('hand', {})
        draw = self.DRAW.get(card, 0)
        base = int(obs['resources'].get('coins', 0)) + self.treasure_money(hand)
        if card == 'Militia':
            base += 2
        first = self.draw_distribution(deck, min(draw, sum(deck.values())))
        remaining = max(0, draw - sum(deck.values()))
        second = self.draw_distribution(me.get('discard', {}), remaining)
        distribution = Counter()
        merchants = me.get('in_play', {}).get('Merchant', 0)
        # At a supported Action prompt the automatic Treasure stage has not
        # started, so every Merchant in play still awaits its first Silver.
        for (cash1, silver1), prob1 in first.items():
            for (cash2, silver2), prob2 in second.items():
                bonus = merchants if hand.get('Silver', 0) or silver1 or silver2 else 0
                distribution[base + cash1 + cash2 + bonus] += prob1 * prob2
        return dict(distribution)

    def _point_frontier(self, obs, max_cash):
        """Exact best immediate VP from up to B buys, including Gardens mass."""
        supply = obs['supply']
        own = obs['self']['owned']
        cards = [c for c in ('Province', 'Duchy', 'Estate', 'Gardens', 'Copper')
                 if supply.get(c, {}).get('count', 0) > 0]
        # Free Copper weakly dominates every non-scoring filler for immediate
        # Gardens mass. Large buy counts fall back instead of truncating search.
        buys = int(obs['resources'].get('buys', 1))
        if buys > 6:
            raise ValueError('point frontier buy bound')
        best = [0] * (max_cash + 1)
        old = self.vp(own)
        def visit(start, spent, left, counts):
            newown = dict(own)
            for c, n in counts.items():
                newown[c] = newown.get(c, 0) + n
            best[spent] = max(best[spent], self.vp(newown) - old)
            if left == 0:
                return
            for i in range(start, len(cards)):
                c = cards[i]
                price = int(supply[c]['cost'])
                if spent + price > max_cash or counts.get(c, 0) >= supply[c]['count']:
                    continue
                counts[c] += 1
                visit(i, spent + price, left - 1, counts)
                counts[c] -= 1
        visit(0, 0, max(0, buys), Counter())
        for cash in range(1, len(best)):
            best[cash] = max(best[cash], best[cash-1])
        return best

    def _terminal_table(self, obs):
        prompt, options = self._options(obs)
        if (not self.terminal_enabled or prompt.get('kind') not in {'action', 'play_action'}
                or obs.get('active_player') != 'agent'
                or int(obs.get('resources', {}).get('actions', 0)) != 1
                or int(obs.get('derived', {}).get('provinces_left', 8)) > 3
                or len(options) < 2 or any(c not in self.TERMINALS for _, c in options)):
            return []
        # Exact local arithmetic is only enabled for the verified Base text.
        markers = {'Smithy': ('3 cards',), 'Moat': ('2 cards',),
                   'Witch': ('2 cards', 'curse'), 'Militia': ('2 money', '3 cards')}
        for _, c in options:
            text = ' '.join(str(obs['supply'].get(c, {}).get('text', '')).lower().split())
            if not all(marker in text for marker in markers[c]):
                return []
        deck = self._visible_deck(obs)
        laws = {c:self._cash_distribution(obs, c, deck) for _, c in options}
        frontier = self._point_frontier(obs, max(cash for law in laws.values() for cash in law))
        opp = obs.get('opponent', {})
        curses = obs.get('supply', {}).get('Curse', {}).get('count', 0)
        rows = []
        for c, law in laws.items():
            # Militia's delayed denial is an explicit strategic estimate. Witch
            # direct Curse VP is certain only if the opponent cannot reveal Moat.
            attack = (self.attack_price if c == 'Militia' and opp.get('hand_count', 0) > 3
                      else 1.0 if c == 'Witch' and curses > 0
                      and not opp.get('owned', {}).get('Moat', 0) else 0.0)
            points = sum(prob * frontier[cash] for cash, prob in law.items())
            rows.append({'card':c, 'cash_law':law,
                         'expected_immediate_vp':round(points, 6),
                         'province_affordable_probability':round(sum(p for cash,p in law.items() if cash >= 8),6),
                         'estimated_attack_margin':attack,
                         'utility':points + attack,
                         'mean_cash':sum(cash*p for cash,p in law.items())})
        return rows

    def _consume_terminal(self, obs):
        if self.terminal_pending is None or not self.terminal_pending.done():
            return None
        future, did = self.terminal_pending, self.terminal_pending_id
        self.terminal_pending = None
        try:
            raw = future.result().text
            data = None
            for match in re.finditer(r'\{', raw if isinstance(raw, str) else ''):
                try:
                    data, _ = json.JSONDecoder().raw_decode(raw[match.start():])
                    if isinstance(data, dict):
                        break
                except Exception:
                    continue
            if not isinstance(data, dict):
                return None
            value = data.get('militia_margin')
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 3:
                self.attack_price = float(value)
            if type(data.get('use_terminal')) is bool:
                self.terminal_enabled = data['use_terminal']
            self.ctx.log('terminal_review', decision_id=did, current_decision=obs.get('decision_id'), review=data)
            self.ctx.memory['terminal_review'] = data
            card = data.get('card')
            if did == obs.get('decision_id') and card in self.TERMINALS:
                return card
        except Exception as exc:
            self.ctx.log('terminal_review_failed', error=type(exc).__name__)
        return None

    def _review_terminal(self, obs, rows, default):
        rem = self._remaining()
        if (self.ctx.done or self.terminal_reviews >= 2 or self.terminal_pending is not None
                or rem.get('system2_calls', 0) <= 2 or rem.get('wall_s', 0) < 365):
            return None
        p = int(obs['derived']['provinces_left'])
        # One decision-specific review, then one genuinely changed race/failure.
        if self.terminal_last_review is not None and p >= self.terminal_last_review and not self.terminal_misses:
            return None
        self.terminal_last_review = p
        view = {k:obs.get(k) for k in ('decision_id','supply','self','opponent','turns_taken',
                                      'resources','derived','prompt','public_log_delta')}
        view.update(exact_terminal_table=rows, baseline=default, progress=self._safe_progress(),
                    failures=self.evidence[-4:], militia_margin=self.attack_price)
        prompt = (
            'Choose this late Dominion TERMINAL ACTION against BMU-1 to maximize FINAL VP MARGIN. '
            'Exactly one Action remains; only Militia/Smithy/Witch/Moat are legal. '
            'The table enumerates the cash distribution from public own deck composition '
            '(owned minus hand/discard/in_play), preserving discard reshuffle timing, and '
            'the exact maximum immediate VP with current coins/buys/supply. '
            'Drawn Actions cannot be played. This is NOT a final-score prediction: '
            'attacking delays opponent scoring and Smithy cycles our deck. The hidden order '
            'is unknown; cash probabilities assume exchangeability. Militia estimated_attack_margin '
            'is merely a tunable opportunity-cost estimate, NOT a measured fact. '
            'BMU plays Smithy then buys Province if owned Treasure money>15, Duchy at P<5, '
            'Estate at P<3, Gold, Duchy at P<7, Smithy if count<treasures/11, Silver. '
            'Choose from the table; consider lead, seat/tie rule, ending, and whether the '
            'attack meaningfully changes the race. Reply ONLY JSON under 100 tokens: '
            '{"card":"Smithy","militia_margin":1.0,"use_terminal":true,"note":"brief reason"}. '
            'militia_margin must be 0..3 and prices uncertain future denial in VP; '
            'use_terminal=false restores the baseline if evidence invalidates the method. STATE=' +
            json.dumps(view, separators=(',', ':')))
        self.terminal_reviews += 1
        self.terminal_pending_id = obs.get('decision_id')
        try:
            self.terminal_pending = self.ctx.spawn(lambda:self.ctx.think(prompt, session_id=None, timeout_s=300.0))
            try:
                self.terminal_pending.result(timeout=min(320.0, max(1, rem['wall_s']-45)))
            except concurrent.futures.TimeoutError:
                self.ctx.log('terminal_wait_cutoff', seconds=320)
            except Exception:
                pass
            return self._consume_terminal(obs)
        except Exception as exc:
            self.ctx.log('terminal_review_start_failed', error=type(exc).__name__)
            return None

    def _pick(self, obs):
        self._consume_terminal(obs)
        # Audit at the next Buy: for these terminal-only turns, sampled cash has
        # no unresolved action-policy uncertainty. Exact support misses disable
        # the mechanism and feed the parent's recovery planner.
        forecast = self.terminal_forecast
        if forecast and (obs.get('prompt') or {}).get('kind') == 'buy':
            self.terminal_forecast = None
            turn = int(obs.get('turns_taken', {}).get('agent', 0))
            if turn == forecast['turn']:
                actual = int(obs.get('resources', {}).get('coins', 0))
                matched = actual in forecast['cash_law']
                self.ctx.log('terminal_cash_audit', actual=actual, supported=matched, **forecast)
                if not matched:
                    self.terminal_misses += 1
                    self.terminal_enabled = False
                    self.evidence.append({'cash_failure':{'terminal_forecast':forecast,'actual':actual}})
                    self.review_due = True
        recovery_override = bool(self.override and self.override[0] == obs.get('decision_id'))
        default_action = super()._pick(obs)
        if recovery_override:
            return default_action
        prompt, options = self._options(obs)
        default_ids = default_action.split()[2:]
        default = next((c for oid,c in options if oid in default_ids), None)
        try:
            rows = self._terminal_table(obs)
        except Exception as exc:
            self.evidence.append({'cash_failure':{'terminal_table':type(exc).__name__, 'reason':str(exc)}})
            self.review_due = True
            self.terminal_enabled = False
            return default_action
        if not rows:
            return default_action
        self.last_terminal_table = rows
        choice = self._review_terminal(obs, rows, default)
        if self.ctx.done:
            return default_action
        if not self.terminal_enabled:
            return default_action
        # Advice may have changed the price of delayed attack value.
        rows = self._terminal_table(obs)
        baseline = next((r for r in rows if r['card'] == default), rows[0])
        selected = max(rows, key=lambda r:(r['utility'], r['card'] == default))
        # Retain the baseline for near ties: this is a conservative local repair.
        if selected['utility'] < baseline['utility'] + 0.35:
            selected = baseline
        if choice in {r['card'] for r in rows}:
            selected = next(r for r in rows if r['card'] == choice)
        self.ctx.log('terminal_choice', decision_id=obs.get('decision_id'), baseline=default,
                     selected=selected['card'], model_choice=choice, alternatives=rows)
        self.terminal_forecast = {'turn':int(obs['turns_taken']['agent']), 'card':selected['card'],
                                  'cash_law':selected['cash_law']}
        oid = next(oid for oid,c in options if c == selected['card'])
        return 'choose ' + str(obs['decision_id']) + ' ' + oid


def build(ctx):
    return TerminalHarness(ctx)
