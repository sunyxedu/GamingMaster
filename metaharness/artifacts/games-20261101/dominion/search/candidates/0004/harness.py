"""Compact-plan Dominion controller.

The model chooses a kingdom strategy and acquisition order; this module owns
legal option IDs, routine card choices, and recovery from stale plans.
"""
from __future__ import annotations

import json
import re
import concurrent.futures
import math
import random
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

    def _remaining(self):
        try:
            return self.ctx.budget.remaining()
        except Exception:
            return {}

    def _review(self, obs, evidence="scheduled review"):
        ctx = self.ctx
        remaining = self._remaining()
        if getattr(ctx, "system2", None) is None or ctx.done or self.planning_disabled or self.reviews >= 12:
            return
        if remaining.get("system2_calls", 0) <= 1 or remaining.get("wall_s", 0) < 250:
            return
        view = {
            "kingdom": obs.get("kingdom", []),
            "supply": obs.get("supply", {}),
            "agent": obs.get("self", {}), "opponent": obs.get("opponent", {}),
            "turns": obs.get("turns_taken", {}), "resources": obs.get("resources", {}),
            "derived": obs.get("derived", {}), "recent_events": obs.get("public_log_delta", [])[-12:],
            "prompt": obs.get("prompt", {}), "progress": self._safe_progress(),
            "recent_buys": self.recent_buys[-8:], "failures": self.evidence[-6:],
        }
        prompt = (
            "You are the strategic planner for Dominion. Maximize FINAL VP MARGIN against BMU-1. "
            "BMU buys Province at deck treasure money>15; Duchy at provinces<5; Estate at <3; "
            "then Gold; Duchy at <7; Smithy if count < treasure cards/11; then Silver. "
            "It never buys attacks, defense or trashers. Witch can decisively slow it. "
            "Choose money with few terminal actions or a coherent engine; Gardens needs fast gains "
            "and an explicit three-pile finish. Avoid slow speculative engines against this race. "
            "Card text and current options below are authoritative.\n"
            "Return ONLY compact JSON with keys: mode (money/engine/witch/gardens), "
            "priority (ordered list of up to 7 card names), targets (desired TOTAL owned counts "
            "for those support cards), duchy_at (Provinces remaining threshold 1-5), "
            "estate_at (threshold 0-3), note (one sentence), and optionally option_ids "
            "(exact selection for CURRENT prompt only, distinct current IDs within min/max). "
            "Executor automatically buys Provinces, prioritizes Gold unless support targets "
            "need repair, stops buying cards at support targets, and gives priority list "
            "a modest boost. It sequences +Actions before terminals, trashes junk while "
            "preserving deck treasure money>=4, and handles nested effects locally. "
            "Set targets thoughtfully: e.g. one Chapel, one Militia, 1-2 draw terminals "
            "in money; engines need action capacity and draw balanced with payload. "
            "If evidence shows a failed step, include option_ids to repair CURRENT prompt. "
            "Keep output below 220 tokens.\n"
            f"Evidence: {evidence}. Previous plan: {json.dumps(self.plan)}. "
            f"State: {json.dumps(view, separators=(',', ':'))}"
        )
        self.reviews += 1
        self.last_review_turn = int((obs.get("turns_taken") or {}).get("agent", 0))
        self.review_due = False
        try:
            # Independent wait protects the main loop even if backend timeout enforcement is late.
            future = ctx.spawn(lambda: ctx.think(prompt, session_id=None, timeout_s=180.0))
            result = future.result(timeout=200.0)
            text = getattr(result, "text", "")
            candidate = self._parse_plan(text, obs)
            if candidate:
                self.plan = candidate
                ctx.memory["plan"] = candidate
                ctx.log("strategy_plan", plan=self.plan, evidence=evidence)
            else:
                ctx.log("invalid_plan", text=str(text)[:150])
        except concurrent.futures.TimeoutError:
            self.planning_disabled = True
            ctx.log("strategy_wait_cutoff", seconds=200, fallback="finish with local policy")
        except Exception as exc:
            self.evidence.append({"model_failure": type(exc).__name__})
            try:
                ctx.log("strategy_review_failed", error=type(exc).__name__)
            except Exception:
                pass

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
                if name in kingdom and isinstance(value, (int, float)) and not isinstance(value, bool):
                    targets[name] = max(0, min(8, int(value)))
        result = {"mode": mode, "priority": priority[:7], "targets": targets,
                  "note": str(data.get("note", ""))[:180]}
        for key, default, limit in (("duchy_at", 3, 5), ("estate_at", 1, 3)):
            value = data.get(key, default)
            result[key] = max(0, min(limit, int(value))) if isinstance(value, (int, float)) else default
        return result

    def _safe_progress(self):
        try:
            p = self.ctx.progress()
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

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
        money = sum(own.get(c, 0)*v for c,v in (("Copper",1),("Silver",2),("Gold",3),("Festival",2),("Market",1),("Militia",2)))
        defaults = {"Chapel":1, "Sentry":1, "Militia":1, "Witch":2 if engine else 1,
                    "Smithy":max(1, n//14) if not engine else max(2, villages+1),
                    "Village":max(0, actions-1) if engine else 0,
                    "Festival":max(1, actions-1) if engine else 1,
                    "Laboratory":4 if engine else 2, "Market":4 if engine else 1,
                    "Cellar":1, "Moat":0, "Merchant":1, "Mine":0, "Remodel":0,
                    "Workshop":4 if mode=="gardens" else 0}
        targets = dict(defaults)
        targets.update(self.plan.get("targets", {}))
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
        own = obs.get("self",{}).get("owned",{})
        p = obs.get("derived",{}).get("provinces_left",8)
        money = sum(own.get(c,0)*v for c,v in (("Copper",1),("Silver",2),("Gold",3),("Festival",2),("Market",1),("Militia",2)))
        selected = []
        for oid, c in sorted(options,key=lambda x:{"Curse":0,"Estate":1,"Copper":2}.get(x[1],5)):
            good = c == "Curse" or (c == "Estate" and p>2)
            if c == "Copper" and p>2 and money>4:
                good = True
                money -= 1
            if good: selected.append(oid)
            if len(selected)>=hi: break
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

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.done:
            return
        obs = _obs(game.observe())
        while not ctx.done and not game.done:
            turn = int((obs.get("turns_taken") or {}).get("agent", 0))
            prov = int((obs.get("derived") or {}).get("provinces_left", 8))
            race = 0 if prov>5 else (1 if prov>3 else (2 if prov>1 else 3))
            if self.last_race is not None and race != self.last_race:
                self.review_due = True
                self.evidence.append({"race_transition": [self.last_race,race], "provinces":prov})
            self.last_race = race
            prompt = obs.get("prompt") or {}
            unknown = prompt.get("kind") not in {"buy","gain","play_action","trash","discard","reaction","yes_no","topdeck","order"}
            if unknown and obs.get("decision_id") != self.last_decision:
                self.review_due = True
                self.evidence.append({"unfamiliar_prompt":prompt})
            if self.review_due or turn >= self.last_review_turn + 6:
                self._review(obs, str(self.evidence[-3:]) if self.evidence else "kingdom planning / six-turn review")
                self.review_due = False
            action = self._pick(obs)
            if (obs.get("decision_id"),action) in self.bad_actions:
                p,opts = self._options(obs)
                lo = int(p.get("min",0))
                alternatives = ["choose " + str(obs.get("decision_id")) + (" " + oid if lo<=1 else " " + " ".join(x[0] for x in opts[:lo])) for oid,_ in opts]
                if lo==0: alternatives.insert(0,"choose " + str(obs.get("decision_id")))
                action = next((a for a in alternatives if (obs.get("decision_id"),a) not in self.bad_actions), action)
            self.last_decision = obs.get("decision_id")
            if prompt.get("kind") == "buy":
                chosen = action.split()[2:]
                label = next((x["label"] for x in prompt.get("options",[]) if str(x["id"]) in chosen), "pass")
                self.recent_buys.append({"turn":turn,"coins":obs.get("resources",{}).get("coins"),"buy":label})
            before_id = obs.get("decision_id")
            before_illegal = int(obs.get("illegal_action_count", 0) or 0)
            prior_obs = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            try:
                nxt = ctx.act(action)
            except Exception as exc:
                if ctx.done:
                    return
                try:
                    ctx.log("action_rejected", action=action, error=type(exc).__name__)
                except Exception:
                    pass
                self.review_due = True
                try:
                    nxt = ctx.observe()
                except Exception:
                    return
            obs = _obs(nxt if nxt is not None else ctx.observe())
            progress = self._safe_progress()
            now_illegal = int(obs.get("illegal_action_count", progress.get("illegal_actions", before_illegal)) or 0)
            new_id = obs.get("decision_id")
            now_obs = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            if now_illegal > before_illegal or (new_id == before_id and now_obs == prior_obs):
                self.review_due = True
                self.stalled += 1
                self.bad_actions.add((before_id,action))
                self.evidence.append({"failed_action":action,"last_error":obs.get("last_error"),"progress":progress})
            else:
                self.stalled = 0
            # progress() is authoritative for accepted/rejected actions; check it after every step.
            if progress.get("illegal_actions", 0) > self.last_illegal:
                self.review_due = True
            self.last_illegal = int(progress.get("illegal_actions", self.last_illegal) or 0)
            if self.stalled >= 2:
                self.review_due = True
                # Make one fresh legal response even if the plan was the source of the stall.
                self.stalled = 0



class RaceHarness(Harness):
    """Keep the leader's opening, search complete purchase bundles near the end.

    Cost, piles and points are exact. Future-hand samples are explicitly a
    forecast using public multisets, never an inferred hidden order.
    """
    def __init__(self, ctx):
        super().__init__(ctx)
        self.rate_cache = {}
        self.pending_review = None
        self.pending_obs = None
        self.stale_reviews = []
        self.last_forecast = None
        self.forecast_error = 0
        self.last_turn_checked = -1
        self.race_horizon_scale = 1.0
        self.last_search = []
        self.expected_purchase = None
        self.last_review_prov = 9

    @staticmethod
    def vp(owned):
        n = sum(owned.values())
        return (owned.get('Estate', 0) + 3*owned.get('Duchy', 0)
                + 6*owned.get('Province', 0) - owned.get('Curse', 0)
                + owned.get('Gardens', 0)*(n//10))

    def _consume_review(self, obs):
        f = self.pending_review
        if f is None or not f.done():
            return
        self.pending_review = None
        try:
            result = f.result()
            raw = getattr(result, 'text', '')
            data = None
            for m in re.finditer(r'\{', raw):
                try:
                    d, _ = json.JSONDecoder().raw_decode(raw[m.start():])
                    if isinstance(d, dict):
                        data = d
                        break
                except Exception:
                    continue
            if not data:
                self.ctx.log('invalid_race_review', text=str(raw)[:160])
                return
            scale = data.get('horizon_scale')
            if isinstance(scale, (int, float)) and math.isfinite(scale):
                self.race_horizon_scale = max(.65, min(1.5, float(scale)))
            # Support plans are accepted only when explicitly provided. The
            # leader's fallback opening is otherwise retained unchanged.
            if data.get('priority'):
                plan_data = {k: v for k, v in data.items() if k != 'option_ids'}
                candidate = self._parse_plan(json.dumps(plan_data), obs)
                if candidate:
                    self.plan = candidate
            ids = data.get('option_ids')
            old = self.pending_obs or {}
            if str(old.get('decision_id')) == str(obs.get('decision_id')) and isinstance(ids, list):
                ids = [str(x) for x in ids]
                p, options = self._options(obs)
                if (len(set(ids)) == len(ids) and set(ids) <= {x[0] for x in options}
                        and int(p.get('min', 0)) <= len(ids) <= int(p.get('max', 1))):
                    self.override = (obs.get('decision_id'), ids)
            self.ctx.memory['race_review'] = data
            self.ctx.log('race_review', review=data)
        except Exception as exc:
            self.ctx.log('race_review_failed', error=type(exc).__name__)

    def _review(self, obs, evidence='scheduled review'):
        self._consume_review(obs)
        turn = int(obs.get('turns_taken', {}).get('agent', 0))
        prov = int(obs.get('derived', {}).get('provinces_left', 8))
        failure = any(k in evidence for k in ('failed_action', 'unfamiliar_prompt', 'forecast_miss', 'purchase_mismatch'))
        if self.reviews >= 4 or self.ctx.done:
            return
        if self.pending_review is not None:
            # A hung opening request must not suppress the failure detector's
            # fresh evidence. Permit one recovery request alongside it, never
            # an unbounded chain of queued work. Stale direct IDs are discarded.
            if not failure or self.stale_reviews:
                return
            self.stale_reviews.append(self.pending_review)
            self.pending_review = None
        # Review at opening, at first endgame, and on evidence of plan failure.
        if self.reviews and not failure and not (prov <= 4 and self.last_review_prov > 4):
            self.last_review_turn = turn
            return
        rem = self._remaining()
        if rem.get('system2_calls', 0) <= 1 or rem.get('wall_s', 0) < 240:
            return
        rates = self._rates(obs.get('self', {}).get('owned', {}), obs, False)
        enemy = self._rates(obs.get('opponent', {}).get('owned', {}), obs, True)
        view = {k: obs.get(k) for k in ('kingdom', 'supply', 'self', 'opponent', 'turns_taken',
                                       'resources', 'derived', 'prompt', 'last_error')}
        view.update(progress=self._safe_progress(), failures=self.evidence[-5:],
                    recent_buys=self.recent_buys[-6:], purchase_candidates=self.last_search[:8],
                    approximate_public_deck_forecast={'agent': rates, 'opponent': enemy})
        prompt = (
            'You advise a Dominion controller against the disclosed BMU-1. Objective is FINAL VP MARGIN. '
            'The reliable fallback buys a little draw/attack/thinning then money and Provinces. '
            'Near the end it enumerates complete legal purchase bundles, exact Gardens/VP/pile arithmetic, '
            'and scores them using a short public-multiset forecast. These forecasts are estimates, '
            'not hidden hands. Correct strategic assumptions when evidence warrants it. '
            'Reply ONLY compact JSON (<150 tokens): horizon_scale (0.65 to 1.5, default 1; '
            'larger means more turns before the game ends), note (one sentence). '
            'Optionally include option_ids for THIS prompt if repairing a failure. '
            'Only if the default opening is clearly wrong, additionally provide mode '
            '(money/engine/witch/gardens), priority (card names) and targets (total owned counts). '
            'Do not overbuild terminal actions. BMU never buys attacks, trashing or defense. '
            'Use observed card text as authority. Evidence: ' + evidence + '\n' +
            json.dumps(view, separators=(',', ':')))
        self.reviews += 1
        self.last_review_turn = turn
        self.last_review_prov = prov
        self.review_due = False
        self.pending_obs = obs
        try:
            self.pending_review = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=None, timeout_s=180.0))
            # Backend queueing can exceed measured model latency. Retain the
            # pending call and finish locally if the independent wait expires.
            try:
                self.pending_review.result(timeout=200.0)
            except concurrent.futures.TimeoutError:
                self.ctx.log('review_wait_cutoff', seconds=200, pending=True)
            except Exception:
                pass
            self._consume_review(obs)
        except Exception as exc:
            self.pending_review = None
            self.ctx.log('review_start_failed', error=type(exc).__name__)

    def _rates(self, owned, obs, opponent=False):
        """Approximate one-turn payload distribution from a public multiset.

        Exact printed draw/action/money effects for the enabled cards; local
        choices approximate trash/cycle priorities. This is an evaluation
        function, not a replacement game simulator or hidden deck estimator.
        """
        supply = obs.get('supply', {})
        prov = int(obs.get('derived', {}).get('provinces_left', 8))
        key = (tuple(sorted(owned.items())), opponent, min(4, prov),
               tuple((c, int(supply.get(c, {}).get('count', 0)))
                     for c in ('Province', 'Duchy', 'Estate', 'Gardens', 'Curse')))
        if key in self.rate_cache:
            return self.rate_cache[key]
        deck0 = [c for c, n in sorted(owned.items()) for _ in range(int(n))]
        if not deck0:
            return {'vp': 0.0, 'province': 0.0, 'coins': 0.0, 'drawn': 5.0}
        rng = random.Random(92371)  # common random numbers, unrelated to game RNG
        values = {'Copper': 1, 'Silver': 2, 'Gold': 3}
        terminal = {'Smithy', 'Witch', 'Militia', 'Moat', 'Chapel', 'Mine', 'Remodel', 'Workshop'}
        cantrip = {'Village', 'Festival', 'Market', 'Laboratory', 'Merchant', 'Sentry', 'Cellar'}
        vp_total = provinces = coins_total = draws_total = 0.0
        for _ in range(96):
            deck = list(deck0)
            rng.shuffle(deck)
            hand = deck[:5]
            deck = deck[5:]
            actions, coins, buys, drawn, merchant, free = 1, 0, 1, len(hand), 0, 0
            def draw(n):
                nonlocal drawn
                got = deck[:n]
                del deck[:n]
                hand.extend(got)
                drawn += len(got)
            for step in range(60):
                if actions <= 0:
                    break
                available = [c for c in hand if c in terminal or c in cantrip]
                if opponent:
                    available = [c for c in available if c == 'Smithy']
                if not available:
                    break
                priority = {'Village': 100, 'Festival': 98, 'Laboratory': 95, 'Market': 93,
                            'Merchant': 92, 'Sentry': 91, 'Cellar': 80, 'Smithy': 60,
                            'Witch': 59, 'Militia': 55, 'Moat': 45, 'Mine': 20,
                            'Workshop': 19, 'Chapel': 0, 'Remodel': 0}
                card = max(available, key=lambda c: priority[c])
                hand.remove(card)
                actions -= 1
                if card == 'Village': actions += 2; draw(1)
                elif card == 'Festival': actions += 2; buys += 1; coins += 2
                elif card == 'Laboratory': actions += 1; draw(2)
                elif card == 'Market': actions += 1; buys += 1; coins += 1; draw(1)
                elif card == 'Merchant': actions += 1; merchant += 1; draw(1)
                elif card == 'Sentry':
                    actions += 1; draw(1)
                    inspect = deck[:2]
                    del deck[:2]
                    deck[:0] = [c for c in inspect if c not in {'Curse', 'Estate', 'Copper', 'Duchy', 'Province', 'Gardens'}]
                elif card == 'Cellar':
                    actions += 1
                    junk = [c for c in hand if c in {'Curse', 'Estate', 'Duchy', 'Province', 'Gardens'}]
                    for c in junk: hand.remove(c)
                    draw(len(junk))
                elif card == 'Smithy': draw(3)
                elif card in {'Witch', 'Moat'}: draw(2)
                elif card == 'Militia': coins += 2
                elif card == 'Mine' and any(c in values and c != 'Gold' for c in hand): coins += 1
                elif card == 'Workshop':
                    if supply.get('Gardens', {}).get('count', 0): free += max(1, (len(deck0)+1)//10)
                    elif prov <= 3 and supply.get('Estate', {}).get('count', 0): free += 1
            coins += sum(values.get(c, 0) for c in hand)
            if 'Silver' in hand: coins += merchant
            coins_total += coins
            draws_total += drawn
            score = 0
            if opponent:
                money = sum(owned.get(c, 0)*v for c, v in values.items())
                if coins >= 8 and money > 15 and supply.get('Province', {}).get('count', 0):
                    score = 6; provinces += 1
                elif coins >= 5 and prov < 5 and supply.get('Duchy', {}).get('count', 0): score = 3
                elif coins >= 2 and prov < 3 and supply.get('Estate', {}).get('count', 0): score = 1
                elif coins < 6 and coins >= 5 and prov < 7 and supply.get('Duchy', {}).get('count', 0): score = 3
            else:
                # A greedy scoring continuation, with actual +Buy resources.
                left = {c: int(supply.get(c, {}).get('count', 0)) for c in ('Province', 'Duchy', 'Gardens', 'Estate')}
                for _b in range(min(buys, 5)):
                    choices = [(6, 8, 'Province'), (3, 5, 'Duchy'),
                               ((len(deck0)+1)//10, 4, 'Gardens'), (1, 2, 'Estate')]
                    affordable = [x for x in choices if x[1] <= coins and left[x[2]] > 0]
                    if not affordable: break
                    value, cost, c = max(affordable, key=lambda x: (x[0], -x[1]))
                    score += value; coins -= cost; left[c] -= 1
                    if c == 'Province': provinces += 1
            vp_total += score + free
        result = {'vp': vp_total/96, 'province': provinces/96,
                  'coins': coins_total/96, 'drawn': draws_total/96}
        self.rate_cache[key] = result
        return result

    def _bundle_choice(self, obs):
        prompt, options = self._options(obs)
        own = obs.get('self', {}).get('owned', {})
        supply = obs.get('supply', {})
        prov = int(obs.get('derived', {}).get('provinces_left', 8))
        empty = sum(1 for p in supply.values() if p.get('count', 0) == 0)
        near = sum(1 for p in supply.values() if p.get('count', 0) <= 2)
        if prov > 4 and near < 3:
            return None
        # Exact VP calculations are used only if they agree with the observed
        # ledger. Unknown scoring rules trigger a review instead of silent error.
        if self.vp(own) != obs.get('self', {}).get('vp', self.vp(own)):
            self.review_due = True
            self.evidence.append({'purchase_mismatch': 'VP model differs from observed ledger'})
            return None
        gain = prompt.get('kind') == 'gain'
        coins = int(obs.get('resources', {}).get('coins', 0))
        buys = 1 if gain else max(0, int(obs.get('resources', {}).get('buys', 1)))
        if buys <= 0:
            return None
        allowed = {c: oid for oid, c in options if c in supply}
        economic = sorted(allowed, key=lambda c: self._buy_rank(c, obs), reverse=True)
        # Search all scoring and pile-ending choices plus the strongest 3 build
        # alternatives. Never approximate costs or card quantities.
        candidates = [c for c in allowed if c in {'Province', 'Duchy', 'Estate', 'Gardens'}
                      or (supply[c]['count'] <= buys and empty >= 1)]
        candidates += [c for c in economic if c not in {'Curse', 'Copper'}][:3]
        if own.get('Gardens', 0) or 'Gardens' in allowed:
            candidates.extend(c for c in allowed if int(supply[c]['cost']) == 0)
        candidates = sorted(set(c for c in candidates if c in allowed))
        # Include any affordable card after the first buy, always a subset of
        # initial legal buys; per-pile counts are checked on every branch.
        bundles = [()]
        def visit(prefix, start, cash):
            if len(prefix) >= min(buys, 5) or len(bundles) >= 1500:
                return
            for i in range(start, len(candidates)):
                c = candidates[i]
                cost = 0 if gain else int(supply[c]['cost'])
                if cost > cash or prefix.count(c) >= supply[c]['count']: continue
                new = prefix + (c,)
                bundles.append(new)
                visit(new, i, cash-cost)
        visit((), 0, coins)
        base = self._rates(own, obs)
        enemy = self._rates(obs.get('opponent', {}).get('owned', {}), obs, True)
        oldvp = self.vp(own)
        margin = float(obs.get('derived', {}).get('vp_margin', 0))
        results = []
        for bundle in bundles:
            if gain and not bundle and int(prompt.get('min', 0)) > 0: continue
            got = Counter(bundle)
            newown = dict(own)
            for c, n in got.items(): newown[c] = newown.get(c, 0)+n
            delta = self.vp(newown)-oldvp
            remainp = prov-got['Province']
            emptied = empty + sum(1 for c, n in got.items() if n == supply[c]['count'])
            end = remainp == 0 or emptied >= 3
            if end:
                value = margin + delta
                horizon = 0
            else:
                future_supply = {c: dict(p) for c, p in supply.items()}
                for c, n in got.items():
                    future_supply[c]['count'] -= n
                future_obs = dict(obs)
                future_obs['supply'] = future_supply
                future_obs['derived'] = dict(obs.get('derived', {}), provinces_left=remainp)
                future_base = self._rates(own, future_obs)
                rate = self._rates(newown, future_obs)
                future_enemy = self._rates(obs.get('opponent', {}).get('owned', {}), future_obs, True)
                speed = max(.4, future_base['province']+future_enemy['province'])
                horizon = min(6.0, max(.5, remainp/speed))*self.race_horizon_scale
                if emptied >= 2: horizon = min(horizon, 2.0)
                # New discard cards may miss the next hand: only part of their
                # estimated deck impact applies before the next shuffle.
                delay = obs.get('self', {}).get('deck_count', 0)/max(5., base['drawn'])
                exposure = max(.2, min(1., (horizon-.5*delay)/max(1., horizon)))
                ours = future_base['vp'] + exposure*(rate['vp']-future_base['vp'])
                # Opponent acts before the next agent turn. With one Province
                # left, its completion probability removes that future turn.
                opponent_first = future_enemy['province'] if remainp == 1 else .15
                value = margin+delta+horizon*(ours-future_enemy['vp'])-opponent_first*ours
            # Preserve a tiny preference for useful economy on true ties, never
            # enough to offset even one immediate VP in a terminal bundle.
            tie = sum(max(0, self._buy_rank(c, obs)) for c in bundle)/10000
            results.append((value, tie, bundle, delta, end, horizon))
        if not results: return None
        results.sort(key=lambda x: (x[0], x[1], -len(x[2])), reverse=True)
        self.last_search = [{'cards': list(x[2]), 'projected_margin': round(x[0], 2),
                             'immediate_vp': x[3], 'ends': x[4], 'horizon': round(x[5], 2)}
                            for x in results[:8]]
        best = results[0]
        try:
            self.ctx.log('purchase_search', choices=self.last_search, own_rate=base, opponent_rate=enemy)
        except Exception:
            pass
        bundle = best[2]
        if not bundle: return 'choose ' + str(obs.get('decision_id'))
        # Execute only one purchase. The next prompt triggers a fresh search,
        # preserving exact legality and allowing all remaining buys after an end.
        first = max(bundle, key=lambda c: (c == 'Province', self.vp({c: 1}), int(supply[c]['cost'])))
        self.expected_purchase = {'card': first, 'count': own.get(first, 0)+1,
                                  'supply': int(supply[first]['count'])-1}
        return 'choose ' + str(obs.get('decision_id')) + ' ' + allowed[first]

    def _pick(self, obs):
        self._consume_review(obs)
        turn = int(obs.get('turns_taken', {}).get('agent', 0))
        if self.expected_purchase:
            e = self.expected_purchase
            got = obs.get('self', {}).get('owned', {}).get(e['card'], 0)
            if got < e['count'] and not obs.get('last_error'):
                self.review_due = True
                self.evidence.append({'purchase_mismatch': e, 'observed_count': got})
            self.expected_purchase = None
        if self.override and self.override[0] == obs.get('decision_id'):
            return super()._pick(obs)
        prompt = obs.get('prompt') or {}
        if prompt.get('kind') in {'buy', 'gain'} and prompt.get('source') != 'Mine':
            if turn != self.last_turn_checked:
                if self.last_forecast and prompt.get('kind') == 'buy':
                    actual = int(obs.get('resources', {}).get('coins', 0))
                    predicted = self.last_forecast.get('coins', 0)
                    self.forecast_error = self.forecast_error+1 if abs(actual-predicted) >= 4 else 0
                    if self.forecast_error >= 2:
                        self.review_due = True
                        self.evidence.append({'forecast_miss': {'expected_coins': round(predicted, 2),
                                                               'actual_coins': actual, 'turn': turn}})
                        self._review(obs, str(self.evidence[-3:]))
                        self.forecast_error = 0
                self.last_turn_checked = turn
                self.last_forecast = self._rates(obs.get('self', {}).get('owned', {}), obs)
            choice = self._bundle_choice(obs)
            if choice is not None:
                return choice
        return super()._pick(obs)


def build(ctx):
    return RaceHarness(ctx)
