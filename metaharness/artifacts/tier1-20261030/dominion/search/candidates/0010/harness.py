"""Compact-plan Dominion controller.

The model chooses a kingdom strategy and acquisition order; this module owns
legal option IDs, routine card choices, and recovery from stale plans.
"""
from __future__ import annotations

import json
import re
import concurrent.futures


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
        # Keep inference off the critical path.  The previous leader spent a
        # fixed 200 s waiting for a plan that never arrived; local play should
        # continue while a bounded request is polled opportunistically.
        self.pending_review = None
        self.pending_review_obs = None

    def _remaining(self):
        try:
            return self.ctx.budget.remaining()
        except Exception:
            return {}

    def _review(self, obs, evidence="scheduled review"):
        ctx = self.ctx
        remaining = self._remaining()
        if getattr(ctx, "system2", None) is None or ctx.done or self.planning_disabled or self.reviews >= 8:
            return
        if self.pending_review is not None and not self.pending_review.done():
            return
        if remaining.get("system2_calls", 0) <= 1 or remaining.get("wall_s", 0) < 120:
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
            # Do not block a tier-1 game on backend queue latency.  The future
            # is consumed only when complete, and its direct option IDs are
            # still checked against the decision that produced this snapshot.
            self.pending_review_obs = dict(obs)
            self.pending_review = ctx.spawn(lambda: ctx.think(prompt, session_id=None, timeout_s=180.0))
            ctx.log("strategy_review_started", evidence=evidence)
        except Exception as exc:
            self.pending_review = self.pending_review_obs = None
            self.evidence.append({"model_failure": type(exc).__name__})

    def _poll_review(self, obs):
        future = self.pending_review
        if future is None or not future.done():
            return
        snapshot = self.pending_review_obs or {}
        self.pending_review = self.pending_review_obs = None
        try:
            result = future.result()
            # Strategy fields may still be useful after a turn boundary, but
            # a direct option list is valid only for the exact prompt that
            # was shown when the request was made.
            parse_obs = obs if snapshot.get("decision_id") == obs.get("decision_id") else dict(obs)
            candidate = self._parse_plan(getattr(result, "text", ""), parse_obs)
            if snapshot.get("decision_id") != obs.get("decision_id"):
                self.override = None
            if candidate:
                self.plan = candidate
                self.ctx.memory["plan"] = candidate
                self.ctx.log("strategy_plan", plan=self.plan)
            else:
                self.ctx.log("invalid_plan", text=str(getattr(result, "text", ""))[:150])
        except Exception as exc:
            self.evidence.append({"model_failure": type(exc).__name__})

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
        if card == "Duchy": return 112 if p <= 3 else -5
        # Once only two Provinces remain, a spare $2 buy is worth a point.
        # The leader's trace at seed 1400 passed exactly this opportunity.
        if card == "Estate": return 98 if p <= 2 else (35 if p == 3 else -10)
        if card == "Curse": return -100
        if card == "Copper": return 5 if mode == "gardens" and own.get("Gardens",0)>=3 else -20
        if card == "Gold": return 88
        if card == "Silver": return 61 if money < 18 else 48
        if card == "Gardens":
            current = own.get("Gardens", 0) * (n // 10)
            after = (own.get("Gardens", 0) + 1) * ((n + 1) // 10)
            delta = after - current
            return 104 if p <= 2 and delta >= 2 else (92 if mode=="gardens" and n >= 17 else -5)
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
            self._poll_review(obs)
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


def build(ctx):
    return Harness(ctx)
