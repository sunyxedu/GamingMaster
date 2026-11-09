"""Single-turn public-zone tactical search supervised by Astra.

Derived from 0001's local executor. Public-zone bookkeeping is adapted from
0011, but no purchase rollout, hidden game state, or full-game search is used.
"""
from __future__ import annotations

import json
import re
import concurrent.futures
import itertools
import math
import random
import time
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


class LocalPolicy:
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


def points(owned):
    return owned.get("Estate",0)+3*owned.get("Duchy",0)+6*owned.get("Province",0)-owned.get("Curse",0)+owned.get("Gardens",0)*(sum(owned.values())//10)

def expand(counts):
    return [c for c,n in sorted(counts.items()) for _ in range(max(0,int(n)))]

TREASURE = {"Copper":1, "Silver":2, "Gold":3}
ACTIONS = {"Cellar", "Chapel", "Festival", "Laboratory", "Market", "Merchant",
           "Militia", "Mine", "Moat", "Remodel", "Sentry", "Smithy", "Village", "Witch", "Workshop"}


class Player:
    def __init__(self, owned, deck, hand, discard, played=()):
        self.owned = Counter(owned)
        self.deck = list(deck)
        self.hand = list(hand)
        self.discard = list(discard)
        self.played = list(played)

    def draw(self, n, rng):
        out = []
        for _ in range(n):
            if not self.deck:
                if not self.discard:
                    break
                self.deck, self.discard = self.discard, []
                rng.shuffle(self.deck)
            out.append(self.deck.pop())
        return out

    def cleanup(self, rng):
        self.discard.extend(self.hand)
        self.discard.extend(self.played)
        self.hand, self.played = self.draw(5, rng), []


class TacticalWorld:
    """Single-turn estimator of disclosed rules, not an engine import.

    Agent zones are conditioned on public multisets. Opponent unknown zones are
    an exchangeable approximation. Own choices share the real local executor.
    """
    def __init__(self, obs, policy, seed):
        self.policy = policy
        self.rng = [random.Random(seed), random.Random(seed ^ 0x6A09E667)]
        self.supply = {c:dict(d) for c,d in obs['supply'].items()}
        self.turns = dict(obs.get('turns_taken', {}))
        me = obs['self']
        unseen = Counter(me['owned'])
        for zone in ('hand','discard','in_play'):
            unseen.subtract(me.get(zone, {}))
        if any(n < 0 for n in unseen.values()) or sum(unseen.values()) != me['deck_count']:
            raise ValueError('own-zone conservation does not match deck_count')
        deck = expand(unseen)
        self.rng[0].shuffle(deck)
        own = Player(me['owned'], deck, expand(me.get('hand',{})),
                     expand(me.get('discard',{})), expand(me.get('in_play',{})))
        enemy = obs['opponent']
        pool = Counter(enemy['owned'])
        pool.subtract(enemy.get('in_play', {}))
        hidden = expand(pool)
        self.rng[1].shuffle(hidden)
        nh, nd = enemy.get('hand_count',5), enemy.get('deck_count',0)
        opp = Player(enemy['owned'], hidden[nh:nh+nd], hidden[:nh], hidden[nh+nd:],
                     expand(enemy.get('in_play',{})))
        self.players = [own,opp]
        self.first_next_coins = None
        self.initial = obs
        self.attack_reward = 0.0
        self.trashed = Counter()
        self.gained = Counter()
        self.gain_history = []
        self.truncated = False

    def ended(self):
        return self.supply.get('Province',{}).get('count',1)<=0 or sum(d['count']==0 for d in self.supply.values())>=3

    def gain(self, who, card, hand=False):
        if card is None or card not in self.supply or self.supply[card]['count'] <= 0:
            return False
        self.supply[card]['count'] -= 1
        p = self.players[who]
        if who == 0: self.gain_history.append((card, Counter(p.owned)))
        p.owned[card] += 1
        if who == 0: self.gained[card] += 1
        (p.hand if hand else p.discard).append(card)
        return True

    def trash(self, p, card):
        p.hand.remove(card)
        p.owned[card] -= 1
        self.trashed[card] += 1

    def obs(self, who, kind, options, source=None, coins=0, buys=1, actions=1, lo=0, hi=1):
        p, e = self.players[who], self.players[1-who]
        return {'decision_id':0, 'supply':self.supply, 'turns_taken':self.turns,
                'resources':{'coins':coins,'buys':buys,'actions':actions},
                'self':{'owned':p.owned,'hand':Counter(p.hand),'discard':Counter(p.discard),
                        'in_play':Counter(p.played),'deck_count':len(p.deck),'discard_count':len(p.discard),
                        'vp':points(p.owned)},
                'opponent':{'owned':e.owned,'vp':points(e.owned)},
                'derived':{'provinces_left':self.supply['Province']['count'],
                           'vp_margin':points(p.owned)-points(e.owned)},
                'prompt':{'kind':kind,'source':source,'min':lo,'max':hi,
                          'options':[{'id':str(i),'label':c} for i,c in enumerate(options)]}}

    def choose(self, who, kind, options, **kwargs):
        o = self.obs(who, kind, options, **kwargs)
        out = self.policy._pick(o).split()[2:]
        return [options[int(i)] for i in out if i.isdigit() and int(i)<len(options)]

    def target(self, who, limit, source=None, treasure=False):
        options = [c for c,d in self.supply.items() if d['count']>0 and d['cost']<=limit
                   and (not treasure or c in TREASURE)]
        if not options:
            return None
        got = self.choose(who,'gain',options,source=source,lo=1)
        return got[0] if got else options[0]

    def turn(self, who=0, first=None, nested=None):
        p, enemy = self.players[who], self.players[1-who]
        label = 'agent' if who==0 else 'opponent'
        resources = self.initial.get('resources', {})
        actions = resources.get('actions', 1)
        coins = resources.get('coins', 0)
        buys = resources.get('buys', 1)
        merchants = p.played.count('Merchant')
        if nested is not None:
            kind, source, cards = nested
            for c in cards:
                if kind == 'trash': self.trash(p, c)
                else: p.hand.remove(c); p.discard.append(c)
            if source == 'Cellar': p.hand.extend(p.draw(len(cards), self.rng[who]))
        for _ in range(80):
            if actions <= 0: break
            options = [c for c in p.hand if c in ACTIONS and (who==0 or c=='Smithy')]
            if not options: break
            if _ == 0 and nested is None:
                if first is None: break
                card = first
            elif who == 1:
                card = 'Smithy'
            else:
                selected = self.choose(0,'play_action',options,actions=actions,coins=coins,buys=buys)
                if not selected: break
                card = selected[0]
            p.hand.remove(card)
            p.played.append(card)
            actions -= 1
            if card == 'Village':
                actions += 2; p.hand.extend(p.draw(1,self.rng[who]))
            elif card == 'Festival':
                actions += 2; coins += 2; buys += 1
            elif card == 'Laboratory':
                actions += 1; p.hand.extend(p.draw(2,self.rng[who]))
            elif card == 'Market':
                actions += 1; coins += 1; buys += 1; p.hand.extend(p.draw(1,self.rng[who]))
            elif card == 'Merchant':
                actions += 1; merchants += 1; p.hand.extend(p.draw(1,self.rng[who]))
            elif card == 'Smithy': p.hand.extend(p.draw(3,self.rng[who]))
            elif card in {'Witch','Moat'}:
                p.hand.extend(p.draw(2,self.rng[who]))
                if card=='Witch' and 'Moat' not in enemy.hand:
                    if self.gain(1-who,'Curse'): self.attack_reward += 1.0
            elif card == 'Militia':
                coins += 2
                if 'Moat' not in enemy.hand:
                    self.attack_reward += 0.65 if len(enemy.hand)>3 else 0
                    count = max(0,len(enemy.hand)-3)
                    # Disclosed BMU default discards first required cards.
                    discarded = enemy.hand[:count]
                    del enemy.hand[:count]
                    enemy.discard.extend(discarded)
            elif card == 'Chapel':
                if hasattr(self, 'chapel_choice'):
                    chosen = self.chapel_choice(p.hand, p.owned, self.supply, coins, buys, self.turns.get('agent',0))
                else:
                    chosen = self.choose(who,'trash',p.hand[:],source=card,hi=min(4,len(p.hand)))
                for c in chosen: self.trash(p,c)
            elif card == 'Cellar':
                actions += 1
                selected = self.choose(who,'discard',p.hand[:],source=card,hi=len(p.hand))
                for c in selected:
                    p.hand.remove(c); p.discard.append(c)
                p.hand.extend(p.draw(len(selected),self.rng[who]))
            elif card == 'Sentry':
                actions += 1; p.hand.extend(p.draw(1,self.rng[who]))
                inspected = p.draw(2,self.rng[who])
                # Policy uses ownership money, not the inspection's position.
                trashed = self.choose(who,'trash',inspected,source=card,hi=len(inspected))
                for c in trashed:
                    inspected.remove(c); p.owned[c] -= 1; self.trashed[c] += 1
                discarded = self.choose(who,'discard',inspected,source=card,hi=len(inspected))
                for c in discarded:
                    inspected.remove(c); p.discard.append(c)
                p.deck.extend(reversed(inspected))
            elif card == 'Workshop':
                self.gain(who,self.target(who,4,card))
            elif card in {'Mine','Remodel'}:
                opts = [c for c in p.hand if c in TREASURE] if card=='Mine' else p.hand[:]
                if opts:
                    got = self.choose(who,'trash',opts,source=card,lo=0 if card=='Mine' else 1)
                    if got:
                        c=got[0]; cost=self.supply[c]['cost']
                        self.trash(p,c)
                        self.gain(who,self.target(who,cost+(3 if card=='Mine' else 2),card,card=='Mine'),card=='Mine')
        coins += sum(TREASURE.get(c,0) for c in p.hand)
        if 'Silver' in p.hand: coins += merchants
        if who==0 and self.first_next_coins is None: self.first_next_coins=coins
        return coins, buys


class Harness(LocalPolicy):
    """Astra supervises a one-turn public-deck tactical executor."""
    def __init__(self, ctx):
        super().__init__(ctx)
        self.local = LocalPolicy(None)
        self.search_enabled = True
        self.attack_weight = 1.0
        self.thin_weight = 1.0
        self.pending = []
        self.forecasts = {}
        self.rows = []
        self.last_tactics = []
        self.cash_history = []
        self.misses = 0
        self.reviewed = set()
        self.need_review = None
        self.calls = 0
        self.revision = 0

    def log(self, event, **fields):
        try: self.ctx.log(event, **fields)
        except Exception: pass

    def valid(self, o, ids):
        p = o.get('prompt') or {}
        legal = {str(x['id']) for x in p.get('options', [])}
        return (isinstance(ids, list) and len(ids) == len(set(ids)) and
                p.get('min', 0) <= len(ids) <= p.get('max', 1) and all(i in legal for i in ids))

    def consume(self, o):
        for future, did in list(self.pending):
            if not future.done(): continue
            self.pending.remove((future, did))
            try:
                result = future.result()
                raw = getattr(result, 'text', '')
                data = None
                for m in re.finditer(r'\{', raw):
                    try:
                        data, _ = json.JSONDecoder().raw_decode(raw[m.start():])
                        if isinstance(data, dict): break
                    except Exception: continue
                if not isinstance(data, dict):
                    self.log('malformed_review', text=str(raw)[:150]); continue
                for key in ('attack_weight', 'thin_weight'):
                    value = data.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        setattr(self, key, max(0.2, min(2.0, float(value))))
                if isinstance(data.get('search'), bool): self.search_enabled = data['search']
                # Strategic names may survive a late reply; occurrence IDs may not.
                plan_data = data.get('plan')
                if isinstance(plan_data, dict):
                    plan_data = dict(plan_data); plan_data.pop('option_ids', None)
                    plan = self._parse_plan(json.dumps(plan_data), o)
                    if plan: self.plan = plan; self.local.plan = dict(plan)
                ids = data.get('option_ids')
                if isinstance(ids, list):
                    ids = [str(x) for x in ids]
                    if did == o.get('decision_id') and self.valid(o, ids): self.override = (did, ids)
                self.revision += 1
                self.ctx.memory['tactical_plan'] = {'plan': self.plan, 'attack_weight': self.attack_weight,
                                                  'thin_weight': self.thin_weight, 'search': self.search_enabled}
                self.log('tactical_review', decision_id=did, revision=self.revision,
                         settings=self.ctx.memory['tactical_plan'], note=str(data.get('note', ''))[:200])
            except Exception as exc:
                self.log('review_failure', error=type(exc).__name__)

    def review(self, o, reason):
        remaining = self._remaining()
        recovery = reason.startswith('failure')
        if self.ctx.done or self.calls >= 4 or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 370:
            return
        if self.pending and (not recovery or len(self.pending) >= 2): return
        self.calls += 1
        self.last_review_turn = o.get('turns_taken', {}).get('agent', 0)
        view = {k: o.get(k) for k in ('decision_id','kingdom','supply','self','opponent','resources','derived','turns_taken','prompt')}
        view['progress'] = self._safe_progress()
        view['current_plan'] = self.plan
        view['evidence'] = self.evidence[-8:]
        view['recent_buys'] = self.recent_buys[-8:]
        view['tactical_alternatives'] = self.rows
        view['last_tactical_decision'] = self.last_tactics
        prompt = (
            'Plan Dominion against the disclosed BMU-1 money/Smithy opponent. Maximize final VP margin. '
            'This experiment uses one-turn public-deck sampling for Action choices and Chapel/Cellar subsets. '
            'It preserves actual own zones but randomizes unknown order. Future local action choices use a fixed '
            'priority; estimates are NOT full-game forecasts. Utility combines affordable buys, early thinning, '
            'and attack bonuses. cash_mean/range predicts this turn\'s FIRST buy prompt; do not compare utility '
            'with current VP or terminal margin. Attack effects and strategic weights are approximations. '
            'Purchases use a money baseline: Province > late Duchy > Gold > capped useful supports > Silver. '
            'Support priority grants only a small boost; targets mean TOTAL owned count. A coherent engine '
            'requires action capacity, draw, payload and time. Witch is strong because BMU never trashes. '
            'Never interpret loss of Estate VP while thinning as strategic collapse by itself. '
            'Inspect evidence and revise tactical weights or disable search if it is unreliable. '
            'Return only JSON below 200 tokens: {"search":true,"attack_weight":1,"thin_weight":1,'
            '"plan":{"mode":"money","priority":["Militia","Smithy"],"targets":{"Militia":1,"Smithy":1},'
            '"duchy_at":3,"estate_at":1},"note":"short reasoning"}. '
            'Optionally include option_ids for the current prompt only to repair a failure. '
            'Weight ranges .2 to 2. Current weights: '+str((self.attack_weight,self.thin_weight))+
            '. Reason: '+reason+'. State: '+json.dumps(view,separators=(',',':')))
        try:
            future = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=None, timeout_s=300.0))
            self.pending.append((future, o.get('decision_id')))
            try: future.result(timeout=min(320.0, max(1.0, remaining['wall_s']-40)))
            except concurrent.futures.TimeoutError: self.log('review_wait_cutoff', seconds=320)
            except Exception: pass
            self.consume(o)
        except Exception as exc: self.log('review_start_failure', error=type(exc).__name__)

    def card_utility(self, c, owned, supply, turn):
        p = supply.get('Province', {}).get('count', 8)
        n = sum(owned.values())
        if c == 'Province': return 6.5
        if c == 'Duchy': return 3.5 if p <= 4 else 0.8
        if c == 'Estate': return 1.2 if p <= 2 else (-0.8 if p>4 else 0.1)
        if c == 'Gardens': return (n+1)//10 + owned.get('Gardens',0)*int((n+1)%10==0) - (0.2 if p<=3 else 1.0)
        if c == 'Curse': return -2
        if c == 'Copper': return -0.3
        if c == 'Gold': return 4.2 if p>3 else 2.4
        if c == 'Silver': return 2.5 if p>3 else 0.7
        # Support value obeys the same acquisition caps as the executable plan.
        fake = {'self':{'owned':owned},'supply':supply,'turns_taken':{'agent':turn},
                'derived':{'provinces_left':p}}
        rank = self.local._buy_rank(c, fake)
        return max(-0.5, min(4.5, rank / 24 - 0.5))

    def buy_value(self, owned, supply, cash, buys, turn):
        # Value the purchases the real local continuation will actually make.
        # This avoids crediting an optimal imaginary bundle that the executor
        # would replace with a different support card at the very next prompt.
        cash=max(0,int(cash));buys=min(6,int(buys));owned=Counter(owned)
        supply={c:dict(d) for c,d in supply.items()};value=0.0
        for _ in range(buys):
            available=[c for c,d in supply.items() if d['count']>0 and d['cost']<=cash]
            if not available:break
            view={'self':{'owned':owned},'supply':supply,'turns_taken':{'agent':turn},
                  'derived':{'provinces_left':supply.get('Province',{}).get('count',8)}}
            card=max(available,key=lambda c:self.local._buy_rank(c,view))
            if self.local._buy_rank(card,view)<=0:break
            value+=self.card_utility(card,owned,supply,turn)
            cash-=supply[card]['cost'];supply[card]['count']-=1;owned[card]+=1
        return value+min(0.15,0.01*cash)

    def thinning_value(self, trashed, provinces):
        early={'Curse':1.6,'Estate':1.05,'Copper':0.55}
        late={'Curse':1.1,'Estate':-1,'Copper':0.05}
        return self.thin_weight*sum(n*(early if provinces>3 else late).get(c,-1) for c,n in trashed.items())

    def chapel_choice(self, hand, owned, supply, coins, buys, turn):
        # Root Chapel play must anticipate a sensible nested trash decision,
        # rather than rejecting Chapel because the old executor overtrashes.
        counts=Counter(hand);best=(-float('inf'),[])
        money=sum(owned.get(c,0)*v for c,v in TREASURE.items())
        cash=coins+sum(counts[c]*v for c,v in TREASURE.items())
        provinces=supply['Province']['count']
        for curse in range(min(4,counts['Curse'])+1):
            for estate in range(min(4-curse,counts['Estate'])+1):
                for copper in range(min(4-curse-estate,counts['Copper'],max(0,money-4))+1):
                    trashed=Counter({'Curse':curse,'Estate':estate,'Copper':copper})
                    after=Counter(owned);after.subtract(trashed)
                    value=self.buy_value(after,supply,cash-copper,buys,turn)+self.thinning_value(trashed,provinces)
                    if value>best[0]:best=(value,expand(trashed))
        return best[1]

    def choices(self, o, baseline):
        p=o['prompt']; kind=p['kind']; src=p.get('source'); opts=p['options']
        base=baseline.split()[2:]
        candidates=[base]
        if kind in {'action','play_action'}:
            seen=set()
            for x in opts:
                if x['label'] not in seen: candidates.append([str(x['id'])]); seen.add(x['label'])
            if p['min']==0:candidates.append([])
        elif kind=='trash' and src=='Chapel':
            # Enumerate distinct junk multisets, preserving at least $4 of total payload.
            relevant=[x for x in opts if x['label'] in {'Copper','Estate','Curse'}]
            money=sum(o['self']['owned'].get(c,0)*v for c,v in TREASURE.items())
            seen=set()
            for n in range(0,min(4,len(relevant))+1):
                for combo in itertools.combinations(relevant,n):
                    labels=tuple(sorted(x['label'] for x in combo))
                    if labels in seen:continue
                    seen.add(labels)
                    if labels.count('Copper')>max(0,money-4):continue
                    if o['derived']['provinces_left']<=2 and ('Estate' in labels or 'Copper' in labels):continue
                    candidates.append([str(x['id']) for x in combo])
        elif kind=='discard' and src=='Cellar':
            # Up to one subset per distinct multiset; hand sizes can grow in engines.
            groups={}
            for x in opts:groups.setdefault(x['label'],[]).append(str(x['id']))
            seen=0
            for amounts in itertools.product(*(range(len(ids)+1) for ids in groups.values())):
                ids=[i for count,group in zip(amounts,groups.values()) for i in group[:count]]
                candidates.append(ids); seen+=1
                if seen>=160:break
        unique=[]
        for ids in candidates:
            if ids not in unique and self.valid(o,ids):unique.append(ids)
        return unique

    def search(self,o):
        self.local.plan=dict(self.plan)
        baseline=self.local._pick(o)
        self.rows=[]
        p=o['prompt'];kind=p['kind'];src=p.get('source')
        eligible=kind in {'action','play_action'} or (kind=='trash' and src=='Chapel') or (kind=='discard' and src=='Cellar')
        if not eligible or not self.search_enabled or self._remaining().get('wall_s',0)<25:return baseline
        choices=self.choices(o,baseline)
        if len(choices)<=1:return baseline
        labels={str(x['id']):x['label'] for x in p['options']}
        turn=o.get('turns_taken',{}).get('agent',0)
        prov=o.get('derived',{}).get('provinces_left',8)
        results=[[] for _ in choices]; cashes=[[] for _ in choices]
        started=time.monotonic()
        for sample in range(96):
            if self.ctx.done:return baseline
            if sample>=24 and time.monotonic()-started>4:break
            for i,ids in enumerate(choices):
                world=TacticalWorld(o,self.local,7919+sample*104729)
                world.chapel_choice = self.chapel_choice
                selected=[labels[x] for x in ids]
                nested=(kind,src,selected) if kind not in {'action','play_action'} else None
                cash,buys=world.turn(first=selected[0] if selected and nested is None else None,nested=nested)
                own=world.players[0].owned
                value=self.buy_value(own,world.supply,cash,buys,turn)
                thin=self.thinning_value(world.trashed,prov)
                gained=sum(self.card_utility(c,before,world.supply,turn) for c,before in world.gain_history)
                value+=thin+self.attack_weight*world.attack_reward+gained
                results[i].append(value);cashes[i].append(cash)
        base=results[0]; n=len(base);rows=[]
        for i,ids in enumerate(choices):
            delta=[v-b for v,b in zip(results[i],base)];lift=sum(delta)/n
            se=math.sqrt(sum((v-lift)**2 for v in delta)/max(1,n-1)/n)
            rows.append({'ids':ids,'cards':[labels[x] for x in ids],'utility':round(sum(results[i])/n,3),
                         'paired_lift':round(lift,3),'se':round(se,3),'selection_value':lift-0.7*se,
                         'cash_mean':round(sum(cashes[i])/n,2),'cash_range':[min(cashes[i]),max(cashes[i])],'n':n})
        best=max(range(len(rows)),key=lambda i:(rows[i]['selection_value'],i==0))
        self.rows=sorted(rows,key=lambda r:r['selection_value'],reverse=True)[:8]
        self.last_tactics={'decision_id':o['decision_id'],'turn':turn,'alternatives':self.rows}
        self.forecasts[turn]={'mean':sum(cashes[best])/n,'low':min(cashes[best]),'high':max(cashes[best]),'decision_id':o['decision_id']}
        self.log('turn_tactics',decision_id=o['decision_id'],baseline=rows[0]['cards'],selected=rows[best]['cards'],
                 alternatives=self.rows,seconds=round(time.monotonic()-started,3))
        ids=choices[best]
        return 'choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')

    def final_buy(self,o,default):
        if o['prompt']['kind']!='buy':return default
        supply=o['supply'];own=Counter(o['self']['owned'])
        if supply['Province']['count']>0 and sum(d['count']==0 for d in supply.values())<3:return default
        cash=o['resources']['coins'];buys=o['resources']['buys'];opts={x['label']:str(x['id']) for x in o['prompt']['options']}
        options=[c for c in opts if c in {'Province','Duchy','Estate','Gardens','Copper'}]
        best=(points(own),[]);used=Counter()
        def rec(start,left,count,bundle):
            nonlocal best
            if points(own)>best[0]:best=(points(own),list(bundle))
            if count<=0:return
            for i in range(start,len(options)):
                c=options[i];cost=supply[c]['cost']
                if cost>left or used[c]>=supply[c]['count']:continue
                used[c]+=1;own[c]+=1
                rec(i,left-cost,count-1,bundle+[c])
                own[c]-=1;used[c]-=1
        rec(0,cash,min(buys,10),[])
        ids=[opts[best[1][0]]] if best[1] else []
        return 'choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')

    def audit(self,before,after,action):
        progress=self._safe_progress();did=before.get('decision_id')
        failed=(progress.get('illegal_actions',0)>self.last_illegal or after.get('illegal_action_count',0)>before.get('illegal_action_count',0)
                or (after.get('decision_id')==did and not after.get('terminal')) or progress.get('since_obs_change',0)>=2)
        self.last_illegal=progress.get('illegal_actions',0)
        if failed:
            self.bad_actions.add((did,action));self.need_review='failure: action rejected or observation did not advance'
            self.evidence.append({'action':action,'error':after.get('last_error'),'progress':progress})
        p=before.get('prompt') or {};ids=action.split()[2:]
        cards=[x['label'] for x in p.get('options',[]) if str(x['id']) in ids]
        if not failed and p.get('kind') in {'buy','gain'} and cards:
            c=cards[0]
            seen=after.get('self',{}).get('owned',{}).get(c,0)>before['self']['owned'].get(c,0)
            seen=seen or any(e.get('player')=='agent' and e.get('card')==c and e.get('event') in {'buy','gain'} for e in after.get('public_log_delta',[]))
            if not seen:self.need_review='failure: acquisition absent from ownership/events';self.evidence.append({'missing_gain':c})
        if p.get('kind')=='buy':self.recent_buys.append({'turn':before['turns_taken']['agent'],'coins':before['resources']['coins'],'cards':cards})
        self.evidence=self.evidence[-12:]

    def run_episode(self,game):
        o=_obs(self.ctx.observe());last_buy_turn=-1
        for step in range(10000):
            if self.ctx.done or game.done or o.get('terminal'):return
            p=o.get('prompt') or {}
            if not p:return
            self.consume(o)
            turn=o.get('turns_taken',{}).get('agent',0);prov=o.get('derived',{}).get('provinces_left',8)
            if p.get('kind')=='buy' and turn!=last_buy_turn:
                last_buy_turn=turn
                self.cash_history.append({'turn':turn,'coins':o['resources']['coins']})
                if (turn>5 and len(self.cash_history)>=3 and max(x['coins'] for x in self.cash_history[-3:])<3
                        and turn>self.last_review_turn+2):
                    self.need_review='failure: three successive turns below $3 despite accepted actions'
                    self.evidence.append({'economic_collapse':self.cash_history[-3:]})
                pred=self.forecasts.pop(turn,None)
                if pred:
                    actual=o['resources']['coins'];miss=actual<pred['low'] or actual>pred['high']
                    self.log('tactical_forecast_check',turn=turn,actual=actual,prediction=pred,miss=miss)
                    self.misses=self.misses+1 if miss else max(0,self.misses-1)
                    if self.misses>=2:
                        self.search_enabled=False;self.need_review='failure: two own-turn cash forecasts missed their sampled envelopes'
                        self.evidence.append({'forecast':pred,'actual':actual});self.misses=0
            if step==0:
                unknown=[c for c,d in o.get('supply',{}).items()
                         if ('Action' in d.get('types',[]) and c not in ACTIONS)
                         or ('Treasure' in d.get('types',[]) and c not in TREASURE)
                         or ('Victory' in d.get('types',[]) and c not in {'Estate','Duchy','Province','Gardens'})]
                if unknown:self.search_enabled=False;self.need_review='failure: unfamiliar rules '+str(unknown)
            if p.get('kind') not in {'buy','gain','play_action','action','trash','discard','reaction','yes_no','topdeck','order'}:
                self.need_review='failure: unfamiliar prompt '+str(p)
            if any(points(o[k]['owned'])!=o[k]['vp'] for k in ('self','opponent')):
                self.search_enabled=False;self.need_review='failure: public VP contradicts own card arithmetic' 
            try: action=self.final_buy(o,self.search(o))
            except Exception as exc:
                self.evidence.append({'estimator_error':type(exc).__name__,'detail':str(exc)[:150]})
                self.search_enabled=False;self.need_review='failure: tactical world could not reconcile public state'
                action=self.local._pick(o)
            reason=None
            if 'opening' not in self.reviewed:self.reviewed.add('opening');reason='opening kingdom and tactical objectives'
            elif prov<=4 and 'race' not in self.reviewed:self.reviewed.add('race');reason='Province race: revise attack/thinning and scoring priorities'
            elif turn>=8 and 'middle' not in self.reviewed:self.reviewed.add('middle');reason='midgame: check tactical forecasts and deck balance'
            if self.need_review:reason=self.need_review;self.need_review=None
            if reason:
                old=(self.revision,self.search_enabled);self.review(o,reason)
                if self.ctx.done:return
                if self.override and self.override[0]==o['decision_id']:
                    ids=self.override[1];self.override=None;self.forecasts.pop(turn,None)
                    action='choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')
                elif old!=(self.revision,self.search_enabled):
                    try:action=self.final_buy(o,self.search(o))
                    except Exception:action=self.local._pick(o)
            ids=action.split()[2:]
            if not self.valid(o,ids):ids=[str(x['id']) for x in p['options'][:p.get('min',0)]];action='choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')
            if (o['decision_id'],action) in self.bad_actions:
                found=None
                for n in range(p['min'],p['max']+1):
                    for combo in itertools.combinations([str(x['id']) for x in p['options']],n):
                        alt='choose '+str(o['decision_id'])+(' '+' '.join(combo) if combo else '')
                        if (o['decision_id'],alt) not in self.bad_actions:found=alt;break
                    if found:break
                if found is None:return
                self.forecasts.pop(turn,None)
                action=found
            before=o
            try:o=_obs(self.ctx.act(action))
            except Exception as exc:
                if self.ctx.done:return
                self.bad_actions.add((before['decision_id'],action));self.need_review='failure: rejected action; repair current prompt'
                self.evidence.append({'action_error':type(exc).__name__});o=_obs(self.ctx.observe())
            if not o:o=_obs(self.ctx.observe())
            self.audit(before,o,action)


def build(ctx):
    return Harness(ctx)
