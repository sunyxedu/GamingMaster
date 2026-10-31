"""Public-zone rollout purchase search with audited Astra recovery.

Only public observations enter the estimator. Random orders are hypothetical,
never recovered simulator state. The game remains the source of truth.
"""
from __future__ import annotations
import concurrent.futures
import itertools
import json
import math
import random
import re
import time
from collections import Counter


def _obs(value):
    if isinstance(value, dict):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _count(mapping, card):
    return int((mapping or {}).get(card, 0) or 0)


def points(owned):
    return (owned.get("Estate", 0) + 3*owned.get("Duchy", 0)
            + 6*owned.get("Province", 0) - owned.get("Curse", 0)
            + owned.get("Gardens", 0)*(sum(owned.values())//10))


def expand(counts):
    return [c for c,n in sorted(counts.items()) for _ in range(max(0,int(n)))]


def liquidation(owned, supply, coins, buys):
    """Exact remaining points once an ending condition is already satisfied."""
    own=Counter(owned)
    options=[c for c in ('Province','Duchy','Gardens','Estate','Copper')
             if c in supply and supply[c]['count']>0 and supply[c]['cost']<=coins
             and (c!='Copper' or own['Gardens']>0 or supply.get('Gardens',{}).get('count',0)>0)]
    used=Counter(); best=(points(own),0,0,())
    def visit(start,cash,left,bundle):
        nonlocal best
        candidate=(points(own),-len(bundle),cash,tuple(bundle))
        if candidate[:3]>best[:3]:best=candidate
        if left<=0:return
        for i in range(start,len(options)):
            c=options[i];d=supply[c]
            if d['cost']>cash or used[c]>=d['count']:continue
            used[c]+=1;own[c]+=1
            visit(i,cash-d['cost'],left-1,bundle+[c])
            used[c]-=1;own[c]-=1
    visit(0,coins,min(buys,12),[])
    return list(best[3])


class LocalPolicy:
    def __init__(self):
        self.plan = {"mode":"money", "priority":[]}
        self.override = None

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


class Rollout:
    """Small forward estimator of the disclosed rules, not an engine import.

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
        self.truncated = False

    def ended(self):
        return self.supply.get('Province',{}).get('count',1)<=0 or sum(d['count']==0 for d in self.supply.values())>=3

    def gain(self, who, card, hand=False):
        if card is None or card not in self.supply or self.supply[card]['count'] <= 0:
            return False
        self.supply[card]['count'] -= 1
        p = self.players[who]
        p.owned[card] += 1
        (p.hand if hand else p.discard).append(card)
        return True

    def trash(self, p, card):
        p.hand.remove(card)
        p.owned[card] -= 1

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

    def buy_card(self, who, coins):
        p = self.players[who]
        available = [c for c,d in self.supply.items() if d['count']>0 and d['cost']<=coins]
        if who == 1:
            owned = p.owned
            n = self.supply['Province']['count']
            money = sum(owned[c]*v for c,v in TREASURE.items())
            nt = sum(owned[c] for c in TREASURE)
            order = []
            if money>15: order.append('Province')
            if n<5: order.append('Duchy')
            if n<3: order.append('Estate')
            order.append('Gold')
            if n<7: order.append('Duchy')
            if owned['Smithy']<nt/11: order.append('Smithy')
            order.append('Silver')
            return next((c for c in order if c in available), None)
        if not available:
            return None
        got = self.choose(who,'buy',available,coins=coins)
        return got[0] if got else None

    def buying(self, who, coins, buys, first=None, force=False):
        if force:
            if first is None:
                return
            if self.gain(who, first):
                coins -= self.supply[first]['cost']
                buys -= 1
        if who==0 and self.ended():
            for c in liquidation(self.players[0].owned,self.supply,coins,buys):
                self.gain(who,c)
            return
        for _ in range(min(buys,12)):
            c = self.buy_card(who, coins)
            if c is None: break
            coins -= self.supply[c]['cost']
            self.gain(who,c)
            if who==0 and self.ended():
                for rest in liquidation(self.players[0].owned,self.supply,coins,buys-_-1):
                    self.gain(who,rest)
                break

    def turn(self, who):
        p, enemy = self.players[who], self.players[1-who]
        label = 'agent' if who==0 else 'opponent'
        self.turns[label] = self.turns.get(label,0)+1
        actions, coins, buys, merchants = 1,0,1,0
        for _ in range(80):
            if actions <= 0: break
            options = [c for c in p.hand if c in ACTIONS and (who==0 or c=='Smithy')]
            if not options: break
            if who == 1:
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
                if card=='Witch' and 'Moat' not in enemy.hand: self.gain(1-who,'Curse')
            elif card == 'Militia':
                coins += 2
                if 'Moat' not in enemy.hand:
                    count = max(0,len(enemy.hand)-3)
                    # Disclosed BMU default discards first required cards.
                    discarded = enemy.hand[:count]
                    del enemy.hand[:count]
                    enemy.discard.extend(discarded)
            elif card == 'Chapel':
                for c in self.choose(who,'trash',p.hand[:],source=card,hi=min(4,len(p.hand))):
                    self.trash(p,c)
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
                    inspected.remove(c); p.owned[c] -= 1
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
        self.buying(who,coins,buys)
        p.cleanup(self.rng[who])

    def evaluate(self, obs, card, horizon=36):
        resources = obs['resources']
        self.buying(0,resources['coins'],resources['buys'],card,True)
        self.players[0].cleanup(self.rng[0])
        for i in range(2*horizon):
            if self.ended(): break
            self.turn(1 if i%2==0 else 0)
        else:
            self.truncated = True
        return points(self.players[0].owned)-points(self.players[1].owned)


class Harness:
    def __init__(self, ctx):
        self.ctx=ctx
        self.local=LocalPolicy()
        self.calls=0
        self.pending=None
        self.retired=[]
        self.request_decision=None
        self.bad=set()
        self.evidence=[]
        self.buys=[]
        self.search_log=[]
        self.need_review=None
        self.override=None
        self.last_illegal=0
        self.last_review_turn=-99
        self.reviewed_race=False
        self.search_enabled=True
        self.shrink=0.6
        self.predictions={}
        self.misses=0

    def log(self,event,**fields):
        try:self.ctx.log(event,**fields)
        except Exception:pass

    def progress(self):
        try:return _obs(self.ctx.progress())
        except Exception:return {}

    def remaining(self):
        try:return self.ctx.budget.remaining()
        except Exception:return {}

    def consume(self,o):
        if self.pending is None or not self.pending.done(): return
        f=self.pending; self.pending=None
        try:
            raw=getattr(f.result(),'text','')
            data={}
            for m in re.finditer(r'\{',raw if isinstance(raw,str) else ''):
                try:
                    v,_=json.JSONDecoder().raw_decode(raw[m.start():])
                    if isinstance(v,dict): data=v; break
                except Exception:pass
            if not data:
                self.log('invalid_rollout_review',text=str(raw)[:180]); return
            if type(data.get('use_rollouts')) is bool: self.search_enabled=data['use_rollouts']
            if type(data.get('uncertainty_penalty')) in (int,float) and math.isfinite(data['uncertainty_penalty']):
                self.shrink=max(0,min(2,float(data['uncertainty_penalty'])))
            for k,limit in (('duchy_at',5),('estate_at',3)):
                if type(data.get(k)) is int: self.local.plan[k]=max(0,min(limit,data[k]))
            ids=data.get('option_ids')
            if self.request_decision==o.get('decision_id') and self.valid(o,ids):
                self.override=[str(i) for i in ids]
            self.ctx.memory['rollout_review']=data
            self.log('rollout_review',review=data)
        except Exception as exc:self.log('review_failure',error=type(exc).__name__)

    def review(self,o,reason):
        self.consume(o)
        rem=self.remaining()
        if self.ctx.done or self.calls>=4 or rem.get('wall_s',0)<350 or rem.get('system2_calls',0)<2:return
        if self.pending is not None:
            if not reason.startswith('failure') or self.retired:return
            self.retired.append(self.pending); self.pending=None
        view={k:o.get(k) for k in ('decision_id','supply','self','opponent','turns_taken','resources','derived','prompt','last_error')}
        view.update(progress=self.progress(), failures=self.evidence[-5:],
                    rollouts=self.search_log, recent_buys=self.buys[-8:])
        prompt=(
            'Audit a Dominion public-belief purchase planner against disclosed BMU-1. Maximize FINAL VP MARGIN. '
            'The local executor buys limited attack/draw/thinning, then money and points; it plays cantrips '
            'before terminals and trashes while keeping at least $4 total deck money. At each buy it compares '
            'legal purchases with complete-game Monte Carlo rollouts, preserving the own hand/discard/deck '
            'multisets and shuffling only when exhausted. Opponent unknown zones are randomly allocated; '
            'its exact disclosed Smithy/buy policy is simulated. These are approximate forecasts, NOT hidden state. '
            'The selected candidate must beat the baseline by uncertainty_penalty times paired standard error. '
            'Look for forecast/observation mismatch and overbuilding. BMU buys Province if total treasure>15, '
            'Duchy if P<5, Estate if P<3, Gold, Duchy if P<7, Smithy if count<treasures/11, Silver. '
            'Return ONLY JSON under 120 tokens: {"use_rollouts":true,"uncertainty_penalty":0.6,'
            '"duchy_at":3,"estate_at":1,"note":"concrete diagnosis"}. Set use_rollouts false if evidence '
            'invalidates the simulator. For a failed CURRENT decision include option_ids matching its legal IDs. '
            'Never infer hidden order. Exact card text/options override the model. '
            'Reason='+reason+'\nSTATE='+json.dumps(view,separators=(',',':')))
        self.calls+=1; self.request_decision=o.get('decision_id')
        self.last_review_turn=o.get('turns_taken',{}).get('agent',0)
        try:
            self.pending=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,timeout_s=300.0))
            try:self.pending.result(timeout=320.0)
            except concurrent.futures.TimeoutError:self.log('review_wait_cutoff',seconds=320)
            except Exception:pass
            self.consume(o)
        except Exception as exc:self.log('review_start_failed',error=type(exc).__name__)

    def valid(self,o,ids):
        if not isinstance(ids,list) or any(type(x) not in (str,int) for x in ids):return False
        ids=[str(i) for i in ids]
        p=o.get('prompt') or {}; opts={str(x['id']) for x in p.get('options',[])}
        return p.get('min',0)<=len(ids)<=p.get('max',1) and len(set(ids))==len(ids) and set(ids)<=opts

    def search(self,o):
        baseline=self.local._pick(o)
        if not self.search_enabled or self.remaining().get('wall_s',0)<30:return baseline
        p=o['prompt']
        if p['kind']!='buy':return baseline
        opts={x['label']:str(x['id']) for x in p['options']}
        chosen=baseline.split()[2:]
        default=next((c for c,i in opts.items() if i in chosen),None)
        prov=o.get('derived',{}).get('provinces_left',8)
        # Exact arithmetic is guarded against the simulator's public ledger.
        if any(points(o[k]['owned'])!=o[k]['vp'] for k in ('self','opponent')):
            self.need_review='failure: VP model mismatch';self.search_enabled=False;return baseline
        if prov==0 or sum(d['count']==0 for d in o['supply'].values())>=3:
            bundle=liquidation(o['self']['owned'],o['supply'],o['resources']['coins'],o['resources']['buys'])
            self.log('exact_final_buys',bundle=bundle,decision_id=o['decision_id'])
            card=bundle[0] if bundle else None
            return 'choose '+str(o['decision_id'])+(' '+opts[card] if card in opts else '')
        candidates=[None]
        for c in opts:
            if c=='Curse':continue
            rank=self.local._buy_rank(c,o)
            if (c==default or rank>0 or (prov<=5 and c in {'Duchy','Estate','Gardens'})
                    or (c=='Copper' and o['self']['owned'].get('Gardens',0)>0)):
                candidates.append(c)
        if len(candidates)<=1:return baseline
        if default not in candidates:candidates.append(default)
        # Constant estimator seeds are independent of simulator/reset RNG.
        values={c:[] for c in candidates}; nextcoins={c:[] for c in candidates}; truncated=0
        started=time.monotonic()
        for sample in range(128):
            if self.ctx.done:return baseline
            if sample>=32 and time.monotonic()-started>12:break
            for c in candidates:
                r=Rollout(o,self.local,730001+sample*104729)
                values[c].append(r.evaluate(o,c))
                if r.first_next_coins is not None:nextcoins[c].append(r.first_next_coins)
                truncated+=int(r.truncated)
        base=values[default]; n=len(base)
        rows=[]
        for c in candidates:
            v=values[c]; delta=[a-b for a,b in zip(v,base)]
            mean=sum(v)/n; lift=sum(delta)/n
            se=(sum((d-lift)**2 for d in delta)/max(1,n-1)/n)**0.5
            score=lift-self.shrink*se
            rows.append({'card':c or 'PASS','mean_margin':round(mean,3),'paired_lift':round(lift,3),
                         'paired_se':round(se,3),'value':score,'n':n})
        rows.sort(key=lambda x:(x['value'], x['card']==(default or 'PASS')),reverse=True)
        winner=rows[0]['card']; best=None if winner=='PASS' else winner
        self.search_log=rows[:8]
        self.log('zone_rollout',decision_id=o['decision_id'],baseline=default,selected=best,
                 candidates=rows,truncated_rollouts=truncated,seconds=time.monotonic()-started,
                 known_deck=o['self']['deck_count'],known_discard=o['self']['discard_count'])
        if nextcoins[best]:
            turn=o.get('turns_taken',{}).get('agent',0)
            self.predictions[turn+1]=nextcoins[best]
        if best is None:return 'choose '+str(o['decision_id'])
        return 'choose '+str(o['decision_id'])+' '+opts[best]

    def audit(self,before,after,action):
        progress=self.progress()
        did=before.get('decision_id')
        illegal=progress.get('illegal_actions',0)
        failed=(illegal>self.last_illegal or after.get('illegal_action_count',0)>before.get('illegal_action_count',0)
                or (after.get('decision_id')==did and not after.get('terminal'))
                or progress.get('since_obs_change',0)>=2)
        self.last_illegal=illegal
        if failed:
            self.bad.add((did,action))
            self.evidence.append({'action':action,'error':after.get('last_error'),'progress':progress})
            self.need_review='failure: rejected or unchanged action; repair current prompt'
        p=before.get('prompt') or {}
        if not failed and p.get('kind') in {'buy','gain'}:
            ids=action.split()[2:]
            card=next((x['label'] for x in p.get('options',[]) if str(x['id']) in ids),None)
            if card:
                gained=after.get('self',{}).get('owned',{}).get(card,0)>before['self']['owned'].get(card,0)
                witnessed=any(e.get('player')=='agent' and e.get('card')==card and e.get('event') in {'buy','gain'} for e in after.get('public_log_delta',[]))
                if not gained and not witnessed:
                    self.need_review='failure: expected purchase missing from public ownership'
                    self.evidence.append({'expected_gain':card,'progress':progress})
            if p['kind']=='buy':self.buys.append({'turn':before.get('turns_taken',{}).get('agent'),
                                                'coins':before.get('resources',{}).get('coins'),'card':card})
        self.evidence=self.evidence[-10:]

    def run_episode(self,game):
        o=_obs(self.ctx.observe())
        unknown=[c for c,d in o.get('supply',{}).items()
                 if ('Action' in d.get('types',[]) and c not in ACTIONS)
                 or ('Treasure' in d.get('types',[]) and c not in TREASURE)
                 or ('Victory' in d.get('types',[]) and c not in {'Estate','Duchy','Province','Gardens'})]
        if unknown:
            self.search_enabled=False
            self.need_review='failure: unsupported card rules in public supply '+str(unknown)
        last_buy_turn=-1
        for step in range(10000):
            if self.ctx.done or game.done or o.get('terminal'):return
            p=o.get('prompt') or {}
            if not p:return
            self.consume(o)
            turn=o.get('turns_taken',{}).get('agent',0)
            if p.get('kind') not in {'buy','gain','play_action','action','trash','discard','reaction','yes_no','topdeck','order'}:
                self.need_review='failure: unfamiliar prompt '+str(p)
            if p.get('kind')=='buy' and turn!=last_buy_turn:
                last_buy_turn=turn
                predicted=self.predictions.pop(turn,None)
                actual=o.get('resources',{}).get('coins',0)
                if predicted:
                    low,high=min(predicted),max(predicted)
                    mean=sum(predicted)/len(predicted)
                    self.log('forecast_check',turn=turn,actual=actual,mean=mean,range=[low,high])
                    self.misses=self.misses+1 if actual<low or actual>high else max(0,self.misses-1)
                    if self.misses>=2:
                        self.evidence.append({'forecast_miss':{'actual':actual,'range':[low,high],'mean':mean}})
                        self.need_review='failure: next-turn spending fell outside rollout envelope twice'
                        self.misses=0
                recent={b['turn']:b['coins'] for b in self.buys[-10:]}
                if len(recent)>=3 and max(list(recent.values())[-3:])<3 and turn>5 and turn>self.last_review_turn+2:
                    self.need_review='failure: three consecutive low-economy turns despite legal progress'
            # Build estimates first, so Astra can inspect a concrete comparison.
            try:action=self.search(o)
            except Exception as exc:
                self.evidence.append({'rollout_error':type(exc).__name__,'detail':str(exc)[:120]})
                self.need_review='failure: rollout could not reconcile public state'
                self.search_enabled=False
                action=self.local._pick(o)
            prov=o.get('derived',{}).get('provinces_left',8)
            if self.calls==0 and p.get('kind')=='buy':self.need_review=self.need_review or 'opening comparison of public-zone purchase rollouts'
            if prov<=4 and not self.reviewed_race and p.get('kind')=='buy':
                self.reviewed_race=True;self.need_review=self.need_review or 'first Province-race transition; audit current forecasts'
            if self.need_review:
                reason=self.need_review;self.need_review=None
                settings=(dict(self.local.plan),self.shrink,self.search_enabled)
                self.review(o,reason)
                if self.ctx.done:return
                if self.override is not None:
                    ids=self.override;self.override=None
                    action='choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')
                elif not self.search_enabled:action=self.local._pick(o)
                elif settings!=(self.local.plan,self.shrink,self.search_enabled):
                    action=self.search(o)
            ids=action.split()[2:]
            if not self.valid(o,ids):
                ids=[str(x['id']) for x in p.get('options',[])[:p.get('min',0)]]
                action='choose '+str(o['decision_id'])+(' '+' '.join(ids) if ids else '')
            if (o.get('decision_id'),action) in self.bad:
                legal=[str(x['id']) for x in p.get('options',[])]
                found=False
                for n in range(p.get('min',0),p.get('max',1)+1):
                    for alt in itertools.combinations(legal,n):
                        candidate='choose '+str(o['decision_id'])+(' '+' '.join(alt) if alt else '')
                        if (o.get('decision_id'),candidate) not in self.bad:
                            action=candidate;found=True;break
                    if found:break
                if not found:return
            before=o
            try:o=_obs(self.ctx.act(action))
            except Exception as exc:
                if self.ctx.done:return
                self.bad.add((before.get('decision_id'),action))
                self.evidence.append({'action_exception':type(exc).__name__})
                self.need_review='failure: action raised; repair current legal selection'
                o=_obs(self.ctx.observe())
            if not o:o=_obs(self.ctx.observe())
            self.audit(before,o,action)


def build(ctx):
    return Harness(ctx)
