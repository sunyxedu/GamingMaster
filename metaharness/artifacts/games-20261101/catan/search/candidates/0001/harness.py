"""Catan policy: use the legal-action list as a verified fast tactical layer.

The previous harness wasted its call budget by asking Astra to restate a move
on every engine action, and its parser dropped action parameters.  This policy
scores current legal moves directly, using Astra only to recover after the
checked policy stalls for an extended stretch.
"""
from __future__ import annotations

import json
import re
import time

from metaharness.core.contract import BudgetExceeded

RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _legal(obs):
    return obs.get("legal_actions", []) if isinstance(obs, dict) else []


class Economy:
    """Public topology and bounded resource/route search; never game internals."""
    def __init__(self, obs, preference=None, stop=None):
        self.stop = stop or (lambda: False)
        self.obs = obs
        self.me = obs['self']['id']
        self.hand = obs['self']['resources']
        self.v = {v['id']: v for v in obs['board']['vertices']}
        self.h = {h['id']: h for h in obs['board']['hexes']}
        self.e = {tuple(sorted(e['id'])): e['owner'] for e in obs['board']['edges']}
        self.prod = {r: 0.0 for r in RES}
        self.siteprod = {}
        for vid, v in self.v.items():
            p = {r: 0.0 for r in RES}
            for hid in v['adjacent_hexes']:
                h = self.h[hid]
                if h.get('terrain') in RES:
                    p[h['terrain']] += PIPS.get(h.get('number'), 0)
            self.siteprod[vid] = p
            if v['owner'] == self.me:
                for r in RES:
                    self.prod[r] += p[r] * (2 if v['building'] == 'CITY' else 1)
        ports = obs['self'].get('ports', [])
        self.rates = {r: 2 if r in ports else 3 if 'ANY' in ports else 4 for r in RES}
        self.goals = []
        pieces = obs['self']['pieces']
        vp = obs['self']['true_vp']
        # Multi-source Dijkstra: our existing roads are already paid for.
        sources = {v for v, d in self.v.items() if d['owner'] == self.me}
        for (a,b), owner in self.e.items():
            if owner == self.me:
                sources.update((a,b))
        dist = {s: (0, []) for s in sources if self.v[s]['owner'] in (None, self.me)}
        queue = [(0,s) for s in dist]
        import heapq
        heapq.heapify(queue)
        while queue:
            if self.stop():
                raise BudgetExceeded('local planning deadline', kind='wall')
            n,u = heapq.heappop(queue)
            if dist[u][0] != n or n >= 3:
                continue
            for v in self.v[u]['neighbors']:
                e = tuple(sorted((u,v)))
                if self.e.get(e) not in (None,self.me) or self.v[v]['owner'] not in (None,self.me):
                    continue
                new = n + (self.e.get(e) != self.me)
                if new < dist.get(v,(99,[]))[0]:
                    dist[v] = (new, dist[u][1] + ([list(e)] if self.e.get(e) != self.me else []))
                    heapq.heappush(queue,(new,v))
        for vid,v in self.v.items():
            p = self.siteprod[vid]
            production_value = sum(p[r] * (1.25 if r in ('WHEAT','ORE') else 1) for r in RES)
            # Immediate points gain value when the race is close.
            point = 2.5 if vp >= 7 else 1.3
            if v['owner'] == self.me and v['building'] == 'SETTLEMENT' and pieces['cities'] > 0:
                self.add('city',vid,[],{'ORE':3,'WHEAT':2},point + .20*production_value)
            if (pieces['settlements'] > 0 and not v['building'] and
                    not any(self.v[n]['building'] for n in v['neighbors']) and vid in dist):
                n,path = dist[vid]
                if n <= min(3,pieces['roads']):
                    diversity = sum(min(p[r],3) / (1+self.prod[r]) for r in RES)
                    port = v.get('port')
                    portvalue = .45 if port == 'ANY' else (.18*self.prod.get(port,0) if port else 0)
                    self.add('settlement',vid,path,{'WOOD':1+n,'BRICK':1+n,'SHEEP':1,'WHEAT':1},
                             point + .17*production_value + .40*diversity + portvalue)
        if obs.get('development_cards_remaining',0) > 0:
            knights = obs['self'].get('played_knights',0)
            army = obs.get('awards',{}).get('largest_army_owner')
            devval = 1.25 if army != self.me and knights < 3 else .85
            if vp >= 8:
                devval += .6
            self.add('development',None,[],{'SHEEP':1,'WHEAT':1,'ORE':1},devval)
        for g in self.goals:
            g['rank'] = g['value'] / (2.3 + self.eta(g['cost'], self.hand))
            if preference and preference.get('kind') == g['kind'] and preference.get('target') == g['target']:
                g['rank'] *= 1.20
        self.goals.sort(key=lambda g: g['rank'], reverse=True)

    def add(self, kind, target, path, cost, value):
        self.goals.append(dict(kind=kind,target=target,path=path,cost=cost,value=value))

    def eta(self,cost,hand):
        # Expected public production, with port conversion of genuine surplus.
        # Used only to rank plans; actual moves are revalidated against legality.
        for ticks in range(0,81,2):
            if self.stop():
                raise BudgetExceeded('local planning deadline', kind='wall')
            projected = {r: hand.get(r,0) + ticks*self.prod[r]/36 for r in RES}
            deficit = sum(max(0,cost.get(r,0)-projected[r]) for r in RES)
            surplus = sum(max(0,projected[r]-cost.get(r,0))/self.rates[r] for r in RES)
            if surplus + 1e-6 >= deficit:
                return ticks/4
        return 25 + sum(max(0,cost.get(r,0)-hand.get(r,0)) for r in RES)

    def trades(self,cost):
        """BFS over current hand, bank and known port rates; at most four trades.
        Return the first CURRENT legal action funding the goal, or None.
        """
        from collections import deque
        initial = tuple(self.hand.get(r,0) for r in RES)
        target = tuple(cost.get(r,0) for r in RES)
        bank = tuple(self.obs.get('bank_counts',{}).get(r,0) for r in RES)
        if all(a>=b for a,b in zip(initial,target)):
            return None, True
        q = deque([(initial,bank,None,0)])
        seen = {initial}
        legal = [a for a in _legal(self.obs) if a['type']=='maritime_trade']
        while q and len(seen)<2500:
            if self.stop():
                raise BudgetExceeded('local planning deadline', kind='wall')
            hand,stock,first,depth = q.popleft()
            if depth >= 4:
                continue
            for i,r in enumerate(RES):
                rate = self.rates[r]
                if hand[i]-target[i] < rate:
                    continue
                for j,s in enumerate(RES):
                    if hand[j] >= target[j] or stock[j] <= 0:
                        continue
                    h=list(hand); b=list(stock)
                    h[i]-=rate; h[j]+=1; b[i]+=rate; b[j]-=1
                    key=tuple(h)
                    if key in seen:
                        continue
                    selected=first
                    if selected is None:
                        selected=next((a for a in legal if isinstance(a['parameters'],list)
                            and a['parameters'][-1]==s and
                            [x for x in a['parameters'][:-1] if x] == [r]*rate),None)
                        if selected is None:
                            continue
                    if all(x>=y for x,y in zip(h,target)):
                        return selected,True
                    seen.add(key); q.append((key,tuple(b),selected,depth+1))
        return None,False

    def action(self, goal):
        kind=goal['kind']
        t = {'city':'build_city','settlement':'build_settlement','development':'buy_development_card'}[kind]
        p = goal['target']
        if goal['path']:
            t='build_road'; p=goal['path'][0]
        return next((a for a in _legal(self.obs) if a['type']==t and a['parameters']==p),None)


def _choose(obs, preference=None, stop=None):
    actions = _legal(obs)
    if not actions:
        return None
    by = {}
    for a in actions:
        by.setdefault(a['type'],[]).append(a)
    mine=obs['self']; me=mine['id']; hand=mine['resources']
    phase=str(obs.get('phase',''))
    eco=Economy(obs,preference,stop)
    if 'initial_settlement' in phase:
        def opening(a):
            p=eco.siteprod[a['parameters']]
            combined={r:eco.prod[r]+p[r] for r in RES}
            weights={'WOOD':1.0,'BRICK':1.0,'SHEEP':.8,'WHEAT':1.4,'ORE':1.3}
            return sum(weights[r]*(combined[r]**.6) for r in RES)+.18*sum(p.values())
        return max(actions,key=opening)
    if 'initial_road' in phase or ('build_road' in by and 'end_turn' not in by and 'roll' not in by):
        def road(a):
            edge=sorted(a['parameters'])
            candidates=[g for g in eco.goals if g['kind']=='settlement' and edge in g['path']]
            return max((g['value']/(1+len(g['path'])) for g in candidates),default=0)
        return max(by.get('build_road',actions),key=road)
    if 'discard_resource' in by:
        goal=eco.goals[0] if eco.goals else {'cost':{}}
        def discard(a):
            r=a['parameters']
            return (hand[r]-goal['cost'].get(r,0),hand[r],eco.prod[r])
        return max(by['discard_resource'],key=discard)
    if 'move_robber' in by:
        players={p['id']:p for p in obs.get('opponents',[])}
        def rob(a):
            coordinate,victim=a['parameters']
            h=next((h for h in eco.h.values() if h['coordinate']==coordinate),{})
            value=0
            for vid in h.get('vertices',[]):
                v=eco.v[vid]
                owner=v['owner']
                if owner is None: continue
                amount=PIPS.get(h.get('number'),0)*(2 if v['building']=='CITY' else 1)
                if owner==me:
                    value-=amount*3
                else:
                    op=players[owner]
                    threat=op['public_vp']+.35*op.get('development_count',0)
                    value+=amount*(.8+.15*threat+max(0,threat-7)*.35)
            if victim in players and players[victim]['resource_count']:
                value+=2+.1*players[victim]['public_vp']
            return value
        return max(by['move_robber'],key=rob)
    # Guaranteed immediate building wins precede all economic estimates.
    if mine['true_vp']>=9:
        for t in ('build_city','build_settlement'):
            if t in by: return by[t][0]
    if 'play_knight_card' in by:
        return by['play_knight_card'][0]
    if 'roll' in by:
        return by['roll'][0]
    if 'play_road_building' in by and any(g['path'] for g in eco.goals):
        return by['play_road_building'][0]
    # Select among concrete projects that can be fully funded by <=4 trades.
    affordable=[]
    for g in eco.goals:
        trade,ok=eco.trades(g['cost'])
        if ok:
            a=trade or eco.action(g)
            if a:
                affordable.append((g['value']/(2.3+.35*len(g['path'])),a,g))
    if affordable:
        _,a,g=max(affordable,key=lambda x:x[0])
        # Do not consume the only ore/wheat for a development card while a
        # high-quality city is one expected round away.
        if g['kind']!='development' or not eco.goals or eco.goals[0]['rank']<g['rank']*1.7:
            return a
    if 'play_year_of_plenty' in by:
        target=eco.goals[0]['cost'] if eco.goals else {}
        def plenty(a):
            h=dict(hand)
            for r in a['parameters']: h[r]=h.get(r,0)+1
            return -eco.eta(target,h)
        best=max(by['play_year_of_plenty'],key=plenty)
        if plenty(best)>-eco.eta(target,hand)+.1:
            return best
    if 'play_monopoly' in by:
        # Conservation gives the aggregate opponent holding exactly, without
        # revealing how cards are divided among opponents.
        def monopoly(a):
            r=a['parameters']
            return 19-obs.get('bank_counts',{}).get(r,19)-hand.get(r,0)
        a=max(by['play_monopoly'],key=monopoly)
        if monopoly(a)>=3: return a
    # Building a route in stages is allowed only for the selected viable site.
    if eco.goals:
        g=eco.goals[0]
        if g['path'] and len(g['path'])<=2:
            a=eco.action(g)
            if a: return a
    # A currently affordable VP investment is preferable to indefinite saving.
    builds=[x for x in affordable if x[2]['kind']!='development']
    if builds: return max(builds,key=lambda x:x[0])[1]
    if 'buy_development_card' in by:
        best=eco.goals[0] if eco.goals else None
        if best is None or eco.eta(best['cost'],hand)>2 or sum(hand.values())>7:
            return by['buy_development_card'][0]
    return by.get('end_turn',actions)[0]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.last_score = None
        self.stall = 0
        self.last_resources = None
        self.replans = 0
        self.last_replan_step = -100
        self.failure = None
        self.consecutive_failures = 0
        self.local_deadline = float('inf')

    def _stopped(self):
        return self.ctx.done or time.monotonic() >= self.local_deadline

    def _ask_setup(self, game, obs):
        """Use one short Astra decision for each initial settlement only.

        Settlement IDs become stale after opponents act, so this is called at
        the current decision and the answer is accepted only if it names a
        currently legal ID.
        """
        ctx = self.ctx
        if (ctx.system2 is None or self._stopped()
                or ctx.budget.remaining().get("system2_calls", 0) <= 0
                or ctx.budget.remaining().get('wall_s', 0) < 270):
            return None
        eco = Economy(obs, stop=self._stopped)
        candidates = []
        for a in _legal(obs):
            v = eco.v[a['parameters']]
            candidates.append(dict(id=a['id'], vertex=a['parameters'],
                pips_by_resource=eco.siteprod[a['parameters']], port=v.get('port'),
                neighbors=v['neighbors']))
        prompt = (
            "Choose a strong Catan opening settlement against three depth-2 bots. "
            "Goal: reach 10 points first. Balance high production, wheat/ore cities, "
            "wood/brick expansion, and sheep for development. Evaluate BOTH settlements "
            "as one economy; second placement should complement the first. A productive "
            "ore/wheat/sheep economy can use cities and army without much brick. "
            "Do not choose a low-pip site merely for coverage. Second placement gives "
            "one resource per adjacent producing hex. Ports enable 2:1 or ANY 3:1 trades. "
            "Return exactly one CURRENT action ID and nothing else.\n"
            + json.dumps(dict(me=obs['self'], existing_production=eco.prod,
                candidates=candidates, buildings=[v for v in eco.v.values() if v['building']],
                hexes=list(eco.h.values())), separators=(",", ":"))
        )
        try:
            result = ctx.think(prompt, session_id=None, timeout_s=240.0)
            self.session = result.session_id
            answer = result.text or ""
            for a in _legal(obs):
                aid = str(a.get("id"))
                if aid and (aid in re.findall(r"r[0-9]+:a[0-9]+", answer) or aid == answer.strip('` \n\r\t"')):
                    return aid
        except BudgetExceeded:
            raise
        except (TimeoutError, RuntimeError):
            return None
        return None

    def _recover(self, game, obs, progress):
        ctx = self.ctx
        if (ctx.system2 is None or ctx.budget.remaining().get("system2_calls", 0) <= 0
                or self.replans >= 6):
            return None
        if ctx.budget.remaining().get('wall_s', 0) < 290:
            return None
        eco = Economy(obs, ctx.memory.get('strategy'), self._stopped)
        projects = [{k:g[k] for k in ('kind','target','path','cost','rank')} for g in eco.goals[:10]]
        prompt = (
            "Replan our Catan policy. We play three AlphaBeta depth-2 opponents. "
            "Choose one current legal action and optionally a persistent project. "
            "Code computes shortest settlement routes and up to four maritime trades, "
            "then verifies the live legal menu and effects after each action. "
            "The previous strategy has stalled or failed verification. Correct it; "
            "prefer productive cities/settlements and a realistic path to 10. "
            "Return short JSON: {\"id\":\"current action ID\",\"kind\":\"city|settlement|development\","
            "\"target\":vertex_or_null,\"reason\":\"short explanation\"}. "
            "kind/target must identify one listed project; omit them if none fits.\n"
            + json.dumps(dict(progress=progress, failure=self.failure,
                previous_strategy=ctx.memory.get('strategy'), phase=obs.get('phase'),
                self=obs['self'], opponents=obs.get('opponents'), awards=obs.get('awards'),
                production=eco.prod, projects=projects,
                buildings=[v for v in eco.v.values() if v['building']],
                roads=[e for e in obs['board']['edges'] if e['owner']],
                legal_actions=_legal(obs)), separators=(",", ":"))
        )
        self.replans += 1
        ctx.log('replan', failure=self.failure, progress=progress, projects=projects)
        try:
            result = ctx.think(prompt, session_id=None, timeout_s=240.0)
            answer = (result.text or "").strip()
            match = re.search(r"\{.*\}", answer, re.S)
            if match:
                try:
                    plan = json.loads(match.group())
                    if isinstance(plan, dict) and any(plan.get('kind')==g['kind'] and
                            plan.get('target')==g['target'] for g in eco.goals):
                        ctx.memory['strategy'] = {k:plan.get(k) for k in ('kind','target','reason')}
                        ctx.log('new_strategy', strategy=ctx.memory['strategy'])
                except (ValueError, TypeError):
                    pass
            ids = re.findall(r'r[0-9]+:a[0-9]+', answer)
            for aid in ids:
                if any(a['id']==aid for a in _legal(obs)):
                    return aid
        except BudgetExceeded:
            raise
        except (TimeoutError, RuntimeError) as exc:
            ctx.log('model_recovery_failure', error=str(exc))
        return None

    def _verify(self, before, after, action, progress_before, progress_after):
        if after.get('terminal'):
            return None
        if progress_after.get('illegal_actions',0)>progress_before.get('illegal_actions',0):
            return 'engine rejected action ' + str(action)
        if before.get('revision') == after.get('revision'):
            return 'action did not advance observation revision: ' + str(action)
        if not action:
            return None
        kind, p = action['type'], action['parameters']
        me = before['self']['id']
        if kind in ('build_settlement','build_city'):
            v = next((v for v in after['board']['vertices'] if v['id']==p), {})
            expected = 'CITY' if kind=='build_city' else 'SETTLEMENT'
            if v.get('owner')!=me or v.get('building')!=expected:
                return 'building effect disagreed with prediction: ' + str(action)
        if kind=='build_road':
            e = next((e for e in after['board']['edges'] if sorted(e['id'])==sorted(p)), {})
            if e.get('owner')!=me:
                return 'road effect disagreed with prediction: ' + str(action)
        if kind=='maritime_trade':
            expected=dict(before['self']['resources'])
            for r in p[:-1]:
                if r: expected[r]-=1
            expected[p[-1]]+=1
            if expected!=after['self']['resources']:
                return 'trade hand delta disagreed with verified list convention'
        return None

    def run_episode(self, game):
        remaining = max(0.0, self.ctx.budget.remaining().get('wall_s', 0.0))
        # Leave time for the final synchronous bot advance and evaluator grace.
        reserve = min(30.0, remaining * 0.4)
        self.local_deadline = time.monotonic() + max(0.0, remaining - reserve)
        try:
            self._run_episode(game, reserve)
        except BudgetExceeded:
            return

    def _run_episode(self, game, reserve):
        ctx = self.ctx
        obs = game.observe()
        while not self._stopped() and not game.done:
            if ctx.budget.remaining().get('wall_s', 0.0) <= reserve:
                return
            progress = ctx.progress()
            score = progress.get("score")
            res = obs.get("self", {}).get("resources", {}) if isinstance(obs, dict) else {}
            signature = (score, tuple(sorted(res.items())))
            if signature == self.last_resources:
                self.stall += 1
            else:
                self.stall = 0
            self.last_resources = signature
            # A long stretch with no VP or hand movement is evidence that the
            # heuristic's plan is failing; request one evidence-conditioned reset.
            step_no = int(progress.get("acts", 0))
            strategy = ctx.memory.get('strategy')
            if strategy and strategy.get('kind')=='settlement':
                eco = Economy(obs, stop=self._stopped)
                target = strategy.get('target')
                if not any(g['kind']=='settlement' and g['target']==target for g in eco.goals):
                    v = eco.v.get(target,{})
                    if v.get('owner') != obs['self']['id']:
                        self.failure = 'chosen settlement site or route is no longer available'
                    ctx.memory.pop('strategy',None)
            should_replan = (self.failure is not None or self.stall >= 18 or
                progress.get("since_score_change", 0) >= 30)
            # Ordinary stagnation is a strategic decision after rolling; a
            # rejected action bypasses that phase/cooldown restriction.
            if not self.failure:
                should_replan = should_replan and obs.get('phase')=='play_turn' and obs['self'].get('has_rolled')
            if should_replan and (self.failure or step_no - self.last_replan_step >= 16):
                self.last_replan_step = step_no
                action = self._recover(game, obs, progress)
                self.stall = 0
                self.failure = None
            else:
                action = None
            # Opening sites are the only choices where a bounded model call is
            # worth its latency; all later actions remain a checked local policy.
            if action is None and "initial_settlement" in str(obs.get("pending_decision", obs.get("phase", ""))):
                phase_key = str(obs.get('self', {}).get('true_vp', 0))
                if phase_key not in getattr(self, "setup_asked", set()):
                    if not hasattr(self, "setup_asked"):
                        self.setup_asked = set()
                    self.setup_asked.add(phase_key)
                    action = self._ask_setup(game, obs)
            if action is None:
                if self._stopped():
                    return
                chosen = _choose(obs, ctx.memory.get("strategy"), self._stopped)
                action = chosen.get("id") if isinstance(chosen, dict) else chosen
            if action is None:
                break
            if self._stopped() or ctx.budget.remaining().get('wall_s', 0.0) <= reserve:
                return
            before = progress
            selected = next((a for a in _legal(obs) if a['id']==action),None)
            old_obs = obs
            try:
                next_obs = ctx.act(action)
            except BudgetExceeded:
                raise
            except Exception as exc:
                if ctx.done:
                    break
                ctx.log("action_exception", error=str(exc))
                obs = ctx.observe()
                self.stall = 18
                self.failure = str(exc)
                self.consecutive_failures += 1
                if self.consecutive_failures >= 4:
                    break
                continue
            after = ctx.progress()
            obs = next_obs if isinstance(next_obs, dict) else ctx.observe()
            self.failure = self._verify(old_obs, obs, selected, before, after)
            if self.failure:
                self.consecutive_failures += 1
                ctx.log('plan_failed', reason=self.failure, action=selected, progress=after)
                ctx.memory.pop('strategy',None)
                if self.consecutive_failures >= 4:
                    break
            else:
                self.consecutive_failures = 0



def build(ctx):
    return Harness(ctx)
