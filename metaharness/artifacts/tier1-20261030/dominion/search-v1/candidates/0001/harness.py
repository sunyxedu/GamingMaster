"""Astra chooses reusable deck plans; validated local policies execute every prompt."""
from __future__ import annotations
import itertools
import json
import math
import re
from collections import Counter

TREASURE = {'Copper': 1, 'Silver': 2, 'Gold': 3}
POINTS = {'Estate': 1, 'Duchy': 3, 'Province': 6, 'Curse': -1}
CANTRIPS = {'Village', 'Festival', 'Market', 'Laboratory', 'Merchant', 'Sentry', 'Cellar'}
TERMINALS = {'Smithy', 'Witch', 'Militia', 'Moat', 'Chapel', 'Workshop', 'Remodel', 'Mine'}


def observation(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def vp(cards):
    return sum(cards.get(c, 0) * n for c, n in POINTS.items()) + cards.get('Gardens', 0) * (sum(cards.values()) // 10)


def parse_json(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', str(text)):
        try:
            obj, _ = decoder.raw_decode(str(text)[match.start():])
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
    return None


def finite_int(x, default, low, high):
    try:
        return max(low, min(high, int(x)))
    except (ValueError, TypeError, OverflowError):
        return default


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = None
        self.calls = 0
        self.last_plan_turn = -10
        self.last_plan_provinces = 8
        self.failures = 0
        self.history = []
        self.override = None
        self.attempts = Counter()

    def default_plan(self, o):
        supply = o.get('supply', {})
        targets = []
        if 'Chapel' in supply:
            targets.append(['Chapel', 1])
        if 'Witch' in supply:
            targets.append(['Witch', 1])
        targets.append(['Silver', 1])
        if 'Sentry' in supply:
            targets.append(['Sentry', 1])
        if 'Witch' not in supply:
            if 'Militia' in supply:
                targets.append(['Militia', 1])
            elif 'Smithy' in supply:
                targets.append(['Smithy', 1])
        targets += [['Gold', 2], ['Silver', 3], ['Laboratory', 3], ['Market', 2]]
        return {'mode': 'money', 'targets': [r for r in targets if r[0] in supply],
                'province_at': 8, 'duchy_at': 3, 'estate_at': 1, 'copper_floor': 5,
                'note': 'Thin early; one attack/draw terminal, then strong money.'}

    def plan_now(self, o):
        ctx = self.ctx
        turn = o.get('turns_taken', {}).get('agent', 0)
        provinces = o.get('supply', {}).get('Province', {}).get('count', 8)
        budget = ctx.budget.remaining()
        if self.calls >= 10 or self.failures >= 2 or budget.get('system2_calls', 0) <= 1 or budget.get('wall_s', 0) < 180:
            return
        urgent = provinces <= 3 < self.last_plan_provinces
        if self.calls and turn < self.last_plan_turn + 3 and not urgent:
            return
        self.last_plan_turn = turn
        self.last_plan_provinces = provinces
        self.calls += 1
        state = {k: o.get(k) for k in ('agent_seat', 'active_player', 'turns_taken', 'resources',
                 'kingdom', 'opponent_id', 'supply', 'self', 'opponent', 'derived', 'prompt')}
        owned = o.get('self', {}).get('owned', {})
        state['computed'] = {
            'deck_size': sum(owned.values()),
            'treasure_money_total': sum(owned.get(c, 0)*v for c, v in TREASURE.items()),
            'terminal_count': sum(owned.get(c, 0) for c in TERMINALS),
            'extra_action_capacity': owned.get('Village', 0)+owned.get('Festival', 0),
            'recent_buys': self.history[-8:],
        }
        prompt = '''You are the strategic planner for two-player Base Dominion. Maximize final VP margin against the disclosed money bot. The executor will finish the entire game locally; you revise its plan every three own turns. No hidden deck order is available. BM buys Province, Duchy at <=4 Provinces, Estate at <=2, else Gold/Silver. SM adds one Smithy early; WM one Witch. All treasures auto-play. Exact supply text and observed options are authoritative.
Return ONLY compact JSON (under 450 tokens):
{"note":"brief strategy/next adaptation", "mode":"money or engine or gardens", "buy_now":"one legal card name or pass", "targets":[["Card",desired_total_owned],...], "province_at":8,"duchy_at":3,"estate_at":1,"copper_floor":5}
Policy semantics: immediately winning end-of-turn baskets are searched in code. Otherwise buy Province when provinces_left<=province_at; Duchy at <=duchy_at; Estate at <=estate_at. Then scan targets IN ORDER and buy the first AFFORDABLE, nonempty card whose total_owned is below its target. Targets are NOT a sequential queue: skip unaffordable/satisfied entries. If no target applies, buy Gold, then Silver, else pass. buy_now overrides this policy for this one prompt only. Specify absolute totals, not additional copies. Targets apply also to constrained action gains, after scoring priorities. All fields are optional; malformed output keeps the previous plan. No executable code.
Choose a coherent, fast plan. Avoid speculative Villages without terminal demand or too many terminals without actions. Early Chapel/Sentry can pay strongly; Chapel trashes multiple cards while retaining at least copper_floor total printed money across the whole deck, counting Militia/Festival/Market. Witch while Curses remain is powerful. Prefer enough economy to draw and buy over aimless engine components. Laboratory/Market can improve a thin deck; Chapel plus just money is also viable. A target Witch should be removed after Curses empty unless more terminal draw is useful. Do not buy junk merely to spend money. Province thresholds may delay greening for an engine, but remaining Provinces<=3 makes scoring urgent. Gardens plans require meaningful extra gains/deck size. Do not infer permanent Moat protection.
Local executor plays nonterminals first, balances terminal selection, handles multi-card Chapel/Sentry trash, Cellar/discards, Mine/Remodel and reactions. You own acquisition balance and transition to scoring. Keep a stable plan unless observation contradicts it.
Previous plan: ''' + json.dumps(self.plan, separators=(',', ':')) + '\nCurrent state: ' + json.dumps(state, separators=(',', ':'))
        future = None
        try:
            # A backend's requested timeout may not cover admission/queue time.
            # Bound our own wait too. If abandoned, finish locally; the evaluator
            # cancels this model-only background call when the episode closes.
            future = ctx.spawn(lambda: ctx.think(
                prompt, session_id=None, timeout_s=min(125.0, budget['wall_s'] - 120)))
            result = future.result(timeout=min(150.0, budget['wall_s'] - 120))
            data = parse_json(result.text)
        except Exception as exc:
            abandoned = future is not None and not future.done()
            self.failures = 2 if abandoned else self.failures + 1
            ctx.log('planner_failure', error=type(exc).__name__, calls=self.calls,
                    abandoned=abandoned)
            return
        if not data:
            ctx.log('planner_unparsed', calls=self.calls)
            return
        new = dict(self.plan)
        if isinstance(data.get('targets'), list):
            targets = []
            for row in data['targets'][:24]:
                if isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str) and row[0] in o['supply']:
                    targets.append([row[0], finite_int(row[1], 0, 0, 20)])
            if targets or not data['targets']:
                new['targets'] = targets
        for key, lo, hi in [('province_at', 3, 8), ('duchy_at', 1, 6), ('estate_at', 0, 3), ('copper_floor', 3, 8)]:
            if key in data:
                new[key] = finite_int(data[key], new[key], lo, hi)
        if data.get('mode') in ('money', 'engine', 'gardens'):
            new['mode'] = data['mode']
        if isinstance(data.get('note'), str):
            new['note'] = data['note'][:500]
        self.plan = new
        if isinstance(data.get('buy_now'), str):
            self.override = (o['decision_id'], data['buy_now'])
        ctx.memory['dominion_plan'] = new
        ctx.log('plan', turn=turn, plan=new, override=self.override)

    def provinces(self, o):
        return o.get('supply', {}).get('Province', {}).get('count', 8)

    def total_money(self, cards):
        return sum(cards.get(c, 0)*n for c, n in TREASURE.items()) + 2*cards.get('Militia', 0) + 2*cards.get('Festival', 0) + cards.get('Market', 0)

    def trash_ids(self, o, opts, maximum=None):
        p = o['prompt']
        source = p.get('source') or ''
        owned = Counter(o['self']['owned'])
        maximum = p.get('max', 1) if maximum is None else maximum
        remaining_money = self.total_money(owned)
        early = self.provinces(o) > 3
        ranked = []
        for op in opts:
            c = op['label']
            value = 0
            if c == 'Curse': value = 100
            elif c == 'Estate' and early and self.plan['mode'] != 'gardens': value = 60
            elif c == 'Copper' and early and self.plan['mode'] != 'gardens': value = 20
            elif c == 'Chapel' and source == 'Sentry' and owned.get('Curse', 0)+owned.get('Estate', 0)+owned.get('Copper', 0) == 0: value = 15
            if value:
                ranked.append((value, op))
        chosen = []
        for _, op in sorted(ranked, key=lambda x: -x[0]):
            if len(chosen) >= maximum:
                break
            c = op['label']
            if c == 'Copper':
                if remaining_money - 1 < self.plan['copper_floor']:
                    continue
                remaining_money -= 1
            chosen.append(op['id'])
        return chosen

    def hand_value(self, cards, o):
        """Small visible-hand utility for forced discard combinatorial search."""
        counts = Counter(cards)
        value = sum(TREASURE.get(c, 0) for c in cards)
        extra = counts['Village'] + counts['Festival']
        draws = {'Smithy': 2.7, 'Witch': 2.8 if o['supply'].get('Curse', {}).get('count', 0) else 1.8,
                 'Militia': 2.6, 'Moat': 1.8, 'Workshop': 1.7, 'Mine': 1.2, 'Remodel': 1.0,
                 'Chapel': 2.3 if counts['Curse']+counts['Estate'] >= 2 and self.provinces(o)>3 else 0.1}
        terminal_values = sorted((draws[c] for c in cards if c in draws), reverse=True)
        value += sum(terminal_values[:1+extra])
        for c, n in counts.items():
            value += n * {'Village': 0.7 if terminal_values else 0.25, 'Festival': 2.1,
                         'Market': 2.0, 'Laboratory': 2.5, 'Merchant': 1.3, 'Sentry': 2.3,
                         'Cellar': 0.8}.get(c, 0)
        return value

    def play_value(self, c, o):
        hand = Counter(o['self'].get('hand', {}))
        actions = o.get('resources', {}).get('actions', 1)
        draw_available = o['self'].get('deck_count', 0)+o['self'].get('discard_count', 0)
        if c in CANTRIPS:
            if c == 'Cellar': return 25
            return {'Village': 100, 'Festival': 99, 'Sentry': 98, 'Market': 96,
                    'Laboratory': 95, 'Merchant': 94}[c]
        if c == 'Chapel':
            options = [{'id': str(i), 'label': v} for i,v in enumerate(hand.elements()) if v != 'Chapel']
            trashes = self.trash_ids(o, options, 4)
            if not trashes: return -1
            curse_n = hand.get('Curse', 0)
            return 60 + 5*len(trashes)+8*curse_n if actions == 1 else 15
        if c == 'Witch':
            return 85 if o['supply'].get('Curse', {}).get('count', 0) else (45 if draw_available else -1)
        if c == 'Smithy': return 65 if draw_available else -1
        if c == 'Militia': return 58 if actions == 1 else 30
        if c == 'Moat': return 40 if draw_available else -1
        if c == 'Mine': return 28 if hand['Copper']+hand['Silver'] else -1
        if c == 'Workshop': return 20
        if c == 'Remodel':
            if hand['Curse'] or hand['Estate'] and self.provinces(o)>2: return 42
            if hand['Gold'] and self.provinces(o)<=3: return 55
            return -1
        return 1

    def finishing_basket(self, o):
        """Enumerate affordable scoring/ending baskets; verify entire-turn VP exactly."""
        supply = o['supply']
        buys = min(o['resources'].get('buys', 1), 6)
        coins = o['resources'].get('coins', 0)
        if buys <= 0:
            return None
        empty = sum(v['count'] == 0 for v in supply.values())
        if supply['Province']['count'] > buys and empty + sum(0 < v['count'] <= buys for v in supply.values()) < 3:
            return None
        names = [c for c,v in supply.items() if v['count']>0 and v['cost']<=coins and
                 (c in ('Province','Duchy','Estate','Gardens','Copper') or v['count']<=buys)]
        names.sort(key=lambda c: (-POINTS.get(c, 0), c))
        base = Counter(o['self']['owned'])
        opponent_vp = o['opponent']['vp']
        best = None
        best_score = -math.inf
        visited = 0
        def visit(index, money, left, gained, used):
            nonlocal best, best_score, visited
            visited += 1
            if visited > 12000:
                return
            new_empty = empty + sum(n == supply[c]['count'] for c,n in used.items())
            end = used.get('Province', 0) == supply['Province']['count'] or new_empty>=3
            if end:
                margin = vp(base + Counter(gained)) - opponent_vp
                turns = o.get('turns_taken', {})
                win = margin>0 or (margin==0 and turns.get('agent', 0)<=turns.get('opponent', 0))
                quality = margin + 0.001*money
                if win and quality > best_score:
                    best_score = quality
                    best = list(gained)
            if left == 0:
                return
            for i in range(index, len(names)):
                c = names[i]
                if used.get(c,0)>=supply[c]['count'] or supply[c]['cost']>money:
                    continue
                used[c] = used.get(c,0)+1
                gained.append(c)
                visit(i, money-supply[c]['cost'], left-1, gained, used)
                gained.pop()
                used[c] -= 1
                if not used[c]: del used[c]
        visit(0, coins, buys, [], {})
        return best

    def losing_end(self, c, o):
        supply = o['supply']
        empty = sum(v['count']==0 for v in supply.values())
        if supply[c]['count'] != 1 or not (c=='Province' or empty>=2):
            return False
        cards = Counter(o['self']['owned']); cards[c] += 1
        margin = vp(cards)-o['opponent']['vp']
        t = o.get('turns_taken', {})
        return margin < 0 or (margin==0 and t.get('agent',0)>t.get('opponent',0))

    def acquisition(self, o, opts):
        allowed = {op['label']:op['id'] for op in opts}
        owned = o['self']['owned']
        provinces = self.provinces(o)
        p = o['prompt']
        buy = p['kind']=='buy'
        if buy:
            basket = self.finishing_basket(o)
            if basket:
                # Subsequent prompts re-enumerate with updated money, piles and VP.
                return [allowed[basket[0]]] if basket[0] in allowed else []
        safe = {c:i for c,i in allowed.items() if not self.losing_end(c,o)}
        if not safe:
            safe = allowed
        if buy and self.override and self.override[0]==o['decision_id']:
            name = self.override[1]
            if name in safe:
                return [safe[name]]
            if name.lower()=='pass' and p.get('min',0)==0:
                return []
        if p.get('source')=='Mine':
            for c in ('Gold','Silver','Copper'):
                if c in safe: return [safe[c]]
        for c,key in [('Province','province_at'),('Duchy','duchy_at'),('Estate','estate_at')]:
            if c in safe and provinces<=self.plan[key]:
                return [safe[c]]
        if 'Gardens' in safe and self.plan['mode']=='gardens' and (sum(owned.values())>=20 or owned.get('Workshop',0)>=2):
            return [safe['Gardens']]
        for c,n in self.plan['targets']:
            if c in safe and owned.get(c,0)<n:
                if c=='Witch' and not o['supply'].get('Curse',{}).get('count',0) and owned.get(c,0):
                    continue
                return [safe[c]]
        for c in ('Gold','Silver'):
            if c in safe: return [safe[c]]
        if self.plan['mode']=='gardens' and 'Copper' in safe and (owned.get('Gardens',0) or sum(owned.values())>=28):
            return [safe['Copper']]
        if p.get('min',0):
            # A mandatory gain cannot pass. Prefer useful cantrips to junk.
            c = max(safe, key=lambda c: {'Market':8,'Laboratory':8,'Merchant':5,'Village':4,'Moat':3,
                   'Estate':2,'Copper':1,'Curse':-100}.get(c,2))
            return [safe[c]]
        return []

    def choose(self, o):
        p = o['prompt']; opts = p.get('options', [])
        kind = p.get('kind'); source = p.get('source') or ''
        if not opts:
            return []
        if kind in ('buy','gain'):
            return self.acquisition(o, opts)
        if kind=='play_action':
            best = max(opts, key=lambda op:self.play_value(op['label'],o))
            return [best['id']] if self.play_value(best['label'],o)>0 else []
        if kind=='trash':
            if source=='Mine':
                labels = {x['label']:x['id'] for x in opts}
                for c,target in [('Silver','Gold'),('Copper','Silver')]:
                    if c in labels and o['supply'].get(target,{}).get('count',0):
                        return [labels[c]]
                return []
            if source=='Remodel':
                labels = {x['label']:x['id'] for x in opts}
                if 'Gold' in labels and self.provinces(o)<=3 and not self.losing_end('Province',o):
                    return [labels['Gold']]
                for c in ('Curse','Estate','Copper'):
                    if c in labels: return [labels[c]]
                return [min(opts,key=lambda x:o['supply'].get(x['label'],{}).get('cost',0))['id']]
            return self.trash_ids(o, opts)
        if kind=='discard':
            if p.get('min',0)>0:
                n=p['min']; size=len(opts)
                if size<=12:
                    choices=itertools.combinations(range(size),n)
                    ids = min(choices,key=lambda ix:-self.hand_value([op['label'] for i,op in enumerate(opts) if i not in ix],o))
                    return [opts[i]['id'] for i in ids]
                return [x['id'] for x in sorted(opts,key=lambda x:self.hand_value([x['label']],o))[:n]]
            # Cellar and Sentry: discard dead green, plus Coppers if draw pool is richer.
            hand = Counter(o['self'].get('hand',{}))
            actions = o.get('resources',{}).get('actions',0)
            owned=o['self']['owned']
            dense = (2*owned.get('Silver',0)+3*owned.get('Gold',0)) / max(1,sum(owned.values())) > 0.85
            out=[]
            for op in opts:
                c=op['label']
                dead = c in ('Curse','Estate','Duchy','Province','Gardens')
                if c=='Copper' and dense: dead=True
                if c=='Chapel' and not self.trash_ids(o,[{'id':str(i),'label':v} for i,v in enumerate(hand.elements())],4): dead=True
                if c in TERMINALS and actions==0: dead=True
                if dead: out.append(op['id'])
            return out
        labels={str(x['label']).lower():x['id'] for x in opts}
        if 'true' in labels and 'false' in labels:
            if source=='Sentry' and len(p.get('relevant_cards',[]))==2:
                cards=p['relevant_cards']
                def cardname(x):
                    return x.get('label',x.get('name','')) if isinstance(x,dict) else str(x)
                first,second=map(cardname,cards)
                swap=self.hand_value([second],o)>self.hand_value([first],o)
                return [labels['true' if swap else 'false']]
            return [labels['true' if source=='Moat' or 'reveal' in p.get('instruction','').lower() else 'false']]
        return [op['id'] for op in opts[:p.get('min',0)]]

    def validate(self, ids, p):
        allowed={str(op['id']) for op in p.get('options',[])}
        result=[]
        for i in ids:
            if str(i) in allowed and str(i) not in result:
                result.append(str(i))
        result=result[:p.get('max',1)]
        for op in p.get('options',[]):
            if len(result)>=p.get('min',0): break
            if str(op['id']) not in result: result.append(str(op['id']))
        return result

    def run_episode(self, game):
        ctx=self.ctx
        o=observation(ctx.observe())
        while not ctx.done:
            if not o or o.get('terminal') or not o.get('prompt'):
                return
            if self.plan is None:
                self.plan=self.default_plan(o)
            p=o['prompt']; did=o.get('decision_id')
            self.attempts[did]+=1
            if p.get('kind')=='buy' and self.attempts[did]==1:
                self.plan_now(o)
            if ctx.done: return
            try:
                ids=self.choose(o)
            except Exception as exc:
                ctx.log('local_policy_failure',error=type(exc).__name__,kind=p.get('kind'),source=p.get('source'))
                ids=[]
            if self.attempts[did]>1:
                ids=[op['id'] for op in p.get('options',[])[:p.get('min',0)]]
            if self.attempts[did]>3:
                ctx.log('stalled_prompt',decision_id=did)
                return
            ids=self.validate(ids,p)
            selected=[op['label'] for op in p.get('options',[]) if str(op['id']) in ids]
            if p.get('kind') in ('buy','gain') and selected:
                self.history.append({'turn':o.get('turns_taken',{}).get('agent',0),'kind':p['kind'],'cards':selected})
            action='choose '+str(did)+(' '+' '.join(ids) if ids else '')
            try:
                raw=ctx.act(action)
                o=observation(raw if raw is not None else ctx.observe())
            except Exception as exc:
                if ctx.done: return
                ctx.log('action_failure',error=type(exc).__name__,decision_id=did)
                o=observation(ctx.observe())


def build(ctx):
    return Harness(ctx)
