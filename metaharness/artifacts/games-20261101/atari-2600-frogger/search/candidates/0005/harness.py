"""Frogger crossover: frame executor, geometric proposals, asynchronous advisers.
All perception is from public observations; no emulator or implementation access.
"""
from __future__ import annotations
import json
import math
from collections import deque
from metaharness.models.jev import Choice, Noul
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

ACTIONS = ('UP', 'RIGHT', 'LEFT', 'DOWN', 'NOOP')

def number(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError):
        return default

def observation(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}

def row_of(o):
    r = int(number(o.get('frog', {}).get('row'), 255))
    return -1 if r == 255 else r

def life_key(o):
    return (o.get('deaths', 0), o.get('frogs_home', 0), o.get('level', 1))

def bounded_json(text):
    if not isinstance(text, str):
        return {}
    try:
        start, end = text.find('{'), text.rfind('}')
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.history = {}
        self.speed = {}
        self.previous = None
        self.last_key = None
        self.hop_px = 8.0
        self.hop_frames = 10
        self.next_hop = 0
        self.press = None
        self.press_tick = -100
        self.press_origin = None
        self.release_until = 0
        self.target = None
        self.plan_version = 0
        self.policy = {'route': 'nearest_open', 'pace': 'balanced', 'focus': 'forward'}
        self.jev_future = None
        self.astra_future = None
        self.jev_meta = None
        self.astra_meta = None
        self.next_jev = 0
        self.next_astra = 0
        self.session = None
        self.advice = None
        self.failures = deque(maxlen=12)
        self.events = deque(maxlen=16)
        self.last_progress = 0
        self.last_action = 'NOOP'
        self.last_candidates = []
        self.counter = 0

    def log(self, name, **data):
        self.ctx.log(name, **data)

    def update(self, o):
        tick = int(number(o.get('tick')))
        key = life_key(o)
        if key != self.last_key:
            if self.last_key is not None:
                e = {'tick': tick, 'before': self.last_key, 'after': key,
                     'frog_before': self.previous.get('frog') if self.previous else {},
                     'last_action': self.last_action, 'reason': o.get('terminal_reason')}
                self.events.append(e)
                if key[0] != self.last_key[0]:
                    self.failures.append(e)
                self.log('life_or_home_transition', **e)
            self.last_key = key
            self.target = None
            self.advice = None
            self.press = None
            self.release_until = tick + 3
            self.next_hop = tick + 3
            self.last_progress = tick
            self.plan_version += 1
        old = self.previous
        if old and (row_of(o) != row_of(old) or o.get('score') != old.get('score')):
            self.last_progress = tick
            self.log('progress', tick=tick, frog=o.get('frog'), score=o.get('score'),
                     deaths=o.get('deaths'), timer=o.get('timer_s'),
                     ram_fields=o.get('ram_fields'), action=self.last_action)
        for lane in o.get('road', []) + o.get('river', []):
            for obj in lane.get('objects', []):
                ident = obj.get('id', str((lane.get('row'), obj.get('index'))))
                x = number(obj.get('x'))
                q = self.history.setdefault(ident, deque(maxlen=32))
                if not q or q[-1][0] != tick:
                    q.append((tick, x))
                candidates = [v for v in q if 12 <= tick-v[0] <= 48]
                if candidates:
                    t0, x0 = candidates[0]
                    dx = (x - x0 + 80) % 160 - 80
                    v = dx / max(1, tick-t0)
                    if abs(v) <= 2:
                        self.speed[ident] = v
                if ident not in self.speed:
                    sign = -1 if obj.get('direction', lane.get('direction')) == 'left' else 1
                    self.speed[ident] = sign / (int(number(lane.get('activity_mask'), 3)) + 1)
        self.previous = o

    def velocity(self, lane):
        vs = [self.speed.get(obj.get('id'), 0) for obj in lane.get('objects', [])]
        return sum(vs) / len(vs) if vs else 0.0

    def lanes(self, o):
        return {int(l['row']): l for l in o.get('road', []) + o.get('river', []) if 'row' in l}

    def intervals(self, lane, dt):
        out = []
        for obj in lane.get('objects', []):
            v = self.speed.get(obj.get('id'), self.velocity(lane))
            x = (number(obj.get('x')) + v * dt) % 160
            width = number(obj.get('width_px_estimate'), 8)
            for shift in (-160, 0, 160):
                out.append((x+shift, x+width+shift, obj))
        return out

    def safety(self, lanes, homes, row, x, dt, difficulty=0):
        if x < 5 or x > 151:
            return -100.0
        if row in (-1, 5):
            return 24.0
        if row >= 11:
            valid = [h for h in homes if not h.get('occupied') and h.get('available', True)
                     and h.get('crocodile') not in (True, 'dangerous', 'present')]
            return max((5.0-abs(x-number(h.get('x_center'))) for h in valid), default=-100.0)
        lane = lanes.get(row)
        if lane is None:
            return -100.0
        ints = self.intervals(lane, dt)
        if row < 5:
            # Frog is treated as a small sprite, with a one-pixel margin.
            clearance = 100.0
            for a, b, obj in ints:
                if a <= x <= b:
                    return -8.0
                clearance = min(clearance, a-x-5 if x<a else x-b-5)
            return clearance
        clearance = -100.0
        for a, b, obj in ints:
            if obj.get('hazard') or obj.get('kind') not in ('log', 'turtle'):
                continue
            clearance = max(clearance, min(x-a-3, b-x-3))
        return clearance

    def transition(self, lanes, homes, r, x, t, action):
        dur = self.hop_frames
        nr, nx = r, x
        if action == 'UP':
            nr += 1
        elif action == 'DOWN':
            nr -= 1
        elif action == 'LEFT':
            nx -= self.hop_px
        elif action == 'RIGHT':
            nx += self.hop_px
        if nr < -1 or nr > 11 or (r == -1 and action == 'DOWN'):
            return None
        v = self.velocity(lanes[nr]) if nr in lanes and nr >= 6 else 0.0
        # Check destination at landing and during the release/recovery window.
        samples = (1, dur//2, dur)
        margin = min(self.safety(lanes, homes, nr, nx+v*max(0,s-1), t+s) for s in samples)
        if margin < 0:
            return None
        return nr, nx+v*(dur-1), t+dur, margin

    def choose_target(self, o):
        x = number(o.get('frog', {}).get('x'), 80)
        homes = [h for h in o.get('homes', []) if not h.get('occupied') and h.get('available', True)]
        if not homes:
            return 80.0
        centers = [number(h.get('x_center'), 80) for h in homes]
        if self.target not in centers:
            route = self.policy.get('route')
            if route == 'left_open':
                self.target = min(centers)
            elif route == 'right_open':
                self.target = max(centers)
            else:
                self.target = min(centers, key=lambda v: abs(v-x))
        return self.target

    def rank(self, o):
        r = row_of(o)
        x = number(o.get('frog', {}).get('x'), 80)
        lanes, homes = self.lanes(o), o.get('homes', [])
        target = self.choose_target(o)
        candidates = []
        # Short, bounded receding-horizon beam search; never calls game.tick.
        for first in ACTIONS:
            root = self.transition(lanes, homes, r, x, 0, first)
            if root is None:
                continue
            nr,nx,t,margin = root
            states = [(nr,nx,t,margin,0.0)]
            best = -1e9
            bestrow = nr
            for depth in range(4):
                next_states = []
                for rr,xx,tt,mm,cost in states:
                    home_distance = abs(xx-target)
                    val = rr*30.0 + min(mm,12)*0.25 - cost
                    if rr >= 5:
                        val -= home_distance * (0.55 if rr >= 8 else 0.12)
                    if rr >= 11:
                        val += 1000
                    if val > best:
                        best, bestrow = val, rr
                    if depth == 3 or rr >= 11:
                        continue
                    for action in ACTIONS:
                        nxt = self.transition(lanes, homes, rr, xx, tt, action)
                        if nxt:
                            r2,x2,t2,m2 = nxt
                            penalty = 0.8 if action == 'NOOP' else (2.0 if action == 'DOWN' else 0.4)
                            next_states.append((r2,x2,t2,min(mm,m2),cost+penalty))
                # Distinct row/x states prevent combinatorial waiting duplicates.
                seen = set(); states=[]
                next_states.sort(key=lambda s: s[0]*30-abs(s[1]-target)*0.15-s[4], reverse=True)
                for s in next_states:
                    key=(s[0], round(s[1]/3))
                    if key not in seen:
                        seen.add(key); states.append(s)
                    if len(states)>=12:
                        break
            immediate = nr*0.9 + min(margin,8)*0.12
            if first == 'NOOP':
                immediate -= 0.3
            if first == 'DOWN':
                immediate -= 1
            candidates.append({'action':first, 'value':best+immediate, 'margin':round(margin,2),
                               'reachable_row':bestrow, 'landing_row':nr, 'landing_x':round(nx,2),
                               'advances':nr>r, 'toward_home':abs(nx-target)<abs(x-target),
                               'safety':'wide' if margin>=4 else 'narrow'})
        candidates.sort(key=lambda c:c['value'], reverse=True)
        self.last_candidates = candidates
        return candidates

    def start_jev(self, o, candidates):
        ctx=self.ctx
        if self.jev_future is not None or not candidates:
            return
        if getattr(ctx,'system1',True) is None or ctx.budget.remaining().get('system1_calls',0)<2:
            return
        tick=int(number(o.get('tick')))
        key=life_key(o); version=self.plan_version; row=row_of(o)
        shortlist=candidates[:3]
        options={c['action']:{'what':'Choose this currently feasible one-hop proposal.',
                             'facts':{k:v for k,v in c.items() if k not in ('value','action')}} for c in shortlist}
        options['UNKNOWN']='No supported selection, or candidate facts are insufficient.'
        state={'rules':'Frogger. Code validates geometry every frame. Never invent safety from unknown crocodile or diving carriers. Prefer forward progress and an unoccupied home; do not stay idle merely because water exists elsewhere.',
               'plan':dict(self.policy), 'zone':o.get('frog',{}).get('zone'),
               'urgent':row>=6 and candidates[0]['safety']=='narrow',
               'timer':'urgent' if number(o.get('timer_s'),30)<6 else 'normal',
               'ranked_candidates':[{k:v for k,v in c.items() if k!='value'} for c in shortlist],
               'current_wait_supported':any(c['action']=='NOOP' for c in candidates),
               'home_uncertainty':'Crocodile is unknown, not confirmed absent.'}
        instructions=('Select a feasible action in ranked_candidates consistent with plan. The ordering is a code-computed progress ranking, not an option-order preference. Prefer the leading proposal unless another has clearly better semantic safety or follows plan focus. UNKNOWN if unsupported. No arithmetic is needed.')
        questions={'move':Choice(instructions,options),
                   'check':Choice(instructions,dict(reversed(list(options.items())))),
                   'escape':Noul('Do the stated current facts require leaving the current carrier rather than waiting? True only when urgent is true; otherwise false.')}
        self.jev_meta={'tick':tick,'key':key,'version':version,'row':row,'x':number(o.get('frog',{}).get('x'),80)}
        def request():
            try:
                start=ctx.clock.now()
                answer=ctx.decide(state,questions,timeout_s=3.0)
                return answer,ctx.clock.now()-start
            except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                raise
            except (TimeoutError,RuntimeError,TypeError):
                return None,0
        self.jev_future=ctx.spawn(request)
        self.next_jev=tick+24

    def start_astra(self,o):
        ctx=self.ctx
        if self.astra_future is not None or getattr(ctx,'system2',True) is None:
            return
        if ctx.budget.remaining().get('system2_calls',0)<2:
            return
        snapshot={'frog':o.get('frog'),'score':o.get('score'),'deaths':o.get('deaths'),
                  'homes':o.get('homes'),'timer_s':o.get('timer_s'),'mode':o.get('mode'),
                  'policy':self.policy,'recent_outcomes':list(self.events),
                  'last_candidates':self.last_candidates[:3]}
        prompt=('You are the slow strategic adviser for Atari Parker Brothers Frogger. A concurrent frame controller handles joystick pulses, NOOP release, measured velocities, collision rejection and short trajectory search. It never waits for you. The old Jev parent held UP forever, yielding one point then timeout; the old Astra parent often never acted before game over. Choose a durable policy, not an action tied to this stale snapshot. Survival and five distinct homes dominate optional bonuses. River geometry is estimated, diving/unknown carriers are rejected. Revise policy after repeated deaths or stalled home progress. Return only a tiny JSON object with route one of nearest_open,left_open,right_open; pace one of balanced,cautious,urgent; focus one of forward,wide_platform,home_alignment. Do not add code or prose. Jev receives this plan plus precomputed feasible candidates, zone, urgency, timer bucket, and supported-wait boolean. It answers two reordered Choice questions over those candidates, plus a Noul for urgent escape. Code accepts only agreeing choices with winning probabilities >=0.65, row/life/plan unchanged, <=30 ticks old, and latest geometric safety. Focus wide_platform prioritizes wide candidates over narrow; home_alignment prioritizes towards-home alternatives on river. Snapshot: '+json.dumps(snapshot,separators=(',',':')))
        version=self.plan_version
        session=self.session
        def request():
            try:
                return ctx.think(prompt,session_id=session,timeout_s=120.0)
            except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                raise
            except (TimeoutError,RuntimeError):
                return None
        self.astra_future=ctx.spawn(request)
        self.astra_meta={'version':version,'level':o.get('level',1)}
        self.next_astra=int(number(o.get('tick')))+3600

    def poll(self,o):
        tick=int(number(o.get('tick')))
        if self.astra_future is not None and self.astra_future.done():
            future,self.astra_future=self.astra_future,None
            res=future.result()
            if res is not None:
                self.session=getattr(res,'session_id',None)
                parsed=bounded_json(getattr(res,'text',''))
                allowed={'route':('nearest_open','left_open','right_open'),
                         'pace':('balanced','cautious','urgent'),
                         'focus':('forward','wide_platform','home_alignment')}
                changes={k:v for k,v in parsed.items() if k in allowed and v in allowed[k]}
                if changes and o.get('level',1)==self.astra_meta['level']:
                    self.policy.update(changes);self.target=None;self.plan_version+=1
                    self.log('astra_policy_adopted',tick=tick,policy=dict(self.policy))
                else:
                    self.log('astra_output_rejected',tick=tick)
        if self.jev_future is not None and self.jev_future.done():
            future,self.jev_future=self.jev_future,None
            answer,latency=future.result()
            meta=self.jev_meta
            if not isinstance(answer,dict):
                return
            a,b=answer.get('move'),answer.get('check')
            chosen=getattr(a,'value',None)
            pa=number(getattr(a,'probabilities',{}).get(chosen,0))
            pb=number(getattr(b,'probabilities',{}).get(chosen,0))
            fresh=(life_key(o)==meta['key'] and self.plan_version==meta['version']
                   and row_of(o)==meta['row'] and tick-meta['tick']<=30
                   and abs(number(o.get('frog',{}).get('x'),80)-meta['x'])<=10)
            if fresh and chosen in ACTIONS and chosen==getattr(b,'value',None) and min(pa,pb)>=0.65:
                self.advice={'action':chosen,'until':meta['tick']+30,'key':meta['key'],
                             'row':meta['row'],'version':meta['version']}
                self.log('jev_advice_accepted',tick=tick,action=chosen,age=tick-meta['tick'],latency_s=latency,probability=min(pa,pb))
            else:
                self.log('jev_advice_discarded',tick=tick,age=tick-meta['tick'],fresh=fresh,latency_s=latency)

    def controller(self,o,candidates):
        tick=int(number(o.get('tick')))
        r=row_of(o); x=number(o.get('frog',{}).get('x'),80)
        # Always perform the electrical release; do not delegate it to a model.
        if self.press is not None:
            origin=self.press_origin
            moved=r!=origin[0] or abs(x-origin[1])>=max(3,self.hop_px*0.7)
            if moved or tick-self.press_tick>=4:
                if moved and r==origin[0] and self.press in ('LEFT','RIGHT'):
                    delta=abs(x-origin[1])
                    if 4<=delta<=20:
                        self.hop_px=delta
                self.log('pulse_released',tick=tick,action=self.press,moved=moved,origin=origin,
                         frog=o.get('frog'),elapsed=tick-self.press_tick)
                self.press=None;self.release_until=tick+3
                return 'NOOP'
            return self.press
        if tick<self.release_until or tick<self.next_hop:
            return 'NOOP'
        if o.get('terminal') or r not in range(-1,12):
            return 'NOOP'
        if not candidates:
            return 'NOOP'
        selected=candidates[0]
        # Durable policy changes only choose among geometrically feasible routes.
        close=[c for c in candidates if c['value']>=selected['value']-12]
        if self.policy.get('focus')=='wide_platform' and r>=5:
            selected=max(close,key=lambda c:(c['safety']=='wide',c['value']))
        elif self.policy.get('focus')=='home_alignment' and r>=8:
            selected=max(close,key=lambda c:(c['toward_home'],c['value']))
        adv=self.advice
        if adv and tick<=adv['until'] and adv['row']==r and adv['key']==life_key(o) and adv['version']==self.plan_version:
            match=next((c for c in close if c['action']==adv['action']),None)
            if match:
                if match['action']!=selected['action']:
                    self.log('jev_changes_action',tick=tick,before=selected['action'],after=match['action'])
                selected=match
            self.advice=None
        action=selected['action']
        if action!='NOOP':
            self.press=action;self.press_tick=tick;self.press_origin=(r,x)
            self.next_hop=tick+self.hop_frames
            self.log('hop_issued',tick=tick,action=action,frog=o.get('frog'),proposal=selected,
                     policy=dict(self.policy),candidates=candidates[:3])
        return action

    def run_episode(self,game):
        ctx=self.ctx
        while not ctx.done:
            o=observation(ctx.observe())
            if not o:
                ctx.act('NOOP');ctx.wait_for_tick();continue
            self.update(o)
            tick=int(number(o.get('tick')))
            self.poll(o)
            # Search only at action boundaries; motor release always has priority.
            ready=self.press is None and tick>=self.next_hop and tick>=self.release_until
            candidates=self.rank(o) if ready else self.last_candidates
            if ready and tick>=self.next_jev:
                self.start_jev(o,candidates)
            if tick>=self.next_astra and (tick==0 or len(self.failures)>0 or tick>1800):
                self.start_astra(o)
            action=self.controller(o,candidates)
            self.last_action=action
            ctx.act(action)
            ctx.wait_for_tick()

def build(ctx):
    return Harness(ctx)
