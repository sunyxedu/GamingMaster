"""0010: learn shape-local contact constraints, not global geometric margins.

Public observations only. The archived positive-margin deaths concentrate on the
right end of stretched shapes 163/170. Those shapes are not solid safe carriers.
Keep a cold-start, episode-local contact book and separate landing failures from
home-entry and supposedly-safe-bank failures.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.models.jev import Choice, Noul
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


# Atari NUSIZ low bits describe separated sprite copies, not a solid bounding box.
COPIES = ((0,), (0, 16), (0, 32), (0, 16, 32), (0, 64), (0,), (0, 32, 64), (0,))
SIZES = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (('UP', 1, 0), ('LEFT', 0, -8), ('RIGHT', 0, 8), ('NOOP', 0, 0), ('DOWN', -1, 0))


def num(v, default=0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def unpack(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}


def small_json(text):
    if not isinstance(text, str):
        return {}
    try:
        return unpack(text[text.index('{'):text.rindex('}') + 1])
    except (ValueError, TypeError):
        return {}


def row(o):
    r = int(num(o.get('frog', {}).get('row'), 255))
    return -1 if r == 255 else r


def epoch(o):
    return (o.get('deaths', 0), o.get('frogs_home', 0), o.get('level', 1))


def segment(o):
    return 'river' if row(o) >= 5 and row(o) <= 11 else 'road'


def merge(intervals):
    merged = []
    # Contact-book bands are tuples while the merged representation is a list;
    # sort by numeric endpoints rather than asking Python to compare tuple/list
    # values when two bands begin at the same coordinate.
    normalized = [(float(a), float(b)) for a, b in intervals]
    for a, b in sorted(normalized, key=lambda z: (z[0], z[1])):
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prev = None
        self.hist = {}
        self.vel = {}
        self.period = {}
        self.step = 6
        self.ready = 3
        self.last_press = -100
        self.last_action = 'NOOP'
        self.last_origin = None
        self.pressed = None
        self.last_epoch = None
        self.last_segment = None
        self.goal = None
        self.goal_until = 0
        self.jev_future = None
        self.jev_meta = None
        self.next_jev = 0
        self.astra_future = None
        self.astra_meta = None
        self.next_astra = 0
        self.session = None
        self.version = 0
        self.policy = {'route': 'balanced', 'road_clearance': 'normal', 'river_support': 'center'}
        self.routes = []
        self.last_plan = None
        self.failures = deque(maxlen=8)
        self.recent = deque(maxlen=20)
        self.road_pad = 1.5
        self.river_pad = 2.0
        self.search_ms = []
        self.last_progress = 0
        self.last_log = -100
        self.s1_latency = deque(maxlen=20)
        self.contact_bans = {}
        self.contact_examples = deque(maxlen=12)
        self.contact_revision = 0
        self.bank_unsafe = False
        self.diagnosis = None
        self.diagnosis_evidence = None
        self.last_diagnosis_revision = -1
        self.home_window = (-5, 1)
        self.policy['focus'] = 'contact'
        self.observe_ring = deque(maxlen=20)
        self.guard_vetoes = 0

    def event(self, name, **data):
        self.ctx.log(name, **data)

    def compact(self, o):
        return {'tick': o.get('tick'), 'frog': o.get('frog'), 'timer': o.get('timer_s'),
                'score': o.get('score'), 'deaths': o.get('deaths'),
                'lanes': [{'row': l.get('row'), 'objects': [
                    {k: z.get(k) for k in ('id', 'x', 'nusiz', 'kind', 'hazard', 'shape_ptr')}
                    for z in l.get('objects', [])]} for l in o.get('road', []) + o.get('river', [])]}

    def update(self, o):
        t = int(num(o.get('tick')))
        if self.prev and num(o.get('deaths')) > num(self.prev.get('deaths')):
            self.learn_failure(o)
        self.observe_ring.append(o)
        k, s = epoch(o), segment(o)
        if self.last_epoch != k or self.last_segment != s:
            self.goal = None
            self.next_jev = t
            if self.last_epoch is not None and k != self.last_epoch:
                death = k[0] != self.last_epoch[0]
                record = {'tick': t, 'death': death, 'before': self.last_epoch, 'after': k,
                          'last_plan': self.last_plan, 'previous': self.compact(self.prev or {}),
                          'current': self.compact(o)}
                self.recent.append(record)
                if death:
                    self.failures.append(record)
                self.event('life_or_home', **record)
                self.ready = t + 3
            self.last_epoch, self.last_segment = k, s
        if self.prev:
            if row(o) != row(self.prev) or o.get('score') != self.prev.get('score'):
                self.last_progress = t
                self.event('progress', tick=t, frog=o.get('frog'), score=o.get('score'),
                           deaths=o.get('deaths'), homes=o.get('frogs_home'))
            if self.last_origin and t - self.last_press <= 4:
                if row(o) != self.last_origin[0] or abs(num(o.get('frog', {}).get('x')) - self.last_origin[1]) >= 7:
                    self.event('pulse_observed', tick=t, origin=self.last_origin,
                               action=self.last_action, frog=o.get('frog'), delay=t-self.last_press)
                    self.last_origin = None
        for lane in o.get('road', []) + o.get('river', []):
            r = int(num(lane.get('row')))
            for z in lane.get('objects', []):
                ident = str(z.get('id', (r, z.get('index'))))
                x = num(z.get('x'))
                q = self.hist.setdefault(ident, deque(maxlen=40))
                if not q or t != q[-1][0]:
                    q.append((t, x))
                old = next((v for v in q if 24 <= t-v[0] <= 48), None)
                sign = -1 if z.get('direction', lane.get('direction')) == 'left' else 1
                default = sign / (int(num(lane.get('activity_mask'), 3)) + 1)
                v = default
                if old:
                    dx = (x-old[1]+80) % 160 - 80
                    observed = dx / max(1, t-old[0])
                    if observed * sign > 0 and abs(observed) <= 2:
                        # These lanes move in integral pixels at an activity cadence.
                        choices = (1/16, 1/8, 1/4, 1/2, 1, 2)
                        v = sign * min(choices, key=lambda p: abs(abs(observed)-p))
                self.vel[ident] = v
        self.prev = o

    def learn_failure(self, o):
        """Learn only from public before/after observations, with cause separation."""
        old = self.prev
        r = row(old)
        x = num(old.get('frog', {}).get('x'))
        t = int(num(o.get('tick')))
        plan = self.last_plan or {}
        origin = plan.get('origin', (-2, 0))
        age = t - int(num(plan.get('tick'), t-99))
        planned_action = plan.get('action')
        actual_action = self.last_action if self.last_action in ('UP', 'RIGHT', 'LEFT', 'DOWN', 'NOOP') else planned_action
        # The route certificate can be many frames old. Cause attribution uses
        # the action that was actually posted immediately before this death;
        # the old plan is retained only as context for timing.
        home_entry = r == 10 and actual_action == 'UP' and age <= 5
        bank = r == 5 and actual_action in ('LEFT', 'RIGHT', 'NOOP')
        transition = r == 5 and actual_action in ('UP', 'DOWN')
        landing = ((6 <= r <= 10 and not home_entry) or transition) and age <= 12
        ev = {'tick': t, 'row': r, 'action': actual_action, 'planned_action': planned_action,
              'landing_failure': landing, 'immediate_home_entry': home_entry,
              'bank_death': bank, 'river_transition_failure': transition,
              'local_contacts': [], 'contact_revision': self.contact_revision+1}
        if bank:
            self.bank_unsafe = True
        if landing:
            lane = next((l for l in old.get('river', []) if int(num(l.get('row'))) == r), {})
            for z in lane.get('objects', []):
                n = int(num(z.get('nusiz'))) & 7
                shape = int(num(z.get('shape_ptr')))
                for off in COPIES[n]:
                    relative = (x-num(z.get('x'))-off+80) % 160-80
                    if -5 <= relative <= SIZES[n]+1 and z.get('kind') in ('log', 'turtle'):
                        key = (r, shape, n)
                        band = (max(-8, int(math.floor(relative))-3), min(SIZES[n]+8, int(math.ceil(relative))+3))
                        prior = self.contact_bans.setdefault(key, [])
                        prior.append(band)
                        self.contact_bans[key] = merge(prior)[-8:]
                        ev['local_contacts'].append({'shape': shape, 'nusiz': n, 'offset': relative, 'quarantined': band})
        self.contact_revision += 1
        self.diagnosis_evidence = ev
        self.contact_examples.append(ev)
        self.goal = None
        self.next_jev = t
        self.event('contact_contradiction', **ev)

    def apply_diagnosis(self, answers, o):
        meta = self.jev_meta or {}
        if meta.get('contact_revision') != self.contact_revision or self.last_diagnosis_revision == self.contact_revision:
            return
        a, b = answers.get('diagnosis'), answers.get('diagnosis_reverse')
        value = getattr(a, 'value', None)
        if not isinstance(value, str) or value != getattr(b, 'value', None):
            return
        pa, pb = getattr(a, 'probabilities', {}) or {}, getattr(b, 'probabilities', {}) or {}
        probability = min(num(pa.get(value)), num(pb.get(value)))
        ev = self.diagnosis_evidence or {}
        supported = {'contact': ev.get('landing_failure') and bool(ev.get('local_contacts')),
                     'entry': ev.get('immediate_home_entry'), 'bank': ev.get('bank_death')}
        if probability < .75 or not supported.get(value):
            return
        self.last_diagnosis_revision = self.contact_revision
        self.diagnosis = value
        if value == 'entry':
            # Empirical successful window is center-5 .. center+1. Prefer its
            # interior after an entry contradiction; do not poison carriers.
            self.home_window = (-4, 0)
        elif value == 'bank':
            self.bank_unsafe = True
        elif value == 'contact':
            self.policy['route'] = 'safest'
        self.event('jev_diagnosis_changes_control', tick=o.get('tick'), diagnosis=value,
                   probability=probability, home_window=self.home_window, bank_unsafe=self.bank_unsafe)

    def guard(self, o, action):
        """One-hop safety check, independent of the longer route score."""
        r = row(o)
        x = num(o.get('frog', {}).get('x'), 80)
        move = next((m for m in MOVES if m[0] == action), MOVES[3])
        nr, nx = r+move[1], x+move[2]
        if not -1 <= nr <= 11 or not 4 <= nx <= 148:
            return False
        if self.bank_unsafe and nr == 5 and action == 'NOOP':
            return False
        if nr in (-1, 5):
            return True
        if nr == 11:
            return any(not h.get('occupied') and h.get('available', True) and
                       num(h.get('x_center'))+self.home_window[0] <= nx <= num(h.get('x_center'))+self.home_window[1]
                       for h in o.get('homes', []))
        lanes, drift = self.geometry(o)
        table, homes = self.grid(o, lanes, drift, 1)
        if table[0].get(nr, [-1]*161)[int(round(nx))] < 0:
            return False
        if nr >= 6:
            endx = nx + drift.get(nr, 0)*self.step
            if not 4 <= endx <= 148 or table[1].get(nr, [-1]*161)[int(round(endx))] < 0:
                return False
        return True

    def geometry(self, o):
        lanes = {}
        drift = {}
        for l in o.get('road', []) + o.get('river', []):
            r = int(num(l.get('row')))
            objects = []
            speeds = []
            for z in l.get('objects', []):
                n = int(num(z.get('nusiz'))) & 7
                ident = str(z.get('id', (r, z.get('index'))))
                v = self.vel.get(ident, 0)
                # Same shape is insufficient: observed purple generic carriers
                # caused immediate deaths despite apparent geometric support.
                safe = not z.get('hazard') and z.get('kind') in ('log', 'turtle')
                shape = int(num(z.get('shape_ptr')))
                # Archive-wide invariant: stretched shapes A3/AA kill at local
                # frog-left offsets 21..26, unlike B8. Never call their heads
                # safe just because color decoding calls them turtles.
                width = 24 if n == 7 and shape in (163, 170) else SIZES[n]
                objects.append((num(z.get('x')), v, COPIES[n], width, safe, shape, n))
                speeds.append(v)
            lanes[r] = objects
            drift[r] = sum(speeds) / len(speeds) if speeds else 0
        return lanes, drift

    def grid(self, o, lanes, drift, depth):
        """Precompute conservative frog-left safe sets; arrays avoid slow inner geometry."""
        table = []
        homes = [h for h in o.get('homes', []) if not h.get('occupied') and h.get('available', True)
                 and h.get('crocodile') not in (True, 'present', 'dangerous')]
        # Each layer describes the entire next six-frame dwell, including display lag.
        for d in range(depth + 1):
            t = d * self.step
            rows = {}
            for r, objects in lanes.items():
                if r < 5:
                    safe = [20.0] * 161
                    for x, v, offsets, width, usable, shape, nusiz in objects:
                        for off in offsets:
                            a = x + off + v * (t + 1)
                            b = x + off + v * (t + self.step + 2)
                            lo, hi = min(a, b), max(a, b) + width
                            for wrap in (-320, -160, 0, 160, 320):
                                first = max(0, math.floor(lo + wrap - 8 - self.road_pad))
                                last = min(160, math.ceil(hi + wrap + self.road_pad))
                                for px in range(first, last + 1):
                                    safe[px] = -1.0
                    rows[r] = safe
                else:
                    # Support uses a six-pixel centre requirement, after union of
                    # adjacent sprite copies. Gaps between copies stay WATER.
                    ints = []
                    for x, v, offsets, width, usable, shape, nusiz in objects:
                        if not usable:
                            continue
                        for off in offsets:
                            a = x + off + v * (t + 1)
                            for wrap in (-320, -160, 0, 160, 320):
                                ints.append((a+wrap, a+width+wrap))
                    safe = [-1.0] * 161
                    for a, b in merge(ints):
                        # x is the left coordinate of an eight-pixel frog.
                        lo, hi = a + self.river_pad - 4, b - self.river_pad - 4
                        first, last = max(0, math.ceil(lo)), min(160, math.floor(hi))
                        for px in range(first, last + 1):
                            safe[px] = max(safe[px], min(px-lo, hi-px))
                    # A contact contradiction invalidates only a local band
                    # of a shape, not every platform and not an entire level.
                    for x, v, offsets, width, usable, shape, nusiz in objects:
                        bans = self.contact_bans.get((r, shape, nusiz), ())
                        if not bans:
                            continue
                        for off in offsets:
                            a = x + off + v * (t + 1)
                            for low, high in bans:
                                for wrap in (-320, -160, 0, 160, 320):
                                    first = max(0, math.floor(a+wrap+low))
                                    last = min(160, math.ceil(a+wrap+high))
                                    for px in range(first, last+1):
                                        safe[px] = -1.0
                    rows[r] = safe
            table.append(rows)
        return table, homes

    def search(self, o):
        started = time.perf_counter()
        r0 = row(o)
        x0 = num(o.get('frog', {}).get('x'), 80)
        if r0 < -1 or r0 > 10:
            return []
        lanes, drift = self.geometry(o)
        river = r0 >= 5
        depth = 22 if river else 15
        table, homes = self.grid(o, lanes, drift, depth)
        centers = [(str(h.get('slot')), num(h.get('x_center'), 80)) for h in homes]
        if not centers:
            return []
        endpoint = 11 if river else 5
        # (row, x, first action, sequence, bottleneck, cumulative cost)
        beam = [(r0, x0, None, (), 20.0, 0.0)]
        routes = {}
        best_partial = None
        partial_value = -1e9
        for d in range(depth):
            nxt = []
            for r, x, first, seq, margin, cost in beam:
                for act, dr, dx in MOVES:
                    nr, nx = r + dr, x + dx
                    if nr < -1 or nr > endpoint or (river and nr < 5):
                        continue
                    if not 4 <= nx <= 148:
                        continue
                    # Retain safe sidewalk/median maneuvering; don't descend to farm score.
                    if nr == 5 and self.bank_unsafe and act == 'NOOP':
                        continue
                    if nr in (-1, 5):
                        m = 20.0
                    elif nr == 11:
                        home = min(centers, key=lambda h: abs(nx-h[1]))
                        # Bay centers describe the opening, while frog x is its
                        # left edge; leave room for rightward carry before sampling.
                        m = min(nx - (home[1]+self.home_window[0]), home[1]+self.home_window[1] - nx)
                        if m < 0:
                            continue
                    else:
                        ix = int(round(nx))
                        m = table[d].get(nr, [-1.0]*161)[ix]
                        if m < 0:
                            continue
                        if nr >= 6:
                            # Endpoint support is checked again, including carrier drift.
                            endx = nx + drift.get(nr, 0) * self.step
                            if not 4 <= endx <= 148:
                                continue
                            j = int(round(endx))
                            endm = table[min(d+1, depth)].get(nr, [-1.0]*161)[j]
                            if endm < 0:
                                continue
                            m = min(m, endm)
                    nm = min(margin, m)
                    nc = cost + 1 + (0.12 if act in ('LEFT', 'RIGHT') else 0) + (0.8 if act == 'DOWN' else 0)
                    path = seq + (act,)
                    firstact = first or act
                    if nr == endpoint:
                        if river:
                            label = 'HOME_' + home[0]
                            goalx = home[1]
                        else:
                            label = 'MEDIAN_' + ('LEFT' if nx < 56 else 'RIGHT' if nx > 104 else 'CENTER')
                            goalx = nx
                        # Prefer completed routes over merely high rows. Margin is a tie-break.
                        val = 200 - nc*2 + min(nm, 6)*0.6
                        if self.policy['route'] == 'safest':
                            val += min(nm, 6)*2
                        if label == self.goal:
                            val += 4
                        prev = routes.get(label)
                        if prev is None or val > prev['value']:
                            routes[label] = {'label': label, 'action': firstact, 'path': path,
                                             'value': val, 'margin': round(nm, 2), 'x': round(goalx, 1),
                                             'complete': True, 'steps': len(path)}
                        continue
                    endx = nx + (drift.get(nr, 0)*self.step if nr >= 6 else 0)
                    home_dist = min(abs(endx-c) for _, c in centers)
                    value = nr*20 - nc*0.5 - home_dist*(0.17 if nr>=8 else 0.015) + min(nm, 5)*0.2
                    # Partial routes favor actual progress, not endless lateral motion.
                    if nr > r0 and value > partial_value:
                        partial_value = value
                        best_partial = {'label': 'PROGRESS', 'action': firstact, 'path': path,
                                        'value': value-400, 'margin': round(nm, 2), 'x': round(endx, 1),
                                        'complete': False, 'steps': len(path)}
                    nxt.append((value, (nr, endx, firstact, path, nm, nc)))
            if not nxt:
                break
            nxt.sort(key=lambda z: z[0], reverse=True)
            used = set()
            beam = []
            for _, state in nxt:
                key = (state[0], round(state[1]/2))
                if key not in used:
                    used.add(key)
                    beam.append(state)
                if len(beam) >= 48:
                    break
            # Once a complete path exists, search a few extra layers for wider alternatives.
            if routes and d > min(z['steps'] for z in routes.values()) + 3:
                break
        result = sorted(routes.values(), key=lambda z: z['value'], reverse=True)
        if not result and best_partial:
            result = [best_partial]
        if not result:
            # A feasible single move is still useful as a survival fallback.
            choices = []
            for act, dr, dx in MOVES:
                nr, nx = r0+dr, x0+dx
                if nr < -1 or nr > 10 or not 4 <= nx <= 148:
                    continue
                if nr == 5 and self.bank_unsafe and act == 'NOOP':
                    continue
                m = 20 if nr in (-1, 5) else table[0].get(nr, [-1]*161)[int(round(nx))]
                if m >= 0:
                    choices.append({'label': 'SURVIVE', 'action': act, 'path': (act,),
                                    'value': nr*10 + min(m, 4) - (0 if act=='NOOP' else 1),
                                    'margin': round(m, 2), 'x': nx, 'complete': False, 'steps': 1})
            result = sorted(choices, key=lambda z:z['value'], reverse=True)
        elapsed = (time.perf_counter()-started)*1000
        self.search_ms.append(elapsed)
        self.search_ms = self.search_ms[-100:]
        return result

    def request_jev(self, o, routes):
        ctx = self.ctx
        if self.jev_future is not None or not routes or getattr(ctx, 'system1', True) is None:
            return
        if ctx.budget.remaining().get('system1_calls', 0) < 2:
            return
        complete = [r for r in routes if r['complete']][:5]
        if not complete:
            return
        t = int(num(o.get('tick')))
        shortest = min(r['steps'] for r in complete)
        widest = max(r['margin'] for r in complete)
        facts = []
        for r in complete:
            facts.append({'goal': r['label'], 'route_verified_in_current_forecast': True,
                          'travel': 'shortest' if r['steps']==shortest else 'detour',
                          'support': 'widest_available' if r['margin']==widest else 'narrower',
                          'clearance': 'comfortable' if r['margin']>=2 else 'tight',
                          'is_current_goal': r['label']==self.goal})
        state = {'game': 'Frogger; reach different open homes, preserve lives.',
                 'phase': segment(o), 'policy': dict(self.policy),
                 'candidates': facts,
                 'timer': 'urgent' if num(o.get('timer_s'), 30)<8 else 'normal',
                 'forecast_warning': 'Sprite-copy gaps are water. Code rechecks every hop; this is a durable goal, not an immediate joystick action.',
                 'recent_prediction_failure': bool(self.failures),
                 'contact_diagnosis': self.diagnosis_evidence,
                 'calibration_rule': 'Local contact bands are quarantined after a contradicted landing. Home-entry deaths are not evidence that a carrier is water. Bank deaths revoke the safe-pause assumption. Shapes A3/AA have a hazardous right end despite turtle labels.',
                 'home_hazard': 'Crocodile unknown, never asserted absent.'}
        options = {r['label']: {'what': 'Choose this complete forecast route goal.', 'facts': f}
                   for r, f in zip(complete, facts)}
        options['UNKNOWN'] = 'None of these route goals is adequately supported.'
        instruction = ('Select a route GOAL from candidates, not an immediate direction. Prefer comfortable support and a shortest route. '
                       'When policy.route is safest prefer widest_available; retain a current goal unless another is clearly safer. '
                       'Urgent timer favors shortest. All candidates avoid known hazards. UNKNOWN if unsupported; no arithmetic.')
        questions = {'goal': Choice(instruction, options),
                     'reverse': Choice(instruction, dict(reversed(list(options.items())))),
                     'review': Noul('Do the supplied facts explicitly report a recent prediction failure that merits reviewing the geometry policy?')}
        diagnoses = {
            'contact': 'A recent failure was a landing or lateral hop on a carrier at a previously uncertified shape-local offset. Keep that local contact band quarantined.',
            'entry': 'A recent failure was an immediate UP from the top river row into a home; carrier geometry is not established as the cause. Prefer the interior of the successful home-entry window.',
            'bank': 'A recent death occurred on the median. Stop treating it as an unlimited safe staging area; traverse it without lateral dwelling.',
            'unknown': 'No recent, well-localized failure establishes one of these explanations.'}
        diagnostic_instruction = ('Using contact_diagnosis, classify the most recent observed failure, not a hypothetical hazard. '
                                  'Use its explicit landing_failure, immediate_home_entry, and bank_death facts. '
                                  'Do not infer a collision from a color name alone. Choose unknown without evidence.')
        questions['diagnosis'] = Choice(diagnostic_instruction, diagnoses)
        questions['diagnosis_reverse'] = Choice(diagnostic_instruction, dict(reversed(list(diagnoses.items()))))
        self.jev_meta = {'tick': t, 'epoch': epoch(o), 'segment': segment(o), 'version': self.version,
                         'contact_revision': self.contact_revision}
        def request():
            try:
                before = ctx.clock.now()
                answer = ctx.decide(state, questions, timeout_s=3.0)
                return answer, ctx.clock.now()-before
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None, 0
        self.jev_future = ctx.spawn(request)
        self.next_jev = t + 90

    def request_astra(self, o):
        ctx = self.ctx
        if self.astra_future is not None or getattr(ctx, 'system2', True) is None:
            return
        if ctx.budget.remaining().get('system2_calls', 0) < 2:
            return
        prompt = ('You are a slow Frogger calibration/strategy adviser, never a frame controller. The executor uses NUSIZ separated sprite '
                  'copies and an 8px frog, searches time-dependent paths to the median or an OPEN home, and releases every hop with NOOP. '
                  'Jev selects a DURABLE route goal from code-verified complete routes, via two reversed Choices (both probability >=.65) '
                  'and a Noul flag for prediction failures. Its state includes semantic shortest/detour, comfortable/tight, widest-available, '
                  'current-goal flags, timer urgency, this policy, and recent failure. Code always revalidates latest geometry. '
                  'Choose policy, not a route from a stale snapshot. Return ONLY JSON with route=balanced|safest, '
                  'road_clearance=normal|extra, river_support=center|inset. Extra expands obstacle exclusion. Inset tightens carrier support '
                  'but may reduce reachability. Prefer the smallest change supported by actual failed predictions. '
                  'Additional field focus=contact|entry|bank is optional. The new executor also learns local contact bands: '
                  'archived shape 163/170 stretched carriers have lethal right ends; immediate home-entry deaths must '
                  'not be blamed on river support, and median deaths revoke the safe-pause assumption. '
                  'Jev has two reversed diagnosis Choices over contact/entry/bank/unknown; matching probabilities >=.75 '
                  'allow only the corresponding evidence-supported conservative update. Choose a focus for that '
                  'diagnosis interface after repeated prediction contradictions, never invent an unobserved mechanism. '
                  'No prose or invented physics. Current summary: ' + json.dumps(
                      {'policy': self.policy, 'score': o.get('score'), 'level': o.get('level'),
                       'failures': list(self.failures)[-2:], 'contact_examples': list(self.contact_examples), 'routes': self.routes[:3]}, separators=(',', ':')))
        session = self.session
        self.astra_meta = {'level': o.get('level')}
        def request():
            try:
                return ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None
        self.astra_future = ctx.spawn(request)
        self.next_astra = int(num(o.get('tick'))) + 3600

    def poll(self, o):
        t = int(num(o.get('tick')))
        if self.jev_future is not None and self.jev_future.done():
            f, self.jev_future = self.jev_future, None
            ans, latency = f.result()
            if isinstance(ans, dict):
                self.s1_latency.append(latency)
                self.apply_diagnosis(ans, o)
                a, b = ans.get('goal'), ans.get('reverse')
                chosen = getattr(a, 'value', None)
                probs_a = getattr(a, 'probabilities', None) or {}
                probs_b = getattr(b, 'probabilities', None) or {}
                p = min(num(probs_a.get(chosen)), num(probs_b.get(chosen))) if isinstance(chosen, str) else 0
                meta = self.jev_meta
                fresh = (epoch(o)==meta['epoch'] and segment(o)==meta['segment'] and self.version==meta['version'] and t-meta['tick']<=90)
                if fresh and isinstance(chosen, str) and chosen.startswith(('HOME_', 'MEDIAN_')) and chosen==getattr(b, 'value', None) and p>=.65:
                    self.goal, self.goal_until = chosen, t+180
                    self.event('jev_goal_adopted', tick=t, goal=chosen, age=t-meta['tick'], probability=p, latency_s=latency)
                else:
                    self.event('jev_goal_rejected', tick=t, fresh=fresh, probability=p, age=t-meta['tick'], latency_s=latency)
                review = num(getattr(ans.get('review'), 'value', 0))
                if review>=.8 and self.failures and t>=self.next_astra:
                    self.request_astra(o)
        if self.astra_future is not None and self.astra_future.done():
            f, self.astra_future = self.astra_future, None
            ans = f.result()
            if ans is not None:
                self.session = getattr(ans, 'session_id', None)
                p = small_json(getattr(ans, 'text', ''))
                allowed = {'route': ('balanced', 'safest'), 'road_clearance': ('normal', 'extra'), 'river_support': ('center', 'inset'), 'focus': ('contact', 'entry', 'bank')}
                changes = {k:v for k,v in p.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if changes and self.astra_meta['level']==o.get('level'):
                    self.policy.update(changes)
                    self.road_pad = 2.5 if self.policy['road_clearance']=='extra' else 1.5
                    self.river_pad = 3 if self.policy['river_support']=='inset' else 2
                    self.version += 1
                    self.goal = None
                    self.event('astra_policy_applied', tick=t, policy=self.policy)
                else:
                    self.event('astra_output_rejected', tick=t)

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            o = unpack(ctx.observe())
            if not o:
                ctx.act('NOOP')
                ctx.wait_for_tick()
                continue
            t = int(num(o.get('tick')))
            self.update(o)
            self.poll(o)
            action = 'NOOP'
            active = int(num(o.get('frog', {}).get('state'))) == 128
            if self.goal and t > self.goal_until:
                self.goal = None
            # The cartridge samples joystick input on alternating frames. A single
            # consumed display frame is not always a sampled edge. Hold only until
            # movement is observed (at most three frames), then explicitly release.
            motor_busy = self.pressed is not None
            if motor_busy:
                if not active or self.last_origin is None or t-self.last_press >= 4:
                    self.pressed = None
                    self.ready = max(self.ready, t+2)
                else:
                    action = self.pressed
            if not motor_busy and active and t >= self.ready and -1 <= row(o) <= 10:
                self.routes = self.search(o)
                if self.routes:
                    chosen = self.routes[0]
                    if self.goal:
                        match = next((r for r in self.routes if r['label']==self.goal and r['value']>=chosen['value']-8), None)
                        if match:
                            if match['action'] != chosen['action']:
                                self.event('jev_changes_route_action', tick=t, before=chosen['action'], after=match['action'], goal=self.goal)
                            chosen = match
                    action = chosen['action']
                    # Planning may cross a wall-clock tick. Check the actual
                    # public state again before exposing a directional press.
                    latest = unpack(ctx.observe())
                    if latest and epoch(latest) == epoch(o) and row(latest) == row(o):
                        if not self.guard(latest, action):
                            alternatives = [a for a, _, _ in MOVES if self.guard(latest, a)]
                            action = alternatives[0] if alternatives else 'NOOP'
                            self.guard_vetoes += 1
                            self.event('contact_guard_veto', tick=latest.get('tick'), before=chosen['action'], after=action)
                    else:
                        action = 'NOOP'
                    chosen = dict(chosen, action=action)
                    self.last_plan = {'tick': t, 'origin': (row(o), o.get('frog', {}).get('x')), **chosen}
                    if action != 'NOOP' or t-self.last_log>=60:
                        self.event('route_action', tick=t, frog=o.get('frog'), selected=chosen,
                                   alternatives=self.routes[:5], search_ms=round(self.search_ms[-1], 2))
                        self.last_log = t
                    if t>=self.next_jev:
                        self.request_jev(o, self.routes)
                elif t-self.last_log>=60:
                    self.event('no_route', snapshot=self.compact(o), search_ms=round(self.search_ms[-1], 2))
                    self.last_log = t
                self.ready = t + (self.step if action!='NOOP' else 3)
                if action!='NOOP':
                    self.pressed = action
                    self.last_press = t
                    self.last_origin = (row(o), num(o.get('frog', {}).get('x')))
            if len(self.failures)>=2 and t>=self.next_astra:
                self.request_astra(o)
            self.last_action = action
            ctx.act(action)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
