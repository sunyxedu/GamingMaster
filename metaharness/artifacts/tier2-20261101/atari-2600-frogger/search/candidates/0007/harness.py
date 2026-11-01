"""Replica-aware Frogger crossover: a clocked motor and semantic route adviser.

Only public observations are used. In particular, NUSIZ is not a solid bounding
box: repeated sprites leave gaps, while stretched sprites have different widths.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

ACTIONS = ('UP', 'LEFT', 'RIGHT', 'NOOP', 'DOWN')
# TIA sprite copies (offset, width), in world pixels. Shape holes are handled
# conservatively by an interior/overlap margin, not by assuming a filled hull.
COPIES = {
    0: ((0, 8),), 1: ((0, 8), (16, 8)),
    2: ((0, 8), (32, 8)), 3: ((0, 8), (16, 8), (32, 8)),
    4: ((0, 8), (64, 8)), 5: ((0, 16),),
    6: ((0, 8), (32, 8), (64, 8)), 7: ((0, 32),),
}


def num(value, default=0.0):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def decode(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def row(o):
    r = int(num(o.get('frog', {}).get('row'), 255))
    return -1 if r == 255 else r


def identity(o):
    return (o.get('deaths', 0), o.get('frogs_home', 0), o.get('level', 1))


def phase(o):
    r = row(o)
    return 'road_crossing' if r < 5 else ('median' if r == 5 else 'river_crossing')


def small_json(text):
    if not isinstance(text, str):
        return {}
    try:
        v = json.loads(text[text.find('{'):text.rfind('}') + 1])
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 0.01:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


class Geometry:
    """A short-horizon approximate simulator, freshly rebuilt at every decision."""
    def __init__(self, o, speeds, hop=8.0, cautious=False):
        self.o = o
        self.lanes = {int(l['row']): l for l in o.get('road', []) + o.get('river', [])}
        self.speeds = speeds
        self.hop = hop
        self.duration = 10
        self.cautious = cautious
        self.cache = {}
        self.homes = [num(h.get('x_center'), 80) for h in o.get('homes', [])
                      if not h.get('occupied') and h.get('available', True)
                      and h.get('crocodile') not in (True, 'present', 'dangerous')]
        self.v = {}
        for r, lane in self.lanes.items():
            vals = [self.speed(lane, b) for b in lane.get('objects', [])]
            self.v[r] = sum(vals) / len(vals) if vals else 0.0

    def speed(self, lane, obj):
        fallback = (-1 if lane.get('direction') == 'left' else 1) / max(1, int(num(lane.get('activity_mask'), 3)) + 1)
        return self.speeds.get(obj.get('id'), fallback)

    def intervals(self, r, t):
        key = (r, t)
        if key in self.cache:
            return self.cache[key]
        lane = self.lanes.get(r, {})
        safe, danger = [], []
        for b in lane.get('objects', []):
            x = num(b.get('x')) + self.speed(lane, b) * t
            kind = b.get('kind')
            usable = not b.get('hazard') and kind in ('log', 'turtle', 'river_carrier')
            dest = safe if usable and r >= 6 else danger
            for offset, width in COPIES.get(int(num(b.get('nusiz'))), ((0, 8),)):
                a = (x + offset) % 160
                for shift in (-160, 0, 160):
                    dest.append((a + shift, a + width + shift))
        value = (merge(safe), merge(danger))
        self.cache[key] = value
        return value

    def safety(self, r, x, t):
        # Frog x is its sprite anchor, not the centre of an eight-pixel box.
        if x < 3 or x > 149:
            return -100.0
        if r in (-1, 5):
            return 20.0
        if r == 11:
            return max((4.0 - abs(x + 4 - h) for h in self.homes), default=-100.0)
        if r not in self.lanes:
            return -100.0
        safe, danger = self.intervals(r, t)
        if r < 5:
            margin = 100.0
            padding = 2.0 + (1.0 if self.cautious else 0.0)
            for a, b in danger:
                margin = min(margin, max(a - (x + 8), x - b) - padding)
            return margin
        # Require the frog's central core to be on actual sprite support.
        # Union adjacent objects first; never bridge a true NUSIZ water gap.
        centre = x + 4
        margin = max((min(centre - a, b - centre) - 1.5 for a, b in safe), default=-100.0)
        for a, b in danger:
            if a - 2 < centre < b + 2:
                return -100.0
        return margin

    def transition(self, r, x, t, action):
        nr = r + (1 if action == 'UP' else -1 if action == 'DOWN' else 0)
        nx = x + (self.hop if action == 'RIGHT' else -self.hop if action == 'LEFT' else 0)
        if nr < -1 or nr > 11 or (r == -1 and action == 'DOWN'):
            return None
        source_v = self.v.get(r, 0.0) if 6 <= r <= 10 else 0.0
        dest_v = self.v.get(nr, 0.0) if 6 <= nr <= 10 else 0.0
        dur = self.duration
        values = []
        if action == 'NOOP':
            values = [self.safety(r, x + source_v * s, t + s) for s in (1, 5, dur + 2)]
            end_x = x + source_v * dur
        else:
            # Observations show joystick sampling takes 1--4 frames. Validate
            # both early and late landing, and the entire neutral recovery.
            for delay in (1, 3):
                for s in (delay, 6, dur + 2):
                    px = nx + source_v * delay + dest_v * max(0, s - delay)
                    values.append(self.safety(nr, px, t + s))
            end_x = nx + source_v * 2 + dest_v * (dur - 2)
            # Sideways river hops must stay within one connected support band;
            # endpoint-only checks used to jump straight into replica gaps.
            if action in ('LEFT', 'RIGHT') and r >= 6:
                for frac in (0.25, 0.5, 0.75):
                    values.append(self.safety(r, x + (nx - x) * frac + source_v * 2, t + 2))
        margin = min(values)
        if margin < 0:
            return None
        return (nr, end_x, t + dur, margin)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.history = {}
        self.speeds = {}
        self.hop = 8.0
        self.pressed = None
        self.press_tick = -100
        self.press_origin = None
        self.release_until = 3
        self.next_hop = 3
        self.last_move = None
        self.key = None
        self.policy = {'route': 'nearest_open', 'safety': 'balanced', 'focus': 'progress'}
        self.version = 0
        self.target = None
        self.jev_future = None
        self.astra_future = None
        self.jev_meta = None
        self.astra_meta = None
        self.advice = None
        self.next_jev = 0
        self.next_astra = 0
        self.session = None
        self.events = deque(maxlen=10)
        self.last_progress = 0
        self.last_candidates = []
        self.last_action = 'NOOP'
        self.last_snapshot = -100

    def log(self, name, **data):
        self.ctx.log(name, **data)

    def update(self, o):
        tick = int(num(o.get('tick')))
        k = identity(o)
        if self.key != k:
            if self.key is not None and self.key[2] != k[2]:
                # A level can replace speed tables; do not carry its predecessor's
                # fitted velocities into the first new crossing.
                self.history.clear()
                self.speeds.clear()
            if self.key is not None:
                event = {'tick': tick, 'before': self.key, 'after': k,
                         'last_move': self.last_move, 'frog': o.get('frog'),
                         'previous_frog': self.previous.get('frog') if self.previous else None}
                self.events.append(event)
                self.log('crossing_outcome', **event)
            self.key = k
            self.target = None
            self.pressed = None
            self.advice = None
            self.next_hop = tick + 4
            self.release_until = tick + 4
            self.last_progress = tick
        if self.previous and (row(o) != row(self.previous) or o.get('score') != self.previous.get('score')):
            self.last_progress = tick
        for lane in o.get('road', []) + o.get('river', []):
            for b in lane.get('objects', []):
                ident = b.get('id')
                h = self.history.setdefault(ident, deque(maxlen=40))
                x = num(b.get('x'))
                if not h or h[-1][0] != tick:
                    h.append((tick, x))
                old = next(((t, z) for t, z in h if 16 <= tick - t <= 40), None)
                if old:
                    t, z = old
                    velocity = ((x - z + 80) % 160 - 80) / (tick - t)
                    if abs(velocity) <= 2:
                        self.speeds[ident] = velocity
        self.previous = o

    def goal(self, o, geometry):
        if not geometry.homes:
            return 80.0
        if self.target not in geometry.homes:
            x = num(o.get('frog', {}).get('x'), 80) + 4
            route = self.policy['route']
            self.target = (min(geometry.homes) if route == 'left_open' else max(geometry.homes)
                           if route == 'right_open' else min(geometry.homes, key=lambda h: abs(h - x)))
        return self.target

    def rank(self, o):
        g = Geometry(o, self.speeds, self.hop, self.policy['safety'] == 'cautious')
        target = self.goal(o, g)
        r, x = row(o), num(o.get('frog', {}).get('x'), 80)
        out = []
        for first in ACTIONS:
            root = g.transition(r, x, 0, first)
            if root is None:
                continue
            rr, xx, tt, mm = root
            states = [(rr, xx, tt, mm, 0.0)]
            best, reach, home = -1e9, rr, False
            # Six-hop horizon; score complete home trajectories much more than
            # merely staying high on the board. Only the first hop is executed.
            for depth in range(6):
                children = []
                for cr, cx, ct, cm, cost in states:
                    dist = abs(cx + 4 - target)
                    value = cr * 35 + min(cm, 10) * 0.4 - cost
                    if cr >= 5:
                        value -= dist * (0.22 if cr < 9 else 0.6)
                    if cr == 11:
                        value += 1000
                        home = True
                    if value > best:
                        best, reach = value, cr
                    if depth == 5 or cr == 11:
                        continue
                    for a in ACTIONS:
                        nxt = g.transition(cr, cx, ct, a)
                        if nxt is not None:
                            nr, nx, nt, nm = nxt
                            penalty = 1.0 if a == 'NOOP' else 3.0 if a == 'DOWN' else 0.6
                            if a in ('LEFT', 'RIGHT') and cr >= 6:
                                penalty += 1.8
                            children.append((nr, nx, nt, min(cm, nm), cost + penalty))
                children.sort(key=lambda s: s[0] * 35 - abs(s[1] + 4 - target) * 0.25 - s[4], reverse=True)
                seen, states = set(), []
                for s in children:
                    key = (s[0], round(s[1] / 2))
                    if key not in seen:
                        seen.add(key)
                        states.append(s)
                        if len(states) >= 10:
                            break
            value = best + rr * 0.4 + min(mm, 8) * 0.2
            if first == 'NOOP':
                value -= 0.3
            category = ('ADVANCE' if first == 'UP' else 'RETREAT' if first == 'DOWN'
                        else 'RIDE' if first == 'NOOP' else 'REALIGN')
            out.append({'action': first, 'category': category, 'value': round(value, 3),
                        'margin': round(mm, 2), 'reach': reach,
                        'home_path': home, 'toward_home': abs(xx + 4 - target) < abs(x + 4 - target),
                        'support': 'wide' if mm >= 4 else 'interior',
                        'landing_row': rr, 'landing_x': round(xx, 2)})
        out.sort(key=lambda c: c['value'], reverse=True)
        self.last_candidates = out
        return out, g

    def start_jev(self, o, candidates, geometry):
        ctx = self.ctx
        if self.jev_future is not None or not candidates or getattr(ctx, 'system1', True) is None:
            return
        if ctx.budget.remaining().get('system1_calls', 0) < 2:
            return
        tick = int(num(o.get('tick')))
        r, x = row(o), num(o.get('frog', {}).get('x'), 80)
        wait_safe = geometry.transition(r, x, 0, 'NOOP') is not None
        categories = []
        for c in candidates[:3]:
            if c['category'] not in categories:
                categories.append(c['category'])
        descriptions = {
            'ADVANCE': 'Take the next geometry-validated forward opening toward an open home.',
            'REALIGN': 'Use a safe lateral move to improve the future route or home alignment; never cross a water gap.',
            'RIDE': 'Wait on safe support for a better opening; the code continues checking drift and hazards.',
            'RETREAT': 'Take a validated backward escape when remaining here is unsafe.',
            'UNKNOWN': 'The supplied facts do not establish a useful maneuver preference.',
        }
        options = {k: descriptions[k] for k in categories}
        options['UNKNOWN'] = descriptions['UNKNOWN']
        state = {
            'rules': 'Frogger: reach unoccupied homes alive. Code checks actual sprite copies, car clearance, water gaps, drift, and joystick release. These are maneuver templates, not commands to execute on an old snapshot.',
            'policy': dict(self.policy), 'phase': phase(o),
            'facts': {'wait_safe': wait_safe, 'escape_needed': not wait_safe,
                      'timer_urgent': num(o.get('timer_s'), 30) <= 7,
                      'near_homes': r >= 9, 'home_aligned': abs(x + 4 - (self.target or 80)) <= 4},
            'proposals_in_progress_order': [{k: v for k, v in c.items() if k not in ('value', 'margin', 'reach', 'landing_x', 'landing_row')} for c in candidates[:3]],
            'uncertainty': 'An unknown home crocodile is not evidence of absence. No optional bonus detours.',
        }
        instructions = ('Choose a maneuver template using facts, policy and proposals_in_progress_order. Prefer the leading progress proposal unless another explicitly improves safety or home alignment. Do not RIDE when escape_needed. Do not perform coordinate arithmetic. UNKNOWN if unsupported.')
        questions = {'maneuver': Choice(instructions, options),
                     'order_check': Choice(instructions, dict(reversed(list(options.items())))),
                     'escape': Noul('Do facts explicitly say escape_needed is true? True means safe waiting is not established; false means waiting is supported.')}
        self.jev_meta = {'tick': tick, 'identity': identity(o), 'phase': phase(o), 'version': self.version}
        self.next_jev = tick + 30
        def request():
            try:
                return ctx.decide(state, questions, timeout_s=3.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None
        self.jev_future = ctx.spawn(request)

    def start_astra(self, o):
        ctx = self.ctx
        if self.astra_future is not None or getattr(ctx, 'system2', True) is None:
            return
        if ctx.budget.remaining().get('system2_calls', 0) < 2:
            return
        snapshot = {'policy': self.policy, 'frog': o.get('frog'), 'homes': o.get('homes'),
                    'score': o.get('score'), 'level': o.get('level'), 'timer_s': o.get('timer_s'),
                    'outcomes': list(self.events), 'candidates': self.last_candidates[:3]}
        prompt = (
            'You advise a concurrent Atari 2600 Frogger controller. Return ONLY tiny JSON with '
            'route: nearest_open|left_open|right_open, safety: balanced|cautious, focus: progress|wide_support|home_alignment. '
            'Your policy lasts across many frames, not one joystick action. The code handles exact NUSIZ sprite replicas '
            'rather than solid estimated bounding boxes, eight-pixel frog extents, safe landing/neutral recovery, and '
            'replanning. No optional bonus detours. Five distinct homes and life preservation are the objective. '
            'Jev receives phase, your policy, semantic candidate facts, safe-wait/escape/home-alignment booleans. '
            'It returns two reordered Choices over ADVANCE/REALIGN/RIDE/RETREAT/UNKNOWN and an escape Noul. '
            'A preference is accepted only on agreement at probability >=.65, unchanged crossing/phase/plan, '
            'at most 90 frames old, and freshly safe geometry. Preferences only select among near-best current '
            'routes. wide_support emphasizes carrier interior, home_alignment emphasizes alignment at the top. '
            'After repeated geometry misses choose cautious; after safe stalls prefer progress or a different open-home '
            'ordering. Do not invent a numeric mapping, code, or immediate action. Snapshot: ' + json.dumps(snapshot, separators=(',', ':')))
        session = self.session
        self.astra_meta = {'level': o.get('level', 1)}
        self.next_astra = int(num(o.get('tick'))) + 3600
        def request():
            try:
                return ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None
        self.astra_future = ctx.spawn(request)

    def poll(self, o):
        tick = int(num(o.get('tick')))
        if self.astra_future is not None and self.astra_future.done():
            f, self.astra_future = self.astra_future, None
            result = f.result()
            if result is not None:
                self.session = getattr(result, 'session_id', None)
                value = small_json(getattr(result, 'text', ''))
                allowed = {'route': ('nearest_open', 'left_open', 'right_open'),
                           'safety': ('balanced', 'cautious'),
                           'focus': ('progress', 'wide_support', 'home_alignment')}
                adopted = {k: v for k, v in value.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if adopted and o.get('level', 1) == self.astra_meta['level']:
                    self.policy.update(adopted)
                    self.version += 1
                    self.target = None
                    self.advice = None
                    self.log('astra_policy_adopted', tick=tick, policy=dict(self.policy))
                else:
                    self.log('astra_output_rejected', tick=tick)
        if self.jev_future is not None and self.jev_future.done():
            f, self.jev_future = self.jev_future, None
            answer = f.result()
            if not isinstance(answer, dict):
                return
            a, b = answer.get('maneuver'), answer.get('order_check')
            choice = getattr(a, 'value', None)
            probs_a, probs_b = getattr(a, 'probabilities', {}), getattr(b, 'probabilities', {})
            pa = num(probs_a.get(choice, 0)) if isinstance(probs_a, dict) and isinstance(choice, str) else 0
            pb = num(probs_b.get(choice, 0)) if isinstance(probs_b, dict) and isinstance(choice, str) else 0
            m = self.jev_meta
            fresh = (identity(o) == m['identity'] and phase(o) == m['phase'] and self.version == m['version'] and tick - m['tick'] <= 90)
            accepted = fresh and choice in ('ADVANCE', 'REALIGN', 'RIDE', 'RETREAT') and choice == getattr(b, 'value', None) and min(pa, pb) >= 0.65
            if accepted:
                self.advice = dict(m, choice=choice, until=m['tick'] + 90)
            escape_probability = num(getattr(answer.get('escape'), 'value', None), -1.0)
            self.log('jev_preference', tick=tick, choice=choice if isinstance(choice, str) else None,
                     accepted=bool(accepted), age=tick - m['tick'], probability=min(pa, pb),
                     escape_probability=escape_probability if 0 <= escape_probability <= 1 else None)

    def motor(self, o, candidates):
        tick = int(num(o.get('tick')))
        r, x = row(o), num(o.get('frog', {}).get('x'), 80)
        # state=8 is observed during death/home animation, not a playable frog.
        if int(num(o.get('frog', {}).get('state'), 128)) != 128 or r not in range(-1, 11):
            self.pressed = None
            self.release_until = tick + 4
            return 'NOOP'
        if self.pressed is not None:
            origin_r, origin_x = self.press_origin
            moved = r != origin_r or abs(x - origin_x) >= 5
            if moved or tick - self.press_tick >= 4:
                self.log('pulse_release', tick=tick, action=self.pressed, moved=moved,
                         origin=self.press_origin, frog=o.get('frog'))
                self.pressed = None
                self.release_until = tick + 4
                return 'NOOP'
            return self.pressed
        if tick < self.release_until or tick < self.next_hop or not candidates:
            return 'NOOP'
        chosen = candidates[0]
        close = [c for c in candidates if c['value'] >= chosen['value'] - 9]
        if self.policy['focus'] == 'wide_support' and r >= 6:
            chosen = max(close, key=lambda c: (c['support'] == 'wide', c['value']))
        elif self.policy['focus'] == 'home_alignment' and r >= 9:
            chosen = max(close, key=lambda c: (c['toward_home'], c['value']))
        advice = self.advice
        if advice and tick <= advice['until'] and identity(o) == advice['identity'] and phase(o) == advice['phase'] and self.version == advice['version']:
            match = next((c for c in close if c['category'] == advice['choice']), None)
            if match and match['action'] != chosen['action']:
                self.log('jev_changes_action', tick=tick, before=chosen['action'], after=match['action'])
                chosen = match
        action = chosen['action']
        if action != 'NOOP':
            self.pressed = action
            self.press_tick = tick
            self.press_origin = (r, x)
            self.next_hop = tick + 10
            self.last_move = {'tick': tick, 'action': action, 'row': r, 'x': x, 'proposal': chosen}
            self.log('hop_issued', **self.last_move, candidates=candidates[:3])
        else:
            self.next_hop = tick + 3
        return action

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            o = decode(ctx.observe())
            if not o:
                ctx.act('NOOP')
                ctx.wait_for_tick()
                continue
            self.update(o)
            self.poll(o)
            tick = int(num(o.get('tick')))
            playable = int(num(o.get('frog', {}).get('state'), 128)) == 128 and row(o) in range(-1, 11)
            ready = playable and self.pressed is None and tick >= self.release_until and tick >= self.next_hop
            candidates = []
            if ready:
                candidates, geometry = self.rank(o)
                if tick >= self.next_jev:
                    self.start_jev(o, candidates, geometry)
                if tick - self.last_snapshot >= 120:
                    self.last_snapshot = tick
                    self.log('geometry_snapshot', observation=o, candidates=candidates[:3], speeds=dict(self.speeds))
            if tick >= self.next_astra and (tick < 4 or self.events or tick - self.last_progress > 300):
                self.start_astra(o)
            action = self.motor(o, candidates)
            self.last_action = action
            ctx.act(action)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
