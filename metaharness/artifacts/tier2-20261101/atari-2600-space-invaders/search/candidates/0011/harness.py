"""0011: preserve the escape heading after an interior stalled traverse.

Only public observations and the HarnessContext interface are used. Models
select among executable spatial programs; code owns their lifetime and fire.
"""
from __future__ import annotations

import json
import math
import statistics
import threading
from collections import deque
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def num(value, default=0.0):
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sign(x):
    return (x > 0) - (x < 0)


def obj(x):
    return x if isinstance(x, dict) else {}


def items(x):
    return [a for a in x if isinstance(a, dict)] if isinstance(x, list) else []


def questions():
    choices = {
        'broad': 'Traverse the living formation, firing along the way. Prefer with many columns, blind threats, or a previous short reversing loop. Uncertain visibility does NOT mean reverse now.',
        'front': 'Traverse the band of lowest surviving invaders. Prefer only when front_distinct is true: an urgent low band differs from the rest. This is not a stationary target lock.',
        'edge': 'Work across the exposed outer flank. Prefer only when edge_viable is true and plan favors finishing an outer column. Never shrink the route into a small left-right oscillation.',
        'unknown': 'The supplied evidence does not support a program; retain the checked fallback.',
    }
    instruction = (
        'Select the next spatial program using evidence, programs, and plan. '
        'Geometry and permissions are computed already. This answer does not '
        'reverse the joystick immediately: code finishes the committed traverse. '
        'Choose broad when the special front/edge preconditions are not met. '
        'An empty bomb list does not prove safety and does not by itself justify '
        'a change. Do not calculate coordinates, count, or invent unseen bombs.'
    )
    return {
        'program': Choice(instruction, choices),
        'crosscheck': Choice(instruction, dict(reversed(list(choices.items())))),
        'fire_mode': Choice(
            'Choose the firing policy from evidence and plan. STREAM means keep '
            'FIRE held while strafing; the ROM still permits only one laser. '
            'SELECTIVE means code opens a predicted shot window, never model-side '
            'aiming. Use STREAM for dense coverage or uncertain beam tracking. '
            'Use SELECTIVE when selective_supported is true and few useful '
            'columns remain. UNKNOWN retains the code default.',
            {'stream': 'Dense firing opportunities or uncertain beam tracking; prioritize throughput.',
             'selective': 'Explicitly supported sparse targets with a stable motion estimate.',
             'unknown': 'Insufficient evidence.'}),
        'repair': Noul(
            'Does evidence explicitly report stalled_motion OR prolonged_no_score? '
            'Visibility uncertainty alone is NOT stalled motion or no progress. '
            'This flags a failed program for replanning at its safe boundary.',
            {'true': 'At least one of those two explicit failure facts is true.',
             'false': 'Neither explicit failure fact is true.'}),
    }


class Advisor:
    """Single persistent Jev thread, bounded latest-state mailbox, no backlog."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.pending = None
        self.answer = None
        self.future = None
        self.enabled = getattr(ctx, 'system1', True) is not None

    def start(self):
        if self.enabled:
            self.future = self.ctx.spawn(self.work)

    def submit(self, packet):
        if self.enabled and not self.stop.is_set():
            with self.lock:
                self.pending = packet
            self.wake.set()

    def take(self):
        if self.future is not None and self.future.done():
            try:
                self.future.result()
            except CancelledError:
                pass
            self.enabled = False
        with self.lock:
            answer, self.answer = self.answer, None
        return answer

    def work(self):
        qs = questions()
        while not self.stop.is_set() and not self.ctx.done:
            self.wake.wait(0.05)
            self.wake.clear()
            if self.stop.is_set() or self.ctx.done:
                return
            with self.lock:
                packet, self.pending = self.pending, None
            if packet is None:
                continue
            if self.ctx.budget.remaining().get('system1_calls', 0) < 2:
                return
            t = self.ctx.clock.now()
            try:
                ans = self.ctx.decide(packet['state'], qs, timeout_s=2.0)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
                self.ctx.log('jev_failed', error=type(exc).__name__)
                self.stop.wait(0.15)
                continue
            with self.lock:
                self.answer = {'answers': ans, 'tag': packet['tag'],
                               'latency': max(0, self.ctx.clock.now() - t)}

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.lock:
            self.pending = None


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = {'focus': 'broad', 'fire': 'stream'}
        self.version = 0
        self.frame = 0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.px, self.py = 90.0, 190.0
        self.old = None
        self.alive = []
        self.columns = {}
        self.shot = {}
        self.motion = deque(maxlen=28)
        self.speeds = deque(maxlen=24)
        self.speed = 0.5
        self.laser_speed = 2.0
        self.vx = 0.0
        self.ready = False
        self.reacquire = 0
        self.last_action = 'fire'
        self.direction = 1
        self.route = None
        self.route_id = 0
        # A stalled route is an escape failure, not evidence that the opposite
        # lane is safe.  Preserve its heading for the next bounded traverse.
        self.stall_escape_pending = False
        self.completed = 0
        self.route_log = deque(maxlen=6)
        self.advice = None
        self.last_progress = 0
        self.last_moved = 0
        self.last_fire = -1000
        self.last_bomb = -10000
        self.tracks = []
        self.blind_deaths = 0
        self.life_event = False
        self.wave_event = False
        self.stalled = False
        self.fires = 0
        self.safety_overrides = 0
        self.accepted = 0
        self.effective = 0
        self.astra_effective = 0
        self.actions = 0

    def observe(self, observation):
        self.life_event = False
        self.wave_event = False
        f = int(num(observation.get('frame'), self.frame + 4))
        df = max(1, f - self.frame)
        p = obj(observation.get('player'))
        px = clamp(num(p.get('x'), self.px), 0, 159)
        self.py = clamp(num(p.get('y'), self.py), 150, 209)
        lives = int(num(observation.get('lives'), self.lives))
        wave = int(num(observation.get('wave'), self.wave))
        shot = obj(observation.get('player_shot'))
        self.life_event = lives < self.lives
        self.wave_event = wave != self.wave
        discontinuity = self.old is not None and abs(px - self.px) > max(12, df * 1.5)
        if self.life_event or self.wave_event or discontinuity:
            if self.life_event:
                blind = f - self.last_bomb > 90
                self.blind_deaths += int(blind)
                self.ctx.log('life_loss', frame=f, previous_x=self.px, blind=blind,
                             route_id=self.route_id, score=self.score)
            self.route = None
            self.advice = None
            self.tracks = []
            self.stall_escape_pending = False
            self.motion.clear()
            self.vx = 0.0
            self.reacquire = f + 12
            self.last_moved = f
        elif self.old is not None:
            dx = px - self.px
            if abs(dx) >= 0.5:
                self.last_moved = f
                self.ready = True
            d = -1 if 'left' in self.last_action else 1 if 'right' in self.last_action else 0
            if d and 0.1 <= dx * d / df <= 1.3:
                self.speeds.append(abs(dx) / df)
                self.speed = clamp(statistics.median(self.speeds), 0.25, 1.0)
            oldshot = obj(self.old.get('player_shot'))
            if oldshot.get('active') and shot.get('active'):
                if abs(num(oldshot.get('x')) - num(shot.get('x'))) <= 2:
                    speed = (num(oldshot.get('y')) - num(shot.get('y'))) / df
                    if 0.5 <= speed <= 4:
                        self.laser_speed = 0.8 * self.laser_speed + 0.2 * speed
        if shot.get('active'):
            self.ready = True
        score = num(observation.get('score'), self.score)
        if score > self.score:
            self.last_progress = f
        self.score, self.frame = score, f
        self.lives, self.wave, self.px, self.shot = lives, wave, px, shot
        self.alive = [a for a in items(observation.get('invaders')) if a.get('alive')
                      and isinstance(a.get('x'), (int, float))
                      and isinstance(a.get('y'), (int, float))]
        self.columns = {}
        for a in self.alive:
            col = int(num(a.get('col'), -1))
            if col not in self.columns or num(a['y']) > num(self.columns[col]['y']):
                self.columns[col] = a
        if self.alive:
            lattice = statistics.median(num(a['x']) - 16 * num(a.get('col')) for a in self.alive)
            self.motion.append((f, lattice))
            while self.motion and f - self.motion[0][0] > 64:
                self.motion.popleft()
            if len(self.motion) >= 3:
                f0, x0 = self.motion[0]
                self.vx = clamp((lattice - x0) / max(1, f - f0), -0.8, 0.8)
                if abs(lattice - x0) >= 1:
                    self.ready = True
        tracks = []
        for bomb in items(observation.get('bombs'))[:12]:
            if not isinstance(bomb.get('x'), (int, float)) or not isinstance(bomb.get('y'), (int, float)):
                continue
            x, y = num(bomb['x']), num(bomb['y'])
            matches = [b for b in self.tracks if abs(b['x'] - x) < 3
                       and 0 < f - b['frame'] <= 16 and y > b['y']]
            v = 1.2
            if matches:
                b = min(matches, key=lambda b: abs(y - b['y'] - 1.2 * (f - b['frame'])))
                v = clamp((y - b['y']) / (f - b['frame']), 0.5, 3)
            tracks.append({'x': x, 'y': y, 'v': v, 'frame': f})
        if tracks:
            self.last_bomb = f
        for b in self.tracks:
            if f - b['frame'] <= 12 and not any(abs(b['x'] - n['x']) < 3 for n in tracks):
                tracks.append(b)
        self.tracks = tracks[:16]
        moving = 'left' in self.last_action or 'right' in self.last_action
        self.stalled = self.ready and moving and f - self.last_moved >= 24
        self.old = observation

    def intercept(self, alien):
        flight = max(0, self.py - 6 - num(alien.get('y'))) / self.laser_speed
        return clamp(num(alien.get('x')) + clamp(self.vx * flight, -12, 12), 18, 141)

    def program_set(self):
        targets = list(self.columns.values())
        xs = [self.intercept(a) for a in targets]
        lo, hi = (min(xs), max(xs)) if xs else (35, 125)
        # Keep an escape traverse even after the target band becomes narrow.
        def band(a, b):
            center = (a + b) / 2
            half = max(18, (b - a) / 2 + 4)
            return [clamp(center - half, 20, 139), clamp(center + half, 20, 139)]
        broad = band(lo, hi)
        bottom = max([num(a['y']) for a in targets] or [0])
        front = [self.intercept(a) for a in targets if num(a['y']) >= bottom - 6]
        front_distinct = len(front) < len(targets) and bottom >= 118 and bool(front)
        front_band = band(min(front), max(front)) if front else broad
        # Flank selection uses live low threats, not fixed training-seed lanes.
        edge_targets = sorted(targets, key=lambda a: num(a['x']))
        if edge_targets:
            left, right = edge_targets[0], edge_targets[-1]
            edge = max((left, right), key=lambda a: (num(a['y']), -abs(self.intercept(a)-self.px)))
            ex = self.intercept(edge)
            edge_band = band(ex - 10, ex + 10)
        else:
            edge_band = broad
        selective = len(targets) <= 3 and len(self.motion) >= 6 and self.ready
        return {
            'broad': {'band': broad, 'allowed': True},
            'front': {'band': front_band, 'allowed': front_distinct},
            'edge': {'band': edge_band, 'allowed': 1 <= len(targets) <= 4},
        }, {'front_distinct': front_distinct, 'edge_viable': 1 <= len(targets) <= 4,
            'selective_supported': selective, 'coverage': 'dense' if len(targets) >= 4 else 'sparse',
            'stalled_motion': bool(self.stalled),
            'prolonged_no_score': bool(self.frame - self.last_progress >= 300),
            'low_formation': bottom >= 150,
            'visibility': 'unknown, never a certificate of safety',
            'blind_death_observed': self.blind_deaths > 0,
            'in_transit': self.route is not None,
            'recent_programs': [r['program'] for r in self.route_log][-3:]}

    def packet(self, programs, evidence):
        return {'tag': {'frame': self.frame, 'life': self.lives, 'wave': self.wave,
                        'version': self.version},
                'state': {'rules': 'Commit to spatial traverses, not joystick alternation. One beam is enforced by ROM. Never infer bomb absence from decoder silence.',
                          'plan': dict(self.plan), 'evidence': evidence,
                          'programs': {k: {'allowed': v['allowed'],
                                         'span': 'wide' if v['band'][1]-v['band'][0] >= 60 else 'moderate'}
                                       for k, v in programs.items()}}}

    def accept(self, result):
        if not isinstance(result, dict):
            return
        tag = obj(result.get('tag'))
        if (tag.get('life') != self.lives or tag.get('wave') != self.wave
                or tag.get('version') != self.version
                or not 0 <= self.frame - num(tag.get('frame'), -10000) <= 48):
            self.ctx.log('jev_discarded', reason='age_or_epoch')
            return
        answers = obj(result.get('answers'))
        a, b = answers.get('program'), answers.get('crosscheck')
        key = getattr(a, 'value', None)
        def supports(answer, choice, threshold):
            probs = obj(getattr(answer, 'probabilities', None))
            confidence = getattr(answer, 'confidence', None)
            return (num(probs.get(choice)) >= threshold
                    and (confidence is None or num(confidence, -1) >= 0.3))
        if (not isinstance(key, str) or key not in ('broad', 'front', 'edge')
                or getattr(b, 'value', None) != key
                or not supports(a, key, 0.6) or not supports(b, key, 0.6)):
            return
        mode = getattr(answers.get('fire_mode'), 'value', None)
        if not isinstance(mode, str) or mode not in ('stream', 'selective') or not supports(answers.get('fire_mode'), mode, 0.65):
            mode = None
        self.advice = {'program': key, 'fire': mode, 'tag': tag,
                       'repair': num(getattr(answers.get('repair'), 'value', 0)) >= 0.8}
        self.accepted += 1
        self.ctx.log('jev_program_accepted', program=key, fire=mode,
                     age_frames=self.frame - tag['frame'], latency_s=result.get('latency'))

    def fresh_advice(self):
        if not self.advice:
            return None
        t = self.advice['tag']
        if (t['life'] == self.lives and t['wave'] == self.wave and t['version'] == self.version
                and self.frame - t['frame'] <= 120):
            return self.advice
        return None

    def commit(self, programs, evidence):
        fallback = self.plan['focus']
        if not programs[fallback]['allowed']:
            fallback = 'broad'
        # Observable stall overrides stale semantic advice, not vice versa.
        if evidence['prolonged_no_score']:
            fallback = 'broad'
        advice = self.fresh_advice()
        chosen = fallback
        if advice and programs[advice['program']]['allowed'] and not evidence['prolonged_no_score']:
            chosen = advice['program']
        if chosen != fallback:
            self.effective += 1
            self.ctx.log('jev_changed_program', fallback=fallback, selected=chosen, frame=self.frame)
        lo, hi = programs[chosen]['band']
        # Route endpoints are frozen until reached; no chasing a moving target.
        if self.stall_escape_pending:
            # The old route stalled away from a boundary.  Continue out of the
            # suspected hidden lane once before allowing the normal sweep logic
            # to choose another heading.
            d = self.direction
            self.stall_escape_pending = False
        elif self.completed == 0 and self.route_id == 0:
            d = 1 if self.px <= (lo + hi) / 2 else -1
        else:
            d = -self.direction
        if self.px < lo + 3:
            d = 1
        elif self.px > hi - 3:
            d = -1
        destination = hi if d > 0 else lo
        if abs(destination - self.px) < 14:
            d = -d
            destination = hi if d > 0 else lo
        self.direction = d
        mode = 'selective' if evidence['selective_supported'] else 'stream'
        if self.plan['fire'] == 'selective' and evidence['selective_supported']:
            mode = 'selective'
        if advice and advice['fire'] == 'stream':
            mode = 'stream'
        elif advice and advice['fire'] == 'selective' and evidence['selective_supported']:
            mode = 'selective'
        if evidence['prolonged_no_score']:
            mode = 'stream'
        self.route_id += 1
        duration = abs(destination - self.px) / max(self.speed, 0.25) + 48
        self.route = {'id': self.route_id, 'program': chosen, 'fire': mode, 'x': destination,
                      'start': self.frame, 'from_x': self.px, 'score': self.score,
                      'expires': self.frame + int(duration), 'version': self.version}
        self.ctx.log('program_committed', **self.route, direction=d)

    def risk(self, direction):
        worst = 0.0
        previous = -1 if 'left' in self.last_action else 1 if 'right' in self.last_action else 0
        for b in self.tracks:
            age = self.frame - b['frame']
            y = b['y'] + age * b['v']
            for velocity in (0.7, b['v'], 2.2):
                for dt in range(0, 49, 4):
                    by = y + velocity * dt
                    if not self.py - 12 <= by <= self.py + 6:
                        continue
                    for delay in (0, 4):
                        x = clamp(self.px + self.speed * (previous * min(delay, dt)
                                  + direction * max(0, dt-delay)), 16, 143)
                        gap = abs(x - b['x'])
                        if gap < 11:
                            worst = max(worst, 100 + 11-gap + (48-dt)/48)
                        elif gap < 18:
                            worst = max(worst, (18-gap)/7)
        return worst

    def action(self, programs, evidence):
        if not self.ready or self.frame < self.reacquire:
            return 'fire', 'reacquire'
        r = self.route
        if r is not None:
            reached = (r['x'] - self.px) * self.direction <= max(2, self.speed*4)
            expired = self.frame >= r['expires']
            blocked = self.stalled and self.frame-r['start'] >= 28
            if reached or expired or blocked:
                result = {'program': r['program'], 'frames': self.frame-r['start'],
                          'distance': round(abs(self.px-r['from_x']), 1),
                          'score_gain': self.score-r['score'],
                          'reason': 'reached' if reached else 'blocked' if blocked else 'expired'}
                self.route_log.append(result)
                self.ctx.log('program_finished', route_id=r['id'], **result)
                self.completed += 1
                if blocked and not reached and not expired:
                    # Do not immediately send the cannon back through the
                    # same interior region.  commit() will still turn at a
                    # real coordinate edge or a too-short destination.
                    self.stall_escape_pending = True
                self.route = None
                self.last_moved = self.frame
        if self.route is None:
            self.commit(programs, evidence)
        r = self.route
        d = self.direction
        risks = {k: self.risk(k) for k in (-1, 0, 1)}
        if risks[d] >= 100:
            safe = min(risks, key=lambda k: (risks[k], k != self.direction, k == 0))
            if risks[safe] < risks[d]:
                d = safe
                self.safety_overrides += 1
                self.ctx.log('bomb_override', frame=self.frame, direction=d, route_id=r['id'])
        if self.px <= 16 and d < 0:
            d = 1
        elif self.px >= 143 and d > 0:
            d = -1
        # The hardware lock, not a guessed cooldown, prevents duplicate lasers.
        # Busy FIRE is harmless and immediately re-arms on a decoder-hidden hit.
        fire = r['fire'] == 'stream' or bool(self.shot.get('active'))
        if not fire:
            launch = self.px + d * self.speed * 2
            fire = any(abs(self.intercept(a)-launch) <= 5 for a in self.alive)
            # Selective mode may not silently suppress firing indefinitely.
            if self.frame-self.last_fire > 90:
                fire = any(abs(self.intercept(a)-launch) <= 12 for a in self.alive)
        if fire:
            self.last_fire = self.frame
            self.fires += 1
        base = 'right' if d > 0 else 'left' if d < 0 else 'noop'
        if self.version > 0 and r['version'] == self.version:
            self.astra_effective += 1
        return ('fire' if base == 'noop' else base+'_fire') if fire else base, r['program']

    def astra_prompt(self):
        data = {'plan': self.plan, 'score': self.score, 'lives': self.lives,
                'blind_deaths': self.blind_deaths, 'completed_routes': list(self.route_log),
                'no_score_frames': self.frame-self.last_progress,
                'observed_columns': sorted(self.columns), 'wave': self.wave}
        return (
            'Choose a policy for a real-time Atari Space Invaders controller. '
            'Reply ONLY a short JSON object: {"focus":"broad","fire":"stream"}. '
            'focus must be broad/front/edge, fire stream/selective. No explanation. '
            'You are off the actuator path; your policy applies to future traverses. '
            'Archive: synchronous typed-column control scored 188.125 but had 55.5% '
            'stale ticks. Best 198.125 used tick-local relocation; its worst episode '
            'chose relocate in all 96 Jev batches and mostly reversed after 8 ticks. '
            'All lives were lost, no bombs were decoded, no wave cleared. '
            'Hypothesis: short reversals revisit unseen bomb lanes; sustained traverses '
            'and hardware-rate firing are better than repeatedly relocating locally. '
            'Current code freezes an endpoint until reached, blocked, or timed out; '
            'visible collision checks override. Broad covers all living columns; front '
            'covers a distinct low band; edge works a live outer flank, never a static '
            'cannon. Stream holds FIRE while moving: the ROM allows ONE laser only; '
            'selective opens computed intercept windows only for sparse tracked targets. '
            'No empty-bomb safety assumption. Your focus is included in Jev state and '
            'changes the code fallback at the next program boundary. '
            'Jev local contract: ctx.decide returns a dict keyed by question ids; '
            'Choice.value is its literal option, with probabilities/confidence; '
            'Noul.value is a yes probability. It cannot do reliable arithmetic. '
            'We supply categorical facts, two reordered Choices broad/front/edge/unknown '
            '(agreement and each probability >=0.60), fire_mode stream/selective/unknown '
            '(>=0.65), and Noul repair (>=0.80 plus a fresh explicit failure). '
            'Unknowns use fallback; replies expire on life/wave/version changes. '
            'Select a policy that helps those existing recognizers; do not invent code '
            'or new API fields. Evidence: ' + json.dumps(data, allow_nan=False)
        )

    def apply_astra(self, text, wave):
        if wave != self.wave or not isinstance(text, str):
            return
        try:
            start, end = text.find('{'), text.rfind('}')
            data = json.loads(text[start:end+1]) if 0 <= start < end else {}
        except (ValueError, TypeError):
            self.ctx.log('astra_rejected', reason='malformed_json')
            return
        if not isinstance(data, dict):
            return
        focus, fire = data.get('focus'), data.get('fire')
        if not isinstance(focus, str) or focus not in ('broad', 'front', 'edge'):
            return
        if not isinstance(fire, str) or fire not in ('stream', 'selective'):
            return
        previous = self.plan
        self.plan = {'focus': focus, 'fire': fire}
        self.version += 1
        self.advice = None
        self.ctx.log('astra_policy_applied', plan=self.plan, previous=previous,
                     version=self.version, takes_effect='next_program_boundary')


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        controller = Controller(ctx)
        advisor = Advisor(ctx)
        legal = set(game.action_spec.names)
        noop = game.action_spec.noop
        astra = None
        astra_wave = None
        astra_count = 0
        last_astra = -10000
        last_query = -10000
        last_frame = None
        try:
            advisor.start()
            while not ctx.done and not game.done:
                observation = ctx.observe()
                if not isinstance(observation, dict):
                    ctx.act('fire' if 'fire' in legal else noop)
                    ctx.wait_for_tick()
                    continue
                if observation.get('game_over'):
                    break
                frame = int(num(observation.get('frame'), (last_frame or 0)+4))
                if frame == last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame = frame
                controller.observe(observation)
                controller.accept(advisor.take())
                if astra is not None and astra.done():
                    try:
                        reply = astra.result()
                        controller.apply_astra(getattr(reply, 'text', ''), astra_wave)
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except (CancelledError, TimeoutError, RuntimeError, TypeError, ValueError) as exc:
                        ctx.log('astra_failed', error=type(exc).__name__)
                    astra = None
                programs, evidence = controller.program_set()
                action, source = controller.action(programs, evidence)
                ctx.act(action if action in legal else noop)
                controller.last_action = action
                controller.actions += 1
                if frame-last_query >= 24 or controller.life_event or controller.wave_event:
                    advisor.submit(controller.packet(programs, evidence))
                    last_query = frame
                remaining = ctx.budget.remaining()
                # Start before a death: archived reactive calls never informed an action.
                initial = astra_count == 0
                revise = (controller.life_event or controller.wave_event
                          or (evidence['prolonged_no_score'] and controller.completed >= 2))
                if (astra is None and astra_count < 2 and (initial or revise)
                        and frame-last_astra >= 1500
                        and remaining.get('system2_calls', 0) >= 2
                        and remaining.get('wall_s', 0) > 40):
                    prompt = controller.astra_prompt()
                    astra_wave = controller.wave
                    astra = ctx.spawn(lambda p=prompt: ctx.think(p, timeout_s=55.0))
                    astra_count += 1
                    last_astra = frame
                    ctx.log('astra_requested', frame=frame, reason='startup' if initial else 'failed_program')
                if controller.actions % 150 == 0:
                    ctx.log('controller_summary', frame=frame, score=controller.score,
                            lives=controller.lives, source=source, completed=controller.completed,
                            jev_accepted=controller.accepted, jev_effective=controller.effective,
                            astra_effective=controller.astra_effective)
                ctx.wait_for_tick()
        finally:
            advisor.close()
            ctx.log('controller_finished', actions=controller.actions,
                    routes=controller.completed, jev_accepted=controller.accepted,
                    jev_effective=controller.effective, astra_requests=astra_count,
                    astra_effective=controller.astra_effective,
                    safety_overrides=controller.safety_overrides)


def build(ctx):
    return Harness(ctx)
