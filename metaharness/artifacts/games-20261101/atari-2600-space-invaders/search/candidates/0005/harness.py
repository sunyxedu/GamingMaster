"""Candidate 0005: bounded quiet periods, shoot-and-relocate, asynchronous advice.

Only symbolic public observations are used. Code owns geometry and timing;
Jev selects a short-lived maneuver, and Astra revises finite strategic presets.
"""
from __future__ import annotations

import json
import math
import statistics
import threading
from concurrent.futures import CancelledError
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(x, low, high):
    return max(low, min(high, x))


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def objects(value):
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def questions():
    options = {
        'engage': 'Approach the selected living column; code aims and fires. Prefer for progress when its route is clear.',
        'relocate': 'Move to the alternative lane while the laser travels, exposure is overdue, or visibility is unreliable. Do not choose a threatened route.',
        'quiet': 'Briefly release movement in cover or while already aligned. Never choose when quiet_allowed is false or exposure is overdue.',
        'unknown': 'No described candidate is supported by the evidence; use the checked fallback.',
    }
    instruction = (
        'Choose a maneuver using candidates, evidence, and plan. All geometry and '
        'comparisons have already been computed. A clear route means no OBSERVED '
        'collision, not proof of safety: invisible bombs are possible. Favor '
        'progress, but relocate when exposure is overdue. Quiet requires '
        'quiet_allowed=true. Do not infer safety from an empty bomb list. '
        'The code rechecks your choice on the current tick.'
    )
    return {
        'maneuver': Choice(instruction, options),
        'crosscheck': Choice(instruction, dict(reversed(list(options.items())))),
        'quiet_supported': Noul(
            'Does evidence.quiet_allowed explicitly permit a brief stationary '
            'period, AND is candidates.quiet.route clear, AND is exposure not '
            'overdue? Answer no if any condition is false or unknown.',
            {'true': 'All three stated conditions are supported.',
             'false': 'Any condition is false or unknown.'}),
    }


class Advisor:
    """One persistent worker: one request in flight, one latest-state mailbox."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.pending = None
        self.result = None
        self.future = None
        self.enabled = getattr(ctx, 'system1', True) is not None

    def start(self):
        if self.enabled:
            self.future = self.ctx.spawn(self.work)

    def submit(self, packet):
        if not self.enabled or self.stop.is_set():
            return
        with self.lock:
            self.pending = packet
        self.wake.set()

    def take(self):
        with self.lock:
            result, self.result = self.result, None
        if self.future is not None and self.future.done():
            # Fatal backend errors are not hidden inside a Future.
            try:
                self.future.result()
            except CancelledError:
                self.enabled = False
                return result
            self.enabled = False
        return result

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
            if self.ctx.budget.remaining().get('system1_calls', 0) <= 2:
                return
            started = self.ctx.clock.now()
            try:
                answers = self.ctx.decide(packet['state'], qs, timeout_s=2.0)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError) as exc:
                self.ctx.log('jev_unavailable_for_tick', error=type(exc).__name__)
                self.stop.wait(0.2)
                continue
            result = {'tag': packet['tag'], 'answers': answers,
                      'latency_s': max(0.0, self.ctx.clock.now() - started)}
            with self.lock:
                self.result = result

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.lock:
            self.pending = None


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = {'focus': 'nearest', 'posture': 'mobile'}
        self.version = 0
        self.previous = None
        self.last_action = 'noop'
        self.frame = 0
        self.wave = 1
        self.lives = 3
        self.px = 90.0
        self.py = 190.0
        self.alive = []
        self.shields = []
        self.bombs = []
        self.tracks = []
        self.shot = {}
        self.shot_speed = 2.0  # Median observed dy/frame in both worst parents.
        self.move_speed = 0.8  # Initial uncertainty; replace with observed motion.
        self.move_samples = deque(maxlen=24)
        self.formation = deque(maxlen=30)
        self.vx = 0.0
        self.ready = False
        self.anchor_x = None
        self.dwell = 0
        self.exposure_limit = 36
        self.last_score = 0.0
        self.last_progress = 0
        self.last_bomb = -100000
        self.unseen_deaths = 0
        self.death_lanes = []
        self.target_col = None
        self.last_fired_col = None
        self.last_launch = -100000
        self.fire_command_frame = -100000
        self.jumps_until = 0
        self.relocate_x = None
        self.relocate_until = 0
        self.events = deque(maxlen=8)
        self.life_event = False
        self.advice = None
        self.advice_accepted = 0
        self.advice_changed = 0
        self.actions = 0
        self.safety_overrides = 0

    def observe(self, obs):
        self.life_event = False
        frame = int(number(obs.get('frame'), self.frame + 4))
        old = self.previous
        df = max(1, frame - self.frame) if old else 4
        self.frame = frame
        self.wave = int(number(obs.get('wave'), self.wave))
        lives = int(number(obs.get('lives'), self.lives))
        player = obs.get('player') if isinstance(obs.get('player'), dict) else {}
        px = clamp(number(player.get('x'), self.px), 0, 159)
        self.py = clamp(number(player.get('y'), 190), 150, 209)
        shot = obs.get('player_shot')
        shot = shot if isinstance(shot, dict) else {}
        self.alive = [a for a in objects(obs.get('invaders')) if a.get('alive') is True
                      and isinstance(a.get('x'), (int, float))
                      and isinstance(a.get('y'), (int, float))]
        self.shields = objects(obs.get('shields'))
        bombs = objects(obs.get('bombs'))
        if lives < self.lives:
            self.life_event = True
            unseen = frame - self.last_bomb > 90
            self.unseen_deaths += int(unseen)
            self.death_lanes.append((self.px, frame))
            self.relocate_x = None
            self.tracks = []
            self.target_col = None
            self.advice = None
            self.dwell = 0
            event = {'event': 'life_lost', 'frame': frame, 'unobserved_threat': unseen}
            self.events.append(event)
            self.ctx.log('life_loss_evidence', **event, previous_lane=round(self.px, 1))
        if old and self.wave != int(number(old.get('wave'), self.wave)):
            self.formation.clear()
            self.relocate_x = None
            self.tracks = []
            self.target_col = None
            self.advice = None
            self.vx = 0.0
            self.last_progress = frame
        if old:
            op = old.get('player') if isinstance(old.get('player'), dict) else {}
            dx = px - number(op.get('x'), self.px)
            if abs(dx) > max(12, 2 * df):
                self.jumps_until = frame + 12
                self.relocate_x = None
                self.advice = None
                self.anchor_x = px
                self.dwell = 0
            elif lives == self.lives and ('left' in self.last_action or 'right' in self.last_action):
                direction = 1 if 'right' in self.last_action else -1
                if 0.15 <= dx * direction / df <= 1.5:
                    self.move_samples.append(abs(dx) / df)
                    self.move_speed = clamp(statistics.median(self.move_samples), 0.4, 1.25)
            os = old.get('player_shot')
            if isinstance(os, dict) and os.get('active') and shot.get('active'):
                if abs(number(os.get('x')) - number(shot.get('x'))) <= 2:
                    speed = (number(os.get('y')) - number(shot.get('y'))) / df
                    if 0.5 <= speed <= 4:
                        self.shot_speed = 0.8 * self.shot_speed + 0.2 * speed
        os = old.get('player_shot', {}) if old else {}
        if not isinstance(os, dict):
            os = {}
        launched = bool(shot.get('active')) and (
            not os.get('active') or number(shot.get('y')) > number(os.get('y')) + 20)
        if launched:
            self.last_launch = frame
            self.last_fired_col = self.target_col
            self.target_col = None  # Relocate during flight, not after it.
            self.ready = True
        self.shot = shot
        self.lives = lives
        self.px = px
        if self.anchor_x is None or abs(px - self.anchor_x) >= 8:
            self.anchor_x = px
            self.dwell = 0
        else:
            self.dwell += df
        score = number(obs.get('score'))
        if score > self.last_score:
            self.last_progress = frame
        self.last_score = score
        if self.alive:
            # Columns retain their IDs even when outer columns disappear.
            lattice = statistics.median(number(a['x']) - 16 * number(a.get('col')) for a in self.alive)
            self.formation.append((frame, lattice))
            while self.formation and frame - self.formation[0][0] > 96:
                self.formation.popleft()
            if len(self.formation) > 1:
                f0, x0 = self.formation[0]
                raw_vx = (lattice - x0) / max(1, frame - f0)
                self.vx = clamp(raw_vx, -0.6, 0.6)
                if abs(lattice - x0) >= 1:
                    self.ready = True
        new_tracks = []
        for b in bombs[:12]:
            if not isinstance(b.get('x'), (int, float)) or not isinstance(b.get('y'), (int, float)):
                continue
            bx, by = number(b['x']), number(b['y'])
            velocity = 1.2
            matches = [t for t in self.tracks if abs(t['x'] - bx) <= 3
                       and 0 < frame - t['frame'] <= 16
                       and 0 < by - t['y'] <= 4 * (frame - t['frame'])]
            if matches:
                t = min(matches, key=lambda t: abs(by - t['y'] - 1.2 * (frame - t['frame'])))
                velocity = clamp((by - t['y']) / (frame - t['frame']), 0.4, 3.0)
            new_tracks.append({'x': bx, 'y': by, 'v': velocity, 'frame': frame})
        if new_tracks:
            self.last_bomb = frame
        # Extrapolate recently seen bombs through a short decoder gap only.
        for t in self.tracks:
            if frame - t['frame'] <= 12 and not any(abs(t['x'] - n['x']) <= 3 for n in new_tracks):
                new_tracks.append(t)
        self.tracks = new_tracks[:16]
        self.bombs = [{'x': t['x'], 'y': t['y'] + t['v'] * (frame - t['frame']),
                       'v': t['v']} for t in self.tracks if t['y'] < self.py + 8]
        self.death_lanes = [(x, f) for x, f in self.death_lanes if frame - f < 240]
        self.exposure_limit = 24 if self.unseen_deaths else (54 if self.plan['posture'] == 'cover' else 36)
        self.previous = obs

    def cover(self, x):
        # Eight mask columns, nine two-scanline rows; conservative center mapping.
        # Cover is an occupancy hint, NEVER a reason to ignore a visible bomb.
        best = 0
        for shield in self.shields:
            sx = number(shield.get('x'), -100)
            masks = shield.get('health_mask')
            if not isinstance(masks, list) or not sx - 4 <= x < sx + 4:
                continue
            bit = 7 - int(clamp(x - (sx - 4), 0, 7))
            occupied = sum(bool(int(number(m)) & (1 << bit)) for m in masks[:9])
            best = max(best, occupied)
        return best

    def intercept(self, alien):
        flight = max(0.0, self.py - 6 - number(alien.get('y'))) / max(0.5, self.shot_speed)
        return clamp(number(alien.get('x')) + clamp(self.vx * flight, -14, 14), 12, 147)

    def target(self):
        columns = {}
        for a in self.alive:
            col = int(number(a.get('col')))
            if col not in columns or number(a['y']) > number(columns[col]['y']):
                columns[col] = a
        if not columns:
            return None
        if self.target_col in columns:
            return columns[self.target_col]
        edges = (min(columns), max(columns))
        lowest = max(number(a['y']) for a in columns.values())
        def cost(item):
            col, a = item
            x = self.intercept(a)
            value = abs(x - self.px) + 0.12 * (lowest - number(a['y']))
            if self.plan['focus'] == 'outer' and col in edges:
                value -= 9
            if self.shot.get('active') and col == self.last_fired_col and len(columns) > 1:
                value += 22
            value += sum(14 for dx, _ in self.death_lanes if abs(x - dx) < 10)
            return value
        self.target_col, a = min(columns.items(), key=cost)
        return a

    def route_risk(self, target_x):
        risk = 0.0
        direction = sign(target_x - self.px)
        for bomb in self.bombs:
            # Use a velocity envelope until a track is well observed.
            for velocity in {0.7, bomb['v'], 2.0}:
                for dt in range(0, 49, 4):
                    y = bomb['y'] + velocity * dt
                    if not self.py - 12 <= y <= self.py + 7:
                        continue
                    # Check both normal response and one-tick sticky delay.
                    for delay in (0, 4):
                        travel = min(abs(target_x - self.px), self.move_speed * max(0, dt - delay))
                        x = self.px + direction * travel
                        gap = abs(x - bomb['x'])
                        if gap < 11:
                            risk = max(risk, 100.0 + (11 - gap) + (48 - dt) / 48)
                        elif gap < 18:
                            risk = max(risk, (18 - gap) / 7)
        return risk

    def candidate_set(self):
        target = self.target()
        target_x = self.intercept(target) if target else clamp(self.px, 24, 136)
        # Select a nearby relocation, not a long blind traversal.
        alternatives = [clamp(self.px - 18, 16, 143), clamp(self.px + 18, 16, 143)]
        def alternative_cost(x):
            cost = 200 * self.route_risk(x) + 0.30 * abs(x - target_x)
            cost += max(0, 24 - x) + max(0, x - 135)
            cost += sum(18 for dx, _ in self.death_lanes if abs(x - dx) < 11)
            return cost
        relocate_x = min(alternatives, key=alternative_cost)
        committed = (self.relocate_x is not None and self.frame <= self.relocate_until
                     and abs(self.relocate_x - self.px) > 3)
        if committed and self.route_risk(self.relocate_x) <= self.route_risk(relocate_x) + 0.01:
            relocate_x = self.relocate_x
        else:
            committed = False
            self.relocate_x = None
        occupied = self.cover(self.px)
        overdue = self.dwell >= self.exposure_limit
        aligned = target is not None and abs(target_x - self.px) <= 4
        quiet_allowed = not overdue and not self.unseen_deaths and (occupied >= 4 or aligned)
        engage_cost = 5.0 + (6 if overdue and abs(target_x - self.px) < 8 else 0)
        relocate_cost = (-3 if committed else 0 if overdue else 10) + (0 if self.shot.get('active') else 5)
        quiet_cost = (2 if aligned and not self.shot.get('active') else 10)
        quiet_cost += 24 if not quiet_allowed else 0
        candidates = {
            'engage': {'x': target_x, 'cost': engage_cost},
            'relocate': {'x': relocate_x, 'cost': relocate_cost},
            'quiet': {'x': self.px, 'cost': quiet_cost},
        }
        for name, c in candidates.items():
            c['risk'] = self.route_risk(c['x'])
            c['cover'] = self.cover(c['x'])
            if name == 'engage' and self.shot.get('active') and abs(c['x'] - self.px) < 8:
                c['cost'] += 10
            c['cost'] += sum(8 for x, _ in self.death_lanes if abs(c['x'] - x) < 10)
        return candidates, {'quiet_allowed': bool(quiet_allowed), 'overdue': overdue,
                            'aligned': aligned, 'occupied': occupied, 'target': target}

    def packet(self, candidates, facts):
        target = facts['target']
        bottom = max([number(a['y']) for a in self.alive] or [0])
        state = {
            'rules': 'One laser in flight; movement is independent of firing. Empty bombs is not reliable safety evidence. Never chase UFOs.',
            'plan': dict(self.plan),
            'evidence': {
                'visibility': 'unreliable: death without detected bomb' if self.unseen_deaths else 'uncertain: sparse bomb decoder',
                'quiet_allowed': facts['quiet_allowed'],
                'exposure': 'overdue' if facts['overdue'] else 'brief',
                'laser': 'in flight; move toward next opportunity' if self.shot.get('active') else 'available; code fires when aligned',
                'progress': 'stalled' if self.frame - self.last_progress > 240 else 'recent or opening',
                'formation': 'low threat' if bottom >= 155 else 'normal height',
                'cover_here': 'occupied' if facts['occupied'] >= 4 else 'open or damaged',
                'target': 'living column selected' if target else 'not observed',
            },
            'candidates': {},
        }
        for key, c in candidates.items():
            state['candidates'][key] = {
                'route': 'collision predicted' if c['risk'] >= 100 else 'near observed bomb' if c['risk'] > 0 else 'clear',
                'movement': 'hold position' if abs(c['x'] - self.px) <= 3 else 'move left' if c['x'] < self.px else 'move right',
                'cover': 'occupied lane' if c['cover'] >= 4 else 'open or damaged lane',
                'aligned_shot': key == 'engage' and facts['aligned'],
            }
        return {'state': state, 'tag': {'frame': self.frame, 'wave': self.wave,
                'lives': self.lives, 'version': self.version, 'x': self.px}}

    def accept(self, result):
        if not result:
            return
        tag = result.get('tag', {})
        if (tag.get('version') != self.version or tag.get('wave') != self.wave
                or tag.get('lives') != self.lives or not 0 <= self.frame - tag.get('frame', -10000) <= 32):
            self.ctx.log('jev_discarded', reason='expired_or_context_changed')
            return
        answers = result.get('answers')
        if not isinstance(answers, dict):
            return
        a, b = answers.get('maneuver'), answers.get('crosscheck')
        choice = getattr(a, 'value', None)
        if not isinstance(choice, str) or choice not in ('engage', 'relocate', 'quiet'):
            return
        if choice != getattr(b, 'value', None):
            return
        for answer in (a, b):
            probabilities = getattr(answer, 'probabilities', None)
            if not isinstance(probabilities, dict) or number(probabilities.get(choice)) < 0.60:
                return
            confidence = getattr(answer, 'confidence', None)
            if confidence is not None and number(confidence, -1) < 0.30:
                return
        if choice == 'quiet' and number(getattr(answers.get('quiet_supported'), 'value', None)) < 0.80:
            return
        self.advice = {'choice': choice, **tag}
        self.advice_accepted += 1
        self.ctx.log('jev_intent_accepted', maneuver=choice, age_frames=self.frame-tag['frame'],
                     latency_s=result.get('latency_s'))

    def action(self, candidates, facts):
        # Ignore coordinates during the opening/respawn placeholder interval.
        if not self.ready or self.frame < self.jumps_until:
            return 'fire', 'reacquire'
        fallback = min(candidates, key=lambda k: (candidates[k]['risk'], candidates[k]['cost']))
        chosen = fallback
        advice = self.advice
        if advice and (advice['version'] == self.version and advice['wave'] == self.wave
                       and advice['lives'] == self.lives and self.frame - advice['frame'] <= 36):
            key = advice['choice']
            c, best = candidates[key], candidates[fallback]
            quiet_ok = key != 'quiet' or (facts['quiet_allowed'] and abs(self.px - advice['x']) <= 8)
            exposure_ok = (not facts.get('overdue', False) or abs(c['x'] - self.px) >= 8
                           or best['risk'] > 0)
            commitment_ok = (self.relocate_x is None or key == 'relocate'
                             or c['risk'] < candidates['relocate']['risk'])
            if (quiet_ok and exposure_ok and commitment_ok
                    and c['risk'] <= best['risk'] + 0.01 and c['cost'] <= best['cost'] + 10):
                chosen = key
        if chosen != fallback:
            self.advice_changed += 1
            self.ctx.log('jev_changed_control', fallback=fallback, selected=chosen, frame=self.frame)
        destination = candidates[chosen]['x']
        if chosen == 'relocate' and self.relocate_x is None:
            self.relocate_x = destination
            self.relocate_until = self.frame + 40
        elif chosen != 'relocate':
            self.relocate_x = None
        delta = destination - self.px
        deadband = max(1.5, self.move_speed * 2.5)
        direction = sign(delta) if abs(delta) > deadband else 0
        if self.px <= 12 and direction < 0 or self.px >= 147 and direction > 0:
            direction = 0
        fire = False
        if not self.shot.get('active') and self.frame - self.fire_command_frame >= 8:
            # A one-tick motion envelope avoids firing far before alignment.
            launch_x = self.px + direction * self.move_speed * 2
            for a in self.alive:
                if abs(self.intercept(a) - launch_x) <= 4.5:
                    fire = True
                    break
        if candidates[chosen]['risk'] >= 100:
            self.safety_overrides += 1
        if fire:
            self.fire_command_frame = self.frame
        base = 'right' if direction > 0 else 'left' if direction < 0 else 'noop'
        return ('fire' if base == 'noop' else base + '_fire') if fire else base, chosen

    def astra_prompt(self):
        evidence = {'plan': self.plan, 'wave': self.wave, 'lives': self.lives,
                    'score': self.last_score, 'unobserved_deaths': self.unseen_deaths,
                    'frames_without_score': self.frame - self.last_progress,
                    'recent_events': list(self.events)}
        return (
            'You adapt a real-time Atari 2600 Space Invaders controller, not individual actions. '
            'One laser at a time, 4 frames/tick, sticky actions 0.25. Public bomb observations '
            'are sparse: two archived losing episodes had zero bombs visible before all three deaths. '
            'Idle survived 46s with score zero; direct Jev got 139 mean but 58% stale ticks. '
            'Code now posts each tick, aims at predicted living columns and relocates during flight. '
            'Jev sees categorical route/cover/exposure facts, not raw arithmetic. In one batch it '
            'answers two reordered Choices over engage/relocate/quiet/unknown, plus Noul quiet_supported. '
            'Choices need agreement and winning probability >=0.60; quiet needs Noul >=0.80. '
            'Code rechecks geometry, freshness and legality; no model may disable its safety checks. '
            'Your finite preset is included in every Jev request; you are not being asked to design '
            'new Jev questions. Return schema: {"focus":"nearest","posture":"mobile","reason":"brief"}. '
            'Choose focus nearest (short travel, immediate shots) or outer (small edge-column preference); '
            'choose posture mobile (short dwell) or cover (longer quiet periods only with current cover '
            'and no observed blind deaths). Do not assume shields are intact or unseen bombs absent. '
            'Do not write code or new questions. Return only a short JSON object with focus, posture, '
            'and reason (at most twelve words). Evidence: ' + json.dumps(evidence, allow_nan=False)
        )

    def apply_astra(self, text, wave):
        if wave != self.wave or not isinstance(text, str):
            return
        try:
            start, end = text.find('{'), text.rfind('}')
            value = json.loads(text[start:end + 1]) if 0 <= start < end else {}
        except (ValueError, TypeError):
            self.ctx.log('astra_rejected', reason='invalid_json')
            return
        if not isinstance(value, dict):
            return
        new = dict(self.plan)
        if value.get('focus') in ('nearest', 'outer'):
            new['focus'] = value['focus']
        if value.get('posture') in ('mobile', 'cover'):
            new['posture'] = value['posture']
        if self.unseen_deaths:
            new['posture'] = 'mobile'
        if new != self.plan:
            self.plan = new
            self.version += 1
            self.target_col = None
            self.advice = None
            self.ctx.log('astra_policy_revision', plan=new, version=self.version)
        else:
            self.ctx.log('astra_policy_retained', plan=new)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        controller = Controller(ctx)
        advisor = Advisor(ctx)
        legal = set(game.action_spec.names)
        astra = None
        astra_wave = None
        astra_count = 0
        astra_last_frame = -10000
        last_query = -10000
        last_frame = None
        try:
            advisor.start()
            while not ctx.done:
                obs = ctx.observe()
                if not isinstance(obs, dict):
                    ctx.act('noop' if 'noop' in legal else game.action_spec.noop)
                    ctx.wait_for_tick()
                    continue
                if obs.get('game_over'):
                    break
                frame = int(number(obs.get('frame'), (last_frame or 0) + 4))
                if frame == last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame = frame
                controller.observe(obs)
                controller.accept(advisor.take())
                if astra is not None and astra.done():
                    try:
                        result = astra.result()
                        controller.apply_astra(getattr(result, 'text', ''), astra_wave)
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except (CancelledError, TimeoutError, RuntimeError, TypeError, ValueError) as exc:
                        ctx.log('astra_unavailable', error=type(exc).__name__)
                    astra = None
                candidates, facts = controller.candidate_set()
                action, source = controller.action(candidates, facts)
                if action not in legal:
                    action = 'noop' if 'noop' in legal else game.action_spec.noop
                ctx.act(action)
                controller.last_action = action
                controller.actions += 1
                if frame - last_query >= 16:
                    advisor.submit(controller.packet(candidates, facts))
                    last_query = frame
                remaining = ctx.budget.remaining()
                trigger = controller.life_event or (controller.ready and frame-controller.last_progress > 480)
                if (trigger and astra is None and astra_count < 2 and frame-astra_last_frame >= 600
                        and remaining.get('system2_calls', 0) > 1 and remaining.get('wall_s', 0) > 15):
                    prompt = controller.astra_prompt()
                    astra_wave = controller.wave
                    astra = ctx.spawn(lambda p=prompt: ctx.think(p, timeout_s=60.0))
                    astra_count += 1
                    astra_last_frame = frame
                    ctx.log('astra_requested', reason='death_or_stall', frame=frame)
                if controller.actions % 150 == 0:
                    ctx.log('controller_summary', frame=frame, score=controller.last_score,
                            lives=controller.lives, source=source, accepted=controller.advice_accepted,
                            changed=controller.advice_changed, unseen_deaths=controller.unseen_deaths)
                ctx.wait_for_tick()
        finally:
            advisor.close()
            ctx.log('controller_finished', actions=controller.actions,
                    jev_accepted=controller.advice_accepted, jev_changed_actions=controller.advice_changed,
                    astra_requests=astra_count, safety_overrides=controller.safety_overrides)


def build(ctx):
    return Harness(ctx)
