"""0017: continuous-fire sweeps with transactional, one-shot column intercepts.

All game information comes from public observations. Model calls never block
the actuator, and no target preference is allowed to become a parked cannon.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


ACTIONS = ('noop', 'fire', 'right', 'left', 'right_fire', 'left_fire')


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError, OverflowError):
        return default


def obj(value):
    return value if isinstance(value, dict) else {}


def rows(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def flag(value):
    return value is True or value == 1 or (isinstance(value, str) and value.lower() == 'true')


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def field(answer, name, default=None):
    return answer.get(name, default) if isinstance(answer, dict) else getattr(answer, name, default)


def trusted(answer, key, threshold=0.60):
    if field(answer, 'value') != key:
        return False
    probability = num(obj(field(answer, 'probabilities')).get(key), 0)
    confidence = field(answer, 'confidence')
    return probability >= threshold and (confidence is None or num(confidence, -1) >= 0.20)


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.x = 90.0
        self.y = 190.0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.direction = 1
        self.last_direction = 1
        self.speed = 0.55
        self.previous = None
        self.columns = {}
        self.alien_count = 36
        self.formation_speed = 0.0
        self.shot = {}
        self.shot_lanes = deque(maxlen=24)
        self.masks = None
        self.lanes = []
        self.last_escape = -10000
        self.last_gain = 0
        self.last_query = -10000
        self.epoch = 0
        self.policy_version = 0
        self.policy = 'SPARSE'
        self.lease = None
        self.cooldown_until = 0
        self.query_future = None
        self.query_tag = None
        self.astra_future = None
        self.astra_requested = False
        self.failures = 0
        self.next_query = 0
        self.history = deque(maxlen=8)
        self.stats = {'actions': 0, 'intercepts': 0, 'changed_actions': 0,
                      'life_losses': 0, 'shield_escapes': 0, 'astra_changes': 0}

    def log(self, event, **data):
        self.ctx.log(event, **data)

    def mask(self, obs):
        cells = {}
        for index, shield in enumerate(rows(obs.get('shields'))[:3]):
            masks = shield.get('health_mask')
            if not isinstance(masks, list):
                continue
            for r, bits in enumerate(masks[:9]):
                for b in range(8):
                    if int(num(bits)) & (1 << (7-b)):
                        cells[(index, r, b)] = num(shield.get('x')) - 4 + b
        return cells

    def end_lease(self, reason):
        if self.lease is None:
            return
        lease = self.lease
        outcome = {'column': lease['col'], 'reason': reason,
                   'duration_frames': self.frame-lease['start'],
                   'score_delta': self.score-lease['score'],
                   'changed_ticks': lease['changed']}
        self.history.append(outcome)
        self.log('intercept_outcome', frame=self.frame, **outcome)
        self.lease = None
        self.cooldown_until = self.frame + 72

    def update(self, raw):
        observation = obj(raw)
        previous = self.previous
        old_frame, old_x = self.frame, self.x
        self.frame = int(num(observation.get('frame'), old_frame+4))
        player = obj(observation.get('player'))
        self.x = clip(num(player.get('x'), self.x), 0, 159)
        self.y = clip(num(player.get('y'), self.y), 150, 209)
        lives = int(num(observation.get('lives'), self.lives))
        wave = int(num(observation.get('wave'), self.wave))
        changed = previous is not None and (lives != self.lives or wave != self.wave)
        score = num(observation.get('score'), self.score)
        if score > self.score:
            self.last_gain = self.frame
        self.score = score
        if changed:
            self.stats['life_losses'] += max(0, self.lives-lives)
            self.log('epoch_changed', frame=self.frame, previous_x=old_x,
                     previous_lives=self.lives, lives=lives, wave=wave,
                     visible_bombs=len(rows(observation.get('bombs'))))
            self.end_lease('life_or_wave_transition')
            self.epoch += 1
            self.lanes.clear()
            self.shot_lanes.clear()
            self.masks = None
        self.lives, self.wave = lives, wave
        dt = max(1, self.frame-old_frame)
        dx = self.x-old_x
        if previous and not changed and 0.1 <= dx*self.last_direction/dt <= 1.2:
            self.speed = clip(0.8*self.speed+0.2*abs(dx)/dt, 0.25, 1.2)
        old_shot = self.shot
        self.shot = obj(observation.get('player_shot'))
        if flag(self.shot.get('active')) and (not flag(old_shot.get('active'))
                or num(self.shot.get('y')) > num(old_shot.get('y')) + 15):
            self.shot_lanes.append((num(self.shot.get('x'), self.x+1), self.frame))
            if self.lease and self.frame > self.lease['start']:
                self.end_lease('new_beam_observed')
        # A laser may hit a shield between observations: include its possible lane.
        if previous and not flag(old_shot.get('active')):
            self.shot_lanes.append((old_x+1, self.frame))
        current_mask = self.mask(observation)
        if self.masks is not None and not changed:
            lost = self.masks.keys()-current_mask.keys()
            if len(lost) > 24:
                self.lanes.clear()
            elif lost:
                unexplained = []
                for key in lost:
                    sh, r, bit = key
                    top = min((k[1] for k in self.masks if k[0] == sh and k[2] == bit), default=9)
                    x = self.masks[key]
                    own = any(abs(x-sx) <= 4.5 and self.frame-sf <= 48
                              for sx, sf in self.shot_lanes)
                    if r <= top+2 and not own:
                        cover = sum(abs(v-x) < 0.1 for v in current_mask.values())
                        if cover <= 2:
                            unexplained.append(x)
                if unexplained:
                    lane = sum(unexplained)/len(unexplained)
                    self.lanes.append({'x': lane, 'frame': self.frame, 'used': False})
                    self.log('weak_exposed_lane', frame=self.frame, x=lane)
        self.masks = current_mask
        self.lanes = [e for e in self.lanes if self.frame-e['frame'] <= 180][-32:]
        aliens = [a for a in rows(observation.get('invaders')) if flag(a.get('alive'))]
        old_columns = self.columns
        self.columns = {}
        for a in aliens:
            c = int(num(a.get('col'), -1))
            if 0 <= c <= 5 and (c not in self.columns or num(a.get('y')) > num(self.columns[c].get('y'))):
                self.columns[c] = a
        self.alien_count = len(aliens)
        deltas = [num(a.get('x'))-num(old_columns[c].get('x'))
                  for c, a in self.columns.items() if c in old_columns]
        if deltas and not changed:
            deltas.sort()
            velocity = deltas[len(deltas)//2]/dt
            if abs(velocity) <= 1:
                self.formation_speed = 0.75*self.formation_speed+0.25*velocity
        self.previous = observation
        if self.lease:
            if self.frame >= self.lease['end'] or self.lease['col'] not in self.columns:
                self.end_lease('expired_or_target_gone')

    def collision_cost(self, direction):
        # Geometry is conservative and only uses explicitly observed bombs.
        cost = 0.0
        for bomb in rows(obj(self.previous).get('bombs')):
            bx, by = num(bomb.get('x'), -999), num(bomb.get('y'), -999)
            for delay in (0, 4):
                for dt in range(0, 41, 4):
                    y = by+1.2*dt
                    if self.y-12 <= y <= self.y+7:
                        x = clip(self.x+self.speed*(self.last_direction*min(delay, dt)
                                 + direction*max(0, dt-delay)), 16, 143)
                        cost = max(cost, max(0.0, 13-abs(x-bx)))
        return cost

    def fallback(self):
        if self.x >= 116:
            self.direction = -1
        elif self.x <= 38:
            self.direction = 1
        direction = self.direction
        costs = {d: self.collision_cost(d) for d in (-1, 0, 1)}
        reason = 'sweep'
        if costs[direction] > 0:
            direction = min(costs, key=lambda d: (costs[d], d == 0, d != self.direction))
            reason = 'visible_bomb'
        elif self.frame-self.last_escape >= 48:
            lanes = [e for e in self.lanes if not e['used'] and abs(e['x']-self.x) <= 28]
            if lanes:
                e = min(lanes, key=lambda q: abs(q['x']-self.x))
                escape = -1 if e['x'] > self.x else 1
                if 38 < self.x < 116 and costs[escape] == 0:
                    direction = escape
                    e['used'] = True
                    self.last_escape = self.frame
                    self.stats['shield_escapes'] += 1
                    reason = 'weak_shield_escape'
        if direction:
            self.direction = direction
        return direction, reason, costs

    def eligible(self):
        sparse = self.alien_count <= 18 and self.alien_count > 0
        stalled = self.frame-self.last_gain >= 180
        if self.policy == 'SWEEP_ONLY':
            return False
        return sparse or (self.policy != 'SPARSE' and stalled)

    def target_x(self, col):
        a = self.columns[col]
        # Modest motion lead; this is not an asserted ROM projectile model.
        lead = clip(self.formation_speed*16, -5, 5)
        return clip(num(a.get('x'))+lead-1, 39, 115)

    def candidate_columns(self, costs):
        result = {}
        for col, alien in self.columns.items():
            tx = self.target_x(col)
            distance = abs(tx-self.x)
            d = -1 if tx < self.x-3 else 1 if tx > self.x+3 else 0
            if distance > 20 or costs[d] > 0:
                continue
            if any(abs(e['x']-tx) <= 12 and not e['used'] for e in self.lanes):
                continue
            result[col] = {'alignment': 'ALIGNED' if distance <= 5 else 'SHORT_APPROACH',
                           'side': 'LEFT' if d < 0 else 'RIGHT' if d > 0 else 'HERE',
                           'front': num(alien.get('y')) >= 130,
                           'outer_column': col in (min(self.columns), max(self.columns)),
                           'target_present': True, 'visible_collision': False}
        return result

    def submit(self, costs):
        if self.query_future is not None or self.ctx.done or not self.eligible():
            return
        if self.lease or self.frame < max(self.cooldown_until, self.next_query):
            return
        if self.frame-self.last_query < 20:
            return
        if self.ctx.budget.remaining().get('system1_calls', 0) <= 0:
            return
        candidates = self.candidate_columns(costs)
        if not candidates:
            return
        criteria = {f'COL{c}': 'Briefly approach this listed living column while firing; never park.'
                    for c in sorted(candidates)}
        criteria['SWEEP'] = 'Continue broad moving fire; prefer this when a useful short intercept is not supported.'
        criteria['UNKNOWN'] = 'The evidence is insufficient for an intercept.'
        instruction = ('Select a short firing opportunity using candidates and plan. Prefer an ALIGNED living '
                       'column, then a SHORT_APPROACH front threat or outer column. Only listed columns exist. '
                       'Choose SWEEP if the shot is not nearly available or there is no useful approach. '
                       'A selected target lasts at most four ticks and ends when a new beam is observed. '
                       'Empty bomb observations never prove safety. Do not calculate coordinates.')
        questions = {
            'target_a': Choice(instruction, criteria),
            'target_b': Choice(instruction, dict(reversed(list(criteria.items())))),
            'opportunity': Noul('Does this state support trying ONE bounded column intercept rather than '
                                'continuing the sweep? Require plan.intercept_eligible and shot_window '
                                'OPEN or NEAR_OPEN, plus a listed candidate. Never infer safety from bomb silence.')
        }
        active = flag(self.shot.get('active'))
        shot_window = 'OPEN' if not active else 'NEAR_OPEN' if num(self.shot.get('y'), 190) < 65 else 'BUSY'
        if shot_window == 'BUSY':
            return
        state = {'schema': 'bounded_column_intercept_v1',
                 'plan': {'policy': self.policy, 'intercept_eligible': True,
                          'rule': 'Keep FIRE pressed. One brief approach, then resume sweep.'},
                 'shot_window': shot_window,
                 'formation': 'SPARSE' if self.alien_count <= 18 else 'SCORE_STALLED',
                 'candidates': {f'COL{c}': facts for c, facts in candidates.items()},
                 'danger_evidence': 'NO_COLLISION_PREDICTED_NOT_PROVEN_SAFE',
                 'last_intercept': self.history[-1] if self.history else None}
        tag = {'frame': self.frame, 'epoch': self.epoch, 'policy': self.policy_version,
               'columns': list(candidates)}
        self.last_query = self.frame
        self.query_tag = tag
        def call():
            try:
                return self.ctx.decide(state, questions, timeout_s=0.85)
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None
        self.query_future = self.ctx.spawn(call)

    def poll(self):
        if self.query_future is not None and self.query_future.done():
            future, tag = self.query_future, self.query_tag
            self.query_future = None
            self.query_tag = None
            try:
                answers = future.result()
            except Exception:
                answers = None
            if not isinstance(answers, dict):
                self.failures += 1
                self.next_query = self.frame+min(240, 20*self.failures)
            elif (tag['epoch'] == self.epoch and tag['policy'] == self.policy_version
                  and 0 <= self.frame-tag['frame'] <= 24 and not self.lease
                  and self.frame >= self.cooldown_until and self.eligible()):
                a, b = answers.get('target_a'), answers.get('target_b')
                key = field(a, 'value')
                valid_keys = {f'COL{c}': c for c in tag['columns']}
                if (isinstance(key, str) and key in valid_keys and trusted(a, key) and trusted(b, key)
                        and num(field(answers.get('opportunity'), 'value'), 0) >= 0.65):
                    col = valid_keys[key]
                    if col in self.columns and abs(self.target_x(col)-self.x) <= 20:
                        self.lease = {'col': col, 'start': self.frame, 'end': self.frame+16,
                                      'score': self.score, 'changed': 0}
                        self.stats['intercepts'] += 1
                        self.log('intercept_accepted', frame=self.frame, column=col,
                                 answer_age=self.frame-tag['frame'], duration=16)
        if self.astra_future is not None and self.astra_future.done():
            future = self.astra_future
            self.astra_future = None
            try:
                text = getattr(future.result(), 'text', '')
                data = json.loads(text[text.find('{'):text.rfind('}')+1]) if isinstance(text, str) else {}
                policy = obj(data).get('policy')
                if isinstance(policy, str) and policy in ('SPARSE', 'SPARSE_OR_STALLED', 'SWEEP_ONLY'):
                    old = self.policy
                    self.policy = policy
                    self.policy_version += 1
                    self.end_lease('astra_policy_revision')
                    self.stats['astra_changes'] += int(old != policy)
                    self.log('astra_policy_applied', frame=self.frame, old=old, new=policy)
            except Exception:
                self.log('astra_ignored', reason='invalid_or_unavailable')

    def maybe_astra(self):
        if self.astra_requested or self.ctx.done:
            return
        # Start early enough for the measured 30--50 second latency; never wait.
        if self.frame < 240 or self.ctx.budget.remaining().get('system2_calls', 0) <= 0:
            return
        if self.ctx.budget.remaining().get('wall_s', 0) < 60:
            return
        evidence = {'score': self.score, 'lives': self.lives, 'remaining_aliens': self.alien_count,
                    'scoreless_frames': self.frame-self.last_gain, 'outcomes': list(self.history)}
        prompt = (
            'Choose a durable policy for a real-time Space Invaders harness, not an immediate action. '
            'Its proven fallback sweeps laterally with FIRE held every tick. The experimental adjunct '
            'asks Jev to recognize a nearby living column, then code permits at most 16 emulator frames '
            'of aiming with FIRE still held; a new beam, danger, death, or deadline ends it. '
            'Jev sees categorical alignment/front/edge facts, never computes geometry. Its two reordered '
            'Choices must agree with probability >=0.60 and a separate Noul >=0.65; a 24-frame-old '
            'answer is discarded. The default enables this only with <=18 remaining aliens. '
            'Empty bombs do not prove safety, shield erosion is weak evidence, and decoder kill counts '
            'can lag score. Choose SPARSE to preserve this cautious default, SPARSE_OR_STALLED to also '
            'permit an intercept after 180 scoreless frames, or SWEEP_ONLY if the evidence argues '
            'against targeting. Output ONLY one short JSON object, for example {"policy":"SPARSE"}. '
            'This changes the eligibility rule supplied to Jev, not its questions. Public evidence: '
            + json.dumps(evidence, allow_nan=False, separators=(',', ':'))
        )
        self.astra_requested = True
        self.astra_future = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=55.0))
        self.log('astra_policy_requested', frame=self.frame)

    def action(self):
        baseline, reason, costs = self.fallback()
        selected = baseline
        if self.lease:
            c = self.lease['col']
            if reason != 'sweep' or c not in self.columns or not self.eligible():
                self.end_lease('current_safety_or_eligibility_veto')
            else:
                tx = self.target_x(c)
                proposed = -1 if tx < self.x-3 else 1 if tx > self.x+3 else 0
                clear_lane = not any(abs(e['x']-tx) <= 12 and not e['used'] for e in self.lanes)
                if costs[proposed] > 0 or not clear_lane or not 38 < self.x < 116:
                    self.end_lease('latest_geometry_veto')
                else:
                    selected = proposed
                    reason = 'jev_column_intercept'
                    if selected != baseline:
                        self.lease['changed'] += 1
                        self.stats['changed_actions'] += 1
                        self.log('jev_changed_action', frame=self.frame, column=c,
                                 fallback=baseline, selected=selected)
        self.last_direction = selected
        self.stats['actions'] += 1
        return ('left_fire' if selected < 0 else 'right_fire' if selected > 0 else 'fire'), costs

    def close(self):
        self.end_lease('episode_end')
        for future in (self.query_future, self.astra_future):
            if future is not None and not future.done():
                future.cancel()
        self.log('crossover_finished', frame=self.frame, score=self.score, **self.stats)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        control = Controller(ctx)
        legal = set(getattr(game.action_spec, 'names', ACTIONS))
        last_frame = None
        try:
            while not ctx.done and not game.done:
                obs = obj(ctx.observe())
                if flag(obs.get('game_over')):
                    break
                frame = int(num(obs.get('frame'), 0 if last_frame is None else last_frame+4))
                if frame == last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame = frame
                control.update(obs)
                control.poll()
                if ctx.done:
                    break
                action, costs = control.action()
                if action not in legal:
                    action = 'fire' if 'fire' in legal else game.action_spec.noop
                ctx.act(action)
                if not ctx.done:
                    control.submit(costs)
                    control.maybe_astra()
                    ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            pass
        finally:
            control.close()


def build(ctx):
    return Harness(ctx)
