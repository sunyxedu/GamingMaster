"""Observation-prefix compiler; models supervise semantics, never key timing.

Only public observations, the documented action grammar, and ctx are used.
An asynchronous Jev certificate enables larger complete chronological prefixes.
Without a certificate, code still services every tick and preserves hold tails.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


def number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def object_observation(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def records(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


class Harness:
    # The observation rounds timers to 0.001 ms. Never release on the wrong
    # side of the true end because of decimal serialization. Shift equal-time
    # heads and tails together so release-before-repress ordering is retained.
    SERIALIZATION_GUARD_MS = 0.002
    PREFIX_MS = 1200.0

    def __init__(self, ctx):
        self.ctx = ctx
        self.certified = False
        self.text_only = False
        self.signature = None
        self.generation = 0
        self.jev = None
        self.astra = None
        self.jev_requests = 0
        self.astra_requested = False
        self.last_error = None
        self.errors = 0
        self.last_feedback_tick = None
        self.last_snapshot = None
        self.plan = 'Compile complete observed event prefixes; preserve every hold tail.'

    def snapshot(self, obs):
        episode = obs.get('episode', {})
        episode = episode if isinstance(episode, dict) else {}
        controls = obs.get('controls', {})
        controls = controls if isinstance(controls, dict) else {}
        chart = obs.get('chart', {})
        chart = chart if isinstance(chart, dict) else {}
        count = number(episode.get('columns'), 0)
        count = int(count) if 1 <= count <= 7 else 0
        now = number(episode.get('time_ms'))
        tick = number(episode.get('tick_s'), 0.04) * 1000.0
        tick = tick if 0 < tick <= 1000 else 40.0
        upcoming = records(obs.get('upcoming'))
        active = records(obs.get('active_holds'))
        schema_ok = now is not None and count > 0
        heads = []
        holds = []
        for note in upcoming:
            lane = number(note.get('column'))
            start = number(note.get('start_ms'))
            end = number(note.get('end_ms'))
            if lane is None or int(lane) != lane or not 0 <= lane < count or start is None:
                schema_ok = False
                continue
            if note.get('kind') == 'hold' and (end is None or end < start):
                schema_ok = False
            heads.append({'id': note.get('id'), 'column': int(lane),
                          'start_ms': start, 'end_ms': end,
                          'kind': note.get('kind'), 'group_id': note.get('group_id'),
                          'group_size': note.get('group_size')})
        for hold in active:
            lane = number(hold.get('column'))
            end = number(hold.get('end_ms'))
            if lane is None or int(lane) != lane or not 0 <= lane < count or end is None:
                schema_ok = False
                continue
            holds.append({'id': hold.get('id'), 'column': int(lane), 'end_ms': end})
        heads.sort(key=lambda x: (x['start_ms'], x['column']))
        windows = episode.get('judgment_windows_ms', {})
        windows = windows if isinstance(windows, dict) else {}
        indexing = controls.get('column_indexing', 'unknown')
        sig = (chart.get('id'), count, episode.get('mod'), indexing,
               bool(schema_ok), bool(self.text_only))
        return {'now': now if now is not None else 0.0, 'tick_ms': tick,
                'tick': episode.get('tick'), 'columns': count, 'heads': heads,
                'holds': holds, 'schema_ok': schema_ok, 'signature': sig,
                'indexing': indexing, 'capped': len(upcoming) >= 64,
                'miss_window': number(windows.get('G'), 200.0),
                'big_p': number(windows.get('Big P'), 40.0),
                'stats': obs.get('stats', {}), 'mod': episode.get('mod')}

    def request_certificate(self, snap):
        ctx = self.ctx
        if self.jev is not None or self.jev_requests >= 3 or ctx.done:
            return
        if getattr(ctx, 'system1', True) is None:
            return
        if ctx.budget.remaining().get('system1_calls', 0) <= 1:
            return
        state = {
            'plan': self.plan,
            'rules': {
                'inputs': 'column is the zero-based DISPLAYED input lane; raw_column is metadata.',
                'times': 'start_ms/end_ms already incorporate BPM and chart offset.',
                'holds': 'Release at or after the tail, never immediately after the head.',
                'batch': 'A sorted batch executes a complete chronological prefix of observed events.',
                'visibility': 'Only the next 2.2 seconds / 64 notes are visible. Never skip unseen heads.',
                'authority': 'Code checks arithmetic, lane bounds, chord completeness and hold ordering.',
            },
            'facts': {
                'schema': 'valid' if snap['schema_ok'] else 'invalid',
                'lane_indexing': snap['indexing'],
                'mod': snap['mod'],
                'tail_support': 'explicit absolute end times',
                'visibility_guard': 'prefix ends inside visible horizon, before a truncated chord',
                'transport': 'rejected' if self.text_only else 'documented JSON batch',
                'recent_error': self.last_error,
            },
            'candidates': {
                'prefix': 'Compile at most 1.2 seconds of ALL known heads and due tails, sorted together.',
                'tick': 'Service the current tick only; retain the exact same lane and hold safety checks.',
                'unknown': 'The semantic contract is not established; continue the executable tick fallback.',
            },
        }
        question = (
            'Using rules and facts, which candidate is supported? Choose prefix only when schema is valid, '
            'lane_indexing is zero_based_displayed, and transport is documented JSON batch. '
            'Choose tick if transport was rejected. Otherwise choose unknown. '
            'This is a semantic compatibility check, not a request to compute note timing.'
        )
        criteria = {
            'prefix': 'All named preconditions for complete chronological prefixes are supported.',
            'tick': 'JSON transport was rejected; current-tick execution is required.',
            'unknown': 'A named precondition is absent, ambiguous, or contradicted.',
        }
        questions = {
            'route': Choice(question, criteria),
            'route_reordered': Choice(question, {k: criteria[k] for k in ('unknown', 'tick', 'prefix')}),
            'lane': Choice(
                'Which observed field is directly usable as the action column under rules.inputs? '
                'Do not mirror a displayed lane a second time.',
                {'column': 'Use the already displayed column directly, including under Flip.',
                 'raw_column': 'Use the original untransformed raw column.',
                 'unknown': 'The input indexing is not established.'}),
        }
        version = self.generation
        signature = snap['signature']
        observed_at = snap['now']
        self.jev_requests += 1
        self.jev = (ctx.spawn(lambda: ctx.decide(state, questions, timeout_s=2.0)),
                    version, signature, observed_at)
        ctx.log('semantic_request', version=version, facts=state['facts'])

    @staticmethod
    def confident(answers, key, option):
        if not isinstance(answers, dict):
            return False
        answer = answers.get(key)
        if getattr(answer, 'value', None) != option:
            return False
        probabilities = getattr(answer, 'probabilities', {})
        if not isinstance(probabilities, dict):
            return False
        probability = number(probabilities.get(option), 0.0)
        confidence = number(getattr(answer, 'confidence', None))
        # Missing confidence is not treated as certainty. The actual winning
        # probability is mandatory; confidence is an additional veto if given.
        return probability >= 0.85 and (confidence is None or confidence >= 0.6)

    def collect(self, snap):
        ctx = self.ctx
        if self.jev is not None and self.jev[0].done():
            future, version, signature, observed_at = self.jev
            self.jev = None
            try:
                answers = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                ctx.log('semantic_failure', fallback='tick')
                answers = {}
            same = version == self.generation and signature == snap['signature']
            safe = snap['schema_ok'] and not self.text_only
            good = all(self.confident(answers, k, v) for k, v in
                       [('route', 'prefix'), ('route_reordered', 'prefix'), ('lane', 'column')])
            if same and safe and good:
                self.certified = True
                ctx.log('semantic_policy_applied', version=version, policy='complete_prefix',
                        observation_age_ms=snap['now'] - observed_at,
                        scope='schema certificate, never a cached note action')
            else:
                ctx.log('semantic_policy_rejected', version=version, stale=not same,
                        fallback='tick', thresholds_met=good)
        if self.astra is not None and self.astra.done():
            future = self.astra
            self.astra = None
            try:
                result = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, ValueError, TypeError):
                return
            token = str(getattr(result, 'text', '')).strip().upper()
            if token == 'TICK':
                self.certified = False
                self.text_only = True
                self.plan = 'Use current-tick textual inputs; JSON transport is not trusted.'
                self.generation += 1
                ctx.log('astra_policy_applied', policy='tick', reason='protocol_diagnosis')
            else:
                ctx.log('astra_policy_rejected', reason='not an actionable safe diagnosis')

    def check_feedback(self, obs, snap):
        feedback = obs.get('action_feedback', {})
        error = feedback.get('error') if isinstance(feedback, dict) else None
        if not error or snap['tick'] == self.last_feedback_tick:
            return
        self.last_feedback_tick = snap['tick']
        self.last_error = str(error)[:300]
        self.errors += 1
        self.certified = False
        self.text_only = True
        self.generation += 1
        self.plan = 'Use current-tick textual inputs after rejected JSON; never sacrifice hold tails.'
        self.ctx.log('protocol_fallback', error=self.last_error, policy='text_tick')
        if self.errors < 2 or self.astra_requested or self.ctx.done:
            return
        if getattr(self.ctx, 'system2', True) is None:
            return
        budget = self.ctx.budget.remaining()
        if budget.get('system2_calls', 0) <= 1 or budget.get('wall_s', 0) < 15:
            return
        self.astra_requested = True
        prompt = (
            'Diagnose an offline Key emulator input-transport failure. Code executes observed '
            'absolute note times and already-transformed displayed columns. Hold tails must '
            'not be released early. The current safe fallback sends textual press/release '
            'actions every tick; never suggest resets, changed chart settings or invented notes. '
            'Should that fallback remain in force? Reply exactly TICK if yes or KEEP if no. '
            'This diagnosis controls a new semantic-plan generation; it never blocks note timing. '
            'Evidence: ' + json.dumps({'error': self.last_error, 'stats': snap['stats']})
        )
        self.astra = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=45.0))

    def compile_action(self, snap):
        now = snap['now']
        tick = snap['tick_ms']
        guard = self.SERIALIZATION_GUARD_MS
        if not snap['schema_ok']:
            return 'noop', {'policy': 'invalid_schema', 'heads': 0, 'tails': 0}
        wide = self.certified and not self.text_only
        # Before semantic certification, only touch heads close to this tick.
        # For the current judge, a half-tick early bound reduces quantization.
        early = min(tick * 0.5, snap['big_p'] * 0.5)
        cutoff = now + (self.PREFIX_MS if wide else early)
        if snap['capped'] and snap['heads']:
            # The last visible group may be cut off by the 64-note limit.
            cutoff = min(cutoff, snap['heads'][-1]['start_ms'] - tick - 0.01)
        events = []
        chosen = []
        for note in snap['heads']:
            start = note['start_ms']
            if start > cutoff or start < now - snap['miss_window'] + guard:
                continue
            at = max(now, start + guard)
            chosen.append(note)
            events.append((at, 1, note['column'], 'press', note['id']))
        release_keys = set()
        # Include only tails INSIDE this complete prefix, not every visible
        # hold's end. A long hold must never jump over still-unseen heads.
        for hold in snap['holds'] + [n for n in chosen if n['kind'] == 'hold']:
            end = hold['end_ms']
            if end is None or end > cutoff:
                continue
            key = (hold['column'], end)
            if key in release_keys:
                continue
            release_keys.add(key)
            at = max(now, end + guard)
            events.append((at, 0, hold['column'], 'release', hold['id']))
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        if self.text_only:
            # Text has no timestamp: never release an actually future tail.
            due_tails = [e for e in events if e[3] == 'release' and e[0] <= now]
            if due_tails:
                return 'release ' + str(due_tails[0][2]), {'policy': 'text_tick', 'tails': 1, 'heads': 0}
            lanes = sorted({n['column'] for n in chosen})
            if lanes:
                return 'press ' + '+'.join(map(str, lanes)), {'policy': 'text_tick', 'heads': len(lanes), 'tails': 0}
            return 'noop', {'policy': 'text_tick', 'heads': 0, 'tails': 0}
        # Equal timestamp events are simultaneous; release old holds first.
        data = [{'type': e[3], 'column': e[2], 'at_ms': e[0]} for e in events]
        summary = {'policy': 'prefix' if wide else 'tick',
                   'heads': sum(e[3] == 'press' for e in events),
                   'tails': sum(e[3] == 'release' for e in events)}
        if data:
            summary['first_ms'] = data[0]['at_ms']
            summary['last_ms'] = data[-1]['at_ms']
            return {'type': 'batch', 'events': data}, summary
        # Advancing only inside an explicitly empty observed prefix is safe.
        # Keep the initial fallback neutral; it remains useful if models fail.
        if wide and cutoff > now + tick and not any(n['start_ms'] <= cutoff for n in snap['heads']):
            return {'type': 'advance', 'to_ms': cutoff}, summary
        return 'noop', summary

    def run_episode(self, game):
        ctx = self.ctx
        if not getattr(game, 'tick_s', None):
            return
        while not ctx.done:
            obs = object_observation(ctx.observe())
            snap = self.snapshot(obs)
            if obs.get('episode', {}).get('done'):
                break
            self.check_feedback(obs, snap)
            if self.signature != snap['signature']:
                self.signature = snap['signature']
                self.generation += 1
                self.certified = False
            self.collect(snap)
            if self.jev is None and not self.certified and self.jev_requests < 3:
                # Startup plus genuine schema/feedback changes only; do not
                # keep asking a model to bless an ambiguous answer every tick.
                if self.last_snapshot is None or self.last_snapshot != self.signature:
                    self.request_certificate(snap)
                    self.last_snapshot = self.signature
            action, summary = self.compile_action(snap)
            if action != 'noop':
                ctx.log('compiled_action', observed_ms=snap['now'], **summary)
            ctx.act(action)
            # No model/network waits on this execution thread. Every consumed
            # action is replaced from a fresh observation on the next tick.
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
