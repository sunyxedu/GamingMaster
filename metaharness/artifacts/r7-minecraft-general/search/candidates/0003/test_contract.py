"""Short offline contract tests; no game, network, or real model access."""
import copy
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


class Question:
    def __init__(self, instructions, criteria=None):
        self.instructions, self.criteria = instructions, criteria


for name in ('metaharness', 'metaharness.models', 'metaharness.models.jev'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['metaharness.models.jev'].Choice = Question
sys.modules['metaharness.models.jev'].Noul = Question
spec = importlib.util.spec_from_file_location('candidate', Path(__file__).with_name('harness.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

MENU = {'goal': 'Acquire one cell, then charge', 'meaning': 'quantity is verified',
        'mode': 'bounded', 'busy_path': ['job'], 'remaining_path': ['left'],
        'feedback_path': ['result'], 'stop': 'abort', 'options': [
            {'command': 'collect obj:7 2', 'ready': 'Object exists, idle, quantity below 2',
             'refs': ['obj:7'], 'guards': [[['quantity'], 'lt', 2]]},
            {'command': 'charge cell 1', 'ready': 'Idle, quantity at least 2',
             'refs': [], 'guards': [[['quantity'], 'ge', 2]]}]}


class ModelUnavailable(RuntimeError):
    pass


class UnavailableTimeout(TimeoutError):
    pass
# Check the mandated exception by MRO name even for timeout inheritance.
UnavailableTimeout.__name__ = 'ModelUnavailable'


class Future:
    def __init__(self, ctx, fn):
        self.ctx, self.fn = ctx, fn
        self.at = ctx.t + ctx.delay
        self.finished = False
        self.error = None
        self.value = None
        self.joined = False

    def done(self):
        return self.ctx.t >= self.at

    def result(self):
        self.joined = True
        if not self.finished:
            self.ctx.t = max(self.ctx.t, self.at)
            self.finished = True
            try:
                self.value = self.fn()
            except Exception as exc:
                self.error = exc
        if self.error:
            raise self.error
        return self.value


class Context:
    system1 = object()

    def __init__(self, output=None, duration=9, error=None, first=False, delay=.5):
        self.output = json.dumps(MENU) if output is None else output
        self.duration, self.error, self.first, self.delay = duration, error, first, delay
        self.t = 0.
        self.clock = types.SimpleNamespace(now=lambda: self.t)
        self.budget = types.SimpleNamespace(remaining=self.remaining)
        self.memory = {}
        self.calls = self.decisions = 0
        self.futures = []
        self.actions = []
        self.events = []
        self.observation = {'job': None, 'left': duration, 'quantity': 0,
                            'result': None, 'objects': ['obj:7']}
        self.task_end = 0
        self.change = lambda: None
        self.exhausted = False

    @property
    def done(self):
        return self.t >= self.duration

    def remaining(self):
        return {'system2_calls': 0 if self.exhausted else 40-self.calls,
                'system1_calls': 4000-self.decisions, 'wall_s': max(0, 30-self.t)}

    def observe(self):
        self.change()
        self.observation['left'] = max(0, self.duration-self.t)
        return copy.deepcopy(self.observation)

    def act(self, action):
        self.actions.append((self.t, action))
        if action.startswith('collect'):
            assert self.observation['job'] is None, 'replaced busy job'
            self.observation['job'] = action
            self.task_end = self.t+1.5
        elif action.startswith('charge'):
            assert self.observation['job'] is None, 'replaced busy job'
            assert self.observation['quantity'] >= 2, 'missing resources'
        return None

    def wait_for_tick(self):
        self.t += .5
        if self.observation['job'] and self.t >= self.task_end:
            self.observation['job'] = None
            self.observation['quantity'] = 2
            self.observation['result'] = {'success': True}

    def think(self, prompt, **kw):
        self.calls += 1
        if self.error:
            raise self.error('stand-in')
        return types.SimpleNamespace(text=self.output, session_id='x')

    def decide(self, state, questions, **kw):
        self.decisions += 1
        answers = {}
        for key, q in questions.items():
            if key == 'revise':
                answers[key] = types.SimpleNamespace(value=0, confidence=None,
                                                     probabilities={'true': 0, 'false': 1})
                continue
            if self.first:
                value = next(iter(q.criteria))
            elif key == 'choose':
                value = next((o['key'] for o in state['options']), 'hold')
            elif key.startswith('ready_'):
                value = 'ready'
            else:
                value = 'clear'
            answers[key] = types.SimpleNamespace(value=value, confidence=None,
                                                 probabilities={value: .95})
        return answers

    def spawn(self, fn):
        future = Future(self, fn)
        self.futures.append(future)
        return future

    def log(self, **fields):
        self.events.append(fields)


class Game:
    tick_s = .5
    action_spec = types.SimpleNamespace(
        names=['idle', 'abort', 'collect', 'charge'], noop='idle', notes='Bounded jobs.',
        descriptions={'idle': 'Continue job.', 'abort': 'Interrupt job.',
                      'collect': 'collect OBJECT [COUNT=1]: acquire.',
                      'charge': 'charge ITEM COUNT: use cell.'})

    def describe(self):
        return 'Synthetic machine game. Idle continues a job; abort interrupts.'


class ContractTests(unittest.TestCase):
    def run_ctx(self, ctx):
        h.build(ctx).run_episode(Game())
        self.assertTrue(all(f.joined for f in ctx.futures))
        return ctx

    def test_garbage_and_first_options(self):
        for garbage in ('not JSON', '{}', '{"options":[null]}',
                        '{"options":[{"command":[]}]}', '{"options":[]}'):
            c = self.run_ctx(Context(output=garbage, first=True))
            self.assertTrue(c.done)
            self.assertTrue(all(a == 'idle' for _, a in c.actions))

    def test_complete_commands_busy_and_pulses(self):
        c = self.run_ctx(Context())
        actions = [a for _, a in c.actions]
        self.assertEqual(actions.count('collect obj:7 2'), 1)
        self.assertEqual(actions.count('charge cell 1'), 1)
        self.assertEqual(actions[actions.index('collect obj:7 2')+1], 'idle')
        self.assertGreater(c.decisions, 0)

    def test_first_option_standins(self):
        c = self.run_ctx(Context(first=True))
        self.assertTrue(all(a == 'idle' for _, a in c.actions))

    def test_vanished_target(self):
        c = Context()
        c.change = lambda: c.observation.update(objects=[]) if c.t >= 1 else None
        self.run_ctx(c)
        self.assertFalse(any(a.startswith('collect') for _, a in c.actions))

    def test_exact_guards(self):
        self.assertTrue(h.guards_hold([[['n'], 'ge', 3]], {'n': 3}))
        self.assertFalse(h.guards_hold([[['n'], 'ge', 3]], {'n': 2}))
        self.assertFalse(h.guards_hold([[['n'], 'eq', 0]], {}))
        self.assertTrue(h.guards_hold([[['n'], 'eq', 0, 0]], {}))
        self.assertFalse(h.guards_hold([[['n'], 'ge', 1]], {'n': 'bad'}))

    def test_timeout_recovery(self):
        self.run_ctx(Context(error=TimeoutError))

    def test_unavailable_propagates_and_joins(self):
        for error in (ModelUnavailable, UnavailableTimeout):
            c = Context(error=error)
            with self.assertRaises(error):
                h.build(c).run_episode(Game())
            self.assertTrue(all(f.joined for f in c.futures))

    def test_no_model_budget_still_finishes(self):
        c = Context()
        c.exhausted = True
        self.run_ctx(c)
        self.assertTrue(c.done)
        self.assertEqual(c.calls, 0)

    def test_stale_judgment_discarded(self):
        c = Context(delay=3.5, duration=15)
        self.run_ctx(c)
        self.assertFalse(any(a.startswith('collect') for _, a in c.actions))
        self.assertTrue(any(e['event'] == 'jev_discarded' for e in c.events))

    def test_arity_and_malformed_fields(self):
        names = Game.action_spec.names
        minimum = h.arities(names, Game.action_spec.descriptions)
        with self.assertRaises(h.Invalid):
            h.command('collect', names, minimum)
        for key, value in [('busy_path', 'job'), ('mode', None), ('options', [None])]:
            menu = copy.deepcopy(MENU)
            menu[key] = value
            with self.assertRaises(h.Invalid):
                h.parse_menu(json.dumps(menu), names, minimum)

    def test_guard_rechecked_after_judgment(self):
        c = Context()
        original = c.decide
        def decide(*args, **kwargs):
            result = original(*args, **kwargs)
            c.observation['quantity'] = 5
            return result
        c.decide = decide
        self.run_ctx(c)
        self.assertFalse(any(a.startswith('collect') for _, a in c.actions))

    def test_low_probability_rejected(self):
        c = Context()
        original = c.decide
        def decide(*args, **kwargs):
            result = original(*args, **kwargs)
            for value in result.values():
                value.probabilities = {str(value.value): .1}
            return result
        c.decide = decide
        self.run_ctx(c)
        self.assertTrue(all(a == 'idle' for _, a in c.actions))

    def test_compact_preserves_late_fields(self):
        summary = h.compact({'large': list(range(200)), 'late': {'count': 4}})
        self.assertEqual(summary['late'], {'count': 4})
        self.assertIn('indexed_sample', summary['large'])


if __name__ == '__main__':
    unittest.main()
