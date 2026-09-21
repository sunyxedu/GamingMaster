"""Short offline checks with stand-in models; no backend or game process is used."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from concurrent.futures import Future


# Match the documented local interface without inspecting implementation source.
class Question:
    def __init__(self, instructions, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


jev = types.ModuleType('metaharness.models.jev')
jev.Choice = jev.Noul = Question
jev.usage_guide = lambda: 'Stand-in usage guide.'
sys.modules['metaharness'] = types.ModuleType('metaharness')
sys.modules['metaharness.models'] = types.ModuleType('metaharness.models')
sys.modules['metaharness.models.jev'] = jev
module_spec = importlib.util.spec_from_file_location('candidate', Path(__file__).with_name('harness.py'))
h = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(h)


POLICY = {
    'goal': 'Transfer to a ready station', 'semantics': 'Synthetic contract fixture',
    'busy': ['ne', ['get', 'o.job'], None], 'feedback': ['get', 'o.result'],
    'failed': ['eq', ['get', 'o.result.status', None], 'failed'],
    'remaining': ['get', 'o.left'], 'stop': 'abort', 'pulse': True,
    'rules': [{'id': 'move', 'priority': 10, 'each': ['get', 'o.sites'],
               'where': ['eq', ['get', 'c.ready'], True],
               'action': ['transfer', ['get', 'c.code'], 2],
               'why': 'A ready transfer site', 'cooldown': 3}]
}


class Answer:
    def __init__(self, value, confidence=0.9):
        self.value = value
        self.probabilities = {value: 0.95}
        self.confidence = confidence


class FakeContext:
    def __init__(self, text=None, error=None, expire=False, duration=12, first=False):
        self.text = json.dumps(POLICY) if text is None else text
        self.error, self.expire, self.first = error, expire, first
        self.time = 0.0
        self.duration = duration
        self.clock = types.SimpleNamespace(now=lambda: self.time)
        self.memory = {}
        self.system1 = object()
        self.budget = types.SimpleNamespace(remaining=lambda: {
            'system2_calls': 40-self.thinks, 'system1_calls': 4000, 'wall_s': 100-self.time})
        self.thinks = self.decides = 0
        self.posted, self.events, self.futures = [], [], []
        self.pending = 'idle'
        self.last_nonidle = None
        self.observe_calls = 0

    @property
    def done(self):
        return self.time >= self.duration

    def observe(self):
        self.observe_calls += 1
        sites = [] if self.expire and self.decides else [{'code': 'dock:7', 'ready': True}]
        return {'job': None, 'result': None, 'sites': sites, 'left': self.duration-self.time}

    def act(self, action):
        self.posted.append((self.time, action))
        self.pending = action

    def wait_for_tick(self):
        self.time += 0.5

    def spawn(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)
        self.futures.append(future)
        return future

    def think(self, prompt, *, session_id=None, timeout_s=300):
        self.thinks += 1
        return types.SimpleNamespace(text=self.text, session_id='unused')

    def decide(self, state, questions, *, timeout_s=5):
        self.decides += 1
        if self.error:
            raise self.error
        chosen = next(iter(questions['option'].criteria)) if self.first else next(
            (k for k in questions['option'].criteria if k.startswith('option_')), 'unknown')
        return {'option': Answer(chosen), 'danger': Answer('clear'), 'revise': Answer(0.0)}

    def log(self, **kwargs):
        self.events.append(kwargs)


GAME = types.SimpleNamespace(
    action_spec=types.SimpleNamespace(names=['idle', 'transfer', 'abort'],
        descriptions={'idle': 'Stay still', 'transfer': 'transfer TARGET COUNT: transfer cargo',
                      'abort': 'Interrupt job'}, noop='idle', notes='Synthetic bounded actions'),
    describe=lambda: 'Synthetic fixture: transfer TARGET COUNT to an observed ready site.',
    tick_s=0.5)


class ContractTests(unittest.TestCase):
    def run_context(self, **kw):
        ctx = FakeContext(**kw)
        h.build(ctx).run_episode(GAME)
        self.assertTrue(ctx.done)
        self.assertTrue(all(f.done() for f in ctx.futures))
        return ctx

    def test_garbage_planners_and_first_option_models(self):
        for text in ['arbitrary prose', '', 'null', '[]', '{"rules":[]}',
                     '{"rules":[null]}', '{"rules":[{"action":[]}],"stop":23}',
                     '{"rules":[{"action":["transfer"]}],"busy":["bad"]}']:
            with self.subTest(text=text):
                ctx = self.run_context(text=text, first=True)
                self.assertTrue(all(a == 'idle' for _, a in ctx.posted))

    def test_complete_arguments_and_neutral_pulse(self):
        ctx = self.run_context()
        actions = [a for _, a in ctx.posted]
        self.assertIn('transfer dock:7 2', actions)
        for i, a in enumerate(actions[:-1]):
            if a == 'transfer dock:7 2':
                self.assertEqual(actions[i+1], 'idle')
        self.assertNotIn('transfer', actions)

    def test_missing_target_rejects_inflight_choice(self):
        ctx = self.run_context(expire=True)
        self.assertTrue(all(a == 'idle' for _, a in ctx.posted))
        self.assertTrue(any(e.get('event') == 'jev_control' and not e['legal_now'] for e in ctx.events))

    def test_timeouts_do_not_end_episode(self):
        ctx = self.run_context(error=TimeoutError('transient'))
        self.assertGreater(ctx.decides, 1)
        self.assertTrue(any(e.get('timeout') is True for e in ctx.events))

    def test_model_unavailable_propagates(self):
        class ModelUnavailable(RuntimeError):
            pass
        ctx = FakeContext(error=ModelUnavailable('offline'))
        with self.assertRaises(ModelUnavailable):
            h.build(ctx).run_episode(GAME)
        self.assertTrue(all(f.done() for f in ctx.futures))

    def test_model_unavailable_as_timeout_propagates(self):
        class ModelUnavailable(TimeoutError):
            pass
        ctx = FakeContext(error=ModelUnavailable('offline'))
        with self.assertRaises(ModelUnavailable):
            h.build(ctx).run_episode(GAME)

    def test_interpreter_exact_arithmetic(self):
        env = {'o': {'stock': {'ore_a': 4, 'ore_b': 7, 'water': 8}}}
        expr = ['sum', ['map', ['filter', ['items', ['get', 'o.stock']],
                               ['startswith', ['get', 'v.key'], 'ore_']], ['get', 'v.value']]]
        self.assertEqual(h.expression(expr, env), 11)
        self.assertEqual(h.expression(['ceil', ['div', 7, 4]], env), 2)
        with self.assertRaises(h.Invalid):
            h.expression(['get', 'o.missing'], env)
        with self.assertRaises(h.Invalid):
            h.expression(['exec', 'anything'], env)

    def test_compaction_keeps_late_fields(self):
        value = {'huge': list(range(1000)), 'late': {'inventory': {'x': 7}}}
        summary = h.compact(value, 32)
        self.assertEqual(summary['late'], value['late'])
        self.assertEqual(summary['huge']['sample_of'], 1000)
        self.assertEqual(json.loads(h.dump(summary)), summary)

    def test_budget_exhaustion_survives(self):
        ctx = FakeContext()
        ctx.budget = types.SimpleNamespace(remaining=lambda: {'system2_calls': 0, 'system1_calls': 0, 'wall_s': 20})
        h.build(ctx).run_episode(GAME)
        self.assertTrue(ctx.done)
        self.assertEqual(ctx.thinks, 0)

    def test_low_probability_rejected(self):
        ctx = FakeContext()
        original = ctx.decide
        def decide(*a, **kw):
            result = original(*a, **kw)
            result['option'].probabilities = {result['option'].value: 0.1}
            result['option'].confidence = None
            return result
        ctx.decide = decide
        h.build(ctx).run_episode(GAME)
        self.assertTrue(all(a == 'idle' for _, a in ctx.posted))

    def test_busy_job_is_not_replaced(self):
        ctx = FakeContext()
        original = ctx.observe
        def observe():
            obs = original()
            obs['job'] = {'running': True} if ctx.time < 3 else None
            return obs
        ctx.observe = observe
        h.build(ctx).run_episode(GAME)
        self.assertTrue(all(a == 'idle' for t, a in ctx.posted if t < 3))
        self.assertTrue(any(a == 'transfer dock:7 2' for t, a in ctx.posted if t >= 3))

    def test_late_decisions_discarded_and_all_futures_joined(self):
        ctx = FakeContext(duration=6)
        outstanding = []
        class Deferred:
            def __init__(self, result, ready):
                self.value, self.ready, self.joined = result, ready, False
            def done(self):
                return ctx.time >= self.ready
            def result(self):
                self.joined = True
                ctx.time = max(ctx.time, self.ready)
                return self.value
        def spawn(fn):
            before = ctx.thinks
            result = fn()
            delay = 0.5 if ctx.thinks > before else 3.0
            future = Deferred(result, ctx.time+delay)
            outstanding.append(future)
            return future
        ctx.spawn = spawn
        h.build(ctx).run_episode(GAME)
        self.assertTrue(all(f.joined for f in outstanding))
        self.assertTrue(all(a == 'idle' for _, a in ctx.posted))
        self.assertTrue(any(e.get('event') == 'jev_discarded' and e['age_s'] >= 3 for e in ctx.events))

    def test_explicit_failure_is_cooled_down(self):
        ctx = FakeContext(duration=8)
        original = ctx.observe
        def observe():
            obs = original()
            if any(a == 'transfer dock:7 2' for _, a in ctx.posted):
                obs['result'] = {'status': 'failed', 'id': 1}
            return obs
        ctx.observe = observe
        h.build(ctx).run_episode(GAME)
        self.assertEqual(sum(a == 'transfer dock:7 2' for _, a in ctx.posted), 1)
        self.assertTrue(any(e.get('event') == 'option_failed' for e in ctx.events))


if __name__ == '__main__':
    unittest.main()
