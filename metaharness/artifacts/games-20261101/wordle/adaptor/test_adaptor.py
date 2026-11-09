"""Offline regression checks; run with the evaluator Python and unittest."""
import itertools
import json
import random
import unittest
from collections import Counter

from game import (ALLOWED_GUESSES, ANSWERS, SEEDS, make_game,
                  mastermind_feedback, wordle_feedback)


class AdaptorTests(unittest.TestCase):
    def test_wordle_duplicate_examples(self):
        for answer, guess, expected in [
            ('abbey', 'bobby', ['yellow','gray','green','gray','green']),
            ('apple', 'allee', ['green','yellow','gray','gray','green']),
            ('cigar', 'array', ['gray','yellow','gray','green','gray']),
            ('level', 'hello', ['gray','green','yellow','yellow','gray']),
        ]:
            self.assertEqual(wordle_feedback(answer, guess), expected)

    def test_wordle_feedback_exhaustive_small_alphabet(self):
        words = [''.join(x) for x in itertools.product('ab', repeat=5)]
        for answer in words:
            for guess in words:
                # Independent oracle removes matching positions from a mutable pool.
                pool = list(answer)
                expected = ['gray'] * 5
                for i in range(5):
                    if answer[i] == guess[i]:
                        pool[i] = None
                        expected[i] = 'green'
                for i in range(5):
                    if expected[i] != 'green' and guess[i] in pool:
                        pool[pool.index(guess[i])] = None
                        expected[i] = 'yellow'
                self.assertEqual(wordle_feedback(answer, guess), expected)

    def test_mastermind_classic_space(self):
        probes = [(0,0,0,0), (0,0,1,1), (0,1,2,3), (5,4,3,2), (1,0,1,0)]
        for code in itertools.product(range(6), repeat=4):
            for guess in probes + [code]:
                exact = sum(a == b for a, b in zip(code, guess))
                total = sum((Counter(code) & Counter(guess)).values())
                self.assertEqual(mastermind_feedback(code, guess),
                                 {'exact': exact, 'misplaced': total - exact})

    def test_win_loss_reset_and_terminal(self):
        for mode in ['wordle', 'mastermind']:
            g = make_game(mode=mode)
            first = g.reset(1001)
            # White-box target access here only to test all terminal scoring paths.
            target = g._target
            bad = next(w for w in ANSWERS if w != target) if mode == 'wordle' else [
                (v + 1) % 6 for v in target]
            for turn in range(1, g.max_guesses + 1):
                obs = g.step({'guess': bad})
                self.assertEqual(obs['turn'], turn)
                self.assertEqual(g.score, 0.0)
            self.assertTrue(g.done)
            self.assertFalse(g.info()['solved'])
            self.assertIn('answer', g.info())
            self.assertEqual(g.step('noop'), obs)
            self.assertEqual(g.reset(1001), first)
            for _ in range(g.max_guesses - 1):
                g.step({'guess': bad})
            obs = g.step({'guess': target})
            self.assertTrue(obs['solved'])
            self.assertAlmostEqual(g.score, 1 / g.max_guesses)
            g.reset(1001)
            g.step({'guess': target})
            self.assertEqual(g.score, 1.0)
            self.assertEqual(g.info()['guesses_used'], 1)
            g.close(); g.close()

    def test_bad_actions_and_noop(self):
        bad = [None, True, 3, float('nan'), {}, {'guess': None}, {'x': 1},
               {'guess': 'crane', 'x': 1}, [], [True]*4, 'guess', '²²²²',
               '１２３４', 'x'*500000, {'guess': [0,1,2,float('inf')]}]
        for mode in ['wordle', 'mastermind']:
            g = make_game(mode=mode)
            for action in bad:
                obs = g.step(action)
                self.assertEqual(obs['turn'], 0)
                self.assertTrue(obs['error'])
                self.assertLess(len(json.dumps(obs, allow_nan=False)), 200000)
            self.assertIsNone(g.step('noop')['error'])

    def test_action_forms_and_observation_isolation(self):
        for mode, actions in [('wordle', ['guess CRANE', ' crane ', {'guess':'crane'}]),
                              ('mastermind', ['guess 0123', '0123', {'guess':[0,1,2,3]}])]:
            results = []
            for action in actions:
                g = make_game(mode=mode); g.reset(1001)
                results.append(g.step(action))
            self.assertEqual(results[0], results[1]); self.assertEqual(results[1], results[2])
            obs = g.observe()
            obs['history'][0]['feedback'] = None
            obs['history'][0]['guess'] = None
            obs['rules']['max_guesses'] = -1
            self.assertNotEqual(obs, g.observe())
            self.assertNotIn('answer', g.observe())
            self.assertNotIn('seed', g.observe())
            self.assertNotIn('answer', g.info())

    def test_seeds_reproduce_and_balance(self):
        all_seeds = [s for seeds in SEEDS.values() for s in seeds]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        for group, seeds in SEEDS.items():
            counts = Counter()
            for seed in seeds:
                w = make_game(); m = make_game(mode='mastermind')
                w.reset(seed); m.reset(seed)
                counts[(len(set(w._target)) < 5, len(set(m._target)))] += 1
                for game in [w,m]:
                    fresh = make_game(mode=game.mode)
                    self.assertEqual(game.observe(), fresh.reset(seed))
                    self.assertEqual(game._target, fresh._target)
            if group != 'dev':
                self.assertEqual(set(counts.values()), {2})
                self.assertEqual(len(counts), 6)

    def test_variants_and_config(self):
        g = make_game(mode='mastermind', positions=5, colors=8, repeats=False, max_guesses=12)
        self.assertEqual(len(set(g._target)), 5)
        self.assertEqual(g.step('guess 00123')['turn'], 0)
        self.assertEqual(g.step('guess 01234')['turn'], 1)
        for config in [{'max_guesses':7}, {'mode':'other'}, {'positions':True},
                       {'mode':'mastermind','repeats':False,'positions':8,'colors':6},
                       {'mode':'mastermind','colors':11}, {'unknown':1}]:
            with self.assertRaises((ValueError, TypeError)):
                make_game(**config)

    def test_random_episodes(self):
        rng = random.Random(8123)
        legal = sorted(ALLOWED_GUESSES)
        for mode in ['wordle','mastermind']:
            for seed in SEEDS['dev']:
                g = make_game(mode=mode); g.reset(seed)
                while not g.done:
                    guess = rng.choice(legal) if mode == 'wordle' else [rng.randrange(6) for _ in range(4)]
                    obs = g.step({'guess':guess})
                    self.assertLess(len(json.dumps(obs, allow_nan=False)), 200000)
                self.assertEqual(g.info()['solved'], obs['solved'])
                self.assertEqual(g.info()['guesses_used'], obs['turn'])
                self.assertEqual(g.score, obs['score'])
                self.assertEqual(g.info()['max_guesses'], 6 if mode == 'wordle' else 10)


if __name__ == '__main__':
    unittest.main()
