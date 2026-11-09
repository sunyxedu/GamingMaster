"""Observation-only baseline evaluation; no target/seed lookup during decisions."""
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from game import ALLOWED_GUESSES, ANSWERS, SEEDS, make_game


def word_clue(answer, guess):
    pool = list(answer)
    clue = ['gray'] * 5
    for i in range(5):
        if answer[i] == guess[i]:
            clue[i] = 'green'
            pool[i] = None
    for i, c in enumerate(guess):
        if clue[i] != 'green' and c in pool:
            pool[pool.index(c)] = None
            clue[i] = 'yellow'
    return tuple(clue)


def peg_clue(answer, guess):
    exact = sum(a == b for a, b in zip(answer, guess))
    return exact, sum((Counter(answer) & Counter(guess)).values()) - exact


def select_guess(candidates, universe, clue):
    if len(candidates) == 1:
        return candidates[0]
    possible = set(candidates)
    best = None
    winner = None
    for guess in universe:
        bins = Counter(clue(answer, guess) for answer in candidates)
        rank = (max(bins.values()), guess not in possible, guess)
        if best is None or rank < best:
            best, winner = rank, guess
    return winner


def run(mode, policy, seed):
    g = make_game(mode=mode)
    obs = g.reset(seed)
    rng = random.Random(4291)  # Policy randomness deliberately independent of episode seed.
    universe = list(ANSWERS) if mode == 'wordle' else list(itertools.product(range(6), repeat=4))
    candidates = universe[:]
    word_guesses = sorted(ALLOWED_GUESSES)
    clue = word_clue if mode == 'wordle' else peg_clue
    while not obs['done']:
        if policy == 'random':
            guess = rng.choice(word_guesses if mode == 'wordle' else universe)
        elif not obs['history']:
            guess = 'slate' if mode == 'wordle' else (0,0,1,1)
        else:
            # Wordle uses candidate-only minimax. Mastermind uses all legal probes.
            guess = select_guess(candidates, candidates if mode == 'wordle' else universe, clue)
        obs = g.step({'guess': guess})
        assert obs['error'] is None
        feedback = obs['history'][-1]['feedback']
        expected = tuple(feedback) if mode == 'wordle' else (feedback['exact'],feedback['misplaced'])
        candidates = [a for a in candidates if clue(a,guess) == expected]
        assert candidates
    result = {k: g.info()[k] for k in ['solved','guesses_used','max_guesses','score']}
    g.close()
    return result


def main():
    output = {}
    # Only dev + train; held-out and validation sets are not used for policy trials.
    seeds = SEEDS['dev'] + SEEDS['train']
    for mode in ['wordle','mastermind']:
        output[mode] = {}
        for policy in ['random','minimax']:
            results = [run(mode,policy,seed) for seed in seeds]
            summary = {'episodes':len(results),'wins':sum(r['solved'] for r in results),
                       'mean_guesses':mean(r['guesses_used'] for r in results),
                       'mean_score':mean(r['score'] for r in results),
                       'max_score':max(r['score'] for r in results), 'episodes_results':results}
            output[mode][policy] = summary
            assert summary['mean_score'] < 1.0
            print(mode,policy,{k:v for k,v in summary.items() if k!='episodes_results'},flush=True)
    runtime = Path(__file__).resolve().parent / 'runtime'
    runtime.mkdir(exist_ok=True)
    (runtime / 'baseline_results.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
