"""Reproducible offline opponent comparison and full-adaptor smoke episodes.
Run with the evaluator Python. Writes only runtime/.
"""
import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path

from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points
from game import IsolatedGame, StrongPlayer, make_game

ROOT = Path(__file__).resolve().parent / 'runtime'


def compare(seed):
    start = time.monotonic()
    game = IsolatedGame([RandomPlayer(c) for c in Color], seed=seed)
    strong_color = game.state.colors[seed % 4]
    old_types = iter([RandomPlayer, WeightedRandomPlayer, VictoryPointPlayer])
    game.state.players = [StrongPlayer(c) if c == strong_color else next(old_types)(c)
                          for c in game.state.colors]
    game.observers = list(game.state.players)
    count = 0
    while game.winning_color() is None and game.state.num_turns < 1000 and count < 20000:
        player = game.state.current_player()
        choice = player.decide(game.copy(), list(game.playable_actions))
        game.execute(choice)
        count += 1
    winner = game.winning_color()
    return {'mode': 'strong_vs_previous', 'seed': seed, 'winner': winner.value if winner else None,
            'strong_color': strong_color.value, 'strong_win': winner == strong_color, 'truncated': winner is None,
            'true_vp': {c.value: get_actual_victory_points(game.state, c) for c in game.state.colors},
            'seat_order': [c.value for c in game.state.colors], 'turns': sum(r.action.action_type == ActionType.END_TURN for r in game.state.action_records),
            'actions': count, 'seconds': time.monotonic() - start}


def episode(args):
    seed, policy = args
    start = time.monotonic()
    game = make_game()
    obs = game.reset(seed)
    rng = random.Random(seed + 9000)
    maximum_bytes = 0
    steps = 0
    while not game.done:
        maximum_bytes = max(maximum_bytes, len(json.dumps(obs, allow_nan=False).encode()))
        if policy == 'random':
            action = rng.choice(obs['legal_actions'])['id']
        else:
            player = ValueFunctionPlayer(game.controlled)
            selected = player.decide(game.game.copy(), list(game.game.playable_actions))
            action = next(aid for aid, a in game._ids.items() if a == selected)
        obs = game.step(action)
        steps += 1
        assert maximum_bytes < 200000
    info = game.info()
    result = {k: info[k] for k in ('seed', 'win', 'true_vp', 'public_vp', 'winner', 'turns',
                                 'action_count', 'termination', 'valid_trial', 'invalid_reason')}
    result.update(mode='adaptor', policy=policy, seconds=time.monotonic()-start,
                  score=game.score, observation_max_bytes=maximum_bytes, controlled_steps=steps)
    game.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compare', 'episodes'], default='compare')
    parser.add_argument('--seeds', type=int, nargs='+', default=[901,902,903,904])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--policy', choices=['random','value'], default='random')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    path = ROOT / ('benchmark_compare.jsonl' if args.mode == 'compare' else f'benchmark_episodes_{args.policy}.jsonl')
    function = compare if args.mode == 'compare' else episode
    values = args.seeds if args.mode == 'compare' else [(s,args.policy) for s in args.seeds]
    with path.open('w') as output, concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(function, v) for v in values]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            line = json.dumps(result, allow_nan=False)
            output.write(line+'\n'); output.flush()
            print(line, flush=True)


if __name__ == '__main__':
    main()
