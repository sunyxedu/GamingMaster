"""Offline, public-observation-only smoke policies and paired evaluation reports."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from game import make_game
from scenarios import SEEDS


def money_action(obs):
    """Agent-side SM-1/WM-1 when present, otherwise BM-1; used only for validation."""
    p = obs['prompt']
    options = p['options']
    target = 'Witch' if 'Witch' in obs['supply'] else 'Smithy'
    selected = []
    if p['kind'] == 'buy':
        provinces = obs['supply']['Province']['count']
        priority = ['Province']
        if provinces <= 4:
            priority.append('Duchy')
        if provinces <= 2:
            priority.append('Estate')
        if provinces >= 5 and not obs['self']['owned'].get(target, 0):
            if target != 'Witch' or obs['supply']['Curse']['count']:
                priority.append(target)
        priority += ['Gold', 'Silver']
        for name in priority:
            matching = [x['id'] for x in options if x['label'] == name]
            if matching:
                selected = matching[:1]
                break
    elif p['kind'] == 'play_action':
        selected = [x['id'] for x in options if x['label'] == target][:1]
    elif p['kind'] == 'yes_no':
        selected = [x['id'] for x in options if x['label'] == 'True']
    elif p['kind'] == 'discard':
        priority = {'Gold': 3, 'Silver': 2, 'Copper': 1, 'Province': .6, 'Duchy': .3, 'Estate': .1, 'Curse': -1}
        keep = []
        terminals = [x for x in options if x['label'] == target]
        if terminals and len(options) > p['min']:
            keep.append(terminals[0]['id'])
        for x in sorted(options, key=lambda x: -priority.get(x['label'], 0)):
            if len(keep) >= len(options) - p['min']:
                break
            if x['id'] not in keep:
                keep.append(x['id'])
        selected = [x['id'] for x in options if x['id'] not in keep]
    else:
        selected = [x['id'] for x in options[:p['min']]]
    return {'decision_id': obs['decision_id'], 'option_ids': selected}


def run(seed, random_actions=False):
    g = make_game()
    rng = random.Random(seed + 999)  # only testing policy randomness, never game state
    try:
        obs = g.reset(seed)
        steps, max_bytes = 0, 0
        while not g.done:
            assert g.info().get('valid_trial'), g.info()
            encoded = json.dumps(obs, allow_nan=False)
            max_bytes = max(max_bytes, len(encoded.encode()))
            assert max_bytes < 200000
            assert 'shuffle_seed' not in encoded and '"seed"' not in encoded
            if random_actions:
                p = obs['prompt']
                n = rng.randint(p['min'], p['max'])
                ids = rng.sample([o['id'] for o in p['options']], n)
                action = dict(decision_id=obs['decision_id'], option_ids=ids)
            else:
                action = money_action(obs)
            obs = g.step(action)
            steps += 1
            assert steps < 20000
        result = g.info()
        assert result['valid_trial'], result
        assert not result['truncated'], result
        assert result['illegal_action_count'] == result['fallback_count'] == 0
        assert g.score == result['agent_vp'] - result['opponent_vp']
        result.update(decisions=steps, max_observation_bytes=max_bytes, random_actions=random_actions)
        return result
    finally:
        g.close()


def aggregate(results):
    groups = defaultdict(list)
    for r in results:
        groups['overall'].append(r)
        for key in ('kingdom_id', 'opponent_id', 'agent_seat'):
            groups[f'{key}={r[key]}'].append(r)
    return {k: {'episodes': len(rs), 'win_rate_draw_half': sum(r['win_score'] for r in rs) / len(rs),
                'mean_vp_margin': sum(r['vp_margin'] for r in rs) / len(rs)}
            for k, rs in groups.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()
    results = [run(seed, args.random) for seeds in SEEDS.values() for seed in seeds]
    report = dict(episodes=results, aggregate=aggregate(results))
    path = Path(__file__).parent / 'runtime' / ('random-report.json' if args.random else 'baseline-report.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['aggregate'], indent=2))
