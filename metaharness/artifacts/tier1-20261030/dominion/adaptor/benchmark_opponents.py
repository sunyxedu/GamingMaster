"""Offline, paired-seat opponent comparison on the real patched engine."""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from engine_patches import apply
from scenarios import KINGDOMS, ADAPTOR_VERSION, ENGINE_COMMIT
from worker import Session, MoneyPolicy, StrongestBuiltInPolicy
from pyminion.bots.examples.big_money import BigMoneyDecider
from pyminion.bots.examples.big_money_smithy import BigMoneySmithyDecider
from pyminion.bots.examples.chapel_bot import ChapelBotDecider

POLICIES = {'BMU-1': StrongestBuiltInPolicy,
            'BM-1': lambda: MoneyPolicy('BM-1'),
            'SM-1': lambda: MoneyPolicy('SM-1'), 'WM-1': lambda: MoneyPolicy('WM-1'),
            'builtin_money': BigMoneyDecider, 'builtin_smithy': BigMoneySmithyDecider,
            'builtin_chapel': ChapelBotDecider}


def match(candidate, baseline, kingdom, seed, seat):
    s = Session(dict(kingdom_id=kingdom, opponent_id='BMU-1', agent_seat=seat,
                     shuffle_seed=seed, max_turns=400))
    s.agent.decider = POLICIES[candidate]()
    s.opponent.decider = POLICIES[baseline]()
    s.publish = lambda prompt: None
    s.run()
    result = s.summary()
    assert not result['truncated'], result
    assert sum(s.agent.owned.values()) == s.agent.get_all_cards_count()
    return dict(candidate=candidate, baseline=baseline, kingdom=kingdom, seed=seed,
                seat=seat, **result)


def aggregate(rows):
    return dict(games=len(rows), wins=sum(r['outcome'] == 'win' for r in rows),
                draws=sum(r['outcome'] == 'draw' for r in rows),
                win_rate=sum(r['outcome'] == 'win' for r in rows) / len(rows),
                win_score=sum(r['win_score'] for r in rows) / len(rows),
                mean_vp_margin=sum(r['vp_margin'] for r in rows) / len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--candidate', default='BMU-1')
    p.add_argument('--suite', choices=['previous', 'builtins'], default='previous')
    p.add_argument('--seed-start', type=int, default=900000)
    p.add_argument('--output', default='runtime/opponent-benchmark.json')
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    apply()
    rows = []
    prior_suite = [('base_intro', 'BM-1'), ('chapel_engine', 'SM-1'),
                   ('curse_race', 'WM-1'), ('gardens_rush', 'SM-1'),
                   ('heldout_thin_gardens', 'BM-1'), ('heldout_witch_gain', 'WM-1')]
    if args.suite == 'previous':
        suite = prior_suite
    else:
        suite = [(k, policy) for k, cards in KINGDOMS.items()
                 for policy in ('builtin_money', 'builtin_smithy', 'builtin_chapel')
                 if policy != 'builtin_chapel' or 'Chapel' in cards]
    for kingdom, baseline in suite:
        offset = list(KINGDOMS).index(kingdom) * 10000
        subset = [match(args.candidate, baseline, kingdom, args.seed_start + offset + i, seat)
                  for i in range(args.pairs) for seat in (0, 1)]
        rows.extend(subset)
        print(kingdom, baseline, aggregate(subset), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['baseline']].append(row)
    report = dict(pairs_per_kingdom=args.pairs, shuffle_seed_start=args.seed_start,
                  suite=args.suite, adaptor_version=ADAPTOR_VERSION, engine_commit=ENGINE_COMMIT,
                  overall=aggregate(rows),
                  by_baseline={k: aggregate(v) for k, v in groups.items()},
                  by_kingdom={k: aggregate([r for r in rows if r['kingdom'] == k])
                              for k in dict.fromkeys(r['kingdom'] for r in rows)},
                  by_seat={str(s): aggregate([r for r in rows if r['seat'] == s]) for s in (0, 1)},
                  rows=rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(report['by_baseline'])


if __name__ == '__main__':
    main()
