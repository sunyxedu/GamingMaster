"""Private acceptance checks; never imported or exposed to a playing harness."""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import time

ROOT = Path(__file__).resolve().parent


def disable_network():
    """Linux x86_64 seccomp: deny IPv4/IPv6 sockets, retain local UNIX IPC."""
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]
    class Program(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    # arch mismatch -> kill, socket(AF_INET/AF_INET6) -> EPERM; other syscalls allowed.
    rows = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xc000003e), (0x06, 0, 0, 0x80000000),
            (0x20, 0, 0, 0), (0x15, 0, 4, 41), (0x20, 0, 0, 16),
            (0x15, 1, 0, 2), (0x15, 0, 1, 10), (0x06, 0, 0, 0x50001),
            (0x06, 0, 0, 0x7fff0000)]
    filters = (Filter * len(rows))(*(Filter(*row) for row in rows))
    program = Program(len(rows), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot install offline seccomp filter')
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.socket(family)
        except PermissionError:
            pass
        else:
            raise AssertionError('Network socket was not blocked')


def audit_suite():
    from game import SEEDS, make_game
    from tw_runtime import resolve
    records = []
    for group, seeds in SEEDS.items():
        assert len(set(s % 3 for s in seeds)) == 3
        assert len(set(Counter(s % 3 for s in seeds).values())) == 1
        for seed in seeds:
            started = time.monotonic()
            path, manifest = resolve(seed)
            obj = json.loads(path.with_suffix('.json').read_text())
            # Ignore unique metadata when detecting semantically duplicate content.
            obj.pop('metadata', None)
            semantic = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
            world = hashlib.sha256(json.dumps(obj['world'], sort_keys=True).encode()).hexdigest()
            game = make_game()
            try:
                observation = game.reset(seed)
                assert game.info()['valid_trial'], game.info()
                assert not game.done and 0 <= game.score <= 1
                assert len(json.dumps(observation, allow_nan=False).encode()) < 200_000
                records.append(dict(group=group, seed=seed, semantic_sha256=semantic,
                                    world_sha256=world, retry=manifest['retry'],
                                    max_score=observation['max_score'],
                                    generation_seconds=manifest['generation_seconds'],
                                    elapsed_seconds=time.monotonic() - started))
                print('accepted', group, seed, observation['max_score'], flush=True)
            finally:
                game.close()
    for key in ('semantic_sha256', 'world_sha256'):
        assert len({r[key] for r in records}) == len(records), f'Duplicate {key}'
    return records


def checks():
    from game import SEEDS, make_game
    from tw_runtime import Interpreter, resolve
    import textworld
    timings = []
    for seed in SEEDS['dev']:
        game = make_game()
        started = time.monotonic()
        initial = game.reset(seed)
        reset_seconds = time.monotonic() - started
        assert game.info()['valid_trial'], game.info()
        sequence = []
        rng = random.Random(seed)
        for _ in range(8):
            if game.done:
                break
            action = rng.choice(game.observe()['admissible_commands'])
            started = time.monotonic()
            obs = game.step(action)
            timings.append(time.monotonic() - started)
            sequence.append((action, obs))
        assert game.reset(seed) == initial, 'Same-instance reset leaked state'
        game.close()
        twin = make_game()
        assert twin.reset(seed) == initial
        for action, expected in sequence:
            assert twin.step(action) == expected, 'Fresh-instance replay changed'
        twin.close(); twin.close()
        print('deterministic replay', seed, 'reset seconds', reset_seconds, flush=True)
    game = make_game()
    first = game.reset(300)
    first['admissible_commands'].append('debug')
    assert 'debug' not in game.observe()['admissible_commands']
    for bad in (None, {}, [], 42):
        obs = game.step(bad)
        assert obs['commands_used'] == 0 and obs['protocol_error']
    for bad in ('look\ninventory', 'look. inventory', 'tw-print state', 'undo', 'restart', '__not_an_action__'):
        before = game.observe()
        obs = game.step(bad)
        assert obs['commands_used'] == before['commands_used'] + 1
        assert obs['moves'] == before['moves'] and obs['protocol_error']
    game.reset(300)
    for _ in range(100):
        game.step('look')
    assert game.done and game.observe()['truncated'] and not game.observe()['lost']
    frozen = game.observe()
    assert game.step('look') == frozen
    assert game._engine is None
    assert game.reset(300)['commands_used'] == 0
    # Verify actual interpreter crash becomes infrastructure invalidity.
    pid = game._engine.process.pid
    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    assert children, 'No native Glulx child'
    os.kill(int(children[0]), signal.SIGKILL)
    obs = game.step('look')
    assert game.done and not game.info()['valid_trial'] and obs['infrastructure_error']
    game.close()
    assert not Path(f'/proc/{pid}').exists(), 'Worker was not reaped'
    # Native partial-credit fixture: take apple earns 2; placing carrot earns 1;
    # eating carrot loses while preserving the already earned two points.
    fixture = ROOT / 'runtime' / 'partial-credit'
    fixture.mkdir(exist_ok=True)
    options = textworld.GameOptions(); options.seeds = 987654
    maker = textworld.GameMaker(options)
    room = maker.new_room('test chamber'); maker.set_player(room)
    apple = maker.new(type='f', name='apple'); carrot = maker.new(type='f', name='carrot')
    table = maker.new(type='s', name='table')
    room.add(apple, carrot, table)
    from textworld.generator.game import Quest, Event
    maker.quests = [Quest(win_events=[Event(conditions=[maker.new_fact('in', apple, maker.inventory)])], reward=2),
                    Quest(win_events=[Event(conditions=[maker.new_fact('on', carrot, table)])],
                          fail_events=[Event(conditions=[maker.new_fact('eaten', carrot)])], reward=1)]
    maker.compile(str(fixture / 'story.ulx'))
    def fixture_game(cap):
        g = make_game()
        g._profile = dict(name='fixture', cap=cap)
        g._manifest = {'acceptance': {'max_score': 3}, 'generation_seed': 987654}
        g._engine = Interpreter(fixture / 'story.ulx')
        g._accept(g._engine.request('reset'))
        return g
    g = fixture_game(100)
    for command in ('take apple', 'take carrot', 'eat carrot'):
        g.step(command)
    assert g.info()['valid_trial'], g.info()
    assert g.done and g.observe()['lost'] and g.score == 2/3, g.observe()
    g.close()
    g = fixture_game(1); g.step('take apple')
    assert g.done and g.observe()['truncated'] and g.score == 2/3
    g.close()
    g = fixture_game(100)
    for command in ('take apple', 'take carrot', 'put carrot on table'):
        g.step(command)
    assert g.observe()['won'] and g.score == 1
    g.close()
    return dict(deterministic_seeds=SEEDS['dev'], partial_loss=2/3, partial_truncation=2/3,
                native_win=1, crash_classified=True,
                mean_step_seconds=statistics.mean(timings), max_step_seconds=max(timings))


def baseline(policy, seed):
    """No private state: objective overlap, object collection, and novelty exploration."""
    from game import make_game
    game = make_game(); obs = game.reset(seed)
    rng = random.Random(seed)
    visits = Counter()
    trace = []
    while not game.done:
        if policy == 'look':
            action = 'look'
        elif policy == 'random':
            action = rng.choice(obs['admissible_commands'])
        else:
            objective = set(re.findall(r"[a-z]+", obs['objective'].lower())) - {
                'the', 'a', 'in', 'on', 'of', 'to', 'and', 'from', 'with', 'you', 'your', 'then'}
            state = (obs['description'], obs['inventory'])
            def value(command):
                words = set(re.findall(r'[a-z]+', command.lower()))
                overlap = len((words - set(('go','take','put','insert','open','close','eat','unlock','lock','drop','examine'))) & objective)
                verb = command.split()[0]
                base = {'look': -10, 'inventory': -10, 'examine': -8, 'drop': -4,
                        'go': 2, 'take': 3, 'open': 4, 'unlock': 5, 'put': 1,
                        'insert': 1, 'close': 0, 'lock': -1, 'eat': -8}.get(verb, 0)
                if verb == 'eat' and ('eat' in objective or 'consume' in objective):
                    base = 5
                return base + 3 * overlap - 7 * visits[(state, command)]
            candidates = sorted(obs['admissible_commands'])
            action = max(candidates, key=value)
            visits[(state, action)] += 1
        obs = game.step(action)
        if len(trace) < 10:
            trace.append(action)
    assert game.info()['valid_trial'], game.info()
    result = dict(policy=policy, seed=seed, score=game.score, **{
        k: game.info()[k] for k in ('won','lost','truncated','commands_used')})
    game.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    args = parser.parse_args()
    if args.offline:
        disable_network()
        print('IPv4/IPv6 sockets blocked by inherited seccomp; local UNIX IPC allowed', flush=True)
    report = dict(offline=args.offline)
    if args.suite:
        report['suite'] = audit_suite()
    if args.checks:
        report['checks'] = checks()
    if args.baselines:
        from game import SEEDS
        report['baselines'] = [baseline(p,s) for p in ('look','random','objective-explorer') for s in SEEDS['dev']]
        print(json.dumps(report['baselines'], indent=2), flush=True)
    name = '-'.join(k for k in ('suite', 'checks', 'baselines') if getattr(args,k))
    (ROOT/'runtime'/f'verification-{name}.json').write_text(json.dumps(report, indent=2))
    print('PASS', name, flush=True)
