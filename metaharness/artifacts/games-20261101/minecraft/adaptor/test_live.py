"""Exercise real, unassisted survival and persist compact evidence under runtime/."""
import json
import random
import time
from pathlib import Path

from game import make_game

ROOT = Path(__file__).resolve().parent


def poll(game, predicate, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        o = game.observe()
        if not game.info()['valid_trial']:
            raise AssertionError(game.info()['invalid_reason'])
        if predicate(o):
            return o
        if game.done:
            raise AssertionError(f'Episode ended: {game.info()}')
        time.sleep(.1)
    raise AssertionError(f'Timed out: {game.observe().get("action_feedback")}')


def act(game, text, timeout=30):
    game.tick('noop')
    time.sleep(.06)
    previous = (game.observe().get('last_result') or {}).get('id', 0)
    game.tick(text)
    result = poll(game, lambda o: (o.get('last_result') or {}).get('id', 0) > previous, timeout)
    print(text, json.dumps(result.get('last_result')), flush=True)
    return result


def main():
    records = {}
    with make_game(max_seconds=300) as g:
        o = g.reset(0)
        records['episode'] = o['episode']
        records['initial'] = o
        assert o['self']['game_mode'] == 'survival' and o['inventory'] == {}
        assert g.score == 0
        start = o['world']['server_age_ticks']
        time.sleep(2)
        o = poll(g, lambda o: o['world']['server_age_ticks'] > start)
        records['idle_tick_delta'] = o['world']['server_age_ticks'] - start
        logs = [c for c in o['candidates'] if c.get('name', '').endswith('_log')]
        print('spawn', o['self']['position'], 'biome', o['world']['biome'], 'logs', len(logs), flush=True)
        assert logs, 'Calibration seed should expose a tree'
        for candidate in logs[:5]:
            o = act(g, f'gather {candidate["id"]} 1')
            if any(n.endswith('_log') for n in o['inventory']):
                break
        o = poll(g, lambda o: any(n.endswith('_log') for n in o['inventory']))
        log = next(n for n in o['inventory'] if n.endswith('_log'))
        planks = log.replace('_log', '_planks')
        o = act(g, f'craft {planks} 4')
        assert o['last_result']['status'] == 'succeeded'
        o = act(g, 'craft crafting_table 1')
        assert o['last_result']['status'] == 'succeeded'
        o = poll(g, lambda o: o['progress']['components']['opening'] == 3)
        records['after_crafting'] = o
        assert g.score == 3
        print('opening score', g.score, 'inventory', o['inventory'], 'advancements', o['advancements'], flush=True)
        for text in random.Random(123).sample(['wait 0.2', 'hotbar 0', 'shield off', 'chat testing ordinary survival', 'equip crafting_table', '__invalid__'], 6):
            act(g, text)
        # Current movement is renewed each tick and expires without further ticker calls.
        origin = g.observe()['self']['position']
        g.tick('controls jump')
        time.sleep(.5)
        g.tick('controls jump')
        time.sleep(1.5)
        g.tick('noop')
        records['after_random'] = g.observe()
        assert g.info()['valid_trial']
        assert len(json.dumps(g.observe(), allow_nan=False)) < 200000
        records['info'] = g.info()
        server, bridge = g._server, g._bridge
    assert server.poll() is not None and bridge.poll() is not None
    g.close()
    records['owned_processes_reaped'] = True
    (ROOT / 'runtime/live-test.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    print('PASS: live mining, crafting, real-time progression, malformed/random actions, cleanup', flush=True)


if __name__ == '__main__':
    main()
