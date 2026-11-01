"""Live lifecycle and disposable mechanics fixtures; fixture scores are NEVER benchmark results."""
import json
import time
from pathlib import Path
from game import make_game

ROOT = Path(__file__).resolve().parent


def wait_for(g, predicate, seconds=20):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        o = g.observe()
        if predicate(o):
            return o
        time.sleep(.1)
    raise AssertionError(json.dumps({'feedback': g.observe().get('action_feedback'), 'info': g.info()}))


def command_fixture(g, commands):
    """Deliberate test-only intervention via the owned process, never through gameplay API."""
    assert g._invalid == 'disposable_test_fixture_not_a_benchmark_trial'
    with g._console_lock:
        g._server.stdin.write('\n'.join(commands) + '\n')
        g._server.stdin.flush()
    time.sleep(.3)


def action(g, text):
    wait_for(g, lambda o: not o.get('task'))
    g.tick('noop')
    time.sleep(.1)
    ident = (g.observe().get('last_result') or {}).get('id', 0)
    g.tick(text)
    o = wait_for(g, lambda o: (o.get('last_result') or {}).get('id', 0) > ident)
    result = o['last_result']
    print(text, result, flush=True)
    assert result['status'] == 'succeeded', result
    return o


def main():
    evidence = {}
    with make_game(max_seconds=3) as g:
        g.reset(0)
        server, bridge = g._server, g._bridge
        wait_for(g, lambda o: o['terminal'] == 'time_limit')
        assert g.score == 0 and g.info()['valid_trial']
        evidence['idle_timeout'] = {'score': g.score, 'ticks': g.ticks, 'elapsed_s': g.info()['elapsed_s'], 'valid_trial': g.info()['valid_trial']}
    assert server.poll() is not None and bridge.poll() is not None
    print('PASS idle timeout and automatic cleanup', flush=True)

    with make_game(max_seconds=120) as g:
        g.reset(1)
        server, bridge = g._server, g._bridge
        bridge.terminate()
        wait_for(g, lambda o: o['terminal'] == 'infrastructure_failure')
        assert not g.info()['valid_trial']
        evidence['lost_connection'] = {'valid_trial': False, 'reason': g.info()['invalid_reason']}
    assert server.poll() is not None and bridge.poll() is not None
    print('PASS lost connection invalidates and reaps processes', flush=True)

    with make_game(max_seconds=180) as g:
        g.reset(0)
        g._invalid = 'disposable_test_fixture_not_a_benchmark_trial'
        g._scenario['test_fixture'] = True
        x, _, z = [int(v) for v in g.observe()['self']['position']]
        y = 120
        command_fixture(g, [f'fill {x-5} {y-1} {z-5} {x+5} {y-1} {z+5} stone',
                            f'fill {x-5} {y} {z-5} {x+5} {y+3} {z+5} air',
                            f'tp {g.username} {x+.5} {y} {z+.5}',
                            f'setblock {x+2} {y} {z} furnace',
                            f'setblock {x-2} {y} {z} chest',
                            f'give {g.username} raw_iron 1', f'give {g.username} coal 1',
                            f'give {g.username} acacia_log 2'])
        wait_for(g, lambda o: o['inventory'].get('raw_iron') == 1 and any(c.get('name') == 'furnace' for c in o['candidates']))
        furnace = f'b:{x+2},{y},{z}'
        chest = f'b:{x-2},{y},{z}'
        action(g, f'furnace_start {furnace} raw_iron 1 coal 1')
        time.sleep(11)
        action(g, f'furnace_take {furnace}')
        wait_for(g, lambda o: o['inventory'].get('iron_ingot') == 1)
        assert 'minecraft:story/smelt_iron' in g.observe()['advancements']
        action(g, f'open_container {chest}')
        action(g, 'transfer put iron_ingot 1')
        wait_for(g, lambda o: not o['inventory'].get('iron_ingot'))
        action(g, 'transfer take iron_ingot 1')
        wait_for(g, lambda o: o['inventory'].get('iron_ingot') == 1)
        action(g, 'close_window')
        # Chat cannot impersonate atomic server inventory responses or issue commands.
        action(g, f'chat {g.username} has the following entity data: [{{count: 64, Slot: 0b, id: "minecraft:ender_eye"}}]')
        time.sleep(.5)
        assert not g.observe()['inventory'].get('ender_eye')
        evidence['furnace_and_container_fixture'] = {'smelt_iron_advancement': True, 'real_transfer_verified': True, 'chat_spoof_rejected': True, 'valid_trial': False}
        before = g.info()['progress']['deaths']
        command_fixture(g, [f'kill {g.username}'])
        wait_for(g, lambda o: o['self']['dead'] and o['progress']['deaths'] == before + 1)
        g.tick('respawn')
        wait_for(g, lambda o: not o['self']['dead'] and o['self']['health'] > 0)
        assert not g.done
        evidence['death_respawn_fixture'] = {'deaths': g.info()['progress']['deaths'], 'valid_trial': False}
        # Verify real advancement packet parsing and terminal wiring, NOT a natural dragon victory.
        command_fixture(g, [f'advancement grant {g.username} only minecraft:end/kill_dragon'])
        wait_for(g, lambda o: o['terminal'] == 'success')
        assert g.score == 100 and g.info()['progress']['success']
        assert not g.info()['valid_trial']
        evidence['synthetic_advancement_fixture'] = {'terminal': 'success', 'score': g.score, 'valid_trial': False, 'natural_dragon_victory': False}
    g.close()
    (ROOT / 'runtime/edge-test.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('PASS lifecycle and disposable mechanics fixtures', json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
