"""Integration checks against the installed engine; outputs only under runtime/."""
import hashlib
import json
import random
from pathlib import Path

from catanatron.models.enums import Action, ActionRecord, ActionType, RESOURCES, DEVELOPMENT_CARDS
from catanatron.models.player import Color
from catanatron.state_functions import player_key
from game import make_game


def check_state(game):
    state = game.game.state
    for i, resource in enumerate(RESOURCES):
        assert state.resource_freqdeck[i] >= 0
        assert state.resource_freqdeck[i] + sum(state.player_state[f'{player_key(state,c)}_{resource}_IN_HAND'] for c in state.colors) == 19
    for color in state.colors:
        key = player_key(state, color)
        buildings = state.buildings_by_color[color]
        for typ, total in [('SETTLEMENT',5),('CITY',4)]:
            assert state.player_state[f'{key}_{typ if typ != "CITY" else "CITIE"}S_AVAILABLE'] + len(buildings[typ]) == total
    assert len(json.dumps(game.observe(), allow_nan=False).encode()) < 200000


def main():
    one, two = make_game(), make_game()
    random.seed(9876)
    before = random.getstate()
    a = one.reset(301)
    assert random.getstate() == before
    b = two.reset(301)
    assert a == b
    rng = random.Random(123)
    trace = []
    old = a['legal_actions'][0]['id']
    for _ in range(24):
        if one.done:
            break
        check_state(one)
        action = rng.choice(a['legal_actions'])['id']
        a = one.step(action)
        # Planning and unrelated random use must not change the second live game.
        copy = two.game.copy()
        if not two.done:
            copy.execute(copy.playable_actions[0])
        for _ in range(50):
            random.random()
        b = two.step(action)
        assert a == b
        trace.append(a)
    assert one.info()['valid_trial'] and two.info()['valid_trial']
    unchanged = one._action_count
    for invalid in ['', '   ', old, [], {}, None, 3, {'id': []}, 'build_road [true,false]']:
        one.step(invalid)
        assert one._action_count == unchanged
    one.step('noop')
    assert one._action_count == unchanged
    controlled = one.controlled
    enemy = next(c for c in Color if c != controlled)
    third = next(c for c in Color if c not in (controlled,enemy))
    for typ,value,result in [(ActionType.BUY_DEVELOPMENT_CARD,None,'VICTORY_POINT'),
                             (ActionType.DISCARD_RESOURCE,'ORE','ORE'),
                             (ActionType.MOVE_ROBBER,((0,0,0),third),'ORE')]:
        event = one._public_event(ActionRecord(Action(enemy,typ,value),result),0)
        assert event['result'] is None
        if typ == ActionType.DISCARD_RESOURCE:
            assert event['parameters'] is None
    event = one._public_event(ActionRecord(Action(enemy,ActionType.MOVE_ROBBER,((0,0,0),controlled)),'ORE'),0)
    assert event['result'] == 'ORE'
    cap = make_game(max_actions=1)
    obs = cap.reset(104)
    obs = cap.step(obs['legal_actions'][0]['id'])
    assert cap.done and cap.info()['termination'] == 'truncated' and cap.info()['win'] == 0
    assert cap.info()['turns'] == 0
    for game in (one,two,cap):
        game.close(); game.close()
    digest = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    print(json.dumps({'status':'PASS','checks':['RNG isolation','same-seed action replay', 'resource/piece invariants',
          'strict JSON size','malformed/stale/noop','private event filtering','cap truncation','idempotent close'],
          'trace_sha256':digest,'replayed_steps':len(trace)}))


if __name__ == '__main__':
    main()
