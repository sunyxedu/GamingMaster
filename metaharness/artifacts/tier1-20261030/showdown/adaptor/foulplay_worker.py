"""Offline Foul Play worker: consumes only one player's protocol and request."""
import copy
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import time

ROOT = Path(os.environ.get('FOUL_PLAY_ROOT', str(Path.home() / 'gm-tools/foul-play')))
PIN = '8f541f80f91f960603673796559f796b10ccf0fb'
if subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
    raise RuntimeError('Foul Play checkout revision mismatch')
sys.path.insert(0, str(ROOT))
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
from fp.battle.state import Battle, LastUsedMove
from fp.battle.protocol import process_battle_updates
from fp.config import FoulPlayConfig
from fp.modes.random_battle import RandomBattleMode
from fp.search.random_battles import prepare_random_battles
from fp.search.poke_engine_helpers import battle_to_poke_engine_state
from fp.data.sets import randbats
from poke_engine import monte_carlo_tree_search

# Never let the upstream dataset loader download anything during gameplay.
DATA = Path(__file__).resolve().parent / 'data/gen9randombattle.json'
randbats.get_randbats_sets_file = lambda _: json.loads(DATA.read_text())
FoulPlayConfig.pokemon_format = 'gen9randombattle'
FoulPlayConfig.search_threads = 1
battle = None
SEARCH_SAMPLES = 4
SAMPLE_MS = 200
DECISION_SECONDS = 2.0


def search(b, connection, seed):
    """Publish completed MCTS samples; the parent can stop native code safely."""
    try:
        random.seed(seed)
        policy, visits = {}, 0
        for sample, weight in prepare_random_battles(b, SEARCH_SAMPLES):
            state = battle_to_poke_engine_state(sample)
            result = monte_carlo_tree_search(
                state, duration_ms=SAMPLE_MS, iterations=0, threads=1)
            visits += result.total_visits
            for option in result.side_one:
                policy[option.move_choice] = policy.get(option.move_choice, 0) + weight * option.visits / max(1, result.total_visits)
            connection.send({'policy': policy, 'visits': visits})
        connection.send({'complete': True})
    except BaseException:
        connection.send({'error': traceback.format_exc()})
    finally:
        connection.close()


def bounded_search(b, deadline):
    # Linux fork retains the already-loaded offline dataset. A Python timer/thread
    # cannot interrupt a native extension holding the GIL; a separate process can.
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=search, args=(b, sender, random.getrandbits(64)), daemon=True)
    latest = None
    complete = False
    child.start()
    sender.close()
    try:
        while (remaining := deadline - time.monotonic()) > 0 and receiver.poll(remaining):
            try:
                reply = receiver.recv()
            except EOFError:
                break
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            if reply.get('complete'):
                complete = True
                break
            latest = reply
    finally:
        if child.is_alive():
            child.terminate()
        child.join(timeout=0.1)
        if child.is_alive():
            child.kill()
            child.join(timeout=0.1)
        receiver.close()
        child.close()
    if latest is None:
        raise RuntimeError('Foul Play deadline expired without a completed MCTS sample')
    return latest['policy'], latest['visits'], not complete

def choose(msg):
    global battle
    start = time.monotonic()
    req = copy.deepcopy(msg['request'])
    req['rqid'] = msg['decision']
    lines = msg['events']
    if battle is None:
        random.seed(msg['seed'])
        battle = Battle('offline')
        battle.pokemon_format = 'gen9randombattle'
        battle.generation = 'gen9'
        battle.battle_type = battle.format_spec.battle_type
        battle.mode = RandomBattleMode()
        battle.mode.datasets.initialize(battle.format_spec)
        battle.user.name = req['side']['id']
        battle.opponent.name = 'p1' if battle.user.name == 'p2' else 'p2'
        battle.user.initialize_first_turn_user_from_json(req)
        battle.started = True
        # As in upstream RandomBattleMode, request initializes our first lead.
        lines = [s for s in lines if not s.startswith('|switch|' + battle.user.name)]
    battle.request_json = req
    battle.force_switch = bool(req.get('forceSwitch'))
    battle.wait = False
    battle.msg_list = lines
    process_battle_updates(battle)
    b = copy.deepcopy(battle)
    b.user.update_from_request_json(req)
    policy, visits, limited = bounded_search(b, start + DECISION_SECONDS)
    def translate(choice):
        if choice.startswith('switch '):
            name = choice[7:]
            for p in b.user.reserve:
                if p.name == name:
                    return f'switch {p.index}'
        else:
            tera = choice.endswith('-tera')
            name = choice.removesuffix('-tera')
            for i, m in enumerate(req.get('active', [{}])[0].get('moves', [])):
                if m['id'] == name:
                    return f'move {i+1}' + (' terastallize' if tera else '')
        return None
    for choice in sorted(policy, key=policy.get, reverse=True):
        action = translate(choice)
        if action in msg['legal']:
            battle.user.last_selected_move = LastUsedMove(battle.user.active.name, choice.removesuffix('-tera'), battle.turn)
            return {'action': action, 'visits': visits, 'deadline_hit': limited,
                    'seconds': time.monotonic() - start}
    raise RuntimeError(f'No legal search result: {policy}; legal={msg["legal"]}')

if __name__ == '__main__':
    for line in sys.stdin:
        try:
            result = choose(json.loads(line))
        except BaseException:
            result = {'error': traceback.format_exc()}
        print(json.dumps(result), flush=True)
