"""Private TextWorld process: generation validation or allowlisted gameplay RPC."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import textworld

FIELDS = ('feedback', 'description', 'inventory', 'objective', 'admissible_commands',
          'score', 'max_score', 'won', 'lost', 'moves')


def infos():
    return textworld.EnvInfos(**{k: True for k in FIELDS})


def snapshot(state, score=None, done=False):
    result = {key: state.get(key) for key in FIELDS}
    result['admissible_commands'] = sorted(set(result['admissible_commands'] or []))
    result['returned_score'] = state.score if score is None else score
    result['terminated'] = bool(done or state.won or state.lost)
    return result


def generate(config, folder):
    options = textworld.GameOptions()
    options.seeds = config['generation_seed']
    options.nb_rooms = config['profile']['rooms']
    options.nb_objects = config['profile']['objects']
    options.quest_length = config['profile']['quest_length']
    options.nb_parallel_quests = 1
    options.quest_breadth = 1
    options.file_ext = '.ulx'
    options.path = str(folder / 'story.ulx')
    options.force_recompile = True
    path, game = textworld.make(options)
    if game.max_score <= 0:
        raise ValueError('Nonpositive generated max_score')
    # Privileged acceptance test, never requested by or revealed to the player.
    env = textworld.start(path, request_infos=infos())
    try:
        state = env.reset()
        trace = [state.score]
        for command in game.metadata['walkthrough']:
            state, score, done = env.step(command)
            if score != state.score or not 0 <= score <= game.max_score:
                raise ValueError('Native cumulative score contract failed')
            trace.append(score)
            if done:
                break
        if not state.won or state.score != game.max_score:
            raise ValueError('Generated solution did not win the compiled game')
    finally:
        env.close()
    (folder / 'accepted.json').write_text(json.dumps({
        'max_score': game.max_score,
        'resolved_seeds': {k: int(v) for k, v in options.seeds.items()},
        'solution_length': len(game.metadata['walkthrough']),
        'native_score_trace': trace,
    }, sort_keys=True))


def serve(path):
    env = None
    try:
        env = textworld.start(path, request_infos=infos())
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            if request['op'] == 'reset':
                response = snapshot(env.reset())
            elif request['op'] == 'step':
                state, score, done = env.step(request['command'])
                response = snapshot(state, score, done)
            else:
                raise ValueError('Unknown RPC operation')
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    if sys.argv[1] == 'generate':
        folder = Path(sys.argv[2])
        generate(json.loads((folder / 'config.json').read_text()), folder)
    elif sys.argv[1] == 'serve':
        serve(sys.argv[2])
    else:
        raise ValueError('Unknown worker mode')
