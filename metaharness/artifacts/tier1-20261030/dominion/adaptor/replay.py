"""Replay accepted decisions from runtime/episode-*/replay.jsonl offline."""
import json
import sys
from pathlib import Path
from game import make_game
from scenarios import ENGINE_COMMIT, ADAPTOR_VERSION


def replay(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    header, *decisions = records
    if header['engine_commit'] != ENGINE_COMMIT or header['adaptor_version'] != ADAPTOR_VERSION:
        raise ValueError('Replay version does not match this adaptor')
    config = header['config']
    game = make_game(max_turns=config['max_turns'], max_decisions=config.get('max_decisions', 20000))
    try:
        game.reset(config['seed'])
        for decision in decisions:
            game.step(decision)
            if game.observe()['last_error']:
                raise ValueError(game.observe()['last_error'])
        return game.info()
    finally:
        game.close()


if __name__ == '__main__':
    print(json.dumps(replay(sys.argv[1]), indent=2))
