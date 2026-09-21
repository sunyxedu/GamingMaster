"""Integration checks against real cached games, plus labeled public-response fixtures."""
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from unittest.mock import patch

from game import make_game, MANIFEST, InterfaceError, ROOT


def check_observation(obs):
    assert len(json.dumps(obs).encode()) < 200_000
    assert len(obs['frame']) == 64 and all(len(row) == 64 for row in obs['frame'])
    assert 0 <= obs['levels_completed'] <= obs['score'] <= obs['win_levels']
    assert obs['completion_fraction'] == obs['score'] / obs['win_levels']
    if obs['screenshot']:
        from PIL import Image
        image = Image.open(obs['screenshot'])
        assert image.size == (512, 512)
        from arc_agi.rendering import COLOR_MAP
        assert image.getpixel((0, 0)) == tuple(bytes.fromhex(COLOR_MAP[obs['frame'][0][0]][1:])[:3])
    if obs['animation_frames_path']:
        frames = json.loads(Path(obs['animation_frames_path']).read_text())
        assert len(frames) == obs['animation_frame_count']
        assert frames[-1] == obs['frame']
    else:
        assert obs['animation_frame_count'] == 1


def public_response(raw, *, levels=None, state=None, frame=None):
    # No simulator internals: only the public response schema is synthesized.
    return SimpleNamespace(
        frame=raw.frame if frame is None else frame,
        levels_completed=raw.levels_completed if levels is None else levels,
        win_levels=raw.win_levels,
        available_actions=raw.available_actions,
        state=SimpleNamespace(value=raw.state.value if state is None else state),
        full_reset=raw.full_reset, action_input=raw.action_input, guid=raw.guid)


def main():
    report = {'backend': 'real pinned OFFLINE SDK', 'initializations': [], 'random_play': [], 'fixtures': []}
    game = make_game(max_actions=1000)
    initial = {}
    try:
        for seed, (gid, levels) in enumerate(MANIFEST):
            obs = game.reset(seed)
            check_observation(obs)
            assert game.info()['game_id'] == gid and obs['win_levels'] == levels
            initial[seed] = obs
            assert game.score == 0.0 and obs['episode_actions'] == 0
            report['initializations'].append({'seed': seed, 'game_id': gid, 'win_levels': levels,
                                             'observation_bytes': len(json.dumps(obs).encode())})
        for seed in (0, 1, 8):
            assert game.reset(seed) == initial[seed % 8]
        # Independent public frame mutation cannot corrupt the retained observation.
        obs = game.observe()
        obs['frame'][0][0] = -1
        assert game.observe()['frame'][0][0] >= 0

        # Count each request, but never send noops or invalid coordinates to the SDK.
        before = game.reset(0)
        after = game.step('noop')
        assert after['frame'] == before['frame'] and game.info()['sdk_action_calls'] == 0
        for bad in ('click', 'click -1 5', 'click 64 0', {'id': 6, 'x': True, 'y': 0},
                    {'id': 6, 'x': 1.5, 'y': 2}, {'id': 6}, {'id': 9}, 'undo'):
            try:
                game.step(bad)
                raise AssertionError(f'Accepted invalid input: {bad}')
            except ValueError:
                pass
        assert game.info()['sdk_action_calls'] == 0 and game.info()['invalid_actions'] == 8
        assert game.observe()['frame'] == before['frame']
        for bad_seed in (-1, True, 1.5, '0'):
            try:
                game.reset(bad_seed)
                raise AssertionError('Accepted invalid seed')
            except ValueError:
                pass

        rng = random.Random(1729)
        max_animation = 1
        for seed in range(4):
            obs = game.reset(seed)
            changed = 0
            max_score = 0
            game_overs = 0
            rewards = 0.0
            animation_max = 1
            for _ in range(160 if seed == 2 else 40):
                if game.done:
                    break
                old = obs['frame']
                if obs['state'] == 'GAME_OVER':
                    game_overs += 1
                    action = 'reset'
                else:
                    action_id = rng.choice(obs['available_actions'])
                    action = {'id': action_id}
                    if action_id == 6:
                        action.update(x=rng.randrange(64), y=rng.randrange(64))
                obs = game.step(action)
                check_observation(obs)
                changed += obs['frame'] != old
                max_score = max(max_score, obs['levels_completed'])
                rewards += obs['reward']
                assert game.score == max_score == rewards
                animation_max = max(animation_max, obs['animation_frame_count'])
            assert changed > 0
            max_animation = max(max_animation, animation_max)
            report['random_play'].append({'seed': seed, 'game_id': game.info()['game_id'],
                                         'actions': obs['episode_actions'], 'changed_frames': changed,
                                         'max_animation_frames': animation_max, 'game_overs': game_overs,
                                         'score': game.score, 'state': obs['state']})
        assert max_animation > 1  # Exercise real temporal frame preservation.
        # Live retry after gameplay vs duplicate reset, and actual undo dispatch.
        game.reset(1)
        game.step('up')
        retry = game.step('reset')
        duplicate = game.step('reset')
        game.step('undo')
        assert game.info()['retries'] == 2 and game.info()['undos'] == 1
        assert game.info()['episode_actions'] == 4
        report['live_retry'] = {'first_full_reset': retry['full_reset'],
                                'duplicate_full_reset': duplicate['full_reset']}

        # These fixtures test hard-to-reach response/accounting paths, not solved games.
        game.reset(0)
        raw = game._env.observation_space
        for levels, expected, reward in ((2, 2, 2), (0, 2, 0), (1, 2, 0), (2, 2, 0), (3, 3, 1)):
            with patch.object(game._env, 'step', return_value=public_response(raw, levels=levels)):
                obs = game.step('reset')
            assert game.score == expected and obs['reward'] == reward
        report['fixtures'].append('high-water progress 2 -> 0 -> 1 -> 2 -> 3 scores only 3')
        with patch.object(game._env, 'step', return_value=public_response(raw, levels=3, state='GAME_OVER')):
            game.step('up')
        assert not game.done
        with patch.object(game._env, 'step', side_effect=AssertionError('post GAME_OVER numbered call')):
            try:
                game.step('up')
                raise AssertionError('numbered action accepted after GAME_OVER')
            except ValueError:
                pass
        with patch.object(game._env, 'step', return_value=public_response(raw, levels=3)):
            assert game.step('reset')['state'] == 'NOT_FINISHED'
        report['fixtures'].append('GAME_OVER blocks numbered calls but permits explicit retry')
        with patch.object(game._env, 'step', return_value=public_response(raw, levels=6, state='WIN')):
            game.step('up')
        assert game.done and game.score == 6.0 and not game.observe()['truncated']
        with patch.object(game._env, 'step', side_effect=AssertionError('post WIN SDK call')):
            obs = game.step('reset')
        assert obs['won'] and obs['reward'] == 0
        report['fixtures'].append('WIN ends episode and no subsequent call reaches SDK')
        for failed_response in (None, public_response(raw, frame=[])):
            game.reset(0)
            before = game.observe()
            with patch.object(game._env, 'step', return_value=failed_response):
                try:
                    game.step('up')
                    raise AssertionError('missing response accepted')
                except InterfaceError:
                    pass
            assert game.done and game.observe()['frame'] == before['frame']
            assert game.observe()['truncation_reason'] == 'interface_failure'
        report['fixtures'].append('None/empty SDK responses preserve last good frame and truncate')
    finally:
        game.close()

    for options, actions, reason in (
        ({'max_actions': 2}, ['noop', 'reset'], 'action_limit'),
        ({'max_invalid_actions': 2}, ['click', 'click'], 'invalid_action_limit'),
    ):
        game = make_game(**options)
        try:
            game.reset(0)
            for action in actions:
                try:
                    game.step(action)
                except ValueError:
                    pass
            assert game.done and game.observe()['truncation_reason'] == reason
            assert game.info()['episode_actions'] == 2 and not game.observe()['won']
        finally:
            game.close()
    game = make_game(max_seconds=1)
    try:
        game.reset(0)
        # Advance only the adaptor clock; do not sleep or alter the live game.
        with patch('game.time.monotonic', return_value=game._started + 2):
            assert game.done and game.observe()['truncation_reason'] == 'time_limit'
    finally:
        game.close()
    report['budgets'] = 'action, invalid request, and wall time limits passed'
    report['network'] = 'all tests ran with requests.Session.request blocked'
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('PASS: real SDK integration and labeled response fixtures')


if __name__ == '__main__':
    with patch('requests.Session.request', side_effect=AssertionError('Network forbidden during verification')):
        main()
