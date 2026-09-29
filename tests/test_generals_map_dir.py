"""Real ladder maps for the generals env (env.generals_env.load_map_dir, GeneralsVecEnv
map_dir): standard 1v1 replay maps are used as they are; maps with features / modes the
rules lack (swamps, lookouts, game modifiers ...), other player counts and broken
garrisons are left out."""

import json

import pytest

from env.generals_env import load_map_dir, map_entry_of
from env.vec_generals import GeneralsVecEnv


def _replay(**kw):
    r = {"id": "x", "mapWidth": 18, "mapHeight": 19, "generals": [5, 300],
         "cities": [40, 41], "cityArmies": [45, 42], "mountains": [7, 8, 9, 100]}
    r.update(kw)
    return r


def test_map_entry_filters():
    ok, why = map_entry_of(_replay())
    assert why == "" and (ok["map_w"], ok["map_h"], ok["generals"], ok["city_armies"]) == (18, 19, [5, 300], [45, 42])
    for bad, reason in [(dict(swamps=[3]), "unsupported features"), (dict(modifiers=[10]), "unsupported features"),
                        (dict(lookouts=[4]), "unsupported features"), (dict(generals=[1, 2, 3]), "not 1v1"),
                        (dict(cityArmies=[-5, 40]), "invalid"), (dict(mountains=[10_000]), "invalid"),
                        (dict(mapWidth=26), "too big")]:
        assert map_entry_of(_replay(**bad)) == (None, reason), bad


def test_vec_env_plays_the_real_maps(tmp_path):
    json.dump(_replay(), open(tmp_path / "a.json", "w"))
    json.dump(_replay(swamps=[3]), open(tmp_path / "b.json", "w"))
    json.dump(_replay(generals=[1, 2, 3]), open(tmp_path / "c.json", "w"))
    assert [m["replay_id"] for m in load_map_dir(str(tmp_path))] == ["a"]
    env = GeneralsVecEnv(n_envs=2, map_dir=str(tmp_path), seed=0)
    for g in env.envs:
        # the replay map sits in the top-left corner of the 25 x 25 grid
        assert (g.real_w, g.real_h) == (18, 19)
        assert sorted(g.mountains) == sorted(i // 18 * 25 + i % 18 for i in (7, 8, 9, 100))
    obs = env.reset()
    for _ in range(3):
        obs, *_ = env.step(env.random_actions(obs))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no usable"):
        GeneralsVecEnv(n_envs=1, map_dir=str(tmp_path / "empty"))
