"""GeneralsEnv._get_observation (whole-grid numpy) == the original per-tile
implementation, kept here verbatim as the oracle: every returned tensor bit for
bit (dtype included) and the sighted-mountain memory it updates — on random
full-size and non-square maps, replay maps smaller than the grid, memory vision
on and off, turn globals, and the terminal state after a general capture."""

from __future__ import annotations

import copy
import math
import os
import random

import pytest
import torch

from env.generals_env import _CITY, _DIR_DC, _DIR_DR, _MOUNTAIN, GeneralsEnv


def reference_observation(self, player_id):
    """The per-tile GeneralsEnv._get_observation before vectorisation."""
    H, W = self.grid_h, self.grid_w  # 25, 25 (padded)
    rH, rW = self.real_h, self.real_w  # real map bounds

    vision = [[0] * W for _ in range(H)]
    for idx in self.player_land[player_id]:
        r, c = divmod(idx, W)
        vision[r][c] = 1   # owned
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rH and 0 <= nc < rW and vision[nr][nc] != 1:
                    vision[nr][nc] = 2  # visible

    grid = torch.zeros(12, H, W, dtype=torch.float32)
    memory = (self.seen_mountains[player_id]
              if getattr(self, "memory_vision", True) else None)
    for r in range(rH):
        for c in range(rW):
            tile = self.map[r][c]
            v = vision[r][c]
            state = self._tile_to_state(tile, v, player_id)
            if memory is not None:
                idx = r * W + c
                if v >= 1 and tile.terrain == _MOUNTAIN:
                    memory.add(idx)
                elif v == 0 and idx in memory:
                    state = 4
            if 0 <= state <= 10:
                grid[state, r, c] = 1.0
            if v >= 1 and tile.terrain != _MOUNTAIN:
                grid[11, r, c] = math.log1p(tile.army) / 10.0

    stats = self._player_stats(player_id)
    opp_stats = self._player_stats(1 - player_id)
    glb = [
        math.log1p(stats["army"]) / 10.0,
        math.log1p(stats["land"]) / 10.0,
        math.log1p(opp_stats["army"]) / 10.0,
        math.log1p(opp_stats["land"]) / 10.0,
    ]
    if self.turn_globals:
        glb += [
            (self.turn % 50) / 50.0,
            float(self.turn % 2),
            self.turn / 500.0,
        ]
    globals_vec = torch.tensor(glb, dtype=torch.float32)

    legal = torch.zeros(1, H, W, dtype=torch.bool)
    for idx in self.player_land[player_id]:
        r, c = divmod(idx, W)
        if self.map[r][c].army > 1:
            legal[0, r, c] = True

    cap_label = torch.zeros(4, H, W, dtype=torch.float32)
    cap_mask = torch.zeros(4, H, W, dtype=torch.bool)
    for idx in self.player_land[player_id]:
        r, c = divmod(idx, W)
        a = self.map[r][c].army
        if a <= 1:
            continue
        for d in range(4):
            nr, nc = r + _DIR_DR[d], c + _DIR_DC[d]
            if not (0 <= nr < rH and 0 <= nc < rW):
                continue
            t = self.map[nr][nc]
            if t.terrain == _MOUNTAIN:
                continue
            defended = (t.occupied == 1 - player_id
                        or (t.occupied == -1 and t.terrain == _CITY))
            if not defended:
                continue
            cap_mask[d, r, c] = True
            if a - 1 > t.army:
                cap_label[d, r, c] = 1.0

    return {"grid": grid, "globals": globals_vec, "legal_mask": legal,
            "capture_label": cap_label, "capture_mask": cap_mask,
            "dir_mask": self.dir_valid}


def _assert_same(got, want, ctx):
    assert set(got) == set(want), ctx
    for k in want:
        a, b = got[k], want[k]
        assert a.dtype == b.dtype and a.shape == b.shape, (ctx, k)
        assert torch.equal(a, b), (ctx, k)


def _check(env, obs, mem_before, ctx):
    """``obs`` (built by the env from memory ``mem_before``) == the oracle."""
    mem_after = copy.deepcopy(env.seen_mountains)
    env.seen_mountains = copy.deepcopy(mem_before)
    for p in (0, 1):
        _assert_same(obs[p], reference_observation(env, p), (ctx, p))
    assert env.seen_mountains == mem_after, ctx


def _random_action(obs, env, rng):
    legal = obs["legal_mask"][0].nonzero().tolist()
    if not legal or rng.random() < 0.1:
        return None
    r, c = rng.choice(legal)
    return {"source_idx": r * env.grid_w + c, "direction": rng.randrange(4),
            "split": rng.randrange(2)}


def _play(env, turns, seed, map_data=None):
    rng = random.Random(seed)
    obs = env.reset(seed=seed, map_data=map_data)
    _check(env, obs, [set(), set()], "reset")          # reset clears the memory
    for t in range(turns):
        mem_before = copy.deepcopy(env.seen_mountains)
        obs, _, done, _ = env.step(_random_action(obs[0], env, rng),
                                   _random_action(obs[1], env, rng))
        _check(env, obs, mem_before, t)
        if done:
            break
    return obs


@pytest.mark.parametrize("memory,turn_globals", [(True, False), (False, True)])
def test_random_full_maps(memory, turn_globals):
    env = GeneralsEnv(n_mountains=40, n_cities=8, max_turns=10_000)
    env.memory_vision = memory
    env.turn_globals = turn_globals
    _play(env, 250, seed=1)


def test_non_square_grid():
    env = GeneralsEnv(grid_h=12, grid_w=17, n_mountains=25, n_cities=6)
    _play(env, 150, seed=2)


def test_crowded_small_map_covers_contact_captures_and_memory():
    """8×8: enemy tiles in sight (states 8-10), owned cities, capture labels
    and remembered mountains all occur (asserted, so the oracle comparison is
    not vacuous)."""
    env = GeneralsEnv(grid_h=8, grid_w=8, n_mountains=6, n_cities=3)
    rng = random.Random(1)
    obs = env.reset(seed=1)
    seen = {"enemy": 0, "own_city": 0, "cap_label": 0, "remembered": 0}
    for t in range(300):
        mem_before = copy.deepcopy(env.seen_mountains)
        obs, _, done, _ = env.step(_random_action(obs[0], env, rng),
                                   _random_action(obs[1], env, rng))
        _check(env, obs, mem_before, t)
        for p in (0, 1):
            g = obs[p]["grid"]
            seen["enemy"] += int(g[8:11].sum() > 0)
            seen["own_city"] += int(g[6].sum() > 0)
            seen["cap_label"] += int(obs[p]["capture_label"].sum() > 0)
            seen["remembered"] += int((g[4] * (g[5:11].sum(0) == 0)).sum() > 0)
        if done:
            break
    assert all(v > 0 for v in seen.values()), seen


def test_replay_map_smaller_than_the_grid_and_injected_memory():
    from data_processing.download.deserializer import deserialize_replay
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "generals", "TyvMRVMys.gior")
    with open(fixture, "rb") as fh:
        replay = deserialize_replay(fh.read())
    env = GeneralsEnv()
    map_data = {"map_w": replay["mapWidth"], "map_h": replay["mapHeight"],
                "generals": replay["generals"], "cities": replay["cities"],
                "city_armies": replay["cityArmies"], "mountains": replay["mountains"]}
    _play(env, 200, seed=3, map_data=map_data)
    assert (env.real_h, env.real_w) == (19, 18)
    # memory entries off the map (or off the grid) are ignored by both
    env.seen_mountains[0].update({24, 10**6, 19 * 25 + 3})
    mem = copy.deepcopy(env.seen_mountains)
    _check(env, env._observations(), mem, "injected")


def test_after_a_general_capture():
    env = GeneralsEnv(n_mountains=10, n_cities=5)
    _play(env, 120, seed=4)
    g1 = env.generals[1]
    r, c = divmod(g1, env.grid_w)
    if g1 in env.player_land[1]:
        env.player_land[1].remove(g1)
    env.map[r][c].occupied = 0
    env.player_land[0].append(g1)
    env._apply_general_capture(0)
    assert env.map[r][c].terrain == _CITY and not env.player_land[1]
    mem = copy.deepcopy(env.seen_mountains)
    _check(env, env._observations(), mem, "captured")


def test_negative_army_on_a_visible_tile_still_raises():
    env = GeneralsEnv()
    env.reset(seed=0)
    g = env.generals[0]
    env.map[g // env.grid_w][g % env.grid_w].army = -3
    with pytest.raises(ValueError, match="math domain error"):
        env._get_observation(0)
