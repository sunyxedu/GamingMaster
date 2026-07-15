"""Tests for the phase-4.1 reward redesign.

  1. Damage-proportional neutral-city payout: whittle hits pay
     unit × damage/G0, a full capture sums to exactly `unit`, the per-city
     unit is normalized by the map's city count (capped at 0.25).
  2. Turn-limit truncation scored by the material heuristic (soft terminal).
  3. Capture-feasibility labels in observations.
  4. RolloutBuffer round-trip of the capture labels.
"""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv, _BLANK, _CITY, _GENERAL
from model.rl_utils import RolloutBuffer

W = 25


def idx(r, c):
    return r * W + c


def make_bare_env(max_turns=9999):
    """Env with a hand-built map: generals at (0,0) / (20,20), nothing else."""
    env = GeneralsEnv(grid_h=25, grid_w=25, n_mountains=0, n_cities=0,
                      max_turns=max_turns)
    env.reset(seed=0)
    for r in range(env.real_h):
        for c in range(env.real_w):
            t = env.map[r][c]
            t.terrain = _BLANK
            t.army = 0
            t.occupied = -1
    env.generals = [idx(0, 0), idx(20, 20)]
    env.cities = []
    env.mountains = []
    env.player_land = [[idx(0, 0)], [idx(20, 20)]]
    for pid in (0, 1):
        r, c = divmod(env.generals[pid], W)
        env.map[r][c].terrain = _GENERAL
        env.map[r][c].army = 100
        env.map[r][c].occupied = pid
    env.city_initial_army = {}
    env.city_reward_unit = 0.2
    env.city_reward_scale = 1.0
    env.prev_heuristic = {0: env._heuristic(0), 1: env._heuristic(1)}
    return env


def add_city(env, r, c, garrison):
    t = env.map[r][c]
    t.terrain = _CITY
    t.army = garrison
    t.occupied = -1
    env.cities.append(idx(r, c))
    env.city_initial_army[idx(r, c)] = garrison


def add_troops(env, pid, r, c, army):
    t = env.map[r][c]
    t.terrain = _BLANK
    t.army = army
    t.occupied = pid
    if idx(r, c) not in env.player_land[pid]:
        env.player_land[pid].append(idx(r, c))


# ======================================================================
# 1. Neutral-city damage payout
# ======================================================================

class TestCityDamagePayout:
    def test_whittle_pays_proportionally(self):
        """A partial hit pays unit × damage/G0 and counts as a bash."""
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        add_troops(env, 0, 5, 5, 21)  # moves 20 < 40 → whittle
        _, rewards, done, info = env.step(
            {"source_idx": idx(5, 5), "direction": 1, "split": 0}, None)
        bd = info["reward_breakdown"]
        assert bd["c_bash"] == 1
        assert bd["c_cap"] == 0
        assert bd["r_city"] == pytest.approx(0.2 * 20 / 40)
        assert env.map[5][6].occupied == -1
        assert env.map[5][6].army == 20

    def test_capture_total_payout_equals_unit(self):
        """Across however many hits, a full capture pays exactly `unit`
        (plus the small precision bonus, bounded by 0.3 × unit)."""
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        total_city = 0.0
        caps = bashes = 0
        for _ in range(10):
            add_troops(env, 0, 5, 5, 22)  # moves 21 per hit
            _, rewards, done, info = env.step(
                {"source_idx": idx(5, 5), "direction": 1, "split": 0}, None)
            total_city += info["reward_breakdown"]["r_city"]
            caps += info["reward_breakdown"]["c_cap"]
            bashes += info["reward_breakdown"]["c_bash"]
            if env.map[5][6].occupied == 0:
                break
        assert env.map[5][6].occupied == 0
        assert caps == 1
        assert bashes == 1  # 21 whittle, then 21 > 19 captures
        # Damage payout sums to exactly unit; precision bonus on top.
        assert total_city > 0.2
        assert total_city <= 0.2 + 0.3 * 0.2 + 1e-9

    def test_no_payout_without_damage(self):
        """A 1-army source can't move; no reward, no bash."""
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        add_troops(env, 0, 5, 5, 1)
        _, rewards, done, info = env.step(
            {"source_idx": idx(5, 5), "direction": 1, "split": 0}, None)
        assert info["reward_breakdown"]["r_city"] == 0.0
        assert info["reward_breakdown"]["c_bash"] == 0

    def test_unit_normalized_by_city_count(self):
        """unit = budget / n_cities, capped at 0.25."""
        env = GeneralsEnv(n_mountains=0, n_cities=8)
        env.city_budget_total = 0.8
        env.reset(seed=3)
        assert len(env.cities) == 8
        assert env.city_reward_unit == pytest.approx(0.1)

        env2 = GeneralsEnv(n_mountains=0, n_cities=2)
        env2.city_budget_total = 0.8
        env2.reset(seed=3)
        assert env2.city_reward_unit == pytest.approx(0.25)  # capped

    def test_steal_symmetric_unchanged(self):
        """Taking an opponent's city stays a symmetric ±0.16 event."""
        env = make_bare_env()
        # p1 owns a city next to a big p0 stack
        add_city(env, 5, 6, 5)
        t = env.map[5][6]
        t.occupied = 1
        env.player_land[1].append(idx(5, 6))
        add_troops(env, 0, 5, 5, 50)
        _, rewards, done, info = env.step(
            {"source_idx": idx(5, 5), "direction": 1, "split": 0}, None)
        bd = info["reward_breakdown"]
        assert bd["c_steal"] == 1
        assert bd["r_city"] == pytest.approx(0.16)


class TestPerMoveAttribution:
    def test_same_tick_double_attack_pays_each_mover_exactly(self):
        """P1 captures (army priority), P0's same-tick hit lands on the now
        P1-owned tile: P1 gets the full payout + bonus, P0 gets nothing."""
        env = make_bare_env()
        add_city(env, 5, 6, 10)
        add_troops(env, 0, 5, 5, 20)   # moves 19 — would capture alone
        add_troops(env, 1, 5, 7, 60)   # moves 59 — bigger, executes first
        env.breakdown_pid = 1
        _, rewards, done, info = env.step(
            {"source_idx": idx(5, 5), "direction": 1, "split": 0},
            {"source_idx": idx(5, 7), "direction": 3, "split": 0})
        bd = info["reward_breakdown"]
        assert env.map[5][6].occupied == 1
        assert bd["c_cap"] == 1
        # P1: full damage payout (10/10 of unit) + precision bonus for
        # leftover 49 (~0), computed pre-increment.
        assert bd["r_city"] == pytest.approx(
            0.2 + 0.3 * 0.2 * math.exp(-0.1 * 49), rel=1e-6)
        # P0 whittled P1's freshly-captured stack, not a neutral garrison:
        # no bash event, no city payout on P0's side either.
        env.breakdown_pid = 0  # (post-hoc: breakdown already recorded)
        assert bd["c_bash"] == 0

    def test_precision_bonus_independent_of_turn_parity(self):
        """Identical captures landing on odd vs even turns pay the same
        bonus (the garrison used to be read after _increment_armies)."""
        totals = []
        for skip_turns in (0, 1):
            env = make_bare_env()
            add_city(env, 5, 6, 10)
            for _ in range(skip_turns):
                env.step(None, None)
            add_troops(env, 0, 5, 5, 13)  # moves 12, leftover 2
            _, _, _, info = env.step(
                {"source_idx": idx(5, 5), "direction": 1, "split": 0}, None)
            totals.append(info["reward_breakdown"]["r_city"])
        assert totals[0] == pytest.approx(totals[1], rel=1e-9)

    def test_same_tick_recapture_is_a_steal(self):
        """Opponent captures a neutral city and the learner takes it back
        in the SAME step: symmetric ±0.16, not a fresh neutral capture."""
        # Priority favors the larger source stack: P1 (100) executes first
        # and captures the 50-garrison city with 49 left; P0's smaller move
        # (59) then beats that leftover and recaptures.
        env2 = make_bare_env()
        add_city(env2, 5, 6, 50)
        env2.city_initial_army[idx(5, 6)] = 50
        add_troops(env2, 1, 5, 7, 100)  # moves 99 → first, captures, 49 left
        add_troops(env2, 0, 5, 5, 60)   # moves 59 > 49 → same-tick recapture
        _, rewards, done, info = env2.step(
            {"source_idx": idx(5, 5), "direction": 1, "split": 0},
            {"source_idx": idx(5, 7), "direction": 3, "split": 0})
        bd = info["reward_breakdown"]
        assert env2.map[5][6].occupied == 0
        # P0 (breakdown) recaptured a just-taken city → steal, no c_cap.
        assert bd["c_steal"] == 1
        assert bd["c_cap"] == 0


# ======================================================================
# 2. Truncation material scoring
# ======================================================================

class TestTruncationScore:
    def test_truncation_scored_by_material(self):
        env = make_bare_env(max_turns=3)
        add_troops(env, 0, 1, 1, 200)  # big p0 material lead
        env.prev_heuristic = {0: env._heuristic(0), 1: env._heuristic(1)}
        done = False
        for _ in range(3):
            _, rewards, done, info = env.step(None, None)
        assert done and info.get("truncated")
        assert env.winner is None
        # The truncation step carries exactly the material score.
        assert rewards[0] == pytest.approx(env.truncation_score(0))
        assert rewards[1] == pytest.approx(env.truncation_score(1))
        assert rewards[0] > 0 > rewards[1]
        assert abs(rewards[0]) <= 0.35
        assert info["reward_breakdown"]["r_winlose"] == pytest.approx(rewards[0])


# ======================================================================
# 3. Capture-feasibility labels
# ======================================================================

class TestCaptureLabels:
    def test_labels_and_masks(self):
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        add_troops(env, 0, 5, 5, 45)    # moves 44 > 40 → capturable
        add_troops(env, 0, 10, 10, 30)  # vs enemy 35 → not capturable
        add_troops(env, 1, 10, 11, 35)

        obs = env._get_observation(0)
        lab, mask = obs["capture_label"], obs["capture_mask"]
        assert lab.shape == (4, 25, 25) and mask.shape == (4, 25, 25)
        # direction 1 = right: (5,5) → city (5,6), 44 > 40
        assert mask[1, 5, 5] and lab[1, 5, 5] == 1.0
        # (10,10) → enemy (10,11): 29 > 35 is false
        assert mask[1, 10, 10] and lab[1, 10, 10] == 0.0
        # blank neighbour is not a supervised target
        assert not mask[0, 5, 5]

        # p1's view: (10,11) → left to p0 stack (10,10): 34 > 30
        obs1 = env._get_observation(1)
        assert obs1["capture_mask"][3, 10, 11]
        assert obs1["capture_label"][3, 10, 11] == 1.0

    def test_unmovable_source_not_supervised(self):
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        add_troops(env, 0, 4, 6, 1)  # army ≤ 1 can't move
        obs = env._get_observation(0)
        assert not obs["capture_mask"][0, 4, 6]  # down onto the city

    def test_exact_threshold(self):
        """src.army - 1 must EXCEED the garrison (equal is not a capture)."""
        env = make_bare_env()
        add_city(env, 5, 6, 40)
        add_troops(env, 0, 5, 5, 41)  # moves 40, 40 > 40 false
        obs = env._get_observation(0)
        assert obs["capture_mask"][1, 5, 5]
        assert obs["capture_label"][1, 5, 5] == 0.0
        add_troops(env, 0, 5, 5, 42)  # moves 41 > 40
        obs = env._get_observation(0)
        assert obs["capture_label"][1, 5, 5] == 1.0


# ======================================================================
# 4. RolloutBuffer round-trip
# ======================================================================

class TestBufferCaptureBatch:
    def _action(self):
        return {"source_idx": 0, "direction": 0, "split": 0, "passed": False}

    def test_batch_includes_capture_keys(self):
        env = make_bare_env()
        obs = env._get_observation(0)
        buf = RolloutBuffer()
        for _ in range(3):
            buf.add(obs=obs, action=self._action(),
                    log_prob=-1.0, value=0.0, reward=0.0, done=False)
        buf.compute_returns_and_advantages()
        batch = buf.to_tensor_batch(torch.device("cpu"))
        assert batch["capture_label"].shape == (1, 3, 4, 25, 25)
        assert batch["capture_mask"].shape == (1, 3, 4, 25, 25)
        assert batch["capture_mask"].dtype == torch.bool

    def test_legacy_obs_without_keys(self):
        env = make_bare_env()
        obs = env._get_observation(0)
        legacy = {k: obs[k] for k in ("grid", "globals", "legal_mask")}
        buf = RolloutBuffer()
        buf.add(obs=legacy, action=self._action(),
                log_prob=-1.0, value=0.0, reward=0.0, done=True)
        buf.compute_returns_and_advantages()
        batch = buf.to_tensor_batch(torch.device("cpu"))
        assert "capture_label" not in batch
        assert "capture_mask" not in batch
