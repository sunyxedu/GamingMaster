"""cfg.memory_vision: the generals sighted-mountain memory as ONE recorded
observation setting (the audit's test_memory_vision_is_one_recorded_setting,
tests/test_audit_generals_leaks.py).

  * a checkpoint written before the field existed resolves to the
    observation it was trained on (model.config.legacy_memory_vision);
  * recorded games get it causally and exactly from their pure-fog frames
    (model.games.generals.apply_memory_vision).
"""
from __future__ import annotations

import os

import pytest
import torch

from env.vec_generals import (
    GeneralsVecEnv,
    map_from_replay,
    replay_actions_by_step,
    replay_tiebreak_by_step,
)
from model.config import checkpoint_config, checkpoint_config_dict, legacy_memory_vision
from model.games.generals import apply_memory_vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "generals", "TyvMRVMys.gior")

#: a config dict as written from 2026-08-11 on (carries joint_decoder), and an older one
AFTER = {"d_model": 64, "n_heads": 4, "joint_decoder": False}
BEFORE = {"d_model": 64, "n_heads": 4}


def _replay_grids(memory_vision: bool) -> torch.Tensor:
    """[2 seats, T, 12, H, W]: the fixture game replayed through GeneralsVecEnv
    (its pure-fog frames are the IL records', tests/test_vec_game_golden.py)."""
    from data_processing.download.deserializer import deserialize_replay
    with open(FIXTURE, "rb") as fh:
        replay = deserialize_replay(fh.read())
    vec = GeneralsVecEnv(1, max_turns=10 ** 6, memory_vision=memory_vision)
    obs = vec.reset(maps=[map_from_replay(replay)])
    by_step = replay_actions_by_step(replay, vec.grid_w)
    tiebreak = replay_tiebreak_by_step(replay)
    end = max(m["turn"] for m in replay["moves"]) + 2
    grids, t = [], 0
    while bool(obs.active.all()) and t < end:
        grids.append(obs.obs["grid_seq"].clone())
        acts = {"source_idx": torch.zeros(2, dtype=torch.long),
                "direction": torch.zeros(2, dtype=torch.long),
                "split": torch.zeros(2, dtype=torch.long),
                "passed": torch.ones(2, dtype=torch.bool)}
        for p, a in by_step.get(t + 1, {}).items():
            for k, v in a.items():
                acts[k][p] = v
        obs, _, _, _ = vec.step(acts, tiebreak_first=[tiebreak.get(t + 1, 0)])
        t += 1
    return torch.stack(grids, dim=1)


@pytest.fixture(scope="module")
def grids():
    return {False: _replay_grids(False), True: _replay_grids(True)}


@pytest.mark.parametrize("ckpt,want", [
    # training/train_rl.py: rl_checkpoint_iter*.pt, rl_ema_iter*.pt, distill seeds
    ({"iteration": 900, "config": AFTER, "opponent_pool": [], "wins": {}}, True),
    ({"iteration": 900, "config": AFTER, "ema": True}, True),
    ({"iteration": 2720, "config": AFTER, "distilled_from": "rl_ema_iter2720.pt"}, True),
    # ... older than the memory (no joint_decoder field)
    ({"iteration": 900, "config": BEFORE, "opponent_pool": []}, False),
    ({"iteration": 900, "config": BEFORE, "ema": True}, False),
    # IL: train_il, train_il_game, single-game exports, train_multi_game
    ({"epoch": 3, "step": 0, "config": AFTER, "loss": 1.0}, False),
    ({"game": "generals", "epoch": 3, "config": AFTER}, False),
    ({"game": "generals", "config": AFTER, "source": {"step": 7}}, False),
    ({"games": ["generals"], "configs": {"generals": AFTER}}, False),
    # train_rl_joint: GeneralsVecEnv's default (off) unless --env_kw switched it on
    ({"games": ["generals"], "configs": {"generals": AFTER}, "stage": "rl_joint",
      "args": {"env_kw": ["generals:max_turns=500"]}}, False),
    ({"games": ["generals"], "configs": {"generals": AFTER}, "stage": "rl_joint",
      "args": {"env_kw": ["generals:memory_vision=True"]}}, True),
    ({"games": ["generals"], "configs": {"generals": AFTER}, "stage": "rl_joint",
      "args": {"env_kw": ["generals:memory_vision=True", "generals:memory_vision=0"]}}, False),
    ({"games": ["generals"], "configs": {"generals": AFTER}, "stage": "rl_joint",
      "args": {"env_kw": ["othello:memory_vision=True"]}}, False),
])
def test_legacy_memory_vision(ckpt, want):
    assert legacy_memory_vision(ckpt) is want
    if "config" in ckpt:
        assert checkpoint_config(ckpt).memory_vision is want
    else:
        assert checkpoint_config_dict(ckpt, "generals")["memory_vision"] is want


def test_a_recorded_setting_wins_over_the_legacy_rule():
    for mv in (False, True):
        assert checkpoint_config({"iteration": 5,
                                  "config": {**AFTER, "memory_vision": mv}}).memory_vision is mv
        assert checkpoint_config({"epoch": 5,
                                  "config": {**AFTER, "memory_vision": mv}}).memory_vision is mv
        multi = {"games": ["generals"], "configs": {"generals": {**AFTER, "memory_vision": mv}},
                 "stage": "rl_joint", "args": {"env_kw": ["generals:memory_vision=True"]}}
        assert checkpoint_config_dict(multi, "generals")["memory_vision"] is mv
    # only a generals network reads the rule; other games keep the plain legacy default
    multi = {"games": ["generals", "othello"], "stage": "rl_joint",
             "configs": {"generals": dict(AFTER), "othello": dict(AFTER)},
             "args": {"env_kw": ["generals:memory_vision=True"]}}
    assert checkpoint_config_dict(multi, "generals")["memory_vision"] is True
    assert checkpoint_config_dict(multi, "othello")["memory_vision"] is False


def test_apply_memory_vision_equals_the_envs_memory(grids):
    fog, mem = grids[False], grids[True]
    assert not torch.equal(fog, mem)                  # the memory shows in this game
    assert torch.equal(apply_memory_vision(fog), mem)
    # causal: a prefix gives the prefix
    t = fog.shape[1] // 3
    assert torch.equal(apply_memory_vision(fog[:, :t]), mem[:, :t])
    # unbatched [T, C, H, W]; zero tail padding stays zero; idempotent
    assert torch.equal(apply_memory_vision(fog[1]), mem[1])
    pad = torch.zeros_like(fog[:, :5])
    assert torch.equal(apply_memory_vision(torch.cat([fog, pad], dim=1))[:, -5:], pad)
    assert torch.equal(apply_memory_vision(mem), mem)
