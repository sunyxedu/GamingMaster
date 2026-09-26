"""cfg.memory_vision: the generals sighted-mountain memory as ONE recorded
observation setting (the audit's test_memory_vision_is_one_recorded_setting,
tests/test_audit_generals_leaks.py).

  * off by default everywhere — the IL records, GeneralsEnv, GeneralsVecEnv,
    ModelConfig;
  * a checkpoint written before the field existed resolves to the
    observation it was trained on (model.config.legacy_memory_vision);
  * every env builder takes it from the network: training/train_rl.py's
    collectors (per seat) and worker pool, train_rl_joint's GeneralsSpec, the
    Elo ladder / depth probe / distillation loader;
  * recorded games get it causally and exactly from their pure-fog frames
    (model.games.generals.apply_memory_vision), so the IL batches of a
    memory_vision network match what its env shows.
"""
from __future__ import annotations

import os
import types

import pytest
import torch

from env.generals_env import GeneralsEnv
from env.vec_generals import (
    GeneralsVecEnv,
    map_from_replay,
    replay_actions_by_step,
    replay_tiebreak_by_step,
)
from model.config import (
    ModelConfig,
    checkpoint_config,
    checkpoint_config_dict,
    legacy_memory_vision,
    upgrade_config_dict,
)
from model.games.generals import apply_memory_vision
from model.network import GeneralsNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "generals", "TyvMRVMys.gior")
TINY = dict(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)
CPU = torch.device("cpu")

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


# ---------------------------------------------------------------------------
# one default, and the legacy rule
# ---------------------------------------------------------------------------

def test_default_is_off_everywhere():
    assert GeneralsEnv().memory_vision is False
    assert GeneralsVecEnv(1).memory_vision is False
    assert GeneralsVecEnv(1).envs[0].memory_vision is False
    assert ModelConfig().memory_vision is False
    assert upgrade_config_dict({})["memory_vision"] is False


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


def test_single_game_exports_record_the_resolved_setting():
    from training.train_multi_game import game_checkpoint
    ck = {"games": ["generals", "othello"], "stage": "rl_joint",
          "configs": {"generals": dict(AFTER), "othello": dict(AFTER)}, "model_state": {},
          "args": {"env_kw": ["generals:memory_vision=True"]}}
    assert game_checkpoint(ck, "generals")["config"]["memory_vision"] is True
    assert "memory_vision" not in game_checkpoint(ck, "othello")["config"]
    ck["args"]["env_kw"] = []
    assert game_checkpoint(ck, "generals")["config"]["memory_vision"] is False


def test_load_multi_resolves_a_train_rl_joint_env_kw(tmp_path):
    from model.games import build_game_network
    from model.multi_game import MultiGameNetwork
    from training.eval_multi_game import load_multi
    net = build_game_network("generals", **TINY)
    saved = {k: v for k, v in net.cfg.__dict__.items() if k != "memory_vision"}
    sd = MultiGameNetwork({"generals": net}).state_dict()
    for env_kw, want in ((["generals:memory_vision=True"], True), ([], False)):
        path = tmp_path / f"rl_{want}.pt"
        torch.save({"games": ["generals"], "configs": {"generals": saved}, "model_state": sd,
                    "stage": "rl_joint", "args": {"env_kw": env_kw}}, path)
        assert load_multi(str(path), CPU)["generals"].cfg.memory_vision is want


def test_ladder_loader_resolves_legacy_train_rl_checkpoints(tmp_path):
    """scripts/elo_ladder.load_net (also depth_probe's and the distillation
    teacher's loader)."""
    from scripts.elo_ladder import load_net
    from training.train_rl import net_memory_vision
    cfg = ModelConfig(**TINY)
    sd = GeneralsNetwork(cfg).state_dict()
    after = {k: v for k, v in cfg.__dict__.items() if k != "memory_vision"}
    before = {k: v for k, v in after.items() if k != "joint_decoder"}
    for name, ck, want in (
            ("rl_ema_new", {"iteration": 100, "ema": True, "config": after}, True),
            ("rl_ema_old", {"iteration": 100, "ema": True, "config": before}, False),
            ("il", {"epoch": 3, "config": after}, False),
            ("recorded", {"iteration": 100, "config": {**after, "memory_vision": False}}, False)):
        torch.save({**ck, "model_state": sd}, tmp_path / f"{name}.pt")
        net, c = load_net(str(tmp_path / f"{name}.pt"), CPU)
        assert c.memory_vision is want and net_memory_vision(net) is want, name


# ---------------------------------------------------------------------------
# recorded games: the memory derived from the frames
# ---------------------------------------------------------------------------

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


def test_il_batches_follow_the_network(grids):
    """train_il_game's generals obs hook (train_il_game / train_multi_game /
    train_rl_joint's replay stream): a memory_vision network's IL frames
    are exactly what its env would show; any other network's are the
    records' own."""
    from training.train_il_game import OBS_HOOKS
    fog = grids[False]
    B, T = fog.shape[:2]
    obs = {"grid_seq": fog, "globals_seq": torch.zeros(B, T, 4),
           "legal_source_mask_seq": torch.zeros(B, T, 1, 25, 25, dtype=torch.bool)}
    assert torch.equal(OBS_HOOKS["generals"](obs, ModelConfig(memory_vision=True))["grid_seq"],
                       grids[True])
    assert OBS_HOOKS["generals"](obs, ModelConfig())["grid_seq"] is fog
    both = OBS_HOOKS["generals"](obs, ModelConfig(memory_vision=True, turn_globals=True,
                                                  global_features=7))
    assert both["globals_seq"].shape == (B, T, 7) and torch.equal(both["grid_seq"], grids[True])


def test_train_rl_il_auxiliary_batches_follow_the_network(grids, monkeypatch):
    import model.games.generals as generals_adapter
    from model.losses import ImitationLoss
    from training.train_rl import il_auxiliary_update
    fed = []
    real = generals_adapter.apply_memory_vision
    monkeypatch.setattr(generals_adapter, "apply_memory_vision",
                        lambda g: fed.append(g) or real(g))
    fog = grids[False][:1, :10]
    T = fog.shape[1]
    zeros = torch.zeros(1, T, 4, dtype=torch.long)
    batch = {"obs": {"grid_seq": fog, "globals_seq": torch.zeros(1, T, 4),
                     "legal_source_mask_seq": fog[:, :, 5:8].sum(2, keepdim=True) > 0},
             "tgt": {"count": torch.zeros(1, T, dtype=torch.long), "src_y": zeros,
                     "src_x": zeros, "dir_id": zeros, "ratio_id": zeros,
                     "value": torch.zeros(1, T)},
             "meta": {"lengths": [T]}}
    for mv in (False, True):
        fed.clear()
        net = GeneralsNetwork(ModelConfig(memory_vision=mv, **TINY))
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        il_auxiliary_update(net, batch, ImitationLoss(grid_w=25), opt, CPU,
                            il_weight=0.1, act_steps=1)
        assert len(fed) == int(mv)


# ---------------------------------------------------------------------------
# env builders
# ---------------------------------------------------------------------------

def test_train_rl_collectors_take_the_setting_from_the_networks():
    """collect_selfplay_episode: each seat its own network's setting (a
    sparring / eval / ladder opponent may come from the other lineage);
    collect_selfplay_episodes_vec: the model's, on every local env."""
    from training.train_rl import collect_selfplay_episode, collect_selfplay_episodes_vec
    torch.manual_seed(0)
    on = GeneralsNetwork(ModelConfig(memory_vision=True, **TINY))
    off = GeneralsNetwork(ModelConfig(**TINY))
    env = GeneralsEnv(max_turns=4)
    kw = dict(max_steps=4, act_steps=1, collect_buffers=False)
    for lp, want in ((0, (True, False)), (1, (False, True))):
        collect_selfplay_episode(on, off, env, CPU, learner_pid=lp, **kw)
        assert env.memory_vision == want
        assert [env.memory_vision_for(p) for p in (0, 1)] == list(want)
    collect_selfplay_episode(on, on, env, CPU, **kw)
    assert env.memory_vision is True
    collect_selfplay_episode(off, off, env, CPU, **kw)
    assert env.memory_vision is False
    # a network without a config keeps the env's setting for its seat
    bare = types.SimpleNamespace(eval=lambda: None)
    from training.train_rl import net_memory_vision
    assert net_memory_vision(bare, True) is True and net_memory_vision(bare) is False
    envs = [GeneralsEnv(max_turns=4) for _ in range(2)]
    collect_selfplay_episodes_vec(on, envs, CPU, max_steps=4, act_steps=1)
    assert all(e.memory_vision is True for e in envs)


def test_vec_collector_refuses_a_worker_pool_with_another_setting():
    from env.vec_workers import EnvWorkerPool
    from training.train_rl import collect_selfplay_episodes_vec
    on = GeneralsNetwork(ModelConfig(memory_vision=True, **TINY))
    pool = EnvWorkerPool(n_envs=2, n_workers=1, env_kwargs=dict(max_turns=4),
                         env_attrs=dict(breakdown_pid=0))
    try:
        assert pool.env_attrs == {"breakdown_pid": 0}
        with pytest.raises(ValueError, match="memory_vision"):
            collect_selfplay_episodes_vec(on, [GeneralsEnv(max_turns=4)], CPU, max_steps=4,
                                          act_steps=1, worker_pool=pool)
    finally:
        pool.close()


def test_rl_joint_generals_env_follows_the_network():
    from env.vec_game import make_vec_env
    from training.train_rl_joint import GeneralsSpec
    spec = GeneralsSpec()
    for mv in (False, True):
        net = types.SimpleNamespace(cfg=ModelConfig(memory_vision=mv))
        kw = spec.env_kwargs(net.cfg)
        assert kw["memory_vision"] is mv
        env = make_vec_env("generals", 1, seed=0, **kw)
        assert env.memory_vision is mv and env.envs[0].memory_vision is mv
        spec.check_env(env, net)
        # an --env_kw override contradicting the network is refused
        other = make_vec_env("generals", 1, seed=0, **{**kw, "memory_vision": not mv})
        with pytest.raises(SystemExit, match="memory_vision"):
            spec.check_env(other, net)


def test_train_il_batches_follow_the_network(grids, monkeypatch):
    """training/train_il.py (the dedicated generals IL trainer) feeds a
    memory_vision network the derived memory too."""
    import training.train_il as train_il
    from model.losses import ImitationLoss
    from training.deep_supervision import BatchCarry
    fed = []
    real = train_il.apply_memory_vision
    monkeypatch.setattr(train_il, "apply_memory_vision", lambda g: fed.append(g) or real(g))
    fog = grids[False][:1, :8]
    T = fog.shape[1]
    zeros = torch.zeros(1, T, 4, dtype=torch.long)
    batch = {"obs": {"grid_seq": fog, "globals_seq": torch.zeros(1, T, 4),
                     "legal_source_mask_seq": fog[:, :, 5:8].sum(2, keepdim=True) > 0},
             "tgt": {"count": torch.zeros(1, T, dtype=torch.long), "src_y": zeros,
                     "src_x": zeros, "dir_id": zeros, "ratio_id": zeros,
                     "value": torch.zeros(1, T)},
             "meta": {"lengths": [T]}}
    args = types.SimpleNamespace(log_interval=100, grad_accum=1, world_model_weight=0.0,
                                 grad_clip=1.0, moe_balance_rate=0.0, save_steps=10 ** 9)
    for mv in (False, True):
        fed.clear()
        net = GeneralsNetwork(ModelConfig(memory_vision=mv, **TINY))
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        carries = [BatchCarry(net.supervision_steps, net.cfg.halt_exploration_prob)]
        train_il.train_epoch(net, [batch], ImitationLoss(grid_w=25), opt, sched, CPU, 1, args,
                             rank0=False, tf_ratio=0.0, carries=carries)
        assert len(fed) == int(mv)
        if mv:
            assert torch.equal(real(fed[0]), grids[True][:1, :8])
