"""training/train_rl_joint.py on gomoku (env/vec_gomoku.py + GomokuSpec).

Tiny Stage-1 checkpoints are trained with the real train_multi_game:
``ck``      gomoku on the checked-in 20x20 fixture gm-shards (gmdata output
            of hand-written games) + othello, attention temporal mixer;
``ck9``     gomoku on synthetic 9x9 shards (shard meta board_size 9),
            recurrent temporal mixer (the fixed-batch actor);
``ck9pass`` the same 9x9 gomoku built with a pass head (allow_pass=True).
Then:

- the env is built on the network's board (20 / 9 — not the env default)
  and checked against it; a mismatching ``--env_kw gomoku:board_size`` is
  refused; the spec is the full-KL placement spec;
- end to end: 2 iterations of joint gomoku + othello RL (mirror + snapshot
  pool) with an IL replay stream of both games, gomoku truncated by
  max_moves — finite metrics, exact canaries, adaptive controllers moving;
- rollout streams: ONE interleaved stream per game, strictly alternating
  movers, no forced frames (pmask == vmask), learner-seat masks against a
  pool opponent, zero-sum ±1 / 0 outcomes on the last frame only;
- PPO exactness on 9x9 (games end by five / full board): the first
  minibatch's ratio is 1 within 1e-5 and the actor's μ is the learner's
  π_old, for the stream and the fixed-batch actor — and for a network WITH a
  pass head, whose sampled pass flags the env ignores (placement-only μ / π).
"""

import math
from pathlib import Path

import pytest
import torch

from data_processing.games.dataset import save_shard
from tests.test_core_moe import _board_samples
from tests.test_rl_joint_trainer import _core_args, _first_minibatch_ratio

ROOT = Path(__file__).resolve().parents[1]
GOMOKU20 = ROOT / "tests" / "fixtures" / "gomoku_env" / "shards20"


@pytest.fixture(scope="module")
def stage1(tmp_path_factory):
    from training import train_multi_game

    root = tmp_path_factory.mktemp("rlj_gomoku")
    (root / "othello").mkdir()
    save_shard(str(root / "othello" / "a.pt"), "othello", _board_samples("othello", 4, 6))
    (root / "gomoku9").mkdir()
    save_shard(str(root / "gomoku9" / "a.pt"), "gomoku", _board_samples("gomoku", 4, 6, board=9),
               meta={"board_size": 9})
    common = ["--batch_size", "2", "--warmup_steps", "1", "--device", "cpu"]
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"gomoku={GOMOKU20}", "--data", f"othello={root / 'othello'}",
        "--steps", "2", "--moe_balance_rate", "0.01", "--save_dir", str(root / "ck"), *common,
        "--override", *_core_args()])
    train_multi_game.main([
        "--data", f"gomoku={root / 'gomoku9'}", "--steps", "1", "--save_dir", str(root / "ck9"),
        *common, "--override", *_core_args(temporal_mixer="recurrent")])
    train_multi_game.main([
        "--data", f"gomoku={root / 'gomoku9'}", "--steps", "1",
        "--save_dir", str(root / "ck9pass"), *common,
        "--override", *_core_args(allow_pass=True)])
    return root


def _args(root: Path, save: str, *extra: str):
    return [
        "--init", str(root / "ck" / "multi_final.pt"), "--rl", "gomoku", "--rl", "othello",
        "--data", f"gomoku={GOMOKU20}", "--data", f"othello={root / 'othello'}",
        "--replay_batch_size", "2", "--tokens_per_iter", "60",
        "--envs", "gomoku=2", "--envs", "othello=2", "--env_kw", "gomoku:max_moves=24",
        "--mirror_prob", "gomoku=0.5", "--mirror_prob", "othello=0.5",
        "--snapshot_every", "1", "--epochs", "2", "--minibatches", "2",
        "--entropy_target", "gomoku=3.0", "--kl_budget", "gomoku=0.01",
        "--moe_balance_rate", "0.01", "--lr", "1e-4", "--ema_rate", "0.9",
        "--snapshot_dir", "memory", "--seed", "7", "--device", "cpu",
        "--save_dir", str(root / save), *extra]


def _trainer9(root: Path, save: str, ck: str = "ck9", *extra: str):
    from training.train_rl_joint import JointRLTrainer, parse_args
    return JointRLTrainer(parse_args([
        "--init", str(root / ck / "multi_final.pt"), "--rl", "gomoku", "--envs", "gomoku=3",
        "--mirror_prob", "gomoku=0.5", "--snapshot_dir", "memory", "--device", "cpu",
        "--save_dir", str(root / save), *extra]))


def test_env_follows_the_network_board(stage1):
    from training.train_rl_joint import GAME_SPECS, GomokuSpec, JointRLTrainer, parse_args

    assert isinstance(GAME_SPECS["gomoku"], GomokuSpec) and GAME_SPECS["gomoku"].full_kl
    tr = JointRLTrainer(parse_args(_args(stage1, "rl_board")))
    net, env = tr.model["gomoku"], tr.rl["gomoku"].env
    assert net.cfg.board_h == net.cfg.board_w == 20                 # from the shard meta
    assert (env.board_h, env.board_w) == (20, 20) and env.max_moves == 24
    assert net.action_decoder.pass_head is None                     # registry: allow_pass=False
    tr9 = _trainer9(stage1, "rl_board9")
    assert tr9.model["gomoku"].cfg.board_h == 9
    assert (tr9.rl["gomoku"].env.board_h, tr9.rl["gomoku"].env.board_w) == (9, 9)
    with pytest.raises(SystemExit, match="network expects"):
        _trainer9(stage1, "rl_board_bad", "ck9", "--env_kw", "gomoku:board_size=15")


def test_spec_checks_the_env_against_the_network():
    from types import SimpleNamespace

    from env.vec_gomoku import GomokuVecEnv
    from training.train_rl_joint import GAME_SPECS

    spec = GAME_SPECS["gomoku"]

    def net(**kw):
        cfg = dict(piece_channels=3, board_h=9, board_w=9, board_global_features=0)
        return SimpleNamespace(cfg=SimpleNamespace(**{**cfg, **kw}))

    assert spec.env_kwargs(net().cfg) == {"board_size": (9, 9)}
    env = GomokuVecEnv(1, **spec.env_kwargs(net(board_w=11).cfg))
    assert (env.board_h, env.board_w) == (9, 11)
    spec.check_env(env, net(board_w=11))
    for bad in (dict(board_w=9), dict(board_w=11, piece_channels=4),
                dict(board_w=11, board_global_features=6)):
        with pytest.raises(SystemExit, match="network expects"):
            spec.check_env(env, net(**bad))


def test_end_to_end_gomoku_and_othello_two_iterations(stage1):
    from training import train_rl_joint
    from training.eval_multi_game import load_multi

    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_e2e", "--iterations", "2"))
    hist = out["history"]
    assert [r["iteration"] for r in hist] == [1, 2]
    for rec in hist:
        assert math.isfinite(rec["gnorm"]) and rec["dropped"] == 0
        for g in ("gomoku", "othello"):
            row = rec["games"][g]
            assert row["tokens"] >= 1 and row["episodes"] >= 2
            for k in ("entropy", "kl", "approx_kl", "value_loss", "canary_log_old_mu",
                      "canary_log_ratio", "load_entropy", "il_loss", "il_drift",
                      "core_gnorm_rl", "core_gnorm_replay"):
                assert math.isfinite(row[k]), (g, k, row[k])
            assert row["canary_log_ratio"] < 1e-5              # exact replay
            assert row["canary_log_old_mu"] < 1e-4             # actor ≈ learner
            assert row["kl"] >= 0.0 and row["core_gnorm_rl"] > 0
    assert hist[-1]["games"]["gomoku"]["ent_coef"] != 0.01
    assert hist[-1]["games"]["gomoku"]["kl_beta"] != 0.05
    model = load_multi(str(stage1 / "rl_e2e" / "rl_final.pt"), torch.device("cpu"))
    assert model["gomoku"].core is model["othello"].core
    assert model["gomoku"].cfg.board_h == 20


def test_rollout_streams_are_interleaved_games(stage1):
    from env.league import League
    from training.train_rl_joint import collect_wave

    tr = _trainer9(stage1, "rl_streams")
    tr._add_snapshot(tag="test")
    r = tr.rl["gomoku"]
    for mirror in (1.0, 0.0):
        r.pool_policy.refresh()
        league = League(r.env, {"pool": r.pool_policy}, mirror_prob=mirror, seed=3)
        streams, st = collect_wave(r.env, league, r.actor, seed=11, max_steps=10_000)
        assert len(streams) == r.env.num_envs == st.episodes
        for s in streams:
            T = s.length
            assert s.done[-1] and not s.done[:-1].any()               # one episode per stream
            assert torch.equal(s.seat, torch.arange(T) % 2)           # strictly alternating
            assert not s.forced.any() and torch.equal(s.pmask, s.vmask)
            if mirror == 1.0:
                assert s.vmask.all()                                   # both movers learn
            else:
                seat = int(s.seat[s.vmask][0])
                assert torch.equal(s.vmask, s.seat == seat) and not s.vmask.all()
            assert s.obs["board_seq"].shape == (T, 3, 9, 9)
            assert s.legal["place"].shape == (T, 81) and not s.legal["pass"].any()
            played = s.actions["place_idx"][s.vmask]
            assert s.legal["place"][s.vmask].gather(1, played[:, None]).all()
            assert not s.actions["passed"].any()                       # no pass head
            assert (s.rewards[:-1] == 0).all() and float(s.rewards[-1].sum()) == 0.0
            assert set(s.rewards[-1].tolist()) <= {-1.0, 0.0, 1.0}
            assert torch.isfinite(s.mu).all() and (s.mu <= 1e-6).all()
    assert r.pool_policy._nets and all(p.stepper.live_streams() == []
                                       for p in r.pool_policy._nets.values())


@pytest.mark.parametrize("actor", ["stream", "fixed"])
def test_first_minibatch_ratio_is_one(stage1, actor):
    tr = _trainer9(stage1, f"rl_exact_{actor}", "ck9", "--actor", actor)
    res, canary = _first_minibatch_ratio(tr)
    ratio_dev, old_mu = res["gomoku"]
    assert ratio_dev < 1e-5, ratio_dev                  # π / π_old == 1 within 1e-5
    assert old_mu < 1e-4, old_mu                        # the actor's μ is π_old
    d = canary.v["gomoku"]
    assert d["canary_n"] > 0 and d["canary_oldmu_sum"] == 0.0
    assert d["canary_ratio_sum"] / d["canary_n"] < 1e-5
    assert d["is_weight_sum"] / d["pol_n"] == pytest.approx(1.0, abs=1e-6)


def test_pass_head_network_keeps_placement_only_logprob(stage1):
    """A gomoku net WITH a pass head samples pass flags the env ignores: the
    spec's μ / π are the placement log-prob of the applied move, so the
    replay stays exact, and the KL still covers the pass head."""
    tr = _trainer9(stage1, "rl_pass", "ck9pass")
    net = tr.model["gomoku"]
    assert net.action_decoder.pass_head is not None
    res, canary = _first_minibatch_ratio(tr)
    ratio_dev, old_mu = res["gomoku"]
    assert ratio_dev < 1e-5 and old_mu < 1e-4, res
    streams = tr.rl["gomoku"].streams
    passed = torch.cat([s.actions["passed"] for s in streams])
    assert passed.any()                                  # the pass head did fire ...
    for s in streams:                                    # ... and the moves were placed anyway
        assert torch.equal(s.seat, torch.arange(s.length) % 2)
    from training.train_rl_joint import build_batch, learner_forward
    r = tr.rl["gomoku"]
    b = build_batch(streams[:1], torch.device("cpu"), ("forced", "vmask"))
    pred = learner_forward(net, r.spec, b, tr.depth, torch.device("cpu"), False)
    terms = r.spec.kl_terms(net, pred, r.spec.kl_targets(b["actions"], b["forced"], net.cfg),
                            b["vmask"], b["legal"])
    assert set(terms) == {"place", "pass"}
