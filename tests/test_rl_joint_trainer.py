"""training/train_rl_joint.py — Stage-2 joint mixed-game RL on the whole model.

A tiny Stage-1 checkpoint is trained with the real train_multi_game (generals
6x6 gio records + othello + chess shards, recurrent-depth MoE core), then:

- end to end: 2 iterations of generals + othello RL (mirror + snapshot-pool
  opponents) with an IL replay stream of othello, generals and chess (the
  IL-only third game); the checkpoint is a train_multi_game checkpoint with
  the PopArt statistics folded into the value heads, the frozen reference
  never moves, the MoE balance bias moves only between iterations;
- rollout streams: interleaved othello streams / per-seat generals streams,
  learner-seat and forced-frame masks, in mirror and opponent episodes;
- PPO exactness: with μ = π_old the first minibatch's ratio is 1 within
  1e-5 (eval mode, frozen routing), and the actor's μ matches π_old — for
  the StreamStepper actor and the fixed-batch actor;
- quotas: each game's realised RL loss weight is its quota share even when
  a wave overshoots it, and the overshoot never starves a game next time;
- resume: a run resumed from its iteration-1 checkpoint reproduces
  iteration 2 bit for bit; metrics.jsonl is cut back; a different --init is
  refused; a file-backed pool survives a resume with opponents off;
- data parallel: 2 gloo ranks with different rollouts keep parameters,
  buffers and EMA bit-identical — on the CPU and, where CUDA is available,
  with the model on CUDA over gloo (the MIG-slice mode).
"""

import json
import math
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path

import pytest
import torch

from data_processing.games.dataset import save_shard
from model.components.core import moe_layers, moe_load_counting
from model.multi_game import is_core_key
from tests.generals_gio import write_gio_dir
from tests.gloo_devices import DEVICES, hide_gpus_if_cpu, local_ranks
from tests.test_core_moe import CORE, _board_samples

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
WORLD = 2
TIMEOUT_S = 300


def _core_args(**extra):
    return [f"{k}={v}" for k, v in {**CORE, **extra}.items()]


@pytest.fixture(scope="module")
def stage1(tmp_path_factory):
    """Tiny data + Stage-1 checkpoints: the main one (attention temporal
    mixer, generals/othello/chess) and a recurrent-mixer one (othello/chess)
    for the fixed-batch actor."""
    from training import train_multi_game

    root = tmp_path_factory.mktemp("rlj")
    write_gio_dir(str(root / "generals"), [9, 7, 8], H=6, W=6)
    for g in ("othello", "chess"):
        (root / g).mkdir()
        # random frames: served as they are (layout per_seat, no seat split)
        save_shard(str(root / g / "a.pt"), g, _board_samples(g, 4, 6), meta={"layout": "per_seat"})
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"generals={root / 'generals'}", "--data", f"othello={root / 'othello'}",
        "--data", f"chess={root / 'chess'}", "--steps", "2", "--batch_size", "2",
        "--game_max_len", "generals=8", "--warmup_steps", "1", "--moe_balance_rate", "0.01",
        "--save_dir", str(root / "ck"), "--device", "cpu", "--override", *_core_args()])
    train_multi_game.main([
        "--data", f"othello={root / 'othello'}", "--data", f"chess={root / 'chess'}",
        "--steps", "1", "--batch_size", "2", "--warmup_steps", "1",
        "--save_dir", str(root / "ck_rec"), "--device", "cpu",
        "--override", *_core_args(temporal_mixer="recurrent")])
    return root


def _args(root: Path, save: str, *extra: str):
    return [
        "--init", str(root / "ck" / "multi_final.pt"), "--rl", "generals", "--rl", "othello",
        "--data", f"othello={root / 'othello'}", "--data", f"chess={root / 'chess'}",
        "--data", f"generals={root / 'generals'}", "--replay_max_len", "generals=8",
        "--replay_batch_size", "2", "--tokens_per_iter", "60",
        "--envs", "generals=2", "--envs", "othello=2",
        "--env_kw", "generals:max_turns=10", "--env_kw", "generals:n_mountains=2",
        "--env_kw", "generals:n_cities=1",
        "--mirror_prob", "othello=0.5", "--mirror_prob", "generals=0.5",
        "--snapshot_every", "1", "--epochs", "2", "--minibatches", "2",
        "--adv_filter_q", "0.5", "--entropy_target", "othello=2.0", "--kl_budget", "generals=0.01",
        "--moe_balance_rate", "0.01", "--lr", "1e-4", "--ema_rate", "0.9",
        "--replay_kl_weight", "chess=0.1", "--seed", "7", "--device", "cpu",
        "--save_dir", str(root / save), *extra]


def _trainer(root: Path, save: str, *extra: str):
    from training.train_rl_joint import JointRLTrainer, parse_args
    return JointRLTrainer(parse_args(_args(root, save, *extra)))


def test_help():
    res = subprocess.run([sys.executable, "-m", "training.train_rl_joint", "--help"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "usage:" in res.stdout and "--rl" in res.stdout


def test_moe_load_counting_in_eval_mode():
    from model.multi_game import build_multi_game

    mg = build_multi_game(["othello"], CORE).eval()
    other = build_multi_game(["othello"], CORE).eval()
    net = mg["othello"]
    x = {"board_seq": torch.rand(2, 3, 3, 8, 8)}
    with torch.no_grad():
        net(x, act_steps=2)
    assert all(float(m.load.sum()) == 0 for m in moe_layers(mg))       # eval: no tally
    with moe_load_counting(mg):
        net(x, act_steps=2)
        other["othello"](x, act_steps=2)
    assert all(float(m.load.sum()) > 0 for m in moe_layers(mg))
    assert all(float(m.load.sum()) == 0 for m in moe_layers(other))    # only mg's layers
    assert not any(m.count_eval_loads for m in moe_layers(mg))         # restored


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e(stage1):
    from training import train_rl_joint

    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_e2e", "--iterations", "2"))
    return stage1, out


def test_end_to_end_two_iterations(e2e):
    root, out = e2e
    hist = out["history"]
    assert [r["iteration"] for r in hist] == [1, 2]
    for rec in hist:
        assert math.isfinite(rec["gnorm"]) and rec["dropped"] == 0
        for g in ("generals", "othello"):
            row = rec["games"][g]
            assert row["tokens"] >= 1 and row["episodes"] >= 2
            for k in ("entropy", "kl", "approx_kl", "value_loss", "canary_log_old_mu",
                      "canary_log_ratio", "load_entropy", "il_loss", "il_drift",
                      "core_gnorm_rl", "core_gnorm_replay"):
                assert math.isfinite(row[k]), (g, k, row[k])
            assert row["canary_log_ratio"] < 1e-5              # exact replay
            assert row["canary_log_old_mu"] < 1e-4             # actor ≈ learner
            assert 0.0 < row["load_entropy"] <= 1.0
            assert row["quota"] > 0 and row["core_gnorm_rl"] > 0
        chess = rec["games"]["chess"]                          # IL-only game
        assert set(chess) >= {"il_loss", "il_drift", "replay_kl", "load_entropy",
                              "core_gnorm_replay"}
        assert "tokens" not in chess and chess["core_gnorm_replay"] > 0
    # the replay stream's weights are split of --replay_weight_total (default 1)
    assert sum(out["trainer"].replay_weight.values()) == pytest.approx(1.0)
    assert hist[0]["games"]["chess"]["il_drift"] == 0.0
    lines = (root / "rl_e2e" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"canary_log_ratio"' in lines[0]
    # adaptive controllers moved: othello's entropy coefficient (target 2.0) and
    # generals' KL beta (budget 0.01)
    assert hist[-1]["games"]["othello"]["ent_coef"] != 0.01
    assert hist[-1]["games"]["generals"]["kl_beta"] >= 0.05    # floored at its initial beta


def test_checkpoint_is_a_multi_game_checkpoint(e2e):
    from training.eval_multi_game import load_multi

    root, out = e2e
    trainer = out["trainer"]
    ck = torch.load(root / "rl_e2e" / "rl_final.pt", weights_only=False)
    assert ck["games"] == ["generals", "othello", "chess"] and ck["stage"] == "rl_joint"
    for k in ("configs", "core_config", "core_state", "model_state", "train_model_state",
              "optimizer_state", "ema_state", "rl_state"):
        assert k in ck, k
    assert ck["rl_state"]["iteration"] == 2
    # model_state = EMA with PopArt folded into the value heads (reward units)
    want = trainer.popart.denormalized_state_dict(trainer.model,
                                                  trainer.ema.ema_model_state(trainer.model))
    for k, v in want.items():
        assert torch.equal(ck["model_state"][k], v), k
    for g in ("generals", "othello"):
        assert trainer.popart.std(g) != 1.0 or trainer.popart.mean(g) != 0.0
    # ... and so are the raw weights and the EMA shadow; rl_state keeps the
    # normalised heads (exactly the live ones) for the RL resume
    live = trainer.model.state_dict()
    want = trainer.popart.denormalized_state_dict(trainer.model, live)
    for k, v in want.items():
        assert torch.equal(ck["train_model_state"][k], v), k
    shadow = trainer.popart.denormalized_state_dict(trainer.model, trainer.ema.shadow)
    for k, v in shadow.items():
        assert torch.equal(ck["ema_state"][k], v), k
    raw = ck["rl_state"]["raw_heads"]
    assert raw["train"] and raw["ema"]
    assert all(".value_head." in k for k in (*raw["train"], *raw["ema"]))
    assert any(not torch.equal(ck["train_model_state"][k], v) for k, v in raw["train"].items())
    for k, v in raw["train"].items():
        assert torch.equal(v, live[k]), k
    for k, v in raw["ema"].items():
        assert torch.equal(v, trainer.ema.shadow[k]), k
    model = load_multi(str(root / "rl_e2e" / "rl_final.pt"), CPU)
    assert model["generals"].core is model["othello"].core
    raw_model = load_multi(str(root / "rl_e2e" / "rl_final.pt"), CPU, raw=True)
    assert raw_model["generals"].core is raw_model["othello"].core
    # pool snapshots: every RL game's own modules, the shared core ONCE
    snap = torch.load(root / "rl_e2e" / "snapshots" / "snapshot_000000.pt", weights_only=True)
    core = {k for k in snap if k.startswith("core.")}
    assert core and all(is_core_key(k[len("core."):]) for k in core)
    assert not any(is_core_key(k.split(".", 2)[2]) for k in snap if k.startswith("nets."))
    assert any(k.startswith("nets.othello.") for k in snap)
    assert any(k.startswith("nets.generals.") for k in snap)
    net = trainer._snapshot_loader("othello")(snap)
    assert next(net.parameters()).device == CPU and not trainer.templates["othello"].training


def test_reference_frozen_and_routing_moves_between_iterations(e2e):
    from training.eval_multi_game import load_multi

    root, out = e2e
    trainer = out["trainer"]
    init = load_multi(str(root / "ck" / "multi_final.pt"), CPU)
    for (n, p), (_, q) in zip(trainer.ref.state_dict().items(), init.state_dict().items()):
        assert torch.equal(p, q), n
    moved = [not torch.equal(a.balance_bias, b.balance_bias)
             for a, b in zip(moe_layers(trainer.model), moe_layers(init))]
    assert any(moved)
    assert any(not torch.equal(p, trainer.ema.shadow[n])
               for n, p in trainer.model.named_parameters())
    assert len(trainer.pool) == 3                  # the Stage-1 policy + one per iteration
    # the IL-only game trains through the replay stream alone
    trained = trainer.model["chess"].action_decoder.state_dict()
    assert any(not torch.equal(trained[k], v)
               for k, v in init["chess"].action_decoder.state_dict().items())


# ----------------------------------------------------------------------
# rollout streams
# ----------------------------------------------------------------------

def test_rollout_streams_masks_and_seats(stage1):
    from env.league import League
    from training.train_rl_joint import collect_wave

    tr = _trainer(stage1, "rl_streams", "--iterations", "1")
    tr._add_snapshot(tag="test")
    for g, n_rows in (("othello", 1), ("generals", 2)):
        r = tr.rl[g]
        for mirror in (1.0, 0.0):
            opps = {"pool": r.pool_policy}
            r.pool_policy.refresh()
            league = League(r.env, opps, mirror_prob=mirror, seed=3)
            streams, st = collect_wave(r.env, league, r.actor, seed=11, max_steps=10_000)
            want = r.env.num_envs * (n_rows if mirror == 1.0 else 1)
            assert len(streams) == want and st.episodes == r.env.num_envs
            assert st.mirror == (r.env.num_envs if mirror == 1.0 else 0)
            for s in streams:
                T = s.length
                assert s.done[-1] and not s.done[:-1].any()           # one episode per stream
                assert torch.equal(s.pmask, s.vmask & ~s.forced)      # forced: value only
                if g == "othello":
                    if mirror == 1.0:
                        assert s.vmask.all()                            # both movers learn
                    else:
                        seat = int(s.seat[s.vmask][0])
                        assert torch.equal(s.vmask, s.seat == seat)     # learner seat only
                        assert not s.vmask.all()
                    # interleaved: the mover alternates except after forced passes
                    assert s.obs["board_seq"].shape == (T, 3, 8, 8)
                else:
                    assert s.vmask.all() and (s.seat == s.seat[0]).all()  # per-seat stream
                    assert s.obs["grid_seq"].shape[1:] == (12, 6, 6)
                assert torch.isfinite(s.mu).all() and (s.mu <= 1e-6).all()
            assert st.tokens == sum(int(s.vmask.sum()) for s in streams)
    # the league released every opponent stream
    for g in ("othello", "generals"):
        for pol in tr.rl[g].pool_policy._nets.values():
            assert pol.stepper.live_streams() == []


# ----------------------------------------------------------------------
# PPO exactness
# ----------------------------------------------------------------------

def _first_minibatch_ratio(tr):
    """Roll out, recompute π_old, then evaluate the first-minibatch log π of
    every stream with the training forward (grad on, eval mode, frozen
    routing): returns per game (max |log π − log π_old|, max |log π_old − μ|)
    over the policy tokens, and the canary of a real first minibatch with
    μ := π_old."""
    from training.train_rl_joint import Diag, build_batch, learner_forward

    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {g: 40 for g in tr.rl_games}, diag)
    for g in tr.rl_games:
        tr._recompute(g)
    tr._targets(diag)
    res = {}
    with moe_load_counting(tr.model):
        for g in tr.rl_games:
            r = tr.rl[g]
            net = tr.model[g]
            b = build_batch(r.streams, CPU, ("lp_old", "mu", "pmask"))
            pred = learner_forward(net, r.spec, b, tr.depth, CPU, False, replay_routes=True)
            lp, _ = r.spec.logp_entropy(net, pred, b["actions"], b["legal"])
            pm = b["pmask"]
            assert lp.requires_grad and pm.any()
            res[g] = (float((lp.detach() - b["lp_old"]).abs()[pm].max()),
                      float((b["lp_old"] - b["mu"]).abs()[pm].max()))
    # a real first minibatch with μ = π_old: ratio 1, IS weight exactly 1
    for g in tr.rl_games:
        for s in tr.rl[g].streams:
            s.mu = s.lp_old.clone()
    canary = Diag(tr.games, tr.n_moe)
    with moe_load_counting(tr.model):
        plan, counts = tr._minibatch_plan(0, 0)
        cnt = {name: {g: float(counts[0, gi, j]) for gi, g in enumerate(tr.rl_games)}
               for j, name in enumerate(("kept", "policy", "value"))}
        for g in tr.rl_games:
            ss = [tr.rl[g].streams[i] for i in plan[0][g]]
            if ss:
                tr._rl_chunk_loss(g, ss, cnt, canary, canary=True).backward()
    tr.optimizer.zero_grad(set_to_none=True)
    return res, canary


def test_replay_split_deals_every_game_once_per_minibatch():
    """--replay_split: over the ranks every replay game once per minibatch, at world x
    its weight (so the ranks' averaged gradient is the every-game objective), the
    costly games spread out (longest first to the least loaded rank)."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer

    weights = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.15, "e": 0.25}
    cost = {"a": 5.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
    draws = []
    for rank in range(3):
        fake = SimpleNamespace(replay=dict.fromkeys(weights), replay_weight=weights,
                               args=SimpleNamespace(replay_games_per_mb=0, replay_split=1),
                               world=3, rank=rank, replay_cost=cost)
        draws.append(JointRLTrainer._replay_draws(fake))
    games = [g for d in draws for g, _ in d]
    assert sorted(games) == sorted(weights)                     # each game exactly once
    for d in draws:
        for g, scale in d:
            assert scale == pytest.approx(3 * weights[g])
    assert [g for g, _ in draws[0]] == ["a"]                    # the costly game alone
    # with the RL games split over the ranks, the replay fills the lightly loaded ranks:
    # rank 0 carries an RL game of 10 s per minibatch, so the costly replay game moves on
    seeded = []
    for rank in range(3):
        fake = SimpleNamespace(replay=dict.fromkeys(weights), replay_weight=weights,
                               args=SimpleNamespace(replay_games_per_mb=0, replay_split=1,
                                                    epochs=2, minibatches=4),
                               world=3, rank=rank, replay_cost=cost, _owner={"rl": 0},
                               learn_cost={"rl": 80.0})
        seeded.append([g for g, _ in JointRLTrainer._replay_draws(fake)])
    assert sorted(sum(seeded, [])) == sorted(weights) and seeded[0] == [] and "a" in seeded[1]
    one = SimpleNamespace(replay=dict.fromkeys(weights), replay_weight=weights,
                          args=SimpleNamespace(replay_games_per_mb=0, replay_split=1),
                          world=1, rank=0, replay_cost=cost)
    assert JointRLTrainer._replay_draws(one) == [(g, w) for g, w in weights.items()]


def test_whole_record_replay_batches(stage1):
    """--replay_frames: the IL replay draws whole records from their first frame,
    batches of similar-length records within the frame budget."""
    tr = _trainer(stage1, "rl_whole", "--iterations", "1", "--snapshot_dir", "memory",
                  "--replay_frames", "othello=13", "--replay_frames", "chess=100")
    rs = tr.replay["othello"]
    assert rs.frames == 13 and rs.dataset.crop == "prefix"
    rs.restart()
    full = {int(L) for L in rs.dataset.sample_lengths()}
    for _ in range(4):
        b = rs.next()
        lens = b["meta"]["lengths"]
        assert set(lens) <= full                                   # whole records
        assert len(lens) * max(lens) <= 13 or len(lens) == 1
    assert tr.replay["generals"].frames == 0                        # windows as before


def test_length_chunks():
    from training.train_rl_joint import _length_chunks

    L = [3, 10, 1, 7, 7, 2]                   # stream lengths; longest first, ties in order
    assert _length_chunks(range(6), L, rows=2) == [[1, 3], [4, 0], [5, 2]]
    # a 14-frame budget: rows x the chunk's longest (first) length <= 14
    assert _length_chunks(range(6), L, rows=2, frames=14) == [[1], [3, 4], [0, 5, 2]]
    assert _length_chunks([1], L, rows=2, frames=5) == [[1]]      # over budget: alone
    assert _length_chunks([], L, rows=2) == []


def test_forward_chunking_leaves_the_gradient(stage1):
    """--rows_per_forward / --frames_per_forward only regroup a minibatch's streams
    into learner forwards: the accumulated gradient is the same up to float."""
    from training.train_rl_joint import Diag

    tr = _trainer(stage1, "rl_chunks", "--iterations", "1", "--snapshot_dir", "memory")
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {g: 40 for g in tr.rl_games}, diag)
    for g in tr.rl_games:
        tr._recompute(g)
    tr._targets(diag)
    plan, counts = tr._minibatch_plan(0, 0)
    cnt = {name: {g: float(counts[0, gi, j]) for gi, g in enumerate(tr.rl_games)}
           for j, name in enumerate(("kept", "policy", "value"))}

    def grads(rows, frames):
        tr.optimizer.zero_grad(set_to_none=True)
        n = 0
        with moe_load_counting(tr.model):
            for g in tr.rl_games:
                r = tr.rl[g]
                r.rows_per_forward, r.frames_per_forward = rows, frames
                for chunk in tr._forward_chunks(g, plan[0][g]):
                    tr._rl_chunk_loss(g, [r.streams[i] for i in chunk], cnt,
                                      Diag(tr.games, tr.n_moe), False).backward()
                    n += 1
        return n, {k: p.grad.clone() for k, p in tr.model.named_parameters()
                   if p.grad is not None}

    n1, one = grads(1, 0)
    n8, eight = grads(8, 0)
    nb, budget = grads(1, 10 ** 6)
    assert n1 > n8 >= nb == len(tr.rl_games)          # one forward per game under the budget
    assert one.keys() == eight.keys() == budget.keys() and one
    for k in one:
        torch.testing.assert_close(eight[k], one[k], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(budget[k], one[k], rtol=1e-4, atol=1e-6)
    tr.optimizer.zero_grad(set_to_none=True)


def test_cached_reference_kl_matches_the_per_forward_reference(stage1):
    """--cache_ref_kl: the frozen reference's KL inputs computed once with π_old give
    the loss, KL and gradient of a reference forward in every learner forward."""
    from training.train_rl_joint import Diag

    tr = _trainer(stage1, "rl_refcache", "--iterations", "1", "--snapshot_dir", "memory")
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {g: 40 for g in tr.rl_games}, diag)
    for g in tr.rl_games:
        tr._recompute(g)
    tr._targets(diag)
    cached = [g for g in tr.rl_games if not tr.kl_off[g]]
    assert cached and all(s.ref_logits is not None or s.ref_lp is not None
                          for g in cached for s in tr.rl[g].streams)
    plan, counts = tr._minibatch_plan(0, 0)
    cnt = {name: {g: float(counts[0, gi, j]) for gi, g in enumerate(tr.rl_games)}
           for j, name in enumerate(("kept", "policy", "value"))}
    # move the policy away from the reference (the cache is the reference's alone)
    ref_params = {id(p) for g in cached for p in tr.ref[g].parameters()}
    torch.manual_seed(3)
    with torch.no_grad():
        for p in tr.model.parameters():
            assert id(p) not in ref_params
            p.add_(0.05 * torch.randn_like(p) * p.abs().mean().clamp(min=1e-3))

    def run():
        tr.optimizer.zero_grad(set_to_none=True)
        d = Diag(tr.games, tr.n_moe)
        loss = 0.0
        with moe_load_counting(tr.model):
            for g in tr.rl_games:
                ss = [tr.rl[g].streams[i] for i in plan[0][g]]
                if ss:
                    l = tr._rl_chunk_loss(g, ss, cnt, d, False)
                    l.backward()
                    loss += float(l.detach())
        grads = {k: p.grad.clone() for k, p in tr.model.named_parameters() if p.grad is not None}
        return loss, {g: d.v[g]["kl_sum"] for g in cached}, grads

    with_cache = run()
    for g in cached:
        for s in tr.rl[g].streams:
            s.ref_logits = s.ref_lp = None
    without = run()
    assert all(v > 1e-4 for v in without[1].values()), without[1]     # a real KL
    assert with_cache[0] == pytest.approx(without[0], rel=1e-5, abs=1e-7)
    for g in cached:
        assert with_cache[1][g] == pytest.approx(without[1][g], rel=1e-3, abs=1e-5)  # float noise
    for k in without[2]:
        torch.testing.assert_close(with_cache[2][k], without[2][k], rtol=1e-4, atol=1e-7)
    tr.optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("actor", ["stream", "fixed"])
def test_first_minibatch_ratio_is_one(stage1, actor):
    if actor == "stream":
        tr = _trainer(stage1, "rl_exact", "--iterations", "1", "--rows_per_forward", "othello=1")
        games = ("generals", "othello")
    else:
        from training.train_rl_joint import JointRLTrainer, parse_args
        tr = JointRLTrainer(parse_args([
            "--init", str(stage1 / "ck_rec" / "multi_final.pt"), "--rl", "othello",
            "--envs", "othello=3", "--actor", "fixed", "--mirror_prob", "othello=0.5",
            "--snapshot_dir", "memory", "--device", "cpu",
            "--save_dir", str(stage1 / "rl_fixed")]))
        games = ("othello",)
    bias = [m.balance_bias.clone() for m in moe_layers(tr.model)]
    res, canary = _first_minibatch_ratio(tr)
    for g in games:
        ratio_dev, old_mu = res[g]
        assert ratio_dev < 1e-5, (g, ratio_dev)            # π / π_old == 1 within 1e-5
        assert old_mu < 1e-4, (g, old_mu)                  # the actor's μ is π_old
        d = canary.v[g]
        assert d["canary_n"] > 0
        assert d["canary_oldmu_sum"] == 0.0                # μ := π_old exactly
        assert d["canary_ratio_sum"] / d["canary_n"] < 1e-5
        assert d["is_weight_sum"] / d["pol_n"] == pytest.approx(1.0, abs=1e-6)
    for m, b in zip(moe_layers(tr.model), bias):           # routing frozen
        assert torch.equal(m.balance_bias, b)


def test_generic_spec_uses_the_decoder_contract(stage1, monkeypatch):
    """A game without a dedicated spec: μ = sample()'s action_log_prob, the
    policy = log_prob_entropy, the anchor = sampled k3 KL — still exact."""
    from training import train_rl_joint as trj

    monkeypatch.setitem(trj.GAME_SPECS, "othello", trj.RLGameSpec())
    tr = trj.JointRLTrainer(trj.parse_args([
        "--init", str(stage1 / "ck" / "multi_final.pt"), "--rl", "othello",
        "--envs", "othello=2", "--tokens_per_iter", "40", "--iterations", "1",
        "--snapshot_dir", "memory", "--device", "cpu", "--save_dir", str(stage1 / "rl_generic")]))
    assert not tr.rl["othello"].spec.full_kl
    rec = tr.run_iteration()["games"]["othello"]
    assert rec["canary_log_ratio"] < 1e-5 and rec["canary_log_old_mu"] < 1e-4
    assert math.isfinite(rec["kl"]) and rec["kl"] >= 0.0


def test_generic_spec_refuses_autoregressive_decoders(stage1, monkeypatch):
    """No dedicated spec + conditioning labels (LABEL_KEYS): the generic spec
    would score stored actions under self-sampled conditioning — refused."""
    from training import train_rl_joint as trj

    monkeypatch.delitem(trj.GAME_SPECS, "generals")
    with pytest.raises(SystemExit, match="teacher-force"):
        trj.game_spec("generals")
    monkeypatch.delitem(trj.GAME_SPECS, "go")                   # (go has its own GoSpec)
    assert type(trj.game_spec("go")) is trj.RLGameSpec          # no labels: generic is exact


def test_quotas_set_the_loss_share_despite_wave_overshoot(stage1):
    """Waves overshoot the tiny quotas (othello by ~3x), yet each game's
    realised RL loss weight is its quota share, and the overshoot is not
    booked as a lead: no game is starved (quota 0) the next iteration."""
    tr = _trainer(stage1, "rl_quota", "--iterations", "3", "--minibatches", "1",
                  "--share", "generals=2", "--snapshot_dir", "memory")
    overshoot = False
    for _ in range(3):
        rec = tr.run_iteration()["games"]
        q = {g: rec[g]["quota"] for g in ("generals", "othello")}
        assert q == {"generals": 40, "othello": 20}               # 2 : 1 of 60, every time
        for g in q:
            assert rec[g]["loss_share"] == pytest.approx(q[g] / 60, abs=1e-9)
        tok = {g: rec[g]["tokens"] for g in q}
        overshoot |= abs(tok["othello"] / sum(tok.values()) - 1 / 3) > 0.1
    assert overshoot                                            # the token share was off
    # the per-token weights give each game w_g·min(q_g, n_g) in every term
    counts = torch.tensor([[[5.0, 10.0, 20.0], [30.0, 60.0, 90.0]]], dtype=torch.float64)
    w, eff = tr._loss_weights(counts, {"generals": 40, "othello": 20})
    assert eff == {"generals": 20.0, "othello": 20.0}           # generals short of its quota
    for j, kind in enumerate(("kept", "policy", "value")):
        assert w[kind]["generals"] * counts[0, 0, j] == pytest.approx(20.0)
        assert w[kind]["othello"] * counts[0, 1, j] == pytest.approx(20.0)


# ----------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------

def _state(path):
    ck = torch.load(path, weights_only=False)
    return ck["train_model_state"], ck["ema_state"], ck["rl_state"], ck["optimizer_state"]


def test_resume_reproduces_the_next_iteration(stage1):
    from training import train_rl_joint

    base = ["--snapshot_dir", "memory", "--save_every", "1"]
    train_rl_joint.main(_args(stage1, "rl_full", "--iterations", "2", *base))
    train_rl_joint.main(_args(stage1, "rl_resumed", "--iterations", "2", *base,
                              "--resume", str(stage1 / "rl_full" / "rl_iter1.pt")))
    full = _state(stage1 / "rl_full" / "rl_iter2.pt")
    res = _state(stage1 / "rl_resumed" / "rl_iter2.pt")
    for k, v in full[0].items():
        assert torch.equal(v, res[0][k]), f"param/buffer {k}"
    for k, v in full[1].items():
        assert torch.equal(v, res[1][k]), f"ema {k}"
    for part in ("popart", "advnorm", "entc", "klb", "sched", "iteration"):
        assert full[2][part] == res[2][part], part
    snaps_a, snaps_b = full[2]["pool"]["snapshots"], res[2]["pool"]["snapshots"]
    assert [s[0] for s in snaps_a] == [s[0] for s in snaps_b]
    for (_, sa, _), (_, sb, _) in zip(snaps_a, snaps_b):
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    sa, sb = full[3]["state"], res[3]["state"]
    assert sa.keys() == sb.keys()
    for i in sa:
        assert torch.equal(sa[i]["exp_avg"], sb[i]["exp_avg"])


def test_resume_truncates_metrics_and_checks_the_reference(stage1, tmp_path):
    """A requeue in the same save dir re-runs the iterations after the
    checkpoint: metrics.jsonl keeps one record per iteration.  A resume
    against a different --init (other weights) is refused."""
    import json
    import shutil

    from training import train_rl_joint

    base = ["--snapshot_dir", "memory", "--save_every", "1"]
    save = tmp_path / "requeue"
    save.mkdir()
    full = stage1 / "rl_full"
    if not (full / "rl_iter1.pt").exists():
        train_rl_joint.main(_args(stage1, "rl_full", "--iterations", "2", *base))
    shutil.copy(full / "metrics.jsonl", save / "metrics.jsonl")
    with open(save / "metrics.jsonl", "a") as f:
        f.write('{"iteration": 3, "partial"')                  # cut short by the crash
    train_rl_joint.main(_args(stage1, str(save), "--iterations", "2", *base,
                              "--resume", str(full / "rl_iter1.pt")))
    iters = [json.loads(line)["iteration"] for line in
             (save / "metrics.jsonl").read_text().splitlines()]
    assert iters == [1, 2]
    # a different Stage-1 checkpoint (same games, other weights) as --init
    ck = torch.load(stage1 / "ck" / "multi_final.pt", weights_only=False)
    k = next(k for k, v in ck["model_state"].items()          # a per-game (unaliased) tensor
             if ".value_head." in k and v.is_floating_point())
    ck["model_state"][k] = ck["model_state"][k] + 1.0
    other = tmp_path / "other_init.pt"
    torch.save(ck, other)
    args = _args(stage1, str(tmp_path / "bad"), "--iterations", "2", *base,
                 "--resume", str(full / "rl_iter1.pt"))
    args[args.index("--init") + 1] = str(other)
    with pytest.raises(SystemExit, match="frozen reference"):
        train_rl_joint.main(args)


def test_resume_file_pool_with_opponents_off(e2e):
    """A file-backed pool checkpoint resumed with pure mirror (no opponents)
    loads (and carries the pool on) instead of crashing."""
    from training.train_rl_joint import JointRLTrainer, parse_args

    root, _ = e2e
    args = _args(root, "rl_mirror_resume", "--iterations", "2",
                 "--snapshot_dir", str(root / "rl_e2e" / "snapshots"),
                 "--resume", str(root / "rl_e2e" / "rl_latest.pt"))
    while "--mirror_prob" in args:
        i = args.index("--mirror_prob")
        del args[i:i + 2]
    tr = JointRLTrainer(parse_args(args))
    assert all(r.pool_policy is None for r in tr.rl.values())
    assert len(tr.pool) == 3 and tr.iteration == 2


# ----------------------------------------------------------------------
# data parallel
# ----------------------------------------------------------------------

def _same(a, b) -> bool:
    """Structural equality with NaN == NaN."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_save(kind: str, placement: str = "spread") -> str:
    base = "rl_ddp" if kind == "cpu" else f"rl_ddp_{kind}"
    return base if placement == "spread" else f"{base}_{placement}"


def _ddp_worker(rank: int, port: int, root: str, kind: str = "cpu", lrs=(0, 1),
                placement: str = "spread") -> None:
    # the trainer pins a MIG slice per LOCAL_RANK before touching CUDA
    hide_gpus_if_cpu(kind)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(lrs[rank]), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    from training import train_rl_joint

    rootp = Path(root)
    save = _ddp_save(kind, placement)
    dev = [] if kind == "cpu" else ["--device", kind, "--dist_backend", "gloo"]
    out = train_rl_joint.main(_args(rootp, save, "--iterations", "2",
                                    "--snapshot_dir", str(rootp / save / "snaps"),
                                    "--dist_timeout", "3", "--rl_placement", placement,
                                    "--replay_split", str(int(placement == "split")), *dev))
    model, ema, tr = out["model"], out["ema"], out["trainer"]
    torch.save({
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
        "ema": {n: v.clone() for n, v in ema.shadow.items()},
        "mu": {g: float(sum(s.mu.sum() for s in tr.rl[g].streams)) for g in tr.rl_games},
        "streams": {g: len(tr.rl[g].streams) for g in tr.rl_games},
        "history": out["history"],
        "pool": tr.pool.ids(),
        "rl": {"popart": tr.popart.state_dict(), "klb": tr.klb.state_dict(),
               "entc": tr.entc.state_dict(), "sched": tr.sched.state_dict()},
        "device": str(next(model.parameters()).device),
    }, os.path.join(root, f"{save}_rank{rank}.pt"))


@pytest.mark.parametrize("kind,placement", [
    *[pytest.param(*(k.values if hasattr(k, "values") else (k,)), "spread",
                   marks=getattr(k, "marks", ())) for k in DEVICES],
    pytest.param("cpu", "split")])
def test_two_gloo_ranks_stay_identical(stage1, kind, placement):
    """Two ranks end bit-identical (params, buffers, EMA, RL statistics, pool) with
    global logs.  --rl_placement split (+ --replay_split): each game is rolled out by
    ONE rank (its whole quota), the other rank holds no stream of it, and each replay
    game is replayed by one rank per minibatch."""
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    lrs = local_ranks(WORLD, kind)
    procs = [ctx.Process(target=_ddp_worker, args=(r, port, str(stage1), kind, lrs, placement))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT_S
    try:
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            wait([p.sentinel for p in procs if p.is_alive()], timeout=1.0)
            bad = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            assert not bad, f"rank failed with exit code {bad[0]}"
        assert not any(p.is_alive() for p in procs), "ranks hung (collective desync?)"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
    save = _ddp_save(kind, placement)
    r0, r1 = (torch.load(stage1 / f"{save}_rank{r}.pt", weights_only=False, map_location="cpu")
              for r in range(WORLD))
    if placement == "split":
        for g in r0["streams"]:
            assert (r0["streams"][g] > 0) != (r1["streams"][g] > 0), (g, r0["streams"], r1["streams"])
        owner = r0["history"][-1]["placement"]
        assert owner == r1["history"][-1]["placement"] and set(owner.values()) == {0, 1}
    assert r0["device"].startswith(kind) and r1["device"].startswith(kind)
    for key in ("params", "buffers", "ema"):
        assert r0[key].keys() == r1[key].keys()
        for n in r0[key]:
            assert torch.equal(r0[key][n], r1[key][n]), f"{key} {n} differs across ranks"
    assert r0["rl"] == r1["rl"] and r0["pool"] == r1["pool"] == [0, 1, 2]
    assert r0["mu"] != r1["mu"]                         # the ranks rolled out different games
    for h0, h1 in zip(r0["history"], r1["history"]):   # logs are global statistics
        assert _same(h0["games"], h1["games"]) and _same(h0["gnorm"], h1["gnorm"])
    ck = torch.load(stage1 / save / "rl_final.pt", weights_only=False, map_location="cpu")
    assert ck["world_size"] == WORLD
    assert sorted(os.listdir(stage1 / save / "snaps")) == [
        f"snapshot_{i:06d}.pt" for i in range(3)]


def test_eval_vs_init_and_alarm_fields(stage1):
    """--eval_every: the learner plays the frozen --init network (from iteration 1: at 0 it
    has the same weights); vs_init / timeout_rate / shaping land in the record, nothing trips."""
    from training import train_rl_joint

    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_eval", "--iterations", "2", "--eval_every", "1",
                                    "--eval_episodes", "4", "--eval_episodes_game", "generals=2",
                                    "--trace_every", "1"))
    assert out["history"][0]["games"]["othello"]["vs_init_n"] == 0     # not at iteration 0
    rec = out["history"][1]
    for g in ("generals", "othello"):
        row = rec["games"][g]
        assert row["vs_init_n"] >= (2 if g == "generals" else 4) and 0.0 <= row["vs_init"] <= 1.0, (g, row)
        assert row["timeout_rate"] == 0.0 or g == "generals"
        assert "alarm" not in row
    assert rec["games"]["generals"]["vs_init_n"] < rec["games"]["othello"]["vs_init_n"]   # its own
    assert "stopped" not in rec and out["trainer"].stopped == {}
    for g in ("generals", "othello"):         # --trace_every 1: one episode per game and iteration
        tr = json.loads((stage1 / "rl_eval" / "traces" / f"{g}_it1.json").read_text())
        assert tr["game"] == g and tr["steps"] and tr["steps"][-1]["done"]
        assert all("applied" in st and "rewards" in st for st in tr["steps"])
    gen = json.loads((stage1 / "rl_eval" / "traces" / "generals_it0.json").read_text())
    assert "state" in gen["steps"][0]            # generals: the board after every step


def test_deadline_stops_before_an_iteration_that_would_not_finish(stage1):
    """--deadline: after one iteration the trainer knows how long one takes; an iteration
    (plus its save) that would end after the deadline is not started — rl_latest.pt is
    written, no rl_final.pt, and the estimates are kept for the next job's first iteration.
    A far deadline changes nothing."""
    import time as _time

    from training import train_rl_joint

    save = stage1 / "rl_deadline"
    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_deadline", "--iterations", "3", "--ckpt_minutes", "0",
                                    "--deadline", str(_time.time() + 5)))
    assert len(out["history"]) == 1
    assert (save / "rl_latest.pt").exists() and not (save / "rl_final.pt").exists()
    ck = torch.load(save / "rl_latest.pt", map_location="cpu", weights_only=False)
    timing = ck["rl_state"]["timing"]
    assert timing["iter_s"] > 0 and "eval_s" not in timing      # (a save's own time: the next one)
    assert out["trainer"].timing["save_s"] >= 0
    torch.manual_seed(0)
    far = train_rl_joint.main(_args(stage1, "rl_deadline_far", "--iterations", "2",
                                    "--deadline", str(_time.time() + 1e6)))
    assert len(far["history"]) == 2 and (stage1 / "rl_deadline_far" / "rl_final.pt").exists()


def test_a_game_that_trains_nowhere_is_still_evaluated(stage1):
    """A zero --share (like a game stopped by an alarm) plays no RL episode, but every
    --eval_every it still plays the frozen --init network: a game left out of RL stays visible
    while the shared core moves under it."""
    from training import train_rl_joint

    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_idle_eval", "--iterations", "2", "--eval_every", "1",
                                    "--eval_episodes", "4", "--eval_episodes_game", "generals=2",
                                    "--share", "othello=0"))
    row = out["history"][1]["games"]["othello"]
    assert row["episodes"] == 0 and row["tokens"] == 0, row
    assert row["vs_init_n"] >= 4 and 0.0 <= row["vs_init"] <= 1.0, row
    assert out["history"][1]["games"]["generals"]["episodes"] > 0
    assert "alarm" not in row and out["trainer"].stopped == {}


def test_a_failing_engine_referee_is_disabled_not_fatal(stage1):
    """--eval_engine with an engine that cannot start: the run goes on, the referee is off."""
    from training import train_rl_joint

    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_badengine", "--iterations", "2", "--eval_every", "1",
                                    "--eval_episodes", "2", "--eval_engine", "othello=/no/such/engine"))
    tr = out["trainer"]
    assert [r["iteration"] for r in out["history"]] == [1, 2]
    assert "othello" in tr._engine_failed and "othello" not in tr._engines
    assert out["history"][1]["games"]["othello"]["vs_engine_n"] == 0


def test_time_windows_match_the_whole_stream(stage1):
    """A stream longer than its game's --frames_per_forward runs in time windows chained
    through the recurrent memory state (truncated BPTT): π_old / V / reference
    log-probs of the recompute and the loss value equal the whole-stream forward; the
    gradient differs only by the cut at window boundaries."""
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args, time_windows

    def trainer(fpf):
        argv = ["--init", str(stage1 / "ck_rec" / "multi_final.pt"), "--rl", "othello",
                "--data", f"othello={stage1 / 'othello'}", "--replay_batch_size", "2",
                "--tokens_per_iter", "60", "--envs", "othello=2", "--epochs", "1",
                "--minibatches", "1", "--seed", "7", "--device", "cpu",
                "--frames_per_forward", f"othello={fpf}",
                "--save_dir", str(stage1 / f"tw{fpf}")]
        torch.manual_seed(0)
        return JointRLTrainer(parse_args(argv))

    whole, win = trainer(0), trainer(7)
    net = win.model["othello"]
    assert net.cfg.temporal_mixer == "recurrent"
    for tr in (whole, win):
        tr._rollouts(0, {"othello": 60}, Diag(tr.games, tr.n_moe))
    # the same rollouts in both: copy the first trainer's streams into the second
    win.rl["othello"].streams = whole.rl["othello"].streams
    T = max(s.length for s in whole.rl["othello"].streams)
    assert len(time_windows(net, T, 7)) > 1
    ref = {}
    for tr in (whole, win):
        tr._recompute("othello")
        ref[tr is win] = [(s.lp_old.clone(), s.v_old.clone(),
                           None if s.ref_lp is None else s.ref_lp.clone())
                          for s in tr.rl["othello"].streams]
    for (a1, v1, r1), (a2, v2, r2) in zip(ref[False], ref[True]):
        assert torch.allclose(a1, a2, atol=1e-4) and torch.allclose(v1, v2, atol=1e-4)
        if r1 is not None:
            assert torch.allclose(r1, r2, atol=1e-4)
    for tr in (whole, win):
        tr._targets(Diag(tr.games, tr.n_moe))
    plan, counts = whole._minibatch_plan(0, 0)
    cnt = {name: {"othello": float(counts[0, 0, j])}
           for j, name in enumerate(("kept", "policy", "value"))}
    ss = whole.rl["othello"].streams
    l_whole = float(whole._rl_chunk_loss("othello", ss, cnt, Diag(whole.games, whole.n_moe),
                                         False).detach())
    parts = [float(l.detach()) for l in win._rl_chunk_losses("othello", ss, cnt,
                                                    Diag(win.games, win.n_moe), False)]
    assert len(parts) > 1
    assert abs(sum(parts) - l_whole) <= 1e-4 * max(1.0, abs(l_whole))


def test_value_warmup_trains_no_policy(stage1):
    """--value_warmup: the first iterations' RL loss has no policy-gradient / entropy term —
    with the KL anchor off, the action decoder gets no gradient at all."""
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    argv = ["--init", str(stage1 / "ck_rec" / "multi_final.pt"), "--rl", "othello",
            "--tokens_per_iter", "60", "--envs", "othello=2", "--epochs", "1",
            "--minibatches", "1", "--seed", "7", "--device", "cpu", "--kl_beta", "othello=0",
            "--save_dir", str(stage1 / "vw")]

    def decoder_grad(warm):
        torch.manual_seed(0)
        tr = JointRLTrainer(parse_args(argv + ["--value_warmup", str(warm)]))
        tr._rollouts(0, {"othello": 60}, Diag(tr.games, tr.n_moe))
        tr._recompute("othello")
        tr._targets(Diag(tr.games, tr.n_moe))
        plan, counts = tr._minibatch_plan(0, 0)
        cnt = {name: {"othello": float(counts[0, 0, j])}
               for j, name in enumerate(("kept", "policy", "value"))}
        tr.optimizer.zero_grad(set_to_none=True)
        tr._rl_chunk_loss("othello", tr.rl["othello"].streams, cnt,
                          Diag(tr.games, tr.n_moe), False).backward()
        dec = tr.model["othello"].action_decoder
        return sum(float(p.grad.abs().sum()) for p in dec.parameters() if p.grad is not None)

    assert decoder_grad(0) > 0
    assert decoder_grad(5) == 0


def test_value_core_grad_zero_trains_the_value_head_only():
    """value_core_grad=0: a value loss reaches the value head and nothing upstream (the core,
    the encoder); 1 (default) reaches both; 0.5 halves the core's gradient."""
    from model.multi_game import build_multi_game

    torch.manual_seed(0)
    net = build_multi_game(["othello"], CORE)["othello"]
    x = {"board_seq": torch.rand(2, 3, 3, 8, 8)}

    net.eval()                              # a deterministic forward (no routing noise)

    def grads(s):
        net.zero_grad(set_to_none=True)
        net.value_core_grad = s
        torch.manual_seed(1)
        out = net(x, act_steps=1)
        v = out["value_logits"].float().square().mean() if "value_logits" in out else out["value"].square().mean()
        v.backward()
        head = sum(float(p.grad.abs().sum()) for p in net.value_head.parameters() if p.grad is not None)
        core = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in net.named_parameters() if not n.startswith("value_head")]
        return head, core

    h1, c1 = grads(1.0)
    h0, c0 = grads(0.0)
    hh, ch = grads(0.5)
    assert h1 > 0 and h0 > 0 and abs(h0 - h1) < 1e-6 * max(1.0, h1)
    assert sum(float(g.abs().sum()) for g in c1) > 0
    assert sum(float(g.abs().sum()) for g in c0) == 0
    assert all(torch.allclose(a * 0.5, b, atol=1e-7) for a, b in zip(c1, ch))


def test_clear_alarms_resumes_a_stopped_game(stage1):
    """--clear_alarms GAME on --resume: a game the checkpoint has stopped trains again."""
    from training import train_rl_joint
    from training.train_rl_joint import JointRLTrainer, parse_args

    torch.manual_seed(0)
    train_rl_joint.main(_args(stage1, "rl_clear", "--iterations", "1"))
    ck_path = stage1 / "rl_clear" / "rl_latest.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck["rl_state"]["stopped"] = {"othello": "iteration 1: test"}
    torch.save(ck, ck_path)
    tr = JointRLTrainer(parse_args(_args(stage1, "rl_clear", "--iterations", "2",
                                         "--resume", str(ck_path))))
    assert tr.stopped == {"othello": "iteration 1: test"}
    tr2 = JointRLTrainer(parse_args(_args(stage1, "rl_clear", "--iterations", "2",
                                          "--resume", str(ck_path), "--clear_alarms", "othello")))
    assert tr2.stopped == {}
    # GAME@ITER clears only the stop made at that iteration: a flag left in a run's settings
    # keeps a later stop
    ck["rl_state"]["stopped"] = {"othello": "iteration 7: score vs the --init network 0.2 < 0.35"}
    torch.save(ck, ck_path)
    kept = JointRLTrainer(parse_args(_args(stage1, "rl_clear", "--iterations", "2",
                                           "--resume", str(ck_path), "--clear_alarms", "othello@1")))
    assert set(kept.stopped) == {"othello"}
    cleared = JointRLTrainer(parse_args(_args(stage1, "rl_clear", "--iterations", "2", "--resume",
                                              str(ck_path), "--clear_alarms", "nogame,othello@7")))
    assert cleared.stopped == {}


def test_stop_file_saves_and_exits_after_the_iteration(stage1):
    """--stop_file: once the file exists the trainer writes rl_latest.pt after the iteration
    and stops — no rl_final.pt (the job's wrapper continues with a handover)."""
    from training import train_rl_joint

    save = stage1 / "rl_stopfile"
    save.mkdir(exist_ok=True)
    stop = save / "STOP"
    stop.write_text("now")
    torch.manual_seed(0)
    out = train_rl_joint.main(_args(stage1, "rl_stopfile", "--iterations", "3",
                                    "--stop_file", str(stop), "--ckpt_minutes", "0"))
    assert len(out["history"]) == 1
    assert (save / "rl_latest.pt").exists() and not (save / "rl_final.pt").exists()
