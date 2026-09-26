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
- resume: a run resumed from its iteration-1 checkpoint reproduces
  iteration 2 bit for bit;
- data parallel: 2 gloo ranks with different rollouts keep parameters,
  buffers and EMA bit-identical.
"""

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
from tests.generals_gio import write_gio_dir
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
        save_shard(str(root / g / "a.pt"), g, _board_samples(g, 4, 6))
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
        "--replay_batch_size", "2", "--tokens_per_iter", "60", "--share_floor", "0.3",
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
                      "canary_log_ratio", "load_entropy", "il_loss", "il_drift"):
                assert math.isfinite(row[k]), (g, k, row[k])
            assert row["canary_log_ratio"] < 1e-5              # exact replay
            assert row["canary_log_old_mu"] < 1e-4             # actor ≈ learner
            assert 0.0 < row["load_entropy"] <= 1.0
        chess = rec["games"]["chess"]                          # IL-only game
        assert set(chess) >= {"il_loss", "il_drift", "replay_kl", "load_entropy"}
        assert "tokens" not in chess
    assert hist[0]["games"]["chess"]["il_drift"] == 0.0
    lines = (root / "rl_e2e" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"canary_log_ratio"' in lines[0]
    # adaptive controllers moved: othello's entropy coefficient (target 2.0) and
    # generals' KL beta (budget 0.01)
    assert hist[-1]["games"]["othello"]["ent_coef"] != 0.01
    assert hist[-1]["games"]["generals"]["kl_beta"] != 0.05


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
    model = load_multi(str(root / "rl_e2e" / "rl_final.pt"), CPU)
    assert model["generals"].core is model["othello"].core


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
            pred = learner_forward(net, r.spec, b, tr.depth, CPU, False)
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


def _ddp_worker(rank: int, port: int, root: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    from training import train_rl_joint

    rootp = Path(root)
    out = train_rl_joint.main(_args(rootp, "rl_ddp", "--iterations", "2",
                                    "--snapshot_dir", str(rootp / "rl_ddp" / "snaps"),
                                    "--dist_timeout", "3"))
    model, ema, tr = out["model"], out["ema"], out["trainer"]
    torch.save({
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
        "ema": {n: v.clone() for n, v in ema.shadow.items()},
        "mu": {g: float(sum(s.mu.sum() for s in tr.rl[g].streams)) for g in tr.rl_games},
        "history": out["history"],
        "pool": tr.pool.ids(),
        "rl": {"popart": tr.popart.state_dict(), "klb": tr.klb.state_dict(),
               "entc": tr.entc.state_dict(), "sched": tr.sched.state_dict()},
    }, os.path.join(root, f"ddp_rank{rank}.pt"))


def test_two_gloo_ranks_stay_identical(stage1):
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_ddp_worker, args=(r, port, str(stage1))) for r in range(WORLD)]
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
    r0, r1 = (torch.load(stage1 / f"ddp_rank{r}.pt", weights_only=False) for r in range(WORLD))
    for key in ("params", "buffers", "ema"):
        assert r0[key].keys() == r1[key].keys()
        for n in r0[key]:
            assert torch.equal(r0[key][n], r1[key][n]), f"{key} {n} differs across ranks"
    assert r0["rl"] == r1["rl"] and r0["pool"] == r1["pool"] == [0, 1, 2]
    assert r0["mu"] != r1["mu"]                         # the ranks rolled out different games
    for h0, h1 in zip(r0["history"], r1["history"]):   # logs are global statistics
        assert _same(h0["games"], h1["games"]) and _same(h0["gnorm"], h1["gnorm"])
    ck = torch.load(stage1 / "rl_ddp" / "rl_final.pt", weights_only=False)
    assert ck["world_size"] == WORLD
    assert sorted(os.listdir(stage1 / "rl_ddp" / "snaps")) == [
        f"snapshot_{i:06d}.pt" for i in range(3)]
