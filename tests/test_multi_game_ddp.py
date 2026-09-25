"""Data-parallel training/train_multi_game.py on a real 2-process gloo group.

Two spawned ranks run the real trainer (torchrun-style environment) for a few
optimizer steps on tiny synthetic shards of two games, one micro-batch per
rank per step, so the ranks regularly draw DIFFERENT games: a game's
encoder/decoder then has a gradient on one rank only.  Afterwards every
parameter, buffer (MoE balance biases included) and EMA shadow must be
bit-identical across ranks; a parameter that is trainable but used by no
forward must keep grad None through every sync (AdamATan2 never sees it: no
state, no weight decay — its value is untouched); and a non-finite gradient
norm on ONE rank must drop the step on BOTH.
"""

import math
import os
import socket
import time
from multiprocessing.connection import wait

import pytest
import torch

from data_processing.games.dataset import save_shard
from tests.test_core_moe import CORE, _board_samples

WORLD = 2
STEPS = 5
NAN_STEP = 3          # rank 1 poisons its (already synced) gradient at this step
TIMEOUT_S = 300


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, root: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    from training import train_multi_game as tmg

    build, sync = tmg.build_net_tolerant, tmg.sync_gradients

    def build_with_probe(game, arch):
        net, dropped = build(game, arch)
        if game == "chess":
            # trainable, but outside every forward: no rank ever has its grad
            net.ddp_probe = torch.nn.Parameter(torch.full((3,), 2.0))
        return net, dropped

    record = []     # per step: (games with a local encoder grad, probe grad None after sync)

    def recording_sync(model, world, **kw):
        local = sorted(g for g in model.games
                       if any(p.grad is not None for p in model[g].obs_encoder.parameters()))
        sync(model, world, **kw)
        record.append((local, model["chess"].ddp_probe.grad is None))
        if rank == 1 and len(record) == NAN_STEP:
            next(p for p in model.parameters() if p.grad is not None).grad.fill_(math.nan)

    tmg.build_net_tolerant, tmg.sync_gradients = build_with_probe, recording_sync
    out = tmg.main([
        "--data", f"othello={root}/othello", "--data", f"chess={root}/chess",
        "--val", f"othello={root}/othello",
        "--steps", str(STEPS), "--games_per_step", "1", "--batch_size", "2",
        "--warmup_steps", "1", "--eval_every", "2", "--log_every", "2",
        "--ema_rate", "0.9", "--moe_balance_rate", "0.01", "--clip_grad", "0.5",
        "--bucket_mb", "0.01", "--seed", "3", "--device", "cpu", "--dist_timeout", "2",
        "--save_dir", f"{root}/ck", "--override", *[f"{k}={v}" for k, v in CORE.items()]])
    model, ema, opt = out["model"], out["ema"], out["optimizer"]
    probe = model["chess"].ddp_probe
    core_p = model["othello"].core.parameters().__next__()
    torch.save({
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
        "ema": {n: v.clone() for n, v in ema.shadow.items()},
        "record": record,
        "probe_in_optimizer": any(p is probe for g in opt.param_groups for p in g["params"]),
        "probe_state": dict(opt.state.get(probe, {})),
        "core_steps": opt.state[core_p]["step"],
    }, os.path.join(root, f"rank{rank}.pt"))


@pytest.fixture(scope="module")
def ranks(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddp")
    for g in ("othello", "chess"):
        (root / g).mkdir()
        save_shard(str(root / g / "a.pt"), g, _board_samples(g, 6, 6))
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, str(root))) for r in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT_S
    try:
        # a rank that dies leaves its peer blocked in a collective: stop at
        # the first failure instead of waiting for the peer's timeout
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
    out = [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(WORLD)]
    ck = torch.load(root / "ck" / "multi_final.pt", weights_only=False)
    return out, ck


def test_ranks_draw_different_games(ranks):
    """rank-specific RNG: some steps have a game's adapter on one rank only
    (the zero-fill path of the sync is exercised)."""
    (r0, r1), _ = ranks
    local0 = [loc for loc, _ in r0["record"]]
    local1 = [loc for loc, _ in r1["record"]]
    assert len(local0) == len(local1) == STEPS
    assert any(a != b for a, b in zip(local0, local1))


def test_parameters_buffers_and_ema_bit_identical(ranks):
    (r0, r1), _ = ranks
    for key in ("params", "buffers", "ema"):
        assert r0[key].keys() == r1[key].keys()
        for n in r0[key]:
            assert torch.equal(r0[key][n], r1[key][n]), f"{key} {n} differs across ranks"
    biases = [v for n, v in r0["buffers"].items() if n.endswith("balance_bias")]
    assert biases and any(b.abs().sum() > 0 for b in biases), "balance bias never moved"


def test_unused_parameter_keeps_grad_none_and_value(ranks):
    (r0, r1), _ = ranks
    for r in (r0, r1):
        assert r["probe_in_optimizer"]
        assert all(none for _, none in r["record"]), "probe got a (zero) grad from the sync"
        assert r["probe_state"] == {}, "AdamATan2 stepped a parameter no rank used"
        assert torch.equal(r["params"]["nets.chess.ddp_probe"], torch.full((3,), 2.0))


def test_nonfinite_norm_on_one_rank_drops_the_step_on_all(ranks):
    (r0, r1), _ = ranks
    assert r0["core_steps"] == r1["core_steps"] == STEPS - 1


def test_rank0_writes_the_checkpoint(ranks):
    _, ck = ranks
    assert ck["world_size"] == WORLD and ck["games"] == ["othello", "chess"]


def _bn_worker(rank: int, init_file: str, root: str) -> None:
    import torch.distributed as dist

    from training.distributed import BatchNormStatSync

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=WORLD)
    try:
        torch.manual_seed(0)
        bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)])
        sync = BatchNormStatSync(bns)
        g = torch.Generator().manual_seed(100 + rank)
        if rank == 0:                  # layer 0: two forwards on rank 0 only
            for _ in range(2):
                bns[0](torch.randn(4, 3, 2, 2, generator=g) * 3 + 1)
        bns[1](torch.randn(4, 3, 2, 2, generator=g) * (rank + 1))   # layer 1: both ranks
        local = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
        sync()
        sync()                         # nothing new since: a no-op
        torch.save({"local": local,
                    "synced": [(m.running_mean.clone(), m.running_var.clone(),
                                int(m.num_batches_tracked)) for m in bns]},
                   os.path.join(root, f"bn{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_batchnorm_stat_sync(tmp_path):
    """One updating rank: exactly its statistics; several: their average;
    num_batches_tracked counts every rank's forwards; a sync with nothing new
    changes nothing."""
    ctx = torch.multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_bn_worker, args=(r, str(tmp_path / "store"), str(tmp_path)))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join(timeout=TIMEOUT_S)
            assert not p.is_alive(), "BatchNormStatSync hung"
            assert p.exitcode == 0
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
    r0, r1 = (torch.load(tmp_path / f"bn{r}.pt") for r in range(WORLD))
    assert all(torch.equal(a, b) for x, y in zip(r0["synced"], r1["synced"])
               for a, b in zip(x[:2], y[:2]))
    # layer 0: only rank 0 updated -> exactly its statistics, 2 forwards counted
    mean0, var0, n0 = r0["synced"][0]
    assert torch.equal(mean0, r0["local"][0][0]) and torch.equal(var0, r0["local"][0][1])
    assert n0 == r1["synced"][0][2] == 2
    # layer 1: both updated once -> the average, 2 forwards counted
    mean1, var1, n1 = r0["synced"][1]
    want_mean = ((r0["local"][1][0].double() + r1["local"][1][0].double()) / 2).float()
    want_var = ((r0["local"][1][1].double() + r1["local"][1][1].double()) / 2).float()
    assert torch.equal(mean1, want_mean) and torch.equal(var1, want_var) and n1 == 2
    assert not torch.equal(r0["local"][1][0], r1["local"][1][0])
