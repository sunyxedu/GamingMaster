"""Data-parallel training/train_multi_game.py on a real 2-process gloo group.

Two spawned ranks run the real trainer (torchrun-style environment) for a few
optimizer steps on tiny synthetic shards of two games, one micro-batch per
rank per step, so the ranks regularly draw DIFFERENT games: a game's
encoder/decoder then has a gradient on one rank only.  Afterwards every
parameter, buffer (MoE balance biases included) and EMA shadow must be
bit-identical across ranks; a parameter that is trainable but used by no
forward must keep grad None through every sync (AdamATan2 never sees it: no
state, no weight decay — its value is untouched); and a non-finite gradient
norm on ONE rank must drop the step on BOTH.  A second run resumes a
single-process checkpoint on 2 ranks with a global micro-batch count (3 =
2 + 1, unequal per rank).  Validation (batches split over the ranks) must
match single-process validation of the same weights.  Where CUDA is
available the first run is repeated on CUDA tensors over gloo (the MIG-slice
mode: --device cuda --dist_backend gloo, every rank pinned to its own slice
when the allocation has one per rank), with the same assertions, and so is
the BatchNorm statistics merge.
"""

import math
import os
import socket
import time
from multiprocessing.connection import wait

import pytest
import torch
from torch.utils.data import DataLoader

from data_processing.games.dataset import collate_game_sequences, save_shard
from tests.gloo_devices import DEVICES, local_ranks, needs_cuda
from tests.test_core_moe import CORE, _board_samples

WORLD = 2
STEPS = 5
NAN_STEP = 3          # rank 1 poisons its (already synced) gradient at this step
RESUME_AT = 2         # the resumed run starts from a single-process step-2 checkpoint
TIMEOUT_S = 300


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _common_args(root: str, save_dir: str, kind: str = "cpu"):
    dev = ["--device", "cpu"] if kind == "cpu" else ["--device", "cuda", "--dist_backend", "gloo"]
    return ["--data", f"othello={root}/othello", "--data", f"chess={root}/chess",
            "--val", f"othello={root}/othello", "--batch_size", "2",
            "--warmup_steps", "1", "--eval_every", "2", "--log_every", "2",
            "--ema_rate", "0.9", "--moe_balance_rate", "0.01", "--clip_grad", "0.5",
            "--seed", "3", *dev, "--save_dir", save_dir,
            "--override", *[f"{k}={v}" for k, v in CORE.items()]]


def _tag(mode: str, kind: str) -> str:
    return mode if kind == "cpu" else f"{mode}_{kind}"


def _worker(rank: int, port: int, root: str, mode: str, kind: str = "cpu",
            lrs=(0, 1)) -> None:
    # main() itself pins a MIG slice per LOCAL_RANK before touching CUDA
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(lrs[rank]), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    from training import train_multi_game as tmg

    build, sync = tmg.build_net_tolerant, tmg.sync_gradients
    prepare_rows, shard_eval = tmg.prepare_rows, tmg.evaluate_sharded
    probe_on = mode == "k1"

    def build_with_probe(game, arch):
        net, dropped = build(game, arch)
        if game == "chess" and probe_on:
            # trainable, but outside every forward: no rank ever has its grad
            net.ddp_probe = torch.nn.Parameter(torch.full((3,), 2.0))
        return net, dropped

    record = []     # per step: (games with a local encoder grad, probe grad None after sync)
    n_mb = [0]      # micro-batches of the current step on this rank
    per_step = []   # n_mb per step
    evals = []      # (sharded result, single-process result on the full val set)

    def counting_prepare_rows(*a, **kw):
        n_mb[0] += 1
        return prepare_rows(*a, **kw)

    def recording_sync(model, world, **kw):
        local = sorted(g for g in model.games
                       if any(p.grad is not None for p in model[g].obs_encoder.parameters()))
        sync(model, world, **kw)
        record.append((local, model["chess"].ddp_probe.grad is None if probe_on else True))
        per_step.append(n_mb[0])
        n_mb[0] = 0
        if probe_on and rank == 1 and len(record) == NAN_STEP:
            next(p for p in model.parameters() if p.grad is not None).grad.fill_(math.nan)

    def checking_eval(model, loaders, device, value_weight, bf16):
        res = shard_eval(model, loaders, device, value_weight, bf16)
        full = {g: DataLoader(ld.dataset, batch_size=2, collate_fn=collate_game_sequences)
                for g, ld in loaders.items()}
        evals.append((res, tmg.evaluate(model, full, device, value_weight, bf16)))
        return res

    tmg.build_net_tolerant, tmg.sync_gradients = build_with_probe, recording_sync
    tmg.prepare_rows, tmg.evaluate_sharded = counting_prepare_rows, checking_eval
    if mode == "k1":
        extra = ["--steps", str(STEPS), "--games_per_step", "1", "--bucket_mb", "0.01"]
    else:
        extra = ["--steps", str(STEPS), "--global_games_per_step", "3",
                 "--resume", f"{root}/sp/multi_latest.pt"]
    tag = _tag(mode, kind)
    out = tmg.main([*_common_args(root, f"{root}/ck_{tag}", kind), *extra,
                    "--dist_timeout", "2"])
    model, ema, opt = out["model"], out["ema"], out["optimizer"]
    probe = model["chess"].ddp_probe if probe_on else None
    core_p = model["othello"].core.parameters().__next__()
    torch.save({
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
        "ema": {n: v.clone() for n, v in ema.shadow.items()},
        "record": record,
        "per_step": per_step,
        "evals": evals,
        "probe_in_optimizer": any(p is probe for g in opt.param_groups for p in g["params"]),
        "probe_state": dict(opt.state.get(probe, {})) if probe_on else {},
        "core_steps": opt.state[core_p]["step"],
        "device": str(next(model.parameters()).device),
    }, os.path.join(root, f"{tag}_rank{rank}.pt"))


def _spawn(root, mode, kind="cpu"):
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    lrs = local_ranks(WORLD, kind)
    procs = [ctx.Process(target=_worker, args=(r, port, str(root), mode, kind, lrs))
             for r in range(WORLD)]
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
    tag = _tag(mode, kind)
    out = [torch.load(root / f"{tag}_rank{r}.pt", weights_only=False, map_location="cpu")
           for r in range(WORLD)]
    ck = torch.load(root / f"ck_{tag}" / "multi_final.pt", weights_only=False,
                    map_location="cpu")
    return out, ck


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddp")
    for g in ("othello", "chess"):
        (root / g).mkdir()
        save_shard(str(root / g / "a.pt"), g, _board_samples(g, 6, 6))
    return root


@pytest.fixture(scope="module")
def ranks(shards):
    return _spawn(shards, "k1")


@pytest.fixture(scope="module")
def cuda_ranks(shards):
    """The ``ranks`` run on CUDA tensors over gloo."""
    return _spawn(shards, "k1", "cuda")


@pytest.fixture(scope="module")
def resumed(shards):
    """2 ranks, --global_games_per_step 3, resumed from a single-process
    checkpoint of step RESUME_AT (written in a spawned process: main()
    reseeds the global RNGs)."""
    ctx = torch.multiprocessing.get_context("spawn")
    p = ctx.Process(target=_single_process_prefix, args=(str(shards),))
    p.start()
    p.join(timeout=TIMEOUT_S)
    assert p.exitcode == 0
    return _spawn(shards, "global")


def _single_process_prefix(root: str) -> None:
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    torch.set_num_threads(1)
    from training import train_multi_game as tmg
    tmg.main([*_common_args(root, f"{root}/sp"), "--steps", str(RESUME_AT),
              "--ckpt_every", str(RESUME_AT), "--games_per_step", "2"])


def _assert_rank_identical(r0, r1):
    for key in ("params", "buffers", "ema"):
        assert r0[key].keys() == r1[key].keys()
        for n in r0[key]:
            assert torch.equal(r0[key][n], r1[key][n]), f"{key} {n} differs across ranks"


def _assert_eval_matches_single_process(r0, r1, n_evals):
    assert len(r0["evals"]) == len(r1["evals"]) == n_evals
    for (sh0, full0), (sh1, _) in zip(r0["evals"], r1["evals"]):
        assert sh0 == sh1                       # every rank returns the global result
        assert sh0.keys() == full0.keys() == {"othello"}
        for k, v in full0["othello"].items():
            assert sh0["othello"][k] == pytest.approx(v, rel=1e-5, abs=1e-6), k


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
    _assert_rank_identical(r0, r1)
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


def test_sharded_validation_matches_single_process(ranks):
    (r0, r1), _ = ranks
    _assert_eval_matches_single_process(r0, r1, n_evals=3)       # steps 2, 4, 5


@needs_cuda
def test_cuda_tensors_over_gloo_keep_every_guarantee(cuda_ranks):
    """--device cuda --dist_backend gloo: the whole contract of the CPU run —
    different draws, bit-identical parameters / buffers / EMA, grad-None
    unused parameter, collective step drop, rank-0 checkpoint, sharded
    validation equal to single-process validation."""
    (r0, r1), ck = cuda_ranks
    assert r0["device"].startswith("cuda") and r1["device"].startswith("cuda")
    local0 = [loc for loc, _ in r0["record"]]
    local1 = [loc for loc, _ in r1["record"]]
    assert len(local0) == STEPS and any(a != b for a, b in zip(local0, local1))
    _assert_rank_identical(r0, r1)
    for r in (r0, r1):
        assert all(none for _, none in r["record"]) and r["probe_state"] == {}
        assert torch.equal(r["params"]["nets.chess.ddp_probe"], torch.full((3,), 2.0))
    assert r0["core_steps"] == r1["core_steps"] == STEPS - 1
    assert ck["world_size"] == WORLD
    _assert_eval_matches_single_process(r0, r1, n_evals=3)


def test_global_micro_batch_count_splits_over_ranks(resumed):
    (r0, r1), _ = resumed
    n = STEPS - RESUME_AT
    assert r0["per_step"] == [2] * n and r1["per_step"] == [1] * n


def test_resume_on_more_ranks_continues_and_stays_identical(resumed):
    (r0, r1), ck = resumed
    _assert_rank_identical(r0, r1)
    # AdamATan2's step counter continues from the checkpoint (no step dropped)
    assert r0["core_steps"] == r1["core_steps"] == STEPS
    assert ck["world_size"] == WORLD
    _assert_eval_matches_single_process(r0, r1, n_evals=2)       # steps 4, 5


def _bn_worker(rank: int, init_file: str, root: str, kind: str = "cpu", lrs=(0, 1)) -> None:
    import torch.distributed as dist

    from tests.gloo_devices import worker_device
    from training.distributed import BatchNormStatSync

    dev = worker_device(kind, lrs[rank])
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=WORLD)
    try:
        torch.manual_seed(0)
        bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3),
                                   torch.nn.BatchNorm2d(3),
                                   torch.nn.BatchNorm2d(3, momentum=None)]).to(dev)
        for m in bns:                  # a non-trivial step-start state P
            m(torch.randn(8, 3, 2, 2, generator=torch.Generator().manual_seed(7)).to(dev) + 2)
        prev = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
        sync = BatchNormStatSync(bns)
        g = torch.Generator().manual_seed(100 + rank)
        same = torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(5)) * 2 - 1
        inputs = {1: [], 3: []}
        if rank == 0:                  # layer 0: two forwards on rank 0 only
            for _ in range(2):
                bns[0]((torch.randn(4, 3, 2, 2, generator=g) * 3 + 1).to(dev))
        x = torch.randn(4, 3, 2, 2, generator=g) * (rank + 1)
        bns[1](x.to(dev))              # layer 1: both ranks, one forward each
        inputs[1].append(x)
        for _ in range(2 - rank):      # layer 2: the same batch, 2 + 1 forwards
            bns[2](same.to(dev))
        for _ in range(1 + rank):      # layer 3 (cumulative): 1 + 2 forwards
            x = torch.randn(4, 3, 2, 2, generator=g) + rank
            bns[3](x.to(dev))
            inputs[3].append(x)
        local = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
        sync()
        after = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
        sync()                         # nothing new since: a no-op
        cpu = [[tuple(t.cpu() for t in pair) for pair in stats] for stats in (prev, local)]
        torch.save({"prev": cpu[0], "local": cpu[1], "inputs": inputs, "same": same,
                    "noop": all(torch.equal(a[0], m.running_mean)
                                and torch.equal(a[1], m.running_var)
                                for a, m in zip(after, bns)),
                    "synced": [(m.running_mean.cpu(), m.running_var.cpu(),
                                int(m.num_batches_tracked)) for m in bns]},
                   os.path.join(root, f"bn{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _bn_reference(momentum, start, batches):
    """A fresh single-process BatchNorm2d at ``start`` (mean, var, count) fed
    ``batches`` in order: the running statistics one process would hold."""
    bn = torch.nn.BatchNorm2d(3, momentum=momentum)
    with torch.no_grad():
        bn.running_mean.copy_(start[0])
        bn.running_var.copy_(start[1])
        bn.num_batches_tracked.fill_(start[2])
    for x in batches:
        bn(x)
    return bn.running_mean, bn.running_var


def _batch_stats(x):
    return x.mean((0, 2, 3)).double(), x.var((0, 2, 3), unbiased=True).double()


@pytest.mark.parametrize("kind", DEVICES)
def test_batchnorm_stat_sync(tmp_path, kind):
    """One updating rank: exactly its statistics.  Several: the step-start
    statistics keep the single-process weight (1-m)^N for the N forwards of
    all ranks, the batches share the rest — exact when the batches coincide
    or for the (order-free) cumulative average.  num_batches_tracked counts
    every rank's forwards; a sync with nothing new changes nothing.  (CUDA:
    the CPU references allow for the CUDA kernels' float32 rounding.)"""
    ctx = torch.multiprocessing.get_context("spawn")
    lrs = local_ranks(WORLD, kind)
    procs = [ctx.Process(target=_bn_worker,
                         args=(r, str(tmp_path / "store"), str(tmp_path), kind, lrs))
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
    tol = {} if kind == "cpu" else {"rtol": 1e-5, "atol": 1e-6}
    r0, r1 = (torch.load(tmp_path / f"bn{r}.pt") for r in range(WORLD))
    assert r0["noop"] and r1["noop"]
    assert all(torch.equal(a, b) for x, y in zip(r0["synced"], r1["synced"])
               for a, b in zip(x[:2], y[:2]))
    counts = [s[2] for s in r0["synced"]]
    assert counts == [s[2] for s in r1["synced"]] == [1 + 2, 1 + 2, 1 + 3, 1 + 3]
    # layer 0: only rank 0 updated -> exactly its statistics
    mean0, var0, _ = r0["synced"][0]
    assert torch.equal(mean0, r0["local"][0][0]) and torch.equal(var0, r0["local"][0][1])
    # layer 1: one forward on each rank -> P keeps 0.9^2, each batch 0.19 / 2
    pm, pv = (t.double() for t in r0["prev"][1])
    (m_a, v_a), (m_b, v_b) = _batch_stats(r0["inputs"][1][0]), _batch_stats(r1["inputs"][1][0])
    mean1, var1, _ = r0["synced"][1]
    torch.testing.assert_close(mean1.double(), 0.81 * pm + 0.19 * (m_a + m_b) / 2, **tol)
    torch.testing.assert_close(var1.double(), 0.81 * pv + 0.19 * (v_a + v_b) / 2, **tol)
    # layer 2: the same batch 2 + 1 times -> what one process holds after 3
    prev2 = (*r0["prev"][2], 1)
    want = _bn_reference(0.1, prev2, [r0["same"]] * 3)
    torch.testing.assert_close(r0["synced"][2][0], want[0], **tol)
    torch.testing.assert_close(r0["synced"][2][1], want[1], **tol)
    # layer 3 (momentum=None): the cumulative average over all 1 + 2 batches
    prev3 = (*r0["prev"][3], 1)
    want = _bn_reference(None, prev3, r0["inputs"][3] + r1["inputs"][3])
    torch.testing.assert_close(r0["synced"][3][0], want[0], **tol)
    torch.testing.assert_close(r0["synced"][3][1], want[1], **tol)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [4, 5, 7])
def test_micro_batch_plan(world, k):
    """--global_games_per_step k: the ranks' counts sum to k (differing by at
    most one) and every micro-batch weighs 1/k after the rank average;
    --games_per_step stays per rank; world 1 is the historical scaling."""
    from training.train_multi_game import micro_batch_plan

    plans = [micro_batch_plan(0, k, 5, r, world) for r in range(world)]
    counts = [K for K, _, _ in plans]
    assert sum(counts) == k and max(counts) - min(counts) <= 1
    assert all(kg == k and world * kd == pytest.approx(k) for _, kg, kd in plans)
    assert micro_batch_plan(2, 0, 5, world - 1, world) == (2, 2 * world, 2)
    assert micro_batch_plan(0, 0, 5, 0, world) == (5, 5 * world, 5)
    assert micro_batch_plan(0, k, 5, 0, 1) == (k, k, k)
    with pytest.raises(SystemExit, match="not both"):
        micro_batch_plan(2, k, 5, 0, world)
    with pytest.raises(SystemExit, match="idle"):
        micro_batch_plan(0, world - 1 or 1, 5, 0, world + 1)


def test_resume_restores_the_training_state_exactly(shards, tmp_path):
    """multi_latest.pt carries everything a resume needs: resuming at its
    step trains nothing and reproduces the run's final checkpoint and
    optimizer state bit for bit; checkpoints without optimizer state, or of
    other games, are refused."""
    from training import train_multi_game as tmg

    root = str(shards)
    tmg.main([*_common_args(root, str(tmp_path / "a")), "--steps", "2", "--ckpt_every", "2"])
    latest = torch.load(tmp_path / "a" / "multi_latest.pt", weights_only=False)
    final = torch.load(tmp_path / "a" / "multi_final.pt", weights_only=False)
    assert latest["step"] == 2 and "optimizer_state" not in final
    out = tmg.main([*_common_args(root, str(tmp_path / "b")), "--steps", "2",
                    "--resume", str(tmp_path / "a" / "multi_latest.pt")])
    again = torch.load(tmp_path / "b" / "multi_final.pt", weights_only=False)
    for key in ("model_state", "train_model_state"):
        assert final[key].keys() == again[key].keys()
        assert all(torch.equal(v, again[key][k]) for k, v in final[key].items()), key
    want, got = latest["optimizer_state"]["state"], out["optimizer"].state_dict()["state"]
    assert want.keys() == got.keys()
    assert all(torch.equal(torch.as_tensor(want[i][k]), torch.as_tensor(got[i][k]))
               for i in want for k in want[i])
    with pytest.raises(SystemExit, match="not resumable"):
        tmg.main([*_common_args(root, str(tmp_path / "c")), "--steps", "3",
                  "--resume", str(tmp_path / "a" / "multi_final.pt")])
    with pytest.raises(SystemExit, match="trained games"):
        tmg.main(["--data", f"chess={root}/chess", "--device", "cpu",
                  "--save_dir", str(tmp_path / "d"),
                  "--resume", str(tmp_path / "a" / "multi_latest.pt")])


def test_rank_seeds_of_a_seed_sweep_do_not_overlap():
    """rank 0 keeps --seed (single-process streams); the ranks of neighbouring
    seeds (a sweep) and of a resumed run get streams of their own."""
    from training.distributed import rank_seed

    assert rank_seed(7, 0) == 7
    fresh = {rank_seed(s, r) for s in range(16) for r in range(8)}
    assert len(fresh) == 16 * 8
    resumed = {rank_seed(s, r, t) for s in range(16) for r in range(8) for t in (500, 1000)}
    assert len(resumed) == 16 * 8 * 2 and not resumed & fresh
