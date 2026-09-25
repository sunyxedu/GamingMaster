"""Rank-invariant gradient sync (model/rl_utils.py::sync_gradients).

Reproduces the production NCCL desync: a 4-GPU phase-2 IL run deadlocked
because one rank's autograd left the TRM q_head without gradients for one
accumulation window — the old ``if p.grad is not None`` loop then posted
two fewer collectives on that rank, NCCL paired mismatched tensors, and
the job died in a 600 s watchdog timeout after silently corrupting one
same-sized pair.

The divergent-coverage test here runs the exact failure shape on a real
2-process gloo group: with the old loop it deadlocks (guarded by a join
timeout); with sync_gradients it completes and the missing rank
contributes zeros. The nowhere-coverage test pins the preserved old
semantics: parameters without gradients on ANY rank stay untouched.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMALL_CFG = dict(
    d_model=32, n_heads=4, d_ff=64, encoder_channels=16, n_st_layers=1,
    core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, dropout=0.0,
    temporal_mixer="recurrent", predict_next_obs=True,
)

WORLD = 2
JOIN_TIMEOUT_S = 180


def _worker(rank, init_file, scenario, queue, sync_kw=None):
    """Runs in a spawned process: build the model, plant per-rank grads,
    sync, and report the first element of every grad back to the parent.

    With ``sync_kw`` (keyword arguments for sync_gradients) the planted
    gradients differ per element and per tensor — (rank+1)·(1 + arange) + i
    for the i-th parameter — and the FULL synced gradients are reported, so a
    bucket that scatters values back into the wrong tensor or offset shows."""
    import torch
    import torch.distributed as dist

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.config import ModelConfig
    from model.network import GeneralsNetwork
    from model.rl_utils import sync_gradients

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD,
    )
    try:
        torch.manual_seed(0)  # identical model on both ranks
        net = GeneralsNetwork(ModelConfig(**SMALL_CFG))

        # Deterministic per-rank gradients: rank 0 → 1.0, rank 1 → 2.0
        for i, (_, p) in enumerate(net.named_parameters()):
            p.grad = torch.full_like(p, float(rank + 1))
            if sync_kw is not None:
                p.grad += (rank + 1) * torch.arange(p.numel(), dtype=p.dtype).view_as(p) + i

        if scenario == "divergent" and rank == 1:
            # The production incident: one rank misses the q_head module
            net.trm.q_head.weight.grad = None
            net.trm.q_head.bias.grad = None
        elif scenario == "nowhere":
            # No rank has these grads (e.g. a head untouched by the objective)
            net.obs_encoder.next_obs_head.weight.grad = None
            net.obs_encoder.next_obs_head.bias.grad = None

        buf = io.StringIO()
        with redirect_stdout(buf):
            sync_gradients(net, WORLD, **(sync_kw or {}))

        # full grads travel as numpy (pickled by value): a torch tensor would
        # go through shared memory that dies with this process
        grads = {
            name: (None if p.grad is None
                   else p.grad.numpy().copy() if sync_kw is not None
                   else p.grad.flatten()[0].item())
            for name, p in net.named_parameters()
        }
        queue.put((rank, grads, "[grad-sync]" in buf.getvalue()))
    finally:
        dist.destroy_process_group()


def _run(scenario, tmp_path, sync_kw=None):
    ctx = torch.multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    init_file = tmp_path / f"store-{scenario}"
    procs = [
        ctx.Process(target=_worker, args=(r, str(init_file), scenario, queue, sync_kw))
        for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    results = {}
    # Drain the queue BEFORE joining — a large result could block the child
    # on the queue's pipe buffer while the parent blocks on join: deadlock.
    for _ in range(WORLD):
        rank, grads, warned = queue.get(timeout=JOIN_TIMEOUT_S)
        results[rank] = (grads, warned)
    for p in procs:
        p.join(timeout=JOIN_TIMEOUT_S)
        if p.is_alive():
            p.terminate()
            pytest.fail(f"worker pid={p.pid} deadlocked (collective desync?)")
        assert p.exitcode == 0, f"worker failed with exit code {p.exitcode}"
    return results


def test_uniform_coverage_averages_grads(tmp_path):
    results = _run("uniform", tmp_path)
    for rank in range(WORLD):
        grads, warned = results[rank]
        assert not warned
        for name, g in grads.items():
            assert g == pytest.approx(1.5), name  # (1.0 + 2.0) / 2


def test_divergent_coverage_completes_and_zero_fills(tmp_path):
    """The incident shape: rank 1 misses q_head grads. Must complete (old
    code deadlocks here), average q_head as (1.0 + 0)/2 on BOTH ranks,
    and warn on the rank that came up short."""
    results = _run("divergent", tmp_path)
    for rank in range(WORLD):
        grads, warned = results[rank]
        assert warned == (rank == 1)
        for name, g in grads.items():
            if name.startswith("trm.q_head."):
                assert g == pytest.approx(0.5), f"rank {rank} {name}"
            else:
                assert g == pytest.approx(1.5), f"rank {rank} {name}"


def test_nowhere_coverage_skips_param_on_all_ranks(tmp_path):
    """Parameters with no gradient on ANY rank keep grad=None everywhere —
    the optimizer never sees them, exactly like the pre-fix behavior for
    heads untouched by the objective (RL never trains next_obs_head)."""
    results = _run("nowhere", tmp_path)
    for rank in range(WORLD):
        grads, warned = results[rank]
        assert not warned
        for name, g in grads.items():
            if name.startswith("obs_encoder.next_obs_head."):
                assert g is None, f"rank {rank} {name}"
            else:
                assert g == pytest.approx(1.5), f"rank {rank} {name}"


@pytest.mark.parametrize("bucket_mb", [0.0, 0.001, 1000.0])
def test_bucketed_sync_is_exact_per_element(tmp_path, bucket_mb):
    """Per-tensor (0), many small buckets and one big bucket all average
    every element into the right tensor and offset; rank 1's missing q_head
    is zero-filled; ``warn_partial=False`` silences the expected-gap warning."""
    results = _run("divergent", tmp_path,
                   sync_kw=dict(bucket_mb=bucket_mb, warn_partial=False))
    for rank in range(WORLD):
        grads, warned = results[rank]
        assert not warned
        for i, (name, g) in enumerate(grads.items()):
            g = torch.from_numpy(g)
            idx = torch.arange(g.numel(), dtype=g.dtype).view_as(g)
            if name.startswith("trm.q_head."):
                want = (1 + idx + i) / 2                   # (rank-0 value + 0) / 2
            else:
                want = 1.5 + 1.5 * idx + i
            assert torch.equal(g, want), f"rank {rank} {name}"


def test_bucketed_sync_keeps_nowhere_params_none(tmp_path):
    results = _run("nowhere", tmp_path, sync_kw=dict(bucket_mb=0.001, warn_partial=False))
    for rank in range(WORLD):
        grads, _ = results[rank]
        for name, g in grads.items():
            assert (g is None) == name.startswith("obs_encoder.next_obs_head."), name
