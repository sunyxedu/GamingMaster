"""training/rl_joint statistics under a real 2-process gloo group.

Rank 0 rolls out only chess, rank 1 chess and othello (go on neither): the
per-game moments must equal the single-process moments of the concatenated
data on BOTH ranks, the advantage normaliser and PopArt (statistics and the
rescaled head weights) must end bit-identical across ranks, and a non-finite
value — or a malformed input (unknown game, wrong mask shape) — on ONE rank
must raise on both instead of leaving the other blocked in a collective.
Statistics over an explicit process group equal the world's.
"""

import os
import socket
import time
from multiprocessing.connection import wait

import pytest
import torch

WORLD = 2
TIMEOUT_S = 120
GAMES = ["chess", "othello", "go"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _data(rank: int):
    g = torch.Generator().manual_seed(10 + rank)
    out = {"chess": (torch.randn(4, 9, generator=g) * (3 + rank) + rank,
                     torch.rand(4, 9, generator=g) > 0.3)}
    if rank == 1:
        out["othello"] = torch.randn(17, generator=g) * 0.2 - 1.0
    return out


def _worker(rank: int, port: int, root: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    import torch.distributed as dist
    import torch.nn as nn

    from model.components.heads import ValueHead
    from training.rl_joint import AdvantageNormalizer, PopArt, per_game_moments

    dist.init_process_group("gloo")
    try:
        data = _data(rank)
        moments = per_game_moments(data, GAMES)
        norm = AdvantageNormalizer(GAMES, decay=0.9)
        norm.update(data)
        norm.update(data)
        torch.manual_seed(0)                       # identical initial heads on both ranks
        heads = nn.ModuleDict({g: ValueHead(8) for g in GAMES})
        pa = PopArt(heads, decay=0.8)
        pa.update(data)
        raised = False
        try:
            bad = {"chess": torch.tensor([float("nan")])} if rank == 1 else {}
            per_game_moments(bad, GAMES)
        except ValueError:
            raised = True
        # malformed input on rank 0 only: rank 0 re-raises its own error,
        # rank 1 gets a RuntimeError — both after the (single) collective
        input_error = None
        try:
            if rank == 0:
                per_game_moments({"chess": (torch.zeros(3), torch.ones(4, dtype=torch.bool))},
                                 GAMES)
            else:
                per_game_moments({"othello": torch.zeros(3)}, GAMES)
        except (ValueError, RuntimeError) as e:
            input_error = type(e).__name__
        group = dist.new_group(list(range(WORLD)))
        group_moments = per_game_moments(data, GAMES, group=group)
        dist.barrier()                             # both ranks still in step afterwards
        torch.save({"moments": moments, "norm": norm.state_dict(), "popart": pa.state_dict(),
                    "heads": heads.state_dict(), "raised": raised,
                    "input_error": input_error, "group_moments": group_moments},
                   os.path.join(root, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(root):
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, str(root))) for r in range(WORLD)]
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
    return [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(WORLD)]


def test_statistics_are_global_and_rank_identical(tmp_path):
    from training.rl_joint import per_game_moments

    r0, r1 = _spawn(tmp_path)
    d0, d1 = _data(0), _data(1)
    cat = {"chess": (torch.cat([d0["chess"][0].flatten(), d1["chess"][0].flatten()]),
                     torch.cat([d0["chess"][1].flatten(), d1["chess"][1].flatten()])),
           "othello": d1["othello"]}
    want = per_game_moments(cat, GAMES, sync=False)
    for r in (r0, r1):
        for g in GAMES:
            assert r["moments"][g] == pytest.approx(want[g], rel=1e-12, abs=1e-12), g
        assert r["raised"]
        assert r["group_moments"] == r["moments"]
    assert r0["input_error"] == "ValueError" and r1["input_error"] == "RuntimeError"
    assert r0["moments"]["go"] == (0.0, 0.0, 0.0)
    assert r0["norm"] == r1["norm"]
    assert r0["popart"] == r1["popart"]
    for k, v in r0["heads"].items():
        assert torch.equal(v, r1["heads"][k]), k

