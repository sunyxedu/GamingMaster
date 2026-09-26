"""Time the data-parallel gradient sync (model.rl_utils.sync_gradients) per
bucket size, on the backend and devices the trainers would use.

A synthetic parameter set of about ``--params`` elements (transformer-like
blocks of width ``--width`` plus ``--small`` tiny adapter tensors, the shape
mix of a multi-game network) gets random gradients on every rank; each
``--bucket_mb`` value is timed over ``--reps`` syncs after one warm-up.  Run
it under torchrun like a trainer (same MIG pinning / --dist_backend logic):

    # 2 CPU ranks over gloo
    uv run torchrun --standalone --nproc_per_node=2 scripts/bench_grad_sync.py --device cpu

    # 2 MIG slices (gloo on CUDA tensors), a 208M-parameter model
    torchrun --standalone --nproc_per_node=2 scripts/bench_grad_sync.py \\
        --params 208e6 --bucket_mb 0 5 25 100 400

Rank 0 prints one line per bucket size: buckets, mean / min seconds per sync
and the algorithm bandwidth (gradient bytes / time).  ``--zero1`` adds, per
bucket size, the extra per-step cost of ``--shard_optimizer`` (training/zero.py):
the owners' broadcast of the updated weights (every parameter stepped).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.distributed import DIST_BACKENDS, pin_rank_device  # noqa: E402  (no CUDA yet)

pin_rank_device()           # before anything below can touch CUDA

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from model.rl_utils import _grad_buckets, sync_gradients  # noqa: E402
from training.distributed import init_distributed  # noqa: E402
from training.zero import Zero1  # noqa: E402


def synthetic_params(n_params: float, width: int, small: int) -> nn.Module:
    """Blocks of 4 d×d + 2 4d×d matrices and 6 d-vectors until ``n_params``,
    plus ``small`` 256-element tensors (per-game adapters, norms)."""
    d = width
    block = [(d, d)] * 4 + [(4 * d, d), (d, 4 * d)] + [(d,)] * 6
    per_block = sum(torch.Size(s).numel() for s in block)
    shapes = block * max(1, round(n_params / per_block)) + [(256,)] * small
    m = nn.Module()
    m.params = nn.ParameterList(nn.Parameter(torch.empty(s)) for s in shapes)
    return m


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--params", type=float, default=208e6)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--small", type=int, default=400)
    p.add_argument("--bucket_mb", type=float, nargs="+", default=[0, 1, 5, 25, 100, 400])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--dist_backend", choices=DIST_BACKENDS, default="auto")
    p.add_argument("--zero1", action="store_true",
                   help="also time --shard_optimizer's per-step weight broadcast")
    args = p.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env, device = init_distributed(device, 10.0, args.dist_backend)
    if not env.enabled:
        raise SystemExit("run under torchrun with --nproc_per_node >= 2")
    model = synthetic_params(args.params, args.width, args.small).to(device)
    n = sum(p.numel() for p in model.parameters())
    nbytes = 4 * n
    gen = torch.Generator(device=device).manual_seed(env.rank)
    for q in model.parameters():
        q.grad = torch.randn(q.shape, generator=gen, device=device)
    grads = [q.grad for q in model.parameters()]

    def sync_once(bucket_mb: float) -> float:
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        sync_gradients(model, env.world, warn_partial=False, bucket_mb=bucket_mb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - t

    opt = torch.optim.SGD(model.parameters(), lr=0.0)     # Zero1 reads its param order

    def broadcast_once(bucket_mb: float) -> float:
        zero = Zero1(model, opt, env.rank, env.world, bucket_mb=bucket_mb)
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        zero._broadcast_params(range(len(zero.params)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - t

    if env.is_main:
        print(f"[bench] world={env.world} backend={dist.get_backend()} device={device} "
              f"params={n / 1e6:.1f}M ({nbytes / 2 ** 20:.0f} MiB fp32) "
              f"tensors={len(grads)}", flush=True)
    for mb in args.bucket_mb:
        n_buckets = len(grads) if mb <= 0 else len(_grad_buckets(grads, int(mb * 2 ** 20)))
        sync_once(mb)                                   # warm-up (allocator, pinned memory)
        times = [sync_once(mb) for _ in range(args.reps)]
        if env.is_main:
            mean = sum(times) / len(times)
            print(f"[bench] bucket_mb={mb:g} buckets={n_buckets} mean={mean:.3f}s "
                  f"min={min(times):.3f}s algbw={nbytes / min(times) / 1e9:.2f} GB/s",
                  flush=True)
        if args.zero1:
            broadcast_once(mb)                          # warm-up
            times = [broadcast_once(mb) for _ in range(args.reps)]
            if env.is_main:
                mean = sum(times) / len(times)
                print(f"[bench] zero1_broadcast bucket_mb={mb:g} mean={mean:.3f}s "
                      f"min={min(times):.3f}s", flush=True)
    env.barrier()
    env.destroy()


if __name__ == "__main__":
    main()
