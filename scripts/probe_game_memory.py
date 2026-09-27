"""Per-game batch geometry and training-step memory at a given config: the
numbers to choose ``--game_batch`` / ``--game_max_len`` with before games join a
joint run (training/train_multi_game.py; e.g. a resume that adds games).

Each game runs alone — its network (the shared core + its own adapters) built
exactly as train_multi_game builds a fresh game (``--override`` + shard meta,
the same crop policy), then ``--batches`` real training batches through the
micro-step (prepare_batch → batch_loss forward + backward, bf16 autocast with
``--bf16``).  Per game it prints the corpus size, the tokens per frame the core
sees (the encoder's tokens + 2 decision tokens), the batch shape, and the peak
memory of the step above the resident weights and gradients (the activations:
what the per-game batch size moves; the joint run adds its resident model /
optimizer / EMA state once, and a micro-batch's activations are freed before
the next one).

    uv run python scripts/probe_game_memory.py --data xiangqi=~/gm-data/xiangqi/train \\
        --data hunl=~/gm-data/hunl/train --batch_size 16 --max_len 64 \\
        --game_batch hunl=128 --batches 4 --bf16 \\
        --override d_model=512 n_heads=8 core_type=transformer core_layers=12 ...
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_processing.games.dataset import collate_game_sequences, training_sampler  # noqa: E402
from training.train_il_game import (  # noqa: E402
    UNCROPPABLE,
    apply_meta_arch_defaults,
    batch_loss,
    build_net_tolerant,
    check_crop_policy,
    check_meta_arch,
    default_tf,
    open_dataset,
    parse_overrides,
    pick_device,
    prepare_batch,
)
from training.train_multi_game import _pairs  # noqa: E402

GiB = 1024 ** 3


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", action="append", required=True, metavar="GAME=DIR")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="network config (train_multi_game --override)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--game_batch", action="append", default=[], metavar="GAME=N")
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--game_max_len", action="append", default=[], metavar="GAME=N")
    p.add_argument("--batches", type=int, default=4, help="batches per game (max over them)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--il_legal_mask", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    device = torch.device(a.device) if a.device else pick_device()
    cuda = device.type == "cuda"
    core_kw = parse_overrides(a.override)
    game_batch = {g: int(n) for g, n in _pairs(a.game_batch, "N").items()}
    game_max_len = {g: int(n) for g, n in _pairs(a.game_max_len, "N").items()}
    torch.manual_seed(a.seed)
    print(f"{'game':<15} {'frames':>12} {'seqs':>10} {'S/frame':>7} {'B':>4} {'T mean':>7} "
          f"{'T max':>5} {'tok/batch':>10} {'weights':>8} {'act peak':>9} {'s/step':>7}")
    for g, d in _pairs(a.data).items():
        d = os.path.expanduser(d)
        ml = game_max_len.get(g) if g in UNCROPPABLE else game_max_len.get(g, a.max_len)
        ds = open_dataset(g, d, ml)
        kw = apply_meta_arch_defaults(g, dict(core_kw), ds.shard_meta)
        net, dropped = build_net_tolerant(g, kw)
        if dropped:
            print(f"  [{g}] config fields unknown to its config, ignored: {dropped}")
        check_meta_arch(g, net.cfg, ds.shard_meta)
        check_crop_policy(g, net.cfg, ml)
        net = net.to(device).train()
        S = net.obs_encoder.n_tokens + 2
        bs = game_batch.get(g, a.batch_size)
        loader = iter(DataLoader(ds, batch_size=bs, sampler=training_sampler(ds),
                                 collate_fn=collate_game_sequences))
        n_par = sum(t.numel() * t.element_size() for t in net.parameters())
        peak, t_sum, T_sum, T_max, n = 0.0, 0.0, 0, 0, 0
        for _ in range(a.batches):
            batch = next(loader)
            mb = prepare_batch(g, batch, device, default_tf(g), cfg=net.cfg,
                               il_legal_mask=a.il_legal_mask)
            lengths = mb["frame_mask"].sum(1)
            T_sum += int(lengths.sum())
            T_max = max(T_max, int(mb["frame_mask"].shape[1]))
            net.zero_grad(set_to_none=False)       # gradients resident, as in training
            if cuda:
                torch.cuda.synchronize(device)
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            loss, _, _ = batch_loss(net, mb, device, 1.0, 0.0, a.bf16, z_loss_weight=1e-4)
            loss.backward()
            if cuda:
                torch.cuda.synchronize(device)
                peak = max(peak, (torch.cuda.max_memory_allocated(device) - base) / GiB)
            t_sum += time.time() - t0
            n += 1
            del loss, mb, batch
        frames = ds.total_frames()
        seqs = len(ds)
        tmean = T_sum / (n * bs)
        print(f"{g:<15} {frames:>12,} {seqs:>10,} {S:>7} {bs:>4} {tmean:>7.1f} {T_max:>5} "
              f"{int(T_sum / n * S):>10,} {2 * n_par / GiB:>7.2f}G "
              f"{peak:>8.2f}G {t_sum / n:>7.2f}", flush=True)
        del net, loader, ds
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
