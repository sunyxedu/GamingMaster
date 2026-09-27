"""Does a pretrained multi-game network use the frames before its training window?

Pretraining (and the RL IL replay) crop records to W-frame windows, each started with
a fresh recurrent state: the network never learns from, and never sees during training,
more than W − 1 frames of history.  Per game, on validation records long enough, the
imitation loss (and value MSE) of the same target frames under different histories:

    fresh    the target frames as a window of their own      (history 0, as trained)
    ctx      the H frames before them as well                 (history H < W, in range)
    full     the whole record from its first frame            (history beyond W: the
                                                              recurrent state of play)

ctx < fresh: the history carries information the frame does not (the crop costs);
full > ctx: the state accumulated beyond the trained horizon is out of distribution
(what a full-game rollout sees).  Two target spans: [W, W + S) and, for long records,
[4W, 4W + S).

    uv run python scripts/context_probe.py --ckpt multi_latest.pt \\
        --data chess=val_dir --data go=val_dir ... [--window 64 --span 32 --n 48]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_processing.games.dataset import GameShardDataset, collate_game_sequences  # noqa: E402
from training.eval_multi_game import load_multi  # noqa: E402
from training.train_il_game import prepare_batch, value_losses  # noqa: E402


def _slice(s: Dict, a: int, b: int) -> Dict:
    return {"obs": {k: v[a:b] for k, v in s["obs"].items()},
            "tgt": {k: v[a:b] for k, v in s["tgt"].items()}, "meta": s.get("meta", {})}


@torch.no_grad()
def span_losses(net, game: str, records: List[Dict], start: int, span: int, history: int,
                device: torch.device, batch: int, bf16: bool) -> Dict[str, float]:
    """Mean IL loss / value MSE of frames [start, start + span) of every record, the
    forward starting ``history`` frames earlier (clipped at the record's start)."""
    a = max(0, start - history)
    il_sum = v_sum = 0.0
    n = 0
    for i in range(0, len(records), batch):
        chunk = [_slice(r, a, start + span) for r in records[i:i + batch]]
        mb = prepare_batch(game, collate_game_sequences(chunk), device, tf_prob=1.0,
                           cfg=net.cfg, il_legal_mask=True)
        mask = mb["frame_mask"].clone()
        mask[:, :start - a] = False                       # score the target frames only
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"],
                       frame_mask=mb["frame_mask"])
            il = net.action_decoder.il_loss(pred, mb["tgt"], mask)["loss"]
            _, v_mse = value_losses(net, pred, mb["tgt"]["value"], mask)
        il_sum += float(il) * len(chunk)
        v_sum += float(v_mse) * len(chunk)
        n += len(chunk)
    return {"il": il_sum / max(n, 1), "value_mse": v_sum / max(n, 1)}


def long_records(ds: GameShardDataset, need: int, n: int, seed: int) -> List[Dict]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for idx in torch.randperm(len(ds), generator=g).tolist()[: 50 * n]:
        s = ds[idx]
        if next(iter(s["obs"].values())).shape[0] >= need:
            out.append(s)
            if len(out) == n:
                break
    return out


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", action="append", default=[], metavar="GAME=DIR")
    p.add_argument("--window", type=int, default=64, help="the training crop W")
    p.add_argument("--span", type=int, default=32, help="target frames per record")
    p.add_argument("--n", type=int, default=48, help="records per game and span")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--raw", action="store_true", help="training weights instead of the EMA")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device)
    model = load_multi(a.ckpt, device, raw=a.raw)
    W, S = a.window, a.span
    res: Dict[str, Dict] = {}
    for item in a.data:
        game, _, path = item.partition("=")
        if game not in model.games:
            print(f"[skip] {game}: not in the checkpoint")
            continue
        net = model[game]
        res[game] = {}
        try:
            ds = GameShardDataset(path, game=game)
        except Exception as e:                            # e.g. gio record directories
            print(f"[skip] {game}: {e}")
            continue
        for start in (W, 4 * W):
            recs = long_records(ds, start + S, a.n, seed=start)
            if len(recs) < max(4, a.n // 4):
                print(f"[skip] {game} frames [{start},{start + S}): {len(recs)} long records")
                continue
            row = {"records": len(recs)}
            for name, hist in (("fresh", 0), ("ctx", W - S), ("full", start)):
                row[name] = span_losses(net, game, recs, start, S, hist, device, a.batch, a.bf16)
            res[game][f"frames_{start}_{start + S}"] = row
            f, c, u = row["fresh"]["il"], row["ctx"]["il"], row["full"]["il"]
            print(f"{game:15s} frames [{start},{start + S}) n={len(recs):3d}  il fresh {f:.4f}  "
                  f"ctx{W - S} {c:.4f} ({100 * (c - f) / f:+.1f}%)  full {u:.4f} "
                  f"({100 * (u - f) / f:+.1f}%)   value_mse {row['fresh']['value_mse']:.4f} / "
                  f"{row['ctx']['value_mse']:.4f} / {row['full']['value_mse']:.4f}", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()
