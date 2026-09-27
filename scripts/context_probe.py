"""How much of a game's history does a pretrained multi-game network use?

The network carries history only in the recurrent state of its spatiotemporal layers.
Per game, on validation records long enough, the imitation loss (per decoder head) and
the value error of the SAME target frames [s, s + S) when the forward starts h frames
earlier (the recurrent state built from h frames of history, h = 0 … the whole record
before them), for several positions s in the game:

    loss(h) − loss(full)   > 0: the frames before s − h carry information the network
                           uses (cutting them costs this much);
                           < 0: history beyond h hurts (state out of distribution).

Per (game, s, h): the pooled mean loss, and the per-record paired difference to the
full history with a 95% interval (records are the unit: 1.96·sd/√n).  Per head for
h = 0 vs full.  Run it before and after a change of training (e.g. full-game training)
on the same records (``--seed``) to see whether the network learns to use more history.

    uv run python scripts/context_probe.py --ckpt multi_latest.pt --bf16 \\
        --data clash_royale=DIR --data lux=DIR ... [--starts 16,64,128,256,512]
        [--histories 0,8,16,32,64,128,256,full] [--span 32 --n 64] --out probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_processing.games.dataset import GameShardDataset, collate_game_sequences  # noqa: E402
from training.eval_multi_game import load_multi  # noqa: E402
from training.train_il_game import prepare_batch  # noqa: E402


def _slice(s: Dict, a: int, b: int) -> Dict:
    return {"obs": {k: v[a:b] for k, v in s["obs"].items()},
            "tgt": {k: v[a:b] for k, v in s["tgt"].items()}, "meta": s.get("meta", {})}


def _per_record(x: torch.Tensor, B: int) -> torch.Tensor:
    """Sum of a [B, T, ...] tensor over everything but the record axis."""
    return x.reshape(B, -1).sum(-1)


@torch.no_grad()
def span_stats(net, game: str, records: List[Dict], start: int, span: int, history: int,
               device: torch.device, batch: int, bf16: bool) -> Dict:
    """Frames [start, start + span) of every record, the forward starting ``history``
    frames earlier: per record the IL loss (Σ_head scale · weighted mean CE) and the
    value MSE; per head the pooled weighted CE sums."""
    a = max(0, start - history)
    rec_il: List[float] = []
    rec_v: List[float] = []
    head_sum: Dict[str, float] = {}
    head_w: Dict[str, float] = {}
    for i in range(0, len(records), batch):
        chunk = [_slice(r, a, start + span) for r in records[i:i + batch]]
        B = len(chunk)
        mb = prepare_batch(game, collate_game_sequences(chunk), device, tf_prob=1.0,
                           cfg=net.cfg, il_legal_mask=True)
        mask = mb["frame_mask"].clone()
        mask[:, :start - a] = False                      # score the target frames only
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"],
                       frame_mask=mb["frame_mask"])
            terms = net.action_decoder.il_head_terms(pred, mb["tgt"], mask)
        il = torch.zeros(B, dtype=torch.float64)
        for h, t in terms.items():
            if t.get("ce") is None:
                continue
            w = t["weight"].float()
            ce = torch.nan_to_num(t["ce"].float()) * w
            s_b, w_b = _per_record(ce, B).double().cpu(), _per_record(w, B).double().cpu()
            head_sum[h] = head_sum.get(h, 0.0) + float(s_b.sum())
            head_w[h] = head_w.get(h, 0.0) + float(w_b.sum())
            il += float(t.get("scale", 1.0)) * s_b / w_b.clamp(min=1.0)
        v = pred["value"].float()
        vt = mb["tgt"]["value"].float()
        vm = mask & mb["tgt"]["value_mask"].bool() if "value_mask" in mb["tgt"] else mask
        err = ((v - vt).square() * vm).reshape(B, -1).sum(-1) / vm.reshape(B, -1).sum(-1).clamp(min=1)
        rec_il.extend(il.tolist())
        rec_v.extend(err.double().cpu().tolist())
    return {"il": rec_il, "value_mse": rec_v,
            "heads": {h: head_sum[h] / max(head_w[h], 1.0) for h in head_sum}}


def paired(a: List[float], b: List[float]) -> Dict[str, float]:
    """Mean of a − b over paired records and its 95% half-width (normal approximation)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1))
    return {"mean": m, "ci95": 1.96 * sd / math.sqrt(n), "n": n}


def long_records(ds: GameShardDataset, need: int, n: int, seed: int) -> List[Dict]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for idx in torch.randperm(len(ds), generator=g).tolist()[: 60 * n]:
        s = ds[idx]
        if next(iter(s["obs"].values())).shape[0] >= need:
            out.append(s)
            if len(out) == n:
                break
    return out


def parse_histories(text: str) -> List[Optional[int]]:
    return [None if h == "full" else int(h) for h in text.split(",")]


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", action="append", default=[], metavar="GAME=DIR")
    p.add_argument("--starts", default="16,64,128,256,512", help="target span starts s")
    p.add_argument("--histories", default="0,8,16,32,64,128,256,full",
                   help="frames of history before s ('full' = the record from frame 0)")
    p.add_argument("--span", type=int, default=32, help="target frames per record")
    p.add_argument("--n", type=int, default=64, help="records per game and start")
    p.add_argument("--min_records", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="record selection (same = paired runs)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--raw", action="store_true", help="training weights instead of the EMA")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device)
    model = load_multi(a.ckpt, device, raw=a.raw)
    starts = [int(s) for s in a.starts.split(",")]
    hists = parse_histories(a.histories)
    S = a.span
    res: Dict[str, Dict] = {"ckpt": a.ckpt, "span": S, "games": {}}
    for item in a.data:
        game, _, path = item.partition("=")
        if game not in model.games:
            print(f"[skip] {game}: not in the checkpoint", flush=True)
            continue
        net = model[game]
        try:
            ds = GameShardDataset(path, game=game)
        except Exception as e:                           # e.g. gio record directories
            print(f"[skip] {game}: {e}", flush=True)
            continue
        res["games"][game] = {}
        for s in starts:
            recs = long_records(ds, s + S, a.n, seed=a.seed * 1000 + s)
            if len(recs) < a.min_records:
                print(f"[skip] {game} s={s}: {len(recs)} records of >= {s + S} frames", flush=True)
                continue
            rows = {}
            for h in hists:
                hh = s if h is None else h
                if hh > s:
                    continue
                key = "full" if h is None or hh == s else str(hh)
                if key in rows:
                    continue
                rows[key] = span_stats(net, game, recs, s, S, hh, device, a.batch, a.bf16)
            full = rows["full"]
            summary = {}
            for key, r in rows.items():
                summary[key] = {
                    "il": sum(r["il"]) / len(r["il"]),
                    "value_mse": sum(r["value_mse"]) / len(r["value_mse"]),
                    "d_il_vs_full": paired(r["il"], full["il"]),
                    "d_value_vs_full": paired(r["value_mse"], full["value_mse"]),
                    "heads": r["heads"]}
            res["games"][game][str(s)] = {"records": len(recs), "rows": summary}
            line = "  ".join(
                f"h={k}: {v['il']:.4f} ({100 * v['d_il_vs_full']['mean'] / full_il:+.1f}%"
                f"±{100 * v['d_il_vs_full']['ci95'] / full_il:.1f})"
                for full_il in [summary["full"]["il"]] for k, v in summary.items() if k != "full")
            print(f"{game:14s} s={s:4d} n={len(recs):3d} full {summary['full']['il']:.4f} | {line}",
                  flush=True)
            if "0" in summary:
                hd = "  ".join(f"{h} {summary['0']['heads'][h]:.3f}→{summary['full']['heads'][h]:.3f}"
                               for h in summary["full"]["heads"])
                print(f"{'':14s}        heads (fresh→full): {hd}   value_mse "
                      f"{summary['0']['value_mse']:.4f}→{summary['full']['value_mse']:.4f}", flush=True)
            if a.out:
                with open(a.out, "w") as fh:
                    json.dump(res, fh, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()
