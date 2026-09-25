"""Evaluate a multi-game checkpoint per game, optionally sweeping the
inference depth r of a recurrent-depth core (does thinking longer help, and
does it extrapolate beyond the training mean r̄?).

    uv run python -m training.eval_multi_game --ckpt checkpoints/multi/multi_final.pt \\
        --val othello=data/othello/val --val chess=data/chess/val --depths 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from data_processing.games.dataset import GameShardDataset, collate_game_sequences
from model.multi_game import MultiGameNetwork
from training.deep_supervision import Q_HALT_WEIGHT
from training.train_il_game import batch_loss, build_net_tolerant, pick_device, prepare_batch


def load_multi(path: str, device: torch.device, raw: bool = False) -> MultiGameNetwork:
    """Rebuild a MultiGameNetwork from a train_multi_game checkpoint
    (EMA weights unless ``raw``)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nets = {}
    for g in ck["games"]:
        net, _ = build_net_tolerant(g, dict(ck["configs"][g]))
        nets[g] = net
    model = MultiGameNetwork(nets)
    model.load_state_dict(ck["train_model_state" if raw else "model_state"])
    return model.to(device).eval()


@torch.no_grad()
def eval_game(net, game: str, loader: DataLoader, device, depth: Optional[int],
              max_batches: int = 0) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        mb = prepare_batch(game, batch, device, tf_prob=1.0)
        _, stats, _ = batch_loss(net, mb, device, 0.0, Q_HALT_WEIGHT, False,
                                 act_steps=depth if depth is not None else net.default_depth)
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val", action="append", required=True, metavar="GAME=DIR")
    p.add_argument("--depths", type=int, nargs="*", default=[],
                   help="inference depths to sweep (default: the model's own)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    p.add_argument("--json", default=None, help="write results here")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device) if a.device else pick_device()
    model = load_multi(a.ckpt, device, raw=a.raw)
    depths = a.depths or [None]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for item in a.val:
        g, _, d = item.partition("=")
        if g not in model.games:
            print(f"[skip] {g}: not in checkpoint games {model.games}")
            continue
        loader = DataLoader(GameShardDataset(d, game=g, max_len=a.max_len),
                            batch_size=a.batch_size, collate_fn=collate_game_sequences)
        for depth in depths:
            m = eval_game(model[g], g, loader, device, depth, a.max_batches)
            key = str(depth if depth is not None else model[g].default_depth)
            results.setdefault(g, {})[key] = m
            acc = m.get("exact_acc", float("nan"))
            print(f"{g} r={key}: loss={m['loss']:.4f} exact_acc={acc:.4f}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
