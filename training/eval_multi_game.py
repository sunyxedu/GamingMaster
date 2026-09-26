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

from data_processing.games.dataset import collate_game_sequences
from model.config import upgrade_config_dict
from model.multi_game import MultiGameNetwork
from training.deep_supervision import Q_HALT_WEIGHT
from training.train_il_game import (
    UNCROPPABLE,
    check_crop_policy,
    batch_loss,
    build_net_tolerant,
    greedy_decoding,
    open_dataset,
    pick_device,
    prepare_batch,
)


def load_multi(path: str, device: torch.device, raw: bool = False) -> MultiGameNetwork:
    """Rebuild a MultiGameNetwork from a train_multi_game checkpoint
    (EMA weights unless ``raw``)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nets = {}
    for g in ck["games"]:
        # fields newer than the checkpoint take their legacy value (e.g.
        # qk_norm=False), so the rebuilt net matches the saved weights
        net, _ = build_net_tolerant(g, upgrade_config_dict(dict(ck["configs"][g])))
        nets[g] = net
    model = MultiGameNetwork(nets)
    model.load_state_dict(ck["train_model_state" if raw else "model_state"])
    return model.to(device).eval()


@torch.no_grad()
def eval_game(net, game: str, loader: DataLoader, device, depth: Optional[int],
              max_batches: int = 0, bf16: bool = False,
              il_legal_mask: bool = False) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        mb = prepare_batch(game, batch, device, tf_prob=1.0, cfg=net.cfg,
                           il_legal_mask=il_legal_mask)
        with greedy_decoding(net):
            _, stats, _ = batch_loss(net, mb, device, 0.0, Q_HALT_WEIGHT, bf16,
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
    p.add_argument("--game_batch", action="append", default=[], metavar="GAME=N",
                   help="per-game batch size (full-length clash_royale sequences are "
                        "~460 frames x 580 tokens: use 1-2)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_len", type=int, default=None,
                   help="crop every game (random window; generals: its first frames)")
    p.add_argument("--game_max_len", action="append", default=[], metavar="GAME=N",
                   help="per-game --max_len, e.g. bound generals (prefix crop, as "
                        "trained) while the other games are evaluated in full")
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    p.add_argument("--il_legal_mask", action="store_true",
                   help="feed the shards' tgt legal_<key> masks to the decoders as "
                        "legal[<key>] (as trained with train_multi_game --il_legal_mask)")
    p.add_argument("--json", default=None, help="write results here")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device) if a.device else pick_device()
    model = load_multi(a.ckpt, device, raw=a.raw)
    depths = a.depths or [None]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    gb = {k: int(v) for k, v in (x.split("=", 1) for x in a.game_batch)}
    gml = {k: int(v) for k, v in (x.split("=", 1) for x in a.game_max_len)}
    for item in a.val:
        g, _, d = item.partition("=")
        if g not in model.games:
            print(f"[skip] {g}: not in checkpoint games {model.games}")
            continue
        # the trainer's crop policy: generals keeps the first max_len frames,
        # clash_royale is never cropped, the others take random windows
        ml = gml.get(g) if g in UNCROPPABLE else gml.get(g, a.max_len)
        check_crop_policy(g, model[g].cfg, ml)
        ds = open_dataset(g, d, ml)
        loader = DataLoader(ds, batch_size=gb.get(g, a.batch_size),
                            collate_fn=collate_game_sequences)
        for depth in depths:
            m = eval_game(model[g], g, loader, device, depth, a.max_batches, a.bf16,
                          il_legal_mask=a.il_legal_mask)
            key = str(depth if depth is not None else model[g].default_depth)
            results.setdefault(g, {})[key] = m
            acc = m.get("exact_acc", float("nan"))
            # value_loss = the head's own training loss (MSE or HL-Gauss CE);
            # value_mse compares scalar and HL-Gauss heads directly
            val = "".join(f" {k}={m[k]:.4f}" for k in ("value_loss", "value_mse") if k in m)
            print(f"{g} r={key}: loss={m['loss']:.4f} exact_acc={acc:.4f}{val}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
