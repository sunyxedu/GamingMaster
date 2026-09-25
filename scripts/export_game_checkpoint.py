"""Export one game's network from a train_multi_game checkpoint as a
single-game checkpoint (the shared core included), for per-game
post-training of a jointly pretrained brain:

    uv run python scripts/export_game_checkpoint.py \\
        --ckpt checkpoints/multi/multi_final.pt --game generals --out generals_brain.pt

    # generals: RL self-play or IL fine-tuning from it
    uv run python -m training.train_rl --il_checkpoint generals_brain.pt ...
    uv run python -m training.train_il --finetune_from generals_brain.pt ...

The output holds ``game``, ``config``, ``model_state`` (EMA weights) and
``train_model_state`` (raw weights); see training.train_multi_game.game_checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_multi_game import game_checkpoint  # noqa: E402


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="train_multi_game checkpoint")
    p.add_argument("--game", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    out = game_checkpoint(ck, a.game)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    n = sum(v.numel() for v in out["model_state"].values())
    print(f"[export] {a.game} from {a.ckpt} ({len(out['model_state'])} tensors, "
          f"{n:,} values) -> {a.out}")
    return a.out


if __name__ == "__main__":
    main()
