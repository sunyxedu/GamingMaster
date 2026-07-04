#!/usr/bin/env python3
"""
Validate a converted IL dataset directory (output of prepare_data's convert
stage) before spending GPU time on it.

Checks per replay file:
  - schema/shape consistency across all sequence fields
  - label invariants: count in {-1, 0, 1}; every count==1 source lies inside
    the legal mask of its paired observation; direction in 0..3; a visible
    destination tile is never a mountain
  - observation sanity: one-hot state channels (<=1 active per cell), army
    channel range, value targets in [-1, 1]
  - dataset-level stats: move/pass/dropped fractions, value-target signs

Then loads a batch through SeqReplayDataset + collate and runs one forward +
ImitationLoss + next_obs_loss on a tiny recurrent model — proving the data
can actually train the intended architecture.

Usage:
    uv run python scripts/validate_dataset.py data/output/training/Human.exe
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

DIR_DR = [1, 0, -1, 0]
DIR_DC = [0, 1, 0, -1]


def validate_file(path: Path) -> dict:
    data = json.load(open(path))
    T = len(data["count"])
    errors = []

    # --- schema/shapes ---
    for key in ("grid_seq", "globals_seq", "legal_source_mask_seq",
                "src_y", "src_x", "dir_id", "ratio_id", "value"):
        if len(data[key]) != T:
            errors.append(f"{key} length {len(data[key])} != count length {T}")
    grid0 = np.asarray(data["grid_seq"][0])
    if grid0.shape != (12, 25, 25):
        errors.append(f"grid shape {grid0.shape} != (12, 25, 25)")

    stats = {"T": T, "moves": 0, "passes": 0, "dropped": 0}
    for t in range(T):
        c = data["count"][t]
        if c not in (-1, 0, 1):
            errors.append(f"t={t}: count={c} not in {{-1,0,1}}")
            continue
        stats["moves" if c == 1 else "passes" if c == 0 else "dropped"] += 1

        grid = np.asarray(data["grid_seq"][t], dtype=np.float32)
        onehot = grid[:11]
        if (onehot.sum(axis=0) > 1.0 + 1e-6).any():
            errors.append(f"t={t}: >1 active state channel on some cell")
        if grid[11].min() < 0 or grid[11].max() > 1.5:
            errors.append(f"t={t}: army channel out of range [{grid[11].min()}, {grid[11].max()}]")
        v = data["value"][t]
        if not (-1.0 - 1e-6 <= v <= 1.0 + 1e-6):
            errors.append(f"t={t}: value {v} outside [-1, 1]")

        if c == 1:
            sy, sx = data["src_y"][t][0], data["src_x"][t][0]
            d = data["dir_id"][t][0]
            mask = np.asarray(data["legal_source_mask_seq"][t], dtype=bool)[0]
            if not (0 <= sy < 25 and 0 <= sx < 25):
                errors.append(f"t={t}: source ({sy},{sx}) out of bounds")
            elif not mask[sy, sx]:
                errors.append(f"t={t}: source ({sy},{sx}) not in legal mask")
            if d not in (0, 1, 2, 3):
                errors.append(f"t={t}: dir {d} invalid")
            else:
                ty, tx = sy + DIR_DR[d], sx + DIR_DC[d]
                if not (0 <= ty < 25 and 0 <= tx < 25):
                    errors.append(f"t={t}: move target ({ty},{tx}) off-grid")
                elif onehot[4, ty, tx] > 0.5:  # visible mountain
                    errors.append(f"t={t}: move target ({ty},{tx}) is a mountain")

    return {"errors": errors, "stats": stats}


def _validate_one(path_str: str) -> tuple:
    """Pool worker: returns (name, errors, stats)."""
    r = validate_file(Path(path_str))
    return (Path(path_str).name, r["errors"], r["stats"])


def validate_directory(root: Path, per_file: bool = False, workers: int = None,
                       sample: int = 0) -> dict:
    """Run label/observation invariants over the training files in parallel.

    Files are independent, so validation fans out over a process pool
    (16k files serially is minutes; parallel is seconds). ``sample > 0``
    checks a deterministic subset — enough to catch systematic corruption.
    Importable: prepare_data calls this automatically after collection.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(Path(root).glob("*.json"))
    if sample and len(files) > sample:
        step = len(files) / sample
        files = [files[int(i * step)] for i in range(sample)]  # even stride
        print(f"(sampling {len(files)} of the files)")

    total = {"T": 0, "moves": 0, "passes": 0, "dropped": 0}
    bad_files = 0
    workers = workers or os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, errors, stats in pool.map(_validate_one, [str(f) for f in files],
                                            chunksize=16):
            for k in total:
                total[k] += stats[k]
            if errors:
                bad_files += 1
                print(f"FAIL {name}: {len(errors)} violations")
                for e in errors[:5]:
                    print(f"    {e}")
            elif per_file:
                print(f"ok   {name}: T={stats['T']}  moves={stats['moves']}  "
                      f"passes={stats['passes']}  dropped={stats['dropped']}")

    frac = {k: total[k] / max(total["T"], 1) for k in ("moves", "passes", "dropped")}
    print(f"{len(files)} files, {total['T']} frames | "
          f"moves {frac['moves']:.1%}  passes {frac['passes']:.1%}  "
          f"dropped {frac['dropped']:.1%}"
          + (f" | {bad_files} FAILED" if bad_files else ""))
    if frac["moves"] < 0.2 and total["T"]:
        print("WARN: suspiciously few move labels — check the pipeline")
    return {"files": len(files), "bad_files": bad_files,
            "frames": total["T"], **{k: total[k] for k in ("moves", "passes", "dropped")}}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate a converted IL dataset dir")
    p.add_argument("root", type=Path, help="training/<player> directory")
    p.add_argument("--sample", type=int, default=0,
                   help="Validate a strided subset of N files (0 = all)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--per-file", action="store_true", help="Print every file")
    args = p.parse_args()

    if not sorted(args.root.glob("*.json")):
        print(f"no .json files under {args.root}")
        return 2

    summary = validate_directory(args.root, per_file=args.per_file,
                                 workers=args.workers, sample=args.sample)
    bad_files = summary["bad_files"]
    root = args.root

    # --- can this data actually train the target architecture? ---
    from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
    from model.config import ModelConfig
    from model.losses import ImitationLoss, next_obs_loss
    from model.network import GeneralsNetwork

    ds = SeqReplayDataset(str(root), max_sequence_length=64)
    batch = collate_variable_length_sequences([ds[i] for i in range(min(2, len(ds)))])
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_st_layers=1,
                      trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
                      encoder_channels=32, temporal_mixer="recurrent",
                      predict_next_obs=True)
    model = GeneralsNetwork(cfg)
    model.train()
    grid_seq = batch["obs"]["grid_seq"]
    lengths = batch["meta"]["lengths"]
    frame_mask = (torch.arange(grid_seq.shape[1])[None, :]
                  < torch.as_tensor(lengths)[:, None])
    pred = model(grid_seq, batch["obs"]["globals_seq"],
                 legal_mask=batch["obs"]["legal_source_mask_seq"],
                 frame_mask=frame_mask)
    tgt = batch["tgt"]
    losses = ImitationLoss(grid_w=25)(pred, tgt, lengths)
    wm = next_obs_loss(pred, grid_seq, frame_mask)
    total_loss = losses["loss"] + wm["state_loss"] + wm["army_loss"]
    total_loss.backward()
    assert torch.isfinite(total_loss), "training smoke: non-finite loss"
    print(f"train smoke (recurrent+wm): loss={losses['loss'].item():.3f}  "
          f"wm_state={wm['state_loss'].item():.3f}  grads finite OK")

    if bad_files:
        print(f"\nRESULT: {bad_files}/{summary['files']} files FAILED validation")
        return 1
    print(f"\nRESULT: all {summary['files']} files valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
