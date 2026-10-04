"""Backgammon cube decisions of checkpoints on recorded matches (problem 130: after RL the
policy loses to gnubg but not to its own starting point — do its doubling / take decisions drift?).

For each checkpoint the backgammon network runs over whole recorded matches (teacher forcing,
eval mode, as IL validation does); at every cube decision — the candidates are {no double,
double} (an offer) or {take, drop} — it reports how often the model's most likely choice is
the recorded one (the gnubg-labelled corpus: gnubg's choice), the mean probability it gives
the recorded choice, and how often it doubles / takes.

    uv run python scripts/probe_cube_decisions.py --ckpt IL.pt --ckpt RL.pt \\
        --data ~/gm-data-local/backgammon2-val [--records 60] [--threads 32]

Candidate features (gmrules backgammon, A_KIND): the first 7 are a one-hot of the decision kind
[pass, full move, split move, no double, double, take, drop]."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KIND_NO_DOUBLE, KIND_DOUBLE, KIND_TAKE, KIND_DROP = 3, 4, 5, 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", type=int, default=60)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from data_processing.games.dataset import GameShardDataset, collate_game_sequences
    from training.eval_multi_game import load_multi

    ds = GameShardDataset(a.data, game="backgammon")
    idx = list(range(min(a.records, len(ds))))
    samples = [ds[i] for i in idx]
    for path in a.ckpt:
        model = load_multi(path, torch.device("cpu"))
        net = model["backgammon"].eval()
        stats = {k: [0, 0, 0.0, 0] for k in ("offer", "take", "move")}   # n, agree, p_label, yes
        with torch.no_grad():
            for s in samples:
                b = collate_game_sequences([s])
                obs, fm = b["obs"], b["meta"]["frame_mask"]
                legal = {"action_idx": obs["action_mask"].bool()} if "action_mask" in obs else None
                out = net(obs, legal=legal, frame_mask=fm)
                logits = next(v for k, v in out.items() if k.endswith("_logits") and v.dim() == 3)
                kinds = obs["actions"][0, :, :, :7]                       # [T, K, 7]
                mask = obs["action_mask"][0].bool()                       # [T, K]
                label = b["tgt"]["action_idx"][0].long()                  # [T]
                lp = torch.log_softmax(logits[0].float().masked_fill(~mask, -1e9), -1)
                for t in range(int(fm[0].sum())):
                    k = kinds[t].argmax(-1)                               # the kind of each candidate
                    present = set(k[mask[t]].tolist())
                    if present == {KIND_NO_DOUBLE, KIND_DOUBLE}:
                        kind, yes = "offer", KIND_DOUBLE
                    elif present == {KIND_TAKE, KIND_DROP}:
                        kind, yes = "take", KIND_TAKE
                    elif int(mask[t].sum()) >= 2:        # a checker-play choice
                        kind, yes = "move", -1
                    else:
                        continue
                    st = stats[kind]
                    pick = int(lp[t].argmax())
                    st[0] += 1
                    st[1] += int(pick == int(label[t]))
                    st[2] += float(lp[t, int(label[t])].exp())
                    st[3] += int(int(k[pick]) == yes)
        print(f"== {path}")
        for kind, (n, agree, p, yes) in stats.items():
            if n:
                tail = (f"  model says {'double' if kind == 'offer' else 'take'} {yes / n:.3f}"
                        if kind != "move" else "")
                print(f"   {kind:5s} decisions {n:5d}  agree with gnubg {agree / n:.3f}  "
                      f"p(gnubg's choice) {p / n:.3f}{tail}", flush=True)
        del model


if __name__ == "__main__":
    main()
