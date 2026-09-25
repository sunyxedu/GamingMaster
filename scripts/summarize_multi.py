"""Summarise train_multi_game runs: last validation metrics per run × game.

    python scripts/summarize_multi.py ~/gm-runs/multi/q1-* [--metric loss exact_acc]

Reads each run's log.txt ("[val STEP] game: k=v ...") and prints, per run,
the final value of each metric for every game plus the macro average over
the games the runs share (so runs over different game sets are not averaged
over different games).  ``il_loss`` (default) removes the value term from the
logged loss.  Note: clash_royale's exact_acc is dominated by "wait" frames —
compare its card/place losses instead.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict

VAL = re.compile(r"^\[val (\d+)\] (\w+): (.*)$")


def parse(log: str) -> Dict[str, Dict[str, float]]:
    """{game: {metric: value}} of the LAST validation of each game."""
    last: Dict[str, Dict[str, float]] = {}
    step: Dict[str, int] = {}
    with open(log) as f:
        for line in f:
            m = VAL.match(line.strip())
            if not m:
                continue
            s, g, rest = int(m.group(1)), m.group(2), m.group(3)
            if s >= step.get(g, -1):
                step[g] = s
                last[g] = {k: float(v) for k, v in (kv.split("=") for kv in rest.split())}
                last[g]["_step"] = s
    return last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--metric", nargs="+", default=["il_loss", "exact_acc"])
    p.add_argument("--value_weight", type=float, default=0.2,
                   help="value-loss weight folded into the logged 'loss'; il_loss = "
                        "loss - value_weight * value_loss (pure imitation loss, "
                        "comparable across games whose value targets differ)")
    a = p.parse_args()
    table = {}
    for r in a.runs:
        log = os.path.join(r, "log.txt") if os.path.isdir(r) else r
        if os.path.exists(log):
            res = parse(log)
            for m in res.values():
                if "loss" in m:
                    m["il_loss"] = m["loss"] - a.value_weight * m.get("value_loss", 0.0)
            table[os.path.basename(os.path.normpath(r))] = res
    if not table:
        raise SystemExit("no logs found")
    games = sorted(set.union(*(set(v) for v in table.values())))
    shared = sorted(set.intersection(*(set(v) for v in table.values())))
    for metric in a.metric:
        print(f"\n## {metric} (last validation)")
        print("| run | step | " + " | ".join(games) + " | macro |")
        print("|---|---|" + "---|" * (len(games) + 1))
        for run, res in sorted(table.items()):
            vals = [res.get(g, {}).get(metric, float("nan")) for g in games]
            sv = [res[g].get(metric, float("nan")) for g in shared]
            macro = sum(sv) / len(sv) if sv else float("nan")   # over games all runs share
            step = max((int(res[g]["_step"]) for g in res), default=0)
            print(f"| {run} | {step} | " + " | ".join(f"{v:.4f}" for v in vals)
                  + f" | {macro:.4f} |")


if __name__ == "__main__":
    main()
