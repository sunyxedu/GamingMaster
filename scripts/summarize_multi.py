"""Summarise train_multi_game runs: last validation metrics per run × game.

    python scripts/summarize_multi.py ~/gm-runs/multi/q1-* [--metric loss exact_acc]

Reads each run's log.txt ("[val STEP] game: k=v ...") and prints, per run,
the final value of each metric for every game plus the macro average over
the games the runs share (so runs over different game sets are not averaged
over different games).
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
    p.add_argument("--metric", nargs="+", default=["loss", "exact_acc"])
    a = p.parse_args()
    table = {}
    for r in a.runs:
        log = os.path.join(r, "log.txt") if os.path.isdir(r) else r
        if os.path.exists(log):
            table[os.path.basename(os.path.normpath(r))] = parse(log)
    if not table:
        raise SystemExit("no logs found")
    games = sorted(set.intersection(*(set(v) for v in table.values())))
    for metric in a.metric:
        print(f"\n## {metric} (last validation)")
        print("| run | step | " + " | ".join(games) + " | macro |")
        print("|---|---|" + "---|" * (len(games) + 1))
        for run, res in sorted(table.items()):
            vals = [res[g].get(metric, float("nan")) for g in games]
            macro = sum(vals) / len(vals) if vals else float("nan")
            step = max(int(res[g]["_step"]) for g in games) if games else 0
            print(f"| {run} | {step} | " + " | ".join(f"{v:.4f}" for v in vals)
                  + f" | {macro:.4f} |")


if __name__ == "__main__":
    main()
