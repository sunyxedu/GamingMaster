"""
City/tower behavior statistics from RL self-play replays (replay_iter*.json).

Answers, per iteration and over training: is the learner (p0) opening fewer
neutral towers? Is it bashing them with insufficient army? Is the net city
reward decline captures drying up, or losses to stronger pool opponents?

Replays are the full-information snapshots saved by train_rl.py
(save_replay_interval): state codes 3=neutral city, 6=p0 city, 9=p1 city;
army grids; per-step actions. Neutral city armies never regenerate, so any
army drop on a neutral city tile is a hit.

Usage (no torch needed, login node is fine):
    python3 scripts/analyze_city_stats.py '/scratch/$USER/gaming-master/checkpoints/rl_v2/replay_iter*.json'
    python3 scripts/analyze_city_stats.py '<glob>' --csv city_stats.csv
"""

import argparse
import glob
import json
import re
import statistics
import sys

_DIR_DR = [1, 0, -1, 0]
_DIR_DC = [0, 1, 0, -1]

NEUTRAL, P0_CITY, P1_CITY = 3, 6, 9
OWN_CODE = {0: P0_CITY, 1: P1_CITY}


def target_idx(action, W, H):
    """Flat 25-wide target tile of a move action, or None."""
    if not action:
        return None
    r, c = divmod(action["source_idx"], W)
    r += _DIR_DR[action["direction"]]
    c += _DIR_DC[action["direction"]]
    if not (0 <= r < H and 0 <= c < W):
        return None
    return r * W + c


def analyze_replay(path):
    with open(path) as f:
        data = json.load(f)
    steps = data["steps"]
    if len(steps) < 2:
        return None
    cities = data["map"]["cities"]
    H = len(steps[0]["state"])
    W = len(steps[0]["state"][0])

    s = {
        "iteration": data.get("iteration", -1),
        "ep_len": data.get("ep_len", len(steps) - 1),
        "winner": data.get("winner"),
        "n_cities": len(cities),
    }
    for pid in (0, 1):
        p = f"p{pid}"
        s[f"{p}_caps"] = 0        # neutral cities captured
        s[f"{p}_bashes"] = 0      # hit a neutral city without capturing it
        s[f"{p}_steals"] = 0      # took a city owned by the opponent
        s[f"{p}_losses"] = 0      # own city taken by the opponent
    cap_margins = {0: [], 1: []}  # army left on tile after neutral capture
    bash_remaining = {0: [], 1: []}  # garrison left after a failed hit

    for t in range(1, len(steps)):
        prev, cur = steps[t - 1], steps[t]
        tgt = {pid: target_idx(cur.get(f"p{pid}_action"), W, H) for pid in (0, 1)}
        for city in cities:
            r, c = divmod(city, W)
            o_prev, o_cur = prev["state"][r][c], cur["state"][r][c]
            a_prev, a_cur = prev["army"][r][c], cur["army"][r][c]
            if o_prev == NEUTRAL:
                if o_cur == NEUTRAL:
                    if a_cur < a_prev:  # neutral armies never grow: this is a hit
                        for pid in (0, 1):
                            if tgt[pid] == city:
                                s[f"p{pid}_bashes"] += 1
                                bash_remaining[pid].append(a_cur)
                else:
                    pid = 0 if o_cur == P0_CITY else 1
                    s[f"p{pid}_caps"] += 1
                    cap_margins[pid].append(a_cur)
            elif o_prev in (P0_CITY, P1_CITY) and o_cur in (P0_CITY, P1_CITY) and o_cur != o_prev:
                winner_pid = 0 if o_cur == P0_CITY else 1
                s[f"p{winner_pid}_steals"] += 1
                s[f"p{1 - winner_pid}_losses"] += 1

    end = steps[-1]["state"]
    s["p0_cities_end"] = sum(row.count(P0_CITY) for row in end)
    s["p1_cities_end"] = sum(row.count(P1_CITY) for row in end)
    for pid in (0, 1):
        s[f"p{pid}_cap_margin_med"] = (
            statistics.median(cap_margins[pid]) if cap_margins[pid] else "")
        s[f"p{pid}_bash_left_med"] = (
            statistics.median(bash_remaining[pid]) if bash_remaining[pid] else "")
    return s


COLS = ["iteration", "ep_len", "winner", "n_cities",
        "p0_caps", "p0_bashes", "p0_steals", "p0_losses",
        "p0_cap_margin_med", "p0_bash_left_med", "p0_cities_end",
        "p1_caps", "p1_bashes", "p1_steals", "p1_losses", "p1_cities_end"]


def main():
    ap = argparse.ArgumentParser(description="City behavior stats from RL replays")
    ap.add_argument("glob", help="Glob for replay_iter*.json (quote it)")
    ap.add_argument("--csv", default=None, help="Also write rows here")
    ap.add_argument("--window", type=int, default=50,
                    help="Iteration bucket size for the summary (default 50)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob),
                   key=lambda p: int(re.search(r"iter(\d+)", p).group(1)))
    if not paths:
        sys.exit(f"no files match {args.glob}")

    rows = [r for r in (analyze_replay(p) for p in paths) if r]
    print("  ".join(f"{c:>{max(len(c), 5)}}" for c in COLS))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):>{max(len(c), 5)}}" for c in COLS))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    # Windowed summary: the trend is the answer
    print(f"\n=== per-{args.window}-iteration means (p0 = learner) ===")
    print(f"{'iters':>12}  {'games':>5}  {'caps':>6}  {'bashes':>6}  "
          f"{'steals':>6}  {'losses':>6}  {'cities_end':>10}")
    bucket = {}
    for r in rows:
        bucket.setdefault(r["iteration"] // args.window, []).append(r)
    for b in sorted(bucket):
        rs = bucket[b]
        def mean(k):
            return sum(x[k] for x in rs) / len(rs)
        print(f"{b * args.window:>5}-{(b + 1) * args.window - 1:<6}  {len(rs):>5}  "
              f"{mean('p0_caps'):>6.2f}  {mean('p0_bashes'):>6.2f}  "
              f"{mean('p0_steals'):>6.2f}  {mean('p0_losses'):>6.2f}  "
              f"{mean('p0_cities_end'):>10.2f}")


if __name__ == "__main__":
    main()
