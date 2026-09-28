"""Star statistics of the generals.io duel ladder index (``gmdata generals index``) — the
evidence behind the thresholds of ``gmdata generals ladder`` (docs: principle 2, a reliable
rating with percentile thresholds; what the number is: each player's stars when the game was
played, as the index lists them).

Reports: games and players by star level; each player's peak and median stars; how well the
star difference predicts the result (the higher-star player's win rate by difference — stars
that did not predict results would not be a skill measure); how many (game, player) records
each candidate threshold keeps, with and without an opponent floor.

    uv run python scripts/generals_ladder_stats.py --index ~/gm-local/generals-index/index.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys


def load(path):
    seen = set()
    for line in open(path):
        try:
            g = json.loads(line)
        except ValueError:
            continue
        r = g.get("ranking") or []
        if g.get("type") != "1v1" or len(r) != 2 or g["id"] in seen:
            continue
        seen.add(g["id"])
        yield g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", required=True)
    p.add_argument("--min-turns", type=int, default=50)
    p.add_argument("--out", help="write the tables as JSON here")
    a = p.parse_args(argv)
    games = [g for g in load(a.index) if (g.get("turns") or 0) >= a.min_turns]
    stars_of = collections.defaultdict(list)
    for g in games:
        for pl in g["ranking"]:
            stars_of[pl["name"]].append(pl.get("stars") or 0)
    peak = {n: max(s) for n, s in stars_of.items()}
    med = {n: statistics.median(s) for n, s in stars_of.items()}
    t0 = min(g["started"] for g in games)
    t1 = max(g["started"] for g in games)
    out = {"games": len(games), "players": len(stars_of), "days": round((t1 - t0) / 86400e3, 1),
           "min_turns": a.min_turns}
    print(f"{len(games)} 1v1 games >= {a.min_turns} turns, {len(stars_of)} players, {out['days']} days")

    # players by peak stars (with >= 20 games): the ladder's top
    ranked = sorted(((peak[n], n) for n in peak if len(stars_of[n]) >= 20), reverse=True)
    out["peak_rank"] = {str(k): ranked[k - 1][0] for k in (1, 10, 25, 50, 100, 200, 500, 1000, 2000) if k <= len(ranked)}
    print("peak stars by rank (players with >= 20 games):", out["peak_rank"])

    # the higher-star player's win rate by star difference
    by_diff = collections.defaultdict(lambda: [0, 0])
    by_level = collections.defaultdict(lambda: [0, 0])
    for g in games:
        r = g["ranking"]
        s = [pl.get("stars") or 0 for pl in r]
        win = [pl.get("kills") == 1 for pl in r]
        if s[0] == s[1] or win[0] == win[1]:
            continue
        hi = 0 if s[0] > s[1] else 1
        d = abs(s[0] - s[1])
        b = min(d // 5 * 5, 40)
        by_diff[b][0] += 1
        by_diff[b][1] += win[hi]
        lo = min(s)
        if lo >= 40:
            lv = min(lo // 10 * 10, 90)
            by_level[(lv, b)][0] += 1
            by_level[(lv, b)][1] += win[hi]
    out["higher_star_win_rate"] = {f"{b}-{b + 4}" if b < 40 else "40+": [n, round(w / n, 3)]
                                   for b, (n, w) in sorted(by_diff.items())}
    print("higher-star player's win rate by difference:", out["higher_star_win_rate"])
    out["higher_star_win_rate_by_lower_stars"] = {f"{lv}|{b}": [n, round(w / n, 3)]
                                                  for (lv, b), (n, w) in sorted(by_level.items()) if n >= 200}

    # (game, player) records per threshold: the player's stars at the game >= X, the
    # opponent's >= floor
    rows = {}
    for x in (50, 55, 60, 65, 70, 75, 80):
        for floor in (0, 40, 50, 60):
            n = players = 0
            who = set()
            for g in games:
                r = g["ranking"]
                s = [pl.get("stars") or 0 for pl in r]
                for i in (0, 1):
                    if s[i] >= x and s[1 - i] >= floor:
                        n += 1
                        who.add(r[i]["name"])
            rows[f"{x}|{floor}"] = [n, len(who)]
    out["records_by_threshold"] = rows
    print("records (player stars >= X, opponent >= F): [records, players]")
    for x in (50, 55, 60, 65, 70, 75, 80):
        print(f"  X={x}:", {f: rows[f"{x}|{f}"] for f in (0, 40, 50, 60)})
    # winner-only (the old recipe): winner >= 70
    old = sum(1 for g in games if any(pl.get("kills") == 1 and (pl.get("stars") or 0) >= 70 for pl in g["ranking"]))
    out["winner_ge_70_games"] = old
    print("games whose winner has >= 70 stars (the old recipe):", old)
    # star levels over time (monthly p90 / max of the higher player's stars)
    month = collections.defaultdict(list)
    import datetime
    for g in games:
        m = datetime.datetime.utcfromtimestamp(g["started"] / 1000).strftime("%Y-%m")
        month[m].append(max(pl.get("stars") or 0 for pl in g["ranking"]))
    out["monthly"] = {m: [len(v), sorted(v)[int(len(v) * 0.9)], max(v)] for m, v in sorted(month.items())}
    print("monthly [games, p90, max] of the higher player's stars:", out["monthly"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
