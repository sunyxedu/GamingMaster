"""Download Pokémon Showdown replays via the public replay search API.

    uv run python -m data_processing.games.pokemon.fetch_replays \\
        --format gen9ou --min_rating 1100 --max_replays 5000 \\
        --sleep 1.0 --output_dir replays/gen9ou

Endpoint: ``https://replay.pokemonshowdown.com/search.json?format=<fmt>``
returns up to 51 newest public replays (id/players/rating/uploadtime);
older pages are cursor-paginated with ``&before=<uploadtime>`` using the
last item's ``uploadtime``.  Each replay is then fetched from
``https://replay.pokemonshowdown.com/<id>.json``.

Every request sets a User-Agent header (the API rejects the default
urllib agent) and sleeps ``--sleep`` seconds between requests — be polite,
this is a free community service.  Existing files are skipped, so the
script is resumable.  ``rating`` on the replay is the LOSER's post-game
rating; ``--min_rating`` filters on the search-page rating field (null
ratings are skipped when the filter is active).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from typing import List, Optional

BASE = "https://replay.pokemonshowdown.com"
USER_AGENT = "GamingMaster-research/0.1 (imitation-learning data pipeline)"
PAGE_SIZE = 51   # server returns ≤51; a full page means more pages exist


def _get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_page(fmt: str, before: Optional[int]) -> List[dict]:
    params = {"format": fmt}
    if before is not None:
        params["before"] = str(before)
    url = f"{BASE}/search.json?{urllib.parse.urlencode(params)}"
    page = _get_json(url)
    return page if isinstance(page, list) else []


def fetch(
    fmt: str,
    output_dir: str,
    min_rating: int = 0,
    max_replays: int = 1000,
    sleep: float = 1.0,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    stats = {"pages": 0, "listed": 0, "downloaded": 0, "skipped_rating": 0,
             "skipped_existing": 0, "errors": 0}
    before: Optional[int] = None

    while stats["downloaded"] < max_replays:
        try:
            page = search_page(fmt, before)
        except Exception as e:  # noqa: BLE001 — network loop must survive
            print(f"[fetch] search page failed ({e}); retrying in 10s")
            time.sleep(10.0)
            continue
        stats["pages"] += 1
        if not page:
            break
        # a full page's last row is the cursor into the next page
        has_more = len(page) >= PAGE_SIZE
        items = page[: PAGE_SIZE - 1] if has_more else page
        before = page[-1].get("uploadtime")

        for item in items:
            if stats["downloaded"] >= max_replays:
                break
            stats["listed"] += 1
            rid = item.get("id")
            if not rid:
                continue
            if min_rating > 0 and (item.get("rating") or 0) < min_rating:
                stats["skipped_rating"] += 1
                continue
            out_path = os.path.join(output_dir, f"{rid}.json")
            if os.path.exists(out_path):
                stats["skipped_existing"] += 1
                continue
            time.sleep(sleep)
            try:
                replay = _get_json(f"{BASE}/{rid}.json")
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                print(f"[fetch] {rid} failed: {e}")
                continue
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(replay, fh)
            os.replace(tmp, out_path)
            stats["downloaded"] += 1
            if stats["downloaded"] % 50 == 0:
                print(f"[fetch] {stats['downloaded']}/{max_replays} "
                      f"(pages={stats['pages']})", flush=True)
        if not has_more or before is None:
            break
        time.sleep(sleep)

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--format", default="gen9ou", help="e.g. gen9ou, gen9randombattle")
    p.add_argument("--min_rating", type=int, default=0,
                   help="skip replays listed below this rating (null → skipped)")
    p.add_argument("--max_replays", type=int, default=1000)
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between HTTP requests")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    stats = fetch(args.format, args.output_dir, min_rating=args.min_rating,
                  max_replays=args.max_replays, sleep=args.sleep)
    print(f"[fetch] done: {stats}")


if __name__ == "__main__":
    main()
