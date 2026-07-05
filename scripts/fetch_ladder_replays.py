#!/usr/bin/env python3
"""Fetch high-star 1v1 replays from the global ladder index.

Unlike ``prepare_data --players`` (which walks specific accounts), this
pages the undocumented global listing endpoint::

    GET https://generals.io/api/replays?count=200&offset=N&l=duel

which returns every ladder game newest-first WITH both players' star
ratings — so we can filter for quality up front and skip bot accounts
(Human.exe, EklipZ, … are bots on the main server; ``isBot`` is checked
via /api/starsAndRanks and cached).

Output layout (resumable — rerunning skips already-downloaded ids)::

    <out>/gior/<id>.gior        raw replays (S3, ~3-15 KB each)
    <out>/manifest.jsonl        one line per kept game (id, stars, players…)
    <out>/botcache.json         username → isBot cache

Typical phase-2 corpus pull (≥70★ winners, ≥100 turns)::

    uv run python scripts/fetch_ladder_replays.py \
        --out /scratch/$USER/gaming-master/data-ladder70 \
        --min_stars 70 --min_turns 100 --max_games 100000

Then feed <out>/gior into the existing simulate+convert stages.

Notes:
  - generals.io API sits behind Cloudflare: python's default UA gets 403,
    so we always send a browser UA.  S3 downloads are plain AWS.
  - The index goes back to ~2024-08 (~1.55M duel games); older games are
    only reachable by enumerating the S3 bucket.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://generals.io/api"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
PAGE_SIZE = 200  # server-side maximum


def http_get(url: str, retries: int = 4, timeout: int = 30) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def fetch_index_page(ladder: str, offset: int) -> list[dict]:
    url = f"{API}/replays?count={PAGE_SIZE}&offset={offset}&l={ladder}"
    return json.loads(http_get(url))


class BotCache:
    """username → isBot, persisted so reruns don't rehit the API."""

    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, bool] = {}
        if path.exists():
            self.cache = json.loads(path.read_text())

    def is_bot(self, username: str) -> bool:
        if username not in self.cache:
            url = f"{API}/starsAndRanks?u={urllib.parse.quote(username)}"
            try:
                info = json.loads(http_get(url))
                self.cache[username] = bool(info.get("isBot", False))
            except RuntimeError:
                # Unresolvable account (renamed/deleted): treat as human,
                # the star filter already did the quality gating.
                self.cache[username] = False
            self.path.write_text(json.dumps(self.cache))
            time.sleep(0.2)  # politeness: this is a Cloudflare-fronted API
        return self.cache[username]


def keep_game(g: dict, args, botcache: BotCache) -> bool:
    if g.get("type") != "1v1" or len(g.get("ranking", [])) != 2:
        return False
    if g.get("turns", 0) < args.min_turns:
        return False
    ranking = g["ranking"]
    winner = next((p for p in ranking if p.get("kills") == 1), ranking[0])
    stars = [p.get("stars") or 0 for p in ranking]
    if args.require_both:
        if min(stars) < args.min_stars:
            return False
    elif (winner.get("stars") or 0) < args.min_stars:
        return False
    if not args.allow_bots:
        # Winner must be human — that's whose moves we imitate.  With
        # --exclude_bots_any the loser must be human too.
        check = ranking if args.exclude_bots_any else [winner]
        if any(botcache.is_bot(p["name"]) for p in check):
            return False
    return True


def download_gior(replay_id: str, server: str, dest: Path) -> bool:
    url = f"https://generalsio-replays-{server}.s3.amazonaws.com/{replay_id}.gior"
    try:
        data = http_get(url, retries=3)
    except RuntimeError:
        return False
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ladder", default="duel", help="ladder id (duel = 1v1)")
    p.add_argument("--server", default="na", help="replay S3 bucket suffix")
    p.add_argument("--min_stars", type=int, default=70,
                   help="winner must have ≥ this many stars")
    p.add_argument("--require_both", action="store_true",
                   help="require BOTH players ≥ min_stars")
    p.add_argument("--min_turns", type=int, default=100)
    p.add_argument("--max_games", type=int, default=100000,
                   help="stop after keeping this many games")
    p.add_argument("--max_pages", type=int, default=None,
                   help="index pages to scan (default: until exhausted)")
    p.add_argument("--start_offset", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.4,
                   help="pause between index pages (Cloudflare politeness)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel S3 downloads")
    p.add_argument("--allow_bots", action="store_true")
    p.add_argument("--exclude_bots_any", action="store_true",
                   help="also drop games where the LOSER is a bot")
    args = p.parse_args()

    gior_dir = args.out / "gior"
    gior_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    botcache = BotCache(args.out / "botcache.json")

    seen: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"[resume] {len(seen)} games already in manifest", flush=True)

    kept = scanned = pages = dl_fail = 0
    offset = args.start_offset
    manifest = manifest_path.open("a")
    pool = ThreadPoolExecutor(max_workers=args.workers)

    try:
        while kept + len(seen) < args.max_games:
            if args.max_pages is not None and pages >= args.max_pages:
                break
            try:
                page = fetch_index_page(args.ladder, offset)
            except RuntimeError as e:
                print(f"[warn] index page offset={offset}: {e}", flush=True)
                time.sleep(10)
                continue
            if not page:
                print("[done] index exhausted", flush=True)
                break
            pages += 1
            scanned += len(page)

            batch = [g for g in page
                     if g["id"] not in seen and keep_game(g, args, botcache)]
            remaining = args.max_games - (kept + len(seen))
            batch = batch[:max(0, remaining)]
            futures = {}
            for g in batch:
                dest = gior_dir / f"{g['id']}.gior"
                if dest.exists():
                    futures[None] = g  # already on disk, just record
                    continue
                futures[pool.submit(download_gior, g["id"], args.server, dest)] = g

            for fut in list(futures):
                g = futures[fut]
                ok = True if fut is None else fut.result()
                if not ok:
                    dl_fail += 1
                    continue
                manifest.write(json.dumps({
                    "id": g["id"],
                    "started": g["started"],
                    "turns": g["turns"],
                    "players": [pl["name"] for pl in g["ranking"]],
                    "stars": [pl.get("stars") or 0 for pl in g["ranking"]],
                    "winner": next((pl["name"] for pl in g["ranking"]
                                    if pl.get("kills") == 1), None),
                }) + "\n")
                seen.add(g["id"])
                kept += 1

            manifest.flush()
            offset += PAGE_SIZE
            if pages % 25 == 0:
                print(f"[page {pages}] offset={offset} scanned={scanned} "
                      f"kept={kept} (+{len(seen) - kept} resumed) "
                      f"dl_fail={dl_fail} bots_cached={len(botcache.cache)}",
                      flush=True)
            time.sleep(args.sleep)
    finally:
        pool.shutdown(wait=True)
        manifest.close()

    print(f"[done] pages={pages} scanned={scanned} kept_new={kept} "
          f"total={len(seen)} dl_fail={dl_fail} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
