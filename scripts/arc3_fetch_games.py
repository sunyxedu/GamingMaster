"""Download the official ARC-AGI-3 games (metadata + game source) from the ARC API into an
``ENVIRONMENTS_DIR`` layout (``<dir>/<game>/<version>/{metadata.json, <class>.py}``, what the
SDK's offline mode scans) — the same two requests the SDK makes when it downloads a game
(arc_agi/base.py ``_fetch_metadata`` / ``_download_game``), for every game the API lists.

The key comes from ``ARC_API_KEY`` (the environment, else the repo's ``.env``); it is only
sent to the ARC API (``ARC_BASE_URL``, default https://three.arcprize.org).

    uv run python scripts/arc3_fetch_games.py --out data/arc3/official
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api_key() -> str:
    key = os.environ.get("ARC_API_KEY", "")
    env = os.path.join(ROOT, ".env")
    if not key and os.path.exists(env):
        for line in open(env):
            name, _, value = line.strip().partition("=")
            if name == "ARC_API_KEY":
                key = value.strip()
    if not key:
        sys.exit("ARC_API_KEY is not set (environment or .env)")
    return key


def get(session: requests.Session, url: str, tries: int = 5):
    for i in range(tries):
        r = session.get(url, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 5 * 2 ** i))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default=os.environ.get("ARC_BASE_URL", "https://three.arcprize.org"))
    p.add_argument("--games", default="", help="comma-separated ids (default: every listed game)")
    a = p.parse_args(argv)
    s = requests.Session()
    s.headers.update({"X-Api-Key": api_key(), "Accept": "application/json"})
    listed = get(s, f"{a.base_url}/api/games").json()
    want = {g for g in a.games.split(",") if g}
    fetched, skipped, failed = [], [], {}
    for g in listed:
        gid = g.get("game_id", "")
        short = gid.split("-")[0]
        if not gid or (want and short not in want and gid not in want):
            continue
        try:
            meta = get(s, f"{a.base_url}/api/games/{short}").json()
            full = meta.get("game_id") or gid
            version = meta.get("version") or (full.split("-", 1)[1] if "-" in full else "")
            if not version:
                raise ValueError(f"no version in {full!r}")
            cls = meta.get("class_name") or (short[0].upper() + short[1:])
            d = os.path.join(a.out, short, version)
            src = os.path.join(d, f"{cls.lower()}.py")
            if os.path.exists(src) and os.path.exists(os.path.join(d, "metadata.json")):
                skipped.append(full)
                continue
            os.makedirs(d, exist_ok=True)
            code = get(s, f"{a.base_url}/api/games/{short}-{version}/source").text
            meta.setdefault("tags", [])
            meta.setdefault("baseline_actions", [])
            meta["local_dir"] = os.path.join("environment_files", short, version)
            meta["date_downloaded"] = datetime.now(timezone.utc).isoformat()
            with open(src + ".part", "w") as fh:
                fh.write(code)
            os.replace(src + ".part", src)
            with open(os.path.join(d, "metadata.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            fetched.append(full)
        except Exception as e:  # noqa: BLE001 — report per game, keep going
            failed[gid] = str(e)[:200]
        time.sleep(0.5)
    print(json.dumps({"listed": len(listed), "fetched": fetched, "already_there": skipped, "failed": failed,
                      "titles": {g.get("game_id"): g.get("title") for g in listed}}, indent=1))


if __name__ == "__main__":
    main()
