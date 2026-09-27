"""Team pools for env/vec_pokemon.py's team formats: metamon's team sets (Hugging Face
``jakegrigsby/metamon-teams``, MIT — Showdown teambuilder files: ``competitive`` sample teams,
``hl_05_26`` 43k high-ladder teams predicted from May 2026 replays and rating-appropriate usage
stats, ``gl_05_26`` the general ladder, ...) validated by Showdown's own TeamValidator at the
simulator's checkout (teams legal when the set was made may not be legal today), written as a
JSONL of {"packed", "source"} (the env's ``teams=``).

    uv run python scripts/pokemon_teams.py --set hl_05_26 --format gen9ou \\
        --out data/pokemon/teams/gen9ou_hl_05_26.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPO = "jakegrigsby/metamon-teams"


def fetch(set_name: str, fmt: str, cache: str) -> bytes:
    path = os.path.join(cache, set_name, f"{fmt}.tar.gz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{set_name}/{fmt}.tar.gz"
        with urllib.request.urlopen(url, timeout=600) as r, open(path + ".part", "wb") as fh:
            fh.write(r.read())
        os.replace(path + ".part", path)
    with open(path, "rb") as fh:
        return fh.read()


def teams_of(blob: bytes):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and not os.path.basename(m.name).startswith("."):
                text = tf.extractfile(m).read().decode("utf-8", "replace")
                if text.strip():
                    yield m.name, text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", default="hl_05_26")
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=os.path.join(ROOT, "data", "pokemon", "metamon-teams"))
    p.add_argument("--max-teams", type=int, default=0, help="0 = all")
    p.add_argument("--showdown-dir", default=None)
    a = p.parse_args(argv)
    from env.vec_pokemon import _Worker

    w = _Worker(a.showdown_dir)
    seen, kept, bad, dup = set(), 0, 0, 0
    reasons = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    batch = []
    with open(a.out + ".part", "w") as out:
        def flush():
            nonlocal kept, bad, dup
            res = w.call({"op": "batch", "cmds": [{"op": "validate", "format": a.format, "team": t} for _, t in batch]})
            for (name, _), r in zip(batch, res["results"]):
                if r.get("error") or not r.get("packed") or r.get("problems"):
                    bad += 1
                    why = (r.get("problems") or [r.get("error") or "no team"])[0]
                    key = why.split(":")[0][:60]
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                if r["packed"] in seen:
                    dup += 1
                    continue
                seen.add(r["packed"])
                out.write(json.dumps({"packed": r["packed"], "source": f"{REPO}/{a.set}/{name}"}) + "\n")
                kept += 1
            batch.clear()
        for name, text in teams_of(fetch(a.set, a.format, a.cache)):
            batch.append((name, text))
            if len(batch) >= 256:
                flush()
            if a.max_teams and kept + bad + dup + len(batch) >= a.max_teams:
                break
        if batch:
            flush()
    os.replace(a.out + ".part", a.out)
    w.close()
    top = dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8])
    print(json.dumps({"set": a.set, "format": a.format, "kept": kept, "invalid": bad, "duplicates": dup,
                      "invalid_reasons": top}))


if __name__ == "__main__":
    main()
