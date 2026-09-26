"""Build tests/fixtures/hunl/acpc_sample.phhs: a small stratified selection of real
ACPC heads-up no-limit hands (PHH, copied verbatim with a provenance comment) from a
sample fetched by ``gmdata hunl fetch``:

    gmdata hunl fetch --out /tmp/hunl_full --max-files-per-year 3
    uv run --no-project --python 3.12 python scripts/reference/make_hunl_fixture.py \\
        --input /tmp/hunl_full --out tests/fixtures/hunl/acpc_sample.phhs

Per competition year (in file order, deterministic): preflop folds, postflop folds,
showdowns, all-ins that were called, and the longest betting sequences.  Data:
"A Dataset of Poker Hand Histories" (Zenodo record 10796885), CC-BY-4.0.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import re
import tomllib

PER_YEAR = {"preflop": 6, "postflop": 6, "showdown": 8, "allin": 5, "longest": 3}

HEADER = """\
# Heads-up no-limit Texas hold'em hands of the Annual Computer Poker Competition
# 2009-2017 logs (PHH format), from "A Dataset of Poker Hand Histories" (Juho Kim et
# al., Zenodo record 10796885, https://doi.org/10.5281/zenodo.10796885), licensed
# CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). A stratified selection
# (preflop folds, postflop folds, showdowns, called all-ins, the longest betting
# sequences) of hands from three match logs per year, copied verbatim; only the
# section numbers are new and a "# source" comment names each hand's archive member.
# Made by scripts/reference/make_hunl_fixture.py.
"""


def sections(text: str):
    """(header, body) of every [n] section of a .phhs text."""
    parts = re.split(r"(?m)^\[(\d+)\]\s*$", text)
    for i in range(1, len(parts), 2):
        yield parts[i], parts[i + 1].strip("\n")


def kind(h: dict) -> str:
    acts = h["actions"]
    boards = sum(a.startswith("d db") for a in acts)
    stack = h["starting_stacks"][0]
    res = h.get("_results", [0, 0])
    if abs(res[0]) == stack:
        return "allin"
    if any(" sm" in a for a in acts):
        return "showdown"
    return "postflop" if boards else "preflop"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.input, "**", "*.phhs*"), recursive=True))
    by_year: dict = {}
    for f in files:
        m = re.search(r"/(20\d\d)/", f)
        if not m:
            continue
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            text = fh.read()
        member = f[len(a.input):].lstrip("/").removesuffix(".gz")
        for _, body in sections(text):
            h = tomllib.loads(body)
            by_year.setdefault(m.group(1), []).append((member, body, h))
    out, n = [HEADER], 0
    for year in sorted(by_year):
        hands = by_year[year]
        picked = []
        for k, want in PER_YEAR.items():
            if k == "longest":
                pool = sorted(hands, key=lambda x: -len(x[2]["actions"]))
            else:
                pool = [x for x in hands if kind(x[2]) == k]
            for x in pool:
                if want == 0:
                    break
                if x not in picked:
                    picked.append(x)
                    want -= 1
        for member, body, _ in picked:
            n += 1
            out.append(f"# source: data/annual-computer-poker-competition/competitions/{member}\n[{n}]\n{body}\n")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(out))
    print(f"{n} hands -> {a.out}")


if __name__ == "__main__":
    main()
