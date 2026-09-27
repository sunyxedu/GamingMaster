"""Regenerate ``showdown_vocab.json``: the exact move / species vocabularies of the Pokémon
pipeline, built from Pokémon Showdown's own data (github.com/smogon/pokemon-showdown, MIT
licence) at a pinned commit.

    uv run python -m data_processing.games.pokemon.make_vocab                 # the pinned commit
    uv run python -m data_processing.games.pokemon.make_vocab --commit <sha>  # move the pin
    uv run python -m data_processing.games.pokemon.make_vocab --showdown DIR  # a local checkout
    uv run python -m data_processing.games.pokemon.make_vocab --check         # still up to date?

Read from Showdown: ``data/moves.ts`` (every move), ``data/pokedex.ts`` (every species and
forme; cosmetic formes become aliases of their species), and the team-preview forme hiding of
``data/rulesets.ts`` / ``sim/battle.ts`` (``Urshifu-*``: one placeholder entry per species whose
forme team preview hides); ``data/items.ts`` (every item), ``data/abilities.ts`` (every ability)
and ``data/typechart.ts`` (the types, Stellar included — tera types) for the revealed held items,
abilities and tera types of converter v4.  The ids (``data_processing/games/pokemon/__init__.py`` reads them in
Python, ``gmdata pokemon convert`` embeds the file at build time):

    0        none (empty name: no move / no Pokémon)
    1        UNK  (a name the vocabulary does not know — counted by the converter)
    2, 3, …  the entries, in table order

Keys are Showdown's ``toID`` of the name (lowercase, ``[a-z0-9]`` only); a species key gets a
``*`` suffix for a team-preview placeholder (``Urshifu-*`` → ``urshifu*``).  The table is
append-only: regenerating keeps every existing entry at its id (entries Showdown dropped stay,
so old ids never change meaning) and appends new keys in Showdown's file order, so a
vocabulary is a prefix of every later one and its size identifies it.  ``--fresh`` ignores
the existing file (renumbers everything: every shard and checkpoint made with the old ids
becomes incompatible — re-convert, retrain).  After a regeneration: rebuild gmdata, re-pin
the pokemon golden digests (tests/fixtures/golden/pokemon.json) and re-convert.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

#: smogon/pokemon-showdown master of 2026-09-23 ("Teams: Tweak user feedback")
SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"
SHOWDOWN_REPO = "https://github.com/smogon/pokemon-showdown"
RAW = "https://raw.githubusercontent.com/smogon/pokemon-showdown/{commit}/{path}"
FILES = ("data/moves.ts", "data/pokedex.ts", "data/rulesets.ts", "sim/battle.ts",
         "data/items.ts", "data/abilities.ts", "data/typechart.ts")
#: the name-keyed tables beside moves / species: (table, Showdown file, plausible minimum size)
EXTRA_TABLES = (("items", "data/items.ts", 500), ("abilities", "data/abilities.ts", 280),
                ("types", "data/typechart.ts", 19))
FORMAT = "gm-pokemon-vocab.v1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showdown_vocab.json")
ID_NONE, ID_UNK, ID_FIRST = 0, 1, 2

#: an entry: (key, display name, Showdown num)
Entry = Tuple[str, str, int]


def to_id(text: str) -> str:
    """Showdown's ``toID`` (sim/dex-data.ts): lowercase, then drop everything but a-z0-9."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parse_table(text: str, path: str) -> List[Dict[str, object]]:
    """The top-level entries of a Showdown data table (``export const X = { key: {...}, }``):
    [{"key": ..., field: raw value}] for the fields written at the entry's first level.
    Multi-line arrays (``cosmeticFormes: [`` … ``],``) are joined."""
    entries: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    arr: Optional[List[str]] = None
    arr_key = ""
    for ln in text.split("\n"):
        m = re.match(r'^\t"?([a-z0-9]+)"?: \{$', ln)
        if m:
            cur = {"key": m.group(1)}
            entries.append(cur)
            arr = None
            continue
        if ln == "\t},":
            cur = None
            continue
        if cur is None:
            continue
        if arr is not None:
            arr.append(re.sub(r"(,|^)\s*//.*$", r"\1", ln.strip()))
            if ln.startswith("\t\t]"):
                cur[arr_key] = re.sub(r",$", "", " ".join(arr))
                arr = None
            continue
        m = re.match(r"^\t\t([A-Za-z]+): (.*)$", ln)
        if m:
            k, v = m.groups()
            if v == "[":
                arr, arr_key = ["["], k
            else:
                cur[k] = re.sub(r",(\s*//.*)?$", "", v)
    if not entries:
        raise SystemExit(f"{path}: no entries parsed — has the file layout changed?")
    return entries


def _string(v: object, what: str) -> str:
    s = json.loads(str(v))
    if not isinstance(s, str):
        raise SystemExit(f"{what}: expected a string, got {v!r}")
    return s


def _strings(v: object, what: str) -> List[str]:
    arr = json.loads(re.sub(r",\s*\]$", "]", str(v)))
    if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
        raise SystemExit(f"{what}: expected a string array, got {v!r}")
    return arr


def parse_moves(text: str) -> List[Entry]:
    out = []
    for e in _parse_table(text, "data/moves.ts"):
        key = str(e["key"])
        if "name" not in e or "num" not in e:
            raise SystemExit(f"data/moves.ts: move {key!r} without name / num")
        out.append((key, _string(e["name"], key), int(str(e["num"]))))
    return out


def parse_named(text: str, path: str) -> List[Entry]:
    """Items / abilities: (key, name, num) of every entry (like :func:`parse_moves`)."""
    out = []
    for e in _parse_table(text, path):
        key = str(e["key"])
        if "name" not in e or "num" not in e:
            raise SystemExit(f"{path}: entry {key!r} without name / num")
        out.append((key, _string(e["name"], key), int(str(e["num"]))))
    return out


def parse_types(text: str) -> List[Entry]:
    """The types of the type chart (keys, capitalised names; num = their order)."""
    keys = re.findall(r"^\t([a-z]+): \{$", text, flags=re.M)
    if not keys:
        raise SystemExit("data/typechart.ts: no types parsed — has the file layout changed?")
    return [(k, k.capitalize(), i + 1) for i, k in enumerate(keys)]


def parse_species(text: str) -> Tuple[List[Entry], Dict[str, str], Dict[str, str]]:
    """(species entries, {cosmetic forme key: its species key}, {key: display name} of the
    cosmetic formes).  Cosmetic formes (``isCosmeticForme`` entries, ``cosmeticFormes``
    lists) play exactly like their species: they are aliases, not entries."""
    species: List[Entry] = []
    cosmetic: Dict[str, str] = {}
    cosmetic_names: Dict[str, str] = {}
    for e in _parse_table(text, "data/pokedex.ts"):
        key = str(e["key"])
        name = _string(e.get("name", '""'), key)
        if e.get("isCosmeticForme") == "true":
            base = _string(e.get("baseSpecies", '""'), key)
            cosmetic[key] = to_id(base)
            cosmetic_names[key] = name
            continue
        if "num" not in e or not name:
            raise SystemExit(f"data/pokedex.ts: species {key!r} without name / num")
        species.append((key, name, int(str(e["num"]))))
        for forme in _strings(e["cosmeticFormes"], key) if "cosmeticFormes" in e else []:
            cosmetic.setdefault(to_id(forme), key)
            cosmetic_names.setdefault(to_id(forme), forme)
    return species, cosmetic, cosmetic_names


#: ``.replace(/(Urshifu|…)…/g, '$1-*')`` — Showdown's team preview hides these species' formes
_HIDDEN_RE = re.compile(r"\.replace\(/\(([A-Za-z|]+)\)[^\n]*?/g,\s*'\$1-\*'\)")


def parse_hidden(texts: Iterable[str]) -> List[str]:
    """Species names whose forme team preview shows as ``NAME-*``, in first-seen order."""
    out: List[str] = []
    for text in texts:
        for m in _HIDDEN_RE.finditer(text):
            for name in m.group(1).split("|"):
                if name not in out:
                    out.append(name)
    if not out:
        raise SystemExit("no team-preview forme hiding found in data/rulesets.ts / sim/battle.ts")
    return out


def _variants(name: str) -> List[str]:
    """toIDs of the Unicode spellings of a name (Showdown writes Flabébé with combining
    accents — toID "flabebe"; a precomposed é gives toID "flabb")."""
    return [to_id(unicodedata.normalize(f, name)) for f in ("NFC", "NFD", "NFKC", "NFKD")]


def _merge(old: List[Entry], new: List[Entry], what: str) -> List[Entry]:
    """Append-only: the old entries keep their ids (display name / num refreshed when Showdown
    still has the key), new keys follow in Showdown's order."""
    cur = {k: (k, n, num) for k, n, num in new}
    out = [cur.get(k, (k, n, num)) for k, n, num in old]
    have = {k for k, _, _ in old}
    gone = [k for k in have if k not in cur]
    if gone:
        print(f"[vocab] {what}: {len(gone)} key(s) no longer in Showdown keep their ids: "
              f"{sorted(gone)[:10]}", file=sys.stderr)
    out += [e for e in new if e[0] not in have]
    return out


def _aliases(entries: List[Entry], extra: Dict[str, str], extra_names: Dict[str, str],
             what: str) -> List[Tuple[str, str]]:
    """[(alias key, entry key)]: ``extra`` (cosmetic formes) plus the Unicode-spelling
    variants of every entry / alias name — never shadowing an entry key."""
    keys = {k for k, _, _ in entries}
    out: Dict[str, str] = {}

    def add(alias: str, target: str):
        if not alias or alias in keys or alias == target:
            return
        if target not in keys:
            raise SystemExit(f"{what}: alias {alias!r} → unknown key {target!r}")
        if out.setdefault(alias, target) != target:
            raise SystemExit(f"{what}: alias {alias!r} is ambiguous ({out[alias]!r} / {target!r})")

    for alias, target in extra.items():
        add(alias, target)
    for key, name, _ in entries:
        if not name.endswith("-*"):
            for v in _variants(name):
                add(v, key)
    for alias, name in extra_names.items():
        for v in _variants(name):
            add(v, extra[alias])
    return sorted(out.items())


def build(texts: Dict[str, str], commit: str, old: Optional[Dict] = None,
          min_sizes: Tuple[int, ...] = (900, 1400)) -> Dict:
    """The vocabulary document from Showdown's files (``texts`` by repo path), appended to
    ``old`` (a previous document) when given; fewer than ``min_sizes`` (moves, species[,
    items, abilities, types] — the last three default to EXTRA_TABLES') entries means the
    parser missed most of the table."""
    moves = parse_moves(texts["data/moves.ts"])
    species, cosmetic, cosmetic_names = parse_species(texts["data/pokedex.ts"])
    by_key = {k: (n, num) for k, n, num in species}
    for name in parse_hidden(texts[p] for p in ("data/rulesets.ts", "sim/battle.ts")):
        if to_id(name) not in by_key:
            raise SystemExit(f"team-preview placeholder {name}-* names no species")
        species.append((to_id(name) + "*", name + "-*", by_key[to_id(name)][1]))
    for what, table in (("moves", moves), ("species", species)):
        keys = [k for k, _, _ in table]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{what}: duplicate keys")
        for k, n, _ in table:
            if not n.endswith("-*") and to_id(n) != k and k not in _variants(n):
                raise SystemExit(f"{what}: key {k!r} is not the toID of its name {n!r}")
    if len(moves) < min_sizes[0] or len(species) < min_sizes[1]:
        raise SystemExit(f"implausibly small tables ({len(moves)} moves, {len(species)} species)")
    extra: Dict[str, List[Entry]] = {}
    for i, (name, path, least) in enumerate(EXTRA_TABLES):
        least = min_sizes[2 + i] if len(min_sizes) > 2 + i else (least if min_sizes[0] else 0)
        table = parse_types(texts[path]) if name == "types" else parse_named(texts[path], path)
        keys = [k for k, _, _ in table]
        if len(set(keys)) != len(keys) or len(table) < least:
            raise SystemExit(f"{name}: duplicate keys or an implausibly small table ({len(table)})")
        for k, n, _ in table:
            if to_id(n) != k and k not in _variants(n):
                raise SystemExit(f"{name}: key {k!r} is not the toID of its name {n!r}")
        extra[name] = table
    if old is not None:
        moves = _merge([tuple(e) for e in old["moves"]], moves, "moves")
        species = _merge([tuple(e) for e in old["species"]], species, "species")
        for name in extra:
            extra[name] = _merge([tuple(e) for e in old.get(name, [])], extra[name], name)
    return {
        "format": FORMAT,
        "version": f"ps-{commit[:12]}",
        "source": {"repo": SHOWDOWN_REPO, "commit": commit, "license": "MIT", "files": list(FILES)},
        "ids": {"none": ID_NONE, "unk": ID_UNK, "first": ID_FIRST},
        "moves": [list(e) for e in moves],
        "move_aliases": [list(a) for a in _aliases(moves, {}, {}, "moves")],
        "species": [list(e) for e in species],
        "species_aliases": [list(a) for a in _aliases(species, cosmetic, cosmetic_names,
                                                      "species")],
        **{name: [list(e) for e in table] for name, table in extra.items()},
    }


def dumps(doc: Dict) -> str:
    """Deterministic JSON, one table entry per line (reviewable diffs)."""
    lines = ["{"]
    for k, v in doc.items():
        if not isinstance(v, list):
            lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    tables = [k for k, v in doc.items() if isinstance(v, list)]
    for i, k in enumerate(tables):
        rows = ",\n".join(f"    {json.dumps(x, ensure_ascii=False)}" for x in doc[k])
        body = f"[\n{rows}\n  ]" if rows else "[]"
        lines.append(f"  {json.dumps(k)}: {body}" + ("," if i + 1 < len(tables) else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def read_sources(commit: str, showdown: Optional[str]) -> Dict[str, str]:
    texts = {}
    for path in FILES:
        if showdown:
            with open(os.path.join(showdown, path), encoding="utf-8") as f:
                texts[path] = f.read()
        else:
            url = RAW.format(commit=commit, path=path)
            with urllib.request.urlopen(url, timeout=60) as r:
                texts[path] = r.read().decode("utf-8")
    return texts


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--commit", default=SHOWDOWN_COMMIT,
                   help="smogon/pokemon-showdown commit to read (default: the pinned one)")
    p.add_argument("--showdown", default=None,
                   help="read the files from this local checkout (at --commit) instead of GitHub")
    p.add_argument("--out", default=OUT)
    p.add_argument("--fresh", action="store_true",
                   help="renumber from scratch instead of appending to the existing file")
    p.add_argument("--check", action="store_true",
                   help="only compare the generated vocabulary with --out (exit 1 if different)")
    a = p.parse_args(argv)
    old = None
    if not a.fresh and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            old = json.load(f)
    doc = build(read_sources(a.commit, a.showdown), a.commit, old)
    text = dumps(doc)
    if a.check:
        with open(a.out, encoding="utf-8") as f:
            same = f.read() == text
        state = "up to date" if same else "DIFFERS from the generated vocabulary"
        print(f"[vocab] {a.out}: {state}")
        return 0 if same else 1
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[vocab] {a.out}: version {doc['version']}, {len(doc['moves'])} moves, "
          f"{len(doc['species'])} species, {len(doc['items'])} items, "
          f"{len(doc['abilities'])} abilities, {len(doc['types'])} types (+ ids 0 none / 1 UNK), "
          f"{len(doc['move_aliases'])} move / {len(doc['species_aliases'])} species aliases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
