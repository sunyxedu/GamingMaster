"""The exact Pokémon Showdown vocabularies (data_processing/games/pokemon): the vendored table
round-trips, every move / species of the committed replay fixtures has its own exact id, the
generator's rules (Showdown parsing, cosmetic-forme aliases, team-preview placeholders,
append-only regeneration) hold on miniature Showdown sources, and the pre-v3 crc32 buckets
really merged the moves / species the review found (the reason for the vocabulary)."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import zlib
from typing import Dict, Set

import pytest

from data_processing.games.dataset import load_gm_samples
from data_processing.games.pokemon import (
    ID_NONE,
    ID_UNK,
    MOVE_VOCAB_SIZE,
    SPECIES_VOCAB_SIZE,
    VOCAB,
    VOCAB_PATH,
    VOCAB_VERSION,
    load_vocab,
    make_vocab,
    species_key,
    to_id,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = [os.path.join(ROOT, "tests", "fixtures", d) for d in ("pokemon", "showdown")]
#: Showdown writes Flabebe with combining accents (toID "flabebe"); a precomposed spelling has
#: toID "flabb" (an alias of the same entry)
FLABEBE_NFD = "Flabe\u0301be\u0301"
FLABEBE_NFC = "Flab\u00e9b\u00e9"


def _doc() -> Dict:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_vocabulary_round_trips():
    doc = _doc()
    assert doc["format"] == make_vocab.FORMAT and doc["source"]["license"] == "MIT"
    assert doc["version"] == VOCAB_VERSION == f"ps-{doc['source']['commit'][:12]}"
    assert doc["source"]["commit"] == make_vocab.SHOWDOWN_COMMIT, "made from the pinned commit"
    assert (MOVE_VOCAB_SIZE, SPECIES_VOCAB_SIZE) == (len(doc["moves"]) + 2, len(doc["species"]) + 2)
    assert MOVE_VOCAB_SIZE > 900 and SPECIES_VOCAB_SIZE > 1400
    for table, lookup, names in (("moves", VOCAB.move_id, VOCAB.move_names),
                                 ("species", VOCAB.species_id, VOCAB.species_names)):
        keys = [e[0] for e in doc[table]]
        assert len(set(keys)) == len(keys), table
        for i, (key, name, _num) in enumerate(doc[table]):
            # dense ids after the reserved 0 / 1; the display name maps back to its own id
            assert lookup(name) == i + 2 and names[i + 2] == name, (table, name)
            assert (species_key(name) if table == "species" else to_id(name)) == key, (table, name)
    entries = {e[0] for e in doc["species"]}
    for alias, target in doc["species_aliases"]:
        assert VOCAB.species_ids[alias] == VOCAB.species_ids[target] > ID_UNK
        assert alias not in entries
    assert (VOCAB.move_names[ID_NONE], VOCAB.species_names[ID_UNK]) == ("", "<UNK>")
    # reserved ids: the empty name, names the vocabulary does not know
    assert (VOCAB.move_id(""), VOCAB.species_id("  ")) == (ID_NONE, ID_NONE)
    assert (VOCAB.move_id("Zz Sentinel Move"), VOCAB.species_id("Zzmon")) == (ID_UNK, ID_UNK)
    assert VOCAB.move_id("---") == ID_UNK, "a non-empty name without a toID is no known move"
    assert load_vocab() is VOCAB


def test_formes_that_play_differently_have_their_own_ids():
    formes = ["Rotom", "Rotom-Wash", "Rotom-Heat", "Urshifu", "Urshifu-Rapid-Strike", "Urshifu-*",
              "Ogerpon", "Ogerpon-Wellspring", "Ogerpon-Hearthflame", "Ogerpon-Cornerstone",
              "Ogerpon-Teal-Tera", "Ogerpon-Wellspring-Tera", "Palafin", "Palafin-Hero",
              "Zamazenta", "Zamazenta-Crowned", "Zamazenta-*", "Terapagos", "Terapagos-Stellar"]
    ids = [VOCAB.species_id(n) for n in formes]
    assert len(set(ids)) == len(formes) and min(ids) > ID_UNK
    # cosmetic formes play like their species; other spellings of a name are the same name
    assert VOCAB.species_id("Gastrodon-East") == VOCAB.species_id("Gastrodon")
    assert VOCAB.species_id("Alcremie-Ruby-Swirl") == VOCAB.species_id("Alcremie")
    assert VOCAB.species_id(FLABEBE_NFC) == VOCAB.species_id(FLABEBE_NFD) > ID_UNK
    assert VOCAB.species_id(FLABEBE_NFC + "-Blue") == VOCAB.species_id(FLABEBE_NFD)
    assert VOCAB.species_id("Nidoran♀") == VOCAB.species_id("Nidoran-F") \
        != VOCAB.species_id("Nidoran♂")
    assert VOCAB.species_id("Farfetch’d") == VOCAB.species_id("farfetchd") > ID_UNK
    assert VOCAB.move_id("U-turn") == VOCAB.move_id("uturn") == VOCAB.move_id(" U-TURN ") > ID_UNK
    # Showdown's `num` is not an identity: G-Max moves share 1000, Hidden Power types 237
    nums = [e[2] for e in _doc()["moves"]]
    assert nums.count(1000) > 1 and nums.count(237) > 1
    hp = {VOCAB.move_id(f"Hidden Power {t}") for t in ("Fire", "Ice", "Grass")}
    assert len(hp | {VOCAB.move_id("Hidden Power")}) == 4


def _legacy_bucket(name: str, n: int) -> int:
    """converter v1 / v2 (gmdata pokemon ``bucket``)."""
    key = "".join(c for c in name.lower() if c.isalnum())
    return zlib.crc32(key.encode("utf-8")) % (n - 1) + 1


def test_the_hashed_buckets_merged_what_the_exact_ids_separate():
    """The review of converter v2 (fresh gen9ou replays): these shared a crc32 bucket — so a
    label and an embedding — under the 1024 move / 512 species buckets."""
    for names, n, lookup in ((("Swords Dance", "Fire Blast"), 1024, VOCAB.move_id),
                             (("Earthquake", "Tidy Up", "Aura Sphere"), 1024, VOCAB.move_id),
                             (("Dragonite", "Hoopa-Unbound"), 512, VOCAB.species_id),
                             (("Primarina", "Rillaboom"), 512, VOCAB.species_id)):
        assert len({_legacy_bucket(x, n) for x in names}) == 1, names
        assert len({lookup(x) for x in names}) == len(names), names


def _fixture_names() -> Dict[str, Set[str]]:
    """Every move / species name the committed replay fixtures log."""
    moves: Set[str] = set()
    species: Set[str] = set()
    for d in FIXTURES:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f, encoding="utf-8") as fh:
                log = json.load(fh)["log"]
            for ln in log.split("\n"):
                p = ln.split("|")
                if len(p) < 4:
                    continue
                if p[1] == "move":
                    moves.add(p[3])
                elif p[1] == "cant" and len(p) > 4 and p[4] and not p[4].startswith("["):
                    moves.add(p[4])
                elif p[1] in ("poke", "switch", "drag", "replace", "detailschange"):
                    species.add(p[3].split(",")[0].strip())
    return {"moves": moves, "species": species}


def test_every_fixture_move_and_species_has_a_distinct_exact_id():
    names = _fixture_names()
    assert len(names["moves"]) > 100 and len(names["species"]) > 60
    for kind, lookup, key in (("moves", VOCAB.move_id, to_id),
                              ("species", VOCAB.species_id, species_key)):
        ids = {n: lookup(n) for n in names[kind]}
        unknown = sorted(n for n, i in ids.items() if i <= ID_UNK)
        assert not unknown, f"{kind} without an exact id: {unknown}"
        # injective up to the name's key (no cosmetic forme shares an entry in the fixtures)
        by_id: Dict[int, Set[str]] = {}
        for n, i in ids.items():
            by_id.setdefault(i, set()).add(key(n))
        shared = {i: k for i, k in by_id.items() if len(k) > 1}
        assert not shared, f"{kind} sharing an id: {shared}"
    assert {"Greninja-*", "Greninja", "Zamazenta-*"} <= names["species"]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """(stats, samples, replays) of `gmdata pokemon convert` over both fixture directories."""
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.skip("cargo not installed")
    ws = os.path.join(ROOT, "data_processing")
    b = subprocess.run([cargo, "build", "--release", "-q", "-p", "gmdata"], cwd=ws,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    root = tmp_path_factory.mktemp("pk_vocab")
    (root / "in").mkdir()
    replays = {}
    for d in FIXTURES:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            shutil.copy(f, root / "in")
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            replays[r["id"]] = r
    res = subprocess.run([os.path.join(ws, "target", "release", "gmdata"), "pokemon", "convert",
                          "--input", str(root / "in"), "--out", str(root / "out")],
                         check=True, capture_output=True, text=True)
    return json.loads(res.stdout), load_gm_samples(str(root / "out")), replays


def test_the_converter_writes_the_vocabulary_ids(converted):
    """gmdata (Rust, the table embedded at build time) and this module (Python) give every
    logged name the same id: each side's lead frame shows the ``|poke|`` species of both
    teams in preview order, every move label is the id of a move the side logged, no fixture
    name is UNK, and the shard meta / convert stats record this vocabulary."""
    stats, samples, replays = converted
    assert stats["vocab"] == {"version": VOCAB_VERSION, "move_vocab": MOVE_VOCAB_SIZE,
                              "species_vocab": SPECIES_VOCAB_SIZE}
    assert stats["unk_moves"] == {} and stats["unk_species"] == {}
    assert stats["samples"] == len(samples) > 40
    labels = 0
    for s in samples:
        log = replays[s["meta"]["replay_id"]]["log"].split("\n")
        me = s["meta"]["side"]
        other = "p2" if me == "p1" else "p1"
        poke = {p: [ln.split("|")[3].split(",")[0].strip() for ln in log
                    if ln.startswith(f"|poke|{p}|")] for p in (me, other)}
        want = [VOCAB.species_id(n) for n in poke[me]] + [0] * (6 - len(poke[me]))
        want += [VOCAB.species_id(n) for n in poke[other]] + [0] * (6 - len(poke[other]))
        assert s["obs"]["species_ids"][0].tolist() == want, s["meta"]
        ids, mask = s["obs"]["species_ids"], s["obs"]["slot_mask"]
        assert bool((ids[mask] > ID_UNK).all()) and bool((ids[~mask] == ID_NONE).all())
        used = set()
        for ln in log:
            p = ln.split("|")
            if len(p) > 3 and p[2].startswith(me) and p[1] in ("move", "cant"):
                used.add(VOCAB.move_id(p[3] if p[1] == "move" else (p[4] if len(p) > 4 else "")))
        got = s["tgt"]["move_id"][s["tgt"]["action_type"] == 0]
        assert set(got.tolist()) <= used, s["meta"]
        seen = s["obs"]["move_ids"]
        assert set(seen[seen > 0].tolist()) <= used | {VOCAB.move_id(n) for n in _foe_moves(log)}
        labels += int(got.numel())
    assert labels > 1000, labels


def _foe_moves(log):
    """Moves a Transform can copy into a side's revealed moves: every logged move."""
    return {ln.split("|")[3] for ln in log if ln.startswith("|move|") and len(ln.split("|")) > 3}


# ------------------------------------------------------------------------------------------
# the generator (make_vocab.py) on miniature Showdown sources
# ------------------------------------------------------------------------------------------

MOVES_TS = """export const Moves: import('../sim/dex-moves').MoveDataTable = {
\t"10000000voltthunderbolt": {
\t\tnum: 719,
\t\tname: "10,000,000 Volt Thunderbolt",
\t\tcondition: {
\t\t\tname: "Not A Move",
\t\t},
\t},
\tabsorb: {
\t\tnum: 71,
\t\tname: "Absorb",
\t\tisNonstandard: "Gmax", // a comment
\t},
\tgmaxbefuddle: {
\t\tnum: 1000,
\t\tname: "G-Max Befuddle",
\t},
\tgmaxcannonade: {
\t\tnum: 1000,
\t\tname: "G-Max Cannonade",
\t},
};
"""

POKEDEX_TS = """export const Pokedex: import('../sim/dex-species').SpeciesDataTable = {
\tflabebe: {
\t\tnum: 669,
\t\tname: "Flabe\\u0301be\\u0301",
\t\tcosmeticFormes: ["Flabe\\u0301be\\u0301-Blue"],
\t},
\tgastrodon: {
\t\tnum: 423,
\t\tname: "Gastrodon",
\t\tcosmeticFormes: [
\t\t\t"Gastrodon-East", // a comment
\t\t],
\t},
\tgastrodoneast: {
\t\tisCosmeticForme: true,
\t\tname: "Gastrodon-East",
\t\tbaseSpecies: "Gastrodon",
\t\tforme: "East",
\t},
\turshifu: {
\t\tnum: 892,
\t\tname: "Urshifu",
\t},
\turshifurapidstrike: {
\t\tnum: 892,
\t\tname: "Urshifu-Rapid-Strike",
\t\tbaseSpecies: "Urshifu",
\t},
};
"""

RULESETS_TS = """\t\t\t\t\t.replace(/(Urshifu|Gastrodon)(-[a-zA-Z?-]+)?/g, '$1-*');"""


def _mini(moves=MOVES_TS, pokedex=POKEDEX_TS):
    return {"data/moves.ts": moves, "data/pokedex.ts": pokedex, "data/rulesets.ts": RULESETS_TS,
            "sim/battle.ts": ""}


def _build(texts, commit, old=None):
    # the real tables have > 900 moves / > 1400 species; the miniature ones do not
    return make_vocab.build(texts, commit, old=old, min_sizes=(0, 0))


def test_generator_parses_showdown_tables(tmp_path):
    doc = _build(_mini(), "0123456789abcdef")
    assert doc["version"] == "ps-0123456789ab" and doc["ids"] == {"none": 0, "unk": 1, "first": 2}
    assert [m[0] for m in doc["moves"]] == ["10000000voltthunderbolt", "absorb", "gmaxbefuddle",
                                            "gmaxcannonade"], "shared nums are no identity"
    # cosmetic formes are aliases, placeholders are entries (after the species, in regex order)
    assert [s[:2] for s in doc["species"]] == [
        ["flabebe", FLABEBE_NFD], ["gastrodon", "Gastrodon"], ["urshifu", "Urshifu"],
        ["urshifurapidstrike", "Urshifu-Rapid-Strike"], ["urshifu*", "Urshifu-*"],
        ["gastrodon*", "Gastrodon-*"]]
    assert dict(map(tuple, doc["species_aliases"])) == {
        "flabb": "flabebe", "flabbblue": "flabebe", "flabebeblue": "flabebe",
        "gastrodoneast": "gastrodon"}
    # the written file is what the loader reads
    path = tmp_path / "vocab.json"
    path.write_text(make_vocab.dumps(doc), encoding="utf-8")
    v = load_vocab(str(path))
    assert (v.n_moves, v.n_species) == (6, 8)
    assert v.species_id(FLABEBE_NFC + "-Blue") == v.species_id(FLABEBE_NFD) == 2
    assert v.species_id(FLABEBE_NFC) == v.species_id("Gastrodon-East") - 1 == 2
    assert v.species_id("Urshifu-*") == 6 and v.move_id("G-Max Cannonade") == 5


def test_generator_is_append_only():
    """A regeneration keeps every existing id (entries Showdown dropped included) and appends
    the new keys; --fresh would renumber."""
    old = _build(_mini(), "0" * 40)
    acid = '\tacid: {\n\t\tnum: 51,\n\t\tname: "Acid",\n\t},\n'
    befuddle = '\tgmaxbefuddle: {\n\t\tnum: 1000,\n\t\tname: "G-Max Befuddle",\n\t},\n'
    assert befuddle in MOVES_TS
    newer = MOVES_TS.replace("\tabsorb: {", acid + "\tabsorb: {").replace(befuddle, "")
    doc = _build(_mini(moves=newer), "1" * 40, old=old)
    assert [m[0] for m in doc["moves"]] == [m[0] for m in old["moves"]] + ["acid"]
    assert [m[0] for m in _build(_mini(moves=newer), "1" * 40)["moves"]] == [
        "10000000voltthunderbolt", "acid", "absorb", "gmaxcannonade"]


def test_generator_refuses_a_changed_layout():
    with pytest.raises(SystemExit, match="no entries parsed"):
        _build(_mini(moves="export const Moves = {absorb: {num: 71}};"), "0" * 40)
    with pytest.raises(SystemExit, match="not the toID"):
        _build(_mini(moves=MOVES_TS.replace('"Absorb"', '"Absorbx"')), "0" * 40)
    with pytest.raises(SystemExit, match="forme hiding"):
        _build({**_mini(), "data/rulesets.ts": ""}, "0" * 40)
    with pytest.raises(SystemExit, match="implausibly small"):
        make_vocab.build(_mini(), "0" * 40)


def test_vendored_file_is_the_generator_output():
    """The vendored file is byte-identical to the generator's rendering of its own content (so
    a regeneration from the same commit changes nothing); set GM_SHOWDOWN_DIR to a
    pokemon-showdown checkout at the pinned commit to regenerate it from the sources."""
    doc = _doc()
    with open(VOCAB_PATH, encoding="utf-8") as f:
        assert f.read() == make_vocab.dumps(doc)
    src = os.environ.get("GM_SHOWDOWN_DIR")
    if src:
        texts = make_vocab.read_sources(make_vocab.SHOWDOWN_COMMIT, src)
        assert make_vocab.dumps(make_vocab.build(texts, make_vocab.SHOWDOWN_COMMIT, old=doc)) == \
            make_vocab.dumps(doc)
