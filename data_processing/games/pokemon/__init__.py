"""Exact Pokémon Showdown vocabularies: move / species names → collision-free ids.

``showdown_vocab.json`` (next to this file) is generated from Pokémon Showdown's own data
(github.com/smogon/pokemon-showdown, MIT licence) by ``make_vocab.py``; the file records the
Showdown commit and its ``version``.  The same table is embedded into ``gmdata pokemon
convert`` (converter v3, data_processing/rs/gmdata/src/pokemon.rs) at build time and read here
by the pokemon adapter (model/games/pokemon.py: embedding sizes, the deploy request mask).

Ids::

    0        none — the empty name (no move / no Pokémon)
    1        UNK  — a name the vocabulary does not know (a move / species newer than the pinned
             Showdown commit, a custom name); never merged with a known id
    2, 3, …  the table entries (every Showdown move; every species and forme that plays
             differently — Rotom-Wash, Urshifu-Rapid-Strike, Ogerpon-Wellspring(-Tera), … —
             plus one team-preview placeholder per species whose forme team preview hides,
             ``Urshifu-*``)

Lookup — identical in the converter::

    move key     toID(name): lowercase, [a-z0-9] only (Showdown's toID)
    species key  toID(name), with Showdown's species.get rule Nidoran♀ / ♂ → nidoranf /
                 nidoranm, and a ``*`` suffix for a team-preview placeholder
                 ("Urshifu-*" → "urshifu*")
    id           0 for an empty name; else the entry of the key, or of the entry it aliases
                 (cosmetic formes — Gastrodon-East plays like Gastrodon — and other Unicode
                 spellings of a name); else 1 (UNK)

The table is append-only across regenerations (make_vocab.py), so a vocabulary is a prefix of
every later one: its size (``n_moves`` / ``n_species``, the shard meta ``move_vocab`` /
``species_vocab``) identifies it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showdown_vocab.json")
ID_NONE = 0
ID_UNK = 1
FORMAT = "gm-pokemon-vocab.v1"
#: the team-preview placeholder suffix of a species name ("Urshifu-*")
HIDDEN_FORME = "-*"


def to_id(text: str) -> str:
    """Showdown's ``toID`` (sim/dex-data.ts): lowercase, then keep only a-z0-9."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def species_key(name: str) -> str:
    """The vocabulary key of a species name as the battle log spells it: the species
    part of a details string ("Urshifu-*" of ``|poke|p1|Urshifu-*, M|``)."""
    t = name.strip()
    key = to_id(t)
    if key == "nidoran" and t.endswith("♀"):
        key = "nidoranf"
    elif key == "nidoran" and t.endswith("♂"):
        key = "nidoranm"
    if t.endswith(HIDDEN_FORME):
        key += "*"
    return key


@dataclass(frozen=True)
class Vocab:
    """One vocabulary file: ``*_ids`` map keys (entries and aliases) to ids, ``*_names[id]``
    is the display name of an id ("" for none, "<UNK>" for UNK)."""

    version: str
    commit: str
    move_ids: Dict[str, int]
    species_ids: Dict[str, int]
    move_names: Tuple[str, ...]
    species_names: Tuple[str, ...]
    #: the revealed held items / abilities / tera types (converter v4): key → id, id → name
    item_ids: Dict[str, int] = None
    ability_ids: Dict[str, int] = None
    type_ids: Dict[str, int] = None
    item_names: Tuple[str, ...] = ()
    ability_names: Tuple[str, ...] = ()
    type_names: Tuple[str, ...] = ()

    @property
    def n_items(self) -> int:
        return len(self.item_names)

    @property
    def n_abilities(self) -> int:
        return len(self.ability_names)

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def _named_id(self, table: Dict[str, int], name: str) -> int:
        if not name.strip():
            return ID_NONE
        return table.get(to_id(name), ID_UNK)

    def item_id(self, name: str) -> int:
        return self._named_id(self.item_ids, name)

    def ability_id(self, name: str) -> int:
        return self._named_id(self.ability_ids, name)

    def type_id(self, name: str) -> int:
        return self._named_id(self.type_ids, name)

    @property
    def n_moves(self) -> int:
        """Move id count, reserved ids included (the embedding / output table size)."""
        return len(self.move_names)

    @property
    def n_species(self) -> int:
        """Species id count, reserved ids included."""
        return len(self.species_names)

    def move_id(self, name: str) -> int:
        if not name.strip():
            return ID_NONE
        return self.move_ids.get(to_id(name), ID_UNK)

    def species_id(self, name: str) -> int:
        if not name.strip():
            return ID_NONE
        return self.species_ids.get(species_key(name), ID_UNK)


def _table(doc: Dict, entries: str, aliases: str = "") -> Tuple[Dict[str, int], Tuple[str, ...]]:
    first = doc["ids"]["first"]
    ids: Dict[str, int] = {}
    names = ["", "<UNK>"]
    if (doc["ids"]["none"], doc["ids"]["unk"], first) != (ID_NONE, ID_UNK, len(names)):
        raise ValueError(f"{entries}: unexpected reserved ids {doc['ids']}")
    for i, (key, name, _num) in enumerate(doc[entries]):
        if key in ids:
            raise ValueError(f"{entries}: duplicate key {key!r}")
        ids[key] = first + i
        names.append(name)
    for alias, target in (doc[aliases] if aliases else []):
        if alias in ids or target not in ids:
            raise ValueError(f"{aliases}: {alias!r} → {target!r} shadows an entry or names none")
        ids[alias] = ids[target]
    return ids, tuple(names)


@lru_cache(maxsize=None)
def load_vocab(path: str = VOCAB_PATH) -> Vocab:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {FORMAT!r}")
    move_ids, move_names = _table(doc, "moves", "move_aliases")
    species_ids, species_names = _table(doc, "species", "species_aliases")
    extra = {name: _table(doc, name) for name in ("items", "abilities", "types") if name in doc}
    empty = ({}, ("", "<UNK>"))
    (item_ids, item_names), (ability_ids, ability_names), (type_ids, type_names) = (
        extra.get(n, empty) for n in ("items", "abilities", "types"))
    return Vocab(version=doc["version"], commit=doc["source"]["commit"], move_ids=move_ids,
                 species_ids=species_ids, move_names=move_names, species_names=species_names,
                 item_ids=item_ids, ability_ids=ability_ids, type_ids=type_ids,
                 item_names=item_names, ability_names=ability_names, type_names=type_names)


#: the vendored vocabulary (what `gmdata pokemon convert` of this checkout writes)
VOCAB = load_vocab()
VOCAB_VERSION = VOCAB.version
MOVE_VOCAB_SIZE = VOCAB.n_moves
SPECIES_VOCAB_SIZE = VOCAB.n_species
ITEM_VOCAB_SIZE = VOCAB.n_items
ABILITY_VOCAB_SIZE = VOCAB.n_abilities
TYPE_VOCAB_SIZE = VOCAB.n_types

#: the formats of converter v5's ``obs.format_id`` (append-only; the same table as
#: ``FORMATS`` in data_processing/rs/gmdata/src/pokemon.rs, pinned by
#: tests/test_pokemon_vocab.py): Gen 9 6v6 singles with the standard mechanics. 0 = a
#: format outside the table
FORMATS = ("", "gen9ou", "gen9uu", "gen9ru", "gen9nu", "gen9pu", "gen9zu", "gen9lc", "gen9ubers",
           "gen9ubersuu", "gen9monotype", "gen9anythinggoes", "gen9randombattle",
           "gen9randombattleblitz")
FORMAT_VOCAB_SIZE = len(FORMATS)


def format_id(format: str) -> int:
    """A format's id (Showdown's toID of its name: "[Gen 9] OU" → gen9ou); 0 = not in
    the table."""
    fid = "".join(c for c in format.lower() if c.isalnum())
    return FORMATS.index(fid) if fid and fid in FORMATS else 0
