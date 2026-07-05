"""Minimal SGF parser for Go game records (pure python, no deps).

Scope (all this pipeline needs — NOT a general SGF library):
  - main line only: at every branch point the FIRST variation is followed,
    sibling variations are discarded;
  - multi-game collections ``(...)(...)`` in one file are supported
    (one game per file is the typical case);
  - properties: root props are kept verbatim in ``SGFGame.props``
    (SZ / HA / KM / RE / PB / PW / BR / WR are given typed accessors),
    AB / AW setup stones and B / W moves are parsed into points;
  - pass moves: ``[]`` always, and ``[tt]`` on boards of size ≤ 19;
  - markup, comments, time stamps etc. are carried in ``props`` but never
    interpreted; ``\\]`` escapes inside property values are honoured so
    parentheses/brackets in comments cannot break tree balancing.

Coordinates: SGF writes ``[xy]`` where x = column, y = row, ``a`` = 0.
We return ``(row, col)`` tuples so they index numpy/torch grids directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

Point = Tuple[int, int]                 # (row, col), 0-based
Move = Tuple[str, Optional[Point]]      # ("B" | "W", point) — None = pass


class SGFError(ValueError):
    """Malformed SGF content (unterminated value, bad coordinate, ...)."""


@dataclass
class SGFGame:
    """One game's main line: raw root properties + typed setup/moves."""

    props: Dict[str, List[str]] = field(default_factory=dict)
    moves: List[Move] = field(default_factory=list)
    setup_black: List[Point] = field(default_factory=list)   # AB stones
    setup_white: List[Point] = field(default_factory=list)   # AW stones

    # -- typed accessors -------------------------------------------------

    def _first(self, key: str) -> Optional[str]:
        vals = self.props.get(key)
        return vals[0] if vals else None

    @property
    def board_size(self) -> int:
        raw = self._first("SZ")
        if raw is None:
            return 19
        try:
            return int(raw.split(":")[0])   # "19" or rectangular "19:19"
        except ValueError:
            raise SGFError(f"bad SZ value {raw!r}") from None

    @property
    def handicap(self) -> int:
        raw = self._first("HA")
        try:
            return int(raw) if raw else 0
        except ValueError:
            return 0

    @property
    def komi(self) -> Optional[float]:
        raw = self._first("KM")
        try:
            return float(raw) if raw not in (None, "") else None
        except ValueError:
            return None

    @property
    def result(self) -> Optional[str]:
        return self._first("RE")

    @property
    def player_black(self) -> Optional[str]:
        return self._first("PB")

    @property
    def player_white(self) -> Optional[str]:
        return self._first("PW")

    @property
    def rank_black(self) -> Optional[str]:
        return self._first("BR")

    @property
    def rank_white(self) -> Optional[str]:
        return self._first("WR")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ("(" | ")" | ";" | "IDENT" | "VALUE", payload) tokens.

    ``[...]`` values are atomic (with ``\\x`` escapes), so brackets and
    parentheses inside comments never reach the tree-structure logic.
    """
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "();":
            yield ch, ch
            i += 1
        elif ch == "[":
            j = i + 1
            buf: List[str] = []
            while j < n and text[j] != "]":
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(text[j])
                j += 1
            if j >= n:
                raise SGFError("unterminated property value")
            yield "VALUE", "".join(buf)
            i = j + 1
        elif ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            # FF[3] compat: lowercase letters in idents are decoration
            # ("White" → "W").
            ident = "".join(c for c in text[i:j] if c.isupper())
            if ident:
                yield "IDENT", ident
            i = j
        else:
            i += 1  # whitespace / stray junk between tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_point(raw: str, size: int) -> Optional[Point]:
    """SGF move value → (row, col); None = pass (``[]``, or ``tt`` ≤ 19×19)."""
    v = raw.strip()
    if v == "":
        return None
    if v == "tt" and size <= 19:
        return None
    if len(v) != 2 or not ("a" <= v[0] <= "z" and "a" <= v[1] <= "z"):
        raise SGFError(f"bad point {raw!r}")
    col, row = ord(v[0]) - 97, ord(v[1]) - 97
    if not (0 <= row < size and 0 <= col < size):
        raise SGFError(f"point {raw!r} off a {size}x{size} board")
    return row, col


def _finish(events: List[Tuple[str, List[str]]]) -> SGFGame:
    """Turn the ordered (ident, values) stream of one game into an SGFGame."""
    size = 19
    for ident, values in events:                # SZ before interpreting "tt"
        if ident == "SZ" and values:
            try:
                size = int(values[0].split(":")[0])
            except ValueError:
                raise SGFError(f"bad SZ value {values[0]!r}") from None
            break

    game = SGFGame()
    for ident, values in events:
        if ident in ("B", "W"):
            game.moves.append((ident, _parse_point(values[0] if values else "",
                                                   size)))
        elif ident in ("AB", "AW"):
            dst = game.setup_black if ident == "AB" else game.setup_white
            for v in values:
                p = _parse_point(v, size)
                if p is not None:               # a setup stone cannot pass
                    dst.append(p)
        else:
            game.props.setdefault(ident, []).extend(values)
    return game


def parse_sgf(text: str) -> List[SGFGame]:
    """Parse an SGF file (possibly a multi-game collection) → main lines.

    Main-line extraction: nodes are collected from a game tree's opening
    ``(`` until the FIRST ``)`` — per the SGF grammar nothing but sibling
    variations can follow a closed subtree, so this is exactly the
    first-variation line. Games whose main line is malformed (bad
    coordinate, bad SZ) are dropped individually; the rest of the
    collection still parses.
    """
    games: List[SGFGame] = []
    events: Optional[List[Tuple[str, List[str]]]] = None
    collecting = False
    depth = 0

    for kind, val in _tokenize(text):
        if kind == "(":
            depth += 1
            if depth == 1:
                events = []
                collecting = True
        elif kind == ")":
            if depth == 0:
                continue                        # stray close paren
            collecting = False                  # first close ends main line
            depth -= 1
            if depth == 0 and events is not None:
                try:
                    games.append(_finish(events))
                except SGFError:
                    pass                        # drop just this game
                events = None
        elif not collecting:
            continue
        elif kind == "IDENT":
            events.append((val, []))
        elif kind == "VALUE":
            if events:                          # ignore values before an ident
                events[-1][1].append(val)
        # ";" needs no action: idents/values already attach in order
    return games
