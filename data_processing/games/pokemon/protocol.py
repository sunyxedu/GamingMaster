"""Pokémon Showdown sim-protocol parser → per-turn observer-view states.

Input is the ``log`` string of a replay JSON (line-oriented sim protocol,
https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md).
Replay logs are OBSERVER VIEW: there are no ``|request|`` lines, so hidden
information (opponent movesets, items, exact stats) must be inferred from
what the log reveals — that is expected and by design.

The parser maintains an evolving battle state for BOTH players and emits
one training sample per (replay, player side):

  - one frame per ``|turn|N`` boundary — the state BEFORE either player's
    turn-N action;
  - the label is that player's visible action that turn: their first
    ``|move|`` (action_type 0) or chosen ``|switch|`` (action_type 1).

Label conventions
-----------------
move:    ``move_idx`` = the used move's index in the acting Pokémon's
         revealed-move list (the move is appended on first use, so a
         first-time move labels the next free slot; if more than 4 moves
         are ever revealed — transform/dynamic movesets — the index is
         clamped to 3).  ``tera`` = 1 when the side terastallized this
         turn (``|-terastallize|`` precedes the move in the log).
switch:  ``switch_idx`` = the incoming Pokémon's own-team slot (0-5,
         team-preview order).

Skipped turns (counted, never mislabeled):
  - no visible action for the side that turn (e.g. only ``|cant|``);
  - the "move" was Struggle (not a real move-slot choice);
  - the move was called by another move (``[from]`` — Sleep Talk etc.);
  - a replacement switch after the side's active fainted that turn
    (forced, and the turn-boundary frame predates the faint);
  - any switch that is not the side's first action of the turn (U-turn /
    Eject Button self-switches).
Team-preview lead selection happens before ``|turn|1`` and is skipped.

State tracked per line type (unknown ``|-minor|`` lines are counted by
prefix in ``ParseResult.unknown``):
    |poke|                     team preview roster (species per slot)
    |switch| |drag| |replace|  active change, HP/status, boosts reset
    |move|                     revealed-move bookkeeping + action label
    |-damage| |-heal| |-sethp| HP fractions ("78/100", "0 fnt", "45/100 par")
    |-status| |-curestatus|    major status
    |-boost| |-unboost| |-setboost| |-clearboost| |-clearallboost|
    |-clearnegativeboost| |-copyboost| |-invertboost| (atk/def/spa/spd/spe)
    |faint|
    |-weather|                 sun/rain/sand/snow/other buckets
    |-fieldstart| |-fieldend|  terrains (other pseudo-weathers ignored)
    |-sidestart| |-sideend|    spikes / tspikes / rocks / web / screens
    |-terastallize|
    |turn| |win| |tie|

``value`` is ±1 from ``|win|`` for the sample's side; replays without a
winner (ties, aborted games) yield no samples.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from model.games.pokemon import (
    FIELD_FEATURE_DIM,
    POKE_FEAT_ACTIVE,
    POKE_FEAT_BOOSTS,
    POKE_FEAT_FAINTED,
    POKE_FEAT_HP,
    POKE_FEAT_REVEALED,
    POKE_FEAT_STATUS,
    POKE_FEAT_TERA,
    POKE_FEATURE_DIM,
    TEAM_SLOTS,
)

# ----------------------------------------------------------------------
# Vocabulary maps
# ----------------------------------------------------------------------

STATUS_TO_IDX = {"": 0, "brn": 1, "par": 2, "psn": 3, "tox": 4, "slp": 5, "frz": 6}
N_STATUS = 7

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")   # accuracy/evasion ignored
N_BOOSTS = len(BOOST_STATS)

# weather one-hot buckets: sun / rain / sand / snow / other (clear = all 0)
WEATHER_TO_IDX = {
    "sunnyday": 0, "desolateland": 0,
    "raindance": 1, "primordialsea": 1,
    "sandstorm": 2,
    "snow": 3, "snowscape": 3, "hail": 3,
}
N_WEATHER = 5           # index 4 = other (e.g. Delta Stream)

TERRAIN_TO_IDX = {
    "electricterrain": 0, "grassyterrain": 1,
    "mistyterrain": 2, "psychicterrain": 3,
}
N_TERRAIN = 4

SIDE_HAZARD_LAYERS = {"spikes": 3, "toxicspikes": 2}
SIDE_FLAGS = ("stealthrock", "stickyweb")
SIDE_SCREENS = ("reflect", "lightscreen", "auroraveil")

N_MOVE_SLOTS = 4


def _norm(name: str) -> str:
    """Showdown id-normalisation: lowercase, strip non-alphanumerics."""
    return "".join(c for c in name.lower() if c.isalnum())


def move_to_id(move: str, buckets: int) -> int:
    """Stable hash-bucket id for a move name; 0 is reserved for
    "slot not yet revealed" (same scheme as :func:`species_to_id`)."""
    if not move:
        return 0
    return zlib.crc32(_norm(move).encode()) % (buckets - 1) + 1


def species_to_id(species: str, buckets: int) -> int:
    """Stable hash-bucket id for a species name; 0 is reserved for unknown."""
    if not species:
        return 0
    return zlib.crc32(_norm(species).encode()) % (buckets - 1) + 1


def base_species(details: str) -> str:
    """``"Ogerpon-Wellspring, F, shiny"`` → ``"Ogerpon-Wellspring"``."""
    return details.split(",")[0].strip()


def parse_hp(hp_str: str) -> Tuple[float, Optional[int], bool]:
    """``"78/100"`` / ``"0 fnt"`` / ``"45/100 par"`` →
    (hp_frac, status_idx or None, fainted)."""
    parts = hp_str.strip().split()
    if not parts:
        return 1.0, None, False
    frac_s, rest = parts[0], parts[1:]
    if rest and rest[0] == "fnt":
        return 0.0, None, True
    if "/" in frac_s:
        num, den = frac_s.split("/", 1)
        try:
            frac = max(0.0, min(1.0, float(num) / max(float(den), 1.0)))
        except ValueError:
            frac = 1.0
    else:
        try:
            frac = max(0.0, min(1.0, float(frac_s) / 100.0))
        except ValueError:
            frac = 1.0
    status = STATUS_TO_IDX.get(rest[0]) if rest else None
    if frac == 0.0:
        return 0.0, status, True
    return frac, status, False


# ----------------------------------------------------------------------
# Battle state
# ----------------------------------------------------------------------

@dataclass
class Pokemon:
    species: str = ""
    hp: float = 1.0
    status: int = 0
    boosts: List[int] = field(default_factory=lambda: [0] * N_BOOSTS)
    fainted: bool = False
    revealed: bool = False       # entered battle at least once
    tera: bool = False
    moves: List[str] = field(default_factory=list)   # revealed moves, in order

    def reset_boosts(self) -> None:
        self.boosts = [0] * N_BOOSTS


@dataclass
class Side:
    name: str = ""
    team: List[Pokemon] = field(default_factory=list)     # ≤ 6, preview order
    active: Optional[int] = None                          # slot of active mon
    nick2slot: Dict[str, int] = field(default_factory=dict)
    hazards: Dict[str, int] = field(default_factory=dict)  # normalised → count
    tera_used: bool = False

    def slot_for(self, nickname: str, details: str) -> int:
        """Resolve a switch-in to a team slot (nickname map → species match
        → first free slot), creating the slot when there was no preview."""
        if nickname in self.nick2slot:
            return self.nick2slot[nickname]
        species = base_species(details)
        norm = _norm(species)
        for i, p in enumerate(self.team):
            if _norm(p.species) == norm and i not in self.nick2slot.values():
                self.nick2slot[nickname] = i
                return i
        # forms (e.g. -Tera / -Mega suffixes): base-name prefix match
        for i, p in enumerate(self.team):
            pn = _norm(p.species)
            if (pn and (norm.startswith(pn) or pn.startswith(norm))
                    and i not in self.nick2slot.values()):
                self.nick2slot[nickname] = i
                return i
        # no preview / unmatched: claim a fresh slot if the roster allows
        if len(self.team) < TEAM_SLOTS:
            self.team.append(Pokemon(species=species))
            self.nick2slot[nickname] = len(self.team) - 1
            return len(self.team) - 1
        # roster full and nothing matches — reuse the active slot (illusion
        # edge cases); never crash on weird logs.
        idx = self.active if self.active is not None else 0
        self.nick2slot[nickname] = idx
        return idx


@dataclass
class Battle:
    sides: Dict[str, Side] = field(default_factory=lambda: {"p1": Side(), "p2": Side()})
    weather: Optional[int] = None      # WEATHER bucket index or None (clear)
    terrain: Optional[int] = None
    turn: int = 0
    winner: Optional[str] = None       # "p1" / "p2"


# ----------------------------------------------------------------------
# Frame encoding (observer view, one side's perspective)
# ----------------------------------------------------------------------

def encode_frame(battle: Battle, side: str) -> Dict[str, torch.Tensor]:
    """Snapshot the current state from ``side``'s perspective.

    Returns per-frame tensors matching the ``PokemonEncoder`` obs keys
    (without the T axis) plus ``active_idx`` for the decoder.
    """
    other = "p2" if side == "p1" else "p1"
    species_ids = torch.zeros(2 * TEAM_SLOTS, dtype=torch.long)
    feats = torch.zeros(2 * TEAM_SLOTS, POKE_FEATURE_DIM)
    mask = torch.zeros(2 * TEAM_SLOTS, dtype=torch.bool)

    for base, s in ((0, battle.sides[side]), (TEAM_SLOTS, battle.sides[other])):
        for i, p in enumerate(s.team[:TEAM_SLOTS]):
            j = base + i
            mask[j] = bool(p.species)
            species_ids[j] = 0  # filled by the converter (needs bucket count)
            feats[j, POKE_FEAT_HP] = p.hp
            feats[j, POKE_FEAT_ACTIVE] = float(s.active == i)
            feats[j, POKE_FEAT_FAINTED] = float(p.fainted)
            feats[j, POKE_FEAT_STATUS + p.status] = 1.0
            for k, b in enumerate(p.boosts):
                feats[j, POKE_FEAT_BOOSTS + k] = max(-1.0, min(1.0, b / 6.0))
            feats[j, POKE_FEAT_REVEALED] = float(p.revealed)
            feats[j, POKE_FEAT_TERA] = float(p.tera)

    fld = torch.zeros(FIELD_FEATURE_DIM)
    if battle.weather is not None:
        fld[battle.weather] = 1.0
    if battle.terrain is not None:
        fld[N_WEATHER + battle.terrain] = 1.0
    fld[N_WEATHER + N_TERRAIN] = min(battle.turn / 50.0, 1.0)
    for off, s in ((10, battle.sides[side]), (15, battle.sides[other])):
        h = s.hazards
        fld[off + 0] = min(h.get("spikes", 0), 3) / 3.0
        fld[off + 1] = min(h.get("toxicspikes", 0), 2) / 2.0
        fld[off + 2] = float(h.get("stealthrock", 0) > 0)
        fld[off + 3] = float(h.get("stickyweb", 0) > 0)
        fld[off + 4] = min(sum(h.get(sc, 0) > 0 for sc in SIDE_SCREENS), 3) / 3.0

    active = battle.sides[side].active
    # Revealed moves of the OWN active mon, in revelation order — the
    # same list the move_idx label indexes, so the label refers to
    # observable content ("" = slot not yet revealed).
    own_team = battle.sides[side].team
    active_moves = [""] * N_MOVE_SLOTS
    if active is not None and active < len(own_team):
        for k, m in enumerate(own_team[active].moves[:N_MOVE_SLOTS]):
            active_moves[k] = m
    return {
        "species": [p.species for p in battle.sides[side].team]
                   + [""] * (TEAM_SLOTS - len(battle.sides[side].team))
                   + [p.species for p in battle.sides[other].team]
                   + [""] * (TEAM_SLOTS - len(battle.sides[other].team)),
        "moves": active_moves,
        "poke_feats": feats,
        "slot_mask": mask,
        "field_seq": fld,
        "active_idx": torch.tensor(active if active is not None else 0,
                                   dtype=torch.long),
    }


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One emitted training frame: state at a |turn| boundary + the side's
    visible action that turn."""
    frame: Dict[str, torch.Tensor]
    action_type: int          # 0 move, 1 switch
    move_idx: int = -1
    switch_idx: int = -1
    tera: int = 0
    turn: int = 0


@dataclass
class ParseResult:
    records: Dict[str, List[TurnRecord]]     # per side
    winner: Optional[str]
    players: Dict[str, str]
    turns: int
    skipped: Dict[str, int]                  # skip-reason histogram
    unknown: Dict[str, int]                  # unhandled |-cmd| histogram


# lines that carry no battle-state information for this pipeline
_IGNORED = {
    "", "j", "l", "c", "c:", "chat", "raw", "html", "uhtml", "inactive",
    "inactiveoff", "t:", "gametype", "gen", "tier", "rated", "rule",
    "clearpoke", "teampreview", "start", "upkeep", "teamsize", "player",
    "-message", "message", "debug", "error", "seed", "choice", "unlink",
    "-hint", "-anim", "n", "join", "leave", "showteam",
}

def _ident(tok: str) -> Tuple[str, str]:
    """``"p1a: Nickname"`` → ("p1", "Nickname"); ``"p2: Name"`` → ("p2", ...)."""
    pos, _, nick = tok.partition(":")
    return pos.strip()[:2], nick.strip()


def parse_log(log: str) -> ParseResult:
    battle = Battle()
    skipped: Dict[str, int] = {}
    unknown: Dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    records: Dict[str, List[TurnRecord]] = {"p1": [], "p2": []}
    # per-turn scratch
    frames: Dict[str, Optional[Dict]] = {"p1": None, "p2": None}
    actions: Dict[str, Optional[TurnRecord]] = {"p1": None, "p2": None}
    consumed: Dict[str, bool] = {"p1": False, "p2": False}   # action seen but unusable
    # Emergency Exit / Wimp Out / Eject Button / Eject Pack force the
    # holder out mid-turn WITHOUT a faint: the replacement |switch| is not
    # the side's turn-start decision and must not be labeled as one.
    forced_pivot: Dict[str, bool] = {"p1": False, "p2": False}
    tera_pending: Dict[str, bool] = {"p1": False, "p2": False}
    active_fainted: Dict[str, bool] = {"p1": False, "p2": False}

    def finalize_turn() -> None:
        for side in ("p1", "p2"):
            if frames[side] is None:
                continue
            rec = actions[side]
            if rec is not None:
                records[side].append(rec)
            else:
                _skip("no_visible_action")
        for side in ("p1", "p2"):
            frames[side] = None
            actions[side] = None
            consumed[side] = False
            tera_pending[side] = False
            active_fainted[side] = False
            forced_pivot[side] = False

    def snapshot_turn(n: int) -> None:
        battle.turn = n
        for side in ("p1", "p2"):
            frames[side] = encode_frame(battle, side)

    for raw_line in log.split("\n"):
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        cmd = parts[1] if len(parts) > 1 else ""
        args = parts[2:]

        if cmd == "player" and len(args) >= 2 and args[0] in battle.sides:
            if args[1]:
                battle.sides[args[0]].name = args[1]
            continue

        if cmd == "poke" and len(args) >= 2:
            side = battle.sides.get(args[0])
            if side is not None and len(side.team) < TEAM_SLOTS:
                side.team.append(Pokemon(species=base_species(args[1])))
            continue

        if cmd in ("switch", "drag", "replace") and len(args) >= 2:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.slot_for(nick, args[1])
            # the chosen-switch label must be taken BEFORE mutating state
            is_choice = (
                cmd == "switch"
                and battle.turn > 0
                and actions[pos] is None
                and not consumed[pos]
                and not active_fainted[pos]
                and not forced_pivot[pos]
                and frames[pos] is not None
            )
            if cmd == "switch" and battle.turn > 0 and not is_choice:
                if forced_pivot[pos]:
                    # Emergency Exit / Eject pivot: the player's actual
                    # turn decision was (usually) a cancelled move that
                    # never reached the log — no valid label this turn.
                    forced_pivot[pos] = False
                    consumed[pos] = True
                    _skip("forced_pivot_switch")
                elif active_fainted[pos]:
                    _skip("forced_replacement_switch")
                elif actions[pos] is not None or consumed[pos]:
                    _skip("mid_turn_switch")
            # state update
            if side.active is not None and side.active < len(side.team):
                prev = side.team[side.active]
                prev.reset_boosts()
            side.active = slot
            p = side.team[slot]
            p.revealed = True
            p.reset_boosts()
            if len(args) >= 3 and args[2]:
                hp, st, fnt = parse_hp(args[2])
                p.hp, p.fainted = hp, fnt
                if st is not None:
                    p.status = st
            if is_choice:
                actions[pos] = TurnRecord(
                    frame=frames[pos], action_type=1, switch_idx=slot,
                    turn=battle.turn,
                )
            continue

        if cmd == "move" and len(args) >= 2:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            move_name = args[1]
            called = any("[from]" in a for a in args[2:])
            if called:
                # called by Sleep Talk / Copycat / ... — not the player's
                # slot choice and possibly not even in the moveset.
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is None or slot >= len(side.team):
                continue
            p = side.team[slot]
            if _norm(move_name) == "struggle":
                if actions[pos] is None and not consumed[pos]:
                    consumed[pos] = True
                    _skip("struggle")
                continue
            # revealed-move bookkeeping (always, even off-label turns)
            norm_moves = [_norm(m) for m in p.moves]
            if _norm(move_name) in norm_moves:
                idx = norm_moves.index(_norm(move_name))
            else:
                p.moves.append(move_name)
                idx = len(p.moves) - 1
            if actions[pos] is None and not consumed[pos] and frames[pos] is not None:
                actions[pos] = TurnRecord(
                    frame=frames[pos], action_type=0,
                    move_idx=min(idx, N_MOVE_SLOTS - 1),
                    tera=int(tera_pending[pos]), turn=battle.turn,
                )
            continue

        if cmd == "cant" and args:
            pos, _ = _ident(args[0])
            if pos in consumed and actions.get(pos) is None:
                consumed[pos] = True
                _skip("cant")
            continue

        if cmd in ("-activate", "-enditem") and len(args) >= 2:
            # Mid-turn pivot triggers: the holder's next |switch| this
            # turn is forced by the ability/item, not a turn decision.
            pos, _ = _ident(args[0])
            effect = args[1]
            if pos in forced_pivot and effect in (
                "ability: Emergency Exit", "ability: Wimp Out",
                "Eject Button", "Eject Pack",
            ):
                forced_pivot[pos] = True
            continue

        if cmd in ("-damage", "-heal", "-sethp") and len(args) >= 2:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is None or slot >= len(side.team):
                continue
            p = side.team[slot]
            hp, st, fnt = parse_hp(args[1])
            p.hp = hp
            if fnt:
                p.fainted = True
            if st is not None:
                p.status = st
            continue

        if cmd == "-status" and len(args) >= 2:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                side.team[slot].status = STATUS_TO_IDX.get(args[1], 0)
            continue

        if cmd == "-curestatus" and args:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                side.team[slot].status = 0
            continue

        if cmd in ("-boost", "-unboost", "-setboost") and len(args) >= 3:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None or args[1] not in BOOST_STATS:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is None or slot >= len(side.team):
                continue
            k = BOOST_STATS.index(args[1])
            try:
                amt = int(args[2])
            except ValueError:
                continue
            b = side.team[slot].boosts
            if cmd == "-boost":
                b[k] = max(-6, min(6, b[k] + amt))
            elif cmd == "-unboost":
                b[k] = max(-6, min(6, b[k] - amt))
            else:
                b[k] = max(-6, min(6, amt))
            continue

        if cmd == "-clearboost" and args:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                side.team[slot].reset_boosts()
            continue

        if cmd == "-clearnegativeboost" and args:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                b = side.team[slot].boosts
                side.team[slot].boosts = [max(0, x) for x in b]
            continue

        if cmd == "-clearallboost":
            for s in battle.sides.values():
                if s.active is not None and s.active < len(s.team):
                    s.team[s.active].reset_boosts()
            continue

        if cmd in ("-copyboost", "-invertboost", "-swapboost") and args:
            # rare volatile-boost shuffles — approximate: invert supported,
            # copy/swap just reset the source-side view (counted as unknown
            # would hide them; they DO occur, so approximate cheaply).
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                if cmd == "-invertboost":
                    side.team[slot].boosts = [-x for x in side.team[slot].boosts]
                else:
                    side.team[slot].reset_boosts()
            continue

        if cmd == "faint" and args:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                p = side.team[slot]
                p.fainted, p.hp = True, 0.0
                p.reset_boosts()
                if side.active == slot:
                    active_fainted[pos] = True
            continue

        if cmd == "-weather" and args:
            w = _norm(args[0])
            if w in ("none", ""):
                battle.weather = None
            else:
                battle.weather = WEATHER_TO_IDX.get(w, N_WEATHER - 1)
            continue

        if cmd == "-fieldstart" and args:
            name = _norm(args[0].split(":")[-1])
            if name in TERRAIN_TO_IDX:
                battle.terrain = TERRAIN_TO_IDX[name]
            # Trick Room etc.: not modelled in v1 — silently fine
            continue

        if cmd == "-fieldend" and args:
            name = _norm(args[0].split(":")[-1])
            if name in TERRAIN_TO_IDX and battle.terrain == TERRAIN_TO_IDX[name]:
                battle.terrain = None
            continue

        if cmd in ("-sidestart", "-sideend") and len(args) >= 2:
            pos, _ = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            name = _norm(args[1].split(":")[-1])
            if name in SIDE_HAZARD_LAYERS or name in SIDE_FLAGS or name in SIDE_SCREENS:
                if cmd == "-sidestart":
                    cap = SIDE_HAZARD_LAYERS.get(name, 1)
                    side.hazards[name] = min(side.hazards.get(name, 0) + 1, cap)
                else:
                    side.hazards[name] = 0
            continue

        if cmd == "-terastallize" and args:
            pos, nick = _ident(args[0])
            side = battle.sides.get(pos)
            if side is None:
                continue
            slot = side.nick2slot.get(nick, side.active)
            if slot is not None and slot < len(side.team):
                side.team[slot].tera = True
            side.tera_used = True
            tera_pending[pos] = True
            continue

        if cmd == "turn" and args:
            try:
                n = int(args[0])
            except ValueError:
                continue
            finalize_turn()
            snapshot_turn(n)
            continue

        if cmd == "win" and args:
            finalize_turn()
            name = args[0].strip()
            for pos, s in battle.sides.items():
                if s.name == name:
                    battle.winner = pos
            continue

        if cmd == "tie":
            finalize_turn()
            continue

        if cmd in _IGNORED or cmd.startswith("c:"):
            continue

        unknown[cmd] = unknown.get(cmd, 0) + 1

    # log ended without |win|/|turn| finalization (forfeit before |win| is
    # still followed by |win|, but be safe)
    finalize_turn()

    return ParseResult(
        records=records,
        winner=battle.winner,
        players={pos: s.name for pos, s in battle.sides.items()},
        turns=battle.turn,
        skipped=skipped,
        unknown=unknown,
    )


# ----------------------------------------------------------------------
# Sample assembly (per replay, per side)
# ----------------------------------------------------------------------

def replay_to_samples(
    replay: Dict, species_buckets: int = 512, move_buckets: int = 256,
) -> Tuple[List[Dict], ParseResult]:
    """One replay JSON dict → list of shard samples (one per side with ≥1
    labeled frame; empty when there is no winner)."""
    result = parse_log(replay.get("log", ""))
    samples: List[Dict] = []
    if result.winner is None:
        return samples, result

    for pos in ("p1", "p2"):
        recs = result.records[pos]
        if not recs:
            continue
        T = len(recs)
        species_ids = torch.zeros(T, 2 * TEAM_SLOTS, dtype=torch.long)
        move_ids = torch.zeros(T, N_MOVE_SLOTS, dtype=torch.long)
        poke_feats = torch.zeros(T, 2 * TEAM_SLOTS, POKE_FEATURE_DIM)
        slot_mask = torch.zeros(T, 2 * TEAM_SLOTS, dtype=torch.bool)
        field_seq = torch.zeros(T, FIELD_FEATURE_DIM)
        action_type = torch.zeros(T, dtype=torch.long)
        move_idx = torch.full((T,), -1, dtype=torch.long)
        switch_idx = torch.full((T,), -1, dtype=torch.long)
        tera = torch.zeros(T, dtype=torch.long)
        active_idx = torch.zeros(T, dtype=torch.long)

        for t, rec in enumerate(recs):
            f = rec.frame
            for j, sp in enumerate(f["species"]):
                species_ids[t, j] = species_to_id(sp, species_buckets)
            for k, mv in enumerate(f.get("moves", [])[:N_MOVE_SLOTS]):
                move_ids[t, k] = move_to_id(mv, move_buckets)
            poke_feats[t] = f["poke_feats"]
            slot_mask[t] = f["slot_mask"]
            field_seq[t] = f["field_seq"]
            active_idx[t] = f["active_idx"]
            action_type[t] = rec.action_type
            move_idx[t] = rec.move_idx
            switch_idx[t] = rec.switch_idx
            tera[t] = rec.tera

        value = torch.full((T,), 1.0 if result.winner == pos else -1.0)
        samples.append({
            "obs": {
                "species_ids": species_ids,
                "move_ids": move_ids,
                "poke_feats": poke_feats,
                "slot_mask": slot_mask,
                "field_seq": field_seq,
            },
            "tgt": {
                "action_type": action_type,
                "move_idx": move_idx,
                "switch_idx": switch_idx,
                "tera": tera,
                "active_idx": active_idx,
                "value": value,
            },
            "meta": {
                "replay_id": replay.get("id", "?"),
                "side": pos,
                "player": result.players.get(pos, "?"),
                "rating": replay.get("rating"),
                "format": replay.get("formatid", replay.get("format", "?")),
                "won": result.winner == pos,
            },
        })
    return samples, result
