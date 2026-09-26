"""Information-leak audit of the Pokémon Showdown pipeline.

gmdata pokemon (data_processing/rs/gmdata/src/pokemon.rs) → model/games/pokemon.py.

Every obs tensor of a (replay, side) sample must be information that player really has when
choosing its action for that turn.  The converter reads spectator logs, so the audit reduces to
two mathematically clean invariances plus a few semantic guards:

FUTURE (causality)   the frame of turn t is a function of the log up to and including the
                     ``|turn|t`` line only — not of anything logged later, in particular not of
                     turn t's own ``|move|`` lines (both players choose simultaneously), nor of
                     moves / items / abilities / tera types revealed later (no whole-battle
                     aggregation).  Tested by re-converting truncated logs (every line, and every
                     turn boundary followed by synthetic actions) and by perturbing turn t and
                     everything after it.
PRIVACY              a side's frames must not depend on what is private to the opponent.  A
                     spectator log only contains information once it is revealed, so this is
                     the opponent-revelation perturbation below, plus guards that out-of-band
                     team data (``|showteam|`` packed sets, a replay ``inputlog``, ratings,
                     chat/timer lines) never reaches obs, and that without team preview the
                     opponent's team grows only as its Pokémon are switched in.
VALUE                ``tgt.value`` (the hindsight outcome) never reaches obs / other targets.
FRAMES               (converter v2) one frame per request the player answers: the team-preview
                     lead, every turn (whatever its outcome — an unseen choice is a masked
                     label, not a missing frame), every forced switch, snapshotted when the
                     request is issued (simultaneous replacements never see each other); every
                     move label names its move (tgt.move_id), and legal masks are functions of
                     the frame's state.
MODEL                the pokemon network is causal over frames, including the ``move_ids``
                     (LEGAL_FROM_OBS) and ``active_idx`` (state label) decode paths.

The fixtures under tests/fixtures/{pokemon,showdown} are always used; point
``GM_AUDIT_POKEMON_REPLAYS`` at a directory of more replay .json files (e.g. the output of
``gmdata pokemon fetch --out /tmp/pk``) to run the converter checks on those too.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from data_processing.games.dataset import load_gm_samples
from model.games.pokemon import (
    FIELD_FORCED_SWITCH,
    FIELD_TEAM_PREVIEW,
    POKE_FEAT_ACTIVE,
    POKE_FEAT_FAINTED,
    move_bucket,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
FIX_DIRS = [
    os.path.join(ROOT, "tests", "fixtures", "pokemon"),
    os.path.join(ROOT, "tests", "fixtures", "showdown"),
]
EXTRA_ENV = "GM_AUDIT_POKEMON_REPLAYS"
EXTRA_MAX = 150
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
TURN_CLIP = 50  # obs.field_seq[..., 9] = min(turn / 50, 1): exact turn numbers below this
MOVE_BUCKETS = 1024  # gmdata pokemon convert --move-buckets default (shard meta "move_buckets")

Key = Tuple[str, str]


# ----------------------------------------------------------------------------------------------
# replay / conversion helpers
# ----------------------------------------------------------------------------------------------


def _replays() -> List[Dict]:
    files = [f for d in FIX_DIRS for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    extra = os.environ.get(EXTRA_ENV)
    if extra and os.path.isdir(extra):
        files += sorted(glob.glob(os.path.join(extra, "*.json")))[:EXTRA_MAX]
    out = []
    for f in files:
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("log"), str) and "|turn|" in r["log"]:
            out.append(r)
    return out


def _side(tok: str) -> Optional[int]:
    tok = tok.strip()
    if tok.startswith("p1"):
        return 0
    if tok.startswith("p2"):
        return 1
    return None


def _names(lines: List[str]) -> Dict[int, str]:
    names = {}
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "player" and _side(p[2]) is not None and p[3]:
            names[_side(p[2])] = p[3]
    return names


def _turn_lines(lines: List[str]) -> List[Tuple[int, int]]:
    """[(line index, turn number)] of the ``|turn|N`` lines."""
    out = []
    for i, ln in enumerate(lines):
        p = ln.split("|")
        if len(p) > 2 and p[1] == "turn" and p[2].strip().isdigit():
            out.append((i, int(p[2])))
    return out


def _convert(gmdata: str, root, tag: str, replays: Dict[str, Dict]) -> Dict[Key, Dict]:
    """Convert {replay id: replay json} in one gmdata call → {(replay id, "p1"/"p2"): sample}."""
    src = root / f"{tag}_in"
    out = root / f"{tag}_out"
    src.mkdir()
    for rid, r in replays.items():
        (src / f"{rid}.json").write_text(json.dumps({**r, "id": rid}))
    subprocess.run(
        [
            gmdata,
            "pokemon",
            "convert",
            "--input",
            str(src),
            "--out",
            str(out),
            "--shard-size",
            "1000000",
        ],
        check=True,
        capture_output=True,
    )
    res = {}
    for s in load_gm_samples(str(out)) if out.exists() else []:
        res[(s["meta"]["replay_id"], s["meta"]["side"])] = s
    return res


def _rec(rid: str, log: str, base: Optional[Dict] = None) -> Dict:
    r = {"formatid": "gen9ou", **(base or {})}
    r["log"] = log
    r["id"] = rid
    return r


def _n(s: Optional[Dict]) -> int:
    return 0 if s is None else int(s["tgt"]["action_type"].shape[0])


def _turns(s: Optional[Dict]) -> List[int]:
    """Turn number of every frame (exact below TURN_CLIP, TURN_CLIP after)."""
    if s is None:
        return []
    return [int(round(float(x) * TURN_CLIP)) for x in s["obs"]["field_seq"][:, 9]]


def _n_upto(s: Optional[Dict], t: int, strict: bool = False) -> int:
    """Number of frames whose turn is ≤ t (< t when strict); requires t < TURN_CLIP."""
    assert t < TURN_CLIP
    return sum(1 for x in _turns(s) if (x < t if strict else x <= t))


def _diff(a: Dict, i: int, b: Dict, j: int, tgt: bool = True) -> List[str]:
    """Keys whose frame a[i] differs from b[j] (obs; plus every tgt except the hindsight value)."""
    assert set(a["obs"]) == set(b["obs"]) and set(a["tgt"]) == set(b["tgt"])
    bad = [f"obs.{k}" for k in sorted(a["obs"]) if not torch.equal(a["obs"][k][i], b["obs"][k][j])]
    if tgt:
        bad += [
            f"tgt.{k}"
            for k in sorted(a["tgt"])
            if k != "value" and not torch.equal(a["tgt"][k][i], b["tgt"][k][j])
        ]
    return bad


#: the targets that name the side's choice (hindsight: the choice is logged after its frame)
ACTION_LABELS = ("action_type", "action_known", "move_id", "move_idx", "switch_idx", "tera")
#: the targets that are functions of the frame's state (legal masks, the acting slot)
STATE_TARGETS = ("active_idx", "legal_action_type", "legal_switch")


def _forced(s: Dict, f: int) -> bool:
    """Frame f answers a forced-switch request (KO replacement, pivot, eject)."""
    return bool(s["obs"]["field_seq"][f, FIELD_FORCED_SWITCH] == 1)


def _preview(s: Dict, f: int) -> bool:
    """Frame f answers the team-preview request (the lead)."""
    return bool(s["obs"]["field_seq"][f, FIELD_TEAM_PREVIEW] == 1)


def _assert_prefix(
    full: Optional[Dict],
    part: Optional[Dict],
    n: int,
    what: str,
    tgt: bool = True,
    unknown_ok: bool = False,
):
    """The first n frames of ``part`` are those of ``full``: obs and (``tgt``) every target
    but the hindsight value.  ``unknown_ok``: a frame whose choice ``part`` has not seen
    (action_known 0 — its log ends before the choice) may differ in the action labels only."""
    assert _n(part) >= n and _n(full) >= n, (what, _n(full), _n(part), n)
    groups = ("obs", "tgt") if tgt else ("obs",)
    if n and full is not None and part is not None:
        assert set(full["obs"]) == set(part["obs"]) and set(full["tgt"]) == set(part["tgt"]), what
    if n == 0 or all(
        torch.equal(full[g][k][:n], part[g][k][:n])
        for g in groups
        for k in full[g]
        if (g, k) != ("tgt", "value")
    ):
        return
    for i in range(n):
        open_ = unknown_ok and not _labelled(part, i)
        skip = {f"tgt.{k}" for k in ACTION_LABELS} if open_ else set()
        d = [k for k in _diff(full, i, part, i, tgt) if k not in skip]
        assert not d, f"{what}: frame {i} differs in {d}"


# ----------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run(
        [CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS, capture_output=True, text=True
    )
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


@pytest.fixture(scope="module")
def replays():
    rs = _replays()
    if not rs:
        pytest.skip("no pokemon replay fixtures")
    return rs


@pytest.fixture(scope="module")
def full(gmdata, replays, tmp_path_factory):
    """Unmodified conversion of every replay, keyed (replay id, side)."""
    res = _convert(
        gmdata, tmp_path_factory.mktemp("pk_full"), "full", {r["id"]: r for r in replays}
    )
    assert res, "the fixtures produced no samples"
    return res


# ----------------------------------------------------------------------------------------------
# FUTURE leaks: truncation
# ----------------------------------------------------------------------------------------------


def test_every_line_truncation_is_a_prefix(gmdata, replays, full, tmp_path):
    """Cut every log after every line (appending a ``|win|`` so the converter keeps it): the
    cut's frames — obs AND every non-value target — are the first frames of the full conversion.
    No obs/label of a decision may depend on anything logged after it (no whole-battle
    normalisers, revealed-move sets, final team, number of turns, ...).  The one exception is
    the decision still open at the cut (a turn whose choice is not logged yet, the second side
    of a simultaneous request): its frame exists — frames never wait for their outcome — with
    the label masked (action_known 0), and only its action labels differ."""
    jobs, want = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        p1 = _names(lines).get(0)
        assert p1, r["id"]
        for i in range(len(lines) + 1):
            cid = f"{r['id']}~L{i}"
            jobs[cid] = _rec(cid, "\n".join(lines[:i] + [f"|win|{p1}"]), r)
            want[cid] = (r["id"], i == len(lines))
    cut = _convert(gmdata, tmp_path, "lines", jobs)
    compared = recovered = opened = 0
    for cid, (rid, last) in want.items():
        for side in ("p1", "p2"):
            part, whole = cut.get((cid, side)), full.get((rid, side))
            if part is None:
                continue
            _assert_prefix(whole, part, _n(part), f"{cid}/{side}", unknown_ok=True)
            # labels never appear retroactively except for the decision open at the cut
            late = [f for f in range(_n(part)) if not _labelled(part, f) and _labelled(whole, f)]
            assert late in ([], [_n(part) - 1]), (cid, side, late, _n(part))
            opened += len(late)
            compared += _n(part)
            if last:
                assert _n(part) == _n(whole), (cid, side)
                recovered += 1
    assert compared > 1000 and recovered >= len(replays) and opened > 100, (
        compared,
        recovered,
        opened,
    )


def _synthetic_turn_cut(lines: List[str], i: int, p1: str) -> str:
    """The log up to and including line i (a ``|turn|t`` line) followed by a Splash for both
    active Pokémon and a ``|win|``: its last frame per side is the turn-t decision frame computed
    WITHOUT any line of turn t."""
    ident = {}
    for ln in lines[:i]:
        p = ln.split("|")
        if len(p) > 2 and p[1] in ("switch", "drag", "replace") and _side(p[2]) is not None:
            ident[_side(p[2])] = p[2].strip()
    extra = [f"|move|{ident[s]}|Splash|{ident[s]}" for s in (0, 1) if s in ident]
    return "\n".join(lines[: i + 1] + extra + [f"|win|{p1}"])


def _state_diff(a: Dict, i: int, b: Dict, j: int) -> List[str]:
    """STATE_TARGETS on which frame a[i] differs from b[j]."""
    return [f"tgt.{k}" for k in STATE_TARGETS if not torch.equal(a["tgt"][k][i], b["tgt"][k][j])]


def test_turn_frame_is_a_function_of_the_log_through_the_turn_line(gmdata, replays, full, tmp_path):
    """Decision-time alignment: for every ``|turn|t`` the frame of turn t equals the one computed
    from the log truncated right after that line (plus synthetic actions), i.e. it contains none
    of turn t's own events — not the simultaneous choices, not the ``|move|`` lines, not the
    terastallization, not the damage — and neither do its legal masks.  Frames before turn t
    match obs + labels.  Every turn has its frame for both sides, followed only by the side's
    forced-switch frames of that turn (the requests answered during it)."""
    jobs, meta = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        for i, t in _turn_lines(lines):
            cid = f"{r['id']}~T{t}"
            jobs[cid] = _rec(cid, _synthetic_turn_cut(lines, i, p1), r)
            meta[cid] = (r["id"], t)
    cut = _convert(gmdata, tmp_path, "turns", jobs)
    by_replay: Dict[Key, List[Tuple[int, int, Dict]]] = {}
    for cid, (rid, t) in meta.items():
        for side in ("p1", "p2"):
            part = cut.get((cid, side))
            assert part is not None and _n(part) >= 1, (
                f"no synthetic decision frame for {cid}/{side}"
            )
            by_replay.setdefault((rid, side), []).append((t, _n(part) - 1, part))
    checked_t = 0
    for (rid, side), cuts in by_replay.items():
        whole = full.get((rid, side))
        cuts.sort(key=lambda x: x[0])
        for idx, (t, k, part) in enumerate(cuts):
            _assert_prefix(whole, part, k, f"{rid}/{side} before turn {t}")
            if t < TURN_CLIP:
                assert _turns(part)[-1] == t, (rid, side, t, _turns(part))
            assert not _forced(part, k) and not _preview(part, k), (rid, side, t)
            k_next = cuts[idx + 1][1] if idx + 1 < len(cuts) else _n(whole)
            assert k_next > k, (rid, side, t, k, k_next)
            between = list(range(k + 1, k_next))
            assert all(_forced(whole, f) for f in between), (rid, side, t, k, k_next)
            if t < TURN_CLIP:
                assert _turns(whole)[k] == t and all(_turns(whole)[f] == t for f in between)
            d = _diff(whole, k, part, k, tgt=False) + _state_diff(whole, k, part, k)
            assert not d, f"{rid}/{side}: turn-{t} frame depends on turn-{t} events ({d})"
            # the synthetic Splash labels the cut's turn-t frame by its identity (never revealed
            # before, so no obs.move_ids slot)
            splash = move_bucket("Splash", MOVE_BUCKETS)
            assert int(part["tgt"]["move_id"][k]) == splash, (rid, side, t)
            assert int(part["tgt"]["move_idx"][k]) == -1, (rid, side, t)
            checked_t += 1
    assert checked_t > 500, checked_t


# ----------------------------------------------------------------------------------------------
# FUTURE leaks: perturbation
# ----------------------------------------------------------------------------------------------


def _turn_blocks(lines: List[str]) -> Dict[int, Tuple[int, int]]:
    """turn t → (first, end) line range of turn t's events (after ``|turn|t``, before the next)."""
    tl = _turn_lines(lines)
    return {
        t: (i + 1, tl[k + 1][0] if k + 1 < len(tl) else len(lines)) for k, (i, t) in enumerate(tl)
    }


def _scramble_turn(lines: List[str], a: int, b: int) -> List[str]:
    """Rewrite the content of every event in lines[a:b]: move names, HP, boosts, status, weather,
    tera type."""
    out = list(lines)
    for i in range(a, b):
        p = out[i].split("|")
        if len(p) < 3:
            continue
        cmd = p[1]
        if cmd == "-weather":
            p[2] = "RainDance"
        elif len(p) < 4:
            pass
        elif cmd == "move":
            p[3] = "Zz Sentinel Move"
        elif cmd in ("-damage", "-heal", "-sethp"):
            p[3] = "37/100"
        elif cmd in ("switch", "drag", "replace") and len(p) > 4:
            p[4] = "37/100 brn"
        elif cmd in ("-boost", "-unboost", "-setboost") and len(p) > 4:
            p[4] = "3"
        elif cmd == "-status":
            p[3] = "tox"
        elif cmd == "-terastallize":
            p[3] = "Stellar"
        out[i] = "|".join(p)
    return out


def test_turn_events_do_not_reach_their_own_decision_frame(gmdata, replays, full, tmp_path):
    """Scrambling everything that happens during turn t (both players' moves, damage, boosts,
    status, weather, tera) leaves every frame up to and including turn t's decision frame
    unchanged (its legal masks included): the frame is the state BEFORE the simultaneous
    choices."""
    jobs, meta = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        for t, (a, b) in _turn_blocks(lines).items():
            if t >= TURN_CLIP:
                continue
            cid = f"{r['id']}~S{t}"
            jobs[cid] = _rec(cid, "\n".join(_scramble_turn(lines, a, b)), r)
            meta[cid] = (r["id"], t)
    got = _convert(gmdata, tmp_path, "scramble", jobs)
    checked = 0
    for cid, (rid, t) in meta.items():
        for side in ("p1", "p2"):
            whole, pert = full.get((rid, side)), got.get((cid, side))
            k = _n_upto(whole, t, strict=True)
            assert _n_upto(pert, t, strict=True) == k, (cid, side)
            _assert_prefix(whole, pert, k, f"{cid}/{side} before turn {t}")
            if _n_upto(whole, t) > k and _n_upto(pert, t) > k:
                d = _diff(whole, k, pert, k, tgt=False) + _state_diff(whole, k, pert, k)
                assert not d, f"{cid}/{side}: turn-{t} frame sees turn-{t} events ({d})"
                checked += 1
    assert checked > 300, checked


# ----------------------------------------------------------------------------------------------
# PRIVACY: opponent information
# ----------------------------------------------------------------------------------------------

_FROM_RE = re.compile(r"^(\[from\] ?(?:item|ability): ?)(.*)$")


def _perturb_opponent_reveals(lines: List[str], start: int, opp: int) -> List[str]:
    """Change every opponent-private fact the log reveals from line ``start`` on: the opponent's
    move names, items, abilities, tera types (``-terastallize`` / ``detailschange ... tera:``)."""
    out = list(lines)
    for i in range(start, len(lines)):
        p = out[i].split("|")
        if len(p) < 4:
            continue
        cmd, subj = p[1], _side(p[2])
        owner = subj
        for x in p[3:]:
            if x.startswith("[of] "):
                owner = _side(x[5:])
        if subj == opp:
            if cmd == "move":
                p[3] = "Zz Hidden " + p[3]
            elif cmd in ("-item", "-enditem", "-ability"):
                p[3] = "Zz Hidden " + p[3]
            elif cmd == "-terastallize":
                p[3] = "Fairy" if p[3] != "Fairy" else "Stellar"
            elif cmd == "detailschange":
                p[3] = re.sub(r"tera:\w+", "tera:Stellar", p[3])
        if owner == opp:
            p = [p[0], p[1], p[2]] + [
                _FROM_RE.sub(lambda m: m.group(1) + "Zz Hidden " + m.group(2), x) for x in p[3:]
            ]
        out[i] = "|".join(p)
    return out


def test_opponent_reveals_do_not_reach_earlier_frames(gmdata, replays, full, tmp_path):
    """Privacy/no-aggregation: rewriting every move, item, ability and tera type the opponent
    reveals after the ``|turn|t`` line leaves the side's frames through its turn-t decision
    bit-identical (obs and labels) — nothing about the opponent's set is known before it is
    revealed.  (The side's forced-switch frames of turn t are snapshots taken after some of turn
    t's reveals, so they are not part of the comparison.)"""
    jobs, meta = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        for i, t in _turn_lines(lines):
            if t >= TURN_CLIP:
                continue
            for me in (0, 1):
                cid = f"{r['id']}~P{me}_{t}"
                jobs[cid] = _rec(cid, "\n".join(_perturb_opponent_reveals(lines, i + 1, 1 - me)), r)
                meta[cid] = (r["id"], t, me)
    got = _convert(gmdata, tmp_path, "opp", jobs)
    checked = 0
    for cid, (rid, t, me) in meta.items():
        side = ("p1", "p2")[me]
        whole, pert = full.get((rid, side)), got.get((cid, side))
        k = _n_upto(whole, t, strict=True) + 1  # through the turn-t decision frame
        assert not _forced(whole, k - 1) and _turns(whole)[k - 1] == t, (cid, side)
        _assert_prefix(whole, pert, k, f"{cid}/{side} up to turn {t}")
        checked += k
    assert checked > 1000, checked


def _perturb_meta(r: Dict) -> Dict:
    """Same battle, different out-of-band data: player names (consistently), ratings, avatars,
    chat / join / timer / timestamp / raw lines dropped, and a replay ``inputlog`` holding a packed
    team (full sets of both players — never visible to the opponent)."""
    lines = r["log"].split("\n")
    names = _names(lines)
    ren = {names[0]: "Audit Alice", names[1]: "Audit Bob"}
    out = []
    for ln in lines:
        p = ln.split("|")
        cmd = p[1] if len(p) > 1 else ""
        if cmd in (
            "j",
            "J",
            "l",
            "L",
            "c",
            "c:",
            "chat",
            "raw",
            "html",
            "uhtml",
            "inactive",
            "inactiveoff",
            "t:",
            "n",
            "N",
        ):
            continue
        if cmd == "player" and len(p) > 3:
            p[3] = ren.get(p[3], p[3])
            p[4:] = ["audit-avatar", "999"][: max(0, len(p) - 4)]
        elif cmd == "win" and len(p) > 2:
            p[2] = ren.get(p[2], p[2])
        out.append("|".join(p))
    team = (
        "Gholdengo||choicescarf|goodasgold|makeitrain,shadowball,trick,nastyplot|Timid"
        "|,,,252,4,252||,0,,,,||50|,,,,,Steel"
    )
    start = '>start {"formatid":"gen9ou"}'
    players = [f'>player p{i} {{"team":"{team}"}}' for i in (1, 2)]
    return {
        **r,
        "log": "\n".join(out),
        "rating": 1,
        "players": list(ren.values()),
        "inputlog": "\n".join([start] + players),
    }


def _inject_showteam(r: Dict) -> Dict:
    """Open-team-sheet lines (``|showteam|``: every move / item / ability / tera type of both
    teams) right after team preview — gen9ou has no open team sheets, so obs must ignore them."""
    lines = r["log"].split("\n")
    pk = "|".join(
        [
            "Gholdengo",
            "",
            "Choice Scarf",
            "Good as Gold",
            "Make It Rain,Shadow Ball,Trick,Nasty Plot",
            "Timid",
            "",
            "",
            "",
            "",
            "50",
            ",,,,,Fairy",
        ]
    )
    at = next((i for i, ln in enumerate(lines) if ln.startswith("|teampreview")), 0) + 1
    ins = [f"|showteam|p1|{pk}]{pk}", f"|showteam|p2|{pk}]{pk}"]
    return {**r, "log": "\n".join(lines[:at] + ins + lines[at:])}


def test_out_of_band_team_data_never_reaches_obs(gmdata, replays, full, tmp_path):
    """Ratings, names, chat/timer lines, a replay ``inputlog`` and ``|showteam|`` packed sets are
    not game state visible to the opponent (or not game state at all): every obs and every target
    of every frame is unchanged."""
    jobs = {}
    for r in replays:
        jobs[f"{r['id']}~M"] = {**_perturb_meta(r), "id": f"{r['id']}~M"}
        jobs[f"{r['id']}~ST"] = {**_inject_showteam(r), "id": f"{r['id']}~ST"}
    got = _convert(gmdata, tmp_path, "oob", jobs)
    for cid in jobs:
        rid = cid.rsplit("~", 1)[0]
        for side in ("p1", "p2"):
            whole, pert = full.get((rid, side)), got.get((cid, side))
            assert _n(whole) == _n(pert), (cid, side)
            _assert_prefix(whole, pert, _n(whole), f"{cid}/{side}")
            if whole is not None:
                assert torch.equal(whole["tgt"]["value"], pert["tgt"]["value"]), (cid, side)


def test_without_team_preview_the_opponent_team_grows_only_by_switch_ins(
    gmdata, replays, full, tmp_path
):
    """Team-preview rules: gen9ou shows both teams' species before turn 1 (``|poke|`` lines), so
    all 12 species are public from frame 0 (the lead frame).  In a format without team preview
    (drop the ``|poke|`` / ``|teampreview|`` lines) the lead is no decision (no turn-0 frame) and
    the opponent's known slots at the turn-t frame must be exactly the Pokémon switched in before
    the ``|turn|t`` line (no end-of-battle team aggregation); at a forced-switch frame of turn t,
    between those and the ones switched in before the frame's own switch line."""
    jobs, meta = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        kept = [
            ln for ln in lines if not (ln.startswith("|poke|") or ln.startswith("|teampreview"))
        ]
        cid = f"{r['id']}~NP"
        jobs[cid] = _rec(cid, "\n".join(kept), r)
        seen: Dict[int, set] = {0: set(), 1: set()}
        at_turn: Dict[int, Dict[int, int]] = {0: {0: 0, 1: 0}}
        # side → the opponent-count before each of its |switch| lines
        before_switch: Dict[int, List[Dict[int, int]]] = {0: [], 1: []}
        for ln in kept:
            p = ln.split("|")
            if len(p) > 2 and p[1] == "switch" and _side(p[2]) is not None:
                before_switch[_side(p[2])].append({s: len(v) for s, v in seen.items()})
            if len(p) > 2 and p[1] in ("switch", "drag", "replace") and _side(p[2]) is not None:
                seen[_side(p[2])].add(p[2].split(":", 1)[1].strip())
            if len(p) > 2 and p[1] == "turn" and p[2].strip().isdigit():
                at_turn[int(p[2])] = {s: len(v) for s, v in seen.items()}
        meta[cid] = (r["id"], at_turn, before_switch)
        # with preview: every frame knows every previewed species of both teams (formats
        # without team preview — e.g. random battles in GM_AUDIT_POKEMON_REPLAYS — have no
        # |poke| lines: their full conversion is already the no-preview case below)
        n_poke = [sum(1 for ln in lines if ln.startswith(f"|poke|p{s + 1}|")) for s in (0, 1)]
        for me, side in enumerate(("p1", "p2")):
            s = full.get((r["id"], side))
            if s is not None and min(n_poke) > 0:
                m = s["obs"]["slot_mask"]
                assert int(m[0, :6].sum()) == min(6, n_poke[me]) and int(m[0, 6:].sum()) == min(
                    6, n_poke[1 - me]
                )
                assert bool((m == m[:1]).all()), (r["id"], side)
    got = _convert(gmdata, tmp_path, "nopreview", jobs)
    checked = forced = 0
    for cid, (rid, at_turn, before_switch) in meta.items():
        for me, side in enumerate(("p1", "p2")):
            s = got.get((cid, side))
            if s is None:
                continue
            # the side's switch-labelled frames answer its |switch| lines in order (the lead
            # line has no frame without team preview)
            sw_frames = [f for f in range(_n(s)) if int(s["tgt"]["action_type"][f]) == 1]
            assert len(sw_frames) == len(before_switch[me]) - 1, (cid, side)
            assert not any(_preview(s, f) for f in range(_n(s))), (cid, side)
            for f, t in enumerate(_turns(s)):
                if t >= TURN_CLIP:
                    continue
                n_opp = int(s["obs"]["slot_mask"][f, 6:].sum())
                lo = min(6, at_turn[t][1 - me])
                if _forced(s, f):
                    line = 1 + sw_frames.index(f) if f in sw_frames else None
                    hi = 6 if line is None else min(6, before_switch[me][line][1 - me])
                    assert lo <= n_opp <= hi, (cid, side, t, n_opp, lo, hi)
                    forced += 1
                else:
                    assert n_opp == lo, (cid, side, t, n_opp, at_turn[t])
                assert not bool(s["obs"]["poke_feats"][f, 6 + n_opp :].any()), (cid, side, t)
                checked += 1
    assert checked > 1000 and forced > 100, (checked, forced)


# ----------------------------------------------------------------------------------------------
# FUTURE leaks: switch / forced-switch labels
# ----------------------------------------------------------------------------------------------


def test_forced_switches_are_never_labelled_as_the_turns_choice(gmdata, replays, tmp_path):
    """A switch caused by something that happens during turn t (replacement after a KO, a pivot
    move, Eject Button) is not what the side chose at the turn-t frame: labelling it would train
    the pre-turn state on a post-KO decision.  It is its own decision: a forced-switch frame of
    turn t (obs flag, legal action types [switch only]) snapshotted when the request is issued
    — after the KO / the pivot hit — and labelled with the incoming Pokémon's previewed slot.
    The turn-t frame keeps the side's own choice (the move) or, when none was visible (KO'd
    before acting, ejected before acting), an unlabelled frame.  Positive control: a voluntary
    switch labels the turn-t frame with the previewed slot, and no forced frame appears."""
    jobs, meta = {}, {}
    for r in replays[:10]:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        preview = [
            ln.split("|")[3].split(",")[0].strip() for ln in lines if ln.startswith("|poke|p1|")
        ]
        for i, t in _turn_lines(lines)[1:6]:
            seen = {
                ln.split("|")[3].split(",")[0].strip()
                for ln in lines[:i]
                if ln.split("|")[1:2]
                and ln.split("|")[1] in ("switch", "drag", "replace")
                and _side(ln.split("|")[2]) == 0
            }
            fresh = [k for k, sp in enumerate(preview) if sp not in seen and "*" not in sp]
            if not fresh:
                continue
            slot = fresh[0]
            sp = preview[slot]
            cut = _synthetic_turn_cut(lines, i, p1).split("\n")[:-1]
            head, m1, m2 = cut[:-2], cut[-2], cut[-1]
            me, opp = m1.split("|")[2], m2.split("|")[2]
            new = f"|switch|p1a: {sp}|{sp}|100/100"
            nxt = [
                "|upkeep|",
                f"|turn|{t + 1}",
                f"|move|p1a: {sp}|Splash|p1a: {sp}",
                m2,
                f"|win|{p1}",
            ]
            ko = [f"|-damage|{me}|0 fnt", f"|faint|{me}"]
            # name: (turn body, the turn-t frame's label, forced frame?, its active fainted?)
            cases = {
                "ko_after_acting": ([m1, m2] + ko + nxt[:1] + [new] + nxt[1:], 0, True, True),
                "ko_before_acting": ([m2] + ko + nxt[:1] + [new] + nxt[1:], -1, True, True),
                "pivot": ([f"|move|{me}|U-turn|{opp}", new, m2] + nxt, 0, True, False),
                "eject_button": ([m2, f"|-enditem|{me}|Eject Button", new] + nxt, -1, True, False),
                "voluntary": ([new, m2] + nxt, 1, False, False),
            }
            for name, (body, want, forced, ko_seen) in cases.items():
                cid = f"{r['id']}~F{t}{name}"
                jobs[cid] = _rec(cid, "\n".join(head + body), r)
                meta[cid] = (t, want, forced, ko_seen, slot)
    assert len(meta) >= 5 * 20, len(meta)
    got = _convert(gmdata, tmp_path, "forced", jobs)
    for cid, (t, want, forced, ko_seen, slot) in meta.items():
        s = got.get((cid, "p1"))
        turns = _turns(s)
        at_t = [f for f, x in enumerate(turns) if x == t and not _forced(s, f)]
        types = [int(s["tgt"]["action_type"][f]) for f in at_t]
        assert types == [want], f"{cid}: the turn-{t} frame is labelled {types}, not {want}"
        if want == 1:
            assert int(s["tgt"]["switch_idx"][at_t[0]]) == slot, cid
        fs = [f for f, x in enumerate(turns) if x == t and _forced(s, f)]
        assert len(fs) == int(forced), (cid, fs)
        for f in fs:
            assert f > at_t[0], cid
            assert int(s["tgt"]["action_type"][f]) == 1 and int(s["tgt"]["switch_idx"][f]) == slot
            assert s["tgt"]["legal_action_type"][f].tolist() == [False, True], cid
            assert bool(s["tgt"]["legal_switch"][f, slot]), cid
            act = int(s["tgt"]["active_idx"][f])
            fainted = float(s["obs"]["poke_feats"][f, act, POKE_FEAT_FAINTED])
            assert fainted == float(ko_seen), f"{cid}: not the request-time state"


def _fresh(lines: List[str], i: int, side: int) -> List[str]:
    """The side's previewed species not switched in before line i (no hidden-form ``*``)."""
    tag = f"|poke|p{side + 1}|"
    preview = [ln.split("|")[3].split(",")[0].strip() for ln in lines if ln.startswith(tag)]
    seen = {
        ln.split("|")[3].split(",")[0].strip()
        for ln in lines[:i]
        if ln.split("|")[1:2]
        and ln.split("|")[1] in ("switch", "drag", "replace")
        and _side(ln.split("|")[2]) == side
    }
    return [sp for sp in preview if sp not in seen and "*" not in sp]


def test_simultaneous_replacements_do_not_see_each_other(gmdata, replays, tmp_path):
    """Both actives faint during turn t: Showdown asks both players for a replacement at once
    (after ``|upkeep|``), so neither decision may see the other's choice.  The order in which the
    two switch-ins are logged must not matter for any frame, and each forced frame shows both
    fainted actives still in place."""
    jobs, meta = {}, {}
    for r in replays[:8]:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        for i, t in _turn_lines(lines)[1:5]:
            f1, f2 = _fresh(lines, i, 0), _fresh(lines, i, 1)
            if t + 1 >= TURN_CLIP or not f1 or not f2:
                continue
            head, m1, m2, me, opp = _pieces(lines, i, p1)
            ko = [
                f"|move|{me}|Outrage|{opp}",
                f"|-damage|{opp}|0 fnt",
                f"|faint|{opp}",
                f"|-damage|{me}|0 fnt|[from] item: Rocky Helmet",
                f"|faint|{me}",
                "|upkeep|",
            ]
            s1 = f"|switch|p1a: {f1[0]}|{f1[0]}|100/100"
            s2 = f"|switch|p2a: {f2[0]}|{f2[0]}|100/100"
            tail = [
                f"|turn|{t + 1}",
                f"|move|p1a: {f1[0]}|Splash|p1a: {f1[0]}",
                f"|move|p2a: {f2[0]}|Splash|p2a: {f2[0]}",
                f"|win|{p1}",
            ]
            for order, sw in (("a", [s1, s2]), ("b", [s2, s1])):
                cid = f"{r['id']}~R{t}{order}"
                jobs[cid] = _rec(cid, "\n".join(head + ko + sw + tail), r)
            meta[(r["id"], t)] = (f"{r['id']}~R{t}a", f"{r['id']}~R{t}b")
    assert len(meta) >= 10, len(meta)
    got = _convert(gmdata, tmp_path, "simul", jobs)
    for (rid, t), (a, b) in meta.items():
        for side in ("p1", "p2"):
            sa, sb = got[(a, side)], got[(b, side)]
            assert _n(sa) == _n(sb), (rid, t, side)
            _assert_prefix(sa, sb, _n(sa), f"{rid} turn {t} {side}: the switch-in order leaks")
            fs = [f for f, x in enumerate(_turns(sa)) if x == t and _forced(sa, f)]
            assert len(fs) == 1 and int(sa["tgt"]["action_type"][fs[0]]) == 1, (rid, t, side, fs)
            feats = sa["obs"]["poke_feats"][fs[0]]
            for lo in (0, 6):  # own and opposing active: both still the fainted Pokémon
                act = feats[lo : lo + 6, POKE_FEAT_ACTIVE].nonzero().squeeze(1)
                assert act.numel() == 1 and float(feats[lo + int(act), POKE_FEAT_FAINTED]) == 1.0


def _forced_commits(lines: List[str]) -> List[Tuple[int, int]]:
    """[(commit line, last switch line)] of every forced-switch decision in a server log: a
    ``|t:|`` commit that is not the start of a turn / of the battle (Showdown's turnLoop logs
    ``|`` + ``|t:|`` when the answered request's actions start), followed by the ``|switch|``
    lines that answer it (up to the next commit / turn)."""
    out, commit, last = [], None, None
    prev = ""
    for i, ln in enumerate(lines):
        p = ln.split("|")
        cmd = p[1] if len(p) > 1 else ""
        if cmd in ("t:", "turn", "win", "tie"):
            if commit is not None and last is not None:
                out.append((commit, last))
            commit = i if cmd == "t:" and prev not in ("turn", "teampreview", "start", "") else None
            last = None
        elif cmd == "switch" and commit is not None:
            last = i
        if cmd not in ("", "t:", "j", "J", "l", "L", "c", "c:", "inactive", "inactiveoff", "raw"):
            prev = cmd
    return out


def test_forced_frames_ignore_post_commit_switch_out_effects(gmdata, replays, full, tmp_path):
    """A forced-switch request is answered at its decision commit (``|t:|``); the switch action
    that follows logs the outgoing Pokémon's switch-out effects — Natural Cure's cure, the end of
    Neutralizing Gas restarting a foe's Drizzle / Intimidate / terrain, a primal weather ending —
    BEFORE the ``|switch|`` line.  Injecting such effects right after the commit of a forced
    switch leaves every frame through that decision's forced frames unchanged: they are the state
    the request was answered in, not the state at the switch line."""
    jobs, meta = {}, {}
    for r in replays:
        lines = r["log"].split("\n")
        p1 = _names(lines).get(0)
        actives: Dict[int, str] = {}
        at_line: Dict[int, Dict[int, str]] = {}
        for i, ln in enumerate(lines):
            p = ln.split("|")
            if len(p) > 2 and p[1] in ("switch", "drag", "replace") and _side(p[2]) is not None:
                actives[_side(p[2])] = p[2].strip()
            at_line[i] = dict(actives)
        for n, (c, last) in enumerate(_forced_commits(lines)[:12]):
            act = at_line[c]
            inject = ["|-weather|RainDance", "|-fieldstart|move: Psychic Terrain"] + [
                x for s in sorted(act) for x in (
                    f"|-unboost|{act[s]}|spe|1",
                    f"|-curestatus|{act[s]}|par|[from] ability: Natural Cure|[silent]")]
            cid = f"{r['id']}~Q{n}"
            jobs[cid] = _rec(cid, "\n".join(lines[: c + 1] + inject + lines[c + 1 :]), r)
            jobs[cid + "c"] = _rec(cid + "c", "\n".join(lines[: last + 1] + [f"|win|{p1}"]), r)
            meta[cid] = r["id"]
    assert len(meta) > 100, len(meta)
    got = _convert(gmdata, tmp_path, "commit", jobs)
    checked = 0
    for cid, rid in meta.items():
        for side in ("p1", "p2"):
            whole, pert = full.get((rid, side)), got.get((cid, side))
            cut = got.get((cid + "c", side))
            assert _n(pert) == _n(whole), (cid, side)
            k = _n(cut)  # the side's frames through this decision (its forced frame last)
            _assert_prefix(whole, pert, k, f"{cid}/{side}: post-commit effects reach the request")
            checked += int(k > 0 and _forced(cut, k - 1))
    assert checked > 100, checked


# ----------------------------------------------------------------------------------------------
# VALUE: the hindsight outcome
# ----------------------------------------------------------------------------------------------


def test_value_target_never_reaches_obs_or_labels(gmdata, replays, full, tmp_path):
    """Flip the winner: obs and every non-value target are unchanged, only tgt.value flips."""
    jobs = {}
    for r in replays:
        lines = r["log"].split("\n")
        names = _names(lines)
        flip = {names[0]: names[1], names[1]: names[0]}
        out = [f"|win|{flip.get(ln[5:], ln[5:])}" if ln.startswith("|win|") else ln for ln in lines]
        jobs[f"{r['id']}~W"] = _rec(f"{r['id']}~W", "\n".join(out), r)
    got = _convert(gmdata, tmp_path, "win", jobs)
    for cid in jobs:
        rid = cid[:-2]
        for side in ("p1", "p2"):
            whole, pert = full.get((rid, side)), got.get((cid, side))
            assert _n(whole) == _n(pert), (cid, side)
            if whole is None:
                continue
            _assert_prefix(whole, pert, _n(whole), f"{cid}/{side}")
            assert torch.equal(pert["tgt"]["value"], -whole["tgt"]["value"]), (cid, side)


# ----------------------------------------------------------------------------------------------
# Frame existence is not conditioned on the outcome of the turn (converter v1 dropped the turn-t
# frame when the side's choice was not visible: |cant|, KO'd / flinched before moving — fixed in
# converter v2, which keeps it with a masked label)
# ----------------------------------------------------------------------------------------------


def test_frame_existence_does_not_depend_on_the_turns_outcome(gmdata, replays, tmp_path):
    """Two continuations of the same turn-t state — the side's Pokémon moves, or it cannot move
    (``|cant|...|flinch``) — must yield the same frames for turns ≤ t (a label mask can mark the
    unobserved choice; the frame and the temporal context must not disappear)."""
    jobs, meta = {}, {}
    for r in replays[:6]:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        for i, t in _turn_lines(lines)[2:8]:
            cut = _synthetic_turn_cut(lines, i, p1).split("\n")[:-1]  # drop the |win|
            mover = cut[-2]  # p1's synthetic Splash
            ident = mover.split("|")[2]
            for tag, act in (("move", mover), ("cant", f"|cant|{ident}|flinch")):
                cid = f"{r['id']}~X{t}{tag}"
                body = cut[:-2] + [
                    act,
                    cut[-1],
                    "|upkeep|",
                    f"|turn|{t + 1}",
                    mover,
                    cut[-1],
                    f"|win|{p1}",
                ]
                jobs[cid] = _rec(cid, "\n".join(body), r)
            meta[(r["id"], t)] = (f"{r['id']}~X{t}move", f"{r['id']}~X{t}cant")
    got = _convert(gmdata, tmp_path, "cant", jobs)
    assert meta
    for (rid, t), (a, b) in meta.items():
        sa, sb = got.get((a, "p1")), got.get((b, "p1"))
        na, nb = _n_upto(sa, t), _n_upto(sb, t)
        assert na == nb, (
            f"{rid} turn {t}: {na} frames up to turn {t} if p1 moves, {nb} if it cannot"
        )
        _assert_prefix(sa, sb, na, f"{rid} turn {t}", tgt=False)
        # the turn-t frame differs in its label only: the Splash vs a masked label
        k = na - 1
        assert _labelled(sa, k) and not _labelled(sb, k), (rid, t)
        assert not _state_diff(sa, k, sb, k), (rid, t)


# ----------------------------------------------------------------------------------------------
# MODEL: causality of the pokemon network over frames
# ----------------------------------------------------------------------------------------------


def test_pokemon_network_is_causal_over_frames():
    """Perturbing frames > t of obs, the move_ids (LEGAL_FROM_OBS) and the teacher-forcing /
    state labels (action_type, active_idx) leaves every output at frames ≤ t unchanged."""
    from model.game_network import GameNetwork
    from model.games.pokemon import PokemonConfig, PokemonDecoder, PokemonEncoder
    from training.train_il_game import LEGAL_FROM_OBS

    torch.manual_seed(0)
    cfg = PokemonConfig(
        d_model=48,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        n_st_layers=2,
        core_type="trm",
        trm_H_cycles=2,
        trm_L_cycles=2,
        trm_block_layers=1,
        halt_max_steps=2,
        pokemon_species_buckets=64,
        pokemon_move_buckets=32,
    )
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg)).eval()
    B, T, S, cut = 2, 6, 12, 3

    def make(seed):
        g = torch.Generator().manual_seed(seed)
        obs = {
            "species_ids": torch.randint(1, 64, (B, T, S), generator=g),
            "move_ids": torch.randint(0, 32, (B, T, 4), generator=g),
            "poke_feats": torch.randn(B, T, S, cfg.pokemon_feature_dim, generator=g),
            "slot_mask": torch.rand(B, T, S, generator=g) < 0.8,
            "field_seq": torch.randn(B, T, cfg.pokemon_field_features, generator=g),
        }
        labels = {
            "action_type": torch.randint(0, 2, (B, T), generator=g),
            "active_idx": torch.randint(0, 6, (B, T), generator=g),
        }
        return obs, labels

    obs, labels = make(1)
    obs2, labels2 = make(2)
    for d, d2 in ((obs, obs2), (labels, labels2)):
        for k in d:
            d2[k][:, :cut] = d[k][:, :cut]
    with torch.no_grad():
        a = net(obs, legal=LEGAL_FROM_OBS["pokemon"](obs), labels=labels, act_steps=2)
        b = net(obs2, legal=LEGAL_FROM_OBS["pokemon"](obs2), labels=labels2, act_steps=2)
    checked = 0
    for k, v in a.items():
        if not torch.is_tensor(v) or v.dim() < 2 or v.shape[:2] != (B, T):
            continue
        torch.testing.assert_close(
            v[:, :cut].float(),
            b[k][:, :cut].float(),
            rtol=0,
            atol=1e-6,
            msg=f"{k} sees future frames",
        )
        checked += 1
    assert checked >= 4, list(a)


# ----------------------------------------------------------------------------------------------
# Independent verification (second pass): refutations pinned as regressions; the confirmed
# deploy mismatches (strict xfails against converter v1) are fixed by converter v2 and pinned as
# regressions too
# ----------------------------------------------------------------------------------------------


def _pieces(lines: List[str], i: int, p1: str):
    """(log head through the ``|turn|t`` line i, p1 Splash line, p2 Splash line, p1 ident,
    p2 ident)."""
    cut = _synthetic_turn_cut(lines, i, p1).split("\n")[:-1]
    head, m1, m2 = cut[:-2], cut[-2], cut[-1]
    return head, m1, m2, m1.split("|")[2], m2.split("|")[2]


def _labelled(s: Dict, f: int) -> bool:
    """Frame f carries a supervised action label (−1 / ``action_known`` 0 mean unlabelled)."""
    if int(s["tgt"]["action_type"][f]) < 0:
        return False
    known = s["tgt"].get("action_known")
    return known is None or bool(known[f])


def _fresh_p1(lines: List[str], i: int) -> List[str]:
    """p1's previewed species not switched in before line i (no hidden-form ``*`` species)."""
    preview = [ln.split("|")[3].split(",")[0].strip() for ln in lines if ln.startswith("|poke|p1|")]
    seen = {
        ln.split("|")[3].split(",")[0].strip()
        for ln in lines[:i]
        if ln.split("|")[1:2]
        and ln.split("|")[1] in ("switch", "drag", "replace")
        and _side(ln.split("|")[2]) == 0
    }
    return [sp for sp in preview if sp not in seen and "*" not in sp]


def test_locked_and_recharge_turns_are_never_labelled(gmdata, replays, tmp_path):
    """Turns on which the player has no choice carry no action label: the 2nd turn of Outrage
    (``[from] lockedmove``), the release turn of a two-turn move (charge turn ``[still]`` +
    ``-prepare`` is a real choice, the release turn is ``[from] lockedmove``), a Hyper Beam
    recharge (``|cant|..|recharge``).  The turns around them (real choices) are labelled."""
    jobs, meta = {}, {}
    for r in replays[:8]:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        for i, t in _turn_lines(lines)[2:6]:
            if t + 2 >= TURN_CLIP:
                continue
            head, m1, m2, me, opp = _pieces(lines, i, p1)
            up1 = ["|upkeep|", f"|turn|{t + 1}"]
            tail = ["|upkeep|", f"|turn|{t + 2}", m1, m2, f"|win|{p1}"]
            charge = [f"|move|{me}|Phantom Force||[still]", f"|-prepare|{me}|Phantom Force"]
            cases = {
                "outrage": [f"|move|{me}|Outrage|{opp}", m2]
                + up1
                + [f"|move|{me}|Outrage|{opp}|[from] lockedmove", m2],
                "twoturn": charge
                + [m2]
                + up1
                + [f"|move|{me}|Phantom Force|{opp}|[from] lockedmove", m2],
                "recharge": [f"|move|{me}|Hyper Beam|{opp}", f"|-mustrecharge|{me}", m2]
                + up1
                + [f"|cant|{me}|recharge", m2],
            }
            for name, body in cases.items():
                cid = f"{r['id']}~K{t}{name}"
                jobs[cid] = _rec(cid, "\n".join(head + body + tail), r)
                meta[cid] = t
    assert len(meta) >= 3 * 20, len(meta)
    got = _convert(gmdata, tmp_path, "locked", jobs)
    for cid, t in meta.items():
        s = got.get((cid, "p1"))
        turns = _turns(s)
        lab = {x: [_labelled(s, f) for f, y in enumerate(turns) if y == x] for x in range(t, t + 3)}
        assert lab[t] == [True] and lab[t + 2] == [True], (cid, lab)
        assert not any(lab[t + 1]), f"{cid}: a no-choice turn is labelled as a decision ({lab})"


def test_natural_cure_on_switch_out_clears_the_benched_status(gmdata, replays, tmp_path):
    """Natural Cure is public in singles (``|-curestatus|..|[from] ability: Natural Cure|[silent]``
    right before the ``|switch|`` line): the benched mon's status is cleared in both sides' frames,
    so the spectator view keeps no stale status (unlike a Regenerator heal, which is not logged)."""
    jobs, meta = {}, {}
    for r in replays[:8]:
        lines = r["log"].split("\n")
        p1 = _names(lines)[0]
        for i, t in _turn_lines(lines)[1:5]:
            fresh = _fresh_p1(lines, i)
            if t + 2 >= TURN_CLIP or not fresh:
                continue
            head, m1, m2, me, opp = _pieces(lines, i, p1)
            sp = fresh[0]
            body = [
                f"|-status|{me}|brn",
                m1,
                m2,
                "|upkeep|",
                f"|turn|{t + 1}",
                f"|-curestatus|{me}|brn|[from] ability: Natural Cure|[silent]",
                f"|switch|p1a: {sp}|{sp}|100/100",
                m2,
                "|upkeep|",
                f"|turn|{t + 2}",
                f"|move|p1a: {sp}|Splash|p1a: {sp}",
                m2,
                f"|win|{p1}",
            ]
            cid = f"{r['id']}~N{t}"
            jobs[cid] = _rec(cid, "\n".join(head + body), r)
            meta[cid] = t
    assert len(meta) >= 10, len(meta)
    got = _convert(gmdata, tmp_path, "natcure", jobs)
    brn = 4  # POKE_FEAT_STATUS + 1
    for cid, t in meta.items():
        s1, s2 = got[(cid, "p1")], got[(cid, "p2")]
        f1 = {x: f for f, x in enumerate(_turns(s1))}
        f2 = {x: f for f, x in enumerate(_turns(s2))}
        slot = int(s1["tgt"]["active_idx"][f1[t + 1]])
        assert s1["obs"]["poke_feats"][f1[t + 1], slot, brn] == 1, cid
        assert s1["obs"]["poke_feats"][f1[t + 2], slot, brn] == 0, f"{cid}: stale own status"
        assert s2["obs"]["poke_feats"][f2[t + 1], 6 + slot, brn] == 1, cid
        assert s2["obs"]["poke_feats"][f2[t + 2], 6 + slot, brn] == 0, f"{cid}: stale opp status"


_HP_AT = {"switch": 4, "drag": 4, "replace": 4, "-damage": 3, "-heal": 3, "-sethp": 3}


def _omniscient(lines: List[str], salt: int) -> List[str]:
    """Rewrite a spectator log as an omniscient (``|split|``) log, the form a local Showdown server
    or a self-play logger writes: every HP-carrying line becomes ``|split|pN``, the private exact-HP
    copy (``E/M``, M = max HP), then the public percentage line (Showdown's addSplit order)."""
    out = []
    for ln in lines:
        p = ln.split("|")
        k = _HP_AT.get(p[1] if len(p) > 1 else "")
        s = _side(p[2]) if k is not None and len(p) > k else None
        m = re.match(r"^(\d+)/100(.*)$", p[k].strip()) if s is not None else None
        if m is None:
            out.append(ln)
            continue
        mx = 250 + 37 * salt + 11 * s
        q = list(p)
        q[k] = f"{(int(m.group(1)) * mx + 99 * salt) // 100}/{mx}{m.group(2)}"
        out += [f"|split|p{s + 1}", "|".join(q), ln]
    return out


def test_private_exact_hp_of_an_omniscient_log_never_reaches_obs(gmdata, replays, full, tmp_path):
    """PRIVACY on omniscient logs: the per-player secret half of a ``|split|`` block (exact HP of
    either side, never shown to the opponent) must not change any obs or target; only the public
    percentage line may.  Two different secret values give the frames of the spectator log."""
    jobs = {}
    for r in replays:
        for salt in (1, 2):
            cid = f"{r['id']}~H{salt}"
            jobs[cid] = _rec(cid, "\n".join(_omniscient(r["log"].split("\n"), salt)), r)
    got = _convert(gmdata, tmp_path, "split", jobs)
    for cid in jobs:
        rid = cid.rsplit("~", 1)[0]
        for side in ("p1", "p2"):
            whole, pert = full.get((rid, side)), got.get((cid, side))
            assert _n(whole) == _n(pert), (cid, side)
            _assert_prefix(whole, pert, _n(whole), f"{cid}/{side}")


def test_every_move_label_names_a_move_in_obs(full):
    """A move label must identify WHICH move was used from what the policy sees (a live player
    sees all 4 of its moves in the request): the labelled slot holds a real move id.  (Converter
    v1 pointed about half of the labels at an unrevealed id-0 slot — every first use collapsed
    onto the first empty slot.  v2: the target is the move identity tgt.move_id over the move
    vocabulary; the legacy slot label tgt.move_idx exists only for moves revealed before the
    frame, and then names the same move.)"""
    bad = tot = n_move = 0
    for s in full.values():
        at, mi, mid = s["tgt"]["action_type"], s["tgt"]["move_idx"], s["tgt"]["move_id"]
        mv = (at == 0) & (mi >= 0)
        idx = mv.nonzero().squeeze(1)
        ids = s["obs"]["move_ids"][idx, mi[idx]]
        bad += int((ids == 0).sum())
        tot += int(idx.numel())
        assert torch.equal(ids, mid[idx]), "the slot label and the identity name different moves"
        assert bool((mid[at == 0] >= 1).all()) and bool((mid[at != 0] == -1).all())
        assert bool((mi[at != 0] == -1).all())
        n_move += int((at == 0).sum())
    assert tot > 500, tot
    assert bad == 0, f"{bad}/{tot} move labels point at an unrevealed slot"
    assert n_move > tot + 300, (n_move, tot)  # first uses: identity only


def _switch_lines(log: str) -> Dict[int, List[Tuple[int, str]]]:
    """side → [(turn, species)] of every ``|switch|`` line of that side.  Each one is a choice of
    that side's player: the lead, a voluntary switch, a KO replacement, a pivot or an eject."""
    out: Dict[int, List[Tuple[int, str]]] = {0: [], 1: []}
    turn = 0
    for ln in log.split("\n"):
        p = ln.split("|")
        if len(p) > 2 and p[1] == "turn" and p[2].strip().isdigit():
            turn = int(p[2])
        elif len(p) > 3 and p[1] == "switch" and _side(p[2]) is not None:
            out[_side(p[2])].append((turn, p[3].split(",")[0].strip()))
    return out


def test_every_in_battle_switch_decision_has_a_frame(replays, full):
    """Every in-battle ``|switch|`` of a side answers a request its live bot must also answer (a
    move/switch request or a forceSwitch request): one switch-labelled frame per such line.
    (Converter v1 gave KO replacements and pivot / eject switches — U-turn, Volt Switch, Flip
    Turn, Parting Shot, Chilly Reception, Teleport, Eject Button/Pack — no frame, so the
    forceSwitch state never occurred in training.)"""
    short = []
    for r in replays:
        sw = _switch_lines(r["log"])
        for me, side in enumerate(("p1", "p2")):
            s = full.get((r["id"], side))
            if s is None:
                continue
            want = sum(1 for t, _ in sw[me] if t >= 1)
            have = sum(
                1
                for f, t in enumerate(_turns(s))
                if t >= 1 and int(s["tgt"]["action_type"][f]) == 1
            )
            if have != want:
                short.append((r["id"], side, have, want))
    assert not short, f"{len(short)} sides miss switch decisions, e.g. {short[:3]}"


def _preview_slot(preview: List[str], species: str) -> Optional[int]:
    for k, sp in enumerate(preview):
        if sp == species:
            return k
    for k, sp in enumerate(preview):  # hidden forms at preview: "Urshifu-*"
        if sp.endswith("-*") and species.startswith(sp[:-2]):
            return k
    return None


def test_the_lead_choice_has_a_frame(replays, full):
    """The lead is chosen at team preview: each side has a turn-0 frame labelled with a switch to
    the lead's team-preview slot (converter v1 never labelled it), flagged as the team-preview
    request, with no active Pokémon on either side (both leads are chosen simultaneously)."""
    checked = 0
    for r in replays:
        lines = r["log"].split("\n")
        sw = _switch_lines(r["log"])
        for me, side in enumerate(("p1", "p2")):
            s = full.get((r["id"], side))
            preview = [
                ln.split("|")[3].split(",")[0].strip()
                for ln in lines
                if ln.startswith(f"|poke|p{me + 1}|")
            ]
            if s is None or not preview or not sw[me] or sw[me][0][0] != 0:
                continue
            slot = _preview_slot(preview, sw[me][0][1])
            f0 = [f for f, t in enumerate(_turns(s)) if t == 0]
            assert f0, f"{r['id']}/{side}: no turn-0 (lead) frame"
            assert int(s["tgt"]["action_type"][f0[0]]) == 1, (r["id"], side)
            assert int(s["tgt"]["switch_idx"][f0[0]]) == slot, (r["id"], side, preview)
            assert f0[0] == 0 and _preview(s, 0) and not _forced(s, 0), (r["id"], side)
            assert s["tgt"]["legal_action_type"][0].tolist() == [False, True], (r["id"], side)
            assert not bool(s["obs"]["poke_feats"][0, :, POKE_FEAT_ACTIVE].any()), (r["id"], side)
            checked += 1
    assert checked, "no replay with team preview and a turn-0 lead"


def test_court_change_swaps_the_side_conditions(gmdata, replays, tmp_path):
    """Court Change swaps every side condition: after p2's Stealth Rock lands on p1's side and p1
    uses Court Change, p1's frame shows the rock on the opponent's side and p2's on its own
    (converter v1 ignored ``|-swapsideconditions|``)."""
    jobs, meta = {}, {}
    for r in replays[:6]:
        lines = r["log"].split("\n")
        names = _names(lines)
        for i, t in _turn_lines(lines)[1:4]:
            if t + 2 >= TURN_CLIP:
                continue
            head, m1, m2, me, opp = _pieces(lines, i, names[0])
            body = [
                f"|move|{opp}|Stealth Rock|{me}",
                f"|-sidestart|p1: {names[0]}|move: Stealth Rock",
                m1,
                "|upkeep|",
                f"|turn|{t + 1}",
                f"|move|{me}|Court Change|{me}",
                "|-swapsideconditions",
                f"|-activate|{me}|move: Court Change",
                m2,
                "|upkeep|",
                f"|turn|{t + 2}",
                m1,
                m2,
                f"|win|{names[0]}",
            ]
            cid = f"{r['id']}~C{t}"
            jobs[cid] = _rec(cid, "\n".join(head + body), r)
            meta[cid] = t
    assert meta
    got = _convert(gmdata, tmp_path, "court", jobs)
    sr_own, sr_opp = 12, 17  # obs.field_seq stealth-rock flags (own side / opponent side)
    for cid, t in meta.items():
        for side, want in (("p1", (0.0, 1.0)), ("p2", (1.0, 0.0))):
            s = got[(cid, side)]
            fs = s["obs"]["field_seq"][_turns(s).index(t + 2)]
            assert (float(fs[sr_own]), float(fs[sr_opp])) == want, (cid, side, fs[10:20].tolist())
