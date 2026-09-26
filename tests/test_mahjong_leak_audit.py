"""Information-leak audit of `gmdata mahjong convert` (riichi mahjong: 4 seats, hidden hands and
wall).

A sample is one seat's decision frames over a game: obs.tiles_seq / obs.players_seq /
obs.table_seq, plus what the decoder reads at decision time (the tgt.legal_* masks, the
teacher-forced action_type) and the frame stream itself (Stage-1 temporal state).  All of it must
be information that seat really has at the decision:

  causality  A frame (its obs, its legal masks, and whether it exists at all) is a function of the
             mjai log up to its DECISION POINT, the event that prompts the seat: its tsumo, its
             chi / pon (forced discard), another seat's dahai (claim).  The seat's response is the
             label (targets may use hindsight), but nothing after the decision point may reach the
             frame.  Test: convert the log cut right after the decision point (own turns: plus the
             seat's own response events, which only set the label) and require its frames to be a
             bit-exact prefix of the full conversion; a claim frame must exist with or without the
             claim response.
  privacy    Seat p's sample is invariant to what only the other seats know: their concealed tiles
             (initial hands and draws, reassigned consistently with every public event) and the
             ura-dora.

The audit found three leaks in converter_version 2, fixed in version 3 and pinned here:
  * test_ron_frames_exist_regardless_of_the_response: a claim frame existed for a seat that was not
    pon/chi-eligible (in riichi, houtei, or no pair / run partner) only when it ronned (v3: claim
    frames exist by exact legality, ron included).
  * test_riichi_declaration_claims_precede_the_stick: claim frames on a riichi declaration tile
    showed the stick deposited (declarer -1000, kyotaku +1), which happens only if nobody rons.
  * test_minkan_dora_is_revealed_after_the_rinshan_decision: the rinshan-draw frame after a
    daiminkan / kakan showed that kan's new dora, which Tenhou flips only with the following
    discard (test_minkan_dora_follows_the_rinshan_response pins that from the logs).
  * test_declaration_tile_claims_do_not_reveal_a_ron / test_rinshan_frame_does_not_reveal_the_response
    re-check the last two by comparing two futures of one decision point (the real response and a
    counterfactual one) instead of a cut log.
test_legal_masks_never_fall_back_to_all_true guards the label-dependent all-true mask fallback.

A discard decision point includes a dora event the log writes right after the dahai: Tenhou flips an
open kan's new dora together with the following discard, and one spelling of the Houou corpus (the
akochan-reviewer converter's) logs it just after that dahai, the other just before it.

Data: tests/fixtures/mahjong (the golden set) and tests/fixtures/mahjong_audit, seven Tenhou Houou
games of the same public HF mirror (hhim8826/tenhou-houou-mjai) chosen for rinshan kaihou after a
kakan / daiminkan, rons on a riichi declaration tile, a pending open-kan dora flipped by a second
kan, legal rons declined by a riichi and a dama seat, and a triple ron (三家和, logged as a bare
ryukyoku after the discard).  GM_MAHJONG_AUDIT_RAW=<directory of
.mjson> repeats every test on more games (skipped when unset; GM_MAHJONG_AUDIT_MAX_GAMES /
_SPECIAL / _PLAIN size the sweep).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
import subprocess
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from data_processing.games.dataset import load_gm_samples
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture)

OBS_KEYS = ("tiles_seq", "players_seq", "table_seq")
LABEL_KEYS = ("action_type", "tile_idx")
COLUMNS = {
    "tiles_seq": ("hand=0", "hand=1", "hand=2", "hand=3", "hand=4", "own_melds", "opp_melds",
                  "disc_self", "disc_shimo", "disc_toimen", "disc_kami", "dora", "visible",
                  "yaochu", "red", "focus"),
    "players_seq": ("score", "riichi", "wind_E", "wind_S", "wind_W", "wind_N", "oya", "melds",
                    "discards", "kans", "focus_actor", "bias"),
    "table_seq": ("bakaze_E", "bakaze_S", "bakaze_W", "bakaze_N", "kyoku", "honba", "kyotaku",
                  "wall", "dora_count", "own_turn", "riichi_count", "bias"),
}
OWN_TURN = COLUMNS["table_seq"].index("own_turn")
TILES = [f"{d}{s}" for s in "mps" for d in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
HONORS = TILES[27:]
MINKAN = ("daiminkan", "kakan")
NEVER = float("inf")

EXTRA_ENV = "GM_MAHJONG_AUDIT_RAW"
MAX_EXTRA_GAMES = int(os.environ.get("GM_MAHJONG_AUDIT_MAX_GAMES", "100"))
# cut logs per test: every "special" decision point (a response other than a plain discard /
# pass) up to this cap, plus a seeded sample of plain ones
MAX_SPECIAL = int(os.environ.get("GM_MAHJONG_AUDIT_SPECIAL", "1200"))
N_PLAIN = int(os.environ.get("GM_MAHJONG_AUDIT_PLAIN", "300"))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _read(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module", params=["fixtures", "extra"])
def games(request) -> Dict[str, List[dict]]:
    """{game name: mjai events} of the committed fixtures, or of $GM_MAHJONG_AUDIT_RAW."""
    if request.param == "fixtures":
        paths = sorted(glob.glob(os.path.join(FIX, "mahjong", "*.mjson")))
        paths += sorted(glob.glob(os.path.join(FIX, "mahjong_audit", "*.mjson")))
    else:
        root = os.environ.get(EXTRA_ENV)
        if not root:
            pytest.skip(f"set {EXTRA_ENV}=<dir of .mjson> for the wide sweep")
        paths = sorted(glob.glob(os.path.join(root, "*.mjson")))[:MAX_EXTRA_GAMES]
    if not paths:
        pytest.skip("no mahjong games found")
    return {os.path.basename(p)[: -len(".mjson")]: _read(p) for p in paths}


def _kyokus(evs: List[dict]) -> List[List[dict]]:
    """The game split at start_kyoku.  Each piece is a self-contained log (start_kyoku carries the
    scores / honba / kyotaku; test_sample_stream_concatenates_the_kyoku_streams pins that)."""
    starts = [i for i, e in enumerate(evs) if e["type"] == "start_kyoku"]
    return [evs[a:b] for a, b in zip(starts, starts[1:] + [len(evs)])]


def _convert(gmdata, records: Dict[str, List[dict]], tmp) -> Dict[str, Dict[int, dict]]:  # noqa: F811
    """Convert every record (a list of mjai events) in one gmdata call → {key: {seat: sample}}."""
    src, out = tmp / "in", tmp / "out"
    src.mkdir(parents=True)
    for key, evs in records.items():
        text = "\n".join(json.dumps(e, separators=(",", ":")) for e in evs) + "\n"
        (src / f"{key}.mjson").write_text(text)
    r = subprocess.run([gmdata, "mahjong", "convert", "--input", str(src), "--out", str(out),
                        "--min-frames", "1", "--shard-size", "1000000"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    stats = json.loads(r.stdout)
    assert stats["skipped"] == 0, f"converter skipped {stats['skipped']} of {len(records)} records"
    got: Dict[str, Dict[int, dict]] = {k: {} for k in records}
    for s in load_gm_samples(str(out)):
        got[s["meta"]["name"][: -len(".mjson")]][int(s["meta"]["seat"])] = s
    return got


# ---------------------------------------------------------------------------
# decision points
# ---------------------------------------------------------------------------

def _decision_points(ky: List[dict]):
    """(kind, index) of the events that prompt a seat: 'draw' (tsumo), 'call' (chi / pon followed
    by the caller's forced dahai), 'discard' (dahai → the other seats' claim decisions)."""
    for i, e in enumerate(ky):
        t = e["type"]
        if t == "tsumo":
            yield "draw", i
        elif t in ("chi", "pon") and i + 1 < len(ky) and ky[i + 1]["type"] == "dahai" \
                and ky[i + 1]["actor"] == e["actor"]:
            yield "call", i
        elif t == "dahai":
            yield "discard", i


def _own_response(ky: List[dict], i: int) -> Optional[List[dict]]:
    """The drawing seat's own response to its tsumo at ``i`` (what sets the label), skipping the
    dora events the log interleaves: [reach, dahai] | [dahai] | [ankan] | [kakan] | [hora] |
    [ryukyoku]."""
    j = i + 1
    while j < len(ky) and ky[j]["type"] == "dora":
        j += 1
    if j >= len(ky):
        return None
    e = ky[j]
    if e["type"] == "reach":
        return ky[j:j + 2]
    if e["type"] in ("dahai", "ankan", "kakan", "hora", "ryukyoku"):
        return [e]
    return None


def _claim_response(ky: List[dict], i: int) -> Tuple[int, bool, bool]:
    """(end, ron, called) of the response to the dahai at ``i``: the bookkeeping the log puts after
    the discard (reach_accepted, dora), then every ron on it, or else one chi / pon / daiminkan of
    it.  ``ky[:end]`` holds the whole response and nothing of the caller's next decision."""
    a = ky[i]["actor"]
    j = i + 1
    while j < len(ky) and ky[j]["type"] in ("reach_accepted", "dora"):
        j += 1
    k = j
    while j < len(ky) and ky[j]["type"] == "hora" and ky[j].get("target") == a \
            and ky[j]["actor"] != a:
        j += 1
    ron = j > k
    called = False
    if not ron and j < len(ky) and ky[j]["type"] in ("chi", "pon", "daiminkan") \
            and ky[j].get("target") == a:
        j += 1
        called = True
    return j, ron, called


def _is_riichi_declaration(ky: List[dict], i: int) -> bool:
    return i > 0 and ky[i - 1]["type"] == "reach" and ky[i - 1]["actor"] == ky[i]["actor"]


def _is_minkan_rinshan_with_dora(ky: List[dict], i: int) -> bool:
    """A tsumo directly followed by a dora event: the rinshan draw after a daiminkan / kakan (the
    log flips that kan's dora right before the drawer's discard / ankan)."""
    return ky[i]["type"] == "tsumo" and i + 1 < len(ky) and ky[i + 1]["type"] == "dora"


def _discard_end(ky: List[dict], i: int) -> int:
    """End of the discard decision point at ``i``: the dahai plus the dora events the log writes
    right after it (an open kan's dora, which Tenhou flips together with this discard; the other
    log spelling writes it right before the dahai)."""
    j = i + 1
    while j < len(ky) and ky[j]["type"] == "dora":
        j += 1
    return j


# ---------------------------------------------------------------------------
# frame comparison
# ---------------------------------------------------------------------------

def _n(sample: Optional[dict]) -> int:
    return 0 if sample is None else int(sample["tgt"]["action_type"].shape[0])


def _first_diff(a: torch.Tensor, b: torch.Tensor, key: str) -> str:
    bad = (a != b).reshape(a.shape[0], -1).any(1).nonzero()[0].item()
    cols = COLUMNS.get(key)
    where = ""
    if cols is not None:
        d = (a[bad] != b[bad]).reshape(-1, len(cols)).any(0)
        where = " [" + ", ".join(c for c, x in zip(cols, d.tolist()) if x) + "]"
    return f"{key} differs at frame {bad}{where}"


def _prefix_diff(part: Optional[dict], full: Optional[dict],
                 free_last: bool = False) -> Optional[str]:
    """Why ``part`` (one seat's frames from a cut log) is not a bit-exact prefix of ``full`` (the
    same seat from the whole log), or None.  obs and legal_action_type must match on every frame;
    the labels and legal_tile_of (the tile row of the labelled type) too, except on the last frame
    when ``free_last`` (a claim decision whose response was cut off)."""
    n = _n(part)
    if n == 0:
        return None
    if n > _n(full):
        return f"{n} frames from the cut log, {_n(full)} from the whole log"
    obs = ["obs." + _first_diff(part["obs"][k], full["obs"][k][:n], k) for k in OBS_KEYS
           if not torch.equal(part["obs"][k], full["obs"][k][:n])]
    if obs:
        return "; ".join(obs)
    a, b = part["tgt"]["legal_action_type"], full["tgt"]["legal_action_type"][:n]
    if not torch.equal(a, b):
        return "tgt." + _first_diff(a, b, "legal_action_type")
    m = n
    if free_last and not all(torch.equal(part["tgt"][k][-1], full["tgt"][k][n - 1])
                             for k in LABEL_KEYS):
        m = n - 1               # the cut removed the response: the last label is not comparable
    for key in LABEL_KEYS + ("legal_tile_of",):
        a, b = part["tgt"][key][:m], full["tgt"][key][:m]
        if not torch.equal(a, b):
            return "tgt." + _first_diff(a, b, key)
    return None


def _budget(special: list, plain: list, seed: int = 0) -> list:
    rng = random.Random(seed)
    if len(special) > MAX_SPECIAL:
        special = rng.sample(special, MAX_SPECIAL)
    return special + rng.sample(plain, min(N_PLAIN, len(plain)))


def _fail(problems: List[str], what: str) -> None:
    assert not problems, f"{len(problems)} {what}; first ones:\n  " + "\n  ".join(problems[:12])


def _kind(p: str) -> int:
    if p in HONORS:
        return 27 + HONORS.index(p)
    d = int(p[0])
    return "mps".index(p[1]) * 9 + (4 if d == 0 else d - 1)


# ---------------------------------------------------------------------------
# the stream
# ---------------------------------------------------------------------------

def test_sample_stream_concatenates_the_kyoku_streams(gmdata, games, tmp_path):  # noqa: F811
    """A seat's sample is the concatenation of its per-kyoku streams: no state crosses a
    start_kyoku except what that event announces (scores, honba, kyotaku — public).  So cutting
    logs inside one kyoku (the causality tests below) is representative of the full stream."""
    records = dict(games)
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            records[f"{g}~k{k}"] = ky
    conv = _convert(gmdata, records, tmp_path)
    keys = [("obs", k) for k in OBS_KEYS]
    keys += [("tgt", k) for k in LABEL_KEYS + ("legal_action_type", "legal_tile_of")]
    for g, evs in games.items():
        full = conv[g]
        assert sorted(full) == [0, 1, 2, 3], g
        for s in range(4):
            parts = [conv[f"{g}~k{k}"].get(s) for k in range(len(_kyokus(evs)))]
            parts = [p for p in parts if p is not None]
            for group, key in keys:
                cat = torch.cat([p[group][key] for p in parts])
                assert torch.equal(cat, full[s][group][key]), (g, s, key)
            v = full[s]["tgt"]["value"]
            assert torch.all(v == v[0]), (g, s)


def test_each_sample_is_one_seats_view(gmdata, games, tmp_path):  # noqa: F811
    """One sample per seat, holding that seat's view: its own concealed hand and seat wind on
    row 0, and exactly one own-turn frame per own tsumo and per own chi / pon (own-turn frames are
    never selected by what the seat did)."""
    conv = _convert(gmdata, games, tmp_path)
    for g, evs in games.items():
        kys = _kyokus(evs)
        for s in range(4):
            smp = conv[g][s]
            own = smp["obs"]["table_seq"][:, OWN_TURN] == 1
            n_own = sum(1 for ky in kys for kind, i in _decision_points(ky)
                        if kind in ("draw", "call") and ky[i]["actor"] == s)
            assert int(own.sum()) == n_own, (g, s)
            # frame 0 lies at or before this seat's first draw (a claim frame may come first,
            # possibly in an earlier kyoku that ended before the seat drew): its tehai, + that
            # draw on an own-turn frame
            k0 = next(k for k, ky in enumerate(kys)
                      if any(e["type"] == "tsumo" and e["actor"] == s for e in ky))
            hands, winds = [], []
            for k in ([k0] if own[0] else range(k0 + 1)):
                want = [0] * 34
                for p in kys[k][0]["tehais"][s]:
                    want[_kind(p)] += 1
                if own[0]:
                    draw = next(e for e in kys[k] if e["type"] == "tsumo" and e["actor"] == s)
                    want[_kind(draw["pai"])] += 1
                hands.append(want)
                winds.append((s - kys[k][0]["oya"]) % 4)
            hand = smp["obs"]["tiles_seq"][0, :, :5].argmax(-1).tolist()
            assert hand in hands, (g, s)
            wind = smp["obs"]["players_seq"][0, 0, 2:6].argmax().item()
            assert wind == winds[hands.index(hand)], (g, s)


# ---------------------------------------------------------------------------
# causality
# ---------------------------------------------------------------------------

def _causal_cases(games: Dict[str, List[dict]]):
    """Whole-kyoku records + (kyoku key, cut key, cut log, seats checked, seats whose last label
    may be cut off) per decision point."""
    records: Dict[str, List[dict]] = {}
    special, plain = [], []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            base = f"{g}~k{k}"
            records[base] = ky
            for kind, i in _decision_points(ky):
                actor = ky[i]["actor"]
                key = f"{base}~{kind}{i}"
                if kind == "draw":
                    resp = _own_response(ky, i)
                    if resp is None:
                        continue
                    case = (base, key, ky[:i + 1] + resp, [actor], ())
                    (plain if resp[0]["type"] == "dahai" else special).append(case)
                elif kind == "call":
                    special.append((base, key, ky[:i + 2], [actor], ()))
                else:
                    others = tuple((actor + r) % 4 for r in (1, 2, 3))
                    case = (base, key, ky[:_discard_end(ky, i)], [actor, *others], others)
                    _, ron, called = _claim_response(ky, i)
                    (special if ron or called else plain).append(case)
    return records, _budget(special, plain)


def test_frames_are_causal(gmdata, games, tmp_path):  # noqa: F811
    """Every frame, cut right after its decision point (own turns: + the seat's own response), is
    a bit-exact prefix of the full conversion: obs, legal_action_type, and (where the response is
    kept) labels and legal_tile_of.  Covers draws / calls / discards, riichi declaration tiles and
    rinshan draws after an open kan included."""
    records, cases = _causal_cases(games)
    for _, key, cut, _, _ in cases:
        records[key] = cut
    conv = _convert(gmdata, records, tmp_path)
    problems = []
    for base, key, _, seats, free in cases:
        for s in seats:
            why = _prefix_diff(conv[key].get(s), conv[base].get(s), free_last=s in free)
            if why:
                problems.append(f"{key} seat {s}: {why}")
    kinds = {key.rsplit("~", 1)[1].rstrip("0123456789") for _, key, *_ in cases}
    assert kinds == {"draw", "call", "discard"}, kinds
    _fail(problems, "frames depend on events after their decision point")


def _check_claim_existence(gmdata, games, tmp_path, ron: bool):  # noqa: F811
    """Per discard (with / without a ron): the log cut right after it and the log cut after its
    whole response must give every seat the same number of frames."""
    records, cases = {}, []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            for kind, i in _decision_points(ky):
                if kind != "discard":
                    continue
                end, is_ron, _ = _claim_response(ky, i)
                cut = _discard_end(ky, i)
                if is_ron != ron or end <= cut:
                    continue
                key = f"{g}~k{k}~discard{i}"
                records[key + "~cut"] = ky[:cut]
                records[key + "~resp"] = ky[:end]
                cases.append(key)
    if not cases:
        pytest.skip("no such discard in the data")
    conv = _convert(gmdata, records, tmp_path)
    problems = []
    for key in cases:
        cut, resp = conv[key + "~cut"], conv[key + "~resp"]
        for s in range(4):
            if _n(cut.get(s)) != _n(resp.get(s)):
                problems.append(f"{key} seat {s}: {_n(cut.get(s))} frames without the response, "
                                f"{_n(resp.get(s))} with it")
    _fail(problems, "claim frames exist only because of the response")


def test_claim_frames_exist_regardless_of_the_call(gmdata, games, tmp_path):  # noqa: F811
    """A claim frame's existence is decided before the claim: cutting the log before a chi / pon /
    daiminkan of a discard (or before its reach_accepted / dora bookkeeping) adds or removes no
    frame."""
    _check_claim_existence(gmdata, games, tmp_path, ron=False)


def test_ron_frames_exist_regardless_of_the_response(gmdata, games, tmp_path):  # noqa: F811
    """Same invariant on the discards somebody rons: the frame of a seat with a legal ron must
    exist whether or not it rons (v2 emitted it only for the ron when the seat had no chi / pon
    option; v3 emits a claim frame whenever the exact claim mask allows anything but pass)."""
    _check_claim_existence(gmdata, games, tmp_path, ron=True)


def test_riichi_declaration_claims_precede_the_stick(gmdata, games, tmp_path):  # noqa: F811
    """Claim frames on a riichi declaration tile are decided before the riichi stick is
    deposited: the log cut right after the declaration discard gives the full log's frames (v2
    applied the following reach_accepted — declarer -1000, kyotaku +1 — first; the stick is
    deposited only if nobody rons)."""
    records, cases = {}, []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            base = f"{g}~k{k}"
            for kind, i in _decision_points(ky):
                if kind == "discard" and _is_riichi_declaration(ky, i):
                    records[base] = ky
                    key = f"{base}~discard{i}"
                    records[key] = ky[:_discard_end(ky, i)]
                    a = ky[i]["actor"]
                    cases.append((base, key, a, tuple((a + r) % 4 for r in (1, 2, 3))))
    if not cases:
        pytest.skip("no riichi declaration in the data")
    conv = _convert(gmdata, records, tmp_path)
    problems, claims = [], 0
    for base, key, a, others in cases:
        for s in (a, *others):
            claims += s in others and _n(conv[key].get(s)) > 0
            why = _prefix_diff(conv[key].get(s), conv[base].get(s), free_last=s in others)
            if why:
                problems.append(f"{key} seat {s}: {why}")
    if not claims:
        pytest.skip("no claim frame on a riichi declaration tile")
    _fail(problems, "claim frames on riichi declaration tiles show an accepted riichi's stick")


def test_minkan_dora_follows_the_rinshan_response(games):
    """Why the rinshan-draw frame must not show a daiminkan / kakan dora: in the logs that dora is
    flipped by the drawer's next action — it comes with a dahai / ankan (written just before it,
    or in the akochan-reviewer spelling of the Houou corpus just after it), never with a rinshan
    kaihou or a second kakan (Tenhou flips an open kan's dora after the discard)."""
    before, after_or_never, spelled_after = set(), set(), set()
    for evs in games.values():
        for i, e in enumerate(evs):
            h = i - 1
            while h > 0 and evs[h]["type"] == "dora":    # a pending dora flipped by this kan
                h -= 1
            if e["type"] != "tsumo" or h < 0 or evs[h]["type"] not in MINKAN:
                continue
            j = i + 1
            while j < len(evs) and evs[j]["type"] == "dora":
                j += 1
            nxt = evs[j]["type"] if j < len(evs) else "end"
            if j == i + 1 and nxt in ("dahai", "ankan") and j + 1 < len(evs) \
                    and evs[j + 1]["type"] == "dora":
                spelled_after.add(nxt)                  # the dora written after the action
                continue
            (before if j > i + 1 else after_or_never).add(nxt)
    assert before <= {"dahai", "ankan"}, before
    assert after_or_never <= {"hora", "kakan", "end"}, after_or_never
    if not after_or_never:
        pytest.skip("no rinshan kaihou / second kakan after an open kan in the data")
    assert before or spelled_after, "no discard after an open kan in the data"


def test_minkan_dora_is_revealed_after_the_rinshan_decision(gmdata, games, tmp_path):  # noqa: F811
    """The rinshan-draw frame after an open kan is the log up to that tsumo plus the drawer's own
    response: dropping the dora event the log places before the response must not change it (v2
    applied that dora to the frame: a dead-wall tile the seat has not seen yet, shown only when
    the drawer then discards / ankans)."""
    records, cases = {}, []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            base = f"{g}~k{k}"
            for kind, i in _decision_points(ky):
                if kind != "draw" or not _is_minkan_rinshan_with_dora(ky, i):
                    continue
                resp = _own_response(ky, i)
                if resp is None:
                    continue
                records[base] = ky
                key = f"{base}~draw{i}"
                records[key] = ky[:i + 1] + resp
                cases.append((base, key, ky[i]["actor"]))
    if not cases:
        pytest.skip("no rinshan draw after an open kan in the data")
    conv = _convert(gmdata, records, tmp_path)
    problems = []
    for base, key, a in cases:
        why = _prefix_diff(conv[key].get(a), conv[base].get(a))
        if why:
            problems.append(f"{key} seat {a}: {why}")
    _fail(problems, "rinshan-draw frames show the open kan's dora before the discard")


# ---------------------------------------------------------------------------
# two futures of one decision point (independent re-check of the known leaks)
# ---------------------------------------------------------------------------
#
# The cut-log tests above compare a frame with the decision-time state.  These compare two
# possible continuations of the same decision point instead: a frame that differs between them
# tells the network which future happened, whatever the decision-time state is.

def _same_obs(x: Optional[dict], y: Optional[dict], tgt_keys: Tuple[str, ...] = ()) -> Optional[str]:
    """Why two samples of one seat differ in frame count, obs or the ``tgt_keys`` targets.  Never
    tgt.value: the final placement is a hindsight target and legitimately depends on the future."""
    if _n(x) != _n(y):
        return f"{_n(x)} vs {_n(y)} frames"
    if x is None:
        return None
    for group, keys in (("obs", OBS_KEYS), ("tgt", tgt_keys)):
        for k in keys:
            if not torch.equal(x[group][k], y[group][k]):
                return f"{group}." + _first_diff(x[group][k], y[group][k], k)
    return None


def test_declaration_tile_claims_do_not_reveal_a_ron(gmdata, games, tmp_path):  # noqa: F811
    """A riichi declaration tile somebody really ronned, against the counterfactual where nobody
    rons (the log's reach_accepted instead of the hora events): the seats that are neither the
    declarer nor a ronner decide on the same tile with the same information, so their samples
    (frame count, obs, labels, masks; not the hindsight value) must be bit-identical."""
    records, cases = {}, []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            for kind, i in _decision_points(ky):
                if kind != "discard" or not _is_riichi_declaration(ky, i):
                    continue
                end, ron, _ = _claim_response(ky, i)
                if not ron:
                    continue
                a = ky[i]["actor"]
                ronners = {e["actor"] for e in ky[i + 1:end] if e["type"] == "hora"}
                key = f"{g}~k{k}~discard{i}"
                records[key + "~ron"] = ky[:end]
                records[key + "~noron"] = ky[:_discard_end(ky, i)] + [{"type": "reach_accepted", "actor": a}]
                records[key + "~pre"] = ky[:i]      # before the declaration tile
                cases.append((key, [s for s in range(4) if s != a and s not in ronners]))
    if not cases:
        pytest.skip("no ronned riichi declaration tile in the data")
    conv = _convert(gmdata, records, tmp_path)
    problems, claims = [], 0
    for key, third in cases:
        for s in third:
            ron, noron = conv[key + "~ron"].get(s), conv[key + "~noron"].get(s)
            claims += _n(ron) > _n(conv[key + "~pre"].get(s))    # a claim frame on this tile
            why = _same_obs(ron, noron, LABEL_KEYS + ("legal_action_type", "legal_tile_of"))
            if why:
                problems.append(f"{key} seat {s}: {why}")
    if not claims:
        pytest.skip("no third-seat claim frame on a ronned riichi declaration tile")
    _fail(problems, "third-seat claim frames on a declaration tile tell whether somebody rons")


def test_rinshan_frame_does_not_reveal_the_response(gmdata, games, tmp_path):  # noqa: F811
    """A rinshan draw after a daiminkan / kakan whose drawer really discarded (the log flips the
    kan's dora in between), against the counterfactual rinshan kaihou (a tsumo hora right after
    the draw: no dora flipped, as in every real rinshan kaihou of the logs): the drawer's frames
    must carry the same obs — only the last label may differ."""
    records, cases = {}, []
    for g, evs in games.items():
        for k, ky in enumerate(_kyokus(evs)):
            for kind, i in _decision_points(ky):
                if kind != "draw" or not _is_minkan_rinshan_with_dora(ky, i):
                    continue
                j = i + 1
                while j < len(ky) and ky[j]["type"] == "dora":
                    j += 1
                if j >= len(ky) or ky[j]["type"] != "dahai":
                    continue
                a = ky[i]["actor"]
                key = f"{g}~k{k}~draw{i}"
                records[key + "~dahai"] = ky[:j + 1]
                records[key + "~hora"] = ky[:i + 1] + [{"type": "hora", "actor": a, "target": a}]
                cases.append((key, a))
    if not cases:
        pytest.skip("no discard after an open kan's rinshan draw in the data")
    conv = _convert(gmdata, records, tmp_path)
    problems = []
    for key, a in cases:
        why = _same_obs(conv[key + "~dahai"].get(a), conv[key + "~hora"].get(a))
        if why:
            problems.append(f"{key} seat {a}: {why}")
    _fail(problems, "rinshan-draw frames tell a discard from a rinshan kaihou")


def test_legal_masks_never_fall_back_to_all_true(gmdata, games, tmp_path):  # noqa: F811
    """A frame whose ground truth falls outside its computed masks gets all-true masks
    (`mask_mismatch`).  That fallback is a function of the label, and the decoder reads the type
    mask at decision time under --il_legal_mask, so it must never trigger on real logs (the
    production conversion of 16000 games reported 0)."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for g, evs in games.items():
        (src / f"{g}.mjson").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    r = subprocess.run([gmdata, "mahjong", "convert", "--input", str(src), "--out", str(out),
                        "--min-frames", "1", "--shard-size", "1000000"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    stats = json.loads(r.stdout)
    assert stats["skipped"] == 0 and stats["games"] == len(games), stats
    assert stats["mask_mismatch"] == 0, stats


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------

def _hide_from(evs: List[dict], protect: int, rng: random.Random, mode: str) -> List[dict]:
    """A copy of ``evs`` whose tiles private to the seats other than ``protect`` are reassigned
    without changing any event ``protect`` observes (discards, calls with their consumed tiles,
    ankan, kakan, riichi, dora markers, deltas):

      'swap'     a physically possible deal (tile counts kept): repeatedly swap two tiles one seat
                 holds at the same time — so a tile it plays later may have been dealt or drawn at
                 another moment — then permute the never-revealed tiles across the other seats;
      'replace'  every never-revealed tile of the other seats becomes a random kind (tile counts
                 not kept).
    ura_markers are randomised in both modes."""
    evs = copy.deepcopy(evs)
    starts = [i for i, e in enumerate(evs) if e["type"] == "start_kyoku"]
    for a, b in zip(starts, starts[1:] + [len(evs)]):
        start = evs[a]
        slots_by_seat, hidden = [], []
        for q in range(4):
            if q == protect:
                continue
            # slot = [acquired at, tile, released at, where the tile is written]
            slots = [[a, p, NEVER, ("tehai", q, n)] for n, p in enumerate(start["tehais"][q])]
            held = list(range(len(slots)))
            for i in range(a + 1, b):
                e = evs[i]
                if e.get("actor") != q:
                    continue
                t = e["type"]
                if t == "tsumo":
                    slots.append([i, e["pai"], NEVER, ("tsumo", i)])
                    held.append(len(slots) - 1)
                    continue
                if t in ("dahai", "kakan"):
                    released = [e["pai"]]
                elif t in ("chi", "pon", "daiminkan", "ankan"):
                    released = list(e["consumed"])
                else:
                    released = []
                for p in released:
                    x = next(x for x in held if slots[x][1] == p)
                    slots[x][2] = i
                    held.remove(x)
            if mode == "swap":
                for _ in range(20 * len(slots)):
                    x, y = slots[rng.randrange(len(slots))], slots[rng.randrange(len(slots))]
                    if x[1] != y[1] and y[0] < x[2] and x[0] < y[2]:
                        x[1], y[1] = y[1], x[1]
                        x[2], y[2] = y[2], x[2]
            slots_by_seat.append(slots)
            hidden += [s for s in slots if s[2] == NEVER]
        tiles = [s[1] for s in hidden]
        if mode == "swap":
            rng.shuffle(tiles)
        else:
            tiles = [rng.choice(TILES) for _ in tiles]
        for s, p in zip(hidden, tiles):
            s[1] = p
        for slots in slots_by_seat:
            for _, p, _, where in slots:
                if where[0] == "tehai":
                    start["tehais"][where[1]][where[2]] = p
                else:
                    evs[where[1]]["pai"] = p
        for i in range(a, b):
            if evs[i]["type"] == "hora" and evs[i].get("ura_markers"):
                evs[i]["ura_markers"] = [rng.choice(TILES) for _ in evs[i]["ura_markers"]]
    return evs


def _same_sample(x: Optional[dict], y: Optional[dict]) -> Optional[str]:
    if _n(x) != _n(y):
        return f"{_n(x)} vs {_n(y)} frames"
    if x is None:
        return None
    for group in ("obs", "tgt"):
        if set(x[group]) != set(y[group]):
            return f"{group} keys differ"
        for key in x[group]:
            if not torch.equal(x[group][key], y[group][key]):
                return f"{group}." + _first_diff(x[group][key], y[group][key], key)
    return None


def test_seat_view_invariant_to_opponent_private_tiles(gmdata, games, tmp_path):  # noqa: F811
    """Seat p's whole sample — obs, legal masks, labels, value, frame count — is bit-identical
    when the other seats' concealed tiles and the ura-dora change (:func:`_hide_from`), while
    those seats' own samples do change (the perturbation reaches what the converter encodes)."""
    rng = random.Random(0)
    records, cases = dict(games), []
    for gi, (g, evs) in enumerate(games.items()):
        for p in range(4):
            mode = ("swap", "replace")[(gi + p) % 2]
            key = f"{g}~hide{p}{mode}"
            records[key] = _hide_from(evs, p, rng, mode)
            assert records[key] != evs
            cases.append((g, key, p))
    conv = _convert(gmdata, records, tmp_path)
    problems, reached = [], 0
    for g, key, p in cases:
        why = _same_sample(conv[g].get(p), conv[key].get(p))
        if why:
            problems.append(f"{key} seat {p}: {why}")
        reached += any(_same_sample(conv[g].get(q), conv[key].get(q))
                       for q in range(4) if q != p)
    assert reached == len(cases), f"no opponent sample changed in {len(cases) - reached} cases"
    _fail(problems, "seat views depend on opponents' private tiles")
