"""Label leak through FRAME SELECTION in ``gmdata mahjong convert``.

converter_version 2 emitted a claim frame (after another seat's discard) for
a seat that called on the discard, or that passed while cheaply chi/pon-
eligible and not in riichi.  A seat that could only ron (or was in riichi)
and passed got NO frame — so on every claim frame where chi/pon/kan were not
legal (riichi seats included) the label was hora with certainty: the frame's
existence was the answer.  The deployed bot is asked at every such discard,
and hora legality was a superset there (yaku / furiten unchecked), so the IL
policy learned "ron whenever the shape allows", and claim-frame IL accuracy
was inflated.

converter_version 3 emits a claim frame whenever the seat may do anything
besides pass under Tenhou's exact rules (the one-han minimum and furiten
included), whatever it then did; these tests pin that.  Skips without cargo.

Independent verification adds the counterfactual form (the decision taken
must not decide whether its frame exists: ron vs declined ron on the same
discard) and shows the ``tgt.legal_*`` all-true fallback is not a network
input.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter

import pytest
import torch

from data_processing.games.dataset import load_gm_samples
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture)

HORA, PASS = 5, 7
CHI, PON, KAN = 2, 3, 4
TABLE_OWN_TURN = 9          # obs.table_seq column: 1 on own-turn frames
PLAYER_RIICHI = 1           # obs.players_seq[:, 0] column: own riichi flag


@pytest.fixture(scope="module")
def claim_frames(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("audit_mj") / "out"
    # the audit fixtures hold legal rons declined by a riichi and by a dama seat (rare in the
    # golden set: phoenix players decline ~0.7% of their legal rons)
    subprocess.run([gmdata, "mahjong", "convert", "--input", f"{FIX}/mahjong", f"{FIX}/mahjong_audit",
                    "--out", str(out)], check=True, capture_output=True)
    only_ron, riichi = Counter(), Counter()
    for s in load_gm_samples(str(out)):
        o, t = s["obs"], s["tgt"]
        for i in range(t["action_type"].shape[0]):
            if o["table_seq"][i, TABLE_OWN_TURN] == 1:
                continue
            lab = int(t["action_type"][i])
            la = t["legal_action_type"][i]
            if la[CHI] == 0 and la[PON] == 0 and la[KAN] == 0:
                only_ron[lab] += 1
            if o["players_seq"][i, 0, PLAYER_RIICHI] == 1:
                riichi[lab] += 1
    return only_ron, riichi


def test_claim_frames_exist_on_the_fixture(claim_frames):
    only_ron, riichi = claim_frames
    assert sum(only_ron.values()) >= 20 and sum(riichi.values()) >= 5


def test_ron_only_claim_frames_are_not_always_hora(claim_frames):
    """Claim frames without a chi/pon/kan option (a legal ron only; riichi seats included) also
    exist when the seat declined the ron (v2: only when it ronned)."""
    only_ron, riichi = claim_frames
    assert only_ron[PASS] > 0 and riichi[PASS] > 0, (dict(only_ron), dict(riichi))


# ---------------------------------------------------------------------------
# Counterfactual form of the same leak: for every ron in the fixture games
# build two event streams that agree up to the discard, ``took`` (the recorded
# ron) and ``passed`` (the ron replaced by an exhaustive draw, i.e. the seat
# declined), convert both, and require every seat to have the same frames,
# with bit-identical obs, in both.  A seat that really ronned had a legal ron
# (yaku, no furiten), so any correct emission rule (superset or exact hora
# legality; v3 is exact) gives it a pass frame in ``passed``.  Unlike the sentinel above
# this does not depend on real declined rons being present in the fixtures.
# ---------------------------------------------------------------------------

_TAIL = [{"type": "end_kyoku"}, {"type": "end_game"}]


def _ron_counterfactuals(evs):
    """[(k, winners, took_events, passed_events)] per ron group (a double ron
    is one group) whose trigger is a discard (chankan is excluded)."""
    out = []
    i = 0
    while i < len(evs):
        e = evs[i]
        if e.get("type") == "hora" and e.get("actor") != e.get("target"):
            j, winners = i, []
            while (j < len(evs) and evs[j].get("type") == "hora"
                   and evs[j].get("target") == e["target"] and evs[j]["actor"] != e["target"]):
                winners.append(evs[j]["actor"])
                j += 1
            trig = max(k for k in range(i) if evs[k].get("type") in ("dahai", "kakan", "ankan"))
            if evs[trig]["type"] == "dahai":
                out.append((len(out), winners, evs[:j] + _TAIL,
                            evs[:i] + [{"type": "ryukyoku", "deltas": [0, 0, 0, 0]}] + _TAIL))
            i = j
        else:
            i += 1
    return out


@pytest.fixture(scope="module")
def ron_pairs(gmdata, tmp_path_factory):  # noqa: F811
    raw = tmp_path_factory.mktemp("audit_mj_cf_raw")
    cases = []
    for fn in sorted(os.listdir(f"{FIX}/mahjong")):
        if not fn.endswith(".mjson"):
            continue
        with open(f"{FIX}/mahjong/{fn}") as f:
            evs = [json.loads(line) for line in f if line.strip()]
        for k, winners, took, passed in _ron_counterfactuals(evs):
            stem = f"{fn[:-len('.mjson')]}__ron{k:02d}"
            for tag, stream in (("took", took), ("passed", passed)):
                with open(raw / f"{stem}__{tag}.mjson", "w") as f:
                    f.write("".join(json.dumps(x) + "\n" for x in stream))
            cases.append((stem, winners))
    out = tmp_path_factory.mktemp("audit_mj_cf") / "out"
    subprocess.run([gmdata, "mahjong", "convert", "--input", str(raw), "--out", str(out),
                    "--min-frames", "1"], check=True, capture_output=True)
    by = {(s["meta"]["name"], int(s["meta"]["seat"])): s for s in load_gm_samples(str(out))}
    pairs = []
    for stem, winners in cases:
        for seat in range(4):
            a = by.get((f"{stem}__took.mjson", seat))
            b = by.get((f"{stem}__passed.mjson", seat))
            if a is None or b is None:
                assert seat not in winners, (stem, seat)
                continue
            # the shared prefix (including the frame AT the discard, when the
            # declined stream has one) is bit-identical: obs never see the decision
            n = min(_n(a), _n(b))
            for k in a["obs"]:
                assert torch.equal(a["obs"][k][:n], b["obs"][k][:n]), (stem, seat, k)
            if seat in winners:
                assert int(a["tgt"]["action_type"][-1]) == HORA, (stem, seat)
                la = a["tgt"]["legal_action_type"][-1]
                pairs.append((stem, seat, bool(la[CHI] or la[PON] or la[KAN]), a, b))
            else:                          # bystanders: same frames, same masks either way
                assert _n(a) == _n(b), (stem, seat)
                for k in ("legal_action_type", "legal_tile_of"):
                    assert torch.equal(a["tgt"][k], b["tgt"][k]), (stem, seat, k)
    return pairs


def _n(s) -> int:
    return int(s["tgt"]["action_type"].shape[0])


def test_ron_counterfactuals_exist_on_the_fixture(ron_pairs):
    assert sum(1 for p in ron_pairs if not p[2]) >= 20      # ron was the only claim
    assert sum(1 for p in ron_pairs if p[2]) >= 5           # chi/pon/kan legal too


def test_ron_frame_exists_either_way_when_a_call_was_also_legal(ron_pairs):
    """Control (passed in v2 already): when chi/pon/kan was legal too, the
    declining seat keeps its frame (label pass, obs identical to the ron
    frame's)."""
    for stem, w, callable_, a, b in ron_pairs:
        if callable_:
            assert _n(a) == _n(b), (stem, w)
            assert int(b["tgt"]["action_type"][-1]) == PASS, (stem, w)


def test_ron_frame_exists_either_way(ron_pairs):
    """Every recorded ron keeps its frame when replaced by a declined ron (v2 lost the frames of
    rons without a chi/pon/kan option)."""
    missing = [(stem, w) for stem, w, _, a, b in ron_pairs if _n(a) != _n(b)]
    assert not missing, f"{len(missing)}/{len(ron_pairs)} rons lose their frame when declined"


# ---------------------------------------------------------------------------
# tgt.legal_* all-true fallback (reported as a leak, refuted): the masks are
# not a network input, they only gate the logits.  Under teacher forcing every
# raw output is bit-identical whether a frame carries rule masks or the
# all-true fallback; only the echoed masks differ.  (The fallback fires 0
# times on the fixtures, see the Rust test fixture_games_have_no_mask_mismatch.)
# ---------------------------------------------------------------------------

def test_mahjong_legal_masks_never_reach_teacher_forced_outputs():
    from tests.test_audit_model_causality import _batch, _labels, _mode, _net, _run
    net = _net("mahjong")
    obs, legal, _, _ = _batch("mahjong", net)
    labels = _labels("mahjong", net.cfg, torch.Generator().manual_seed(3), self_sample=False)
    all_true = {k: torch.ones_like(v) for k, v in legal.items()}
    echoes = {"action_type_mask", "tile_mask"}
    for mode in ("eval", "train"):
        with _mode(net, mode):
            a = _run(net, obs, legal, labels)
            b = _run(net, obs, all_true, labels)
        assert echoes <= set(a)
        for k in sorted(set(a) - echoes):
            assert torch.equal(a[k], b[k]), (mode, k)
        assert not torch.equal(a["action_type_mask"], b["action_type_mask"])
