"""Label leak through FRAME SELECTION in ``gmdata mahjong convert``.

A claim frame (after another seat's discard) is emitted for a seat that
called on the discard, or that passed while cheaply chi/pon-eligible and not
in riichi.  A seat that could only ron (or is in riichi) and passed gets NO
frame — so on every claim frame where chi/pon/kan are not legal (riichi
seats included) the label is hora with certainty: the frame's existence is
the answer.  The deployed bot is asked at every such discard, and hora
legality is a superset there (yaku / furiten unchecked), so the IL policy
learns "ron whenever the shape allows", and claim-frame IL accuracy is
inflated.

Strict xfail until the converter also emits pass frames for ron-shape (and
riichi) opportunities that were not taken.  Skips without cargo.
"""
from __future__ import annotations

import subprocess
from collections import Counter

import pytest

from data_processing.games.dataset import load_gm_samples
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture)

HORA, PASS = 5, 7
CHI, PON, KAN = 2, 3, 4
TABLE_OWN_TURN = 9          # obs.table_seq column: 1 on own-turn frames
PLAYER_RIICHI = 1           # obs.players_seq[:, 0] column: own riichi flag


@pytest.fixture(scope="module")
def claim_frames(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("audit_mj") / "out"
    subprocess.run([gmdata, "mahjong", "convert", "--input", f"{FIX}/mahjong", "--out", str(out)],
                   check=True, capture_output=True)
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


@pytest.mark.xfail(strict=True, reason="AUDIT LEAK: mahjong claim frames without a chi/pon/kan "
                                       "option exist only when the seat ronned (label = hora)")
def test_ron_only_claim_frames_are_not_always_hora(claim_frames):
    only_ron, riichi = claim_frames
    assert only_ron[PASS] > 0 and riichi[PASS] > 0, (dict(only_ron), dict(riichi))
