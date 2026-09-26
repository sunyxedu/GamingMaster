"""go converter v2 (`gmdata go convert`): obs.globals_seq [T,6] — side to move,
komi from the mover's view, komi known, area scoring, handicap, previous move
was a pass — on the committed KGS fixtures, and the shard meta the IL
trainers read the model width from ({global_features: 6, board_size: N})."""
from __future__ import annotations

import os
import re
import subprocess

import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from model.games import build_game_network
from tests.test_rust_converters_parity import CASES, gmdata  # noqa: F401  (fixture)

SMALL = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, core_type="trm",
             trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1, halt_max_steps=1,
             dropout=0.0, encoder_channels=8)
AREA_RULES = {"chinese", "aga", "nz", "new zealand", "area"}


def _convert(gm, args, out):
    subprocess.run([gm, *args, "--out", str(out)], check=True, capture_output=True)
    return load_gm_samples(str(out)), GameShardDataset(str(out))


@pytest.fixture(scope="module")
def converted(gmdata, tmp_path_factory):  # noqa: F811
    args, _ = CASES["go_handicap_noresult"]     # handicap + even games, both rulesets
    return _convert(gmdata, args, tmp_path_factory.mktemp("go") / "rs")


def _sgf(source):
    """(move colours, lower-cased RU) of a single-line KGS record: property
    values are blanked first so a B/W identifier is only ever matched as a
    property (a move may follow other properties in its node)."""
    text = open(source, encoding="utf-8", errors="replace").read()
    ru = re.search(r"RU\[([^\]]*)\]", text)
    idents = re.sub(r"\[(?:\\.|[^\]\\])*\]", "[]", text)
    colours = re.findall(r"(?<![A-Z])([BW])\[\]", idents)
    return colours, (ru.group(1).strip().lower() if ru else None)


def test_globals_follow_the_record(converted):
    samples, ds = converted
    assert ds.converter_version == 2
    assert ds.shard_meta == {"global_features": 6, "board_size": 19}
    seen = {"white_first": False, "pass": False, "territory": False, "handicap": False}
    for s in samples:
        g, m = s["obs"]["globals_seq"], s["meta"]
        T = s["tgt"]["place_idx"].shape[0]
        assert g.dtype == torch.float32 and g.shape == (T, 6)
        assert s["obs"]["board_seq"].shape == (T, 4, 19, 19)          # planes unchanged

        # 0: mover is black, from the SGF's own B/W properties
        colours, ru = _sgf(m["source"])
        assert len(colours) == T == m["n_moves"]
        black = torch.tensor([c == "B" for c in colours])
        assert torch.equal(g[:, 0], black.float())
        seen["white_first"] |= colours[0] == "W"

        # 1: komi / 20 for white, sign-flipped for black; 2: komi known
        assert m["komi"] is not None
        k = torch.tensor(m["komi"] / 20.0, dtype=torch.float64).clamp(-1, 1).float()
        assert torch.equal(g[~black, 1], k.expand(int((~black).sum())))
        assert torch.equal(g[black, 1], (-k).expand(int(black.sum())))
        assert (g[:, 2] == 1).all()

        # 3: area scoring from RU; 4: handicap stones
        area = float(ru in AREA_RULES)
        assert (g[:, 3] == area).all()
        seen["territory"] |= area == 0.0
        assert torch.allclose(g[:, 4], torch.full((T,), min(m["handicap"], 9) / 9.0))
        seen["handicap"] |= m["handicap"] > 0

        # 5: the previous move (either colour) was a pass
        passed = s["tgt"]["passed"].float()
        assert g[0, 5] == 0
        assert torch.equal(g[1:, 5], passed[:-1])
        seen["pass"] |= bool(passed[:-1].any())
    assert all(seen.values()), seen                     # the fixtures cover every case


def test_komi_unknown_and_edge_rules(gmdata, tmp_path):  # noqa: F811
    """Missing / unparseable KM → komi 0 and 'known' 0; territory rules → 0;
    a huge komi clips at ±1; HA over 9 clips at 1."""
    src = tmp_path / "sgf"
    src.mkdir()
    moves = ";B[dd];W[pp];B[dp];W[pd];B[];W[qq]"
    (src / "a.sgf").write_text(f"(;GM[1]SZ[19]RU[Japanese]RE[B+R]{moves})")
    (src / "b.sgf").write_text(f"(;GM[1]SZ[19]KM[x]RU[Korean]RE[W+R]{moves})")
    (src / "c.sgf").write_text(f"(;GM[1]SZ[19]KM[45]RU[New Zealand]RE[W+R]{moves})")
    samples, ds = _convert(gmdata, ["go", "convert", "--input", str(src), "--min-moves", "1"],
                           tmp_path / "rs")
    by = {os.path.basename(s["meta"]["source"]): s["obs"]["globals_seq"] for s in samples}
    assert set(by) == {"a.sgf", "b.sgf", "c.sgf"}
    for name in ("a.sgf", "b.sgf"):
        g = by[name]
        assert (g[:, 1:5] == 0).all(), name
        assert g[:, 0].tolist() == [1, 0, 1, 0, 1, 0]
        assert g[:, 5].tolist() == [0, 0, 0, 0, 0, 1]
    c = by["c.sgf"]
    assert c[:, 1].tolist() == [-1, 1, -1, 1, -1, 1]
    assert (c[:, 2] == 1).all() and (c[:, 3] == 1).all()
    assert ds.shard_meta == {"global_features": 6, "board_size": 19}


def test_board_encoder_consumes_go_globals(converted):
    """A network built with board_global_features=6 (the width fresh runs
    take from the shard meta) runs forward on a converted sample and its
    output depends on the globals; the legacy registry default (0) still
    builds and simply ignores obs.globals_seq."""
    samples, _ = converted
    s = samples[0]
    crop = {k: v[:12] for k, v in s["obs"].items()}
    batch = collate_game_sequences([{"obs": crop, "tgt": {k: v[:12] for k, v in s["tgt"].items()},
                                     "meta": s["meta"]}])
    obs, fm = batch["obs"], batch["meta"]["frame_mask"]
    assert obs["globals_seq"].shape == (1, 12, 6)

    torch.manual_seed(0)
    net = build_game_network("go", board_global_features=6, **SMALL).eval()
    with torch.no_grad():
        out = net(obs, act_steps=1, frame_mask=fm)
        other = {**obs, "globals_seq": 1 - obs["globals_seq"]}
        flipped = net(other, act_steps=1, frame_mask=fm)
    assert out["place_logits"].shape == (1, 12, 361)
    assert torch.isfinite(out["place_logits"]).all() and torch.isfinite(out["value"]).all()
    assert not torch.allclose(out["place_logits"], flipped["place_logits"])

    legacy = build_game_network("go", **SMALL).eval()
    assert legacy.cfg.board_global_features == 0 and legacy.obs_encoder.global_proj is None
    with torch.no_grad():
        assert torch.isfinite(legacy(obs, act_steps=1, frame_mask=fm)["place_logits"]).all()
