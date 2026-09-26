"""Xiangqi end to end on REAL records (tests/fixtures/xiangqi/ccpd_sample.zip: 31 files
of the Chinese Chess Practical Dataset, CC BY 4.0, original Big5 bytes and paths — see
tests/fixtures/xiangqi/CCPD_ATTRIBUTION.txt):

    CCPD PGN ─► gmdata xiangqi convert (moves resolved against the legal moves, every
      record replayed by the gmrules rules) ─► gm-shards (fromto.v1)
      ─► BoardEncoder / FromToDecoder (build_fromto_network) ─► train_multi_game
      ─► gmsim env (GmSimVecEnv) ─► train_rl_joint (FromToSpec)

- spec + registrations (model registry, LABEL_KEYS / META_ARCH_KEYS, env, RL spec);
- the converter on the real sample: every record class ends where it should
  (resignations, checkmate, stalemate, perpetual check / chase, repetition, move limit,
  cuts after an unclaimed repetition or a mate, results from the [Event] text,
  conflicting results, duplicates across the CCPD views, the rejections: a
  transcription error, a header contradicting a mate, an empty record, an endgame
  start); its moves are the ones Fairy-Stockfish replayed
  (tests/fixtures/xiangqi/ffish_ccpd_sample.json.gz);
- shard frames == gmsim.replay frames; value target = the mover's outcome;
- GOLDEN: the RL env stepped with the records' own action keys reproduces every
  converter frame bit for bit (obs, legality = the IL masks, applied keys) and ends
  exactly where the rules end each record, with the outcome as reward;
- a network's sampled (from, to) is always legal in the env;
- invariance: simulator battery (causality; perfect information — nothing to perturb)
  and converter causality on real records (their truncations are frame prefixes);
- IL contract on real batches: il_head_terms decomposes il_loss, the shards' IL masks
  make every target legal, soft targets (the tensors a Pikafish labeller would write:
  from / to heads, sparse) — one-hot teachers = the hard loss;
- train_multi_game with existing games (the gomoku fixture shards, kuhn_poker), then
  train_rl_joint on xiangqi + kuhn_poker: μ = π_old, finite losses, the compact
  FromToSpec replay legality.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
    load_gm_samples,
)
from env.vec_game import has_simulator, make_vec_env
from env.vec_gmsim import GmSimVecEnv
from model.games import build_game_network
from model.games.gmrules import rules_games
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
FIX = ROOT / "tests" / "fixtures" / "xiangqi"
SAMPLE = FIX / "ccpd_sample.zip"
FFISH_REAL = FIX / "ffish_ccpd_sample.json.gz"
GOMOKU = ROOT / "tests" / "fixtures" / "gomoku_env" / "shards20"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
N = 90
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent")
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
CPU = torch.device("cpu")
M = "Dataset/對局/大師對局/"

#: record → (outcome for Red, how the record ends) of every converted sample record;
#: "adjudicated" = the recorded result (resignation / agreement), else the rules' end
EXPECTED = {
    M + "以棋手分類/吳貴臨/00000002.pgn": (1, "adjudicated"),
    M + "以棋手分類/吳貴臨/00000003.pgn": (1, "adjudicated"),
    M + "以賽事分類/世界智力運動會/00000011.pgn": (-1, "adjudicated"),
    M + "以賽事分類/世界智力運動會/00000012.pgn": (-1, "adjudicated"),
    M + "以開局分類/中炮局/00000027.pgn": (0, "adjudicated"),
    M + "以開局分類/中炮局/00000029.pgn": (0, "adjudicated"),
    M + "以棋手分類/呂欽/00000006.pgn": (0, "adjudicated"),            # 319 plies
    "Dataset/對局/電腦對局/人機賽/00000002.pgn": (1, "adjudicated"),
    M + "以棋手分類/楊官璘/00000686.pgn": (-1, "adjudicated"),         # * + 先負 in [Event]
    M + "以棋手分類/胡榮華/00001826.pgn": (-1, "adjudicated"),         # copy of a conflict
    M + "以棋手分類/呂欽/00000193.pgn": (-1, "checkmate"),
    M + "以棋手分類/呂欽/00000570.pgn": (-1, "checkmate"),
    M + "以棋手分類/李來群/00001400.pgn": (1, "checkmate"),            # 和 vs 先勝: mate decides
    M + "以棋手分類/楊官璘/00000643.pgn": (1, "checkmate"),            # * : mate decides
    M + "以棋手分類/胡榮華/00000463.pgn": (-1, "stalemate"),           # stalemate loses
    M + "以賽事分類/全國賽對局譜/00001873.pgn": (1, "perpetual_check"),
    M + "以棋手分類/呂欽/00000072.pgn": (1, "perpetual_check+cut"),
    M + "以棋手分類/許銀川/00000461.pgn": (-1, "perpetual_chase"),     # drawn by the players
    M + "以棋手分類/呂欽/00000029.pgn": (-1, "perpetual_chase+cut"),
    M + "以棋手分類/呂欽/00000227.pgn": (0, "repetition"),
    M + "以棋手分類/吳貴臨/00000009.pgn": (0, "repetition+cut"),
    M + "以賽事分類/五羊杯/00000103.pgn": (0, "move_limit"),
    M + "以棋手分類/呂欽/00000381.pgn": (0, "move_limit+cut"),
    "Dataset/對局/電腦對局/電腦對局競賽/00000011.pgn": (-1, "checkmate+cut"),  # moves after mate
}


def _run(*args) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("xq")
    st = _run(gmdata, "xiangqi", "convert", "--input", SAMPLE, "--out", root / "all",
              "--store-events")
    hq = _run(gmdata, "xiangqi", "convert", "--input", SAMPLE, "--out", root / "hq", "--hq")
    _run(gmdata, "kuhn-poker", "generate", "--out", root / "raw" / "h.log", "--hands", "200")
    _run(gmdata, "kuhn-poker", "convert", "--input", root / "raw", "--out", root / "kuhn")
    return root, st, hq


def _by_record(samples: List[Dict]) -> Dict[str, Dict]:
    return {s["meta"]["record"]: s for s in samples}


# ---------------------------------------------------------------------------
# spec, registrations
# ---------------------------------------------------------------------------

def test_spec_and_registrations():
    spec = gmsim.spec_of("xiangqi", {})
    assert spec["schema"] == "fromto.v1" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"], t["dtype"], t["bits"]) for t in spec["obs"]] == [
        ("board_seq", [19, 10, 9], "f32", True), ("globals_seq", [7], "f32", False)]
    assert [(t["name"], t["shape"]) for t in spec["legal"]] == [("from", [N]), ("to_cond", [N, N])]
    assert [t["name"] for t in spec["target"]] == ["from_idx", "to_idx", "legal_from",
                                                   "legal_to_of"]
    assert spec["action_keys"] == ["from_idx", "to_idx"]
    assert spec["meta"] == {"board_h": 10, "board_w": 9, "piece_channels": 19,
                            "global_features": 7}
    entry = rules_games()["xiangqi"]
    assert not entry.is_candidate and has_simulator("xiangqi")
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS
    from training.train_rl_joint import GAME_SPECS, FromToSpec
    assert LABEL_KEYS["xiangqi"] == ("from_idx", "to_idx")
    assert META_ARCH_KEYS["xiangqi"] == {"board_h": ("board_h",), "board_w": ("board_w",),
                                         "piece_channels": ("piece_channels",),
                                         "global_features": ("board_global_features",)}
    assert isinstance(GAME_SPECS["xiangqi"], FromToSpec)
    net = build_game_network("xiangqi", **TINY)                # dims: the simulator's spec
    assert (net.cfg.board_h, net.cfg.board_w, net.cfg.piece_channels,
            net.cfg.board_global_features, net.cfg.n_promotions) == (10, 9, 19, 7, 0)
    assert net.obs_encoder.n_tokens == N and net.action_decoder.promotion_head is None


# ---------------------------------------------------------------------------
# the converter on real records
# ---------------------------------------------------------------------------

def test_converter_on_real_records(shards):
    root, st, hq = shards
    assert (st["files"], st["games"], st["records"], st["converted"]) == (31, 31, 29, 24)
    assert (st["illegal"], st["unparseable"], st["returns_mismatch"], st["truncated"],
            st["empty"]) == (0, 1, 1, 2, 1)
    assert (st["other_start"], st["duplicates"], st["cut"], st["result_from_event"],
            st["result_conflict"]) == (1, 1, 5, 2, 2)
    assert st["frames"] == 2567 and st["adjudicated"] == 10
    errors = "\n".join(st["errors"])
    assert "其他/00000753.pgn: ply 70: \"士６進５\": no legal move matches" in errors
    assert "世界盃/00000143.pgn: recorded outcome [-1.0, 1.0], the rules end" in errors
    samples = _by_record(load_gm_samples(str(root / "all")))
    assert set(samples) == set(EXPECTED)
    for rec, (red, how) in EXPECTED.items():
        m = samples[rec]["meta"]
        assert m["outcome"] == [red, -red], rec
        assert m.get("adjudicated", False) == (how == "adjudicated"), rec
        end = how.split("+")[0]
        if end in ("perpetual_check", "perpetual_chase", "repetition", "move_limit") \
                or how.endswith("+cut"):
            assert m["rules_end"] == end, rec
        assert ("cut_at" in m) == how.endswith("+cut"), rec
        if "cut_at" in m:
            assert m["cut_at"] == m["plies"] < m["record_plies"], rec
        assert m["plies"] == len(m["events"]) == samples[rec]["obs"]["board_seq"].shape[0]
    # the conflicting copy (全盤戰術: header 1-0, [Event] 先負) claimed nothing: its later
    # copy with the right result was kept; the duplicate of a kept game was dropped
    assert "Dataset/全盤戰術/00000103.pgn" not in samples
    assert samples[M + "以棋手分類/李來群/00001400.pgn"]["meta"]["result_source"] == "conflict"
    assert samples[M + "以棋手分類/楊官璘/00000686.pgn"]["meta"]["result_source"] == "event"
    # --hq: master games with a known result: not the 2 computer games, nor the 2 whose
    # result only the rules' checkmate decides ([Result "*"]; header vs [Event] conflict)
    assert (hq["converted"], hq["truncated"]) == (20, 0)


def test_converter_moves_are_the_ones_fairy_stockfish_replayed(shards):
    """The converted real games = the move lists of the Fairy-Stockfish fixture
    (tests/test_xiangqi_reference.py compares every position of those)."""
    root, _, _ = shards
    samples = _by_record(load_gm_samples(str(root / "all")))
    with gzip.open(FFISH_REAL, "rt") as fh:
        ref = json.load(fh)["trajectories"]
    assert {t["record"] for t in ref} == set(samples)
    for t in ref:
        st = gmsim.State("xiangqi", {})
        moves = []
        for seat, a in samples[t["record"]]["meta"]["events"]:
            moves.append(st.action_string(seat, a))
            st.apply_action(seat, a)
        assert moves == t["moves"], t["record"]
        assert (t["result"] is None) == samples[t["record"]]["meta"].get("adjudicated", False)


def test_shards_are_the_rules_frames(shards):
    root, st, _ = shards
    ds = GameShardDataset(str(root / "all"), game="xiangqi")
    m = ds.shard_meta
    assert (m["schema"], m["layout"], m["board_h"], m["board_w"], m["piece_channels"],
            m["global_features"], m["rules_version"]) == ("fromto.v1", "interleaved", 10, 9,
                                                          19, 7, 1)
    assert ds.total_frames() == st["frames"]
    s = ds[0]
    assert set(s["obs"]) == {"board_seq", "globals_seq"}
    assert set(s["tgt"]) == {"from_idx", "to_idx", "legal_from", "legal_to_of", "value"}
    for smp in load_gm_samples(str(root / "all")):
        r = gmsim.replay("xiangqi", {}, smp["meta"]["events"])
        for k in r["obs"]:
            assert np.array_equal(smp["obs"][k].numpy(), r["obs"][k]), k
        for k in r["target"]:
            assert np.array_equal(smp["tgt"][k].numpy(), r["target"][k]), k
        # interleaved: the value target is the MOVER's outcome, frame by frame
        want = [smp["meta"]["outcome"][s] for s in r["seat"]]
        assert smp["tgt"]["value"].tolist() == want
        assert r["terminal"] == (not smp["meta"].get("adjudicated", False))


# ---------------------------------------------------------------------------
# GOLDEN: env frames == converter frames on the real records
# ---------------------------------------------------------------------------

def test_env_reproduces_the_converter_frames_of_real_records(shards):
    """One batched GmSimVecEnv row per record, stepped with the record's own action
    keys (the shard targets): every observation / legality table equals the
    converter's frame and IL masks bit for bit, the applied keys are the record's,
    and each row ends exactly where the rules end its record (outcome = reward);
    rows whose record ended off the rules (resignation) play on randomly."""
    root, _, _ = shards
    samples = load_gm_samples(str(root / "all"))
    n = len(samples)
    lengths = [s["obs"]["board_seq"].shape[0] for s in samples]
    env = GmSimVecEnv("xiangqi", n, seed=0, max_turns=max(lengths) + 5, threads=2)
    obs = env.reset(seed=0)
    gen = torch.Generator().manual_seed(0)
    compared, t = 0, 0
    ended = [False] * n
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1).tolist()
        assert set(obs.step_idx[rows].tolist()) == {t}          # interleaved, lockstep
        acts = env.random_actions(obs, generator=gen)
        for r in rows:
            s = samples[r]
            if t >= lengths[r]:
                continue
            for k in ("board_seq", "globals_seq"):
                assert torch.equal(obs.obs[k][r], s["obs"][k][t]), (r, t, k)
            f, to = int(s["tgt"]["from_idx"][t]), int(s["tgt"]["to_idx"][t])
            assert torch.equal(obs.legal["from"][r].float(), s["tgt"]["legal_from"][t]), (r, t)
            assert torch.equal(obs.legal["to_cond"][r][f].float(), s["tgt"]["legal_to_of"][t])
            assert torch.equal(obs.legal["from"][r], obs.legal["to_cond"][r].any(-1))
            acts["from_idx"][r], acts["to_idx"][r] = f, to
            compared += 1
        obs, rew, done, info = env.step(acts)
        for r in rows:
            if t < lengths[r]:
                s = samples[r]
                assert (int(info["applied"]["from_idx"][r]), int(info["applied"]["to_idx"][r])) \
                    == (int(s["tgt"]["from_idx"][t]), int(s["tgt"]["to_idx"][t]))
            if t == lengths[r] - 1:
                terminal = not samples[r]["meta"].get("adjudicated", False)
                assert bool(done[r]) == terminal and not bool(info["truncated"][r]), r
                if terminal:
                    assert rew[r].tolist() == samples[r]["meta"]["outcome"], r
                    ended[r] = True
        t += 1
    assert compared == sum(lengths) == 2567
    assert sum(ended) == sum(not s["meta"].get("adjudicated", False) for s in samples) == 14
    for r, s in enumerate(samples):
        ev = [list(e) for e in env.trajectory(r).tolist()]
        assert ev[:lengths[r]] == s["meta"]["events"], r


def test_network_samples_legal_moves():
    from env.league import StreamStepper

    torch.manual_seed(0)
    net = build_game_network("xiangqi", **TINY).eval()
    env = GmSimVecEnv("xiangqi", 3, seed=1, max_turns=80)
    stepper = StreamStepper(net)
    obs = env.observe()
    steps = 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
        if restart.numel():
            stepper.reset(restart.tolist())
        with torch.no_grad():
            out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                               {k: v[rows] for k, v in obs.legal.items()},
                               obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
        f, to = out["action_from"].cpu(), out["action_to"].cpu()
        for j, r in enumerate(rows.tolist()):        # every sampled (from, to) is legal
            assert bool(obs.legal["to_cond"][r, f[j], to[j]])
        acts = {"action_from": torch.zeros(obs.n_rows, dtype=torch.long),
                "action_to": torch.zeros(obs.n_rows, dtype=torch.long)}
        acts["action_from"][rows], acts["action_to"][rows] = f, to
        obs, _, _, info = env.step(acts)
        assert bool(info["acted"][rows].all())
        steps += 1
    assert steps > 0 and bool(env.finished.all())


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------

def test_simulator_invariance_battery():
    """Causality over random playouts (the frames of decisions <= t never depend on
    later events); perfect information: the rules perturb nothing."""
    endgame = {"fen": "4k4/4a4/9/2p1p1p2/9/9/2P1P1P2/9/4A4/4K4 b - - 0 1"}   # short games
    st = invariance.check_sim_battery("xiangqi", endgame, n_games=2, seed=5)
    assert st["decisions"] > 50 and st["privacy"] == 0
    for seed in (1, 2):             # the opening and middlegame of full games (O(n^2) check)
        ev = gmsim.random_trajectory("xiangqi", {}, seed)[:120]
        assert invariance.check_sim_causality("xiangqi", {}, ev) == 120
    ev = gmsim.random_trajectory("xiangqi", {}, 7)
    assert all(gmsim.perturb_hidden("xiangqi", {}, ev, seat, 0) is None for seat in (0, 1))
    assert "terminal: 4" in gmsim.check_game("xiangqi", {}, 4, 3)


def _pgn_parts(raw: bytes):
    """(header lines, move tokens) of a CCPD PGN file."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("big5")
    head, _, moves = text.replace("\r\n", "\n").partition("\n\n")
    toks = [w for w in re.split(r"[\s　]+", moves)
            if w and not re.fullmatch(r"\d+\.|1-0|0-1|1/2-1/2|\*", w)]
    return [ln for ln in head.splitlines() if ln.startswith("[")], toks


def test_converter_causality_on_real_records(gmdata, tmp_path):
    """A real record and its truncations after every few plies ([Result "*"], converted
    with --allow-truncated): each truncation's frames and labels are a prefix of the
    full record's — the converter never lets a frame see later moves."""
    names = [M + "以棋手分類/呂欽/00000029.pgn",                   # a perpetual chase, cut
             "Dataset/對局/電腦對局/人機賽/00000002.pgn"]
    with zipfile.ZipFile(SAMPLE) as z:
        entries = {n.split("/", 1)[1]: n for n in z.namelist()}
        raws = [z.read(entries[n]) for n in names]
    calls = [0]

    def convert(records: List[str]) -> List[Dict]:
        calls[0] += 1
        d = tmp_path / f"c{calls[0]}"
        (d / "in").mkdir(parents=True)
        for i, text in enumerate(records):
            (d / "in" / f"{i:03d}.pgn").write_text(text, encoding="utf-8")
        _run(gmdata, "xiangqi", "convert", "--input", d / "in", "--out", d / "out",
             "--allow-truncated")
        return load_gm_samples(str(d / "out"))

    n = 0
    for raw in raws:
        head, toks = _pgn_parts(raw)
        head = [ln if not ln.startswith("[Result") else '[Result "*"]' for ln in head]
        full = "\n".join(head) + "\n\n" + " ".join(toks) + "\n"
        cuts = sorted({1, 2, 7, len(toks) // 2, len(toks) - 1})
        truncs = ["\n".join(head) + "\n\n" + " ".join(toks[:k]) + "\n" for k in cuts]
        n += invariance.check_converter_causality(convert, full, truncs)
    assert n == 10


# ---------------------------------------------------------------------------
# IL contract on real batches
# ---------------------------------------------------------------------------

def test_il_contract_on_real_batches(shards):
    from training.train_il_game import batch_loss, check_meta_arch, prepare_batch

    root, _, _ = shards
    ds = GameShardDataset(str(root / "all"), game="xiangqi", max_len=64, crop="prefix")
    torch.manual_seed(0)
    net = build_game_network("xiangqi", **TINY)
    check_meta_arch("xiangqi", net.cfg, ds.shard_meta)
    with pytest.raises(ValueError, match="piece_channels"):
        check_meta_arch("xiangqi", build_game_network("xiangqi", **TINY, piece_channels=14).cfg,
                        ds.shard_meta)
    batch = collate_game_sequences([ds[i] for i in range(4)])
    mb = prepare_batch("xiangqi", batch, CPU, tf_prob=1.0, cfg=net.cfg, il_legal_mask=True)
    assert set(mb["legal"]) == {"from", "to_of"} and set(mb["labels"]) == {"from_idx", "to_idx"}
    net.eval()
    pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"])
    dec = net.action_decoder
    mask = mb["frame_mask"]
    il = dec.il_loss(pred, mb["tgt"], mask)
    terms = dec.il_head_terms(pred, mb["tgt"], mask)
    assert set(terms) == {"from", "to"}
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    assert float(total.detach()) == pytest.approx(float(il["loss"].detach()), abs=1e-5)
    # the shards' IL masks make every recorded move legal, and teacher forcing commits
    # to the recorded from-square on every frame
    assert float(il["illegal_target_frac"]) == 0.0
    assert torch.equal(pred["from_idx"][mask], mb["tgt"]["from_idx"][mask])
    assert bool(terms["to"]["weight"][mask].eq(1).all())
    # soft targets as a Pikafish labeller would write them (sparse from / to heads):
    # a one-hot teacher on the recorded move = the hard objective exactly
    tgt = dict(mb["tgt"])
    pad = torch.full_like(tgt["from_idx"], -1)
    for h in ("from", "to"):
        tgt[f"{h}_probs_idx"] = torch.stack([tgt[f"{h}_idx"], pad, pad], -1)
        tgt[f"{h}_probs_val"] = torch.stack([torch.ones_like(tgt["value"]),
                                             torch.zeros_like(tgt["value"]),
                                             torch.zeros_like(tgt["value"])], -1)
    def run(t, alpha=1.0):
        net.zero_grad()
        loss, stats, _ = batch_loss(net, {**mb, "tgt": t}, CPU, 1.0, 0.0, False,
                                    soft_target_weight=alpha)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
        return float(loss.detach()), grad, stats

    hard, g_hard, _ = run(mb["tgt"])
    soft, g_soft, s_soft = run(tgt)
    assert soft == pytest.approx(hard, abs=1e-5) and torch.allclose(g_soft, g_hard, atol=1e-6)
    assert s_soft["from_teacher_frac"] == pytest.approx(1.0)
    assert s_soft["to_teacher_illegal_mass"] == 0.0
    # a genuine teacher (uniform over the legal destinations of the recorded from-square,
    # dense): a different gradient, no illegal mass, a finite KL
    to_of = mb["tgt"]["legal_to_of"]
    tgt2 = dict(mb["tgt"])
    tgt2["to_probs"] = to_of / to_of.sum(-1, keepdim=True).clamp(min=1)
    _, g2, s2 = run(tgt2)
    assert not torch.allclose(g2, g_hard, atol=1e-6)
    assert torch.isfinite(g2).all() and np.isfinite(s2["to_teacher_kl"])
    assert s2["to_teacher_illegal_mass"] == pytest.approx(0.0)
    assert s2["to_teacher_frac"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# IL with an existing game, then RL on both
# ---------------------------------------------------------------------------

def test_train_multi_game_then_rl_with_existing_games(shards, tmp_path):
    """IL on the real xiangqi shards together with an existing game (gomoku: the
    fixture gm-shards of the Gomocup converter) and kuhn_poker, then RL on xiangqi and
    kuhn_poker jointly (the gomoku env would only make the smoke slow)."""
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _, _ = shards
    torch.manual_seed(0)
    out = train_multi_game.main([
        "--data", f"xiangqi={root / 'all'}", "--data", f"gomoku={GOMOKU}",
        "--data", f"kuhn_poker={root / 'kuhn'}", "--val", f"xiangqi={root / 'hq'}",
        "--steps", "3", "--batch_size", "4", "--max_len", "48", "--val_max_len", "48",
        "--warmup_steps", "1", "--eval_every", "3", "--device", "cpu", "--il_legal_mask",
        "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    cx = ck["configs"]["xiangqi"]
    assert (cx["board_h"], cx["board_w"], cx["piece_channels"], cx["board_global_features"],
            cx["n_promotions"]) == (10, 9, 19, 7, 0)
    assert ck["configs"]["gomoku"]["board_h"] == 20
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "xiangqi",
            "--rl", "kuhn_poker", "--tokens_per_iter", "120", "--envs", "xiangqi=3",
            "--envs", "kuhn_poker=4", "--env_kw", "xiangqi:max_turns=40",
            "--mirror_prob", "xiangqi=0.5", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    for g in ("xiangqi", "kuhn_poker"):
        row = rec["games"][g]
        assert row["tokens"] >= 1 and row["episodes"] > 0
        assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
        for k in ("entropy", "kl", "value_loss"):
            assert np.isfinite(row[k]), (g, k)
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"xiangqi": 60, "kuhn_poker": 40}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["xiangqi"].streams:
        # FromToSpec stores the masks the rollout sampled under, [N] per frame
        assert set(s.legal) == {"from", "to_of"} and s.legal["to_of"].shape[1:] == (N,)
    for g in tr.rl_games:
        tr._recompute(g)
        for s in tr.rl[g].streams:
            assert torch.allclose(s.lp_old, s.mu, atol=1e-4), g
    # an env the network was not built for is refused (a 14-plane board network)
    with pytest.raises(SystemExit, match="board_seq"):
        tr.rl["xiangqi"].spec.check_env(make_vec_env("xiangqi", 1),
                                        build_game_network("xiangqi", **TINY, piece_channels=14))


def test_fetch_script_is_valid_bash():
    script = ROOT / "scripts" / "fetch_xiangqi.sh"
    assert script.exists() and os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "gmdata\" xiangqi fetch" in text or "xiangqi fetch" in text
    assert "--split train" in text and "--split val" in text
