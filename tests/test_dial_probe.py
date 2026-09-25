"""DIAL step-0 headroom probe: training/othello_probe.py + scripts/dial_probe.py.

Rules / frame semantics are pinned to the converter (golden digests of
`gmdata othello convert` on the WTHOR fixture), the solver to hand-checked
endgames and a pruning-free brute force, the search's perspective handling to
independent minimax references (stub values and the real network), and the
whole script to an end-to-end run on a tiny train_multi_game checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys

import numpy as np
import pytest
import torch

from data_processing.games.dataset import save_shard
from data_processing.games.othello import rules
from model.games import build_game_network
from training import othello_probe as op
from training.othello_probe import BLACK, PASS

sys.path.insert(0, os.path.dirname(__file__))
from golden_digests import sample_digests  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1, moe_experts=4)


def parse_wtb(data: bytes):
    """WTHOR records → move lists (mirrors gmdata's parse_wtb)."""
    n1 = struct.unpack_from("<I", data, 4)[0]
    assert len(data) == 16 + 68 * n1
    games = []
    for i in range(n1):
        off = 16 + 68 * i
        moves = []
        for byte in data[off + 8: off + 68]:
            if byte == 0:
                break
            moves.append((byte // 10 - 1) * 8 + byte % 10 - 1)
        games.append(moves)
    return games


_SAMPLES = None


def wthor_samples():
    """The fixture converted by the Python replay (same tensors as gmdata)."""
    global _SAMPLES
    if _SAMPLES is None:
        with open(os.path.join(FIX, "othello", "WTH_2021.wtb"), "rb") as fh:
            games = parse_wtb(fh.read())
        out = []
        for i, moves in enumerate(games):
            try:
                frames, b, w = op.replay_wthor(moves)
            except ValueError:
                continue
            obs, tgt = op.frames_to_tensors(frames, b, w)
            out.append({"obs": obs, "tgt": tgt, "meta": {
                "game_index": i, "replayed_black": b.bit_count(),
                "n_pass_frames": sum(f.action == PASS for f in frames)}})
        _SAMPLES = out
    return _SAMPLES


def board(*rows):
    """ASCII rows (row 0 first; B / W / .) → (black, white) bitboards."""
    b = w = 0
    for r, row in enumerate(rows):
        for c, ch in enumerate(row.replace(" ", "")):
            if ch == "B":
                b |= 1 << (r * 8 + c)
            elif ch == "W":
                w |= 1 << (r * 8 + c)
    return b, w


def random_positions(seed, n, min_empties=0, max_empties=60):
    """Decision positions (black, white, mover) from random playouts."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b, w, p = op.INITIAL_BLACK, op.INITIAL_WHITE, BLACK
        target = rng.randint(min_empties, max_empties)
        while not op.is_game_over(b, w):
            own, opp = op.sides(b, w, p)
            moves = list(op.iter_bits(op.legal_moves(own, opp)))
            if moves and op.n_empties(b, w) <= target:
                out.append((b, w, p))
                break
            if moves:
                b, w = op.colours(*op.play(own, opp, rng.choice(moves)), p)
            p = op.other(p)
    return out


# ---------------------------------------------------------------------------
# rules and frame semantics
# ---------------------------------------------------------------------------

def test_replay_reproduces_the_converter_bit_for_bit():
    """Python replay == gmdata othello convert on the WTHOR fixture: every
    ply from the side to move, forced pass frames, W/D/L value per mover."""
    with open(os.path.join(FIX, "golden", "othello.json")) as fh:
        golden = json.load(fh)
    samples = wthor_samples()
    got = sample_digests(samples, ("game_index", "replayed_black", "n_pass_frames"))
    assert len(got) == len(golden["samples"]) > 0
    assert got == golden["samples"]
    assert sum(s["meta"]["n_pass_frames"] for s in samples) > 0     # passes exercised


def test_bitboards_match_reference_rules_and_planes():
    rng = random.Random(0)
    n_pass = 0
    for _ in range(25):
        bd, player = rules.initial_board(), BLACK
        while not rules.is_game_over(bd):
            b, w = op.board_from_bytes(bd)
            own, opp = op.sides(b, w, player)
            ref = {r * 8 + c for r, c in rules.legal_moves(bd, player)}
            assert set(op.iter_bits(op.legal_moves(own, opp))) == ref
            assert np.array_equal(op.planes_batch([(own, opp)])[0],
                                  rules.board_planes(bd, player).numpy())
            if ref:
                sq = rng.choice(sorted(ref))
                want = sum(1 << i for i in rules.flips(bd, player, sq // 8, sq % 8))
                assert op.flips(own, opp, sq) == want
                rules.apply_move(bd, player, sq // 8, sq % 8)
            else:
                n_pass += 1
            player = op.other(player)
        assert op.is_game_over(*op.board_from_bytes(bd))
    assert n_pass > 0


def test_frames_from_sample_roundtrip_and_drift_detection():
    s = next(x for x in wthor_samples() if x["meta"]["n_pass_frames"])
    frames = op.frames_from_sample(s)
    assert [max(f.action, 0) for f in frames] == s["tgt"]["place_idx"].tolist()
    assert [f.action == PASS for f in frames] == s["tgt"]["passed"].tolist()
    # a side-to-move flip (the classic perspective bug) must be caught
    bad = {"obs": {"board_seq": s["obs"]["board_seq"].clone()}, "tgt": s["tgt"]}
    bad["obs"]["board_seq"][7] = bad["obs"]["board_seq"][7][[1, 0, 2]]
    with pytest.raises(ValueError, match="board_seq"):
        op.frames_from_sample(bad)


# ---------------------------------------------------------------------------
# exact solver
# ---------------------------------------------------------------------------

def brute(own, opp):
    """Pruning-free negamax (reference for the alpha-beta solver)."""
    moves = op.legal_moves(own, opp)
    if not moves:
        if not op.legal_moves(opp, own):
            return op.final_diff(own, opp)
        return -brute(opp, own)
    best = -65
    for sq in op.iter_bits(moves):
        o2, p2 = op.play(own, opp, sq)
        best = max(best, -brute(p2, o2))
    return best


def test_solver_hand_checked_endgames():
    # black's only move a1 flips the six whites of row 0: 8 black vs 56 white
    b, w = board(". W W W W W W B", *["W W W W W W W W"] * 7)
    assert op.solve(b, w) == -48
    assert op.exact_move_values(b, w) == {0: -48}
    # white to move has no move: forced pass, then the same line → +48
    assert op.legal_moves(w, b) == 0 and op.solve(w, b) == 48
    # finished games: empties go to the winner
    b, w = board("B . . . . . . .", *["........"] * 6, ". . . . . . W W")
    assert op.solve(b, w) == -62 and op.solve(w, b) == 62
    b, w = board("B . . . . . . .", *["........"] * 6, ". . . . . . . W")
    assert op.solve(b, w) == 0
    b, w = board(*["........"] * 3, "...BB...", *["........"] * 4)
    assert op.solve(b, w) == 64 and op.solve(w, b) == -64


def test_solver_matches_brute_force_including_passes():
    passes = 0
    for b, w, p in random_positions(1, 40, min_empties=4, max_empties=7):
        own, opp = op.sides(b, w, p)
        vals = op.exact_move_values(own, opp)
        want = {}
        for sq in op.iter_bits(op.legal_moves(own, opp)):
            o2, p2 = op.play(own, opp, sq)
            want[sq] = -brute(p2, o2)
            if not op.legal_moves(p2, o2) and op.legal_moves(o2, p2):
                passes += 1
        assert vals == want
        assert op.solve(own, opp) == max(want.values())
        # a narrow window still returns a correct bound (fail-soft)
        v = op.solve(own, opp, -1, 1)
        best = max(want.values())
        assert (v == best) if -1 < best < 1 else (v >= 1 if best >= 1 else v <= -1)
    assert passes > 0
    assert op.solve_positions([op.sides(b, w, p) for b, w, p in
                               random_positions(2, 3, 5, 6)], workers=2) == [
        op.exact_move_values(*op.sides(b, w, p)) for b, w, p in random_positions(2, 3, 5, 6)]


# ---------------------------------------------------------------------------
# lookahead search: perspective and pass handling
# ---------------------------------------------------------------------------

class StubEval:
    """feed() interface with histories as tuples of frame keys; the value of
    a frame is ``fn(planes)`` (the frame mover's perspective)."""

    def __init__(self, fn):
        self.fn = fn
        self.log = []

    def empty_ref(self):
        return ((), 0)

    def feed(self, refs, planes, r, keep_cache=True):
        new, vals = [], []
        for (hist, _), p in zip(refs, planes):
            h = hist + (p.tobytes(),)
            self.log.append(h)
            vals.append(self.fn(p))
            new.append((h, 0))
        logits = np.stack([self.logit(p) for p in planes]) if len(planes) else np.zeros((0, 64))
        return logits, np.array(vals, dtype=np.float64), (new if keep_cache else None)

    @staticmethod
    def logit(p):
        return np.sin(np.arange(64) * 0.37 + p[0].sum() * 1.3 + p[1].sum() * 0.7)


def disc_diff_value(p):
    return float(p[0].sum() - p[1].sum()) / 64.0


_W = np.random.RandomState(3).randn(2, 64)


def arbitrary_value(p):
    return float(np.tanh(_W[0] @ p[0].ravel() - 0.5 * _W[1] @ p[1].ravel()))


def ref_node(fn, b, w, to_move, depth, root):
    """Independent depth-limited minimax in the root mover's frame."""
    own, opp = op.sides(b, w, to_move)
    if not op.legal_moves(own, opp):
        if not op.legal_moves(opp, own):
            return op.outcome_sign(*op.sides(b, w, root))
        to_move = op.other(to_move)                      # forced pass: free
        own, opp = opp, own
    if depth == 0:
        v = fn(op.planes_batch([(own, opp)])[0])
        return v if to_move == root else -v
    vals = [ref_node(fn, *op.colours(*op.play(own, opp, sq), to_move),
                     op.other(to_move), depth - 1, root)
            for sq in op.iter_bits(op.legal_moves(own, opp))]
    return max(vals) if to_move == root else min(vals)


def ref_move_values(fn, b, w, mover, depth):
    own, opp = op.sides(b, w, mover)
    return {sq: ref_node(fn, *op.colours(*op.play(own, opp, sq), mover),
                         op.other(mover), depth - 1, mover)
            for sq in op.iter_bits(op.legal_moves(own, opp))}


def _forces_pass(b, w, p):
    own, opp = op.sides(b, w, p)
    for sq in op.iter_bits(op.legal_moves(own, opp)):
        o2, p2 = op.play(own, opp, sq)
        if not op.legal_moves(p2, o2) and op.legal_moves(o2, p2):
            return sq
    return None


def _pass_positions(n):
    out, seed = [], 100
    while len(out) < n:
        for b, w, p in random_positions(seed, 60, 2, 30):
            sq = _forces_pass(b, w, p)
            if sq is not None:
                out.append((b, w, p, sq))
        seed += 1
    return out[:n]


def test_la1_forced_pass_reads_the_movers_own_frame():
    b, w, p, sq = _pass_positions(1)[0]
    ev = StubEval(disc_diff_value)
    root_hist = (b"root",)
    res = op.search(ev, [op.Root(b, w, p, (root_hist, 0), np.zeros(64))], 1, r=1)
    (_, vals), = res
    own, opp = op.sides(b, w, p)
    for mv, v in vals.items():
        o2, p2 = op.play(own, opp, mv)
        if op.is_game_over(o2, p2):
            assert v == op.outcome_sign(o2, p2)
        else:
            # antisymmetric value: whoever's frame is read, the mover's value
            # must be its own disc lead — a sign slip on the pass path flips it
            assert v == pytest.approx((o2.bit_count() - p2.bit_count()) / 64)
    # the history behind the pass move holds the opponent's pass frame first,
    # then the mover's decision frame — exactly the training stream
    o2, p2 = op.play(own, opp, sq)
    want = (op.planes_batch([(p2, o2)])[0].tobytes(), op.planes_batch([(o2, p2)])[0].tobytes())
    assert root_hist + want in ev.log


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("fn", [disc_diff_value, arbitrary_value])
def test_search_equals_independent_minimax(fn, depth):
    pos = [x[:3] for x in _pass_positions(3)] + random_positions(7, 6, 0, 50)
    ev = StubEval(fn)
    roots = [op.Root(b, w, p, ((b"r%d" % i,), 0), StubEval.logit(op.planes_batch(
        [op.sides(b, w, p)])[0])) for i, (b, w, p) in enumerate(pos)]
    res = op.search(ev, roots, depth, r=1)
    for (b, w, p), rt, (mv, vals) in zip(pos, roots, res):
        want = ref_move_values(fn, b, w, p, depth)
        assert vals.keys() == want.keys()
        for k in want:
            assert vals[k] == pytest.approx(want[k], abs=1e-12)
        best = max(want.values())
        tied = [k for k in want if abs(want[k] - best) < 1e-12]
        assert mv == max(tied, key=lambda k: rt.logits[k])     # tie-break by policy logit


def test_search_tie_break_prefers_policy_logit():
    # two mirror-image moves (each forces white to pass) with equal values:
    # only the root policy logit decides
    b, w = board(". W W W W W W B", *["W W W W W W W W"] * 6, ". W W W W W W B")
    ev = StubEval(disc_diff_value)
    for fav in (0, 56):
        logits = np.zeros(64)
        logits[fav] = 1.0
        (mv, vals), = op.search(ev, [op.Root(b, w, BLACK, (("x",), 0), logits)], 1, 1)
        assert set(vals) == {0, 56} and vals[0] == vals[56]
        assert mv == fav


def expected_history(actions):
    """Frame keys a training sample of the game with these actions holds:
    one frame per action (forced passes included), side-to-move planes."""
    b, w, p = op.INITIAL_BLACK, op.INITIAL_WHITE, BLACK
    keys = []
    for a in actions:
        own, opp = op.sides(b, w, p)
        keys.append(op.planes_batch([(own, opp)])[0].tobytes())
        if a == PASS:
            assert not op.legal_moves(own, opp) and op.legal_moves(opp, own)
        else:
            b, w = op.colours(*op.play(own, opp, a), p)
        p = op.other(p)
    assert op.is_game_over(b, w)
    return tuple(keys)


def test_play_games_streams_training_frames():
    ev = StubEval(disc_diff_value)
    ops = [tuple(f.action for f in op.frames_from_sample(s)[:4]) for s in wthor_samples()[:40]]
    raw, la1 = op.PlayerSpec(0, 1), op.PlayerSpec(1, 1)
    res = op.play_games(ev, [(o, raw, la1) for o in ops], parallel=7)
    log = set(ev.log)
    for r, o in zip(res, ops):
        assert r["actions"][:4] == list(o)
        assert expected_history(r["actions"]) in log
        assert r["black"] + r["white"] <= 64 and r["frames"] == len(r["actions"])
    assert any(PASS in r["actions"] for r in res)          # the pass path ran


# ---------------------------------------------------------------------------
# the real network: batched / cloned histories == sequential streaming
# ---------------------------------------------------------------------------

def _tiny_net(mixer):
    torch.manual_seed(0)
    return build_game_network("othello", **TINY, temporal_mixer=mixer).eval()


def _step1(net, cache, t, own_opp, r):
    obs = {"board_seq": torch.from_numpy(op.planes_batch([own_opp]))}
    with torch.no_grad():
        out = net.step(obs, st_cache=cache, step_idx=t, act_steps=r)
    return out["st_cache"], t + 1, float(out["value"][0])


def net_ref_node(net, r, b, w, to_move, depth, root, cache, t):
    own, opp = op.sides(b, w, to_move)
    if not op.legal_moves(own, opp):
        if not op.legal_moves(opp, own):
            return op.outcome_sign(*op.sides(b, w, root))
        cache, t, _ = _step1(net, cache, t, (own, opp), r)        # pass frame
        to_move, own, opp = op.other(to_move), opp, own
    cache, t, v = _step1(net, cache, t, (own, opp), r)
    if depth == 0:
        return v if to_move == root else -v
    vals = [net_ref_node(net, r, *op.colours(*op.play(own, opp, sq), to_move),
                         op.other(to_move), depth - 1, root, cache, t)
            for sq in op.iter_bits(op.legal_moves(own, opp))]
    return max(vals) if to_move == root else min(vals)


@pytest.mark.parametrize("mixer", ["recurrent", "attention"])
def test_net_lookahead_matches_sequential_streaming(mixer):
    net = _tiny_net(mixer)
    s = next(x for x in wthor_samples() if x["meta"]["n_pass_frames"])
    frames = op.frames_from_sample(s)
    # root: the decision right before the first pass frame (its move forces a pass)
    k = next(t for t, f in enumerate(frames) if f.action == PASS) - 1
    r = 3
    ev = op.NetEvaluator(net, torch.device("cpu"), max_batch=4)
    ref = ev.empty_ref()
    cache, t = None, 0
    for f in frames[:k + 1]:
        logits, _, new = ev.feed([ref], op.planes_batch([f.own_opp()]), r)
        ref = new[0]
        cache, t, _ = _step1(net, cache, t, f.own_opp(), r)
    root = frames[k]
    own, opp = root.own_opp()
    for depth in (1, 2):
        (_, vals), = op.search(ev, [op.Root(root.black, root.white, root.mover, ref,
                                            logits[0])], depth, r)
        for sq, v in vals.items():
            b2, w2 = op.colours(*op.play(own, opp, sq), root.mover)
            want = net_ref_node(net, r, b2, w2, op.other(root.mover), depth - 1,
                                root.mover, cache, t)
            assert v == pytest.approx(want, abs=1e-5), (depth, sq)


def test_evaluator_gathers_rows_across_blocks():
    """Histories referenced from several cache blocks (e.g. games fed in
    different r groups) are gathered row-exactly and continue correctly."""
    net = _tiny_net("recurrent")
    ev = op.NetEvaluator(net, torch.device("cpu"), max_batch=2)
    seq = wthor_samples()[0]["obs"]["board_seq"].numpy()
    _, _, a = ev.feed([ev.empty_ref()] * 3, seq[0:3], 1)
    _, _, b = ev.feed([ev.empty_ref()] * 2, seq[3:5], 2)
    refs = [b[1], a[2], a[0], b[0]]
    got = ev._gather(refs)
    S = ev.S
    for j, (blk, row) in enumerate(refs):
        for g, c in zip(got, blk.cache):
            assert torch.equal(g[j * S:(j + 1) * S], c[row * S:(row + 1) * S])
    nxt = seq[10:14]
    lg, v, _ = ev.feed(refs, nxt, 2)
    for j, ref in enumerate(refs):
        lg1, v1, _ = ev.feed([ref], nxt[j:j + 1], 2)
        assert np.allclose(v[j], v1[0], atol=1e-5) and np.allclose(lg[j], lg1[0], atol=1e-5)


def test_history_is_depth_independent_and_matches_training_forward():
    """The Stage-1 cache does not depend on r (players with different r can
    share one stream), and streaming a game reproduces the full-sequence
    training forward."""
    net = _tiny_net("recurrent")
    s = wthor_samples()[1]
    seq = s["obs"]["board_seq"]
    ev = op.NetEvaluator(net, torch.device("cpu"))
    ref = ev.empty_ref()
    vals, logits = [], []
    for t in range(seq.shape[0]):
        a = ev.feed([ref], seq[t:t + 1].numpy(), 1)
        b = ev.feed([ref], seq[t:t + 1].numpy(), 3)
        for ca, cb in zip(a[2][0][0].cache, b[2][0][0].cache):
            assert torch.equal(ca, cb)
        vals.append(b[1][0])
        logits.append(b[0][0])
        ref = a[2][0]
    with torch.no_grad():
        out = net({"board_seq": seq[None]}, act_steps=3)
    assert np.allclose(np.array(vals), out["value"][0].numpy(), atol=1e-4)
    assert np.allclose(np.stack(logits), out["place_logits"][0].numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# match statistics and the end-to-end script
# ---------------------------------------------------------------------------

def test_match_stats():
    st = op.match_stats([(1, 1), (1, 0), (0.5, 0.5), (0, 1)], seed=0)
    assert (st["W"], st["D"], st["L"], st["games"], st["pairs"]) == (4, 2, 2, 8, 4)
    assert st["score"] == pytest.approx(0.625)
    assert st["elo"] == pytest.approx(400 * np.log10(0.625 / 0.375))
    lo, hi = st["elo_ci95_normal"]
    assert lo < st["elo"] < hi
    blo, bhi = st["elo_ci95_bootstrap"]
    assert blo <= st["elo"] <= bhi
    sweep = op.match_stats([(1, 1)] * 5)
    assert np.isfinite(sweep["elo"]) and sweep["elo"] > 0


def test_dial_probe_end_to_end(tmp_path):
    from scripts import dial_probe
    from training import train_multi_game

    samples = wthor_samples()
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    save_shard(str(tmp_path / "train" / "a.pt"), "othello", samples[:8])
    val = [s for s in samples[8:80] if s["meta"]["n_pass_frames"]][:3] + samples[8:14]
    save_shard(str(tmp_path / "val" / "a.pt"), "othello", val)
    core = [f"{k}={v}" for k, v in TINY.items()] + ["temporal_mixer=recurrent"]
    train_multi_game.main([
        "--data", f"othello={tmp_path / 'train'}", "--steps", "2", "--batch_size", "2",
        "--warmup_steps", "1", "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
        "--override", *core])
    out = tmp_path / "probe.json"
    res = dial_probe.main([
        "--ckpt", str(tmp_path / "ck" / "multi_final.pt"), "--val_dir", str(tmp_path / "val"),
        "--pairs", "2", "--opening_plies", "6", "--depths", "1", "2", "--empties", "6",
        "--max_positions", "15", "--device", "cpu", "--solver_workers", "1",
        "--parallel", "3", "--json", str(out)])
    with open(out) as fh:
        disk = json.load(fh)
    assert disk["gate"]["match"] == "la1@2 vs raw@2"
    assert [m["a"] for m in disk["matches"]] == ["la1@2", "la2@2"]
    for m in disk["matches"]:
        assert m["games"] == 4 and m["W"] + m["D"] + m["L"] == 4
        assert len(m["per_game"]) == 4 and m["elo_ci95_normal"] is not None
        for g in m["per_game"]:
            assert g["frames"] >= 6 and g["black"] + g["white"] <= 64
    # seat swap: the same opening is played with A as black, then as white
    pg = disk["matches"][0]["per_game"]
    assert pg[0]["actions"][:6] == pg[1]["actions"][:6]
    assert pg[0]["a_black"] and not pg[1]["a_black"]
    for part in ("endgame", "human"):
        assert set(disk[part]["per_depth"]) == {"1", "2"}
    e = disk["endgame"]["per_depth"]["2"]
    assert e["n"] == disk["endgame"]["n_positions"] == 15
    assert 0 <= e["raw_best"] <= 1 and e["raw_regret"] >= 0 and e["la1_regret"] >= 0
    h = disk["human"]["per_depth"]["1"]
    assert h["n"] == 15 and 0 <= h["raw_top1"] <= 1 and 0 <= h["la1_top1"] <= 1
    assert res["skipped_val_games"] == []
