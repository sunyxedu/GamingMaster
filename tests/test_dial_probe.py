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


def overshoot_value(p):
    """Unbounded like the real ValueHead: confident frames exceed |V| = 1."""
    return 1.3 * arbitrary_value(p)


def ref_node(fn, b, w, to_move, depth, root):
    """Independent depth-limited minimax in the root mover's frame; returns
    (exactness class, value): an exact win beats every estimate, an exact
    loss loses to every estimate."""
    own, opp = op.sides(b, w, to_move)
    if not op.legal_moves(own, opp):
        if not op.legal_moves(opp, own):
            z = op.outcome_sign(*op.sides(b, w, root))
            return (z, z)
        to_move = op.other(to_move)                      # forced pass: free
        own, opp = opp, own
    if depth == 0:
        v = fn(op.planes_batch([(own, opp)])[0])
        return (0.0, v if to_move == root else -v)
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
    vals = res[0].values
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
@pytest.mark.parametrize("fn", [disc_diff_value, arbitrary_value, overshoot_value])
def test_search_equals_independent_minimax(fn, depth):
    pos = ([x[:3] for x in _pass_positions(3)] + random_positions(7, 6, 0, 50)
           + random_positions(8, 12, 0, 3))                  # terminals within reach
    ev = StubEval(fn)
    roots = [op.Root(b, w, p, ((b"r%d" % i,), 0), StubEval.logit(op.planes_batch(
        [op.sides(b, w, p)])[0])) for i, (b, w, p) in enumerate(pos)]
    res = op.search(ev, roots, depth, r=1)
    n_terminal = 0
    for (b, w, p), rt, (mv, vals, term) in zip(pos, roots, res):
        want = ref_move_values(fn, b, w, p, depth)
        assert vals.keys() == want.keys()
        for k in want:
            assert vals[k] == pytest.approx(want[k][1], abs=1e-12)
        best = max(want.values())
        tied = [k for k in want if want[k][0] == best[0] and abs(want[k][1] - best[1]) < 1e-12]
        assert mv == max(tied, key=lambda k: rt.logits[k])     # tie-break by policy logit
        n_terminal += term
    assert n_terminal > 0


def _stub_const(v):
    """Evaluator whose every frame has value ``v`` (for the frame's mover)."""
    return StubEval(lambda p: v)


@pytest.mark.parametrize("v", [-1.0, -1.001, -1.05])
def test_exact_win_beats_an_overshooting_estimate(v):
    """Review regression: white to move, 57 ends the game in a win (+10),
    56 loses (-2).  With the opponent's frame after 56 read at V <= -1 (so
    >= +1 for white) the search must still take the certain win."""
    b, w = 0x3f03072f5d373f, 0xfcc0fcf8d0a2c8c0
    own, opp = op.sides(b, w, op.WHITE)
    assert op.exact_move_values(own, opp) == {56: -2, 57: 10}
    o2, p2 = op.play(own, opp, 57)
    assert op.is_game_over(o2, p2) and op.outcome_sign(o2, p2) == 1.0
    logits = np.zeros(64)
    logits[56] = 5.0                                         # the policy prefers the loss
    (mv, vals, term), = op.search(_stub_const(v), [op.Root(b, w, op.WHITE, ((), 0), logits)],
                                  1, 1)
    assert vals[57] == 1.0 and vals[56] == pytest.approx(-v)
    assert mv == 57 and term


def test_estimate_beats_an_exact_loss_and_compares_with_an_exact_draw():
    """Play on at V = -1.05 rather than end the game in a certain loss; an
    exact draw compares with estimates by value (a tie goes to the logit).
    Sparse boards, black to move: one live move and one game-ending move."""
    b, w, live, dead = 0x80010, 0x8000400000015080, 5, 20      # 20 ends in a loss
    kinds = {sq: op.is_game_over(*op.play(b, w, sq)) for sq in op.iter_bits(op.legal_moves(b, w))}
    assert kinds == {live: False, dead: True}
    assert op.outcome_sign(*op.play(b, w, dead)) == -1.0
    logits = np.zeros(64)
    logits[dead] = 5.0
    for v in (1.05, 1.0, -1.05):                               # live move scores -1.05 / -1 / ..
        (mv, vals, _), = op.search(_stub_const(v), [op.Root(b, w, BLACK, ((), 0), logits)], 1, 1)
        assert vals[dead] == -1.0 and mv == live

    b, w, live, dead = 0x800000600000, 0x200000028000508, 36, 37    # 37 ends in a draw
    kinds = {sq: op.is_game_over(*op.play(b, w, sq)) for sq in op.iter_bits(op.legal_moves(b, w))}
    assert kinds == {live: False, dead: True}
    assert op.outcome_sign(*op.play(b, w, dead)) == 0.0
    seen = set()
    for v in (0.2, -0.2):
        (mv, vals, _), = op.search(_stub_const(v), [op.Root(b, w, BLACK, ((), 0), np.zeros(64))],
                                   1, 1)
        assert vals[dead] == 0.0 and abs(vals[live]) == pytest.approx(0.2)
        assert mv == (live if vals[live] > 0 else dead)
        seen.add(mv)
    assert seen == {live, dead}
    for fav in (live, dead):                                   # exact tie: policy logit
        logits = np.zeros(64)
        logits[fav] = 1.0
        (mv, _, _), = op.search(_stub_const(0.0), [op.Root(b, w, BLACK, ((), 0), logits)], 1, 1)
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
        (_, vals, _), = op.search(ev, [op.Root(root.black, root.white, root.mover, ref,
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


def test_bf16_keeps_value_head_and_decoder_in_fp32():
    """Under bf16 autocast the value head / decoder run in fp32 (a bf16 value
    has 8 significant bits: artificial lookahead ties); the overrides are
    removed afterwards and the parameters untouched."""
    net = _tiny_net("recurrent")
    seq = wthor_samples()[0]["obs"]["board_seq"].numpy()
    dtypes = []
    hook = net.value_head.register_forward_hook(lambda m, i, o: dtypes.append(o.dtype))
    obs = {"board_seq": torch.from_numpy(seq[:4])}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        plain = net.step(obs, act_steps=2)
    assert dtypes[-1] == torch.bfloat16 and plain["value"].dtype == torch.bfloat16
    ev16 = op.NetEvaluator(net, torch.device("cpu"), bf16=True)
    lg16, v16, _ = ev16.feed([ev16.empty_ref()] * 4, seq[:4], 2)
    assert dtypes[-1] == torch.float32
    hook.remove()
    assert "forward" not in net.value_head.__dict__ and "decode" not in net.action_decoder.__dict__
    ev32 = op.NetEvaluator(net, torch.device("cpu"))
    lg32, v32, _ = ev32.feed([ev32.empty_ref()] * 4, seq[:4], 2)
    assert np.allclose(v16, v32, atol=0.1) and np.allclose(lg16, lg32, atol=0.2)
    assert not np.array_equal(v16, v32)                     # the trunk did run in bf16


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


def test_gate_verdict_is_three_way():
    def st(elo, lo, hi, pairs=200):
        return {"elo": elo, "pairs": pairs, "elo_ci95_normal": [lo, hi],
                "elo_ci95_bootstrap": [lo + 1, hi - 1]}
    assert op.gate_verdict(st(45, 12, 78), 30)["verdict"] == "GO"
    assert op.gate_verdict(st(45, -3, 93), 30)["verdict"] == "INCONCLUSIVE"   # CI hits 0
    assert op.gate_verdict(st(25, 2, 48), 30)["verdict"] == "INCONCLUSIVE"    # point < 30
    assert op.gate_verdict(st(5, -20, 29), 30)["verdict"] == "NO-GO"
    assert op.gate_verdict(st(-40, -70, -10), 30)["verdict"] == "NO-GO"
    assert op.gate_verdict(st(45, 12, 78, pairs=20), 30)["verdict"] == "INCONCLUSIVE"
    g = op.gate_verdict(st(45, 12, 78), 30)
    assert g["go"] and g["ci95"] == [12, 78] and g["pairs"] == 200     # union of the CIs
    assert op.gate_verdict({"elo": 50.0, "pairs": 1, "elo_ci95_normal": None,
                            "elo_ci95_bootstrap": None}, 30)["verdict"] == "INCONCLUSIVE"


def test_cluster_mean_ci_resamples_games():
    groups = [0, 0, 0, 1, 2, 2]
    vals = [1.0, 1.0, np.nan, 0.0, 1.0, 0.0]
    r = op.cluster_mean_ci(groups, vals, n_boot=500)
    assert (r["n"], r["games"]) == (5, 3) and r["mean"] == pytest.approx(3 / 5)
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1]
    assert op.cluster_mean_ci([0, 0], [1.0, 0.0])["ci95"] is None     # one game: no CI
    assert op.cluster_mean_ci([0], [np.nan])["n"] == 0
    # 100 rows from 2 games are 2 units, not 100: the CI stays wide
    wide = op.cluster_mean_ci([0] * 50 + [1] * 50, [1.0] * 50 + [0.0] * 50, n_boot=500)
    assert wide["ci95"][1] - wide["ci95"][0] > 0.5


def test_openings_dedupe_positions_and_raise_plies():
    from scripts.dial_probe import collect_openings
    samples = wthor_samples()[:300]
    order = list(range(len(samples)))
    seqs, boards = set(), set()
    for x in samples:
        fr = op.frames_from_sample(x)
        if len(fr) > 6:
            seqs.add(tuple(f.action for f in fr[:6]))
            boards.add((fr[6].black, fr[6].white, fr[6].mover))
    assert len(boards) < len(seqs)                       # the fixture has transpositions
    ops, plies = collect_openings(samples, order, 6, 10 ** 6, {}, 6)
    assert plies == 6 and len(ops) == len(boards)
    # not enough 6-ply positions: the smallest sufficient even ply count is used
    want = len(boards) + 5
    ops, plies = collect_openings(samples, order, 6, want, {}, 16)
    assert plies > 6 and len(ops) == want
    keys = set()
    for o in ops:
        b, w, p = op.INITIAL_BLACK, op.INITIAL_WHITE, BLACK
        for a in o:
            if a != PASS:
                b, w = op.colours(*op.play(*op.sides(b, w, p), a), p)
            p = op.other(p)
        keys.add((b, w, p))
    assert len(keys) == want and all(len(o) == plies for o in ops)
    ops2, plies2 = collect_openings(samples, order, 6, want, {}, plies - 2)
    assert plies2 == plies - 2 and len(ops2) < want      # capped: the most there are


def test_select_positions_spreads_and_skips_forced_moves():
    from scripts.dial_probe import select_positions
    samples = wthor_samples()[:120]
    sel = select_positions(samples, range(len(samples)), 10, 100, {}, human_stride=4,
                           human_per_game=3, endgame_per_game=2, seed=1)
    n_end = sum(len(e) for _, _, e in sel)
    n_hum = sum(len(h) for _, h, _ in sel)
    assert n_end == 100 and n_hum == 100
    assert sum(1 for _, _, e in sel if e) >= 50          # spread over >= 50 games
    for g, hum, end in sel:
        assert len(hum) <= 3 and len(end) <= 2
        for t in hum | end:
            f = g.frames[t]
            assert f.action != PASS and op.legal_moves(*f.own_opp()).bit_count() >= 2
        for t in end:
            assert op.n_empties(g.frames[t].black, g.frames[t].white) <= 10
        for t in hum:
            assert (t + g.index) % 4 == 0


def test_skipped_validation_games_counted_once():
    from scripts.dial_probe import collect_openings, select_positions
    samples = list(wthor_samples()[:30])
    bad = dict(samples[3])
    bad["obs"] = {"board_seq": samples[3]["obs"]["board_seq"].clone()}
    bad["obs"]["board_seq"][5] = bad["obs"]["board_seq"][5][[1, 0, 2]]
    samples[3] = bad
    skipped = {}
    collect_openings(samples, range(30), 6, 10 ** 6, skipped, 6)
    select_positions(samples, range(30), 10, 10 ** 6, skipped)
    assert list(skipped) == [3]


def test_play_games_search_stats_and_solver_endgame():
    ev = StubEval(disc_diff_value)
    ops = [tuple(f.action for f in op.frames_from_sample(s)[:4]) for s in wthor_samples()[:12]]
    raw, la1 = op.PlayerSpec(0, 1), op.PlayerSpec(1, 1)
    res = op.play_games(ev, [(o, la1, raw) for o in ops], parallel=5)
    tot = {k: sum(r["stats"]["black"][k] for r in res) for k in op._STATS0}
    assert all(r["stats"]["white"]["searched"] == 0 for r in res)        # raw never searches
    assert tot["searched"] > 0 and tot["terminal"] > 0 and tot["solved"] == 0
    assert tot["terminal_changed"] <= tot["terminal"] <= tot["searched"]
    # solver endgame: every decision at <= 6 empties plays an exactly optimal move
    res = op.play_games(ev, [(o, la1, raw) for o in ops], parallel=5, solve_empties=6)
    n_solved = 0
    for r in res:
        n_solved += r["stats"]["black"]["solved"] + r["stats"]["white"]["solved"]
        b, w, p = op.INITIAL_BLACK, op.INITIAL_WHITE, BLACK
        for a in r["actions"]:
            own, opp = op.sides(b, w, p)
            if a != PASS:
                if op.n_empties(b, w) <= 6:
                    ex = op.exact_move_values(own, opp)
                    assert ex[a] == max(ex.values())
                b, w = op.colours(*op.play(own, opp, a), p)
            p = op.other(p)
    assert n_solved > 0


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
        "--max_positions", "15", "--endgame_per_game", "3", "--device", "cpu",
        "--solver_workers", "1", "--parallel", "3", "--json", str(out)])
    with open(out) as fh:
        disk = json.load(fh)
    assert disk["gate"]["match"] == "la1@2 vs raw@2"
    # 2 pairs < --gate_min_pairs: never a GO / NO-GO on a degenerate CI
    assert disk["gate"]["verdict"] == "INCONCLUSIVE" and disk["gate"]["pairs"] == 2
    assert disk["gate"]["opening_plies"] == disk["opening_plies"] == 6
    assert [m["a"] for m in disk["matches"]] == ["la1@2", "la2@2"]
    for m in disk["matches"]:
        assert m["games"] == 4 and m["W"] + m["D"] + m["L"] == 4
        assert len(m["per_game"]) == 4 and m["elo_ci95_normal"] is not None
        assert m["search"]["a"]["searched"] > 0 and m["search"]["b"].get("searched", 0) == 0
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
    assert e["games"] == disk["endgame"]["n_games"] >= 15 / 3
    assert 0 <= e["raw_best"] <= 1 and e["raw_regret"] >= 0 and e["la1_regret"] >= 0
    assert e["n_contested"] <= e["n"] and e["n_critical"] <= e["n"]
    assert set(e["la1_minus_raw"]) == {"best_contested", "regret_contested", "blunder_critical"}
    h = disk["human"]["per_depth"]["1"]
    assert h["n"] == 15 and 0 <= h["raw_top1"] <= 1 and 0 <= h["la1_top1"] <= 1
    assert h["la1_minus_raw_top1"]["n"] == 15
    assert res["skipped_val_games"] == []
