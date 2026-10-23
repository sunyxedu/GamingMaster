"""scripts/analyze_strength.py: archived evaluation episodes replayed, every learner decision judged
by a full-strength engine (loss against the engine's best move).  Engine-backed parts run only
where the engines are installed (~/gm-engines)."""
import json
import os

import numpy as np
import pytest

from scripts.analyze_strength import KataGoAnalysis, decisions, judge_file, make_engine, sample
from training.train_rl_joint import archive_eval_wave

STOCKFISH = os.path.expanduser("~/gm-engines/bin/stockfish")
KATAGO = os.path.expanduser("~/gm-engines/katago/v1.18.1-eigenavx2")
B10 = os.path.expanduser("~/gm-engines/nets/katago/kata1-b10c128-s1141046784-d204142634.txt.gz")


def _random_games(game, n, plies, seed=1):
    import gmsim._native as gm
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        st, traj = gm.State(game), []
        for ply in range(plies):
            seat = ply % 2
            legal = list(st.legal_actions(seat))
            if not legal:
                break
            a = int(rng.choice(legal))
            traj.append([seat, a])
            st.apply_action(seat, a)
        out.append(np.array(traj, dtype=np.int64))
    return out


class _Env:
    def __init__(self, trajs):
        self.trajs, self.num_envs = trajs, len(trajs)

    def trajectory(self, e):
        return self.trajs[e]


def test_decisions_are_the_learner_seats_moves_in_order():
    traj = _random_games("chess", 1, 6)[0]
    both = decisions("chess", traj, [True, True])
    white = decisions("chess", traj, [True, False])
    assert [t for t, _, _ in both] == [0, 1, 2, 3, 4, 5]
    assert [t for t, _, _ in white] == [0, 2, 4]
    assert all(len(before) == t for t, before, _ in both)
    assert sample(list(range(10)), 4) != list(range(10)) and len(sample(list(range(10)), 4)) == 4
    assert sample([1, 2], 5) == [1, 2]


@pytest.mark.skipif(not os.path.exists(STOCKFISH), reason="stockfish not installed")
def test_chess_loss_against_stockfish(tmp_path):
    trajs = _random_games("chess", 2, 10)
    path = str(tmp_path / "traces" / "eval_chess_init_it3_r0_w0.npz")
    archive_eval_wave(path, _Env(trajs), [[True, False], [False, True]],
                      {"game": "chess", "opponent": "init", "iteration": 3, "rank": 0, "wave": 0})
    eng = make_engine("chess", STOCKFISH, "go depth 6", 1)
    try:
        rec = judge_file(path, eng, "chess", "go depth 6", 8)
    finally:
        eng.close()
    assert rec["positions"] == 8 and rec["iteration"] == 3 and rec["engine"].startswith("Stockfish")
    assert 0.0 <= rec["mean_loss"] <= 1000.0 and 0.0 <= rec["agree"] <= 1.0
    assert rec["mean_loss"] > 0.0          # random moves lose centipawns
    # against the --init network: the other seats are the init network, judged in the same games
    assert rec["init_positions"] == 8 and rec["init_mean_loss"] > 0.0 and 0.0 <= rec["init_agree"] <= 1.0
    path2 = str(tmp_path / "traces" / "eval_chess_engine_it3_r0_w0.npz")
    archive_eval_wave(path2, _Env(trajs), [[True, False], [False, True]],
                      {"game": "chess", "opponent": "engine", "iteration": 3})
    eng = make_engine("chess", STOCKFISH, "go depth 6", 1)
    try:
        rec2 = judge_file(path2, eng, "chess", "go depth 6", 8)
    finally:
        eng.close()
    assert rec2["positions"] == 8 and "init_positions" not in rec2     # the referee is not judged


@pytest.mark.skipif(not (os.path.exists(KATAGO) and os.path.exists(B10)), reason="katago not installed")
def test_go_loss_against_katago(tmp_path):
    cmd = (f"{KATAGO}/katago analysis -model {B10} -config {KATAGO}/analysis_example.cfg "
           f"-override-config numAnalysisThreads=1,numSearchThreadsPerAnalysisThread=1,reportAnalysisWinratesAs=SIDETOMOVE,logDir=")
    eng = KataGoAnalysis(cmd, visits=2)
    try:
        wr = eng.winrates(["Q16", "D4", "A1"], [2, 3])
        assert set(wr) == {2, 3} and wr[3] > 0.6          # B to move after W's A1: well ahead
        trajs = _random_games("go", 1, 6, seed=2)
        path = str(tmp_path / "traces" / "eval_go_init_it3_r0_w0.npz")
        archive_eval_wave(path, _Env(trajs), [[True, False]], {"game": "go", "opponent": "init", "iteration": 3})
        rec = judge_file(path, eng, "go", "visits 2", 3)
    finally:
        eng.close()
    assert rec["positions"] == 3 and 0.0 <= rec["mean_loss"] <= 100.0 and rec["agree"] is None
    assert rec["init_positions"] == 3 and 0.0 <= rec["init_mean_loss"] <= 100.0


def test_an_engine_that_does_not_start_skips_its_game(tmp_path, capsys):
    """main(): a game whose engine does not start (an AVX2 build on an SSE2-only node exits at
    once) is skipped with a message — its files stay unjudged for the next run — and the other
    games are still judged."""
    from scripts import analyze_strength

    run = tmp_path / "run"
    for game, plies in (("chess", 6), ("xiangqi", 6)):
        archive_eval_wave(str(run / "traces" / f"eval_{game}_init_it3_r0_w0.npz"),
                          _Env(_random_games(game, 1, plies)), [[True, False]],
                          {"game": game, "opponent": "init", "iteration": 3, "rank": 0, "wave": 0})
    calls = []

    class FakeEngine:
        name = "fake"

        def judge(self, start, moves, played, go):
            calls.append(played)
            return 10.0, 0.0, False

        def close(self):
            pass

    def fake_make(game, command, go, threads):
        if game == "xiangqi":
            raise analyze_strength.EngineError("pikafish: no 'uciok' (exited -4)")
        return FakeEngine()

    orig = analyze_strength.make_engine
    analyze_strength.make_engine = fake_make
    try:
        analyze_strength.sys.argv = ["analyze_strength.py", "--run", str(run), "--engine", "chess=x",
                                     "--engine", "xiangqi=y", "--go", "chess=go depth 1"]
        analyze_strength.main()
    finally:
        analyze_strength.make_engine = orig
    rows = [json.loads(l) for l in open(run / "strength.jsonl")]
    assert [r["game"] for r in rows] == ["chess"] and calls
    assert "xiangqi: the engine does not start" in capsys.readouterr().err


def test_archive_names_keep_multi_word_games():
    """archive_game: chess_variants' archives are chess_variants, not chess (the chess engine
    must not judge them)."""
    from scripts.analyze_strength import archive_game

    assert archive_game("eval_chess_init_it40_r0_w2.npz") == "chess"
    assert archive_game("eval_chess_variants_init_it40_r0_w0.npz") == "chess_variants"
    assert archive_game("eval_clash_royale_engine_it7_r1_w12.npz") == "clash_royale"
    assert archive_game("hunl_it25.json") is None
