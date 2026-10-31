"""--timeout_reward: a time-out never scores better than a loss (train_rl_joint.apply_timeout)."""

import torch

from training.train_rl_joint import TIMEOUT_IS_DRAW, TIMEOUT_REWARD, apply_timeout, parse_args


def test_truncated_episodes_score_the_timeout_reward_for_every_seat():
    rew = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [0.3, 0.1]])
    done = torch.tensor([True, True, False, True])
    info = {"truncated": torch.tensor([True, False, True, False]), "winner": torch.tensor([-1, 0, -1, -1])}
    out, hit = apply_timeout(rew, done, info, -1.0)
    assert hit.tolist() == [True, False, False, False]      # a truncated row that is not done: no
    assert out[0].tolist() == [-1.0, -1.0]
    assert torch.equal(out[1:], rew[1:])                     # wins / losses / shaping untouched
    assert rew[0].tolist() == [0.0, 0.0]                     # not in place


def test_draw_rule_catches_a_turn_limit_draw_but_not_a_decided_game():
    rew = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]])
    done = torch.tensor([True, True, False])
    info = {"truncated": torch.zeros(3, dtype=torch.bool), "winner": torch.tensor([-1, 0, -1])}
    out, hit = apply_timeout(rew, done, info, -1.0, draw=True)
    assert hit.tolist() == [True, False, False]
    assert out[0].tolist() == [-1.0, -1.0] and out[1].tolist() == [1.0, -1.0]
    # without the draw rule (every game but Tetris) a draw stays a draw
    out, hit = apply_timeout(rew, done, info, -1.0)
    assert not hit.any() and torch.equal(out, rew)


def test_none_keeps_the_env_score():
    rew = torch.zeros(2, 2)
    info = {"truncated": torch.ones(2, dtype=torch.bool)}
    out, hit = apply_timeout(rew, torch.ones(2, dtype=torch.bool), info, None)
    assert out is rew and not hit.any()


def test_defaults_cover_the_games_whose_time_limit_beat_a_loss():
    assert set(TIMEOUT_REWARD) == {"sc2", "pokemon", "hok", "tetris"}
    assert all(v == -1.0 for v in TIMEOUT_REWARD.values())
    assert TIMEOUT_IS_DRAW == {"tetris"}
    a = parse_args(["--init", "x.pt", "--timeout_reward", "sc2=none", "--timeout_reward", "hok=-0.5"])
    assert a.timeout_reward == ["sc2=none", "hok=-0.5"]


def test_hok_shaping_starts_below_the_outcome_and_fades_out():
    from training.train_rl_joint import SHAPING, SHAPING_ANNEAL, shaping_weight
    s0, n = SHAPING["hok"], SHAPING_ANNEAL["hok"]
    assert s0 * 22 < 1.0                      # an expert episode's shaped sum (~22) < a win
    assert shaping_weight(s0, n, 0) == s0
    assert abs(shaping_weight(s0, n, n // 2) - s0 / 2) < 1e-12
    assert shaping_weight(s0, n, n) == 0.0 and shaping_weight(s0, n, 10 * n) == 0.0
    assert shaping_weight(0.3, 0, 10 ** 6) == 0.3


def test_alarm_reasons():
    from training.train_rl_joint import alarm_reasons
    kw = dict(timeout=0.05, len_x=2.0, kl_x=3.0)
    ok = {"timeout_rate": 0.01, "ep_len": 150.0, "kl": 0.05}
    assert alarm_reasons(ok, 100.0, 0.03, **kw) == []
    why = alarm_reasons({"timeout_rate": 0.2, "ep_len": 250.0, "kl": 0.1}, 100.0, 0.03, **kw)
    assert len(why) == 3 and "time-out" in why[0] and "length" in why[1] and "KL" in why[2]
    assert alarm_reasons({"timeout_rate": float("nan"), "ep_len": 250.0}, None, 0.03, **kw) == []
    assert alarm_reasons({"timeout_rate": 0.9}, None, 0.03, timeout=0, len_x=0, kl_x=0) == []


def test_alarm_stops_a_game_after_patience_iterations_and_zeroes_its_quota():
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    logs = []
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2,
                             alarm_min_episodes=8, alarm_min_tokens=1000, tokens_per_iter=100, rl_placement="all"),
        rl_games=["sc2", "chess"], kl_budget={"sc2": 0.03, "chess": 0.03}, alarm_base_len={},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={}, engine_streak={}, stopped={}, iteration=0, log=logs.append, world=1,
        sched=SimpleNamespace(quotas=lambda n: {"sc2": 50, "chess": 50}))
    good = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01}
    bad = {"episodes": 16, "timeout_rate": 0.5, "ep_len": 100.0, "kl": 0.01}
    few = {"episodes": 2, "timeout_rate": 1.0, "ep_len": 900.0, "kl": 1.0}

    def it(sc2, chess=good):
        rec = {"games": {"sc2": dict(sc2), "chess": dict(chess)}}
        JointRLTrainer._alarms(t, rec)
        t.iteration += 1
        return rec
    it(good)                                    # the baseline iteration
    assert t.alarm_base_len == {"sc2": 100.0, "chess": 100.0}
    it(bad); assert not t.stopped               # one bad iteration: patience 2
    it(good); it(bad); assert not t.stopped     # the streak restarts
    it(few); assert not t.stopped               # too few episodes to judge (streak kept)
    rec = it(bad)
    assert list(t.stopped) == ["sc2"] and "time-out rate" in t.stopped["sc2"]
    assert rec["stopped"] == t.stopped and any("[alarm] sc2" in m for m in logs)
    q, per_rank = JointRLTrainer._global_quotas(t)
    assert q == {"sc2": 0, "chess": 50} and per_rank == q


def test_referee_alarm_counts_evaluations_not_iterations():
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8, alarm_min_tokens=1000),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={}, alarm_streak={},
        vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={}, engine_streak={}, stopped={}, iteration=0,
        log=lambda m: None)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01}
    for vs in (float("nan"), 0.2, float("nan"), float("nan")):   # baseline, a bad eval, 2 plain its
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_init": vs, "vs_init_n": 32}}})
    assert not t.stopped and t.vs_init_streak == {"chess": 1}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_init": 0.25, "vs_init_n": 32}}})
    assert "score vs the --init network" in t.stopped["chess"]


def test_engine_alarm_is_relative_to_the_first_engine_evaluation():
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8, alarm_min_tokens=1000),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={"chess": 100.0},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={}, engine_streak={}, stopped={},
        iteration=0, log=lambda m: None)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01, "vs_engine_n": 128}
    for ve in (0.10, 0.05, 0.12):          # a strong engine: low scores are no alarm by themselves
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert t.engine_base == {"chess": 0.10} and not t.stopped
    t.engine_base["chess"], t.engine_base_n["chess"] = 0.40, 128
    for ve in (0.20, 0.22):                # 0.18+ below the first evaluation, twice
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert "score vs the engine" in t.stopped["chess"]



def test_engine_baseline_is_taken_on_the_first_iteration_too():
    """--eval_at_start: iteration 0's engine score (the IL policy) is the baseline, although
    that record only sets the episode-length baseline and is not judged otherwise."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8, alarm_min_tokens=1000),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={}, alarm_streak={},
        vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={}, engine_streak={}, stopped={}, iteration=0,
        log=lambda m: None)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.40, "vs_engine_n": 128}}})
    assert t.engine_base == {"chess": 0.40} and t.alarm_base_len == {"chess": 100.0}
    assert t.engine_base_n == {"chess": 128}
    for ve in (0.20, 0.21):
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve, "vs_engine_n": 128}}})
    assert "score vs the engine" in t.stopped["chess"]


def test_few_episode_games_are_judged_on_kl():
    """A game with 1-2 (long) episodes an iteration (hok, sc2) is judged on its KL — over its
    thousands of tokens — but not on a time-out rate or length of one episode."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8,
                             alarm_min_tokens=1000),
        rl_games=["hok", "sc2"], kl_budget={"hok": 0.03, "sc2": 0.03}, alarm_base_len={},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={}, engine_streak={}, stopped={},
        iteration=0, log=lambda m: None)
    hok = {"episodes": 1, "tokens": 4400, "timeout_rate": 0.0, "ep_len": 2233.0, "kl": 0.10}
    sc2 = {"episodes": 1, "tokens": 8400, "timeout_rate": 1.0, "ep_len": 8441.0, "kl": 0.006}
    for _ in range(2):
        JointRLTrainer._alarms(t, {"games": {"hok": dict(hok), "sc2": dict(sc2)}})
    assert "KL" in t.stopped["hok"] and "sc2" not in t.stopped     # one time-out: not judged
    tiny = {"episodes": 1, "tokens": 200, "kl": 0.5}                    # too few tokens
    t.stopped.clear(); t.alarm_streak.clear()
    for _ in range(2):
        JointRLTrainer._alarms(t, {"games": {"hok": dict(tiny)}})
    assert not t.stopped



def test_engine_alarm_needs_the_drop_beyond_the_noise():
    """A 0.17 drop between two 40-game evaluations is within the noise (2.5 standard errors
    of the difference ~0.27): no alarm; the same drop between 400-game evaluations is not."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer

    def run(n):
        t = SimpleNamespace(
            args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                                 alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2,
                                 alarm_min_episodes=8, alarm_min_tokens=1000),
            rl_games=["backgammon"], kl_budget={"backgammon": 0.03},
            alarm_base_len={"backgammon": 300.0}, alarm_streak={}, vs_init_streak={},
            engine_base={"backgammon": 0.47}, engine_base_n={"backgammon": n}, init_base={},
            init_base_n={}, engine_streak={},
            stopped={}, iteration=0, log=lambda m: None)
        row = {"episodes": 10, "timeout_rate": 0.0, "ep_len": 300.0, "kl": 0.002,
               "vs_engine": 0.30, "vs_engine_n": n}
        for _ in range(2):
            JointRLTrainer._alarms(t, {"games": {"backgammon": dict(row)}})
        return t.stopped

    assert not run(40)
    assert "score vs the engine" in run(400)["backgammon"]



def test_vs_init_alarm_of_multi_seat_games_uses_their_equal_strength_level():
    """mahjong (4 seats): ~0.31 vs the --init network is par (a quarter of the wins + half
    credit for drawn hands), no alarm against the fixed 0.35; its iteration-0 baseline
    (learner = init) is the level a drop is judged from."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_vs_init_drop=0.15, alarm_engine_drop=0.15, alarm_engine_z=2.5,
                             alarm_patience=2, alarm_min_episodes=8, alarm_min_tokens=1000),
        rl={"mahjong": SimpleNamespace(env=SimpleNamespace(n_seats=4))},
        rl_games=["mahjong"], kl_budget={"mahjong": 0.03}, alarm_base_len={}, alarm_streak={},
        vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={},
        engine_streak={}, stopped={}, iteration=0, log=lambda m: None)
    base = {"episodes": 12, "timeout_rate": 0.0, "ep_len": 550.0, "kl": 0.002, "vs_init_n": 128}
    JointRLTrainer._alarms(t, {"games": {"mahjong": {**base, "vs_init": 0.31}}})   # iteration 0
    assert t.init_base == {"mahjong": 0.31}
    for vi in (0.30, 0.29):                      # par, not < 0.35's two-player meaning
        JointRLTrainer._alarms(t, {"games": {"mahjong": {**base, "vs_init": vi}}})
    assert not t.stopped
    for vi in (0.10, 0.09):                      # far below its own level, twice
        JointRLTrainer._alarms(t, {"games": {"mahjong": {**base, "vs_init": vi}}})
    assert "score vs the --init network" in t.stopped["mahjong"]
    # no baseline (a run resumed from before it existed): not judged on this score
    t2 = SimpleNamespace(**{**vars(t), "init_base": {}, "init_base_n": {}, "stopped": {},
                            "vs_init_streak": {}, "alarm_base_len": {"mahjong": 550.0}})
    for it, vi in ((41, 0.05), (61, 0.05)):
        JointRLTrainer._alarms(t2, {"iteration": it, "games": {"mahjong": {**base, "vs_init": vi}}})
    assert not t2.stopped and t2.init_base == {}           # no baseline taken after iteration 0


def test_referee_ladder_steps_up_after_two_high_scores_and_rebaselines():
    """--eval_engine_ladder: a referee the learner beats (score >= --eval_engine_step_up twice in
    a row) is replaced by the next rung — its settings applied, the engine process dropped, the
    alarm baseline cleared so the next evaluation measures it again — rung by rung to the end."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer, parse_ladder

    ladder = parse_ladder(["chess=eps=0.25;eps=0,go=go+nodes+10;opt.UCI_Elo=1800"])
    assert ladder == {"chess": [{"eps": "0.25"}, {"eps": "0", "go": "go nodes 10"}, {"opt.UCI_Elo": "1800"}]}
    closed = []
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8,
                             alarm_min_tokens=1000, eval_engine_step_up=0.9, eval_engine_step_down=0.0),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={"chess": 100.0},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={},
        engine_streak={}, stopped={}, iteration=0, log=lambda m: None,
        engine_ladder=ladder, engine_rung={}, engine_high={}, engine_low={},
        eval_engine_eps={"chess": 0.5}, eval_engine_go={"chess": "go nodes 5"}, eval_engine_opt={},
        engine_base_settings={"chess": (0.5, "go nodes 5", {})},
        _engines={"chess": SimpleNamespace(close=lambda: closed.append(1))})
    t._apply_rung = lambda g, k: JointRLTrainer._apply_rung(t, g, k)
    t._set_rung = lambda g, k, why: JointRLTrainer._set_rung(t, g, k, why)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01, "vs_engine_n": 64}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.60}}})      # the baseline
    assert t.engine_base == {"chess": 0.60} and t.engine_rung == {} and t.engine_high == {"chess": 0}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.95}}})
    assert t.engine_rung == {} and t.engine_high == {"chess": 1}                        # once is not enough
    rec = {"games": {"chess": {**base, "vs_engine": 0.92}}}
    JointRLTrainer._alarms(t, rec)
    assert t.engine_rung == {"chess": 1} and t.engine_high == {"chess": 0} and closed == [1]
    assert t.eval_engine_eps["chess"] == 0.25 and "chess" not in t.engine_base and "chess" not in t._engines
    assert rec["games"]["chess"]["referee_rung"] == 1 and "alarm" not in rec["games"]["chess"]
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.55}}})      # the new baseline
    assert t.engine_base == {"chess": 0.55}
    for ve in (0.9, 0.9):
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert t.engine_rung == {"chess": 2} and t.eval_engine_eps["chess"] == 0.0 and t.eval_engine_go["chess"] == "go nodes 10"
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.50}}})
    for ve in (0.95, 0.95):
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert t.engine_rung == {"chess": 3} and t.eval_engine_opt == {"chess": {"UCI_Elo": "1800"}}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.50}}})
    for ve in (0.95, 0.95, 0.95):                                                       # no rung left
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert t.engine_rung == {"chess": 3} and not t.stopped
    rec = {"games": {"chess": {**base, "vs_engine": 0.05}}}
    for _ in range(3):                                   # --eval_engine_step_down 0 (off): stays
        JointRLTrainer._alarms(t, rec)
    assert t.engine_rung == {"chess": 3} and rec["games"]["chess"]["referee_at"] == 3


def test_referee_ladder_steps_down_after_two_low_scores_rebuilt_from_its_base():
    """--eval_engine_step_down: a referee the learner scores at most that against twice in a row
    goes back one rung — its settings rebuilt from the base settings plus the lower rungs (an eps
    the higher rung had removed comes back), the engine process dropped, the baseline cleared;
    never below rung 0.  Every evaluation records the rung it was played at (referee_at)."""
    from types import SimpleNamespace

    from training.train_rl_joint import JointRLTrainer, parse_ladder

    ladder = parse_ladder(["go=eps=0.25;eps=0;go=visits+16"])
    closed = []
    logs = []
    t = SimpleNamespace(
        args=SimpleNamespace(alarm_timeout=0.05, alarm_len_x=2.0, alarm_kl_x=3.0, alarm_vs_init=0.35,
                             alarm_engine_drop=0.15, alarm_engine_z=2.5, alarm_patience=2, alarm_min_episodes=8,
                             alarm_min_tokens=1000, eval_engine_step_up=0.65, eval_engine_step_down=0.25),
        rl_games=["go"], kl_budget={"go": 0.03}, alarm_base_len={"go": 100.0},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_base_n={}, init_base={}, init_base_n={},
        engine_streak={}, stopped={}, iteration=0, log=logs.append,
        engine_ladder=ladder, engine_rung={}, engine_high={}, engine_low={},
        eval_engine_eps={"go": 0.5}, eval_engine_go={"go": "visits 1"}, eval_engine_opt={},
        engine_base_settings={"go": (0.5, "visits 1", {})}, _engines={})
    t._apply_rung = lambda g, k: JointRLTrainer._apply_rung(t, g, k)
    t._set_rung = lambda g, k, why: JointRLTrainer._set_rung(t, g, k, why)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01, "vs_engine_n": 64}

    def ev(ve):
        rec = {"games": {"go": {**base, "vs_engine": ve}}}
        JointRLTrainer._alarms(t, rec)
        return rec["games"]["go"]

    assert ev(0.70)["referee_at"] == 0 and t.engine_rung == {}
    row = ev(0.66)                                       # 0.70, 0.66 >= 0.65: up to rung 1
    assert row["referee_at"] == 0 and row["referee_rung"] == 1 and t.engine_rung == {"go": 1}
    for ve in (0.8, 0.8):                                # up to rung 2: eps 0, the base search
        ev(ve)
    assert t.engine_rung == {"go": 2} and t.eval_engine_eps == {"go": 0.0}
    for ve in (0.9, 0.9):
        ev(ve)
    assert t.engine_rung == {"go": 3} and t.eval_engine_go == {"go": "visits 16"}
    t._engines["go"] = SimpleNamespace(close=lambda: closed.append(1))
    assert ev(0.30)["referee_at"] == 3 and t.engine_rung == {"go": 3}   # above step_down
    ev(0.20)
    row = ev(0.10)                                       # twice <= 0.25: back to rung 2
    assert row["referee_rung"] == 2 and t.engine_rung == {"go": 2} and closed == [1]
    assert t.eval_engine_go == {"go": "visits 1"} and t.eval_engine_eps == {"go": 0.0}
    assert "go" not in t.engine_base and "go" not in t._engines
    for ve in (0.1, 0.1, 0.1, 0.1):                      # down to rung 1, then rung 0
        ev(ve)
    assert t.engine_rung == {"go": 0} and t.eval_engine_eps == {"go": 0.5}
    for ve in (0.1, 0.1):                                # nothing below rung 0
        ev(ve)
    assert t.engine_rung == {"go": 0} and not t.stopped
    assert any("stepped down" in m for m in logs) and any("stepped up" in m for m in logs)


def test_archive_eval_wave_keeps_gmrules_trajectories(tmp_path):
    """archive_eval_wave: a gmsim env's finished episodes (seat, action id) and the learner's
    seats land in one .npz that replay_state rebuilds; an env without trajectories writes none."""
    import numpy as np

    from env.engine_policy import replay_state
    from env.vec_game import make_vec_env
    from training.train_rl_joint import archive_eval_wave

    env = make_vec_env("chess", 2, seed=3)
    env.reset(seed=3)
    path = str(tmp_path / "traces" / "eval_chess_init_it5_r0_w0.npz")
    n = archive_eval_wave(path, env, [[True, False], [False, True]], {"game": "chess", "iteration": 5})
    assert n == 2
    z = np.load(path)
    assert z["traj0"].shape[1] == 2 and z["traj0"].dtype == np.int64
    assert z["learner_seats"].tolist() == [[True, False], [False, True]]
    import json
    assert json.loads(str(z["meta"]))["iteration"] == 5
    st, moves = replay_state("chess", z["traj0"])
    assert moves == [] and str(st).startswith("rnbqkbnr")
    assert archive_eval_wave(str(tmp_path / "x.npz"), object(), []) == 0 and not (tmp_path / "x.npz").exists()


def test_deal_group_gives_consecutive_envs_the_same_deal():
    """VecEnvBase.deal_group G: a seeded reset gives envs k*G .. k*G+G-1 the episode seed
    seed + k — the same deal (here HUNL's hole and board cards, every chance event) — and
    the next group another one; G = 1 keeps seed + k per env.  Unseeded resets ignore it."""
    import numpy as np

    from env.vec_game import make_vec_env

    def chance(env, e):
        t = np.asarray(env.trajectory(e)).reshape(-1, 2)
        return t[t[:, 0] < 0, 1].tolist()

    env = make_vec_env("hunl", 4, seed=1)
    assert env.deal_group == 1
    assert env._episode_seeds(4, 7) == [7, 8, 9, 10]
    env.deal_group = 2
    assert env._episode_seeds(5, 7) == [7, 7, 8, 8, 9]
    assert len(set(env._episode_seeds(4, None))) > 1
    env.reset(seed=7)
    deals = [chance(env, e) for e in range(4)]
    assert deals[0] and deals[0] == deals[1] and deals[2] == deals[3] and deals[0] != deals[2]
    env.deal_group = 1
    env.reset(seed=7)
    assert chance(env, 0) == deals[0] and chance(env, 1) != deals[0]
