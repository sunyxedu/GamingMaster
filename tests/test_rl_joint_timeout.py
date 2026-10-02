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
                             alarm_engine_drop=0.15, alarm_patience=2,
                             alarm_min_episodes=8, tokens_per_iter=100, rl_placement="all"),
        rl_games=["sc2", "chess"], kl_budget={"sc2": 0.03, "chess": 0.03}, alarm_base_len={},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_streak={}, stopped={}, iteration=0, log=logs.append, world=1,
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
                             alarm_engine_drop=0.15, alarm_patience=2, alarm_min_episodes=8),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={}, alarm_streak={},
        vs_init_streak={}, engine_base={}, engine_streak={}, stopped={}, iteration=0,
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
                             alarm_engine_drop=0.15, alarm_patience=2, alarm_min_episodes=8),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={"chess": 100.0},
        alarm_streak={}, vs_init_streak={}, engine_base={}, engine_streak={}, stopped={},
        iteration=0, log=lambda m: None)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01, "vs_engine_n": 32}
    for ve in (0.10, 0.05, 0.12):          # a strong engine: low scores are no alarm by themselves
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve}}})
    assert t.engine_base == {"chess": 0.10} and not t.stopped
    t.engine_base["chess"] = 0.40
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
                             alarm_engine_drop=0.15, alarm_patience=2, alarm_min_episodes=8),
        rl_games=["chess"], kl_budget={"chess": 0.03}, alarm_base_len={}, alarm_streak={},
        vs_init_streak={}, engine_base={}, engine_streak={}, stopped={}, iteration=0,
        log=lambda m: None)
    base = {"episodes": 16, "timeout_rate": 0.0, "ep_len": 100.0, "kl": 0.01}
    JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": 0.40, "vs_engine_n": 32}}})
    assert t.engine_base == {"chess": 0.40} and t.alarm_base_len == {"chess": 100.0}
    for ve in (0.20, 0.21):
        JointRLTrainer._alarms(t, {"games": {"chess": {**base, "vs_engine": ve, "vs_engine_n": 32}}})
    assert "score vs the engine" in t.stopped["chess"]
