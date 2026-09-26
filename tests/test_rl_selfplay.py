"""Tests for the board-game self-play env and the generic PPO trainer.

The flagship invariant: rollout log-probs collected through
``GameNetwork.step()`` + ``BoardSelfPlayEnv`` equal the PPO replay's
recomputation through the full ``forward`` at the same ACT depth, so the
ratio is exactly ~1 before the first optimizer step.

Stream convention (training/train_rl_game.py): one interleaved mover-POV
sequence per game through one Stage-1 cache, loss on the learner seats,
GAE per seat.
"""

import random
import sys

import pytest
import torch

from data_processing.games.othello.rules import (
    disc_counts,
    has_legal_move,
    initial_board,
    is_game_over,
    legal_moves,
)
from env.board_selfplay_env import BoardSelfPlayEnv
from model.games import build_game_network
from model.rl_utils import compute_gae
from training import train_rl_game
from training.train_rl_game import batch_trajectories, collect, play_game, ppo_update

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)
CPU = torch.device("cpu")
PPO = dict(ppo_epochs=1, clip=0.2, value_coef=0.5, entropy_coef=0.01,
           clip_grad=1.0, act_steps=1)


def _net(game="othello", seed=0, **overrides):
    torch.manual_seed(seed)
    return build_game_network(game, **SMALL, **overrides).eval()


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

def test_othello_env_tracks_rules():
    """A full random game through the env matches the rules engine
    frame-for-frame (legal masks, forced passes, termination, winner)."""
    random.seed(0)
    env = BoardSelfPlayEnv("othello")
    board = initial_board()
    plies = 0
    while not env.done:
        mover = env.mover
        mask = env.legal_mask()
        expected = {r * 8 + c for r, c in legal_moves(board, mover)}
        assert set(torch.nonzero(mask).flatten().tolist()) == expected
        # obs planes come from the same board state
        planes = env.obs()["board_seq"]
        assert planes.shape == (3, 8, 8)
        own = {i for i, v in enumerate(board) if v == mover}
        assert set(torch.nonzero(planes[0].flatten()).flatten().tolist()) == own

        if not expected:
            assert not has_legal_move(board, mover)
            env.step(None)
        else:
            choice = random.choice(sorted(expected))
            from data_processing.games.othello.rules import apply_move
            apply_move(board, mover, choice // 8, choice % 8)
            env.step(choice)
        plies += 1
        assert plies < 200

    assert is_game_over(board)
    black, white = disc_counts(board)
    expected_winner = 1 if black > white else 2 if white > black else 0
    assert env.winner == expected_winner


def test_othello_env_rejects_bad_pass():
    env = BoardSelfPlayEnv("othello")
    with pytest.raises(ValueError, match="pass with legal moves"):
        env.step(None)


def test_gomoku_env_win_and_masks():
    env = BoardSelfPlayEnv("gomoku", board_size=9)
    assert env.legal_mask().sum() == 81
    # black builds five in a row on row 0; white replies on row 8
    for col in range(4):
        assert not env.step(col)          # black
        assert not env.step(72 + col)     # white
    assert env.step(4)                    # black's fifth stone wins
    assert env.done and env.winner == 1
    with pytest.raises(RuntimeError):
        env.step(40)


def test_gomoku_env_no_pass():
    env = BoardSelfPlayEnv("gomoku", board_size=9)
    with pytest.raises(ValueError, match="no pass"):
        env.step(None)
    # occupied point rejected by the rules engine
    env.step(40)
    with pytest.raises(Exception):
        env.step(40)


def test_env_to_play_and_legal():
    """to_play = the frame owner (alternates every ply, forced passes
    included); legal() = {'place': legal_mask()} or None on a forced pass /
    a finished game; to_play is read-only."""
    random.seed(3)
    env = BoardSelfPlayEnv("othello")
    assert env.to_play == 1
    with pytest.raises(AttributeError):
        env.to_play = 2
    n_pass, expect = 0, 1
    while not env.done:
        assert env.to_play == env.mover == expect
        lg = env.legal()
        mask = env.legal_mask()
        if lg is None:
            assert not mask.any()
            n_pass += 1
            env.step(None)
        else:
            assert set(lg) == {"place"} and torch.equal(lg["place"], mask)
            env.step(random.choice(torch.nonzero(mask).flatten().tolist()))
        expect = 3 - expect
    assert env.legal() is None                         # finished game

    g = BoardSelfPlayEnv("gomoku", board_size=9)
    assert g.to_play == 1 and g.legal()["place"].all()
    g.step(40)
    assert g.to_play == 2 and not g.legal()["place"][40]
    assert g.legal()["place"].sum() == 80


# ----------------------------------------------------------------------
# rl_utils
# ----------------------------------------------------------------------

def test_freeze_encoder_on_game_network():
    from model.rl_utils import freeze_encoder, unfreeze_all

    net = build_game_network("othello", **SMALL)
    freeze_encoder(net)
    assert all(not p.requires_grad for p in net.obs_encoder.parameters())
    assert all(not p.requires_grad for p in net.st_layers.parameters())
    assert all(p.requires_grad for p in net.action_decoder.parameters())
    unfreeze_all(net)
    assert all(p.requires_grad for p in net.parameters())


# ----------------------------------------------------------------------
# Trainer: one interleaved sequence per game
# ----------------------------------------------------------------------

def _check_stream(t, learner_seats=(1, 2)):
    """The per-game dict invariants of the stream convention."""
    T = t["plies"]
    assert len(t["obs"]) == T
    for k in ("place", "lp", "value", "seat", "pmask", "vmask"):
        assert t[k].shape == (T,), k
    assert t["legal"].shape[0] == T
    # both movers' frames, alternating (forced passes are frames too)
    assert t["seat"].tolist() == [1 + (i % 2) for i in range(T)]
    forced = ~t["legal"].any(-1)
    learner = torch.tensor([int(s) in learner_seats for s in t["seat"]])
    assert torch.equal(t["vmask"], learner)
    assert torch.equal(t["pmask"], learner & ~forced)
    assert (t["place"][forced] == 0).all()
    assert (t["lp"][~t["pmask"]] == 0).all()
    assert (t["lp"][t["pmask"]] <= 0).all()
    w = t["winner"]
    assert t["reward"] == {1: 0.0 if w == 0 else (1.0 if w == 1 else -1.0),
                           2: 0.0 if w == 0 else (1.0 if w == 2 else -1.0)}


@pytest.mark.parametrize("game,board_size,overrides", [
    ("othello", None, {}),
    ("gomoku", 9, {"board_h": 9, "board_w": 9}),
])
def test_selfplay_ppo_smoke(game, board_size, overrides):
    """Two self-played games → two interleaved sequences → one PPO update:
    the first-epoch ratio is exactly ~1 (rollout and replay define the same
    policy), losses are finite, and trainable parameters move."""
    random.seed(0)
    net = _net(game, **overrides)

    trajs = collect(net, game, board_size, n_games=2, act_steps=1, device=CPU)
    assert len(trajs) == 2  # one sequence per game
    for t in trajs:
        _check_stream(t)
        if game == "gomoku":
            assert t["pmask"].all()

    batch = batch_trajectories(trajs, CPU)
    T_max = max(t["plies"] for t in trajs)
    assert batch["place"].shape == batch["seat"].shape == (2, T_max)
    assert torch.equal(batch["frame_mask"].sum(1), torch.tensor([t["plies"] for t in trajs]))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in net.parameters()]

    stats = ppo_update(net, batch, optimizer, **PPO)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    # rollout (step + cache) and replay (full forward) agree bit-for-bit
    # up to the step-vs-forward cache equivalence
    assert stats["ratio_dev"] < 1e-4
    assert any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, net.parameters())
    )


def test_selfplay_second_update_ratio_moves():
    """After one optimizer step the policy changed, so a second update on
    the SAME batch must see ratios ≠ 1 — pins that old_lp is genuinely
    frozen rollout data, not recomputed."""
    random.seed(0)
    net = _net()
    trajs = collect(net, "othello", None, n_games=1, act_steps=1, device=CPU)
    batch = batch_trajectories(trajs, CPU)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ppo_update(net, batch, optimizer, **PPO)
    stats = ppo_update(net, batch, optimizer, **PPO)
    assert stats["ratio_dev"] > 1e-6


def _replay_values(net, trajs):
    """Full-forward replay of the batch (the PPO path): mover-POV values
    at every frame, learner and opponent frames alike."""
    batch = batch_trajectories(trajs, CPU)
    with torch.no_grad():
        pred = net(batch["obs"], act_steps=1, frame_mask=batch["frame_mask"])
    return pred["value"], batch


def test_one_cache_for_the_whole_game():
    """The learner's rollout values (one cache, step_idx = ply) equal the
    full-sequence replay at EVERY frame — both movers' frames live in one
    context, as in the IL samples."""
    random.seed(1)
    net = _net()
    trajs = collect(net, "othello", None, n_games=2, act_steps=1, device=CPU)
    values, batch = _replay_values(net, trajs)
    for i, t in enumerate(trajs):
        assert torch.allclose(values[i, :t["plies"]], t["value"], atol=1e-4)


# ----------------------------------------------------------------------
# Per-seat GAE
# ----------------------------------------------------------------------

def _old_per_seat_split(traj):
    """The pre-convention trajectory layout: one trajectory per seat with
    that seat's frames only (values, terminal reward)."""
    out = []
    for s in (1, 2):
        idx = (traj["seat"] == s).nonzero().squeeze(1)
        if idx.numel():
            out.append((s, idx, {"place": traj["place"][idx],
                                 "value": traj["value"][idx],
                                 "reward": traj["reward"][s]}))
    return out


def _old_gae(trajs, gamma, lam):
    """Reference copy of the old batch_trajectories GAE loop (per-seat
    trajectories, terminal reward only)."""
    res = []
    for t in trajs:
        T = len(t["place"])
        rewards = torch.zeros(T)
        rewards[-1] = t["reward"]
        dones = torch.zeros(T, dtype=torch.bool)
        dones[-1] = True
        res.append(compute_gae(rewards, t["value"], dones, gamma=gamma, lam=lam))
    return res


def _game_with_forced_pass(net, max_seeds=40):
    for seed in range(max_seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        t = play_game(net, "othello", None, 1, CPU)
        if (~t["legal"].any(-1)).any():
            return t
    pytest.fail("no othello self-play game with a forced pass")


def _synthetic(seats, reward1, seed):
    g = torch.Generator().manual_seed(seed)
    T = len(seats)
    return {"obs": [{"board_seq": torch.zeros(3, 8, 8)}] * T,
            "place": torch.zeros(T, dtype=torch.long),
            "legal": torch.ones(T, 64, dtype=torch.bool),
            "lp": torch.zeros(T), "value": torch.rand(T, generator=g) * 2 - 1,
            "seat": torch.tensor(seats), "pmask": torch.ones(T, dtype=torch.bool),
            "vmask": torch.ones(T, dtype=torch.bool),
            "reward": {1: reward1, 2: -reward1}, "winner": 1 if reward1 > 0 else 2,
            "plies": T}


@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.5)])
def test_per_seat_gae_matches_old_per_seat_code(gamma, lam):
    """adv/ret at each seat's frames equal the old per-seat computation on
    the same frames — real self-play games (one with an othello forced
    pass: the passing seat's frame stays in its own subsequence and its
    placements stop alternating with the opponent's), plus synthetic seat
    orders that do not alternate at all."""
    net = _net()
    trajs = [_game_with_forced_pass(net)]
    random.seed(0)
    trajs += collect(net, "othello", None, n_games=1, act_steps=1, device=CPU)
    trajs += [_synthetic([1, 2, 2, 1, 1, 1, 2], 1.0, 0),
              _synthetic([2, 1, 1, 2], -1.0, 1), _synthetic([1, 1, 1], 1.0, 2)]
    trajs[-1]["reward"] = {1: 0.0, 2: 0.0}                # draw, seat 2 absent

    batch = batch_trajectories(trajs, CPU, gamma=gamma, lam=lam)
    for i, t in enumerate(trajs):
        T = len(t["place"])
        assert (batch["adv"][i, T:] == 0).all() and (batch["ret"][i, T:] == 0).all()
        split = _old_per_seat_split(t)
        ref = _old_gae([tr for _, _, tr in split], gamma, lam)
        for (s, idx, _), (a, r) in zip(split, ref):
            torch.testing.assert_close(batch["adv"][i, idx], a, rtol=0, atol=0)
            torch.testing.assert_close(batch["ret"][i, idx], r, rtol=0, atol=0)
            if gamma == lam == 1.0:
                # Monte-Carlo: every frame's return is the seat's outcome
                assert torch.allclose(r, torch.full_like(r, t["reward"][s]))


# ----------------------------------------------------------------------
# Learner seats / frozen opponent
# ----------------------------------------------------------------------

def test_learner_seat_subset_masks_and_exact_ratio():
    """learner_seats=(1,): the policy/value loss lives on seat-1 frames
    only, seat-2 frames still stream through the learner's cache, and the
    first-epoch ratio stays exact."""
    random.seed(2)
    net = _net()
    trajs = collect(net, "othello", None, n_games=2, act_steps=1, device=CPU,
                    learner_seats=(1,))
    for t in trajs:
        _check_stream(t, learner_seats=(1,))
    batch = batch_trajectories(trajs, CPU)
    fm = batch["frame_mask"]
    assert torch.equal(batch["vmask"], (batch["seat"] == 1) & fm)
    assert torch.equal(batch["pmask"] & (batch["seat"] == 2), torch.zeros_like(fm))
    assert batch["pmask"].any()

    # value loss is normalised over the learner frames only: a huge return
    # on the opponent frames must not change it
    b2 = {**batch, "ret": torch.where(batch["seat"] == 2, 1e3, batch["ret"])}
    s1 = ppo_update(_net(), batch, torch.optim.SGD(_net().parameters(), lr=0.0), **PPO)
    s2 = ppo_update(_net(), b2, torch.optim.SGD(_net().parameters(), lr=0.0), **PPO)
    assert s1["value_loss"] == pytest.approx(s2["value_loss"], rel=1e-6)

    stats = ppo_update(net, batch, torch.optim.AdamW(net.parameters(), lr=1e-4), **PPO)
    assert stats["ratio_dev"] < 1e-4


class _Recorder:
    """Wraps a net's step() to log the calls the rollout makes."""

    def __init__(self, net):
        self.net, self.calls = net, []

    def step(self, obs, legal=None, st_cache=None, step_idx=0, act_steps=None):
        out = self.net.step(obs, legal=legal, st_cache=st_cache,
                            step_idx=step_idx, act_steps=act_steps)
        self.calls.append((step_idx, st_cache is None, act_steps,
                           int(out["action_place"].item())))
        return out


@pytest.mark.parametrize("learner", [1, 2])
def test_frozen_opponent_streams_every_ply(learner):
    """A differently-seeded frozen opponent steps every ply on its own
    cache (step_idx = ply) and its samples are the non-learner seat's
    actions; the learner's log-probs replay exactly."""
    random.seed(4)
    net = _net(seed=0)
    opp = _Recorder(_net(seed=1))
    trajs = collect(net, "othello", None, n_games=2, act_steps=1, device=CPU,
                    learner_seats=(learner,), opponent=opp, opponent_act_steps=2)
    start = 0
    for t in trajs:
        _check_stream(t, learner_seats=(learner,))
        calls = opp.calls[start:start + t["plies"]]
        start += t["plies"]
        assert [c[0] for c in calls] == list(range(t["plies"]))
        assert [c[1] for c in calls] == [True] + [False] * (t["plies"] - 1)
        assert {c[2] for c in calls} == {2}
        forced = ~t["legal"].any(-1)
        for ply, c in enumerate(calls):
            if t["seat"][ply] != learner and not forced[ply]:
                assert t["place"][ply] == c[3]
    assert start == len(opp.calls)

    values, batch = _replay_values(net, trajs)
    for i, t in enumerate(trajs):
        assert torch.allclose(values[i, :t["plies"]], t["value"], atol=1e-4)
    stats = ppo_update(net, batch, torch.optim.AdamW(net.parameters(), lr=1e-4), **PPO)
    assert stats["ratio_dev"] < 1e-4


def test_opponent_needs_a_free_seat():
    with pytest.raises(ValueError, match="non-learner seat"):
        play_game(_net(), "othello", None, 1, CPU, opponent=_net(seed=1))
    with pytest.raises(ValueError, match="subset"):
        play_game(_net(), "othello", None, 1, CPU, learner_seats=(3,))


def test_main_one_iteration(tmp_path, monkeypatch, capsys):
    """CLI smoke: one iteration against a frozen opponent checkpoint; the
    log counts games (not per-seat trajectories) and the save format is
    unchanged."""
    net = _net()
    ck = tmp_path / "othello_il.pt"
    torch.save({"game": "othello", "config": net.cfg.__dict__,
                "model_state": net.state_dict()}, ck)
    monkeypatch.setattr(train_rl_game, "pick_device", lambda: CPU)
    monkeypatch.setattr(sys, "argv", [
        "train_rl_game", "--game", "othello", "--il_checkpoint", str(ck),
        "--opponent_checkpoint", str(ck), "--learner_seats", "2",
        "--iters", "1", "--games_per_iter", "2", "--ppo_epochs", "1",
        "--act_steps", "1", "--save_dir", str(tmp_path / "rl")])
    train_rl_game.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[it0] "))
    p1 = float(line.split("p1_win=")[1].split()[0])
    lw = float(line.split("learner_win=")[1].split()[0])
    assert p1 in (0.0, 0.5, 1.0) and lw in (0.0, 0.5, 1.0)
    saved = torch.load(tmp_path / "rl" / "othello_rl_iter1.pt", weights_only=False)
    assert set(saved) == {"game", "iter", "model_state", "optimizer_state",
                          "config", "config_overrides"}

    monkeypatch.setattr(sys, "argv", [
        "train_rl_game", "--game", "othello", "--il_checkpoint", str(ck),
        "--opponent_checkpoint", str(ck), "--iters", "1"])
    with pytest.raises(ValueError, match="single"):
        train_rl_game.main()
