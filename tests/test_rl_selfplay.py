"""Tests for the board-game self-play env and the generic PPO trainer.

The flagship invariant: rollout log-probs collected through
``GameNetwork.step()`` + ``BoardSelfPlayEnv`` equal the PPO replay's
recomputation through the full ``forward`` at the same ACT depth, so the
ratio is exactly ~1 before the first optimizer step.
"""

import random

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
from training.train_rl_game import batch_trajectories, collect, ppo_update

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)


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
# Trainer
# ----------------------------------------------------------------------

@pytest.mark.parametrize("game,board_size,overrides", [
    ("othello", None, {}),
    ("gomoku", 9, {"board_h": 9, "board_w": 9}),
])
def test_selfplay_ppo_smoke(game, board_size, overrides):
    """Two self-played games → one PPO update: the first-epoch ratio is
    exactly ~1 (rollout and replay define the same policy), losses are
    finite, and trainable parameters move."""
    torch.manual_seed(0)
    random.seed(0)
    net = build_game_network(game, **SMALL, **overrides)
    net.eval()

    trajs = collect(net, game, board_size, n_games=2, act_steps=1,
                    device=torch.device("cpu"))
    assert len(trajs) == 4  # two seats per game
    for t in trajs:
        assert t["place"].shape == t["lp"].shape == t["pmask"].shape
        if game == "gomoku":
            assert t["pmask"].all()

    batch = batch_trajectories(trajs, torch.device("cpu"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in net.parameters()]

    stats = ppo_update(
        net, batch, optimizer, ppo_epochs=1, clip=0.2,
        value_coef=0.5, entropy_coef=0.01, clip_grad=1.0, act_steps=1,
    )
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
    torch.manual_seed(0)
    random.seed(0)
    net = build_game_network("othello", **SMALL)
    net.eval()
    trajs = collect(net, "othello", None, n_games=1, act_steps=1,
                    device=torch.device("cpu"))
    batch = batch_trajectories(trajs, torch.device("cpu"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ppo_update(net, batch, optimizer, ppo_epochs=1, clip=0.2,
               value_coef=0.5, entropy_coef=0.01, clip_grad=1.0, act_steps=1)
    stats = ppo_update(net, batch, optimizer, ppo_epochs=1, clip=0.2,
                       value_coef=0.5, entropy_coef=0.01, clip_grad=1.0,
                       act_steps=1)
    assert stats["ratio_dev"] > 1e-6
