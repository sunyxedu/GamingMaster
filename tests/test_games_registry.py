"""Registry-level tests: every registered game builds a coherent network."""

import pytest
import torch

from model.game_network import GameNetwork
from model.games import GAME_REGISTRY, build_game_network
from model.games.gmrules import rules_games

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)

GAME_OVERRIDES = {
    "generals": dict(grid_h=6, grid_w=6),
    "chess": dict(board_h=6, board_w=6),
    "go": dict(board_h=7, board_w=7),
    "othello": {},
    "gomoku": dict(board_h=9, board_w=9),
    "shogi": dict(board_h=5, board_w=5),
    "lux": dict(map_h=6, map_w=6, max_units=5),
    "mahjong": {},
    "pokemon": {},
    "clash_royale": dict(arena_h=8, arena_w=6, n_cards=20),
    "hearthstone": {},
    "starcraft": dict(minimap_h=16, minimap_w=16, max_units=8),
}
# gmrules games register themselves (model/games/gmrules/): every one
# must build with its simulator's frame dims
GAME_OVERRIDES.update({g: {} for g in rules_games() if g not in GAME_OVERRIDES})


def test_registry_covers_all_games():
    assert set(GAME_OVERRIDES) == set(GAME_REGISTRY)


@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_build_game_network(game):
    net = build_game_network(game, **SMALL, **GAME_OVERRIDES[game])
    assert isinstance(net, GameNetwork)

    # The TRM reads q_halt from the HL decision token, which sits directly
    # after the game tokens — the core must have rewired the index.
    assert net.trm.q_token_idx == net.obs_encoder.n_tokens

    # PE covers game tokens + the two decision tokens.
    assert net._build_pe().shape == (1, net.obs_encoder.n_tokens + 2, 48)

    # Every parameter is reachable by the optimizer grouping used in training.
    from model.rl_utils import optim_param_groups
    groups = optim_param_groups(net, weight_decay=0.01)
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == sum(1 for p in net.parameters() if p.requires_grad)


def test_unknown_game_raises():
    with pytest.raises(KeyError, match="unknown game"):
        build_game_network("tetris")


def test_config_overrides_forwarded():
    net = build_game_network("go", **SMALL, board_h=9, board_w=9)
    assert net.obs_encoder.n_tokens == 81
