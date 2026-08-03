"""Tests for the expander magnet heuristic anchor (model/magnet.py)."""

import math

import torch

from model.magnet import _dest, expander_magnet_heads

H = W = 25


def _grid(state_cells, army_cells):
    """Build a [1, 1, 12, H, W] obs grid from {state: [(r,c),...]} and
    {(r,c): army}."""
    g = torch.zeros(1, 1, 12, H, W)
    for state, cells in state_cells.items():
        for r, c in cells:
            g[0, 0, state, r, c] = 1.0
    for (r, c), a in army_cells.items():
        g[0, 0, 11, r, c] = math.log1p(a) / 10.0
    return g


def test_dest_shifts_no_wraparound():
    x = torch.zeros(1, H, W)
    x[0, 0, 5] = 1.0  # top row
    # DOWN (dr=+1): dest[r,c] = x[r+1,c] → the value moves to row -1 → gone
    assert _dest(x, 1, 0)[0, H - 1, 5] == 0.0  # no wrap to bottom
    # UP (dr=-1): dest[1,5] = x[0,5] = 1
    assert _dest(x, -1, 0)[0, 1, 5] == 1.0
    # jnp.roll would have wrapped; make sure we did not
    assert _dest(x, 1, 0).sum() == 0.0


def test_city_capture_scores_highest():
    # My general at (5,5) with 10 army; neutral city with 3 army at (5,6)
    # (RIGHT); blank at (5,4) (LEFT); rest fog.
    g = _grid(
        {7: [(5, 5)], 3: [(5, 6)], 2: [(5, 4)]},
        {(5, 5): 10, (5, 6): 3},
    )
    src = torch.tensor([[5 * W + 5]])
    out = expander_magnet_heads(g, src)
    dirs = out["direction_logits"][0, 0]  # DOWN, RIGHT, UP, LEFT
    # RIGHT = capturable city (10 > 3+1) → 5 (+2 top5 boost) = 7
    # LEFT  = neutral blank → 2 (+2) = 4 ; DOWN/UP = default 1 (+2) = 3
    assert dirs[1] == 7.0
    assert dirs[3] == 4.0
    assert dirs[0] == 3.0 and dirs[2] == 3.0
    # Source marginal puts the most mass on the general's cell
    src_probs = torch.softmax(out["source_logits"][0, 0], dim=-1)
    assert src_probs.argmax().item() == 5 * W + 5


def test_uncapturable_city_not_boosted():
    # 2 army vs city with 30 army: no capture score (2 > 31 is false)
    g = _grid({5: [(5, 5)], 3: [(5, 6)]}, {(5, 5): 2, (5, 6): 30})
    src = torch.tensor([[5 * W + 5]])
    dirs = expander_magnet_heads(g, src)["direction_logits"][0, 0]
    # RIGHT: city present but not capturable and not neutral-blank → default
    assert dirs[1] == dirs[0]  # same as empty DOWN


def test_capturable_enemy_tile():
    g = _grid({5: [(5, 5)], 8: [(6, 5)]}, {(5, 5): 10, (6, 5): 4})
    src = torch.tensor([[5 * W + 5]])
    dirs = expander_magnet_heads(g, src)["direction_logits"][0, 0]
    # DOWN = capturable enemy land (10 > 4+1) → 3 (+2 boost) = 5
    assert dirs[0] == 5.0


def test_head_shapes_and_pass_rate():
    g = _grid({5: [(3, 3), (4, 4)]}, {(3, 3): 5, (4, 4): 9})
    g = g.expand(1, 4, 12, H, W).contiguous()  # T=4
    src = torch.full((1, 4), 3 * W + 3, dtype=torch.long)
    out = expander_magnet_heads(g, src)
    assert out["source_logits"].shape == (1, 4, H * W)
    assert out["direction_logits"].shape == (1, 4, 4)
    assert out["split_logits"].shape == (1, 4, 2)
    assert out["pass_logits"].shape == (1, 4)
    for v in out.values():
        assert torch.isfinite(v).all()
    # Magnet pass prob sits in the low single digits (~5%)
    p_pass = torch.sigmoid(out["pass_logits"][0, 0]).item()
    assert 0.005 < p_pass < 0.15
    # Uniform split prior
    assert (out["split_logits"] == 0).all()


def test_direction_mask_applied():
    g = _grid({5: [(5, 5)]}, {(5, 5): 10})
    src = torch.tensor([[5 * W + 5]])
    dm = torch.zeros(1, 1, 4, H, W, dtype=torch.bool)
    dm[0, 0, 1, 5, 5] = True  # only RIGHT valid at the source
    dirs = expander_magnet_heads(g, src, dir_mask=dm)["direction_logits"]
    assert dirs[0, 0, 1] > -1e8
    assert (dirs[0, 0, [0, 2, 3]] <= -1e8).all()
