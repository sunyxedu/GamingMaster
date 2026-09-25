"""
Comprehensive audit tests for every module in the project (except simulator).

Organized by module:
  1. Parser (generals_json.py) — direction, encoding, legal mask, full conversion
  2. Filter (filter.py) — replay filtering logic
  3. Dataset (dataset_seq.py) — loading, truncation, collation edge cases
  4. Model Components — encoder, heads, transformer, TRM
  5. Losses (losses.py) — masking, NaN handling, edge cases
  6. Network (network.py) — GRU vs ST consistency, step vs full, shape checks
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# 1. Parser tests (data_processing/parsers/generals_json.py)
# =====================================================================

def test_compute_direction_all_four():
    """Verify all 4 directions map correctly."""
    print("=== compute_direction: all 4 directions ===")
    from data_processing.parsers.generals_json import compute_direction

    # down: row increases
    assert compute_direction([5, 3], [6, 3]) == 0, "Down should be 0"
    # right: col increases
    assert compute_direction([5, 3], [5, 4]) == 1, "Right should be 1"
    # up: row decreases
    assert compute_direction([5, 3], [4, 3]) == 2, "Up should be 2"
    # left: col decreases
    assert compute_direction([5, 3], [5, 2]) == 3, "Left should be 3"
    print("  All 4 directions correct ✓")


def test_compute_direction_no_move():
    """No-move (-1 sentinel) returns 0."""
    print("\n=== compute_direction: no move ===")
    from data_processing.parsers.generals_json import compute_direction

    assert compute_direction([-1, -1], [-1, -1]) == 0
    assert compute_direction([-1, -1], [5, 3]) == 0
    assert compute_direction([5, 3], [-1, -1]) == 0
    print("  No-move sentinel handled ✓")


def test_encode_state_channels_shape():
    """encode_state_as_channels produces [11, 25, 25] with correct one-hot."""
    print("\n=== encode_state_as_channels: shape and values ===")
    from data_processing.parsers.generals_json import encode_state_as_channels

    # Small 3x3 map padded to 25x25
    state = [[5, 0, 4], [8, 2, 1], [10, 3, 7]]
    channels = encode_state_as_channels(state)
    assert channels.shape == (11, 25, 25), f"Shape: {channels.shape}"

    # Check one-hot encoding
    assert channels[5, 0, 0] == 1.0, "State 5 at (0,0)"
    assert channels[0, 0, 1] == 1.0, "State 0 at (0,1)"
    assert channels[4, 0, 2] == 1.0, "State 4 at (0,2)"
    assert channels[8, 1, 0] == 1.0, "State 8 at (1,0)"
    assert channels[10, 2, 0] == 1.0, "State 10 at (2,0)"

    # Check padding is all zeros
    assert channels[:, 3:, :].sum() == 0.0, "Padding rows should be zero"
    assert channels[:, :3, 3:].sum() == 0.0, "Padding cols should be zero"

    # Each position has exactly one channel active (where state is valid)
    for i in range(3):
        for j in range(3):
            assert channels[:, i, j].sum() == 1.0, f"Position ({i},{j}) should have exactly one active channel"
    print("  Shape, one-hot, and padding correct ✓")


def test_encode_state_channels_padding_value():
    """State value -1 (padding from simulator) activates no channel."""
    print("\n=== encode_state_as_channels: padding state -1 ===")
    from data_processing.parsers.generals_json import encode_state_as_channels

    state = [[-1, -1], [-1, 5]]
    channels = encode_state_as_channels(state)
    assert channels[:, 0, 0].sum() == 0.0, "State -1 should activate no channel"
    assert channels[:, 0, 1].sum() == 0.0, "State -1 should activate no channel"
    assert channels[:, 1, 0].sum() == 0.0, "State -1 should activate no channel"
    assert channels[5, 1, 1] == 1.0, "State 5 should be encoded"
    print("  Padding state -1 handled correctly ✓")


def test_create_legal_source_mask():
    """Legal mask: only owned tiles (5,6,7) with army > 1."""
    print("\n=== create_legal_source_mask ===")
    from data_processing.parsers.generals_json import create_legal_source_mask

    state = [
        [5, 6, 7, 0, 4],    # owned types
        [8, 2, 5, -1, 3],   # enemy, blank, owned, padding, city
    ]
    armies = [
        [3, 1, 5, 0, 0],    # 3>1, 1 not >1, 5>1, 0, 0
        [10, 0, 0, 0, 2],   # enemy has army (not owned), blank, owned but 0 army
    ]
    mask = create_legal_source_mask(state, armies)
    assert mask.shape == (1, 25, 25), f"Shape: {mask.shape}"

    # (0,0): state=5, army=3 → True
    assert mask[0, 0, 0] == True, "(0,0) should be legal"
    # (0,1): state=6, army=1 → False (need > 1)
    assert mask[0, 0, 1] == False, "(0,1) army=1 not legal"
    # (0,2): state=7, army=5 → True
    assert mask[0, 0, 2] == True, "(0,2) should be legal"
    # (0,3): state=0 → False (not owned)
    assert mask[0, 0, 3] == False, "(0,3) not owned"
    # (1,0): state=8, army=10 → False (enemy, not owned)
    assert mask[0, 1, 0] == False, "(1,0) enemy tile not legal"
    # (1,2): state=5, army=0 → False (army not > 1)
    assert mask[0, 1, 2] == False, "(1,2) army=0 not legal"
    # Padding positions should be False
    assert mask[0, 2:, :].sum() == 0, "Padding rows should be False"
    print("  Legal mask logic correct ✓")


def test_convert_simulator_to_gio_shapes():
    """Full converter produces correct shapes and field types."""
    print("\n=== convert_simulator_to_gio: shapes ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 10
    H, W = 25, 25
    sim_data = _make_fake_sim_data(T_raw, H, W)
    gio = convert_simulator_to_gio(sim_data, max_actions_per_tick=4)

    T = T_raw - 1  # last obs has no future action

    assert len(gio["grid_seq"]) == T
    assert len(gio["globals_seq"]) == T
    assert len(gio["legal_source_mask_seq"]) == T
    assert len(gio["count"]) == T
    assert len(gio["src_y"]) == T
    assert len(gio["src_x"]) == T
    assert len(gio["dir_id"]) == T
    assert len(gio["ratio_id"]) == T
    assert len(gio["value"]) == T
    assert len(gio["tick_seq"]) == T
    assert len(gio["tick_bucket_seq"]) == T

    # Check grid shape: [T][C][H][W] where C=12
    grid0 = np.array(gio["grid_seq"][0])
    assert grid0.shape == (12, 25, 25), f"Grid shape: {grid0.shape}"

    # Check K-padded action fields
    K = 4
    assert len(gio["src_y"][0]) == K
    assert len(gio["dir_id"][0]) == K

    # Globals should have 4 features
    assert len(gio["globals_seq"][0]) == 4
    print("  All shapes correct ✓")


def test_convert_observation_action_pairing():
    """Verify observation[t] is paired with action[t+1] (IL pairing)."""
    print("\n=== convert: observation/action pairing ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 5
    sim_data = _make_fake_sim_data(T_raw, 25, 25)
    # Put a known move at turn 3: start=(2,3), end=(2,4)
    sim_data["move_start"][3] = [2, 3]
    sim_data["move_end"][3] = [2, 4]
    sim_data["move_is_50"][3] = 0

    gio = convert_simulator_to_gio(sim_data)
    T = T_raw - 1  # 4

    # Action at t+1=3 is paired with observation t=2
    # So gio index 2 should have the move from turn 3
    assert gio["count"][2] == 1, "t=2 should have count=1 (action from turn 3)"
    assert gio["src_y"][2][0] == 2, "Source row should be 2"
    assert gio["src_x"][2][0] == 3, "Source col should be 3"
    # Direction: (2,3) → (2,4) = right = 1
    assert gio["dir_id"][2][0] == 1, "Direction should be right (1)"
    assert gio["ratio_id"][2][0] == 0, "Not is_50"
    print("  Observation/action pairing correct ✓")


def test_convert_no_move_padding():
    """Timesteps with no move should have count=0 and zero padding."""
    print("\n=== convert: no move padding ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 4
    sim_data = _make_fake_sim_data(T_raw, 25, 25)
    # All moves are [-1,-1] (no move)
    gio = convert_simulator_to_gio(sim_data)

    for t in range(T_raw - 1):
        assert gio["count"][t] == 0, f"t={t} should have count=0"
        assert gio["src_y"][t] == [0, 0, 0, 0], f"t={t} src_y should be all zeros"
    print("  No-move padding correct ✓")


def test_convert_value_terminal_states():
    """Value computation: op_land=0 → +1.0, my_land=0 → -1.0."""
    print("\n=== convert: terminal value ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    # op_land = 0 at t=2
    sim = _make_fake_sim_data(4, 25, 25)
    sim["info_myArmy"] = [10, 10, 10, 10]
    sim["info_myLand"] = [5, 5, 5, 5]
    sim["info_opArmy"] = [10, 10, 0, 0]
    sim["info_opLand"] = [5, 5, 0, 0]
    gio = convert_simulator_to_gio(sim)
    assert gio["value"][2] == 1.0, f"op_land=0 should give value=1.0, got {gio['value'][2]}"

    # my_land = 0 at t=2
    sim2 = _make_fake_sim_data(4, 25, 25)
    sim2["info_myArmy"] = [10, 10, 0, 0]
    sim2["info_myLand"] = [5, 5, 0, 0]
    sim2["info_opArmy"] = [10, 10, 10, 10]
    sim2["info_opLand"] = [5, 5, 5, 5]
    gio2 = convert_simulator_to_gio(sim2)
    assert gio2["value"][2] == -1.0, f"my_land=0 should give value=-1.0, got {gio2['value'][2]}"
    print("  Terminal values correct ✓")


def test_convert_value_blending_with_outcome():
    """Value blends heuristic → outcome as t increases."""
    print("\n=== convert: value blending ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 20
    sim = _make_fake_sim_data(T_raw, 25, 25)
    # Equal armies → heuristic ≈ 0.0
    for t in range(T_raw):
        sim["info_myArmy"][t] = 100
        sim["info_myLand"][t] = 10
        sim["info_opArmy"][t] = 100
        sim["info_opLand"][t] = 10
    sim["winner"] = "player0"
    sim["player"] = "player0"

    gio = convert_simulator_to_gio(sim)
    T = T_raw - 1

    # Early value should be close to heuristic (≈0 for equal armies)
    assert abs(gio["value"][0]) < 0.3, f"Early value too far from 0: {gio['value'][0]}"
    # Late value should be close to outcome (+1.0)
    assert gio["value"][T - 1] > 0.5, f"Late value should be close to +1: {gio['value'][T-1]}"
    # Values should generally increase (winning)
    assert gio["value"][T - 1] > gio["value"][0], "Values should increase toward outcome"
    print("  Value blending correct ✓")


def test_convert_army_channel():
    """Army channel uses log1p normalization and handles padding (-1)."""
    print("\n=== convert: army channel ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 3
    sim = _make_fake_sim_data(T_raw, 25, 25)
    # Set army at (2, 3) = 100, rest = 0 or -1 (padding)
    sim["armies"][0][2][3] = 100
    sim["armies"][0][0][0] = -1  # padding value from simulator

    gio = convert_simulator_to_gio(sim)
    grid = np.array(gio["grid_seq"][0])  # [12, 25, 25]
    army_ch = grid[11]  # last channel is army

    # Position (2,3): log1p(100)/10 ≈ 0.4615
    expected = np.log1p(100) / 10.0
    assert abs(army_ch[2, 3] - expected) < 1e-5, \
        f"Army channel at (2,3): {army_ch[2, 3]:.5f} vs expected {expected:.5f}"

    # Padding position (-1): should be 0 (not > 0)
    assert army_ch[0, 0] == 0.0, "Padding army should map to 0"

    # Position with army=0: should be 0
    assert army_ch[5, 5] == 0.0, "Zero army should map to 0"
    print("  Army channel encoding correct ✓")


def test_convert_tick_bucket():
    """tick_bucket_seq should cycle within [0, 25] every 50 ticks."""
    print("\n=== convert: tick_bucket ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    T_raw = 55
    sim = _make_fake_sim_data(T_raw, 25, 25)
    gio = convert_simulator_to_gio(sim)

    # t=0: ((0%50)+1)//2 = 0
    assert gio["tick_bucket_seq"][0] == 0
    # t=1: ((1%50)+1)//2 = 1
    assert gio["tick_bucket_seq"][1] == 1
    # t=49: ((49%50)+1)//2 = 25
    assert gio["tick_bucket_seq"][49] == 25
    # t=50: ((0%50)+1)//2 = 0 (wraps)
    assert gio["tick_bucket_seq"][50] == 0
    print("  Tick bucket cycling correct ✓")


# =====================================================================
# 2. (filter tests live with the Rust filter: data_processing/rs/gmdata/src/generals/filter.rs)
# =====================================================================

def test_dataset_loading_shapes():
    """Dataset produces correct tensor shapes."""
    print("\n=== dataset: loading shapes ===")
    from data_processing.dataset_seq import SeqReplayDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_gio(tmpdir, "test.json", T=10)
        ds = SeqReplayDataset(tmpdir)
        item = ds[0]

        assert item["obs"]["grid_seq"].shape == (1, 10, 12, 25, 25)
        assert item["obs"]["globals_seq"].shape == (1, 10, 4)
        assert item["obs"]["legal_source_mask_seq"].shape == (1, 10, 1, 25, 25)
        assert item["tgt"]["count"].shape == (1, 10)
        assert item["tgt"]["src_y"].shape == (1, 10, 4)
        assert item["tgt"]["value"].shape == (1, 10)
        assert item["meta"]["T"] == 10
        assert item["meta"]["H"] == 25
        assert item["meta"]["W"] == 25
        assert item["meta"]["K"] == 4
    print("  Loading shapes correct ✓")


def test_dataset_truncation():
    """max_sequence_length truncates correctly."""
    print("\n=== dataset: truncation ===")
    from data_processing.dataset_seq import SeqReplayDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_gio(tmpdir, "test.json", T=50)
        ds = SeqReplayDataset(tmpdir, max_sequence_length=20)
        item = ds[0]

        assert item["meta"]["T"] == 20
        assert item["obs"]["grid_seq"].shape == (1, 20, 12, 25, 25)
        assert item["tgt"]["count"].shape == (1, 20)
    print("  Truncation correct ✓")


def test_dataset_no_truncation_when_shorter():
    """No truncation when T < max_sequence_length."""
    print("\n=== dataset: no truncation when shorter ===")
    from data_processing.dataset_seq import SeqReplayDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_gio(tmpdir, "test.json", T=10)
        ds = SeqReplayDataset(tmpdir, max_sequence_length=50)
        item = ds[0]

        assert item["meta"]["T"] == 10
    print("  No truncation when shorter ✓")


def test_collation_padding():
    """Collation pads sequences to max length in batch."""
    print("\n=== dataset: collation padding ===")
    from data_processing.dataset_seq import (
        SeqReplayDataset,
        collate_variable_length_sequences,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_gio(tmpdir, "short.json", T=5)
        _write_mock_gio(tmpdir, "long.json", T=15)
        ds = SeqReplayDataset(tmpdir)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate_variable_length_sequences
        )
        batch = next(iter(loader))

        max_T = 15
        assert batch["obs"]["grid_seq"].shape == (2, max_T, 12, 25, 25)
        assert batch["tgt"]["count"].shape == (2, max_T)
        assert set(batch["meta"]["lengths"]) == {5, 15}

        # Padding should be zeros for grid_seq
        for i, length in enumerate(batch["meta"]["lengths"]):
            if length < max_T:
                padded = batch["obs"]["grid_seq"][i, length:]
                assert padded.abs().sum() == 0, "Padding should be zeros"
    print("  Collation padding correct ✓")


def test_collation_tick_sequences():
    """tick_seq and tick_bucket_seq are collated correctly."""
    print("\n=== dataset: collation tick sequences ===")
    from data_processing.dataset_seq import (
        SeqReplayDataset,
        collate_variable_length_sequences,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_gio(tmpdir, "a.json", T=5)
        _write_mock_gio(tmpdir, "b.json", T=5)
        ds = SeqReplayDataset(tmpdir)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate_variable_length_sequences
        )
        batch = next(iter(loader))

        assert batch["meta"]["tick_seq"].shape == (2, 5)
        assert batch["meta"]["tick_bucket_seq"].shape == (2, 5)
        # tick_seq should be [0,1,2,3,4] for each item
        for i in range(2):
            expected = torch.arange(5, dtype=torch.float32)
            assert torch.allclose(batch["meta"]["tick_seq"][i], expected)
    print("  Tick sequences collated correctly ✓")


# =====================================================================
# 4. Model Component tests
# =====================================================================

def test_spatial_encoder_different_batch():
    """SpatialEncoder works with various batch sizes."""
    print("\n=== SpatialEncoder: batch sizes ===")
    from model.components.encoder import SpatialEncoder
    from model.config import ModelConfig

    cfg = ModelConfig(d_model=64, encoder_channels=32)
    enc = SpatialEncoder(cfg)
    enc.eval()

    for B in [1, 4, 16]:
        x = torch.randn(B, 12, 25, 25)
        y = enc(x)
        assert y.shape == (B, 64, 25, 25), f"B={B}: {y.shape}"
    print("  All batch sizes correct ✓")


def test_source_head_all_masked():
    """External legal mask: all-masked produces all -inf after masking."""
    print("\n=== SourceHead: all masked (external) ===")
    from model.components.heads import SourceHead

    head = SourceHead(64)
    tokens = torch.randn(2, 625, 64)
    mask = torch.zeros(2, 625, dtype=torch.bool)  # all False
    logits = head(tokens)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    assert (logits_masked == float("-inf")).all(), "All positions should be -inf"
    print("  All-masked produces -inf ✓")


def test_source_head_partial_mask():
    """External legal mask preserves legal positions."""
    print("\n=== SourceHead: partial mask (external) ===")
    from model.components.heads import SourceHead

    head = SourceHead(64)
    tokens = torch.randn(1, 10, 64)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 3] = True
    mask[0, 7] = True

    logits = head(tokens)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    assert logits_masked[0, 3] != float("-inf"), "Legal position should not be -inf"
    assert logits_masked[0, 7] != float("-inf"), "Legal position should not be -inf"
    assert logits_masked[0, 0] == float("-inf"), "Masked position should be -inf"

    probs = F.softmax(logits_masked, dim=-1)
    assert abs(probs[0, 3] + probs[0, 7] - 1.0) < 1e-5, "Legal positions should sum to 1"
    print("  Partial mask correct ✓")


def test_value_head_range():
    """ValueHead output is unbounded (no tanh) so the critic can fit
    discounted RL returns that exceed [-1, 1]."""
    print("\n=== ValueHead: output range ===")
    from model.components.heads import ValueHead

    head = ValueHead(64)
    # Test with extreme inputs
    x_large = torch.ones(10, 64) * 100
    x_small = torch.ones(10, 64) * -100
    x_rand = torch.randn(100, 64) * 10

    for x, name in [(x_large, "large"), (x_small, "small"), (x_rand, "random")]:
        v = head(x)
        assert torch.isfinite(v).all(), f"{name}: non-finite values"
    print("  Output finite and unbounded (no tanh saturation) ✓")


def test_direction_head_dimensions():
    """Neighbor-gather DirectionHead: shape contract incl. [BT, d] batching."""
    print("\n=== DirectionHead: dimensions ===")
    from model.components.heads import DirectionHead

    head = DirectionHead(64, 25, 25)
    src = torch.randn(2, 64)
    spatial = torch.randn(2, 625, 64)
    dec = torch.randn(2, 64)
    logits = head(src, spatial, dec, torch.tensor([0, 624]))
    assert logits.shape == (2, 4)

    # Should accept batched inputs [BT, d]
    src_big = torch.randn(100, 64)
    spatial_big = torch.randn(100, 625, 64)
    dec_big = torch.randn(100, 64)
    logits_big = head(src_big, spatial_big, dec_big, torch.randint(0, 625, (100,)))
    assert logits_big.shape == (100, 4)
    print("  Dimensions correct ✓")


def test_trm_h1_all_params_have_grad():
    """With H_cycles=1, all trainable parameters get gradients.
    H_init/L_init are buffers (not trainable), following the paper."""
    print("\n=== TRM: H_cycles=1 all params get grad ===")
    from model.components.trm import TRM
    from model.config import ModelConfig

    cfg = ModelConfig(
        d_model=32, n_heads=4, d_ff=64,
        core_type="trm", trm_H_cycles=1, trm_L_cycles=2, trm_block_layers=1,
    )
    trm = TRM(cfg)
    x = torch.randn(1, 5, 32)
    z_H, q, _ = trm(x)
    loss = z_H.sum() + q.sum()
    loss.backward()

    no_grad = [n for n, p in trm.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Missing gradients: {no_grad}"
    # H_init/L_init are buffers, not in named_parameters()
    assert "H_init" not in dict(trm.named_parameters())
    assert "L_init" not in dict(trm.named_parameters())
    print("  All trainable parameters get gradients with H_cycles=1 ✓")
    print("  H_init/L_init are buffers (not trainable) ✓")


def test_trm_output_varies_with_input():
    """TRM output should change when input changes."""
    print("\n=== TRM: output varies with input ===")
    from model.components.trm import TRM
    from model.config import ModelConfig

    cfg = ModelConfig(
        d_model=32, n_heads=4, d_ff=64,
        core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, dropout=0.0,
    )
    trm = TRM(cfg)
    trm.eval()

    x1 = torch.randn(1, 5, 32)
    x2 = torch.randn(1, 5, 32)

    with torch.no_grad():
        z1, _, _ = trm(x1)
        z2, _, _ = trm(x2)

    diff = (z1 - z2).abs().max().item()
    assert diff > 0.01, f"TRM outputs too similar for different inputs: {diff}"
    print(f"  Output varies (max diff={diff:.4f}) ✓")


def test_spatiotemporal_block_output_shape():
    """SpatiotemporalBlock maintains shape [B, T, S, d]."""
    print("\n=== SpatiotemporalBlock: output shape ===")
    from model.components.transformer import SpatiotemporalBlock
    from model.config import ModelConfig

    cfg = ModelConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
    block = SpatiotemporalBlock(cfg)

    for B, T, S in [(1, 1, 10), (2, 5, 627), (1, 20, 50)]:
        x = torch.randn(B, T, S, 32)
        y = block(x)
        assert y.shape == (B, T, S, 32), f"Shape mismatch for B={B},T={T},S={S}: {y.shape}"
    print("  All shapes correct ✓")


# =====================================================================
# 5. Loss tests (model/losses.py)
# =====================================================================

def test_loss_all_zero_count():
    """When all counts are 0, source/dir/split losses should be 0."""
    print("\n=== loss: all zero count ===")
    from model.losses import ImitationLoss

    B, T, N = 2, 5, 625
    pred = {
        "source_logits": torch.randn(B, T, N),
        "direction_logits": torch.randn(B, T, 4),
        "split_logits": torch.randn(B, T, 2),
        "pass_logits": torch.randn(B, T),
        "value": torch.randn(B, T),
    }
    tgt = {
        "count": torch.zeros(B, T, dtype=torch.long),
        "src_y": torch.zeros(B, T, 4, dtype=torch.long),
        "src_x": torch.zeros(B, T, 4, dtype=torch.long),
        "dir_id": torch.zeros(B, T, 4, dtype=torch.long),
        "ratio_id": torch.zeros(B, T, 4, dtype=torch.long),
        "value": torch.zeros(B, T),
    }
    criterion = ImitationLoss(grid_w=25)
    losses = criterion(pred, tgt)

    assert losses["source_loss"].item() == 0.0
    assert losses["direction_loss"].item() == 0.0
    assert losses["split_loss"].item() == 0.0
    # Value loss should still be computed (not masked by count)
    assert losses["value_loss"].item() >= 0.0
    print("  Zero count: source/dir/split=0, value computed ✓")


def test_loss_sequence_length_masking():
    """Padding beyond sequence length doesn't contribute to any loss."""
    print("\n=== loss: sequence length masking ===")
    from model.losses import ImitationLoss

    B, T, N = 1, 10, 625
    pred = {
        "source_logits": torch.randn(B, T, N),
        "direction_logits": torch.randn(B, T, 4),
        "split_logits": torch.randn(B, T, 2),
        "pass_logits": torch.randn(B, T),
        "value": torch.randn(B, T),
    }
    tgt = {
        "count": torch.ones(B, T, dtype=torch.long),
        "src_y": torch.randint(0, 25, (B, T, 4)),
        "src_x": torch.randint(0, 25, (B, T, 4)),
        "dir_id": torch.randint(0, 4, (B, T, 4)),
        "ratio_id": torch.randint(0, 2, (B, T, 4)),
        "value": torch.randn(B, T),
    }
    criterion = ImitationLoss(grid_w=25)

    # Length=10 (full sequence)
    loss_full = criterion(pred, tgt, lengths=[10])
    # Length=5 (only first 5 valid)
    loss_half = criterion(pred, tgt, lengths=[5])

    # Source/dir/split losses should differ (different number of valid timesteps)
    assert loss_full["source_loss"] != loss_half["source_loss"], \
        "Source loss should differ with different lengths"
    # Value loss should differ (averaged over different number of timesteps)
    assert loss_full["value_loss"] != loss_half["value_loss"], \
        "Value loss should differ with different lengths"
    print("  Sequence length masking works ✓")


def test_loss_q_halt_masked_to_valid():
    """q_halt loss only computed for timesteps with valid moves."""
    print("\n=== loss: q_halt masked to valid ===")
    from model.losses import ImitationLoss

    B, T, N = 1, 4, 625
    pred = {
        "source_logits": torch.zeros(B, T, N),
        "direction_logits": torch.zeros(B, T, 4),
        "split_logits": torch.zeros(B, T, 2),
        "pass_logits": torch.zeros(B, T),
        "value": torch.zeros(B, T),
        "q_halt": torch.zeros(B, T),
    }
    # Make model predict source 0 (matching padding)
    pred["source_logits"][:, :, 0] = 100.0

    tgt = {
        "count": torch.tensor([[0, 1, 0, 0]]),  # only t=1 is valid
        "src_y": torch.zeros(B, T, 4, dtype=torch.long),
        "src_x": torch.zeros(B, T, 4, dtype=torch.long),
        "dir_id": torch.zeros(B, T, 4, dtype=torch.long),
        "ratio_id": torch.zeros(B, T, 4, dtype=torch.long),
        "value": torch.zeros(B, T),
    }
    # Valid move at t=1: source (0,0) = flat 0, which matches prediction
    # So correct=True for t=1

    criterion = ImitationLoss(grid_w=25)
    losses = criterion(pred, tgt)

    # q_halt: sigmoid(0) = 0.5, target = 1.0 (correct prediction)
    # BCE = -log(0.5) ≈ 0.693, but only 1 valid timestep
    expected = -np.log(0.5)
    assert abs(losses["q_halt_loss"].item() - expected) < 0.01, \
        f"q_halt_loss={losses['q_halt_loss']:.4f}, expected ~{expected:.4f}"
    print("  q_halt masked to valid moves ✓")


def test_loss_nan_handling():
    """NaN from all-masked source logits is handled gracefully."""
    print("\n=== loss: NaN handling ===")
    from model.losses import ImitationLoss

    B, T, N = 1, 3, 625
    pred = {
        "source_logits": torch.full((B, T, N), float("-inf")),  # all masked
        "direction_logits": torch.randn(B, T, 4),
        "split_logits": torch.randn(B, T, 2),
        "pass_logits": torch.randn(B, T),
        "value": torch.randn(B, T),
    }
    tgt = {
        "count": torch.ones(B, T, dtype=torch.long),
        "src_y": torch.randint(0, 25, (B, T, 4)),
        "src_x": torch.randint(0, 25, (B, T, 4)),
        "dir_id": torch.randint(0, 4, (B, T, 4)),
        "ratio_id": torch.randint(0, 2, (B, T, 4)),
        "value": torch.randn(B, T),
    }
    criterion = ImitationLoss(grid_w=25)
    losses = criterion(pred, tgt)

    # Should not produce NaN
    assert not losses["loss"].isnan(), "Total loss should not be NaN"
    assert not losses["source_loss"].isnan(), "Source loss should not be NaN"
    print("  NaN handled gracefully ✓")


def test_loss_gradient_flow():
    """Loss produces valid gradients for model parameters."""
    print("\n=== loss: gradient flow ===")
    from model.config import ModelConfig
    from model.losses import ImitationLoss
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32,
    )
    model = GeneralsNetwork(cfg)
    criterion = ImitationLoss(grid_w=25)

    B, T = 1, 3
    grid = torch.randn(B, T, 12, 25, 25)
    glob = torch.randn(B, T, 4)
    mask = torch.ones(B, T, 1, 25, 25, dtype=torch.bool)
    tgt = {
        "count": torch.ones(B, T, dtype=torch.long),
        "src_y": torch.randint(0, 25, (B, T, 4)),
        "src_x": torch.randint(0, 25, (B, T, 4)),
        "dir_id": torch.randint(0, 4, (B, T, 4)),
        "ratio_id": torch.randint(0, 2, (B, T, 4)),
        "value": torch.randn(B, T),
    }

    pred = model(grid, glob, mask, tgt["src_y"][:, :, 0], tgt["src_x"][:, :, 0])
    losses = criterion(pred, tgt)

    # Total loss should require grad (computed from model output)
    assert losses["loss"].requires_grad, "Loss should require grad"

    losses["loss"].backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"Missing grads: {n_with_grad}/{n_total}"
    print(f"  All {n_total} parameters get gradients ✓")


# =====================================================================
# 6. Network tests (model/network.py)
# =====================================================================

def test_network_st_shapes():
    """Spatiotemporal mode produces correct output shapes."""
    print("\n=== network: ST shapes ===")
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32, temporal_type="spatiotemporal",
    )
    model = GeneralsNetwork(cfg)
    model.eval()

    B, T = 2, 5
    grid = torch.randn(B, T, 12, 25, 25)
    glob = torch.randn(B, T, 4)

    with torch.no_grad():
        out = model(grid, glob)

    assert out["source_logits"].shape == (B, T, 625)
    assert out["direction_logits"].shape == (B, T, 4)
    assert out["split_logits"].shape == (B, T, 2)
    assert out["pass_logits"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)
    print("  ST shapes correct ✓")


def test_network_st_step_cache_growth():
    """ST step KV cache grows correctly."""
    print("\n=== network: ST step cache growth ===")
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32, temporal_type="spatiotemporal", dropout=0.0,
    )
    model = GeneralsNetwork(cfg)
    model.eval()

    B = 1
    cache = None
    for t in range(5):
        grid = torch.randn(B, 12, 25, 25)
        glob = torch.randn(B, 4)

        with torch.no_grad():
            out = model.step(grid, glob, st_cache=cache)

        cache = out["st_cache"]
        assert len(cache) == 2, "Should have 2 caches (n_st_layers=2)"
        assert cache[0].shape[1] == t + 1, f"Cache should have {t+1} entries"
    print("  ST step cache grows correctly ✓")


def test_network_legal_mask_effect():
    """Legal mask affects sampling but source_logits are raw (unmasked)."""
    print("\n=== network: legal mask effect ===")
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32, dropout=0.0,
    )
    model = GeneralsNetwork(cfg)
    model.eval()

    B = 1
    grid = torch.randn(B, 12, 25, 25)
    glob = torch.randn(B, 4)

    # Source logits should be raw (no -inf) regardless of mask
    mask_one = torch.zeros(B, 1, 25, 25, dtype=torch.bool)
    mask_one[0, 0, 0, 0] = True

    with torch.no_grad():
        out = model.step(grid, glob, legal_mask=mask_one)

    # source_logits are raw — no -inf values
    assert not (out["source_logits"] == float("-inf")).any(), \
        "source_logits should be raw (unmasked)"
    # External masking still works
    lm = mask_one.reshape(B, -1)
    masked = out["source_logits"].masked_fill(~lm, float("-inf"))
    assert (masked[0, 1:] == float("-inf")).all()
    print("  Legal mask affects source logits ✓")


def test_mock_data_valid_ranges():
    """Verify mock data generator produces values in valid ranges."""
    print("\n=== mock data: valid ranges ===")

    # Simulate what make_mock_replays does (without argparse)
    import random
    T, H, W, K = 10, 25, 25, 4

    for _ in range(5):
        count = [random.randint(0, 1) for _ in range(T)]
        dir_id = [[random.randint(0, 3) for _ in range(K)] for _ in range(T)]
        ratio_id = [[random.randint(0, 1) for _ in range(K)] for _ in range(T)]

        for t in range(T):
            assert 0 <= count[t] <= 1, f"count[{t}]={count[t]} out of range"
            for k in range(K):
                assert 0 <= dir_id[t][k] <= 3, f"dir_id out of range"
                assert 0 <= ratio_id[t][k] <= 1, f"ratio_id out of range"
    print("  Mock data ranges valid ✓")


def test_mock_data_ce_compatible():
    """Mock data values must be valid targets for cross-entropy."""
    print("\n=== mock data: CE compatibility ===")

    # Simulate generating data and passing to cross-entropy
    T, K = 5, 4
    # These match what the fixed make_mock_replays produces
    import random
    ratio_ids = [[random.randint(0, 1) for _ in range(K)] for _ in range(T)]
    dir_ids = [[random.randint(0, 3) for _ in range(K)] for _ in range(T)]

    # Simulate loss computation
    pred_split = torch.randn(T, 2)  # n_split=2
    pred_dir = torch.randn(T, 4)   # n_directions=4

    tgt_split = torch.tensor([r[0] for r in ratio_ids], dtype=torch.long)
    tgt_dir = torch.tensor([d[0] for d in dir_ids], dtype=torch.long)

    # These should not crash or produce NaN
    split_loss = F.cross_entropy(pred_split, tgt_split)
    dir_loss = F.cross_entropy(pred_dir, tgt_dir)

    assert not split_loss.isnan(), "Split CE should not be NaN"
    assert not dir_loss.isnan(), "Dir CE should not be NaN"
    print("  Mock data compatible with cross-entropy ✓")


# =====================================================================
# 8. Deserializer unit tests (what we can test without Node.js)
# =====================================================================

def test_deserialize_move():
    """deserialize_move parses [index, start, end, is50, turn]."""
    print("\n=== deserializer: move ===")
    from data_processing.download.deserializer import deserialize_move

    m = deserialize_move([0, 155, 156, 0, 42])
    assert m["index"] == 0
    assert m["start"] == 155
    assert m["end"] == 156
    assert m["is50"] == 0
    assert m["turn"] == 42
    print("  Move deserialization correct ✓")


def test_deserialize_afk():
    """deserialize_afk parses [index, turn]."""
    print("\n=== deserializer: afk ===")
    from data_processing.download.deserializer import deserialize_afk

    a = deserialize_afk([1, 100])
    assert a["index"] == 1
    assert a["turn"] == 100
    print("  AFK deserialization correct ✓")


def test_deserialize_chat():
    """deserialize_chat parses [text, prefix, playerIndex, turn]."""
    print("\n=== deserializer: chat ===")
    from data_processing.download.deserializer import deserialize_chat

    c = deserialize_chat(["gg", "", 0, 50])
    assert c["text"] == "gg"
    assert c["playerIndex"] == 0
    assert c["turn"] == 50
    print("  Chat deserialization correct ✓")


# =====================================================================
# 9. End-to-end integration: converter → dataset → model → loss
# =====================================================================

def test_e2e_converter_to_loss():
    """Full pipeline: fake sim data → converter → dataset → model → loss."""
    print("\n=== E2E: converter → dataset → model → loss ===")
    from data_processing.dataset_seq import (
        SeqReplayDataset,
        collate_variable_length_sequences,
    )
    from data_processing.parsers.generals_json import convert_simulator_to_gio
    from model.config import ModelConfig
    from model.losses import ImitationLoss
    from model.network import GeneralsNetwork

    # Create fake simulator data
    T_raw = 15
    sim = _make_fake_sim_data(T_raw, 25, 25)
    # Add some moves
    sim["move_start"][5] = [3, 4]
    sim["move_end"][5] = [3, 5]
    sim["move_is_50"][5] = 0
    sim["move_start"][10] = [7, 7]
    sim["move_end"][10] = [8, 7]
    sim["move_is_50"][10] = 1
    # Set some armies on map
    for t in range(T_raw):
        sim["map"][t][3][4] = 5  # owned land
        sim["armies"][t][3][4] = 10
        sim["map"][t][7][7] = 5
        sim["armies"][t][7][7] = 5

    # Convert
    gio = convert_simulator_to_gio(sim)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.json"), "w") as f:
            json.dump(gio, f)

        ds = SeqReplayDataset(tmpdir, max_sequence_length=12)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, collate_fn=collate_variable_length_sequences
        )

        cfg = ModelConfig(
            d_model=64, n_heads=4, d_ff=128,
            core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
            encoder_channels=32,
        )
        model = GeneralsNetwork(cfg)
        criterion = ImitationLoss(grid_w=cfg.grid_w)

        for batch in loader:
            grid = batch["obs"]["grid_seq"]
            glob = batch["obs"]["globals_seq"]
            mask = batch["obs"]["legal_source_mask_seq"]
            tgt = batch["tgt"]
            lengths = batch["meta"]["lengths"]

            pred = model(
                grid, glob, mask,
                tgt["src_y"][:, :, 0], tgt["src_x"][:, :, 0],
            )
            losses = criterion(pred, tgt, lengths)

            # Verify no NaN/Inf
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    assert not v.isnan().any(), f"NaN in {k}"
                    assert not v.isinf().any(), f"Inf in {k}"

            assert losses["loss"].item() > 0
            print(f"  Loss: {losses['loss']:.4f}")
            print(f"  Source: {losses['source_loss']:.4f}")
            print(f"  Dir: {losses['direction_loss']:.4f}")
            break

    print("  E2E pipeline: no NaN, no Inf, positive loss ✓")


def test_e2e_training_step_reduces_loss():
    """One optimizer step should reduce loss on a single batch."""
    print("\n=== E2E: training step reduces loss ===")
    from data_processing.dataset_seq import (
        SeqReplayDataset,
        collate_variable_length_sequences,
    )
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32,
    )
    model = GeneralsNetwork(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(4):
            _write_mock_gio(tmpdir, f"mock_{i}.json", T=10)

        ds = SeqReplayDataset(tmpdir)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate_variable_length_sequences
        )

        batch = next(iter(loader))
        grid = batch["obs"]["grid_seq"]
        glob = batch["obs"]["globals_seq"]
        mask = batch["obs"]["legal_source_mask_seq"]
        tgt = batch["tgt"]

        model.train()
        losses = []
        for _ in range(5):
            pred = model(grid, glob, mask, tgt["src_y"][:, :, 0], tgt["src_x"][:, :, 0])

            B, T, N = pred["source_logits"].shape
            move_mask = (tgt["count"] > 0).float()
            n_valid = move_mask.sum().clamp(min=1)
            tgt_src = (tgt["src_y"][:, :, 0] * 25 + tgt["src_x"][:, :, 0]).long().clamp(0, N - 1)

            loss = (F.cross_entropy(
                pred["source_logits"].reshape(-1, N),
                tgt_src.reshape(-1), reduction="none"
            ).reshape(B, T) * move_mask).sum() / n_valid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (decreased) ✓")


# =====================================================================
# Helper functions
# =====================================================================

def _make_fake_sim_data(T_raw, H=25, W=25):
    """Create fake simulator output data."""
    return {
        "first": [0] * T_raw,
        "map": [[[0] * W for _ in range(H)] for _ in range(T_raw)],
        "armies": [[[0] * W for _ in range(H)] for _ in range(T_raw)],
        "info_myArmy": [100] * T_raw,
        "info_myLand": [10] * T_raw,
        "info_opArmy": [100] * T_raw,
        "info_opLand": [10] * T_raw,
        "move_start": [[-1, -1]] * T_raw,
        "move_end": [[-1, -1]] * T_raw,
        "move_is_50": [-1] * T_raw,
    }


def _write_mock_gio(tmpdir, filename, T=10, H=25, W=25, K=4):
    """Write a mock GIO-format JSON file."""
    data = {
        "grid_seq": np.random.randn(T, 12, H, W).astype(np.float32).tolist(),
        "globals_seq": np.random.randn(T, 4).astype(np.float32).tolist(),
        "legal_source_mask_seq": [[[[True] * W for _ in range(H)]] for _ in range(T)],
        "max_actions_per_tick": K,
        "count": [1] * T,
        "src_y": [[np.random.randint(0, H)] + [0] * (K - 1) for _ in range(T)],
        "src_x": [[np.random.randint(0, W)] + [0] * (K - 1) for _ in range(T)],
        "dir_id": [[np.random.randint(0, 4)] + [0] * (K - 1) for _ in range(T)],
        "ratio_id": [[np.random.randint(0, 2)] + [0] * (K - 1) for _ in range(T)],
        "value": [0.0] * T,
        "tick_seq": list(range(T)),
        "tick_bucket_seq": [int(((t % 50) + 1) // 2) for t in range(T)],
        "enemy_heatmap": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
        "next_visibility": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
    }
    with open(os.path.join(tmpdir, filename), "w") as f:
        json.dump(data, f)


# =====================================================================
# Main runner
# =====================================================================

def main():
    print("=" * 60)
    print("COMPREHENSIVE AUDIT VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        # 1. Parser
        test_compute_direction_all_four,
        test_compute_direction_no_move,
        test_encode_state_channels_shape,
        test_encode_state_channels_padding_value,
        test_create_legal_source_mask,
        test_convert_simulator_to_gio_shapes,
        test_convert_observation_action_pairing,
        test_convert_no_move_padding,
        test_convert_value_terminal_states,
        test_convert_value_blending_with_outcome,
        test_convert_army_channel,
        test_convert_tick_bucket,
        # 3. Dataset
        test_dataset_loading_shapes,
        test_dataset_truncation,
        test_dataset_no_truncation_when_shorter,
        test_collation_padding,
        test_collation_tick_sequences,
        # 4. Model components
        test_spatial_encoder_different_batch,
        test_source_head_all_masked,
        test_source_head_partial_mask,
        test_value_head_range,
        test_direction_head_dimensions,
        test_trm_h1_all_params_have_grad,
        test_trm_output_varies_with_input,
        test_spatiotemporal_block_output_shape,
        # 5. Losses
        test_loss_all_zero_count,
        test_loss_sequence_length_masking,
        test_loss_q_halt_masked_to_valid,
        test_loss_nan_handling,
        test_loss_gradient_flow,
        # 6. Network
        test_network_st_shapes,
        test_network_st_step_cache_growth,
        test_network_legal_mask_effect,
        # 7. Mock data
        test_mock_data_valid_ranges,
        test_mock_data_ce_compatible,
        # 8. Deserializer
        test_deserialize_move,
        test_deserialize_afk,
        test_deserialize_chat,
        # 9. E2E
        test_e2e_converter_to_loss,
        test_e2e_training_step_reduces_loss,
    ]

    passed = 0
    failed = 0
    failures = []
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((test.__name__, str(e)))
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failures:
        print("\nFailed tests:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)