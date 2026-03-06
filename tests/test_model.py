"""
Comprehensive tests for the generals.io neural network.

Tests each module independently and then the full end-to-end pipeline,
including forward pass, backward pass, loss computation, and dataset integration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model.components.encoder import HistoryEncoder, ResBlock, SpatialEncoder
from model.components.heads import DirectionHead, SourceHead, SplitHead, ValueHead
from model.components.transformer import SpatiotemporalBlock, TransformerBlock
from model.components.world_model import WorldModel
from model.config import ModelConfig
from model.losses import ImitationLoss
from model.network import GeneralsNetwork


def test_resblock():
    print("=== ResBlock ===")
    block = ResBlock(64)
    x = torch.randn(2, 64, 25, 25)
    y = block(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    # Residual: output should not be identical to input
    assert not torch.allclose(x, y), "ResBlock output equals input (no effect)"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} ✓")


def test_spatial_encoder():
    print("\n=== SpatialEncoder ===")
    cfg = ModelConfig(d_model=128, encoder_channels=64)
    enc = SpatialEncoder(cfg)
    x = torch.randn(4, 12, 25, 25)
    y = enc(x)
    assert y.shape == (4, 128, 25, 25), f"Shape mismatch: {y.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} ✓")
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Params: {n_params:,}")


def test_history_encoder():
    print("\n=== HistoryEncoder ===")
    cfg = ModelConfig(d_model=128)
    enc = HistoryEncoder(cfg)
    x = torch.randn(2, 10, 128)  # B=2, T=10, d=128
    out, h = enc(x)
    assert out.shape == (2, 10, 128), f"Output shape: {out.shape}"
    assert h.shape == (1, 2, 128), f"Hidden shape: {h.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}, Hidden: {h.shape} ✓")

    # Test with initial hidden state
    out2, h2 = enc(x, h)
    assert out2.shape == out.shape
    print(f"  With hidden: {out2.shape} ✓")


def test_transformer_block():
    print("\n=== TransformerBlock ===")
    cfg = ModelConfig(d_model=128, n_heads=4, d_ff=256, dropout=0.0)
    block = TransformerBlock(cfg)
    x = torch.randn(2, 627, 128)  # B=2, N=625+2, d=128
    y = block(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} ✓")

    # Test pondering (applying the same block multiple times)
    z = x
    for _ in range(4):
        z = block(z)
    assert z.shape == x.shape
    print(f"  After 4 ponder steps: {z.shape} ✓")


def test_source_head():
    print("\n=== SourceHead ===")
    head = SourceHead(128)
    tokens = torch.randn(4, 625, 128)
    logits = head(tokens)
    assert logits.shape == (4, 625), f"Shape: {logits.shape}"
    print(f"  Input:  {tokens.shape}")
    print(f"  Output: {logits.shape} ✓")

    # With legal mask
    mask = torch.zeros(4, 625, dtype=torch.bool)
    mask[:, :10] = True  # only first 10 positions legal
    logits_masked = head(tokens, mask)
    assert logits_masked.shape == (4, 625)
    # Masked positions should be -inf
    assert (logits_masked[:, 10:] == float("-inf")).all(), "Masked positions not -inf"
    # Can still softmax
    probs = F.softmax(logits_masked, dim=-1)
    assert probs[:, 10:].sum() < 1e-6, "Masked positions have nonzero probability"
    print("  Masked output ✓ (10 legal positions)")


def test_direction_head():
    print("\n=== DirectionHead ===")
    head = DirectionHead(128, 4)
    source_embed = torch.randn(4, 128)
    decision = torch.randn(4, 128)
    logits = head(source_embed, decision)
    assert logits.shape == (4, 4), f"Shape: {logits.shape}"
    print(f"  Input:  source={source_embed.shape}, decision={decision.shape}")
    print(f"  Output: {logits.shape} ✓")


def test_split_head():
    print("\n=== SplitHead ===")
    head = SplitHead(128, 2)
    source_embed = torch.randn(4, 128)
    decision = torch.randn(4, 128)
    logits = head(source_embed, decision)
    assert logits.shape == (4, 2), f"Shape: {logits.shape}"
    print(f"  Output: {logits.shape} ✓")


def test_value_head():
    print("\n=== ValueHead ===")
    head = ValueHead(128)
    x = torch.randn(4, 128)
    v = head(x)
    assert v.shape == (4,), f"Shape: {v.shape}"
    assert (v >= -1).all() and (v <= 1).all(), "Value out of [-1, 1] range"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {v.shape}, range=[{v.min():.3f}, {v.max():.3f}] ✓")


def test_world_model():
    print("\n=== WorldModel ===")
    cfg = ModelConfig(d_model=128, use_world_model=True)
    wm = WorldModel(cfg)
    state = torch.randn(4, 128)
    action_dim = 25 * 25 + 4 + 2
    action = torch.randn(4, action_dim)
    pred = wm(state, action)
    assert pred.shape == (4, 128), f"Shape: {pred.shape}"
    print(f"  Input:  state={state.shape}, action={action.shape}")
    print(f"  Output: {pred.shape} ✓")


def test_generals_network_forward():
    print("\n=== GeneralsNetwork Forward (GRU mode) ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_ponder_steps=2, encoder_channels=32, dropout=0.0,
        temporal_type="gru",
    )
    model = GeneralsNetwork(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    B, T = 2, 5
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    legal_mask = torch.ones(B, T, 1, 25, 25, dtype=torch.bool)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    out = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

    assert out["source_logits"].shape == (B, T, 625), f"source: {out['source_logits'].shape}"
    assert out["direction_logits"].shape == (B, T, 4), f"dir: {out['direction_logits'].shape}"
    assert out["split_logits"].shape == (B, T, 2), f"split: {out['split_logits'].shape}"
    assert out["value"].shape == (B, T), f"value: {out['value'].shape}"
    assert out["hidden"] is not None
    print(f"  source_logits:    {out['source_logits'].shape} ✓")
    print(f"  direction_logits: {out['direction_logits'].shape} ✓")
    print(f"  split_logits:     {out['split_logits'].shape} ✓")
    print(f"  value:            {out['value'].shape} ✓")


def test_spatiotemporal_block():
    print("\n=== SpatiotemporalBlock ===")
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, dropout=0.0)
    block = SpatiotemporalBlock(cfg)
    B, T, S, d = 2, 8, 627, 64
    x = torch.randn(B, T, S, d)
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    y = block(x, causal)
    assert y.shape == (B, T, S, d), f"Shape: {y.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} ✓")

    # test forward_step with cache
    x_cur = torch.randn(B, 1, S, d)
    out1, cache1 = block.forward_step(x_cur, None)
    assert out1.shape == (B, 1, S, d)
    assert cache1.shape == (B * S, 1, d)
    out2, cache2 = block.forward_step(x_cur, cache1)
    assert cache2.shape == (B * S, 2, d)
    print("  forward_step: cache grows 1→2 ✓")


def test_spatiotemporal_forward():
    print("\n=== GeneralsNetwork Forward (spatiotemporal + TRM) ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, n_ponder_steps=3,
        encoder_channels=32, dropout=0.0,
        temporal_type="spatiotemporal", use_act=False,
    )
    model = GeneralsNetwork(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    B, T = 2, 8
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    legal_mask = torch.ones(B, T, 1, 25, 25, dtype=torch.bool)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    out = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

    assert out["source_logits"].shape == (B, T, 625)
    assert out["direction_logits"].shape == (B, T, 4)
    assert out["split_logits"].shape == (B, T, 2)
    assert out["value"].shape == (B, T)
    print(f"  source_logits:    {out['source_logits'].shape} ✓")
    print(f"  direction_logits: {out['direction_logits'].shape} ✓")
    print(f"  split_logits:     {out['split_logits'].shape} ✓")
    print(f"  value:            {out['value'].shape} ✓")


def test_spatiotemporal_causal():
    """Verify temporal attention is causal: output at t depends only on t'<=t."""
    print("\n=== Spatiotemporal Causality ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=1, n_ponder_steps=2,
        encoder_channels=32, dropout=0.0,
        temporal_type="spatiotemporal", use_act=False,
    )
    model = GeneralsNetwork(cfg)
    model.eval()

    B, T = 1, 6
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)

    with torch.no_grad():
        out_full = model(grid_seq, globals_seq)
        out_prefix = model(grid_seq[:, :3], globals_seq[:, :3])

    # output at t=0,1,2 should be identical whether T=6 or T=3
    diff = (out_full["source_logits"][:, :3] - out_prefix["source_logits"]).abs().max()
    assert diff < 1e-4, f"Causal violation: max diff = {diff:.6f}"
    print(f"  Max diff (T=6 vs T=3, first 3 steps): {diff:.6f} ✓")


def test_spatiotemporal_step():
    print("\n=== Spatiotemporal + TRM Single-step Inference ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, n_ponder_steps=3,
        encoder_channels=32, dropout=0.0,
        temporal_type="spatiotemporal", use_act=False,
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
        assert out["source_logits"].shape == (B, 625)
        assert len(cache) == 2  # n_st_layers=2
        assert cache[0].shape[1] == t + 1
    print(f"  5 sequential steps, cache sizes = {[c.shape[1] for c in cache]} ✓")


def test_spatiotemporal_backward():
    print("\n=== Spatiotemporal + TRM Backward ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, n_ponder_steps=3,
        encoder_channels=32,
        temporal_type="spatiotemporal", use_act=True,
    )
    model = GeneralsNetwork(cfg)

    B, T = 1, 5
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    out = model(grid_seq, globals_seq, source_y=source_y, source_x=source_x)
    loss = (
        out["source_logits"].mean()
        + out["direction_logits"].mean()
        + out["split_logits"].mean()
        + out["value"].mean()
    )
    loss.backward()

    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"No gradient: {no_grad}"
    print(f"  All {sum(1 for _ in model.parameters())} params have gradients ✓")


def test_generals_network_inference():
    print("\n=== GeneralsNetwork Inference (no source_y/x) ===")
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32, temporal_type="gru")
    model = GeneralsNetwork(cfg)
    model.eval()

    B, T = 1, 3
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)

    with torch.no_grad():
        out = model(grid_seq, globals_seq)

    assert out["source_logits"].shape == (B, T, 625)
    assert out["direction_logits"].shape == (B, T, 4)
    print("  Inference shapes correct ✓")


def test_generals_network_step():
    print("\n=== GeneralsNetwork Single-step (GRU) ===")
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32, temporal_type="gru")
    model = GeneralsNetwork(cfg)
    model.eval()

    B = 1
    grid = torch.randn(B, 12, 25, 25)
    globals_vec = torch.randn(B, 4)
    legal_mask = torch.ones(B, 1, 25, 25, dtype=torch.bool)

    with torch.no_grad():
        out = model.step(grid, globals_vec, legal_mask)

    assert out["source_logits"].shape == (B, 625), f"source: {out['source_logits'].shape}"
    assert out["direction_logits"].shape == (B, 4), f"dir: {out['direction_logits'].shape}"
    assert out["split_logits"].shape == (B, 2), f"split: {out['split_logits'].shape}"
    assert out["value"].shape == (B,), f"value: {out['value'].shape}"
    print(f"  source_logits:    {out['source_logits'].shape} ✓")
    print(f"  direction_logits: {out['direction_logits'].shape} ✓")
    print(f"  split_logits:     {out['split_logits'].shape} ✓")
    print(f"  value:            {out['value'].shape} ✓")

    # Test sequential inference with hidden state
    hidden = out["hidden"]
    out2 = model.step(grid, globals_vec, legal_mask, hidden=hidden)
    assert out2["hidden"] is not None
    print("  Sequential inference with hidden state ✓")


def test_backward_pass():
    print("\n=== Backward Pass ===")
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32)
    model = GeneralsNetwork(cfg)

    B, T = 2, 5
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    out = model(grid_seq, globals_seq, source_y=source_y, source_x=source_x)

    loss = (
        out["source_logits"].mean()
        + out["direction_logits"].mean()
        + out["split_logits"].mean()
        + out["value"].mean()
    )
    loss.backward()

    # Check gradients exist
    grad_count = 0
    no_grad = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
        else:
            no_grad.append(name)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {grad_count}/{total_params}")
    if no_grad:
        print(f"  Missing gradients: {no_grad}")
    assert grad_count == total_params, f"Missing gradients for {total_params - grad_count} params"
    print("  All parameters have gradients ✓")


def test_imitation_loss():
    print("\n=== ImitationLoss ===")
    criterion = ImitationLoss(grid_w=25)

    B, T, N = 2, 5, 625
    pred = {
        "source_logits": torch.randn(B, T, N),
        "direction_logits": torch.randn(B, T, 4),
        "split_logits": torch.randn(B, T, 2),
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
    lengths = [5, 3]

    losses = criterion(pred, tgt, lengths)
    assert "loss" in losses
    assert losses["loss"].requires_grad is False  # detached components
    print(f"  Total loss: {losses['loss']:.4f}")
    print(f"  Source:     {losses['source_loss']:.4f}")
    print(f"  Direction:  {losses['direction_loss']:.4f}")
    print(f"  Split:      {losses['split_loss']:.4f}")
    print(f"  Value:      {losses['value_loss']:.4f}")

    # Test with count=0 masking
    tgt["count"] = torch.zeros(B, T, dtype=torch.long)
    losses_zero = criterion(pred, tgt, lengths)
    assert losses_zero["source_loss"].item() == 0.0, "Source loss should be 0 with no moves"
    print("  Zero-count masking ✓")


def test_imitation_loss_backward():
    print("\n=== ImitationLoss Backward ===")
    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32)
    model = GeneralsNetwork(cfg)
    criterion = ImitationLoss(grid_w=25)

    B, T = 2, 5
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    legal_mask = torch.ones(B, T, 1, 25, 25, dtype=torch.bool)

    tgt = {
        "count": torch.ones(B, T, dtype=torch.long),
        "src_y": torch.randint(0, 25, (B, T, 4)),
        "src_x": torch.randint(0, 25, (B, T, 4)),
        "dir_id": torch.randint(0, 4, (B, T, 4)),
        "ratio_id": torch.randint(0, 2, (B, T, 4)),
        "value": torch.randn(B, T),
    }

    source_y = tgt["src_y"][:, :, 0]
    source_x = tgt["src_x"][:, :, 0]

    pred = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

    # Need to recompute with grad (criterion returns detached components)
    loss = _compute_loss_with_grad(pred, tgt, criterion)
    loss.backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    assert len(grad_norms) > 0, "No gradients computed"
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Grad norms: min={min(grad_norms):.6f}, max={max(grad_norms):.4f}")
    print("  Full IL backward pass ✓")


def _compute_loss_with_grad(pred, tgt, criterion):
    """Recompute loss keeping gradients (criterion returns detached components)."""
    B, T, N = pred["source_logits"].shape
    W = 25

    move_mask = (tgt["count"] > 0).float()
    n_valid = move_mask.sum().clamp(min=1)

    tgt_src = (tgt["src_y"][:, :, 0] * W + tgt["src_x"][:, :, 0]).long().clamp(0, N - 1)
    source_loss = (F.cross_entropy(
        pred["source_logits"].reshape(-1, N),
        tgt_src.reshape(-1),
        reduction="none",
    ).reshape(B, T) * move_mask).sum() / n_valid

    dir_loss = (F.cross_entropy(
        pred["direction_logits"].reshape(-1, 4),
        tgt["dir_id"][:, :, 0].long().reshape(-1),
        reduction="none",
    ).reshape(B, T) * move_mask).sum() / n_valid

    split_loss = (F.cross_entropy(
        pred["split_logits"].reshape(-1, 2),
        tgt["ratio_id"][:, :, 0].long().reshape(-1),
        reduction="none",
    ).reshape(B, T) * move_mask).sum() / n_valid

    value_loss = F.mse_loss(pred["value"], tgt["value"])

    return source_loss + dir_loss + 0.5 * split_loss + 0.5 * value_loss


def test_dataset_integration():
    """Test model with data from the actual dataset format (mock data)."""
    print("\n=== Dataset Integration ===")
    import json
    import tempfile

    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32)
    model = GeneralsNetwork(cfg)
    model.eval()

    # Create minimal mock data in the dataset format
    T = 10
    H, W = 25, 25
    K = 4

    mock_data = {
        "grid_seq": torch.randn(T, 12, H, W).tolist(),
        "globals_seq": torch.randn(T, 4).tolist(),
        "prev_action_seq": [[0.0] * 8 for _ in range(T)],
        "legal_source_mask_seq": [[[[True] * W for _ in range(H)]] for _ in range(T)],
        "max_actions_per_tick": K,
        "count": [1] * T,
        "src_y": [[5] + [0] * (K - 1) for _ in range(T)],
        "src_x": [[3] + [0] * (K - 1) for _ in range(T)],
        "dir_id": [[1] + [0] * (K - 1) for _ in range(T)],
        "ratio_id": [[0] + [0] * (K - 1) for _ in range(T)],
        "value": [0.0] * T,
        "tick_seq": list(range(T)),
        "tick_bucket_seq": [0] * T,
        "enemy_heatmap": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
        "next_visibility": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            path = os.path.join(tmpdir, f"mock_{i}.json")
            with open(path, "w") as f:
                json.dump(mock_data, f)

        from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
        dataset = SeqReplayDataset(tmpdir, max_sequence_length=10)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_variable_length_sequences,
        )

        for batch in loader:
            grid_seq = batch["obs"]["grid_seq"]
            globals_seq = batch["obs"]["globals_seq"]
            legal_mask = batch["obs"]["legal_source_mask_seq"]
            tgt = batch["tgt"]
            lengths = batch["meta"]["lengths"]

            source_y = tgt["src_y"][:, :, 0]
            source_x = tgt["src_x"][:, :, 0]

            with torch.no_grad():
                out = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

            print("  Batch shapes:")
            print(f"    grid_seq:       {grid_seq.shape}")
            print(f"    globals_seq:    {globals_seq.shape}")
            print(f"    source_logits:  {out['source_logits'].shape}")
            print(f"    dir_logits:     {out['direction_logits'].shape}")
            print(f"    split_logits:   {out['split_logits'].shape}")
            print(f"    value:          {out['value'].shape}")

            # Verify loss computation
            criterion = ImitationLoss(grid_w=25)
            losses = criterion(out, tgt, lengths)
            print(f"    Loss: {losses['loss']:.4f}")
            print("  Dataset integration ✓")
            break


def test_real_replay_integration():
    """Test with real replay data if available."""
    print("\n=== Real Replay Integration ===")
    test_data_dir = "/workspace/tests/test_data"
    replay_file = os.path.join(test_data_dir, "asS2S5O_y.json")

    if not os.path.exists(replay_file):
        print("  Skipped (no real replay data)")
        return

    import json

    from data_processing.parsers.generals_json import convert_simulator_to_gio
    from data_processing.processing.simulator import ReplaySimulator

    with open(replay_file, "r") as f:
        replay_data = json.load(f)

    player = replay_data["usernames"][0]
    sim = ReplaySimulator(replay_data, player)
    result = sim.simulate()
    gio = convert_simulator_to_gio(result)

    # Truncate for test
    max_t = 20
    grid_seq = torch.tensor(gio["grid_seq"][:max_t], dtype=torch.float32).unsqueeze(0)
    globals_seq = torch.tensor(gio["globals_seq"][:max_t], dtype=torch.float32).unsqueeze(0)
    legal_mask = torch.tensor(
        gio["legal_source_mask_seq"][:max_t], dtype=torch.bool
    ).unsqueeze(0)

    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32)
    model = GeneralsNetwork(cfg)
    model.eval()

    source_y = torch.tensor([gio["src_y"][t][0] for t in range(max_t)]).unsqueeze(0)
    source_x = torch.tensor([gio["src_x"][t][0] for t in range(max_t)]).unsqueeze(0)

    with torch.no_grad():
        out = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

    print(f"  Player: {player}")
    print(f"  Timesteps: {max_t}")
    print(f"  source_logits: {out['source_logits'].shape}")
    print(f"  value: {out['value'].shape}")
    print("  Real replay integration ✓")


def test_world_model_integration():
    print("\n=== World Model Integration ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_ponder_steps=2, encoder_channels=32,
        use_world_model=True,
    )
    model = GeneralsNetwork(cfg)
    assert model.world_model is not None, "World model not created"

    B, T = 1, 3
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    wm_embed = torch.zeros(B, T, 64)  # zeros for IL

    out = model(grid_seq, globals_seq, world_model_embed=wm_embed)
    assert out["source_logits"].shape == (B, T, 625)
    print("  Forward with zero world-model embed (IL mode) ✓")

    # With random embed (simulating RL)
    wm_embed_rl = torch.randn(B, T, 64)
    out_rl = model(grid_seq, globals_seq, world_model_embed=wm_embed_rl)
    assert out_rl["source_logits"].shape == (B, T, 625)
    print("  Forward with random world-model embed (RL mode) ✓")


def test_gradient_flow():
    """Verify gradients flow through all major components."""
    print("\n=== Gradient Flow (spatiotemporal + TRM) ===")
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, n_ponder_steps=3,
        encoder_channels=32,
        temporal_type="spatiotemporal", use_act=True,
    )
    model = GeneralsNetwork(cfg)

    B, T = 1, 3
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    out = model(grid_seq, globals_seq, source_y=source_y, source_x=source_x)

    loss = (
        out["source_logits"].mean()
        + out["direction_logits"].mean()
        + out["split_logits"].mean()
        + out["value"].mean()
        + out["act_loss"]
    )
    loss.backward()

    components = {
        "spatial_encoder": model.spatial_encoder,
        "global_proj": model.global_proj,
        "st_layers": model.st_layers,
        "trm_block": model.trm_block,
        "halt": model.halt,
        "source_head": model.source_head,
        "direction_head": model.direction_head,
        "split_head": model.split_head,
        "value_head": model.value_head,
    }

    for name, module in components.items():
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        assert has_grad, f"No gradient flow through {name}"
        print(f"  {name}: gradient ✓")

    # Check decision tokens
    assert model.hl_token.grad is not None and model.hl_token.grad.abs().sum() > 0
    assert model.ll_token.grad is not None and model.ll_token.grad.abs().sum() > 0
    print("  hl_token: gradient ✓")
    print("  ll_token: gradient ✓")

    # Check positional encodings
    assert model.row_embed.grad is not None
    assert model.col_embed.grad is not None
    print("  row_embed: gradient ✓")
    print("  col_embed: gradient ✓")


def main():
    print("=" * 60)
    print("GENERALS.IO MODEL TEST SUITE")
    print("=" * 60)

    tests = [
        test_resblock,
        test_spatial_encoder,
        test_history_encoder,
        test_transformer_block,
        test_source_head,
        test_direction_head,
        test_split_head,
        test_value_head,
        test_world_model,
        test_generals_network_forward,
        test_generals_network_inference,
        test_generals_network_step,
        test_backward_pass,
        test_imitation_loss,
        test_imitation_loss_backward,
        test_dataset_integration,
        test_real_replay_integration,
        test_world_model_integration,
        test_gradient_flow,
        # spatiotemporal tests
        test_spatiotemporal_block,
        test_spatiotemporal_forward,
        test_spatiotemporal_causal,
        test_spatiotemporal_step,
        test_spatiotemporal_backward,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
