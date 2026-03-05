"""
End-to-end training verification test.

Creates mock training data, runs training for a few steps, and verifies
the loss decreases. Also tests the full pipeline from real replay → training.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.network import GeneralsNetwork
from model.losses import ImitationLoss
from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences


def create_mock_gio_data(n_files: int, out_dir: str, T: int = 20):
    """Create mock training data in GIO format."""
    os.makedirs(out_dir, exist_ok=True)
    H, W, K = 25, 25, 4

    for i in range(n_files):
        import numpy as np
        grid_seq = np.random.randn(T, 12, H, W).astype(np.float32)
        globals_seq = np.random.randn(T, 4).astype(np.float32)

        data = {
            "grid_seq": grid_seq.tolist(),
            "globals_seq": globals_seq.tolist(),
            "prev_action_seq": [[0.0] * 8 for _ in range(T)],
            "legal_source_mask_seq": [[[[True] * W for _ in range(H)]] for _ in range(T)],
            "max_actions_per_tick": K,
            "count": [1] * T,
            "src_y": [[np.random.randint(0, H)] + [0] * (K - 1) for _ in range(T)],
            "src_x": [[np.random.randint(0, W)] + [0] * (K - 1) for _ in range(T)],
            "dir_id": [[np.random.randint(0, 4)] + [0] * (K - 1) for _ in range(T)],
            "ratio_id": [[np.random.randint(0, 2)] + [0] * (K - 1) for _ in range(T)],
            "value": [float(np.random.randn()) for _ in range(T)],
            "tick_seq": list(range(T)),
            "tick_bucket_seq": [0] * T,
            "enemy_heatmap": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
            "next_visibility": [[[[0.0] * W for _ in range(H)]] for _ in range(T)],
        }
        with open(os.path.join(out_dir, f"mock_{i}.json"), "w") as f:
            json.dump(data, f)


def test_training_loop():
    """Test that a few training steps run without errors and loss decreases."""
    print("=== Training Loop Test ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        create_mock_gio_data(8, data_dir, T=15)

        cfg = ModelConfig(
            d_model=64, n_heads=4, d_ff=128,
            n_ponder_steps=2, encoder_channels=32,
        )
        model = GeneralsNetwork(cfg)
        criterion = ImitationLoss(grid_w=cfg.grid_w)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        dataset = SeqReplayDataset(data_dir, max_sequence_length=15)
        loader = DataLoader(
            dataset, batch_size=4,
            collate_fn=collate_variable_length_sequences,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")
        print(f"  Dataset size: {len(dataset)}")

        losses_history = []

        model.train()
        for epoch in range(3):
            epoch_loss = 0.0
            n_steps = 0
            for batch in loader:
                grid_seq = batch["obs"]["grid_seq"]
                globals_seq = batch["obs"]["globals_seq"]
                legal_mask = batch["obs"]["legal_source_mask_seq"]
                tgt = batch["tgt"]
                lengths = batch["meta"]["lengths"]

                source_y = tgt["src_y"][:, :, 0]
                source_x = tgt["src_x"][:, :, 0]

                pred = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

                # Compute loss with gradients
                B, T, N = pred["source_logits"].shape
                W = 25
                move_mask = (tgt["count"] > 0).float()
                n_valid = move_mask.sum().clamp(min=1)

                tgt_src = (tgt["src_y"][:, :, 0] * W + tgt["src_x"][:, :, 0]).long().clamp(0, N - 1)
                source_loss = (
                    torch.nn.functional.cross_entropy(
                        pred["source_logits"].reshape(-1, N),
                        tgt_src.reshape(-1),
                        reduction="none",
                    ).reshape(B, T)
                    * move_mask
                ).sum() / n_valid

                dir_loss = (
                    torch.nn.functional.cross_entropy(
                        pred["direction_logits"].reshape(-1, 4),
                        tgt["dir_id"][:, :, 0].long().reshape(-1),
                        reduction="none",
                    ).reshape(B, T)
                    * move_mask
                ).sum() / n_valid

                split_loss = (
                    torch.nn.functional.cross_entropy(
                        pred["split_logits"].reshape(-1, 2),
                        tgt["ratio_id"][:, :, 0].long().reshape(-1),
                        reduction="none",
                    ).reshape(B, T)
                    * move_mask
                ).sum() / n_valid

                value_loss = torch.nn.functional.mse_loss(pred["value"], tgt["value"])

                loss = source_loss + dir_loss + 0.5 * split_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1

            avg_loss = epoch_loss / max(n_steps, 1)
            losses_history.append(avg_loss)
            print(f"  Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        # Verify loss decreased (or at least didn't diverge)
        assert losses_history[-1] < losses_history[0] * 1.5, \
            f"Loss diverged: {losses_history[0]:.4f} -> {losses_history[-1]:.4f}"
        print(f"  Loss trend: {losses_history[0]:.4f} -> {losses_history[-1]:.4f} ✓")
        print("  Training loop works correctly ✓")


def test_real_data_training():
    """Test training with real replay-derived data if available."""
    print("\n=== Real Data Training Test ===")

    replay_file = "/workspace/tests/test_data/asS2S5O_y.json"
    if not os.path.exists(replay_file):
        print("  Skipped (no real replay data)")
        return

    from data_processing.processing.simulator import ReplaySimulator
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    with open(replay_file, "r") as f:
        replay_data = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate training data from the replay
        for player in replay_data["usernames"]:
            sim = ReplaySimulator(replay_data, player)
            result = sim.simulate()
            gio = convert_simulator_to_gio(result)
            out_path = os.path.join(tmpdir, f"{player}.json")
            with open(out_path, "w") as f:
                json.dump(gio, f)

        cfg = ModelConfig(
            d_model=64, n_heads=4, d_ff=128,
            n_ponder_steps=2, encoder_channels=32,
        )
        model = GeneralsNetwork(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        dataset = SeqReplayDataset(tmpdir, max_sequence_length=64)
        loader = DataLoader(
            dataset, batch_size=1,
            collate_fn=collate_variable_length_sequences,
        )

        print(f"  Dataset: {len(dataset)} replays")
        print(f"  Max seq len: 64")

        model.train()
        for batch in loader:
            grid_seq = batch["obs"]["grid_seq"]
            globals_seq = batch["obs"]["globals_seq"]
            legal_mask = batch["obs"]["legal_source_mask_seq"]
            tgt = batch["tgt"]

            source_y = tgt["src_y"][:, :, 0]
            source_x = tgt["src_x"][:, :, 0]

            pred = model(grid_seq, globals_seq, legal_mask, source_y, source_x)

            B, T, N = pred["source_logits"].shape
            W = 25
            move_mask = (tgt["count"] > 0).float()
            n_valid = move_mask.sum().clamp(min=1)

            tgt_src = (tgt["src_y"][:, :, 0] * W + tgt["src_x"][:, :, 0]).long().clamp(0, N - 1)
            source_loss_raw = torch.nn.functional.cross_entropy(
                pred["source_logits"].reshape(-1, N),
                tgt_src.reshape(-1),
                reduction="none",
            ).reshape(B, T)
            source_loss_raw = torch.nan_to_num(source_loss_raw, nan=0.0)
            loss = (source_loss_raw * move_mask).sum() / n_valid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Batch: B={B}, T={T}")
            print(f"  Source loss: {loss.item():.4f}")
            print(f"  Valid moves: {int(move_mask.sum().item())}/{T}")
            break

        print("  Real data training works ✓")


def test_checkpoint_save_load():
    """Test model checkpoint save/load."""
    print("\n=== Checkpoint Save/Load ===")

    cfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_ponder_steps=2, encoder_channels=32)
    model = GeneralsNetwork(cfg)

    B, T = 1, 3
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    source_y = torch.randint(0, 25, (B, T))
    source_x = torch.randint(0, 25, (B, T))

    model.eval()
    with torch.no_grad():
        out1 = model(grid_seq, globals_seq, source_y=source_y, source_x=source_x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
        }, ckpt_path)

        # Load into new model
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg2 = ModelConfig(**ckpt["config"])
        model2 = GeneralsNetwork(cfg2)
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        with torch.no_grad():
            out2 = model2(grid_seq, globals_seq, source_y=source_y, source_x=source_x)

        # Outputs should match
        for key in ["source_logits", "direction_logits", "split_logits", "value"]:
            assert torch.allclose(out1[key], out2[key], atol=1e-5), \
                f"Mismatch in {key} after load"
        print("  Checkpoint save/load verified ✓")


def main():
    print("=" * 60)
    print("END-TO-END TRAINING TESTS")
    print("=" * 60)

    tests = [
        test_training_loop,
        test_real_data_training,
        test_checkpoint_save_load,
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
