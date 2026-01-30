"""
Test script for the TRM model.

Verifies:
1. Model instantiation
2. Forward pass shapes
3. Integration with existing dataset
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.config import ModelConfig
from model.model import GeneralsModel


def test_model_instantiation():
    """Test that model can be instantiated."""
    print("Testing model instantiation...")

    config = ModelConfig()
    model = GeneralsModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  ✓ Model instantiation successful")

    return model


def test_forward_pass():
    """Test forward pass with random data."""
    print("\nTesting forward pass...")

    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    # Create dummy batch
    B, T = 2, 10
    C, H, W = config.grid_channels, config.grid_height, config.grid_width
    G = config.global_features

    grid_seq = torch.randn(B, T, C, H, W)
    globals_seq = torch.randn(B, T, G)
    tick_bucket_seq = torch.randint(0, 26, (B, T))

    # Forward pass (no legal_mask needed - uses penalty-based approach)
    with torch.no_grad():
        outputs = model(
            grid_seq=grid_seq,
            globals_seq=globals_seq,
            tick_bucket_seq=tick_bucket_seq
        )

    # Check output shapes
    assert outputs['source_logits'].shape == (B, T, H * W), \
        f"Expected source_logits shape {(B, T, H*W)}, got {outputs['source_logits'].shape}"
    assert outputs['dir_logits'].shape == (B, T, 4), \
        f"Expected dir_logits shape {(B, T, 4)}, got {outputs['dir_logits'].shape}"
    assert outputs['ratio_logits'].shape == (B, T, 2), \
        f"Expected ratio_logits shape {(B, T, 2)}, got {outputs['ratio_logits'].shape}"
    assert outputs['value'].shape == (B, T), \
        f"Expected value shape {(B, T)}, got {outputs['value'].shape}"
    assert outputs['halt_probs'].shape == (B, T, config.n_supervision_steps), \
        f"Expected halt_probs shape {(B, T, config.n_supervision_steps)}, got {outputs['halt_probs'].shape}"

    print(f"  source_logits: {outputs['source_logits'].shape}")
    print(f"  dir_logits: {outputs['dir_logits'].shape}")
    print(f"  ratio_logits: {outputs['ratio_logits'].shape}")
    print(f"  value: {outputs['value'].shape}")
    print(f"  halt_probs: {outputs['halt_probs'].shape}")
    print("  ✓ Forward pass successful")


def test_inference():
    """Test single-frame inference."""
    print("\nTesting inference...")

    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    B = 1
    C, H, W = config.grid_channels, config.grid_height, config.grid_width
    G = config.global_features

    grid = torch.randn(B, C, H, W)
    globals_vec = torch.randn(B, G)
    history_latent = model.get_initial_latent(B, grid.device)
    tick_bucket = torch.tensor([0])

    # Inference (no legal_mask needed - uses penalty-based approach)
    with torch.no_grad():
        output = model.inference(
            grid=grid,
            globals_vec=globals_vec,
            history_latent=history_latent,
            tick_bucket=tick_bucket
        )

    print(f"  source_probs: {output['source_probs'].shape}")
    print(f"  dir_probs: {output['dir_probs'].shape}")
    print(f"  ratio_probs: {output['ratio_probs'].shape}")
    print(f"  value: {output['value'].shape}")
    print(f"  action: {output['action']}")
    print("  ✓ Inference successful")


def test_history_latent_update():
    """Test incremental history latent update."""
    print("\nTesting history latent update...")

    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    B = 1
    C, H, W = config.grid_channels, config.grid_height, config.grid_width
    G = config.global_features

    # Initial latent
    latent = model.get_initial_latent(B, torch.device('cpu'))
    print(f"  Initial latent shape: {latent.shape}")

    # Update with new observation
    grid = torch.randn(B, C, H, W)
    globals_vec = torch.randn(B, G)

    with torch.no_grad():
        new_latent = model.update_history_latent(latent, grid, globals_vec)

    print(f"  Updated latent shape: {new_latent.shape}")
    print(f"  Latent changed: {not torch.allclose(latent, new_latent)}")
    print("  ✓ History latent update successful")


def test_with_dataset():
    """Test model with existing dataset if available."""
    print("\nTesting with existing dataset...")

    from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
    from torch.utils.data import DataLoader

    # Try to find training data
    data_dirs = [
        project_root / "data" / "training",
        project_root / "data" / "my_replays" / "training",
    ]

    data_dir = None
    for d in data_dirs:
        if d.exists():
            # Look for player subdirectories
            subdirs = [x for x in d.iterdir() if x.is_dir()]
            if subdirs:
                # Find first subdir with JSON files
                for sd in subdirs:
                    if list(sd.glob("*.json")):
                        data_dir = sd
                        break
        if data_dir:
            break

    if data_dir is None:
        print("  No training data found, skipping dataset test")
        return

    print(f"  Found data in: {data_dir}")

    # Load dataset
    dataset = SeqReplayDataset(str(data_dir), max_sequence_length=20)
    print(f"  Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("  Dataset is empty, skipping")
        return

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_variable_length_sequences
    )

    # Get a batch
    batch = next(iter(dataloader))

    print(f"  Batch grid_seq shape: {batch['obs']['grid_seq'].shape}")
    print(f"  Batch globals_seq shape: {batch['obs']['globals_seq'].shape}")

    # Test with model
    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    # Prepare batch for model
    B, T = batch['obs']['grid_seq'].shape[:2]

    # Compute target source indices
    src_y = batch['tgt']['src_y'][:, :, 0]  # Take first action
    src_x = batch['tgt']['src_x'][:, :, 0]
    target_source_idx = (src_y * 25 + src_x).long()

    # Forward pass (no legal_mask needed - uses penalty-based approach)
    with torch.no_grad():
        outputs = model(
            grid_seq=batch['obs']['grid_seq'],
            globals_seq=batch['obs']['globals_seq'],
            tick_bucket_seq=batch['meta']['tick_bucket_seq'].long(),
            target_source_idx=target_source_idx
        )

    print(f"  Output value range: [{outputs['value'].min():.3f}, {outputs['value'].max():.3f}]")
    print("  ✓ Dataset integration successful")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRM Model Tests")
    print("=" * 60)

    try:
        test_model_instantiation()
        test_forward_pass()
        test_inference()
        test_history_latent_update()
        test_with_dataset()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
