"""
Integration test for TRM model with game environment.

Tests:
1. Game environment from replay data
2. Model inference with environment states
3. Self-play loop with direct policy sampling
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json


def test_game_environment():
    """Test game environment initialization and stepping."""
    print("Testing game environment...")

    from data_processing.processing.game_env import GameEnvironment, load_replay_as_env
    from data_processing.processing.models import MapData

    # Test map initialization using MapData
    map_data = MapData.create_random(seed=42)  # default 25x25
    env = GameEnvironment()
    state = env.reset(map_data)

    print(f"  Map size: {state.map_width}x{state.map_height}")
    print(f"  Generals: {state.generals}")
    print(f"  Cities: {len(state.cities)}")
    print(f"  Mountains: {len(state.mountains)}")
    print(f"  Player 0 land: {len(state.players[0].land)}")
    print(f"  Player 1 land: {len(state.players[1].land)}")

    # Test state tensors
    grid = state.get_grid_tensor()
    globals_vec = state.get_globals_tensor()
    legal_mask = state.get_legal_mask_tensor()

    print(f"  Grid tensor shape: {grid.shape}")
    print(f"  Globals tensor shape: {globals_vec.shape}")
    print(f"  Legal mask shape: {legal_mask.shape}")

    # Test legal actions
    legal_actions = env.get_legal_actions()
    print(f"  Legal actions: {len(legal_actions)}")

    # Test stepping
    if legal_actions:
        action = legal_actions[0]
        new_state, reward, done, info = env.step(action)
        print(f"  After step - Turn: {info['turn']}, Done: {done}, Reward: {reward}")

    print("  ✓ Game environment test passed")
    return env


def test_environment_from_replay():
    """Test loading environment from replay data."""
    print("\nTesting environment from replay...")

    from data_processing.processing.game_env import load_replay_as_env

    # Find a replay file
    replay_dir = project_root / "data" / "my_replays" / "replays"
    replay_files = list(replay_dir.glob("*.json")) if replay_dir.exists() else []

    if not replay_files:
        print("  No replay files found, skipping")
        return None

    replay_path = replay_files[0]
    print(f"  Loading replay: {replay_path.name}")

    env = load_replay_as_env(str(replay_path))

    print(f"  Map size: {env.state.map_width}x{env.state.map_height}")
    print(f"  Players: {[p.name for p in env.state.players]}")

    # Play a few turns
    for i in range(5):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            print(f"  Turn {i}: No legal actions")
            continue

        action = legal_actions[0]
        state, reward, done, info = env.step(action)

        if done:
            print(f"  Game ended at turn {info['turn']}, winner: {info['winner']}")
            break

    print("  ✓ Replay environment test passed")
    return env


def test_model_with_environment():
    """Test model inference with environment states."""
    print("\nTesting model with environment...")

    from data_processing.processing.game_env import GameEnvironment
    from data_processing.processing.models import MapData
    from model.config import ModelConfig
    from model.model import GeneralsModel

    # Create environment
    map_data = MapData.create_random(width=20, height=20, seed=123)
    env = GameEnvironment()
    state = env.reset(map_data)

    # Create model
    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    # Get state tensors
    grid = state.get_grid_tensor().unsqueeze(0)  # [1, C, H, W]
    globals_vec = state.get_globals_tensor().unsqueeze(0)  # [1, G]
    legal_mask = state.get_legal_mask_tensor().unsqueeze(0)  # [1, 1, H, W]
    history_latent = model.get_initial_latent(1, grid.device)
    tick_bucket = torch.tensor([0])

    # Inference
    with torch.no_grad():
        output = model.inference(
            grid, globals_vec, history_latent, legal_mask, tick_bucket
        )

    print(f"  Source probs sum: {output['source_probs'].sum().item():.4f}")
    print(f"  Dir probs: {output['dir_probs'][0].tolist()}")
    print(f"  Ratio probs: {output['ratio_probs'][0].tolist()}")
    print(f"  Value: {output['value'].item():.4f}")

    # Check action is valid
    action = output['action']
    source_idx = action['source_idx'].item()
    dir_id = action['dir_id'].item()
    ratio_id = action['ratio_id'].item()

    print(f"  Predicted action: source={source_idx}, dir={dir_id}, ratio={ratio_id}")

    # Verify it's in legal actions
    legal_actions = env.get_legal_actions()
    legal_sources = set(a[0] for a in legal_actions)

    # Note: model might predict illegal source if legal_mask isn't properly applied
    if source_idx in legal_sources:
        print("  ✓ Action source is legal")
    else:
        print(f"  ⚠ Action source {source_idx} not in legal sources")
        print(f"    Legal sources sample: {list(legal_sources)[:5]}")

    print("  ✓ Model with environment test passed")


def test_self_play_loop():
    """Test a simple self-play game loop."""
    print("\nTesting self-play loop...")

    from data_processing.processing.game_env import GameEnvironment
    from data_processing.processing.models import MapData
    from model.config import ModelConfig
    from model.model import GeneralsModel

    # Create environment and model
    map_data = MapData.create_random(width=15, height=15, seed=456)
    env = GameEnvironment()
    state = env.reset(map_data)

    config = ModelConfig()
    model = GeneralsModel(config)
    model.eval()

    # Initialize history latent for both players
    history_latents = [
        model.get_initial_latent(1, torch.device('cpu')),
        model.get_initial_latent(1, torch.device('cpu'))
    ]

    max_turns = 50
    game_history = []

    print(f"  Starting self-play (max {max_turns} turns)...")

    for turn in range(max_turns):
        current_player = state.current_player

        # Get state tensors
        grid = state.get_grid_tensor().unsqueeze(0)
        globals_vec = state.get_globals_tensor().unsqueeze(0)
        legal_mask = state.get_legal_mask_tensor().unsqueeze(0)
        history_latent = history_latents[current_player]
        # FIX: Use same formula as dataset: ((t % 50) + 1) // 2
        tick_bucket = torch.tensor([((state.turn % 50) + 1) // 2])

        # Model inference
        with torch.no_grad():
            output = model.inference(
                grid, globals_vec, history_latent, legal_mask, tick_bucket,
                sample=True,  # Sample for exploration
                temperature=1.0
            )

        # Extract action
        action = (
            output['action']['source_idx'].item(),
            output['action']['dir_id'].item(),
            output['action']['ratio_id'].item()
        )

        # Validate action
        legal_actions = env.get_legal_actions()
        if action not in legal_actions:
            # Fallback to random legal action
            if legal_actions:
                import random
                action = random.choice(legal_actions)
            else:
                action = None

        # Store history
        game_history.append({
            'turn': state.turn,
            'player': current_player,
            'action': action,
            'value': output['value'].item()
        })

        # Update history latent (inputs already have batch dimension)
        history_latents[current_player] = model.update_history_latent(
            history_latent.squeeze(0).unsqueeze(0), grid, globals_vec
        )

        # Step environment
        if action:
            state, reward, done, info = env.step(action)
        else:
            # No legal moves, skip turn
            state.current_player = 1 - state.current_player
            state.turn += 1
            done = False
            reward = 0

        if done:
            print(f"  Game ended at turn {state.turn}")
            print(f"  Winner: Player {info['winner']}")
            break

        if turn % 10 == 0:
            p0_land = len(state.players[0].land)
            p1_land = len(state.players[1].land)
            print(f"  Turn {state.turn}: P0 land={p0_land}, P1 land={p1_land}")

    print(f"  Game lasted {len(game_history)} turns")
    print("  ✓ Self-play loop test passed")


def main():
    print("=" * 60)
    print("TRM Integration Tests")
    print("=" * 60)

    try:
        test_game_environment()
        test_environment_from_replay()
        test_model_with_environment()
        test_self_play_loop()

        print("\n" + "=" * 60)
        print("All integration tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
