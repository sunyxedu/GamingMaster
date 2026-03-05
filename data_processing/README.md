# Generals.io Data Processing Pipeline

Complete data processing pipeline for generals.io replays.

## Overview

This pipeline processes generals.io replays through the following stages:

1. **Download** - Fetch replay metadata and files from generals.io API and S3
2. **Filter** - Filter replays based on criteria (1v1, minimum turns, etc.)
3. **Simulate** - Simulate replays from each player's perspective with fog of war
4. **Convert** - Transform simulator output into training format

## Directory Structure

```
data_processing/
├── download/
│   ├── api_client.py       # Fetch replay metadata from API
│   ├── decompress.js       # Node.js LZ-string decompressor
│   ├── deserializer.py     # Decompress .gior files
│   └── replay_fetcher.py   # Download replay files from S3
├── processing/
│   ├── filter.py           # Filter replays by criteria
│   └── simulator.py        # Simulate game from player perspective
├── parsers/
│   └── generals_json.py    # Convert to training format
├── mock/
│   └── make_mock_replays.py # Generate mock data for testing
├── dataset_seq.py          # PyTorch dataset for loading data
└── __init__.py             # Public API exports

scripts/
└── prepare_data.py         # CLI script for data pipeline
```

## Installation

```bash
uv sync
```

Node.js dependency (auto-installed on first use, or manually):

```bash
(cd data_processing/download && npm install lz-string)
```

## Quick Start

### Full Pipeline

Process replays for specific players:

```bash
uv run python scripts/prepare_data.py \
    -p MeltedToast Mithraaaa \
    -o data/my_replays \
    --max-replays 1000 \
    --min-turns 80
```

This will:
1. Download up to 1000 replay metadata entries per player
2. Filter for 1v1 games with ≥80 turns
3. Download the actual replay files
4. Simulate each replay from the player's perspective
5. Convert to training format

Output structure:
```
data/my_replays/
├── metadata/              # Raw replay metadata
├── filtered/              # Filtered metadata
├── replays/               # Downloaded .gior files (decompressed)
├── simulated/             # Simulated game states
└── training/              # Final training data
```

### Run Specific Stages

Run only certain stages:

```bash
# Only download and filter
uv run python scripts/prepare_data.py \
    --players MeltedToast \
    --output data/test \
    --stages download filter

# Only convert existing simulated data
uv run python scripts/prepare_data.py \
    --players MeltedToast \
    --output data/test \
    --stages convert
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--players` | `-p` | Player usernames to process | (required) |
| `--output` | `-o` | Output directory | (required) |
| `--stages` | `-s` | Stages to run: download, filter, simulate, convert | all |
| `--max-replays` | | Maximum replays per player | 10000 |
| `--max-concurrent` | | Maximum concurrent downloads | 10 |
| `--min-turns` | | Minimum turns required | 80 |
| `--min-opponent-stars` | | Minimum opponent stars | None |

### Using Programmatically

```python
import asyncio
from data_processing import (
    fetch_replay_list,
    download_replays,
    filter_replays,
    simulate_replay,
    parse_replay_to_gio,
)

async def main():
    # 1. Fetch replay metadata
    replays = await fetch_replay_list("MeltedToast", total=100)

    # 2. Filter replays
    filtered = filter_replays(replays, min_turns=80)

    # 3. Download replay files
    replay_ids = [r['id'] for r in filtered]
    await download_replays(replay_ids, "output/replays")

    # 4. Simulate a replay
    simulate_replay(
        "output/replays/replay123.json",
        "MeltedToast",
        "output/simulated/replay123.json"
    )

    # 5. Convert to training format
    parse_replay_to_gio(
        "output/simulated/replay123.json",
        "output/training/replay123.json"
    )

asyncio.run(main())
```

## Data Format

### Simulator Output

The simulator produces per-turn game state from a player's perspective:

```json
{
  "id": "replay_id",
  "player": "player_name",
  "first": [1, 0, 1, ...],
  "map": [[[state_values]]],
  "armies": [[[army_counts]]],
  "info_myArmy": [10, 15, ...],
  "info_myLand": [5, 6, ...],
  "info_opArmy": [12, 14, ...],
  "info_opLand": [6, 7, ...],
  "move_start": [[row, col], ...],
  "move_end": [[row, col], ...],
  "move_is_50": [0, 1, ...]
}
```

State encoding (11 categories):
- 0: Unseen Blank
- 1: Unseen not Blank
- 2: Seen Blank
- 3: Seen Uncaptured City
- 4: Seen Mountain
- 5: Seen Your Land
- 6: Seen Your City
- 7: Seen Your General
- 8: Seen Opponent's Land
- 9: Seen Opponent's City
- 10: Seen Opponent's General

### Training Format

The final training format:

```json
{
  "grid_seq": [T, C, H, W],
  "globals_seq": [T, G],
  "prev_action_seq": [T, A],
  "legal_source_mask_seq": [T, 1, H, W],
  "max_actions_per_tick": 4,
  "count": [T],
  "src_y": [T, K],
  "src_x": [T, K],
  "dir_id": [T, K],
  "ratio_id": [T, K],
  "value": [T],
  "enemy_heatmap": [T, 1, H, W],
  "next_visibility": [T, 1, H, W]
}
```

Where:
- T = number of turns
- C = 12 channels (11 state + 1 army)
- H = 25 (padded height)
- W = 25 (padded width)
- G = 4 global features
- K = max actions per tick (default: 4)

## Loading Data for Training

```python
from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from torch.utils.data import DataLoader

# Create dataset
dataset = SeqReplayDataset('data/my_replays/training/MeltedToast')

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_variable_length_sequences
)

# Iterate
for batch in dataloader:
    obs = batch['obs']       # Observations
    tgt = batch['tgt']       # Targets
    meta = batch['meta']     # Metadata

    grid = obs['grid_seq']   # [B, T, C, H, W]
    # ... train your model
```

## Performance Tips

1. **Parallel Downloads**: Increase `--max-concurrent` for faster downloads
2. **Batch Processing**: Process multiple players simultaneously
3. **Resume**: The pipeline skips already-processed files
4. **Filtering**: Use stricter filters to reduce data volume