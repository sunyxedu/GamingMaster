# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML data pipeline (`gaming-master`) for processing generals.io game replays into PyTorch training data.

### Services

| Component | How to run | Notes |
|---|---|---|
| Data pipeline CLI | `uv run python scripts/prepare_data.py --help` | Download/filter/simulate/convert stages. Download stage needs internet + Node.js. |
| Mock data generator | `uv run python data_processing/mock/make_mock_replays.py --out /tmp/mock --n 10` | Creates synthetic training data (no network needed). |
| PyTorch DataLoader | See `data_processing/dataset_seq.py` | `SeqReplayDataset` + `collate_variable_length_sequences` |

### Lint / Test / Build

- **Lint:** `uv run ruff check .` — config in `pyproject.toml` (`[tool.ruff]`). Pre-existing lint issues exist in the repo.
- **Test:** `uv run pytest` — only one test file exists (`tests/test_simulator_visual.py`), and it is an interactive visual test (requires GUI/matplotlib display), not a standard pytest suite. No tests are collected by default.
- **Build:** `uv build` — uses Hatchling backend.

### Non-obvious caveats

- The Node.js `lz-string` package must be installed in `data_processing/download/` (not the repo root) for the replay decompressor to work. Run: `cd data_processing/download && npm install lz-string`.
- The download stage calls `node decompress.js` as a subprocess — Node.js must be on `PATH`.
- `uv` must be on `PATH`. If not found, ensure `$HOME/.local/bin` is in `PATH`.
- The mock data generator is the best way to test the dataloader pipeline offline without network access.
