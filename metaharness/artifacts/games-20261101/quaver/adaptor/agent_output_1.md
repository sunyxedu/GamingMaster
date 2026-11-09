Implemented the Quaver Tier-2 adaptor:

- Added deterministic headless `.qua` simulator in `game.py`
- Added official-style timing, LN, accuracy, combo, score, grade, and performance rating
- Added seeded 4K/7K scenario generation with held-out seeds
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`
- Supports optional native chart roots and rejects unsupported mines/timing groups
- Runtime artifacts are stored under `runtime/`

Validation passes with `expected_tier=2`; perfect-play and random-action smoke tests also passed.