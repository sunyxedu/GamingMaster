Implemented and verified the TextWorld adaptor.

- Added Tier 1 native Glulx adaptor in [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/game.py).
- Added deterministic generation, caching, integrity checks, bounded workers, and cleanup.
- Added documentation, offline setup, pinned requirements, and environment template:
  - [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/ADAPTOR.md)
  - [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/requirements.txt)
  - [requirements.lock](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/requirements.lock)
  - [setup.sh](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/setup.sh)
  - [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/textworld/adaptor/env.example)

Verification completed:

- `validate_adaptor(..., expected_tier=1)` → **PASS**
- TextWorld 1.6.2 installed in the required evaluator venv.
- All 30 declared seeds generated and compiled offline.
- Determinism, score normalization, partial credit, truncation, malformed actions, crash handling, cleanup, and artifact uniqueness passed.
- `pip check` passed.