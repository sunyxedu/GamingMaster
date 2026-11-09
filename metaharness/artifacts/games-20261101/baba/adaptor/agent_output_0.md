Implemented and validated the Baba Is AI adaptor.

- Added Tier 1 `TurnGame` in [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/baba/adaptor/game.py).
- Vendored the upstream Baba engine under `baba/` for offline execution.
- Added deterministic seed-to-level mapping with train, held-out, dev, and validation sets.
- Added structured symbolic observations, native win scoring, action descriptions, malformed-action handling, metadata, and cleanup.
- Added [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/baba/adaptor/ADAPTOR.md), `requirements.txt`, `env.example`, and license.

Validation passed:

`validate_adaptor(..., expected_tier=1)` → **PASS**

Also verified `pip check`, random actions, deterministic resets, and a native puzzle solved through the adaptor with score `1.0`.