Implemented the Sokoban Tier 1 adaptor.

- Added [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/sokoban/adaptor/game.py) with deterministic Boxoban parsing, seed mapping, turn-based actions, JSON observations, scoring, action caps, and idempotent cleanup.
- Bundled pinned Boxoban levels and license under `runtime/boxoban/`.
- Added [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/sokoban/adaptor/ADAPTOR.md), [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/sokoban/adaptor/env.example), and [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/sokoban/adaptor/requirements.txt).

Validation passed with `expected_tier=1`, including deterministic resets, strict observations, malformed actions, and close behavior. Random-action and solved-level smoke tests also passed.