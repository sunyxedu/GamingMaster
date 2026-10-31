Implemented the Jericho Tier 1 adaptor.

- Added multi-game `FrotzEnv` adaptor in [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/jericho/adaptor/game.py).
- Staged eight distinct suite games under `runtime/games/`, with SHA-256/MD5 validation and native score maxima.
- Added train/heldout/dev/validation seed partitions and deterministic game/VM seed mapping.
- Updated [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/jericho/adaptor/ADAPTOR.md), [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/jericho/adaptor/env.example), and manifest provenance.
- Verified replay determinism, JSON observations, malformed actions, native scoring, and random play across all games.

Required validator result: **PASS**.