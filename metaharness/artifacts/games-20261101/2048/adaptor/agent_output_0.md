Implemented the offline Tier 1 2048 adaptor.

- [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/2048/adaptor/game.py): seeded 4×4 engine, exact merge rules, 90/10 spawning, observations, actions, scoring, and cleanup.
- [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/2048/adaptor/ADAPTOR.md): interface and scenario documentation.
- [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/2048/adaptor/requirements.txt), [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/2048/adaptor/env.example).

Validation passed with `validate_adaptor(..., expected_tier=1)`. Canonical merge cases and random seeded play also passed.