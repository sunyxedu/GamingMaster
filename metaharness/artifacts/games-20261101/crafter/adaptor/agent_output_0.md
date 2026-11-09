Implemented the Crafter adaptor.

- Added [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/crafter/adaptor/game.py) with seeded deterministic `TurnGame`, semantic local map, inventory, achievements, action handling, scoring, and cleanup.
- Added [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/crafter/adaptor/ADAPTOR.md), [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/crafter/adaptor/requirements.txt), and [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/crafter/adaptor/env.example).
- Installed `crafter==1.8.3` in the evaluator venv.
- `validate_adaptor(..., expected_tier=1)` passes.
- Verified random play and wood collection against the real local Crafter environment.