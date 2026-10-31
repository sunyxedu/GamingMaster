Implemented and validated the MiniGrid/BabyAI adaptor.

- Added [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/minigrid/adaptor/game.py)
- Added [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/minigrid/adaptor/ADAPTOR.md)
- Added pinned [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/minigrid/adaptor/requirements.txt)
- Added [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/minigrid/adaptor/env.example)
- Installed and verified MiniGrid 3.0.0, Gymnasium 1.4.0, NumPy 2.5.4, Pygame 2.6.1.

`validate_adaptor(..., expected_tier=1)` passes all checks. Random gameplay, deterministic resets, scenario selection, max-step handling, malformed actions, and successful scoring were also verified.