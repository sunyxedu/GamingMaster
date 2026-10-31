Implemented and verified the NetHack adaptor.

- Added [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/nethack/adaptor/game.py)
- Added [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/nethack/adaptor/ADAPTOR.md)
- Added [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/nethack/adaptor/requirements.txt)
- Added [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/nethack/adaptor/env.example)
- Installed `nle==1.3.0` in the evaluator venv.
- Supports deterministic seeded resets, structured JSON observations, score/depth/ascension breakdown, score and staircase scenarios, malformed actions, and idempotent cleanup.

Validation completed successfully: `Adaptor validation: PASS`. Random-action and deterministic-reset checks also passed.