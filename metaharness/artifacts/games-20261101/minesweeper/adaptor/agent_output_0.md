Implemented the offline deterministic Minesweeper adaptor.

- [game.py](/home/hemlix/GamingMaster-mh-tier1/runs/minesweeper/adaptor/game.py:1): Tier 1 `TurnGame`, seeded boards, first-click safety, reveal/flag/chord actions, JSON observations, scoring, diagnostics, and cleanup.
- [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/minesweeper/adaptor/ADAPTOR.md:1): interface, seed mapping, rules, and scoring documentation.
- [env.example](/home/hemlix/GamingMaster-mh-tier1/runs/minesweeper/adaptor/env.example:1): no environment variables required.
- [requirements.txt](/home/hemlix/GamingMaster-mh-tier1/runs/minesweeper/adaptor/requirements.txt:1): standard library only.

Validation passes with `expected_tier=1`. Manual random-action, deterministic-sequence, win, loss, and score checks also passed.