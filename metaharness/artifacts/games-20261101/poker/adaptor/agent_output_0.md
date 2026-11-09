Implemented and validated the poker adaptor.

- Added [`game.py`]( /home/hemlix/GamingMaster-mh-tier1/runs/poker/adaptor/game.py ) using PokerKit 0.7.6.
- Added stronger frozen Monte Carlo/pot-odds opponent in [`poker_opponent_v2.py`]( /home/hemlix/GamingMaster-mh-tier1/runs/poker/adaptor/poker_opponent_v2.py ).
- Updated [`ADAPTOR.md`]( /home/hemlix/GamingMaster-mh-tier1/runs/poker/adaptor/ADAPTOR.md ) and opponent manifest.
- No credentials or environment variables required.
- Validator result: **PASS**.
- Random-action match completed successfully with valid scoring and zero illegal actions.
- Recorded 1,000-hand offline benchmark in [`runtime/benchmark_1000.txt`]( /home/hemlix/GamingMaster-mh-tier1/runs/poker/adaptor/runtime/benchmark_1000.txt ).