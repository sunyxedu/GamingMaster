# Adaptor validation: PASS

- [x] load game.py and make_game() — name=ARC-AGI-3 tier=tier1 realtime=False
- [x] action_spec is an ActionSpec with >=1 action
- [x] describe() returns a non-trivial string — 2651 chars
- [x] seed 0: observe() JSON-serialisable — 13330 bytes
- [x] seed 0: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 0: applied 9 actions without exception
- [x] seed 0: info() is a dict
- [x] seed 1: observe() JSON-serialisable — 15969 bytes
- [x] seed 1: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 1: applied 9 actions without exception
- [x] seed 1: info() is a dict
- [x] different seeds give different starts (scenario variety)
