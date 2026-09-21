# Adaptor validation: PASS

- [x] load game.py and make_game() — name=Snake tier=tier2 realtime=True
- [x] action_spec is an ActionSpec with >=1 action
- [x] realtime: tick_s > 0 and noop action set — tick_s=0.2 noop=NOOP
- [x] describe() returns a non-trivial string — 1765 chars
- [x] seed 0: observe() JSON-serialisable — 547 bytes
- [x] seed 0: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 0: applied 6 actions without exception
- [x] seed 0: ticks counter advanced — ticks=6 applied=6
- [x] seed 0: info() is a dict
- [x] seed 1: observe() JSON-serialisable — 546 bytes
- [x] seed 1: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 1: applied 6 actions without exception
- [x] seed 1: ticks counter advanced — ticks=6 applied=6
- [x] seed 1: info() is a dict
- [x] DETERMINISTIC: reset(seed) reproduces the first observation
- [x] different seeds give different starts (scenario variety)

