# Adaptor validation: PASS

- [x] load game.py and make_game() — name=Minecraft Java 1.21.1 — natural survival tier=tier2 realtime=True
- [x] action_spec is an ActionSpec with >=1 action
- [x] realtime: tick_s > 0 and noop action set — tick_s=0.5 noop=noop
- [x] describe() returns a non-trivial string — 4877 chars
- [x] seed 0: observe() JSON-serialisable — 17978 bytes
- [x] seed 0: done is bool and score is a number after reset — done=False score=3.775000028933088e-05
- [x] seed 0: applied 12 actions without exception
- [x] seed 0: ticks counter advanced — ticks=12 applied=12
- [x] seed 0: info() is a dict
- [x] seed 1: observe() JSON-serialisable — 11870 bytes
- [x] seed 1: done is bool and score is a number after reset — done=False score=3.887500303486983e-05
- [x] seed 1: applied 12 actions without exception
- [x] seed 1: ticks counter advanced — ticks=12 applied=12
- [x] seed 1: info() is a dict
- [x] different seeds give different starts (scenario variety)
