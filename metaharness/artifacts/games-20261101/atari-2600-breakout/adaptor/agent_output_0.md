Implemented the Atari 2600 Breakout adaptor.

- Tier 2 `RealtimeGame`, 60 Hz `tick_s`, ALE `frameskip=1`
- Official headless ALE RAM interface with OCAtari symbolic decoding
- JSON observations with ball, paddle, bricks, score, lives, walls, rewards, and flags
- Seed declarations, deterministic reset, malformed-action handling, idempotent close
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`

Validation: **PASS** with the exact required command. Random gameplay also confirmed ALE reward matches decoded cartridge score.