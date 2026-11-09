Implemented the Atari Space Invaders tier-2 adaptor.

- Added `game.py`, `ADAPTOR.md`, `requirements.txt`, and `env.example`
- Uses headless ALE/Gymnasium with RAM plus symbolic RGB decoding
- Supports seeded scenarios, realtime ticks, score/lives/waves/projectiles/UFO state
- Handles malformed actions and idempotent cleanup
- Validation: **PASS**
- Random-action smoke tests completed successfully

Validator warning: initial observations are identical across seeds, but trajectories diverge via seeded sticky-action RNG as intended.