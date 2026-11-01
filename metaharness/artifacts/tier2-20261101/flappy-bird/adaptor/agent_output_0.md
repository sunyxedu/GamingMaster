Implemented the Flappy Bird tier-2 adaptor.

- Headless `flappy-bird-gymnasium==0.4.0`
- 30 FPS `RealtimeGame` with `IDLE`/`FLAP`
- Seeded train/held-out splits: 8 each
- Strict JSON observations and score diagnostics
- Idempotent cleanup, malformed-action handling
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`

Validation passed headlessly. Random and heuristic episodes produced finite scores and valid termination diagnostics.