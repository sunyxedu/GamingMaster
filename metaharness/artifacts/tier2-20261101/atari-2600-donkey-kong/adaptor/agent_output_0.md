Implemented the Atari 2600 Donkey Kong Tier 2 adaptor.

Files added:
- `game.py`
- `ADAPTOR.md`
- `requirements.txt`
- `env.example`
- `runtime/`

Verified:
- ALE/Stella + OCAtari RAM interface
- Real-time 66.7 ms ticks
- JSON observations and cumulative rewards
- Sticky-action seed variation
- Malformed action handling
- Idempotent cleanup
- Generic validation: **PASS**