Implemented and verified the Atari Frogger Tier 2 adaptor.

- Fixed seed replay by using deterministic seed-derived live ALE traffic phases.
- Added phase metadata and documented behavior.
- Corrected truncated episodes to set `done=True`.
- Tested random play, secondary modes, strict JSON, and idempotent close.
- `validate_adaptor(..., expected_tier=2)` now passes.