Implemented the seed-divergence fix.

- Enabled deterministic ALE sticky actions at `0.25` in defaults and `EVAL_KWARGS`.
- Updated `ADAPTOR.md`.
- Verified ghost modes 0–3.
- Random-action smoke test passed with valid score/state updates.
- `validate_adaptor(..., expected_tier=2)` now reports **PASS**.