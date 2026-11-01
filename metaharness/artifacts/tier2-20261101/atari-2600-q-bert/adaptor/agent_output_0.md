Implemented and verified the Atari 2600 Q*bert adaptor:

- Tier 2 ALE/Gymnasium real-time interface with 4-frame cadence.
- Symbolic 21-cube, Q*bert, enemy, disc, score, lives, and phase observations.
- Seeded novice/expert modes, sticky-action handling, malformed-action tolerance.
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`.
- Exact validator result: **PASS**.
- Random-action smoke tests confirmed valid JSON and score/ROM consistency.

Validator warning is expected: seeds share the fixed opening but diverge during play.