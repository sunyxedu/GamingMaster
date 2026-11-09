Implemented the Atari 2600 Frogger adaptor:

- Added ALE/Gymnasium-backed `game.py` with Tier 2 realtime interface.
- Added strict JSON RAM observations, score/life bookkeeping, seed/config metadata, malformed-action handling, and idempotent `close()`.
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`.
- Validation passes:
  - Tier 2 confirmed
  - Determinism confirmed
  - Strict JSON confirmed
  - Random gameplay and secondary modes tested
  - Score/reward and terminal behavior verified