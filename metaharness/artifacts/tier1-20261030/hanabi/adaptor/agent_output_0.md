Implemented the offline Hanabi adaptor with deterministic deals, masked observations, complete rules, and 0–25 scoring.

- Generic validator: **PASS**, no warnings.
- All **12 conformance tests passed**.
- Random and deck-exhaustion games verified.
- No packages, credentials, GUI, or network required.

See [ADAPTOR.md](/home/hemlix/GamingMaster-mh-tier1/runs/hanabi/adaptor/ADAPTOR.md) for usage and limitations. The harness must maintain separate private memory per seat to preserve hidden-information play.