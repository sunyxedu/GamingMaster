# Crafter adaptor

This adaptor runs the official `crafter==1.8.3` Python simulator directly in the evaluator's
Crafter virtual environment. It starts no server, opens no account, and makes no network requests.
`SDL_VIDEODRIVER` defaults to `dummy` for headless Linux. There are no required credentials; the
optional setting is documented in `env.example`.

Crafter is tier 1 because its simulation advances only when `step(action)` is called. `reset(seed)`
creates a fresh `crafter.Env(seed=seed)` and calls its reset, preserving the standard 64x64 world and
10,000-tick episode length unless evaluator kwargs override them. The declared train, dev, validation,
and held-out seeds all select independent procedural worlds; no route is memorized.

The action vocabulary is read from `crafter.constants.actions` at construction. Actions may be submitted
as their names (the normal harness form) or integer IDs. Invalid actions are converted to `noop` and
reported in the next observation. Each observation is strict JSON: player coordinate/facing/vitals,
inventory, cumulative achievement counters and unlocked names, the last action/reward, and a 9x9 local
semantic map. Map cells contain coordinates, material, and nearby object names (including health and
plant growth status where available). The optional RGB observation from Crafter is intentionally omitted
to keep the interface textual and compact; the official environment remains available for debugging.

`score` is the fraction of the standard achievement set unlocked in the current episode, and
`score_breakdown()` reports the unlocked count and denominator. For a batch of held-out episodes, the
benchmark Crafter score is obtained by calculating each achievement's success rate and taking the
geometric mean, as described in the study. Raw environment reward is exposed only as `last_reward` and
is not used as the primary score.

`close()` is idempotent and releases the current environment. No runtime files or processes are created.
