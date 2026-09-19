# Documentation

GamingMaster contains three control paths. Tiers 1 and 2 use generated
foundation-model harnesses; tier 3 uses the neural network and per-game adapters.
All are available on `main`.

## Meta-Harness: Astra and Jev

| Document | Purpose |
|---|---|
| [Meta-Harness guide](meta-harness.md) | Architecture, configuration, CLI and evaluation protocol |
| [Implementation and live-model audit](meta-harness-audit-2026-09-19.md) | Review of both tiers, measured performance and remaining limitations |
| [Tier-2 experiment report](meta-harness-tier2-audit-2026-09-19.md) | Snake, Lane Runner and model-removal controls |
| [Jev context and usage](../metaharness/models/jev_usage.md) | The versioned model guide embedded in proposal prompts |
| [Published experiment archives](../metaharness/artifacts/README.md) | Generated programs, evaluations, traces and provenance |

Minecraft and Genshin are intended applications, not implemented integrations.
The ARC results cover selected offline games rather than an official benchmark run.

## Neural agents and data

| Document | Purpose |
|---|---|
| [Multi-game architecture](multi-game.md) | Shared network, observation encoders and action decoders |
| [Training](training.md) | Imitation learning, reinforcement learning and memory sizing |
| [Data pipeline](data-pipeline.md) | Replay download, filtering, simulation and conversion |
| [Game data sources](game-data-sources.md) | Dataset and environment coverage per game |
| [Clash Royale](clash-royale.md) | Adapter, game-state interface and current scope |
| [Known limitations](known-limitations.md) | Outstanding constraints and unsupported behavior |

## Working directories and checks

- `metaharness/artifacts/` holds published research records. Keep their original
  programs and measurements intact; add a new archive for a new protocol.
- `runs/`, `data/` and `checkpoints/` are ignored local outputs.
- `.env` and `.claude/` hold ignored local credentials/settings; `.env.example`
  documents supported environment variables without credentials.
- `uv run pytest tests/ -q` runs the repository suite. Meta-Harness-only checks
  use `uv run pytest tests/metaharness -q`; its live Jev check is skipped when
  no credential configuration exists. CI has no model credentials and uses
  the offline tests.

Historical reports refer to the branches and commits used during experiments.
Their dates and protocols remain part of the evidence after integration into `main`.
