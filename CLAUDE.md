# GamingMaster — project instructions

## Meta-Harness (tiers 1/2)
- `metaharness/` is the Meta-Harness for non-latency-critical games; design doc `docs/meta-harness.md`.
- Model *backends* live in `metaharness/models/` (codex CLI, claude CLI, TypeSafe HTTP); which model runs on a
  backend is configuration in `metaharness/models.toml` (profiles: astra, sol, claude, jev, ...). Never hard-code
  a model name in a backend class, never hard-code API keys (`.env`, see `.env.example`).
- Anything touching Jev / TypeSafe (`metaharness/models/jev.py`, question design in harnesses or task specs):
  use the `typesafe-ai` skill (`claude plugin install typesafe@typesafe-ai`) and read the live docs it points to
  (`https://docs.typesafe.ai/llms.txt`, `/api.md`, `/primitives/*.md`, `/confidence.md`) before changing the contract.
- Never edit `runs/` while a loop is running; run `uv run pytest tests/metaharness` before committing.
