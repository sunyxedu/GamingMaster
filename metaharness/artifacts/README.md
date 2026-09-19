# Published Meta-Harness experiments

These are immutable research records, distinct from editable source in
`metaharness/core/`, `models/`, `games/` and `tiers/`. New runs belong in the
ignored `runs/` directory until their results are checked and deliberately
published. Keep failed candidates and timeout records alongside successes.

## Current audited evidence

| Archive | What it contains |
|---|---|
| [Tier-1 discovery](audit-20260919-tier1-discovery/) | Initial search, two baselines and the completed program from a timed-out proposer |
| [Tier-1 final evaluation](audit-20260919-tier1/) | Fresh validation, rule-puzzle tests and ARC `ls20`/`cd82` transfer, including two timeouts |
| [Tier-2 audit](audit-20260919-tier2/) | 48 live Astra/Jev episodes covering search, independent/shared memory, transfer and ablations |
| [Jev context integration probe](jev-context-smoke-20260919/) | One real Astra-generated typed request sent to Jev; interface verification, not gameplay performance |

Read the [combined audit](../../docs/meta-harness-audit-2026-09-19.md) and
[tier-2 report](../../docs/meta-harness-tier2-audit-2026-09-19.md) before comparing
scores. The selected programs are tier 1's
[`0001-previous_astra_winner`](audit-20260919-tier1/candidates/0001-previous_astra_winner/harness.py)
and tier 2's [`0004`](audit-20260919-tier2/candidates/0004/harness.py).
Selection used training results; transfer scores did not change those choices.

Each audited archive has an `archive_manifest.json` listing file hashes.
Trace files ending in `.jsonl.gz` are complete gzip-compressed JSONL records.
The recorded absolute paths identify the original execution directories and
need not exist when reading a published archive.

## Historical runs

| Archive group | Models and protocol |
|---|---|
| `astra-tier1-rulepuzzle-heldout-arc3`, `astra-tier2-snake-heldout-lanerunner` | Earlier real Astra runs and the prior champions used in the audit |
| `exp-tier1-rulepuzzle-claude`, `exp-tier2-snake-jev-claude` | Explicit Claude stand-in experiments |
| `exp2-tier1-rulepuzzle-heldout-arc3`, `exp2-tier2-snake-heldout-lanerunner` | Earlier transfer experiments using the stand-in backend |

Historical runs used different protocols, including cross-seed memory sharing.
Their leaderboards are preserved as measured and should not be combined into
one ranking with the independent-episode audit. See the
[historical experiment notes](../../docs/meta-harness.md#historical-experiments-2026-09-19-shared-memory-protocol).
