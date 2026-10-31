# Tier-1 search archives, 2026-10-30 / 31 (branch `metaharness-tier1-v2`)

One directory per game, produced from the game's name by `python -m metaharness run` (study → adapt → search →
validation → held-out) with Astra (gpt-6-astra) as study/adapt agent, proposer and player. Per game:
`study/` (brief, interfaces, strategies, study.json), `adaptor/` (the generated game.py and helpers, ADAPTOR.md;
runtime data, game assets and binaries omitted), `search*/` (task.md, leaderboard.md, notes.md, manifest.json,
calibration.json; per candidate harness.py, proposal.md, eval.json without final observations, pre-flight reports;
`validation/` and `heldout/` eval.json). `search-v1*` are earlier archives whose adaptor was replaced (saturated
score or an opponent that was too weak / too slow); `game_v1.py` etc. are the replaced adaptors. Traces are not
included (tens of MB per candidate); they stay in the run directory.

Results summary and the general lessons: `docs/meta-harness.md` (section "Tier-1 search v2") and
`docs/decision-log.md`.
