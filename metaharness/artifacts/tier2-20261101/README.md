# Tier-2 search archives, 2026-10-29 / 11-01 (branch `metaharness-tier2-v2`)

One directory per game, produced from the game's name by `python -m metaharness run` (study → adapt with an
observation review → search → validation → held-out → ablation). Astra (gpt-6-astra through the claude CLI) is
the study/adapt agent, proposer and System-2 player; Jev (jev-latest, jev-1.14) is System-1. Real time on the wall
clock, a 900 s episode deadline. Searches: generations of 3 proposals (refine / rewrite / crossover / explore /
repair), up to 8 generations, stopping after 3 without a new best (2 before 10-31 08:00). Per game: `study/`,
`adaptor/` (generated code and docs, review.md; runtime data, environments and traces omitted), `search*/`
(task.md, leaderboard.md, lessons.md, reflections, manifest.json, calibration.json, loop.log, champion.json; per
candidate harness.py, proposal.md, directive.json, pre-flight reports and eval.json without final observations,
also `validation/`, `heldout/` and `ablation-no-models/`). REPAIRS.md records operator interventions.

| game | champion | train | validation | held-out (seeds) | models failing | proposals |
|---|---|---|---|---|---|---|
| Atari 2600 Asteroids | 0022 | 28,303 | 27,040 | 29,594 (8) | 28,208 | 24 |
| Atari 2600 Breakout | 0026 | 793 | 647 | 560 (8) | 742 | 24 |
| Atari 2600 Donkey Kong | 0018 | 4,133 | 4,150 | 4,075 (8) | 4,133 | 24 |
| Atari 2600 Frogger | 0017 | 4,433 | 4,874 | 4,764 (8) | 5,123 | 24 |
| Atari 2600 Ms. Pac-Man | 0023 | 3,945 | 4,367 | 3,639 (8) | 3,342 | 21 |
| Atari 2600 Pong | 0009 | 20.33 | 19 | 14.5 (4) | 20.67 | 9 |
| Atari 2600 Q*bert | 0016 | 21,808 | 23,125 | 21,566 (8) | 21,208 | 24 |
| Atari 2600 Space Invaders | 0025 | 394 | 434 | 331 (8) | 394 | 24 |
| Doom | 0011 | 10.96 | 10.96 | 6.57 (16) | 11.62 | 18 |
| Flappy Bird | 0006 | 716 | 716 | 716 (8) | 716 | 6 |
| Malody | 0004 | 1 | 1 | 1 (8) | 1 | 3 |
| Minecraft | 0015 | 3 | 3 | 1 (3) | 2 | 24 |
| osu!mania | 0008 | 1,000,000 | 1,000,000 | 1,000,000 (8) | 0 | 6 |
| Overcooked | 0024 | 258 | 237 | 227 (6) | 258 | 24 |
| Quaver | 0004 | 1,000,000 | 1,000,000 | 1,000,000 (16) | 1,000,000 | 3 |
| Snake | 0020 | 141 | 141 | 140 (7) | 141 | 18 |
| Tetris | 0017 | 63,217 | 66,181 | 62,259 (7) | 61,753 | 21 |

"models failing" is the ablation: the champion on its training seeds with every Jev and Astra call failing after
the models' calibrated latency (pure code if it holds). Notes:
- Scores are each game's own: Malody is weighted accuracy (1 = every note perfect); Quaver and osu!mania 1,000,000 =
  the maximum; Snake 141 = the 12x12 board filled; Breakout 864 = both walls; Flappy Bird is capped by the deadline.
- Doom's held-out is transfer (unseen maps E1M5-E1M8 at skill 4; training E1M1-E1M4). Overcooked's held-out uses
  the training layouts; its transfer run on 8 unseen layouts (`overcooked/transfer-0024/`) averages 199.6, with 0 on
  the three forced_coordination layouts. Pong's held-out is its unseen second variation.
- osu!mania's champion waits for Jev approvals of its key schedules, so it presses nothing without models (0);
  its first ablation (System-1 set to None) is kept in `ablation-no-models.system1-none/`.
- Donkey Kong's champion farms barrel jumps (+100 each) and completes no screen. Minecraft reaches only the opening
  sub-milestones of "Free the End" within 15 minutes.
- Frogger's `search.pre-review/` (best 14.5, no frog home) is the archive from before the observation review
  added decoded hazards; the reviewed adaptor's search reached 4,433 (41-63 frogs home per episode).

Decisions and problems: `docs/decision-log.md` (entries 174-200) and `docs/problems-log.md` (144-153).
