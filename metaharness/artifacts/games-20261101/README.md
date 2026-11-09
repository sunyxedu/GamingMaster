# Meta-Harness search archives: 35 games from their names (2026-10-29 … 11-01)

One directory per game, produced from the game's name by `python -m metaharness run` (study → adapt → search →
validation → held-out). Astra (gpt-6-astra) is the study / adapt agent, the proposer and the reasoning player in
every run. Per game: `study/` (brief, interfaces, strategies, study.json), `adaptor/` (the generated game.py and
helpers, ADAPTOR.md; runtime data, game assets, environments and binaries omitted), `search*/` (task.md,
leaderboard.md, notes.md or lessons.md with reflections, manifest.json, calibration.json; per candidate harness.py,
proposal.md, pre-flight reports and eval.json without final observations; `validation/` and `heldout/` eval.json).
Traces are not included (tens of MB per candidate).

**These runs predate the single task spec** (`metaharness/task.py`, 2026-11-02). Until then the pace of a game also
decided its model set, on two branches: the 18 turn-based games were searched with Astra alone
(`metaharness-tier1-v2`, through the codex CLI), the 17 real-time games with Astra and Jev (`metaharness-tier2-v2`,
through the claude CLI; Jev = jev-latest, jev-1.14; wall clock, a 900 s episode deadline; generations of 3
proposals, up to 8 generations, stopping after 3 without a new best). That is why the archived task.md, manifests,
study.json and seed-candidate names still say "tier 1" / "tier 2", and why no turn-based harness here calls Jev.
Since the merge every game gets Astra and Jev, and the pace comes from the adaptor; `study.json` files with `tier`
still load.

| game | pace | champion | train | validation | held-out (episodes) | models failing | proposals |
|---|---|---|---|---|---|---|---|
| 2048 | turn-based | 0013 | 3,068 | 3,017 | 3,066 (8) | – | 18 |
| Atari 2600 Asteroids | real-time | 0022 | 28,303 | 27,040 | 29,594 (8) | 28,208 | 24 |
| Atari 2600 Breakout | real-time | 0026 | 793 | 647 | 560 (8) | 742 | 24 |
| Atari 2600 Donkey Kong | real-time | 0018 | 4,133 | 4,150 | 4,075 (8) | 4,133 | 24 |
| Atari 2600 Frogger | real-time | 0017 | 4,433 | 4,874 | 4,764 (8) | 5,123 | 24 |
| Atari 2600 Ms. Pac-Man | real-time | 0023 | 3,945 | 4,367 | 3,639 (8) | 3,342 | 21 |
| Atari 2600 Pong | real-time | 0009 | 20.33 | 19 | 14.5 (4) | 20.67 | 9 |
| Atari 2600 Q*bert | real-time | 0016 | 21,808 | 23,125 | 21,566 (8) | 21,208 | 24 |
| Atari 2600 Space Invaders | real-time | 0025 | 394 | 434 | 331 (8) | 394 | 24 |
| Baba Is AI | turn-based | 0003 | 0.875 | 0.75 | 1.0 (8) | – | 3 |
| Crafter | turn-based | 0010 | 0.79 | 0.676 | 0.79 (8) | – | 18 |
| Dominion (pyminion) | turn-based | 0016 | 14.5 | 11.0 | 1.63 (8) | – | 18 |
| Doom | real-time | 0011 | 10.96 | 10.96 | 6.57 (16) | 11.62 | 18 |
| Factorio | turn-based | 0011 | 4,073 | 3,607 | 3,343 (4) | – | 12 |
| Flappy Bird | real-time | 0006 | 716 | 716 | 716 (8) | 716 | 6 |
| Hanabi | turn-based | 0013 | 23.25 | 23.0 | 23.5 (4) | – | 18 |
| Heads-up no-limit Texas Hold'em | turn-based | 0013 | 931 | 744 | 878 (4) | – | 18 |
| Jericho (Zork and other text adventures) | turn-based | 0005 | 0.149 | 0.121 | 0.101 (8) | – | 18 |
| Malody | real-time | 0004 | 1 | 1 | 1 (8) | 1 | 3 |
| Minecraft: Java Edition | real-time | 0015 | 3 | 3 | 1 (3) | 2 | 24 |
| Minesweeper | turn-based | 0011 | 0.999 | 0.671 | 0.837 (12) | – | 12 |
| MiniGrid / BabyAI | turn-based | 0002 | 0.918 | 0.941 | 0.907 (8) | – | 12 |
| NetHack | turn-based | 0016 | 123 | 77.9 | 114 (8) | – | 18 |
| osu!mania | real-time | 0008 | 1,000,000 | 1,000,000 | 1,000,000 (8) | 0 | 6 |
| Overcooked-AI | real-time | 0024 | 258 | 237 | 227 (6) | 258 | 24 |
| Pokémon Showdown (Gen 9 Random Battle) | turn-based | 0000 (naive Astra) | −0.687 | −0.489 | −0.901 (10) | – | 9 |
| Quaver | real-time | 0004 | 1,000,000 | 1,000,000 | 1,000,000 (16) | 1,000,000 | 3 |
| Settlers of Catan | turn-based | 0009 | 0.583 | 0.585 | 1.10 (4) | – | 18 |
| Snake | real-time | 0020 | 141 | 141 | 140 (7) | 141 | 18 |
| Sokoban (Boxoban) | turn-based | 0005 | 10,393 | 10,393 | 10,392 (10) | – | 6 |
| Tetris (Guideline marathon) | real-time | 0017 | 63,217 | 66,181 | 62,259 (7) | 61,753 | 21 |
| TextArena | turn-based | 0018 | 0.609 | 0.656 | 0.578 (8) | – | 18 |
| TextWorld | turn-based | 0001 | 1.0 | 1.0 | 1.0 (9) | – | 3 |
| Wordle / Mastermind | turn-based | 0016 | 0.611 | 0.597 | 0.569 (12) | – | 18 |
| Yu-Gi-Oh! TCG (EDOPro) | turn-based | 0008 | 0.541 | 0.476 | 0.299 (12) | – | 18 |

Every number is the mean of the game's own score and comes from the champion's rows in `search/leaderboard.md`;
scores of different games are not comparable. `pace` is what the adaptor implements (`tick` present = real-time).
"models failing" is the ablation of the real-time searches: the champion on its training seeds with every Jev and
Astra call failing after the models' calibrated latency (pure code if it holds); the turn-based searches did not
run it. Pong, Tetris, Ms. Pac-Man, Flappy Bird, Space Invaders, Malody, Snake and Doom ran the earlier ablation
with the models set to None; both agree for a harness that does not test for a missing model (osu!mania's does,
and was re-run).

Notes on the real-time games:
- Malody is weighted accuracy (1 = every note perfect); Quaver and osu!mania 1,000,000 = the maximum; Snake 141 =
  the 12x12 board filled; Breakout 864 = both walls; Flappy Bird is capped by the deadline.
- Doom's held-out is transfer (unseen maps E1M5-E1M8 at skill 4; training E1M1-E1M4). Overcooked's held-out uses
  the training layouts; its transfer run on 8 unseen layouts (`overcooked/transfer-0024/`) averages 199.6, with 0 on
  the three forced_coordination layouts. Pong's held-out is its unseen second variation.
- osu!mania's champion waits for Jev approvals of its key schedules, so it presses nothing without models (0);
  its first ablation (System-1 set to None) is kept in `ablation-no-models.system1-none/`.
- Donkey Kong's champion farms barrel jumps (+100 each) and completes no screen. Minecraft reaches only the opening
  sub-milestones of "Free the End" within 15 minutes.
- Frogger's `search.pre-review/` (best 14.5, no frog home) is the archive from before the observation review
  added decoded hazards; the reviewed adaptor's search reached 4,433 (41-63 frogs home per episode).
- REPAIRS.md in a game's directory records operator interventions.

Notes on the turn-based games:
- Opponents: Hold'em plays a Monte-Carlo pot-odds bot, Catan three Catanatron AlphaBeta players (the held-out
  score 1.10 is 4 wins of 4), Yu-Gi-Oh! EDOPro's WindBot (0.299 is about 8 wins of 12), Dominion pyminion's
  big-money bot (its held-out uses unseen kingdoms), Hanabi is two-seat self-play (out of 25).
- Pokémon Showdown plays Foul Play (MCTS). No proposal beat the naive one-call-per-decision baseline, so the
  baseline is the champion and it loses most battles. `showdown/search-v1/` is the earlier search against a
  heuristic opponent (0.85 on training, 5 wins of 10 held-out) and `search-v2/` the first Foul Play attempt, whose
  battles did not finish before the deadline.
- `search-v1*` directories are earlier archives whose adaptor was replaced (a saturated score, or an opponent that
  was too weak or too slow); `game_v1.py` etc. are the replaced adaptors.
- Sokoban solves every held-out Boxoban level (10 of 10); TextWorld and Baba Is AI reach the maximum on held-out.
  Jericho (8 interactive-fiction games, normalised score) is the weakest result.
