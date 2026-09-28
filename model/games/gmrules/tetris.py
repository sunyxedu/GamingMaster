"""Versus Tetris — the 1v1 garbage battle under guideline rules, a candidate-action
game of the gmrules framework trained on Cold Clear self-play:

    rules + frames    data_processing/rs/gmrules/src/games/tetris.rs (SRS with its kicks,
                      T-spin / mini detection, attack table, back-to-back, combos,
                      perfect clears, garbage cancelling and insertion, one 7-bag
                      sequence for both players; Cold Clear's libtetris conventions)
    teacher           Cold Clear (MinusKelvin, MPL-2.0; data_processing/rs/gmtetris, its
                      own cargo workspace): `gmtetris selfplay` plays both seats on OUR
                      rules — every teacher move must be one of our legal placements
                      (0 mismatches on 264 test games)
    data              gmtetris selfplay → trajectory JSONL → `gmdata rules convert --game
                      tetris` (replays and verifies every record)
    simulator         gmsim.Sim("tetris") → env/vec_gmsim.py
    model             the generic CandidateEncoder / CandidateDecoder: 20 column tokens
                      (my field, the opponent's), a global token (pieces, hold, queues,
                      back-to-back, combo, pending garbage), ≤ 160 placement candidates
                      with their after-state features
    RL                train_rl_joint's generic CandidateSpec (per-seat streams: both
                      players place one piece per turn — a simultaneous turn)

One decision per piece: the final placement (hold or not, rotation, position, T-spin
status), not the key presses — every placement reachable at 0G is a candidate. One
episode = one game (outcome ±1, a draw 0 after ``max_turns``).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="tetris",
    sim_config={},
    description="Versus Tetris (guideline garbage battle, one piece per player per turn)",
)
