"""Backgammon — match play with the doubling cube (money play as a variant), a
candidate-action game of the gmrules framework trained on gnubg self-play:

    rules + frames    data_processing/rs/gmrules/src/games/backgammon.rs (gnubg's rules:
                      bar, blocking, bearing off, the max-dice / larger-die rules,
                      gammons / backgammons, cube, Jacoby, Crawford / post-Crawford)
    reference         gnubg (GNU Backgammon, GPL, an external process in a throw-away
                      uv env): legal move sets (`gmdata backgammon diff`), its own
                      match loop (scripts/reference/gnubg_selfplay_dump.py +
                      `gmdata backgammon verify`), tests/test_backgammon.py
    data              `gmdata backgammon generate` (gnubg 2-ply plays both seats, our
                      engine runs the matches and checks gnubg's legal list at every
                      checker play) → `gmdata backgammon convert` (replays and verifies
                      every record; gnubg's equities → soft targets, its match-winning
                      chances → engine values); scripts/fetch_backgammon.sh end to end
    simulator         gmsim.Sim("backgammon") → env/vec_gmsim.py (dice from the
                      per-episode seed: a generated record replays exactly in the env)
    model             the generic CandidateEncoder / CandidateDecoder: 28 entity tokens
                      (points, bars, trays), a global token, 64 candidate tokens
    RL                train_rl_joint's generic CandidateSpec (interleaved stream: one
                      row per match, every frame from the mover's view)

Decisions (one frame each): the cube decision of the player on roll (no double /
double), the opponent's answer (take / drop), and the checker play — every candidate
a FULL move (the distinct resulting positions of the roll, gnubg's move list) with
its after-state features; a roll with more than 64 distinct moves (4% of gnubg's
checker plays, big doubles) is split into single checker moves until the rest fits.
One episode = one whole match (outcome ±1); `--env_kw backgammon:match_length=N`
(0 = money play, outcome clamp(points / money_scale)). The frames carry the absolute
away scores, so one network plays every match length.

Full-scale data (cluster; `scripts/fetch_backgammon.sh OUT MATCHES WORKERS`)::

    scripts/fetch_backgammon.sh ~/gm-data/backgammon 100000 256
    # = gmdata backgammon generate --out OUT/raw --matches 100000 --workers 256 --plies 2
    #       --seed 1 --keep-going   (+ 2% more matches with --seed 2 into OUT/raw_val)
    #   gmdata backgammon convert --input OUT/raw --out OUT/train --soft-k 8 --engine-values
    #   gmdata backgammon convert --input OUT/raw_val --out OUT/val --soft-k 8 --engine-values

Expected corpus (7-point matches, gnubg 2-ply, measured on samples): ~5 games and
~350 decisions per match; 100k matches ≈ 35M frames ≈ 75 GB of shards (~2.2 KB per
frame: bit-packed candidates) + ~1 GB of gz transcripts; ~13 s per match per worker
(one CPU core each) → ~1.4 h on 256 cores. Needs uv and a glibc >= 2.28 x86-64 or
macOS arm64 host (the gnubg 1.1.0a16 wheels).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="backgammon",
    sim_config={"match_length": 7},
    description="Backgammon with the doubling cube, 7-point matches (money play: match_length=0)",
)
