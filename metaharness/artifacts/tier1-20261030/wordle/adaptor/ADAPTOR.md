# Wordle and Mastermind adaptor

This is a local, offline Tier 1 implementation of the selected turn-based games. It
starts no server, process, GUI, network connection, or account. `close()` is
idempotent and has no external resources to release.

## Wordle

`make_game(mode="wordle")` implements normal English Wordle: a five-letter hidden
answer, six guesses, no hard-mode constraint, and duplicate-aware green/yellow/gray
feedback. `reset(seed)` uses Python's seeded PRNG to choose from 2,315 answers. The
bundled historical answer snapshot is from Josh Stephenson's `Wordle-Solver`
repository, commit `48c32e1ac0cb1ec87740cab009214e89b2c636c8` (2025-04-11), which
documents the 2,315 NYT answer words. The 12,972-word allowed guess list comes from
Tab Atkins' `wordle-list` initial snapshot, commit
`ae0403f957ae7312d72929403432becc0b0f5d63` (2022-01-13). Both repositories include
MIT licenses; copies are included in `WORDLIST_LICENSE.txt` and
`GUESSLIST_LICENSE.txt`. These are community-maintained snapshots of Wordle lists,
not a live or NYT-supported feed. The pinned SHA-256 values are:

- `answers.txt`: `f40e47975cf360ff9febe7d49c8e285054d8fa3fb9644cea575922060951d6ee`
- `allowed_guesses.txt`: `feacc549249b5dcb43e2799fbbcda14ebc33d55904fb15d043779104e60c1579`

Both lists are public and their paths, version, counts, and hashes are included in
the observation rules, so a harness can filter candidates without receiving the
secret. Every answer is accepted as a guess; guess-only words are accepted from the
second list. The answer is only included in evaluator-only `info()` after the episode
ends.

## Mastermind

`make_game(mode="mastermind")` defaults to classic Mastermind: four positions, six
color IDs (`0` through `5`), repeats allowed, and ten guesses. A seeded PRNG selects
from the complete configured code space. Feedback first counts exact-position pegs,
then counts matching multiplicities among unmatched positions. `positions`,
`colors`, `repeats`, and `max_guesses` can be passed as explicit JSON kwargs for a
variant; they remain fixed across seeds for that evaluation. The code is only
included in evaluator-only `info()` after completion.

## Interface and scoring

Wordle accepts `guess crane`, the bare string `crane`, or `{"guess":"crane"}`.
Mastermind accepts `guess 0123`, the bare digit string, or `{"guess":[0,1,2,3]}`.
`noop` is the only parameterless action. Invalid actions return an observation with
an error and consume no guess; valid repeated guesses do consume one. Observations
contain public rules, turn count, history and feedback, terminal flags, error, and
the explicit `solved`, `guesses_used`, and `max_guesses` result fields. They never
contain the seed or hidden answer. Scalar score is zero for an unfinished or lost
game and `(max_guesses + 1 - guesses_used) / max_guesses` for a win. `info()` and
`score_breakdown()` retain the primary result fields.

The train, dev, validation, and held-out groups are disjoint. Their seeds were
selected so each group has the same mix of answer words with and without repeated
letters, and Mastermind targets with two, three, or four distinct colors, with and
without repeats. The rules and target-selection process are identical in each group.
There are no opponents in these single-player puzzles. For headroom verification, a
candidate-elimination Wordle solver and Knuth-style Mastermind minimax policy were
run on the 12 train and two dev seeds (14 episodes per game), without consulting
held-out or validation results. Wordle solved 14/14 in a mean 3.64 guesses (mean
score 0.56); Mastermind solved 14/14 in a mean 4.36 guesses (mean score 0.66). A
random legal guess policy solved 0/14 in both games. The detailed episode results
are kept under `runtime/baseline_results.json`.
