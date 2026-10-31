# Hanabi adaptor

This adaptor is a local, dependency-free Python implementation of the standard Hanabi rules. It runs in-process under the evaluator's Python 3.12 environment and starts no server, GUI, account, or network connection. `make_game()` returns a tier-1 `TurnGame`; `step()` applies one action synchronously and returns the next current-seat observation.

The default profile has two cooperating seats, seat zero starts, five colors (`R`, `Y`, `G`, `W`, `B`), ranks 1-5, card multiplicities 3/2/2/2/1, five-card hands, eight shared clues, and three shared lives. The implementation also accepts `num_players=3`, `4`, or `5` as separately labelled variants (five-card hands for 2-3 seats and four-card hands for 4-5 seats). `EVAL_KWARGS` fixes evaluation at two players. `info()["suite"]` labels the profile as `standard-2p`, etc.; report extension suites separately, never aggregate them into the primary two-player score.

Every reset creates and uniformly shuffles a fresh standard 50-card deck with a private `random.Random(seed)`. The initial multiset is ordered by color R,Y,G,W,B, then rank 1-5, then repeated copies. After Python's shuffle, dealing is round-robin starting at seat 0, popping from the end of the list; replacements also pop from that end. IDs begin at 0 and increment at deal/draw time. Reproducibility is scoped to `hanabi-standard-offline-1` and this Python RNG implementation, not HLE or other engines. All integer seeds are accepted, but the declared disjoint evaluation sets are:

| Split | Seeds | Purpose |
| --- | --- | --- |
| train | 11, 12, 13, 14 | Harness tuning |
| dev | 201 | Proposer experiments and local tests |
| validation | 301, 302, 303 | Finalist re-evaluation |
| heldout | 101, 102, 103, 104 | Final reporting |

These seeds were assigned without selecting for favorable outcomes. Seeds appear only in evaluator metadata, never observations or `describe()`. A reset discards all engine history, clues, counters, IDs and diagnostics.

The observation is JSON-safe structured state for the acting absolute seat. Its own cards expose only opaque card IDs and literal candidate color/rank masks; other hands expose their actual faces. Public fireworks, clue/life counts, deck size, discard pile, action history, final-round countdown, and all currently valid action objects are included. A clue updates positive matches and negative exclusions for every card in the target hand. Invalid or malformed actions leave state unchanged and set an `error` field in the next observation.

Actions are dictionaries: `PLAY` or `DISCARD` with `card_index`, or `REVEAL_COLOR` / `REVEAL_RANK` with an absolute `target` and color/rank. Alternatively use `PLAY 0`, `DISCARD 0`, `REVEAL_COLOR 1 R`, or `REVEAL_RANK 1 2`. Verbs and color letters are case-insensitive. `NOOP`, the action-spec no-op, does not advance a turn and is not a game pass. Bare parameterized verbs are rejected. There is no automatic replacement of bad actions; the harness should limit retries. `step()` returns the observation dictionary directly, not a Gym tuple.

Playing the next required rank advances a firework; a misplay loses a life and is discarded. Discards and rank-5 plays recover a clue (capped at eight). Discard is unavailable at eight clues. When the last card is drawn, the triggering action does not consume the countdown; exactly one further action per seat is allowed, unless all lives are lost or all five fireworks reach 5 first. Immediate life-exhaustion/perfect endings omit the useless replacement draw. Terminal observations keep the last actor's masked view and contain no legal actions; further steps do nothing.

`score` reports the fireworks sum during play and at a normal terminal, or zero when all lives are exhausted. Evaluate the final value once; do not sum cumulative scores. `fireworks_sum` is exposed separately in observations and `score_breakdown()` so progress remains visible after a bomb-out. `info()["reward"]` is the change in benchmark score on the last accepted step, including the negative change on a bomb-out; rejected actions and no-ops set it to zero. `score_breakdown()` also reports clue/life counts as diagnostics, not objective bonuses. Gameplay failures are valid trials; no external infrastructure is used. `info()["valid_trial"]` is therefore true.

The adaptor emits only the current seat's view and owns no policy sessions or private policy memory. The harness MUST route observations by `observer` into separate per-seat sessions and scratchpads, and clear them at every reset. Shared memory may contain only public events and preagreed conventions. A single accumulated transcript containing different seats' private views makes this centralized/omniscient play and must not be reported as standard hidden-information Hanabi. The adaptor cannot erase the caller's memory or sandbox Python introspection; policy code must not access engine fields, evaluator seed metadata, or the seed declarations. No built-in opponent, bot, convention, or open-hand mode is provided.

No credentials or environment variables are required; `env.example` is intentionally empty of mandatory values. `requirements.txt` documents that there are no additional dependencies to install. The game performs no file writes, downloads, telemetry, or background work. `close()` is idempotent; no external resources need releasing. Verification artifacts live under `runtime/`, which the source hash skips.

Verification from the repository root, using the required evaluator venv:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/hanabi/bin/python -m unittest discover -s runs/hanabi/adaptor -p 'test_game.py' -v
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/hanabi/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/hanabi/adaptor', expected_tier=1).to_markdown())"
```

The conformance suite exercises exact deck multiplicities and card conservation; reproducible reset/action sequences; own-hand/deck noninterference; positive/negative/repeated clues; new-card masks; discard legality and shifting slots; safe/duplicate/skipped plays; five-token recovery; perfect/bomb-out/deck-exhaustion scoring; all final-round action types for 2-5 seats; JSON detachment; malformed inputs; and full random/discard episodes. It runs against this actual local simulator, with explicit fixtures only for rare rule transitions. This is an original equivalent simulator validated against the fixed study profile, not a certified HLE port. An external HLE differential oracle is unavailable and was not claimed or used. No benchmark performance claim is inferred from random test scores.
