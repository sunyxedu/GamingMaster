"""Heads-up no-limit Texas hold'em (HUNL) — the ACPC game (50 / 100 blinds, 20000-chip
stacks = 200 big blinds, stacks reset every hand), through every layer:

    rules + frames    data_processing/rs/gmrules/src/games/hunl/ (ACPC dealer rules =
                      OpenSpiel universal_poker ids; schema "candidate_abstract.v1": the
                      candidate layout over 15 ABSTRACT candidates — fold, check/call,
                      all-in, min-raise, pot fractions 0.25..5 — while the rules and the
                      records keep exact amounts)
    reference         PokerKit differential tests (tests/test_hunl_reference.py,
                      scripts/reference/pokerkit_hunl.py)
    data              ACPC 2009-2017 hand histories (PHH, Zenodo 10796885, CC-BY-4.0):
                      scripts/fetch_hunl.sh → `gmdata hunl fetch|convert` (every hand
                      replayed and verified with its recorded payoffs)
    simulator         gmsim.Sim("hunl") → env/vec_gmsim.py: 2 seats, per-seat streams
                      (hidden cards), alternating, deck dealt from the episode seed
    model             the generic CandidateEncoder / CandidateDecoder (candidate layout)
    RL                train_rl_joint's generic CandidateSpec (``--actor stream``)
    evaluation        scripts/slumbot_client.py (Slumbot's public API, optional)

IL targets: ``action_idx`` (the recorded action's nearest candidate: the decoder's
label), plus the exact size of raises — ``raise_to`` (chips, total commitment) and
``raise_frac`` (raise increment / pot after calling) — for a later continuous size
head.  Outcome (value target, RL reward) = chips won / stack.  Frames never contain
the opponent's hole cards or future board cards (causality / privacy batteries).
Frame dims come from the simulator's spec meta (they do not depend on the blinds or
stacks, so one network plays every config).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="hunl",
    sim_config={"small_blind": 50, "big_blind": 100, "stack": 20000},
    description="Heads-up no-limit Texas hold'em (ACPC rules, 200bb), abstracted bet sizes",
)
