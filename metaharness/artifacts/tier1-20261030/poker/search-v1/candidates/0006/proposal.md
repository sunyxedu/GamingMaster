parent: 0002

## Hypothesis

The opponent's free checks leave a profitable middle ground that both parent
0002 and the range-heavy leader underuse. Heuristic-v1 value-bets freely only
at fairly high random-hand equity; otherwise it checks, except for its small
mixed branch. In position, a small flop or turn probe with an ace/king blocker
or a real draw should therefore win some checked pots while risking far less
than the leader's repeated 200bb jams. Against a wager, the opposite signal is
useful: a large price should be treated as value-heavy and called only with an
equity margin over pot odds.

## What changed vs. the parent

This candidate starts from parent 0002's board-aware, code-only loop and keeps
its exact candidate mapping, hole-card help test, and no-model-call completion.
It adds deterministic showdown sampling for price decisions, explicit sizing by
street, and a new action branch keyed to the public history: after the opponent
checks on the flop or turn while hero has position, bet about one-third pot with
a high-card blocker or genuine draw when showdown equity is not already strong.
Value bets use modest fractions of the pot, deep reraises are reserved for high
equity, and river bluffs are suppressed. Every action remains an id from the
current observation's legal candidates; malformed stand-in model text cannot
affect the loop because Astra is not called.

## Evidence that would confirm or refute it

Confirmation would be complete, zero-error matches whose traces contain probe
bets after opponent checks, positive chips/hand, and fewer very large pots than
candidate 0003 while retaining value extraction. In particular, probe pots
should show folds or cheap calls often enough to exceed their one-third-pot
cost. Refutation would be a lower mean with probes losing at showdown or being
called too frequently, or no meaningful probes because the position/history
condition is wrong. Independent match scores matter more than any one hand;
the two available development seeds are diagnostic only.

## Local validation

Preflight passed with 153 legal actions and no model calls or errors. One
independent development seed (901) completed all 100 hands in about 60 seconds,
with zero illegal actions and 391.81 chips/hand. This is diagnostic only; it is
not a training or held-out result.

## Trace evidence motivating the test

The leader's seed-2207 trace repeatedly posts near-stack targets (for example
`raise_to 19500` with `8c5s`, `raise_to 19550` with `5c8h`, and `raise_to 20000`
with `9d9s`) while that completed match scores only 83.0 chips/hand. Its seed
4409 trace also has a checked-through hand with Qc6s on Kh 7h Ts 5c Jc, ending at
-250 chips after the opponent's revealed Ac9d; this is the kind of free-street
node where a small, position-aware probe is a testable alternative to automatic
checking or a deep commitment.
