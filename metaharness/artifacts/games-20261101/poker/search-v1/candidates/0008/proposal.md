parent: 0006

Preflight cause note: the 0.35s pre-`ctx.act` margin let the frozen opponent's multi-second equity step race the 60s deadline; the harness now stops with an 8s margin and checks immediately before and after policy work.

## Hypothesis

The fast equity policy still treats the river as mostly a fresh showdown
decision. Public continuation actions carry information that its random-hand
equity omits: a heuristic-v1 call on the flop or turn removes many weak hands
before the river. The leader's seed-2207 hand-22 is a concrete failure: after
the opponent called a turn bet, candidate 0003 jammed 19,500 with Js8c on
Td4h9dJcTs and lost 20,000 to JhQc. The turn call should make a later thin
value jam less attractive even when raw river equity is high against all legal
hands.

## What changed vs. the parent

Starting from candidate 0006's complete, code-only policy, this harness keeps
its cheap legal candidate mapping, board-aware hand checks, and sampled
showdown equity. It adds a public-history continuation posterior. On river
calls, prior opponent calls and raises add an explicit realization margin. On
river value bets after that evidence, the equity threshold rises and the target
shrinks to roughly half pot; untouched rivers retain the parent sizing. The
same evidence is used when deciding whether to call a large river wager.
Caching remains per hand and all computation has cooperative deadline checks.

## Evidence that would confirm or refute it

Confirmation would be completed 100-hand matches with zero illegal actions,
fewer 200bb river pots after earlier calls, and improved mean or reduced
match-block variance relative to 0006. Traces should show the history branch
on hands where an earlier call is followed by a river value decision. Refutation
would be a lower mean caused by missed thin value against the heuristic's
price-sensitive calls, or no reduction in large-pot losses. Two development
seeds remain diagnostic; independent training matches are the selection test.
