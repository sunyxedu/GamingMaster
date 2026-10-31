parent: 0000

Hypothesis: the leader's failure is primarily throughput and horizon, not illegal
actions. Its trace shows only 2--8 completed hands after 1,800 seconds because every
turn spends a 40--69 second Astra call; the worst seed is still deciding whether to
continue with pocket aces on a flop when the deadline arrives. The resulting score is
therefore dominated by a tiny, seed-dependent prefix. A verified coarse policy can
play many more hands, while reserving model time for an actual policy failure.

What changed: 0002 owns a legal, position-aware preflop policy (wide button opens,
price-sensitive big-blind defense), and a conservative postflop policy based on
made-hand class, draw detection, and pot price. It computes only from the public
observation and always selects an advertised candidate. After every action it checks
progress and illegal-action counters. If the script ever violates its assumption, it
hands the exact observation and error to Astra, sanitizes the response against legal
candidates, and then verifies the recovery step. This is structurally different from
0000's one-call-per-decision conversational player.

Evidence that would confirm/refute it: materially more completed hands and a stable
score across the four training seeds would confirm that latency was the main failure
mode. A lower score despite substantially higher hand count, especially concentrated
losses after postflop calls, would refute the coarse-strength hypothesis and motivate
model-guided range/equity decisions in a later iteration. Any illegal-action count or
stalled progress would refute the legality/postcondition implementation.
