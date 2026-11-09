parent: 0000

## Hypothesis

The baseline is failing primarily at the interface boundary, before its strategy can matter. In seed 105 Astra repeatedly returned `build_road 11 29` while the current legal actions were IDs `a0=[10,11]`, `a1=[11,32]`, and `a2=[11,12]`; the engine left the agent in `build_initial_road` for 18 steps. Seed 106 shows the same pattern with `build_settlement 49` when vertex 49 was not legal. Because the adaptor reports rejected/stale actions without raising, a textual parser can silently deadlock for the whole 1,800-second episode.

## What changed vs. parent

This harness treats `legal_actions[].id` as the only executable vocabulary. It ranks setup vertices from the observed hex numbers, resource types, robber, ports, and topology, and ranks normal cities, settlements, roads, trades, and robber targets with deterministic bookkeeping. Astra is called only at the first meaningful choice of a turn (or setup/robber), receives a compact current legal-ID table, and its answer is accepted only if it names an ID in that exact table. Forced rolls/discards and routine continuation use code, and malformed, timed-out, or rejected calls fall back to another current legal ID. A bounded plan note is retained without growing the model conversation.

## Evidence that would confirm or refute it

Confirmation would be a large reduction in repeated unchanged observations and a rise above the baseline's 1–2 terminal VP, especially on seeds 105/106, with zero legal-action errors. If scores remain near 2 after the harness progresses through normal turns, then the interface deadlock was only the first bottleneck; traces would show whether production-aware placement, build ordering, robber choices, or the sparse Astra interventions need revision. A high rate of Astra answers rejected as non-ID text would refute the prompt/parser assumption and motivate even less model control.
