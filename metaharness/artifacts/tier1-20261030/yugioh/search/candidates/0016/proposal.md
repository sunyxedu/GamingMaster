parent: 0013

## Hypothesis

The phase-completion repair in 0013 fixes missed Book/Rush setup, but its strategic review can still expose a defensive wall. In the archived seed-1111 trace, a 2000-DEF board is eventually changed into a weak attack position while the opponent has several face-up attackers; the next opponent battle reduces the agent to 0 LP. This is a high-confidence public-board failure, unlike hidden trap outcomes.

## What changed

I copied 0013's constrained parser, chain projection, setup-at-end-turn repair, visible combat search, and bounded Astra reviews. Candidate 0016 adds one narrow `unsafe_defense_flip` gate. It rejects a current `change position` action only when the selected monster is face-up Defense, every visible opposing monster is at least as strong in the relevant battle statistic, and no face-down defender leaves uncertainty. The gate applies both to Astra's parsed answer and to the final local fallback. It then chooses another live menu action (Battle/Main 2/End when available), preserving fresh-token validation and all existing consequence audits.

## Evidence that would confirm or refute it

Replay should show fewer defense-to-attack transitions into a visibly stronger board, no increase in illegal or stale actions, and preserved Book/Rush setup behavior. Training and unseen results would confirm the strategic value if losses from counterattack fall without sacrificing profitable attacks. Refute it if the gate blocks necessary lethal attacks, causes avoidable stalls, or leaves native completion unchanged while the same counterattack failures persist. The check intentionally does not infer hidden cards or override engine legality.
