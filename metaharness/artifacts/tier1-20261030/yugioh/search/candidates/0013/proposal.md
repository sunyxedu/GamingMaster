parent: 0008

## Hypothesis

The leader's remaining weakness is a turn-completion error in its cheap policy. It treats the presence of a legal Battle Phase as evidence that setting a Quick-Play or Trap is undesirable, even when there is no useful attack. That can leave the agent with no field and no interaction while ending a turn that had a legal defensive setup available.

The clearest evidence is `trace_seed1111.jsonl`: at steps 39 and 43, the agent has Book of Moon in hand, no monsters, and an available `set spell/trap` choice, but chooses `end turn`; steps 80 and 81 repeat the same pattern. The same shape appears elsewhere with Rush Recklessly (seed 1109 steps 105 and 121, seed 1105 steps 7 and 15). These are observed missed setup actions, not claims that every set would have changed the duel result.

## Change

Starting from 0008, retain the validated constrained-action parser, public combat allocator, chain projection, target binding, and guarded Astra recovery. Replace the static Quick-Play set penalty with a small phase-completion test. For each offered set, inspect the current face-up attackers and visible defenders. If no profitable attack is available, value setting Book, Trap Hole, Mirror Force, Compulsory Evacuation Device, or Call of the Haunted; value Rush and Mystical Space Typhoon when the opponent presents a visible threat or there is no own attacker. If a profitable attack exists, keep immediate activation or Battle ahead of a speculative set. Battle Phase itself is scored low when it cannot produce a visible attack, allowing a useful set to win the current menu.

This is deliberately local and information-limited. It does not assume hidden card identities, guarantee that a set will resolve, or queue a future action. Every native step still gets a fresh menu and the existing progress, token, illegal-action, and consequence checks can clear the local preference and escalate to Astra when the board stalls.

## Confirmation / refutation

Confirm if replay shows the previously skipped Book/Rush set opportunities now become legal setup actions, while profitable attacks and immediate effect windows remain unchanged, and native completion/error rates stay clean. Strategic confirmation requires more wins on training and unseen seeds, especially the weak seats and deck matchups. Refute if the policy sets interaction while a necessary attack was available, loses tempo by overvaluing empty-board cards, or does not improve native outcomes despite correcting the observed phase error.

The archive's latency measurements support retaining the 0008 planner guard: Astra remains a recovery mechanism rather than a call on every ordinary menu. The evaluator's training result is required to distinguish a real win-rate gain from a mechanically correct setup repair.

## Development check

The permitted development run on seed 3301 completed all three duels natively at 0.3785417 (2 wins, 1 loss), with 449 actions, one Astra request, and no errors, forfeits, truncations, or invalid actions. Its setup audit verified Rush and Book placements. This is completion evidence only; the training evaluation must establish whether the repaired setup converts into more wins.
