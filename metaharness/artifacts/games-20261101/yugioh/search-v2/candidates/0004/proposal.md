parent: 0003

## Hypothesis

The leader already wins all 18 training duels. Its remaining demonstrable failure is action semantics: the interface uses the combined label `change position / flip summon`, but the controller tests the substring `flip summon` before `position`. It consequently values turning an existing attacker into Defense Position as if that action were bringing a monster into Attack Position. This sacrifices damage, sometimes a visible-board lethal, and exposes low-DEF attackers to future battles. Resolving ambiguous commands against the current visible state should preserve the leader's reliable completion while reducing missed lethals and unnecessary engine decisions.

## Trace evidence

The worst-scoring parent episode is seed 1101 (1.090833), with all three native wins. In `candidates/0003/trace_seed1101.jsonl`, line 65 posts `choose r1p31 0`, labeled `change position / flip summon`. Gene-Warped Warwolf is already in `face-up attack` with 2000 ATK and 100 DEF; the opponent has no monsters. The returned event is `position: face-up defense`, so a potential 2000 direct attack is lost. At line 131, `choose r1p64 0` similarly turns Alexandrite Dragon to defense while our two 2000-ATK attackers face an empty monster field and 3100 opposing LP. This removes known-board lethal. The duel eventually wins on turn 11 instead of taking that turn-9 damage opportunity. Unknown interaction remains a caveat; this is a missed public-board lethal, not proof of a guaranteed earlier win.

The defect is not confined to the worst score: seed 1102 line 363 switches an already face-up Summoned Skull to defense, and seed 1104 lines 135 and 179 switch 2000-ATK monsters to their 100-DEF positions. These legal commands cannot be caught by a token or subset validator.

## Change from parent

Keep the parent's full control loop, complete-action validation, mode-first subset handling, battle assignment search, tactical scores, sparse bounded Astra calls, and fallback paths. Add one semantic normalization layer: derive the actual position transition from the hydrated current monster before scoring the action. Face-down monsters retain the Flip Summon branch; face-up monsters reach the parent's previously shadowed position-change branch. The original exact action string is never rewritten. Astra's ranked options also receive the resolved transition, so adviser and executor see the same operation.

This is deliberately a targeted correction to a measured defect in a near-ceiling leader, rather than a wholesale policy change or a seed-specific rule. All decisions use the current public/owned observation only. No game implementation, private state, hidden identities or held-out results are used.

## Confirmation and refutation

Confirm the mechanism by replaying archived public observations: the cited attack-to-defense moves should fall below entering battle, while useful face-down Flip Summons and defense-to-attack moves remain available. Native dev and training runs should retain wins and zero forfeits/errors, with fewer turns/actions and no gratuitous conversion of open-board attackers to defense. A score gain is possible only through the small LP component unless the change improves generalization; equal score with fewer calls/actions is also useful under the selection rule.

Refutation would be lost wins, reduced damage, or new stalls caused by the now-reachable position logic; for example, a position change the old mistake accidentally avoided could expose a weak attacker. Archived-state checks validate the mechanism, not downstream outcome. Dev results and preflight results will be recorded below after the complete harness is tested.
