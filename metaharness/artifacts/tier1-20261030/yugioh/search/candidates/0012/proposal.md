parent: 0007

# Hypothesis

Summon selection undervalues converting bodies that cannot win a battle into a tribute monster that can. The parent's separate summon heuristics compare summed ATK and raw tribute value, rather than comparing the combat outcomes of the resulting boards. This creates a deployment horizon error even when attack assignment itself is correct.

Evidence: the current leader 0008, seed 1111, steps 103–110 holds two Blue-Eyes in hand while Luster Dragon (1900) and Battle Ox (1700) cannot attack through Mystical Elf (2000 DEF). It ends two Battle Phases without attacking and adds an 800-ATK Mystical Elf. Step 111 finally draws Summoned Skull, tributes the Elf and breaks the blockade. This duel eventually wins; the observed stall motivates the conversion hypothesis but does not establish that an earlier tribute would win sooner. In the same seed's losing third duel, step 180 Sets Giant Soldier instead of converting Luster plus Mystical Elf into the available Blue-Eyes; step 183 exits Battle with no productive attack. These are counterfactual opportunities, not evidence that a different summon guarantees victory.

# Change

Use 0007 as parent (not the leader). Preserve its legal action adapter, live-menu combat, Rush expansion and response safeguards. Replace deployment selection, only at menus offering a Tribute Summon, with a shared short public-combat horizon for every summon/set option and conserving the summon. Enumerate possible tribute sets, evaluate the resulting board and known battle exchanges, and bind the chosen material set to the actual native tribute selection. Do not assume hidden identities or future tokens.

Audit the expected summoned body after native resolution. A missing summon, mismatched material selection, rejected action or semantic loop invalidates the intent and sends concrete evidence to Astra. No projected battle is executed as a queue. Astra receives the competing deployment scores and assumptions; inference timeout is 235 seconds with an independent 210-second wait guard, above the observed 108–120 second model latency.

# Falsification and validation

Confirm mechanically if archived blockade positions choose a useful legal tribute and preserve productive attackers when the tribute reduces useful damage. Confirm strategically only with improved native wins/completion across deck/seat configurations. Refute if card disadvantage, opposing removal or an inaccurate public horizon outweighs the breakthrough gains. Replay cannot establish a counterfactual win. The checks below test the mechanism; training evaluation must establish playing strength.

## Development and archive replay

Final preflight passed: 382 actions, game_done, zero model calls in stand-in
mode, no exception. Separate fixtures exercised malformed model text,
TimeoutError, RuntimeError, stale/duplicate selections and a missing summoned
body; all recovered legally or raised the intended re-planning signal.

Both permitted development evaluations were used:

- Seed 3301, initial integration draft: 3 native completions, 2 wins, score
  **0.3785417**, 433 actions, 217 seconds. Its opening Astra call hit the
  210-second wait guard and was discarded. This run preceded the final fix
  preserving removal/setup priority over deployment comparisons; it is
  completion evidence for the draft, not an evaluation of the final file.
- Seed 3302, final file (SHA-256
  `cd9364d867fd16d8562b9e5c833ff0a642cc6efb6865bac6cb9e871785b1e508`):
  3 native completions, 2 wins, score **0.3685417**, 257 actions, 376 seconds.
  Two Astra requests produced one applied answer; the other hit the wait
  guard. Five deployment audits all returned `verified: true`, including
  Summoned Skull tributing Giant Soldier at turn 10 and Luster at turn 18.

Neither run had errors, forfeits or truncations. The final development trace
confirms that the new tribute selection and consequence audit execute in the
native game; it does not isolate a win-rate gain from this intervention.

The final archive replay covered 3,591 leader observations with zero invalid
generated actions. It changed 9 of 23 detected tribute forks. At seed 1111,
step 180 selected a face-up Blue-Eyes Tribute Summon from a previously stalled
board, while the baseline selected a low-impact Set. At step 178 it correctly
kept Fissure before deploying. It did not take the earlier Blue-Eyes
opportunity at step 106; the search does not consistently identify every
blocked-board conversion. A 2-attacker fixture confirms that an otherwise
available Blue-Eyes summon is rejected when the existing open-board attackers
have better projected pressure. These are mechanism checks only; there is no
counterfactual result for archived actions and no training evaluation here.

## Remaining limits

The horizon intentionally omits new opposing summons, spells, traps and Flip
resolution, and cannot guarantee combat outcomes through hidden interaction.
It scores unknown defenders as unresolved blockers. The search only intervenes
when a legal Tribute Summon is offered; ordinary openings retain the parent
policy. Selection validation enforces the native constraints after the
projection, and a discrepancy discards the projected materials. Failure
recovery remains bounded by available Astra time and budget.
