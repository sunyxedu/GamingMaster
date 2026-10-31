parent: 0007

# Hypothesis

The local controller prices the enemy cards it can destroy this turn but not
the opposing attack sequence against the board it leaves behind. This favors
flipping a defensive wall into a weak attacker to take a small useful trade,
even when retaining defense avoids a decisive counterattack. A two-sided
public combat horizon should improve survival and resource preservation without
requiring a slow model call for every attack.

The leader 0008's weakest training seed, 1111, is -0.373333. At steps 190–194
its sole Giant Soldier of Stone changes from 2000 DEF to 1300 ATK and attacks
Magician of Faith for 1000 damage. Gene-Warped Warwolf (2000 ATK), Luster Dragon
(1900), and Giant Soldier (1300) remain on the enemy field. Step 195 ends the
duel at 0 LP. The opponent has also summoned another Warwolf by then, so the
trace does not prove that retaining defense would win. It does show a concrete
exposure the one-sided search fails to price. The other well-documented Rush
deadlock already has a tested repair in 0007; this proposal uses that parent
rather than repeating its hypothesis or modifying the leader's chain recipe.

# Change under test

Keep 0007's legal adapter, Rush expansion and observed-stat verification.
Replace the one-sided combat utility with a bounded two-sided visible battle
search: enumerate our attacks and then the opponent's possible attack order
against the surviving board. Include defense position, equal-ATK destruction,
direct damage, resource trades, native LP lethal, and expiry of the pool's
temporary Rush buff. Unknown defenders and unrevealed interaction remain
uncertain; no hidden identity or future summon is generated. Compare legal
position-change choices with keeping the current position under this same
horizon. Execute one current action and recompute after the engine responds.

Record the predicted enemy damage and compare against observed LP/field
changes, with explicit uncertainty for newly revealed cards and effects.
Discrepancies and stalled public boards clear the plan and request Astra repair
with concrete evidence and ctx.progress(). Model output remains advisory and
is validated against the current menu. Calls retain a 235-second timeout and
210-second independent wait guard, above the observed 81-second recent maximum.

# Confirmation and refutation

Mechanically confirm if the archived vulnerable Giant Soldier stays in defense,
profitable safe attacks still occur, and legal prompts remain valid. Strategically
confirm only with more native wins and improved weak seeds; a replay cannot
establish a counterfactual win. Refute if pessimistic counterattack estimates
cause passivity, neglect Flip value, or sacrifice lethal pressure. Public-only
search cannot guarantee survival through unknown removal or a new summon.

# Validation

Implementation and preflight results will be recorded below before submission.
