parent: 0004

## Hypothesis

The leader delegates tactical timing to a prose policy and scores most fallback
choices in isolation. A switch that looks defensively sensible can lose an entire
attack and expose the replacement to two hits before it acts. A two-turn exchange
search should use Astra more effectively as a source of uncertain threat hypotheses
and team priorities, rather than as the immediate action selector. This tests a
different architecture from the leader's continuation recipe and from the parallel
proposals' threat-ledger repair and model-call ambiguity gate.

Concrete evidence: in 0004 seed 232003, step 17 leaves healthy Salamence against
Lucario at 47% after Dual Wingbeat. Astra then says "Bring Scyther in to outspeed
and KO Lucario"; step 18 spends the turn switching and Flash Cannon takes Scyther
from 214/249 to 72/249. In seed 333667, Astra says "Iron Jugulis absorbs the Fire
pressure best" after Eruption has already KO'd Volcanion. Jugulis is then KO'd
from full HP before Earth Power; Raichu subsequently acts first and weakens
Eruption. These are testable exchange-order failures, not just bad type labels.

## Changes

Keep the parent's observation-only mechanics helpers and legal fallback, but
replace its action/policy controller with bounded two-turn simultaneous-action
search. Astra supplies an uncertain opponent move portfolio and per-team strategic
priorities at sparse reveal checkpoints. It does not choose the submitted command.
Each legal root action is compared against the same plausible opponent responses;
search accounts for switch tempo, entry hazards, speed/priority, damage-dependent
Eruption, Tera, healing, setup and misses. Follow-up choices are computed anew after
each real observation, never executed as an open-loop queue. Public move definitions
are collected from completed archived observation move_dex fields only.

## Confirmation / refutation

Confirm if the search selects replacements that can actually act, pays fewer free
hits on switches, uses fast damage to weaken HP-dependent attacks, and completes
battles with fewer model calls while improving outcomes. Refute if approximate
rollouts miss mechanics, model threat hypotheses cause excessive caution, or the
short horizon ignores a necessary sacrifice/setup line. Log root values and threat
provenance so later iterations can distinguish a planning failure from a model or
mechanics error. No seed-conditioned policy or hidden state is used.

## Validation

Completed validation is recorded below.

### Offline evidence and a qualification

The initial pre-flight passed (18 accepted actions, no error). Its short virtual
budget did not admit Astra, so separate stand-in checks exercised garbage text,
wrong JSON types, TimeoutError and RuntimeError recovery. Search returned legal
root actions on all 1,105 usable decision observations from completed archived
candidates. This checks robustness and coverage, not off-policy win rate.

The replay sharpened the diagnosis: the Scyther/Lucario switch cited above remains
the highest-value search choice. Its HP cost is real but does not by itself show
the switch was wrong; this counterexample argues against a blanket anti-switch
rule. At the Eruption forced replacement, adding the observed move's public
mechanics makes search select Raichu over Iron Jugulis. The opening Giratina
position also selects an immediate Clefable switch instead of taking two Dragon
Pulses with Volcanion. These are mechanism checks, not completed counterfactual
battles.

One dev evaluation was started on seed 869431 using initial revision SHA-256
253e77c95809bedc2082ec4620d5aea5b65f8057201cbcd3f2577ccd58645f05.
Final small fixes after launch merge hypotheses before constructing prompts,
validate malformed hypothesis field types, re-consult Astra for a newly revealed
move lacking a definition, and apply publicly activated Protosynthesis/Quark Drive
stat modifiers. Results and final pre-flight will be recorded below.

### Scope of the tactical approximation

The search is a small counterfactual evaluator, not a cloned Showdown simulator.
It enumerates up to three opponent moves plus one revealed switch target and uses
a blend of expected value and the worst branch. Unknown bench Pokémon are never
invented. Root attack misses have explicit branches; opposing misses use expected
damage. A replacement request does not pay a free enemy attack, while a voluntary
switch does. It spends one-use Tera only on the legal current root action.

Known limits include approximate hidden stats/items, opponent Tera uncertainty,
pivots (valued without a full nested replacement search), Transform/Illusion,
weather changes caused by simulated switches, partial status/volatile semantics,
and a two-turn horizon's inability to prove an entire setup endgame. The generic
STAB fallback remains explicitly hypothetical. These limits can invalidate search
values, so root scores and model threat portfolios are logged for diagnosis.
Contrary/Simple, White Herb and Technician multi-hit corrections were made after
the initial dev launch and checked offline. The final pre-flight again passed
with 18 accepted actions, no exception and no admitted model call.

### Completed dev evaluation

The single dev run at `dev/20261030-083709/` completed normally: **win +1.1579145711**,
19 accepted decisions / 19 turns, all six own Pokémon surviving, own HP fraction
0.6316582846, no episode error, and zero pending tasks at closure. Wall time was
1681.21 seconds. One System-2 call was admitted but **no answer arrived**; the
120-second reserve ended the wait and every actual decision came from code. The
trace contains no search or legality failure. This confirms graceful completion
under a stuck call and provides one favorable result for the search fallback. It
does not confirm that Astra-generated threat hypotheses improve play, nor does a
single development seed establish generalization. No second dev run or full
training evaluation was performed.

Final validation also checks that model hypotheses cannot overwrite public move
definitions, Contrary/Simple and White Herb causal transitions, and a pending-call
reserve with a stand-in future. Final pre-flight passes (18 accepted actions,
no exception). Later edits cap unrevealed named hypotheses to remaining move slots
and avoid duplicating a supplied STAB attack with a synthetic one. They were not
part of the real dev revision; the final harness was separately pre-flighted.

Final harness SHA-256: `45db3d9e84130354f42fa69a6e05bec6beafe39c2acc08fe29c3fdc6c1bb100b`.
