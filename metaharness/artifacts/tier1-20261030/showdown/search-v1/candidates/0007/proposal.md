parent: 0004

## Hypothesis

The leader's remaining weakness is an incomplete threat model, not a broken
control loop. Missing revealed-move definitions silently become zero damage;
observed hits on voluntary switches are discarded; one global attacker danger
multiplier cannot distinguish move power, STAB, category, or a variable-power
attack. These estimates enter both Astra's prompt and the continuation guard,
so the same blind spot can corrupt model advice and automatic policy execution.

Concrete evidence from 0004's training traces:

- Seed 232003: Torkoal reveals Solar Beam on turn 1. Lanturn's Thunderbolt policy
  nevertheless continues on turns 13 and 14; the latter Solar Beam takes it
  from 322/367 to 34/367. Solar Beam is missing from the parent's public move
  table, so its damage function returns zero for this revealed move.
- Seed 333667: Typhlosion reveals Eruption while KOing switched-in Volcanion.
  Astra then says Iron Jugulis "absorbs the Fire pressure best" and selects it;
  Jugulis is KO'd at full HP before Earth Power. Raichu subsequently moves first
  and survives an Eruption. Eruption is also absent from the parent's table,
  and its switch-damage learner exits when the active identity changes.
- The same loss later has Chimecho Calm Mind into Gigaton Hammer, taking
  195/290 HP, followed by a Knock Off KO. Gigaton Hammer is likewise absent.

Across all ten parent traces, 54 of 187 parseable decision observations contain
an active foe with at least one revealed move missing from the reference. There
are 19 such move IDs, all damaging attacks in this sample. Large serialized
observations truncated by the archive logger were excluded from this count.

## Changes versus parent

Keep 0004's legal-action handling, guarded named-move policies, independent Astra
calls, timeout recovery, and wall-time reserve. Replace the incomplete threat
information with a move-specific evidence ledger. An archive-wide check found
the same 115 public move definitions as the parent:
the missing enemy attacks never occur in these own-team move_dex dictionaries.
Rather than inventing archive evidence, ask Astra for typed public move facts
(type, category, base power, accuracy, priority) for already revealed moves absent
from that dictionary. Validate the schema and ranges, reject a Status category
contradicted by observed direct damage, label those facts as model
inferences, and check their predictions against observed damage. Unknown facts
remain unknown. No species sets, seed mappings, hidden teams, or implementation
files are consulted. Current observation data always overrides inferred facts.

Record direct move damage across switches, retain censored KO observations as
lower bounds, and calibrate the matching move's estimate instead of spreading a
single multiplier over every attack. Track the conditions that affect a damage
estimate and flag unknown revealed moves explicitly. Include the resulting
per-move threats and empirical evidence in Astra's prompt. Unknown revealed
moves invalidate automatic continuation until Astra can reassess them. Account
for publicly activated Flash Fire when estimating Fire damage.

## Confirmation and refutation

Confirm if revealed high-power attacks appear in prompts and continuation guards,
previously ignored switch hits enter the evidence ledger, and fewer selected
replacements faint before acting. Win rate and weak-episode outcomes should
improve without undoing the parent's eight wins or increasing episode errors.
Refute if the richer model becomes overconservative, transfers damage evidence
across incompatible states, or adds information without changing bad decisions.
This is a single threat-information intervention, not a new call scheduler or
a seed-specific policy. The ordinary damage model remains approximate and no
unrevealed opponent set is treated as known.

## Validation

Pre-flight passed: 11 accepted actions, no exception, zero model calls because
the short virtual budget is below the parent's call reserve. The reproducible
`verify.py` checks all 187 parseable parent decision observations for legal
ranking and prompt construction and captures 35 switch-hit samples. Focused
checks exercise KO lower bounds, typed-fact validation, HP-dependent damage,
garbage replies, structured replies, recovery after a model timeout, and a
queued call abandoned at the wall reserve without launching another call.

In the Typhlosion loss replay, a stand-in answer supplying Eruption's public
mechanics changes Iron Jugulis's predicted incoming damage to 1.60 HP fractions.
The observed Volcanion KO is recorded as at least 97/256 HP, not an exact attack
strength. This verifies information flow, not a counterfactual win.

One live dev smoke test is pending. The submitted final revision also tightens
evidence matching for volatile effects and screens and reconstructs public
field/volatile changes before a hit; these bounded fixes have offline coverage.
