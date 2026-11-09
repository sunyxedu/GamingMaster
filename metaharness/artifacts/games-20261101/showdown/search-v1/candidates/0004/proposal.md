parent: 0001

## Hypothesis

The leader's poor results have a coupled cause: per-decision Astra calls consume
the deadline, while a single 100-second timeout permanently switches to a weak
fallback. The fix should preserve Astra's tactical intent over several decisions,
with feedback guards, rather than ask the same isolated question every turn or
blindly queue slot numbers. A threat-aware fallback is necessary for this test.

Concrete evidence: in candidate 0001 seed 282617 the opening call times out,
Empoleon faints to Supercell Slam, and step 2 selects Braviary-Hisui, which also
faints to Supercell Slam at step 3 without attacking. In seed 333667 the second
call times out; Eiscue and then Iron Jugulis are sacrificed to Typhlosion's
Eruption, while the later Raichu outspeeds and almost KOs it. Seeds 130363,
206369, 257053 and 308081 end at the deadline rather than completing. Seed
257053 repeatedly cycles Drednaw/Flareon through Moltres/Vileplume while paying
one model call per request. Parent code also infers resistance by reversing
weaknesses, which is not a valid type chart (Dragon resisting Water is missed).

## Changes

Astra returns one legal immediate action and a short, named-move matchup policy.
Code resolves names against each new request and reuses the policy only while
identity, field and threat guards hold. Ordinary damage does not invalidate a
policy; dangerous damage, a changed foe, changed typing, enemy setup, failure or a
binding volatile does. Recovery is conditional on HP and survival. A forced
replacement can carry a plan for the selected teammate. Explicit bounded memory
replaces an ever-growing model session. Timeouts are caught and do not permanently
disable later calls. A deterministic threat-aware fallback uses the complete type
chart, exact own stats, approximate opponent stats, observed damage, entry hazards,
speed, priority, abilities and Tera; estimates remain labeled as estimates.
The standard directional chart is adapted from candidate 0002, replacing the
leader's invalid reverse-weakness approximation. A static move reference contains
115 generic move definitions from archived public `move_dex` observations; it
contains no seed keys, opponent teams, species-set tables or hidden information.

Calls use independent sessions and a maximum of 22 attempts, with a short cooldown
after failures. The final revision wraps `ctx.think` in tracked `ctx.spawn` work:
if total waiting leaves less than 100 seconds, it finishes with legal code actions
and leaves the evaluator to cancel the model. It never launches another call while
that call is pending. This handles queue time that is absent from reported model
latency. Initial call admission still requires more than 185 seconds remaining.

Final guard fixes accept a selected teammate's own expected weather/terrain entry
change (while checking the opponent is unchanged), model terrain priority/damage,
and do not invalidate a priority KO merely because another move has more raw
power. Unexpected changes still trigger replanning.

## Confirmation / refutation

Confirm with zero errors, fewer Astra calls per completed battle, accepted guarded
continuations after routine damage, and fewer replacements immediately KO'd before
acting. Improvement must include wins and HP margin, not just fewer deadlines.
Refute if continuation guards miss decisive state changes, setup/recovery loops
replace progress, or completing more battles merely converts unfinished games to
losses. New model calls and policy executions will be logged separately.

## Validation and limitations

- Final pre-flight passed: 11 accepted actions, no exception. Its short virtual
  budget caused zero System-2 calls, so this is a control-loop check, not a test
  of actual model quality.
- Offline replay of 182 legal archived decision observations exercised ranking
  and public-memory updates without errors; every returned option was legal.
  Focused checks covered malformed output, illegal Tera rejection, timeout
  recovery, structured-policy reuse, foe-Tera invalidation, known terrain entry,
  priority-KO continuation and a queued model abandoned at the wall reserve.
- One real dev evaluation was run, seed 869431, under
  `dev/20261030-062409/`: win **+1.160339**, four survivors, HP fraction 0.641356,
  15 decisions / 13 turns, 12 System-2 calls, 1290 seconds, no episode error.
  A 145-second timeout was caught; fallback Shadow Ball KO'd Gallade, and later
  Astra calls succeeded. Two policy continuations were used, including Nasty
  Plot followed by the planned Focus Blast KO on Arceus-Steel.
- That dev run evaluated the initial complete revision, hash
  `398c1f15cfa2fbe4f8ab5b3092efb370896e585992d6f5ddae2d174d9ee7532d`.
  The final public-item/damage-memory, wall-wait cutoff, terrain and continuation
  fixes were checked offline and pre-flighted, but were not evaluated in a
  second live episode. Only one dev run was used; no full training evaluation
  or held-out results were accessed.

The single dev win is evidence of working integration and recovery, not a robust
win-rate estimate. Only two continuations were executed, so the call-saving
hypothesis is still weakly tested. Opponent stats/damage remain estimates and
unseen coverage, unusual abilities, variable-power moves, Transform and Illusion
are not completely simulated. Raw public events and explicitly labeled estimates
let Astra override code; fallback play remains the main risk under long queues.
