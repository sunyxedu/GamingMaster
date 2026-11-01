# Candidate 0018 — persistent collection missions, temporary evasions

## Parent and diagnosis
Declared parent: **0013**. Keep its observation-derived dot graph, power-node
snapping and uninterrupted cardinal execution, but discard the refuted rule
that reserves become goals only after life loss. Also retain the best candidate
**0016**'s bounded ghost projection and independently detected resource epochs.
This is a rewrite of the hierarchy, not a reserve-count or probability tweak.

Concrete observations from the archive:
- **0013 seed 1**, lines 648–671: at frames 1644–1708, score 800, the Jev
  state remains at `[74,123]`, both choices are `danger`, and `escape_option`
  is `none` despite two visible powers. Lines 707 and 832 show subsequent
  life-loss observations with both powers still present. The terminal score is
  1230 with one power remaining. Those later locations are respawns, not proven
  collision sites; stationary states can also be death animation.
- **0016 seed 3**, line 2371: frame 6060 has score 1780 and only two dots left.
  The next ten-point gain is not until frame 7488 (line 2923), after a second
  life loss. Terminal frame 7601 leaves `[8,103]` as the final dot and score 1790.
  The trace records **212 route_lease events** and 114 accepted Jev route answers.
  At frames 6504, 6664, 6684, 6696, 6720, 6740, 6756, and 6828, the strategic
  target oscillates among the dot `[8,55]` and empty retreat waypoints
  `[16,55]`, `[16,79]`, and `[16,103]`. This supports goal churn as a failure
  mechanism, not a claim that every retreat was unnecessary.
- 0016's `_code_action` calls `_install_goal` on safer short retreat packets;
  `_install_goal` overwrites the collectible target. Comparing a short retreat
  against a longer collection approach also gives the retreat a shorter danger
  horizon. These are specific architectural weaknesses despite the large score gain.
- Its first Astra answer in this episode takes 72.686s and returns
  `RUBRIC=cleanupRUBRIC=cleanup RUBRIC=cleanup`; the strict parser rejects it.
  A repeated consistent assignment need not be treated as contradictory output.

## Hypothesis
A safety maneuver should change **how to reach a collectible**, not **which
collectible remains the objective**. Separating these lifetimes should reduce
empty-corridor goal churn, improve late-maze completion, and retain the best
controller's continuous safety response. Every temporary evasion has a computed
continuation back to the same mission; it is evaluated with that continuation,
not as an artificially short safe endpoint.

## New division of labour
- **Code:** public-observation graph and resource bookkeeping; durable target
  existence/consumption checks; risk-weighted route search; candidate corridor
  prefixes and rejoining paths; ghost arrival approximation; short contact
  guard; action every tick. Nearest-node association is no longer treated as
  collection: the physical approach continues until the collectible disappears.
  A bounded mission timeout prevents indefinite commitment without progress.
- **Jev:** select an actual collection mission (keep, dot run, nearby dot, or
  reachable power), and recognize evidence that an existing mission is stalled.
  It cannot select an empty retreat, post a direction, or overwrite temporary
  maneuvers. Mission release needs independent progress evidence, not merely a
  new local hazard label. A demonstrably safer power approach can replace a
  pressured mission before a death.
- **Astra:** choose the next Jev mission-selection rubric after evidence of
  life loss, resource renewal, or repeated unproductive evasions. It selects
  completion, mobility, or resource use, not a joystick and not executable code.
  Repeated identical explicit assignments are accepted; conflicting ones ignored.

## System-1 state, questions, bridge
Each request supplies `rules`, `policy`, `context`, and `options`. Options contain
semantic approach risk, short/medium/long travel, dot-yield bucket, collectible
type, whether it continues the mission, and previous unproductiveness. Context
contains cleanup/last-life flags, measured mission and score stalls, repeated
escape evidence, and uncertain recent power use. No raw RAM or path arithmetic
is delegated to Jev.

One request batches two reordered **Choice** questions (`mission_a/b`) and two
independent **Noul** questions (`release`, `audit`). The Choices must agree and
both selected-option probabilities must be **>=0.70**. A current mission can be
replaced for a stall only with **release >=0.80**, >=240 native frames without
mission progress, and >=180 frames without score progress. A newly safer reachable
power is a separate guarded exception. **audit >=0.80** plus measured score stall
requests an Astra review. Missing/malformed probabilities are not confidence.

Responses are bound to resource epoch, death count, mission serial, and interface
version, and expire after **90 native frames**. The target must still exist and
the approach is rebuilt against current ghost observations; the model cannot
increase the current risk class. A retained mission is not a cached action.

## Scheduling and failure policy
Continuous code control posts before any inference and waits for ticks. One Jev
request at a time, at least 96 native frames apart, **2.5s timeout** (archived
normal p50 ~0.19s, maximum ~1.16s). One Astra call at a time, at most three,
**110s timeout**, short answer, >=1800 native frames between starts, and only
with enough remaining wall budget. Model failures, disagreement, stale state,
unknown answers and arbitrary text leave code control active. All inference uses
ctx; no extra background loops, private clients or emulator access.

## Confirmation / refutation
Confirm only with error-free near-zero-stale evaluation, fewer strategic goal
changes per collected dot, more late-maze completion, and higher mean ALE score.
`maneuver` logs must retain the same mission serial through evasions, followed
by `mission_complete` due to object disappearance. `jev_mission_changed` and
`astra_mission_interface` distinguish useful bridge effects from mere call counts.
Resource epochs must renew the power records on observed dot repopulation.

Refute if persistent missions repeatedly time out, the safety layer still circles
without progress, or committed approaches lose lives sooner. The inference graph,
velocity projection and contact guard are approximations, not verified walls or
frightened-state knowledge. No seed-specific route or timing is encoded.

## Validation status
Both deliverables are written before refinement. Next: syntax/import restrictions,
stand-in-model contract tests, and bounded archived-observation replay. No long
self-evaluation or direct model backend calls are planned.
