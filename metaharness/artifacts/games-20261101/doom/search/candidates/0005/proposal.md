# Candidate 0005 — Guarded semantic options, measured motor execution

## Parents and hypothesis

Crossover of **0000-tier2_jev_only** and **0002-tier2_idle**. The former supplies cheap contextual judgments and some combat/progress; the latter supplies a surprisingly important invariant: do not commit the player to unknown geometry or wake a fight merely because a model is still thinking. This is not an average of their action distributions. It is a separation of semantic intention from motor execution: Jev chooses among small contextual options, while a per-tic controller measures whether the chosen option is working and brakes or changes it.

**Hypothesis:** the primary baseline failure is an ungrounded, open-loop action interface, not insufficient model sophistication. An actor-only observation cannot establish a walkable route, a door, or an exit. Repeated raw `forward` decisions are not independent evidence that moving forward works. Odometry, finite collision probes, explicit useful-resource candidates, and aimed standoff combat should retain the idle parent's survivability while producing substantially more useful progress than the raw-action parent. Astra is a delayed policy/interface reviewer, not a prerequisite for the first safe action.

## Archive evidence

Only the supplied archive and public interface were inspected. No `lessons.md` exists at the archive root (nor in the initial depth-two file search).

- `0000-tier2_jev_only/trace_seed1001.jsonl`: at tic 76, the player has already reached `(640, 79.951, 8)` and lost health to 94; by tic 406, health is 1, the action is still `forward`, and y is still **79.951**. The episode ends at tic 439 by death, with **zero kills**. Tick actions comprise 373 `forward`, 52 `fire_forward`, and six initial noops. This is measured collision blindness, not merely a slow inference.
- `trace_seed1005.jsonl`: the same qualitative failure, not a route assumption: health falls from 100 to 7 while the forward coordinate remains **79.951**, then death around 9.4 seconds. No kills. The new controller uses no seed- or coordinate-specific rule.
- `trace_seed1002.jsonl`: 4,000 Jev calls and 22,710 forward ticks produce only 27 distinct integer xy positions, ending at `(224, 912.049)` with no kills/items. The first model answer chooses forward with probability **0.27**, confidence **0.24**. The old bridge treats this weak plurality as an unconditional motor commitment.
- `0002-tier2_idle/trace_seed1001.jsonl`: one xy position, 100 health, 50 selected-weapon ammo, no damage, and no progress through tic 31,504. Its 4.25 score is genuinely a survival floor, not useful exploration. We retain brief quiet braking, not permanent idling.
- Live labels distinguish corpses: `trace_seed1003.jsonl`, tic 223, has `DeadLostSoul/Gore`; `trace_seed1011.jsonl`, tic 391, includes `DeadShotgunGuy/Gore`. The controller requires the public `Monster` category and excludes `Dead*`, rather than repeatedly shooting corpse names.
- Parents have zero exits. The new candidate must improve routing, not simply optimize the minor health/time components. No reward or evaluator metadata controls the policy.

## Structural change

1. **Code owns the fast loop.** Observe, recompute geometry and safety, and post a legal action each new observed tic. Angle/bearing arithmetic, turn-sign/rate calibration, aim, weapon selection, motion success, damage deltas and target legality are not delegated to Jev.
2. **Observed-world memory.** Maintain visited pose cells, actual traversed adjacency, a breadcrumb trail, attempted headings and timed blocked headings. These are measurements of the player's own motion, not unseen map data. Landmarks originate only in rendered labels. A visible high ledge is not a reachable straight-line goal.
3. **Bounded motor commitments.** Ten tics without ten units of translation marks a failed movement heading. First appropriate forward collisions get one USE pulse, an 18-tic quiet wait, a 12-tic movement test, and (if still blocked) a seven-tic backoff. Failed resource goals are deferred. Exploration tries less-visited, unblocked headings instead of replaying forward indefinitely.
4. **Combat without rushing.** Rank currently visible threats using declared categories and a small named priority taxonomy; re-aim on current coordinates every tic. Shoot at standoff, strafe active threats, switch to owned usable conventional weapons, and refuse unverified rocket/BFG firing into unknown nearby geometry. No `fire_forward` rush is needed.
5. **Two independent model timescales.** At most one Jev call and one Astra call are in flight. Jev runs asynchronously; the game never waits for its answer. Up to three Astra reviews select a bounded question profile/resource priority after actual local history has accumulated. The 60–213 second archived System-2 latency cannot own the combat loop.

## Jev interface and answer-to-action bridge

Every request is a compact named object containing:

- `rules`: exit objective, actor-only observations, no walkability/door inference;
- `policy`: Astra-approved priority, profile, handedness;
- `health`, `ammo`: code-derived semantic buckets; `damage_recent`;
- up to three `threats`: current id, name/category, range/side/height buckets, active-motion/damage evidence, hitscanner flag;
- up to four `resources`: observed id, category, range/height and explicit benefit;
- `motion`: blocked, current goal type, stalled, known retreat availability;
- executable fallback description.

Batched questions:

| Question | Type | Control effect |
|---|---|---|
| `mode` | Choice: collect/engage/explore/recover/wait | Selects a local intention; can interrupt an exploration probe for a useful resource or request measured retreat. |
| `attack` | Noul | Optional engagement requires this >=0.76 **and** a confident engage intention. Immediate active threats remain a code-level safety responsibility. |
| `continue` | Noul | Classifies current-goal support; reserved for commitment invalidation as collision evidence accumulates. Collision braking itself never depends on a yes. |
| `target` | Choice over at most three visible IDs plus none | Selects the current threat to aim at; absent/nonvisible IDs are ignored. |
| `pickup` | Choice over at most four resource IDs plus none | Selects a currently legal resource destination; code rechecks utility, reachability evidence and deferral. |
| `obstruction` | Choice: use once/change heading/backtrack/unknown, only when blocked | Supplies contextual obstruction advice; the bounded executor remains the legal/safety authority. |

Choice acceptance requires the chosen option probability >=0.56, and rejects a supplied confidence below 0.15; missing confidence is not converted to certainty. Noul uses `.value` as probability, not `.confidence`. Mode option order is reversed after policy revision. Optional combat is corroborated across two independently answered judgments rather than trusting a low-probability raw action.

Requests carry observation time and policy version. Discard answers older than 1.5 seconds or from a superseded policy. Accepted intention expires after 1.25 seconds; actor IDs and all motor preconditions are checked on the latest observation. Empty/garbage results fall back to the code executor. A 3-second call timeout is deliberately above the measured 0.15–0.30 second ordinary latency and 1.88-second tail; late answers may still be discarded without having left a movement action unbounded. Normal Jev admission is at most every 0.32 seconds in combat and 0.75 seconds outside it, further paced by remaining calls/wall time.

## Astra interface revision

First review is eligible after 20 seconds; later reviews are event-gated on progress stalls or damage and separated by at least 100 seconds, with at most three calls. Give it the compact state, twelve recent measured snapshots, and the relevant exact local Jev contract. It returns only a bounded JSON policy: `priority` in resource_first/exit_first/recover; `profile` in ordinary/collision/survival; handedness left/right. Those values choose the emphasis of the next Jev questions. Free text, coordinates, invented routes, generated code and arbitrary thresholds are not executed. Invalid JSON leaves the existing policy unchanged. A revision invalidates pending semantic commitments; it never changes the per-tic safety layer.

The testable revision hypothesis is that sustained collision evidence needs **a different local question**, not another raw forward-action classification; falling health instead needs survival/escape emphasis. The slow response is a persistent policy update, not a stale motor action.

## Late-decision fallback and robustness

- Safe initial noop while the first judgment is pending, bounded to 0.6 seconds.
- Continue a measured, finite option when valid; collision and imminent damage override it immediately.
- Quiet holds are useful during a door-opening test and genuine safe uncertainty. They do not freeze the game, reset it, or terminate it to preserve a score.
- No model-generated Python, no map assets, no private API, no direct client, no seed routing, no cross-episode cache.
- Catch ordinary timeouts/backend errors, preserve legal defaults, and let budget/episode/model-unavailability signals propagate according to contract. Background tasks contain only the bounded inference call, not immortal worker loops.

## Confirmation / refutation

**Confirm:** fewer multi-second zero-displacement forward streaks; longer survival on the exposed starts; useful weapon acquisition instead of passing visible weapons; kills from aligned fire rather than accidental rush-fire; broader measured cell coverage; continued control through slow/model-invalid replies. Ultimately a genuine exit is the main success criterion. Logs distinguish Jev intent changes, stale responses, bounded USE probes, movement after a probe, and Astra policy revisions.

**Refute:** short survival despite braking (combat/wake policy wrong), long stationary but safe episodes (exploration policy inadequate), repeated resource deferrals without new coverage (odometry-only navigation insufficient), or stable code behavior that never changes in response to Jev/Astra (interface not contributing). A lower stale-tick fraction alone does not confirm the hypothesis. These are architectural tests, not claims of evaluated performance.

## Validation

Initial implementation and proposal are written before experiments. Local stand-in tests and syntax checks will exercise garbage outputs, asynchronous completion, blocked movement, terminal conditions, and the exact context signatures. No live game experiment or direct model-backend call is planned; the evaluator owns performance measurement.
