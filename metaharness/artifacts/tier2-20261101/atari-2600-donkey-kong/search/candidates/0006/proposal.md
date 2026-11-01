# Candidate 0006 — feedback route identification

## Parent and hypothesis

Parent: `candidates/0001-tier2_astra_only/` (with the later route/async traces used as evidence).

The best prior candidate does not fail because it is late. Candidate 0004 keeps nearly every tick fresh, yet its trace posts `up` from tick 2 through 238 while Mario remains on the bottom band, then posts `noop` for the rest of the episode after the death transition. The concrete route error is that its `nearest_ladder` heuristic chooses the visually nearest upper ladder around x=50, rather than first crossing to the bottom ladder around x=108. Candidate 0003 corrected the initial horizontal target, but its trace reaches the right edge and then alternates `up`, `down`, and diagonal inputs without an observed height transition. It neither verifies that the ladder accepted UP nor tries a bounded alternate.

**Hypothesis:** the missing capability is closed-loop route identification, not another larger model prompt or a faster Jev cadence. A controller must treat a ladder choice as a hypothesis, probe it for a bounded number of quanta, and advance only after the observation shows upward motion. It must also own the death/transition restart. If the first screen can be completed by this feedback machine, positive reward should appear before strategic model quality matters.

## Structural change

This harness changes the division of labour:

* **Code is the real-time actor and route learner.** It reads player/hazard geometry, raw jump state, rivet bytes, lives and stage; selects the public alternating ladder candidates; aligns by sprite centers; probes `UP`; detects an actual decrease in player height; and switches to an alternate candidate after a bounded failed probe. It does not select the nearest ladder and does not let a stale model action hold the game.
* **Code has a hard lifecycle state machine.** It posts one FIRE to leave idle, releases movement during transition, and posts one restart FIRE after the observed death-release interval when a spare life remains. Active-play FIRE is reserved for a directional jump guard or a known rivet gap.
* **Jev is now an interrupt recognizer, not a route driver.** It sees a compact semantic state and can confirm a closing-hazard jump. It cannot suppress a code-level imminent jump and cannot invent a route or numerical movement. A late, malformed, low-value, or stale answer leaves the feedback controller in charge.
* **Astra is an asynchronous policy reviewer.** It receives current facts only on initial entry and major phase changes and returns a short risk/ramp/rivet preference. The output is advisory; route correctness is tested by observations rather than by waiting for a 50–177 second action answer. The actor continues every quantum while Astra thinks.

The same phase machine is reused on the rivet screen: raw row bytes drive a bottom-to-top serpentine target, the expected side is inferred from a row-byte change, cleared locations are treated as gaps, and climbing resumes only after the row is complete.

## Evidence addressed

* `0001` made only 18 Astra calls and had 100% stale ticks; its long blocking answer produced no usable action sequence.
* `0000` made 1,907 Jev calls but selected `up` while a hazard crossed the bottom approach; unconstrained action selection did not establish a route.
* `0003` did issue the initial rightward motion, but its post-arrival trace oscillated between `up` and `down` at the bottom ladder and lost the life without a height confirmation.
* `0004` demonstrated that asynchronous execution fixes cadence (`0.0025%` stale ticks) but not strategy: the action runs were `fire`, `up`, `leftfire`, then `noop` through the 900-second deadline. Its transition/life events show the harness never sent the required restart FIRE.

The new controller therefore makes the first measurable milestones explicit: first active state, first horizontal arrival, first observed upward y change, restart after life loss, and row-byte progress.

## System-1 schema and questions

Every outstanding Jev request receives strict JSON containing only derived facts:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail port",
  "tick": 123,
  "stage": "ramp|rivet",
  "player": {
    "side": "left|middle|right",
    "band": "low|middle|high",
    "motion": "up|down|level",
    "jumping": false
  },
  "route": {
    "fallback": "left|right|up|...",
    "ramp_level": 0,
    "ramp_choice": 0,
    "up_probe_count": 3,
    "rivet_level": 0,
    "rivet_rows": [0, 6, 12, 18]
  },
  "hazard": {
    "present": true,
    "urgency": "none|watch|soon|immediate",
    "side": "left|right|none",
    "closing": true,
    "jump_direction": "left|right|none",
    "count": 1
  },
  "progress": {"lives": 2, "screens": 0, "rivets_removed": 0, "bonus": 4900},
  "last_posted_action": "right"
}
```

The batched questions are:

1. `tactic`: a `Choice` over `continue_route`, `jump_left`, `jump_right`, and `hold`. The descriptions explicitly say that `route.fallback` owns exact movement. When code has `soon`/`immediate` hazard facts, a matching jump is the only model result that can be used; on a clear route the answer is observational only.
2. `danger`: a `Noul` asking whether the supplied hazard is a closing object on the current walkway. This is a redundant recognition signal for traces and future revisions; the numeric guard remains authoritative.

Requests are started asynchronously at most once every three ticks, with one outstanding request and a 0.34-second timeout (above the measured Jev p90 while never blocking the actor). Answers are tagged with stage and capture tick and discarded after three ticks or any stage/life change. Unknown values, missing probabilities, timeout, or ordinary backend failure use the deterministic action. The hard guard starts a directional jump for an `immediate`/`soon` closing hazard regardless of Jev; Jev cannot turn a lethal state into a walk.

## Astra prompt and revision trigger

Astra is asked for fewer than 100 words of JSON with `risk`, `ramp`, `rivet`, and `replan`. It is started asynchronously initially and after a stage, life, or screen change, up to three calls. The prompt states that code is already probing ladders and asks Astra to reason about macro route/recovery and gap policy rather than emit joystick timings. Garbage or non-JSON output is ignored. This makes Astra useful for interpreting an unusual route without allowing its latency to stop the world.

A future revision should inspect traces at these boundaries, rather than average model confidence:

* If the first post-start `UP` probe yields no y change, adjust the ladder candidate/center interpretation or candidate order; this would refute the geometry prior while confirming the need for empirical probing.
* If y changes upward and then a later band is reached, but the controller dies on a hazard, revise only the derived hazard row/closing guard and jump hold.
* If row bytes change but the controller dies crossing the second rivet, revise side inference/gap crossing; do not replace the route with free-form Jev actions.
* If stage reaches `transition` with a spare life and no subsequent `fire`, the lifecycle state machine is wrong; the intended evidence is a new active state after one release interval.

Positive reward, any observed upward band transition, a successful restart, or a changed rivet byte would support the hypothesis. A run that remains stationary despite bounded ladder probes would refute the coordinate/topology prior and identify the next calibration target. The harness remains executable with arbitrary stand-in model text: all model output is optional and parsed defensively.
