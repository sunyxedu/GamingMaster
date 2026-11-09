# Candidate 0007 — certified route beam with bounded semantic intent

## Parents and hypothesis

Crossover parents: `candidates/0005/` (the current best) and `candidates/0004/` (the first successful code-side MPC rewrite). The harness also compares their traces because the archive records where each succeeds and fails.

The important mechanism in **0005** is not its model output. It explicitly revokes a FLAP before doing work after the next tick, posts a fresh action on nearly every frame, verifies the observed gravity order, and uses backward reachable sets as an action-safety gate. That combination survived all six training seeds to the 900-second deadline with score 716 on every seed, only 0–1 stale ticks, and at most 2–3 consecutive flap ticks. Jev profiles were accepted, but only 14 profile changes were reported in seed 0 and Astra never preceded an action; the high score therefore comes principally from the local pulse executor.

The useful mechanism in **0004** is its forward short-horizon beam: it evaluates competing sparse pulse schedules and exposes the following gap before it reaches the bird. Its weakness is that the search and model bridge were not sufficiently robust at route changes. It reached 215 on seed 0 but only 19 on seed 5; both failures ended while crossing a pipe with negative top clearance (`-16` and `-19`). Its Jev calls mostly timed out (`1172/1197`) and never informed actions, while `0005`'s one-frame-safe fallback remained robust.

**Hypothesis:** 0005 has already solved real-time execution and the immediate safety problem, but its preference is mostly current-gap/centre tracking. At a transition, it can select a locally safe pulse that leaves too much height error for the following gap. A forward route beam can improve this without sacrificing robustness if it is used only as a *preference* and is always filtered through 0005’s backward reachability certificate. Jev should judge the semantic choice between “follow the route schedule” and “protect the current corridor,” not calculate arithmetic or directly choose FLAP. Astra should select only a coarse posture that changes route-vs-current emphasis; neither model may override the certificate or block the clock.

This is a structural crossover rather than a margin tweak: two code-side controllers are composed with explicit authority ordering:

1. observation-validated backward certificate defines the legal one-frame action set;
2. forward route beam supplies a sparse route preference using current and following visible pipes;
3. Jev provides a fresh, duplicated/reversed intent judgment when available;
4. local centre/velocity recovery wins whenever the judgment is stale, uncertain, malformed, or unsafe;
5. Astra changes only posture at a pipe boundary or quiet buffer.

## What changed

* **Certified route planning.** The harness uses the public feature/convenience views only. It selects the first non-placeholder pipe not behind the bird, retains the following visible pipe, and simulates sparse action sequences with the observed gravity/flap order and pipe motion. It scores clearance and following-gap alignment, but never treats the forward beam as a proof of safety.
* **Safety authority retained.** A backward integer-position reachability calculation, adapted from 0005, computes actions with a collision-free continuation across every visible corridor and a short post-route coast. It tries the posture margin, then bounded 6/2/0-pixel relaxations. The model and route beam cannot manufacture a legal action.
* **Pulse execution retained and made explicit.** After every observed FLAP, the harness posts IDLE before polling Astra, Jev, or running a planner. It posts a fresh action every loop, checks a second observation before publishing work that may have crossed a tick, and uses an observation-based bounded recovery if the certificate is unavailable. This prevents the 0000/0003 failure mode where a slow decision held FLAP for 36 frames.
* **A real model-to-action bridge.** Jev receives a compact derived state containing qualitative vertical band/motion, phase, following-gap direction, route first action and margin class, and the certificate’s safe action list. Two reversed-order Choices independently select `route` or `protect`; they must agree and pass probability thresholds. `route` selects the already-computed beam first action, while `protect` selects a local centre/velocity preference. The chosen intent is still intersected with the certificate.
* **Slow supervision without live blocking.** Astra is spawned once at reset with a small opening summary and may return only `BALANCED`, `TRANSITION`, or `CAUTIOUS`. A changed posture is committed only at a target transition or after a quiet buffer. Garbage text, timeout, cancellation, and model exceptions leave the current posture unchanged. There is no Python background loop: spawned work is one managed call and is polled by the game loop.
* **Freshness and diagnostics.** Jev answers expire after 12 frames, a target identity change, a plan revision, or a 48-pixel horizontal shift. The target identity uses observed gap edges plus world-space reconstructed x, not a seed or private future state. Runtime transition verification logs physics mismatches. Controller events record route margin, certificate horizon, safe relaxations, accepted/rejected judgments, and posture.

## Evidence and test of the hypothesis

Evidence supporting the safety half already exists in the archive: 0005’s traces show 26,999 alive frames on every training seed, `physics_verified` after 12 transitions, sparse FLAP pulses, and final positive clearances. Evidence supporting the need for a route component is weaker but concrete: 0004’s worst failures occur at pipe boundaries with top clearance below zero, and 0005’s accepted profiles rarely change a posted action. The new harness specifically tests whether a following-gap route preference can add value while the 0005 certificate preserves the 716-level robustness.

**Confirmation:** training episodes remain error-free with approximately zero stale ticks and no held FLAP bursts; the minimum/mean score is at least the 0005 baseline, and route-controller events show that following-gap intent changes some certified actions before transitions. Held-out seeds should show fewer transition deaths than 0004 without losing 0005’s survival.

**Refutation or narrowing:**

* any death before a first passage with verified dynamics and a nonempty certificate would implicate target selection or the pulse preference;
* a reduction toward 0004’s 19–215 range, especially with negative top clearance at `CROSSING`, would show that route lookahead is being trusted too aggressively or the target identity is stale;
* route intents are accepted but rarely alter actions, while the score is unchanged, would show that model participation is decorative and the code preference is already sufficient;
* frequent physics mismatches would refute the assumed transition order and should disable the certificate rather than be hidden;
* stale ticks, multi-frame FLAP runs, or any escaped model exception would refute the real-time execution design and make the candidate ineligible.

The principal comparison is against 0005, not shaped reward: direct pipe-pair score and survival are primary; route intent counts and Jev/Astra-informed-action fractions are diagnostics.

## System-1 state and questions

Every Jev request carries strict JSON of this form (no raw feature array and no arithmetic task):

```json
{
  "game": "flappy_bird",
  "frame": 742,
  "plan": {
    "posture": "BALANCED",
    "revision": 0,
    "owner": "route_beam_plus_certificate"
  },
  "bird": {
    "motion": "FALLING_OR_LEVEL",
    "vertical_band": "INTERIOR",
    "top_margin": 23,
    "bottom_margin": 53
  },
  "route": {
    "phase": "CROSSING",
    "following_direction": "HIGHER",
    "transition_intent": "PREPARE_UP",
    "route_first_action": "IDLE",
    "route_margin_class": "WIDE"
  },
  "certificate": {
    "safe_actions": ["IDLE", "FLAP"],
    "safety_is_non_overrideable": true,
    "relaxation_level": 0
  },
  "score": 19
}
```

The `bird` labels are code-derived from the observed y, velocity, gap edges, and rectangle height. `route.following_direction` is code-derived from the next visible gap: `HIGHER`, `LOWER`, `LEVEL`, or `UNKNOWN`. `route_margin_class` is a label from the forward beam’s worst simulated clearance: `WIDE`, `TIGHT`, or `EDGE`. `certificate.safe_actions` is the only action legality information supplied to the model; Jev does not subtract, count, simulate, or infer hidden pipes.

The one request batches three independent typed questions:

1. `intent_forward`: Choice of `route` versus `protect`.
2. `intent_reverse`: the same Choice with criteria order reversed, reducing first-option bias.
3. `transition_check`: a Noul asking whether the supplied following-gap facts support preparing for the transition. It is diagnostic and cannot legalize an action.

The Choice instructions define `route` as using the forward beam’s first action only when its class is not `EDGE`, and `protect` as the conservative local centre/velocity rule. Both Choices must return the same option. Winning-option probability must be at least 0.82 when confidence is absent, or at least 0.62 when the model supplies confidence; the answer is rejected on malformed values, disagreement, low probability, changed plan revision, changed target, age over 12 frames, or horizontal movement over 48 pixels. The Noul threshold is 0.65 only for a diagnostic memory field and does not alter the action.

When a fresh consensus says `route`, the code uses the beam’s first action. Otherwise it uses `local_preference`, which prefers IDLE while rising or above the slightly-low anchor and asks for a sparse flap while descending below it. Whichever preference is selected, the final action must be in `safe_actions`. If IDLE is certified, it is the tie-breaker; FLAP is selected only when it is the sole certified action. If the certificate is empty because the observation is outside the integer assumptions, recovery uses the local rule with explicit ceiling IDLE and near-ground FLAP guards. Thus a late/garbage Jev result is harmless and never becomes a held action.

## Astra role, timing, and revision trigger

Astra is not the live actuator. It receives the initial public opening facts and is asked for one token:

* `BALANCED`: ordinary route-beam preference;
* `TRANSITION`: modestly reward preparation for the following visible gap;
* `CAUTIOUS`: shift toward the local patient centre rule.

Its call is spawned with a 90-second timeout while the controller immediately posts IDLE and continues. A valid changed posture is committed only when the observed target token changes or at least 18 frames have passed since the last route replan. It increments a revision, invalidating route/intent assumptions. No Astra output changes physics constants, safety margins outside the bounded local choices, or legal action sets. If Astra is never ready, the stable `BALANCED` posture is used.

## Why this is a distinct crossover

0004 supplied forward route awareness but let inference mostly time out and did not turn a model answer into an action. 0005 supplied the winning real-time certificate and pulse discipline but its Jev profile only weakly affected actions and its route focus was largely advisory. Candidate 0007 gives the following-gap beam a controlled path into the action choice while preserving 0005’s authority ordering and fallbacks. The decisive experiment is whether this extra route information improves transitions without allowing a stale or optimistic plan to cross the certificate boundary.
