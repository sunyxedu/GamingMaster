# Candidate 0025 — carrier guard during animation waits

## Parent and targeted hypothesis

**Parent:** `candidates/0023/` (the 0017 route-portfolio architecture plus its long-horizon carrier-wait veto).

The parent is strong at throughput, but its carrier veto can be skipped on the exact frames where it is needed. 0023 gates the lookahead on the decoder's `frog.moving_with_platform` bit. That bit is a useful fact, but it is a single-frame named status; a carrier handoff or update phase can make it false even while the observation's frog x overlaps a decoded safe river carrier. In that case the controller accepts repeated `NOOP` waits without running the parent's edge/actor forecast.

The parent traces give a concrete, testable version of this hypothesis:

* In `candidates/0023/trace_seed127.jsonl`, tick **19937** logs `carrier_wait_lookahead_unresolved` at river row 8, frog x **142**, projected x **151**, projected margin **-100**, actor `[8, 118]`, and no replacement. The frog then continues on `NOOP` and the gate failure at tick **19954** is row 8 x **150 -> 151**, with a safe-turtle contact and actor `[8, 134]`. The same unresolved-then-loss pattern recurs at ticks **21216 -> 21234**.
* In `trace_seed109`, row 8 reaches x **158 -> 159** at tick **19365** after `NOOP`, with carrier contacts at offsets `28` and `-4` and an auxiliary actor at x **142**; the preceding plan at tick 19335 had selected `RIGHT`, but no long-wait veto is logged during the subsequent wait.
* The parent also loses while apparently supported on rows without an auxiliary actor: `trace_seed107` records row 9 x **109** at tick **19421** after `NOOP` (contacting a log, plan margin `0.25`) and row 10 x **109** at tick **21405** after `NOOP` (turtle contact, plan margin `9.875`). These show that the named bit/short margin is not a sufficient reason to exempt a wait; they do not justify blindly forcing a hop.

The isolated hypothesis is therefore: **during a non-steerable animation wait (`frog.state != 128`), the normal control loop is not searching routes or applying the carrier lookahead at all. If the current observation independently shows the frog overlapping a decoded safe river carrier, a bounded, support-checked invocation of the existing lookahead can recognize an impending carrier-edge/actor danger and post a checked hop before the next normal route-planning frame.** It must not substitute a contact-only inference during ordinary active control; that protects against treating the parent’s uncertain contact geometry as a new general support certificate.

## What changed

The only behavioral change is a small extension of the parent’s existing lookahead bridge. Its calculation is factored into `carrier_wait_guard()` and is still invoked unchanged from `pick()` during normal route control. Additionally, while `run_episode()` is in the parent’s non-active (`state != 128`) wait path, a bounded cadence (at most once every three ticks) invokes that same guard on `NOOP`, but permits contact-based support only in this animation path and only when the overlap maps to a non-hazardous decoded `log` or `turtle`. It can replace the wait only with an immediate non-NOOP action that passes the parent’s current and 18-frame support checks. Otherwise, it preserves `NOOP`. The posted hop is tracked with the existing press/neutral release machinery, and the existing no-steerable action fallback remains in place when there is no certified move.

This diagnoses a control-loop coverage gap, not a route-search defect: `trace_seed109`'s row-8 failure at tick 19365 follows a RIGHT plan at tick 19335 and 30 ticks of NOOP, with no further lookahead events. The same trace logs unresolved lookahead at ticks 45172 and 45175 (frog x135/137), then its next recorded observation at tick 45191 establishes the relevant non-active state: state 8, row 8, x145, carry bit false, a decoded turtle overlap, and NOOP, with no further lookahead. The extension makes the existing guard reachable in that observed animation path without claiming the sparse trace proves that state 8 caused the earlier loss.


## Confirmation and refutation

I will treat the hypothesis as supported if evaluation traces show animation-path guard events at observed non-active safe-carrier waits near the edge (especially the x142–159 class), fewer subsequent carrier `gate_failure`s and deaths/reserve loss, and no material reduction in homes or increase in unresolved/no-route stalls. A veto count alone is not a save; the useful evidence is a later crossing outcome after the intervention.

It is refuted if the animation-path guard does not activate at supported non-active NOOP frames, if it repeatedly selects a certified action that is followed by a loss, or if the added checks materially reduce route throughput while carrier failures remain unchanged. Per episode, compare score, frogs home, deaths, reserve frogs, `carrier_wait_lookahead_veto`, unresolved waits, and the carrier-failure witnesses against 0023.

## Runtime division of labor

The route executor remains code-owned because it has exact legality, interval arithmetic, neutral rearming, and current object tracks. Astra is an asynchronous protocol reviewer, not a frame-action source. Jev supplies occasional typed supervisory judgments; garbage, timeout, stale, or low-confidence answers cannot disable local exclusions or become joystick actions.

### System-1 state schema

Every Jev request receives the compact JSON returned by `model_state()`:

* `game`: the objective and life-preservation priority;
* `phase`: `road_launch`, `bank_transit`, `river`, or `dock`;
* `policy`: Astra/code-compiled `itinerary`, `launch`, and `dock` enums;
* `homes`: ordered open-bay records. Each record contains `bay`, `upstream_frontier`, `nearest_now`, `current_commitment`, `failed_entry_this_level`, `matched_home_alert`, and `complete_route_now`;
* `facts`: `higher_level_bank_threat`, `complete_launch_exists`, `auxiliary_on_playable_row`, `auxiliary_on_frog_row`, `auxiliary_spawn_reserved`, `target_has_matched_alert`, `urgent_timer`, and the most recent `entry_failed`, `bank_failed`, `repeat_entry`, and `carrier_failed` flags;
* `rules`: short explicit meanings: choose a whole-crossing bay, treat the bank as transit, do not infer safety from unknown crocodile status or a failed final hop, and let code recheck geometry.

The state intentionally omits raw RAM and irrelevant object arithmetic. Code computes coordinates, intervals, scores, epochs, and route availability before sending the JSON. The new animation-path invocation is code-only and is not delegated to Jev.

### Questions, thresholds, and freshness

One batched request contains five independent typed questions:

1. `itinerary`: `Choice` over each open bay plus `UNKNOWN`, choosing a whole-crossing commitment;
2. `itinerary_check`: the same Choice with option order reversed, to reduce first-option bias;
3. `gate`: `Choice` over `TRANSIT`, `STAGE`, `RECOVER`, `SCREEN`, and `UNKNOWN`;
4. `gate_check`: the reversed gate Choice;
5. `review`: `Noul`, asking whether explicit repeated-entry or bank evidence merits an Astra protocol review.

A pair is accepted only when the two Choices agree and the smaller winning-option probability is at least **0.70**. Missing or malformed probabilities reject the pair. Advice is accepted only if its epoch matches the current `(deaths, frogs_home, level)`, its policy version is current, and it is no more than **90 ticks** old. A route goal expires after 180 ticks and is revalidated against current open homes and current `Field` routes. A single request is outstanding; requests use a 3-second timeout and exceptions are converted to no advice (apart from mandatory context/budget termination exceptions).

The `review` Noul can request asynchronous Astra, but it does not directly alter the motor. Astra's allowed output is parsed as the three policy enums (`itinerary`, `launch`, `dock`); anything else is rejected. A valid policy increments the version and causes a fresh Jev request.

### Action bridge and late-answer fallback

At every frame code searches time-expanded routes with body-aware sprite-copy collision intervals, checks the selected action again on the latest observation, and emits only legal actions. Standard one-hop directions are pulse-held only until movement or a short release timeout, then followed by centered `NOOP`; this preserves edge-triggered cartridge semantics. `NOOP` is the normal executable fallback while a Jev/Astra request is pending.

Jev never selects an action. Its accepted bay/protocol advice only changes route ordering before code's legality checks. If advice is late, disagreeing, low probability, malformed, or from an old epoch, the current code policy remains. If the selected route fails the current guard, code tries other searched routes and then a deterministic legal ordering; unsupported river jumps are never forced. The new animation-only condition likewise never forces an exit: if the parent lookahead has no supported replacement, it preserves the parent’s unresolved `NOOP` fallback.

## Expected diagnostic distinction

The candidate should primarily add **animation-interval** guard events, not alter the route search or Jev protocol: in particular, the tick-45191 class (state 8, river row 8, x145, carry bit false, current NOOP and a decoded non-hazardous turtle contact) should become eligible for the same lookahead. A successful result is fewer subsequent carrier losses, not merely more events or more conservative waiting.
