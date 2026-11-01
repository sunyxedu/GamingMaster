# Candidate 0013 — quarantine the failed home-entry offset

Parent: `candidates/0010/` (the shape-local contact-learning controller, retaining the
route search, neutral rearming, live guard, durable Jev goals, and asynchronous Astra
review). Mode: refine.

## Hypothesis

The parent's remaining worst-case failure is not solved by carrier clearance or by
asking Jev to classify the failure after the fact.  A final `UP` can be geometrically
certified as a complete home route, yet the same bay/approach is immediately lethal;
the executor then retries that approach on every new frog.  A small, episode-local
quarantine of the *failed bay and frog-left offset* should preserve the bay's other
entry offsets and other homes while preventing repeated loss of lives to the same
contradiction.

This is based on the parent trace, not a seed-specific route:

* In `candidates/0010/trace_seed109.jsonl`, the first four deaths after the run had
  already reached nine homes are immediate home-entry failures at ticks **4270,
  4900, 5552, and 6192**.  Each is `row=10`, `action=UP`, target `HOME_0`, and the
  prior frog is at **x=12, 11, 11, 11**.  The corresponding route records at ticks
  4267, 4899, 5550, and 6190 all select the sole
  `HOME_0` route with `margin=0.0`; the exact trace records are, for example,
  `tick 4270 ... immediate_home_entry=true` and `tick 4270 ... before=[0,9,2]
  after=[1,9,2]`, followed by the same `HOME_0`/x=11 route at the next three
  life losses.  The parent did not stop replaying the failed approach until its
  Jev diagnosis arrived much later at tick 7917.
* The parent also demonstrates why a global safety pad is the wrong isolated test:
  successful home transitions occur at several edge-looking coordinates (for
  example seed 109 reaches home at x=13 and seed 127 has successful entries at
  x=43, 75, 107 and 140), while the repeated seed-109 failure is localized to
  `HOME_0` near x=11/12.  Rejecting every tight home or inflating all river
  geometry would throw away reachable homes rather than test the observed repeat.
* The parent's later median deaths (seed 127, ticks 6849, 7015, 7401 and 9467)
  are a separate bank/staging failure.  This candidate intentionally does not
  change that policy, so the evaluation can distinguish home-entry recovery from
  median or carrier effects.

## One targeted change

`home_entry_bans` is a cold-start, episode-local book keyed by `(level, bay slot)`.
When public observations show a death immediately after an `UP` from river row 10,
code identifies the planned `HOME_n` (or an unambiguous nearest bay) and records only
`[failed_x-1, failed_x+1]` as a forbidden frog-left entry interval.  The current
level is in the key so the same bay coordinates can be reconsidered after the
five-home progression.  The ban is applied in both the multi-frame search and the
one-hop live guard:

* a candidate endpoint in that bay/offset interval is not a complete route;
* the bay remains available at other x positions, and other open bays are untouched;
* no model answer can override the ban; an invalid or missing plan cannot crash the
  feedback path.

This is deliberately narrower than 0011's global `river_pad` increase and does not
alter `road_pad`, carrier widths, contact bands, the home window, pulse timing, or
Astra's policy vocabulary.  It acts on the first observed contradiction, rather than
waiting for the asynchronous diagnosis question.  The normal fallback remains the
newest code-verified route, a verified partial/one-hop move, then `NOOP`.

## What would confirm or refute it

Confirmation would be a sharp reduction in repeated immediate-home deaths after the
first such death, especially the four-tick-class sequence at `HOME_0` in seed 109,
with the controller selecting another home or a different x offset and preserving
more frogs.  A useful confirming trace will contain `home_entry_ban` followed by no
later route whose endpoint lies in that interval, while still showing successful
entries through unbanned offsets and no increase in model errors or stale ticks.

Refutation would be: the same bay/offset still kills after the ban; deaths occur at
unbanned offsets with no reduction in home-entry losses; or the narrow ban causes
large `no_route` intervals and fewer homes than the parent by preventing otherwise
safe bays.  Median deaths should remain approximately unchanged; if only median
failures change, that is not evidence for this hypothesis.  The primary comparison
is homes/deaths and score, not Jev call count alone.

## System-1 interface and bridge

The interface is inherited from 0010 because the experiment changes only the
home-entry failure response.  Each asynchronous Jev request receives strict JSON
containing:

* `phase` (`road`/`river`), frog zone, timer urgency, level/home/death epoch, and
  the current bounded Astra policy;
* code-generated, current-observation route candidates only, each labelled by
  endpoint (`MEDIAN_*` or `HOME_n`), shortest/detour, widest/narrower support,
  comfortable/tight clearance, and current-goal status;
* the fact that separated sprite copies leave water gaps, unknown home crocodile
  hazards are not declared safe, and the recent localized failure evidence;
* candidate routes already filtered by the code-owned home-entry quarantine.  Jev
  is not asked to rediscover or override that geometry.

One request batches independent typed judgments:

1. `goal`: a `Choice` over the current complete route labels plus `UNKNOWN`.
2. `reverse`: the identical `Choice` with its options reversed, guarding against
   option-order bias.
3. `review`: a `Noul` asking whether explicit prediction-failure evidence merits
   an Astra policy review.
4. `diagnosis` and `diagnosis_reverse`: `Choice`s over `contact`, `entry`, `bank`,
   and `unknown`, used only to update the parent's existing contact/window/bank
   policy.  The new home quarantine does not depend on Jev diagnosis.

A goal is adopted only when both Choices agree, the selected probability is at least
`0.65` in both distributions, and the answer is no older than 90 game ticks with
matching epoch, phase, and policy version.  It lasts at most 180 ticks and can only
bias a route that the latest code search still certifies.  Diagnosis requires
agreement and a minimum probability of `0.75` plus a matching code-supported failure
flag.  `review >= 0.8` may schedule Astra subject to its existing cooldown.
Probabilities that are absent/non-numeric, disagreement, `UNKNOWN`, stale answers,
timeouts, and arbitrary stand-in model output are rejected.

Jev never selects an immediate joystick action and never performs coordinate
arithmetic.  Code owns all legality, velocity, collision intervals, home occupancy,
entry-ban membership, freshness checks, and the answer-to-action bridge.  While a
request is outstanding, the motor posts each tick and uses the newest verified
route; a held direction is capped and explicitly released with `NOOP`.  If no route
survives the current state or quarantine, it falls back to a verified one-hop
survival move and finally `NOOP`.  Astra remains a non-blocking reviewer whose
malformed/late JSON leaves the executable policy unchanged.

## Expected diagnostic events

In addition to the parent's `contact_contradiction`, the harness logs
`home_entry_ban` inside that event with the level, slot, failed x and banned interval.
Evaluation can therefore compare the first ban with subsequent `route_action` and
`no_route` events, distinguish a bay/offset recovery from a generic reduction in
hops, and determine whether the repeated-home death sequence disappears without
sacrificing unbanned home entries.
