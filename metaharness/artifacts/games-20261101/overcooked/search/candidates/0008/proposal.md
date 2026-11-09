# Candidate 0008 — partner transaction leases

## Parent and diagnosis

Parent: `candidates/0005/`.  The parent is the strongest evaluated harness
(mean 195.56), and I retained its useful symbolic geometry, but its worst
training episode is diagnostic rather than random.  On seed 3 it delivers only
40 points.  Around ticks 56–69 the trace shows a partial pot with two onions,
the controlled chef fetching the third onion, and the scripted partner also
ending up with an onion.  The controlled chef then starts the full pot while
the partner is nearby with the redundant onion.  During the next cycle the
partner occupies the only walkable pose below the pot with that onion.  The
controlled chef has a dish, but the parent's `_interaction_path` treats that
pose as blocked and its local escape alternates north/south from tick 127
through 400.  The final observation is a ready soup, a dish in the controlled
chef's hand, and the partner still holding an onion.  This is not a failure of
BFS or recipe arithmetic: it is a race caused by both chefs independently
claiming the same missing ingredient, followed by an irreversible one-cell
traffic deadlock.

## Hypothesis

The partner is a scripted stateful worker, not an independent random obstacle.
A visible partial pot plus a partner who has just interacted at its approach (or
who is carrying the matching ingredient on a reachable route) is evidence of a
short-lived **partner-owned fill transaction**.  If the controlled chef stops
fetching a duplicate until that transaction commits, the partner should not
arrive at the next cooking phase with a useless ingredient.  A phase-aware
lease should therefore remove the one-cell deadlock while preserving the
parent's production loop.  Leases are finite (18 ticks) so a misread partner
cannot make the controlled chef idle forever; after expiry, the controller
takes the job itself.

This is a structural change rather than a Jev threshold tweak: the scheduler
now has an explicit transaction/ownership layer between symbolic bookkeeping
and the primitive motor.  It also holds a global parking route instead of the
parent's local two-step escape, and lets a dish holder reserve a cooking/ready
pot without inventing an interaction.

## What changed

* `partner_claim_base` detects a matching ingredient with a reachable partner
  route, and an empty-handed partner standing at a partial pot approach.  A
  claim persists while the partner travels to the dispenser or pot, is keyed by
  pot phase and ingredient signature, and expires after 18 unchanged
  ticks, and is recomputed from observations rather than trusted forever.
* An empty controlled chef yields to an active claim instead of fetching a
  duplicate.  It starts full idle pots and protects ready soup before beginning
  another run.  Held objects still follow exact legal destinations; surplus
  ingredients use a free counter when one exists.
* All actions remain code-generated: BFS reaches an adjacent interaction pose,
  the final action first turns to the target, and `interact` is emitted only
  after the pose/facing preconditions are true.  A global parking search replaces
  local north/south oscillation after repeated blockage.
* Astra is one nonblocking strategic review.  Its output can only select the
  same safe phase already selected by code; stale or malformed JSON is ignored.
  It cannot emit directions.  This tests the lease hypothesis without making
  the realtime clock wait for the slow model.

Confirmation would be a higher score on the parent's cramped-room seed 3,
with no extra stuck interval after a ready soup, and fewer collision/stuck ticks
without reducing the parent's 220–260 point asymmetric/coordination episodes.
Trace evidence should show the controlled chef waiting while the partner fills a
partial pot, then starting/staging after the pot signature changes.  Refutation
would be repeated 18-tick lease expiry with no partner progress, or a new class
of failures where partner leases suppress necessary ingredient work on the ring
or asymmetric layouts.  In that case the next revision should learn partner
intent from a longer action/position history rather than using a local lease.

## Tier-2 Jev interface

The Jev state is a strict JSON object containing:

* `tick`, `layout`, the controlled and partner positions/facings/held names;
* one derived entry per pot: position, `empty|partial|full_idle|cooking|ready`,
  ingredient names, missing recipe ingredients, and `partner_lease`;
* active orders, the last few transition events, and at most eight code-built
  candidates with exact IDs, kinds, and descriptions.

One batched request asks two independent typed questions:

1. `task`: a `Choice` over those current candidate IDs.  Jev may break ties
   within the deterministic phase (for example, two dispenser routes), but the
   bridge accepts it only if its kind matches the code-selected safe task.  It
   can never turn a `yield_partner`/delivery/start phase into speculative work,
   and an unknown or low-probability ID is discarded.
2. `lease`: a `Noul` asking whether the derived partner lease should be
   respected.  Code already owns the lease legality; a high-confidence true
   only records a conservative yield hint when a lease is actually present.

Requests are made only on phase/candidate changes, with a six-tick minimum for
urgent ready/start/lease states and an 18-tick heartbeat otherwise.  The
three-second timeout and broad failure handling treat timeout, malformed
answers, budget exhaustion, and stand-in garbage as normal fallback events.
A late answer is used for no more than ten ticks and is revalidated against the
current candidate IDs and safe task kind.

When Jev or Astra is unavailable, the deterministic fallback is: deliver held
soup; pick up ready soup; start full idle pots; respect a live partner lease;
fill a compatible pot/fetch its missing ingredient; stage a dish during cooking;
otherwise wait.  Every chosen primitive is regenerated from the latest
observation, so a stale model answer cannot produce a stale interaction.
