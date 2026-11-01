# Candidate 0015 — pre-stage the next ingredient during a full-pot cook

## Parent and targeted hypothesis

Parent: `candidates/0012/`. I retain its exact symbolic pot/recipe bookkeeping,
BFS interaction poses, just-in-time dish route, soft lane choice, asynchronous
Jev advisory, and deterministic per-tick motor.

The targeted hypothesis is that 0012's remaining low-score episodes are caused
by an empty-handed cook interval being treated as an unavoidable wait even when
a free counter can hold the next ingredient. This is a narrower change than a
new traffic lease: use the cooking window to stage a bounded number of
code-selected ingredients, then let the existing counter retrieval and pot
filling jobs consume them.

The evidence is concrete. In `0012/trace_seed0.jsonl`, after `park_cook` at
tick 66 the controller posts `stay` on ticks 67–84, and after the next
`park_cook` at tick 269 it posts `stay` through tick 287; the episode ends at
180 points with a cooking pot at cooking tick 4 and an onion left on a
counter. Seed 3 similarly stays from ticks 64–82 after the first start and
from 265–283 later, ending at 180 with a cooking pot at tick 5. Seed 6 has the
same pattern at ticks 100–109 and 138–156; it ends at 180 with a cooking pot
at tick 10 and an onion at `[1,0]`. The parent does already stage an item
before some starts, but its empty-hand scheduler only creates a new ingredient
run when a pot is `empty`/`partial`; while the only pot is `cooking`, it falls
through to `park_cook`/`wait`. Thus the available counter buffer is not used
for the next batch.

## Change

When at least one pot is cooking, no pot is currently buildable, the controlled
chef is empty-handed, and a free counter exists, the scheduler can commit to a
normal `fetch_ingredient` job **only in a serial kitchen**: one pot total, or
multiple pots whose complete walkable interaction-pose sets share a gate. This
keeps the change focused on the contention pattern seen in the ring and the
single-pot cramped room; distributed multi-pot layouts retain 0012's proven
continuous pipeline. After pickup, the unchanged parent logic puts that
ingredient on a free counter. It repeats only until two staged ingredients are
present (or fewer if the layout has fewer usable counter sites), and never fills
the last usable counter: the counter is a shared traffic buffer, not a private
queue. Ready-soup delivery, full-idle start, held-object legality, and all
interaction preconditions still outrank staging. Staging is therefore one
bounded productive fallback, not an indefinite ownership lease. Ingredient
selection is derived from `all_orders` and already staged names; it does not
hardcode onion or training layouts.

Confirmation would be a reduction in the long `park_cook`/`wait` runs and in
pot idle time on cramped seeds 0/3/6, with the next `fetch_counter`/`fill_pot`
sequence beginning immediately after delivery, while the clean asymmetric
16–18 delivery traces remain intact. It would be refuted if staged objects
consume the route's only useful counter, cause the partner to stop delivering,
produce wrong-recipe soups, or increase collision/stuck ticks enough to erase
the saved travel. The decisive trace signs are a staged ingredient that is
never consumed, a ready pot whose dish route is blocked by a stage, or a
score drop on the parent’s 360-point episodes.

## Tier-2 Jev interface and fallback

The Jev interface is inherited and unchanged: each asynchronous request carries
strict JSON with tick/layout, self and partner positions/held objects, pot
phases and derived cook remainder, the current macro job, lane, and compact
candidate labels. A batched `Choice` asks for `prepare|fetch_dish|wait`; a
`Noul` asks whether an urgent dish/soup-carrying partner should get a brief
yield. Their answers never create a raw direction or staging target. Closed
world Choice values require the parent’s `0.60` option probability and Noul
requires `0.72`; requests use the parent’s short `1.0` second timeout. Stale
epochs, missing probabilities, exceptions, and timeouts are ignored. Code
always revalidates the current state and falls back to the BFS job motor, with
advertised `stay` as the final legal action. The new
staging branch is code-owned and cannot be vetoed into an illegal interaction
by Jev; ready delivery and held-object rules override any timing hint.

Astra remains the parent’s one background layout-level advisory and cannot
alter routes, recipe choice, counter bound, or legality. The episode loop keeps
posting one legal action and waiting for each realtime tick while model calls
run asynchronously.
