# Candidate 0003 proposal: delay-aware Jev safety shield

## Parent and targeted change

Parent: `candidates/0000-tier2_jev_only/`.

I kept its architecture: one synchronous Jev `Choice` request, followed by an
immediate `ctx.act`, with no Astra call.  The one targeted change is a
mechanical safety/freshness shield around that choice.  The harness now:

1. simulates the exact Snake transition from the observation, including tail
   vacating on non-food moves and retaining the tail on food moves;
2. removes `NOOP` and immediate reversals from Jev's options;
3. offers only directions that remain collision-free for three repeated ticks
   when possible; and
4. re-observes after Jev returns and rejects an answer that is no longer safe,
   using a safe direction as the fallback.

The three-tick horizon is deliberately short.  The parent measured roughly
0.13--0.19 seconds per Jev call, while the game advances every 0.10 seconds,
so a posted action commonly persists for one or two ticks before the next
answer arrives.  This is a safety intervention, not a full food planner.

## Failure evidence and hypothesis

The parent did not fail because it could not call Jev: all 37 System-1 calls
succeeded, but its raw five-way question let Jev select a non-action while the
snake continued toward a wall.  In `trace_seed4.jsonl`, the initial state has
`direction: "LEFT"` and `safe_directions_now: ["UP", "DOWN", "LEFT"]`; Jev
answered `"NOOP"` with probability 0.40.  The same happened at request ticks
1, 3, and 4 (NOOP probabilities 0.38, 0.42, and 0.41), and the episode ended
at tick 6 with `attempted_head: [6, -1]` and `terminal_reason: "wall_collision"`.
`trace_seed3.jsonl` shows the same pattern: the tick-0 answer was NOOP (0.32)
and the tick-3 answer was NOOP (0.28), followed by a tick-6 wall collision.

Even when Jev did steer, the one-step mask was not enough for a delayed
controller.  In `trace_seed1.jsonl`, at request tick 7 the head was
`[4, 1]`, the direction was LEFT, and the one-step safe set was
`["UP", "DOWN", "LEFT"]`; Jev chose LEFT (probability 0.45).  The next
uninterrupted movement reached column zero and the episode ended by wall
collision.  The leaderboard confirms the common symptom: zero score in every
training episode, 42% stale ticks, and a mean of only 8.83 ticks survived.

**Hypothesis:** most of the parent's zero score is caused by an action-interface
failure, not by difficult long-horizon Snake planning.  Specifically, Jev's
semantic preference for `NOOP`/straight travel and the one-tick legality test
are unsafe when an answer is delayed.  If the same Jev judgments are restricted
to short-horizon legal steering actions and stale answers are revalidated, the
snake should turn before the wall and live long enough for Jev to react to
food.  This isolates the effect of the shield without adding Astra or a new
planner.

There was no `lessons.md` file in the supplied archive.  The evidence above is
from the parent's traces and is the basis for this change.

## System-1 bridge

The state sent to Jev is a strict JSON object:

```text
{
  "observation": <complete game observation>,
  "candidate_directions": ["UP", ...],
  "one_step_safe_directions": ["UP", ...],
  "candidate_safety_mode": "safe for 3 repeated ticks" | "safe for the next tick",
  "horizon_ticks": 3
}
```

The adaptor's complete observation remains available so Jev can see food,
body order, heading, and score.  Code computes the exact candidate mask; Jev
is not asked to count cells or prove a route.

There is one `Choice` question, `act`.  Its criteria contain only the current
`candidate_directions`, with each direction described as a code-verified
three-tick or one-tick-safe option.  `NOOP` is never a normal criterion.  The
question asks Jev to prefer visible food progress among these already-checked
options.  A response is usable only when its `.value` is an exact current
candidate.

The safety thresholds are:

- first filter out the direction opposite the observed heading;
- require collision-free simulation for three repeated requests, treating the
  tail as vacated only on a non-food move;
- if no direction passes three ticks, fall back to the exact one-step-safe set;
- after the model call, recompute both sets from a fresh observation and never
  post a stale or unsafe answer.

A failed or timed-out Jev call does not escape as an ordinary harness error. It
causes the same fresh-observation fallback, except for the contract's normal
`BudgetExceeded`/`ModelUnavailable` termination exceptions, which are
propagated. If no safe move exists, the harness waits for the next tick rather
than asking Jev to invent an illegal action. If an answer is malformed or
arbitrary (including pre-flight stand-in output), it is treated as rejected and
the fallback chooses the current heading when safe, otherwise the first
verified safe direction.

## Confirmation or refutation

The hypothesis is supported if candidate 0003 has substantially more than
zero food, survives past the parent's roughly 6--12 ticks, and shows no early
wall collisions caused by NOOP/straight persistence.  A useful diagnostic is
that Jev choices in the trace are among the code-provided candidates and that
turns occur before a three-tick wall deadline.

It is refuted if the score remains zero despite long survival and the traces
show the shield selecting legal turns, or if the score/length improves only
when the shield overrides Jev and the Jev choices themselves systematically
ignore food.  It would also be refuted if three-tick filtering causes too much
conservative looping and lowers food count relative to a one-step shield.  That
would motivate a later, separate food-route or Hamiltonian-cycle proposal;
this candidate intentionally does not change that planning layer.
