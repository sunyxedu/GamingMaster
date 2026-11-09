# Candidate 0018 proposal

## Parents and hypothesis

Parents: `candidates/0014/` and the best candidate, `candidates/0017/`.

The parent controller solved the real-time survival problem, but its strategic
interface was still mostly a safe-placement interface.  `0014`'s code chose
among `safe`, `clear`, `future`, and `recovery`; `0017` added a bounded queue
beam.  Neither explicitly represented the highest-value scoring event as a
control objective.  This matters because every episode already reaches the
200-piece cap, while the remaining score is dominated by line-clear points and
there is very little back-to-back bonus.

The concrete evidence is the `0017` worst run, seed 4: its final public
observation reports `77` lines, `back_to_back: false`, one hole, and a score of
`57630`, with `46300` line-clear points and `0` back-to-back points.  Its lock
trace contains isolated T-spin events (for example a full T-spin at piece 52
and piece 68), but the chain is not sustained.  In contrast, seed 3's trace
contains full/mini T-spins and ends with a `2000` back-to-back bonus and
`70450` score.  The lesson is not that more generic line weight is needed:
`0015` raised the line coefficient and still produced no B2B bonus.  The
untested structural change is to make difficult-clear opportunities explicit
and phase-controlled, while retaining the proven safety envelope.

**Hypothesis:** a controller that (1) computes a rotation-qualified T-spin /
Tetris event value in code, (2) asks Jev whether to pursue that event or keep a
smooth/clear/recovery track, and (3) lets Astra set a persistent `ATTACK`,
`BALANCED`, or `RECOVER` phase will create more safe difficult-clear chains
than a generic safe/clear role chooser.  Code remains authoritative for all
arithmetic, exact geometry, SRS reachability, and execution; models decide
whether a verified alternative is strategically appropriate.

## What changed

* The foreground is a cheap depth-0 exact placement enumeration.  A background
  finite search adds two queue plies and cannot delay a tick.  A fresh `noop`
  is posted at every new-piece boundary before foreground enumeration, then the
  observation and piece identity are revalidated.  This preserves the
  parent's guardian/clock guarantee rather than making a model or search the
  realtime driver.
* Each candidate carries projected lines, holes, hole depth, height, surface,
  corner occupancy around a T pivot, and whether its route ends with a rotation.
  A full T-spin or mini is valued only when the target differs in orientation
  and the verified BFS route's final edge is a rotation.  Tetris and
  line-clearing full T-spins receive difficult-clear and current B2B value in
  the code score; no-line spins are setup signals, not score hallucinations.
  The engine's lock event remains authoritative.
* Exact candidates are grouped into four semantic tracks: `attack` (a safe
  difficult clear or full-spin setup), `clear` (a safe ordinary line clear),
  `smooth` (low/flexible surface), and `recover` (holes/top-out reduction).
  The default track is code-selected, with an attack candidate admitted only
  inside a safety envelope: no new holes, bounded hole depth, and no more than
  five additional maximum-height cells.
* Jev is no longer asked to pick a generic placement role whose description
  does not expose the scoring objective.  It chooses between the four tracks;
  code maps the chosen track to the exact already-simulated placement.  Two
  independent Choices use reverse option order and must agree.  Jev never
  emits a primitive action.
* Astra is a slow phase planner rather than a one-shot policy-weight reviewer.
  It receives compact public buckets and queue/chain facts and returns one
  leading token: `ATTACK`, `BALANCED`, or `RECOVER`.  The token affects future
  code ranking and is never used as an action.  At most two calls are started,
  asynchronously, with the second only after a later danger transition.
* The verified executor rechecks active piece, hold permission, current pose,
  SRS kicks, route progress, and lock urgency before every action.  A spin
  target is abandoned and hard-dropped if its final-rotation route disappears;
  this is safer than late speculative rotation.

## System-1 state and questions

Every Jev request carries strict JSON derived from the current observation:

```text
{
  ruleset, phase, piece, hold, hold_available, queue,
  stack: {height: low|moderate|high|critical,
          holes: none|few|several|many,
          surface: flat|uneven|rough},
  chain: active|inactive,
  combo: active|inactive,
  alternatives: {
    attack: description,
    clear: description,
    smooth: description,
    recover: description
  }
}
```

Descriptions state purpose, whether hold is used, the named clear class
(`full T-spin`, `Tetris`, `single`, etc.), projected height/holes/surface, and
whether an active difficult chain is preserved.  Raw grids and arithmetic are
kept out of Jev's context.  The complete rule in each question says that the
alternatives are legal code-generated placements; `recover` wins for danger,
`attack` is allowed only when its described difficult event is safe, and an
active chain should be preserved when possible.

The request contains two independent `Choice` questions, `first` and
`second`, with the same criteria in opposite orders.  An answer is accepted
only when both values are the same valid track.  If probabilities are present,
that track must have probability at least **0.38** in both answers.  The
request is submitted asynchronously with a **0.80 s** timeout (longer than the
measured Jev tail); late, malformed, contradictory, or low-probability results
are ignored.  The answer is tagged in code by piece identity and is discarded
if the piece or hold state no longer matches.

## Astra interface and revision trigger

Astra sees the ruleset, piece count, bucketed stack facts, lines/level,
combo/B2B state, active and held piece, five-piece queue, current phase, and
whether the call is the initial or danger review.  It must return a leading
phase token only.  A phase is revised after the initial phase and, at most
once more, after at least 20 further pieces when holes or height make the
current phase suspect.  This uses Astra for the multi-piece offense/recovery
choice rather than wasting its slow output on coordinates or actions.  A
missing/late token leaves the existing phase unchanged.

## Fallback and real-time policy

There is always a code-owned plan.  At a piece boundary the harness posts a
fresh `noop`, re-observes, computes a depth-0 guardian, and posts a verified
movement or hard drop; background depth-2 search and Jev never block the clock.
If no model answer is ready, the default track is `recover` in the recovery
phase, a safe difficult clear or rotation-qualified full-spin setup when its
envelope is satisfied, then a safe ordinary clear, and otherwise `smooth`.
If Jev disagrees, times out, returns garbage, or becomes stale, the current
selected track or this deterministic default remains.  Near lock, hard drop
takes precedence over model/search results unless a verified movement/rotation
is still needed to complete the selected target.  Hold is used only
when `can_hold` is true and is marked consumed after one hold.  Any failed
worker result is discarded; model-unavailable is allowed to propagate as the
contract requires, while ordinary timeout/runtime errors are treated as no
answer.

## Evidence that would confirm or refute it

Confirmation would preserve `0017`'s zero-illegal-action, 200-piece, zero-stale
behavior while increasing difficult line clears: more full T-spin/Tetris
lines, longer B2B chains, B2B points, and mean score above `63216.67`,
especially on the seed-4-like safe-but-score-poor tails.  Traces should show
`attack` tracks selected only for code-certified candidates and phase/track
changes preceding subsequent difficult clears.

The hypothesis is refuted if attack tracks create holes/top-outs, increase
stale ticks or illegal actions, reduce 200-piece completion, or leave B2B and
difficult-clear frequency unchanged while lowering score.  If survival stays
perfect but B2B remains near zero, the event recognizer or track gate is too
conservative; if B2B rises while holes/height rise materially, the safety
threshold—not the division of labour—is the problem.  If Astra/JeV answers
rarely change tracks but the code scorer improves, that supports the event
representation and rejects the model role as necessary; if only model choices
change outcomes, later iterations should tune the state/questions rather than
expand geometric search.
