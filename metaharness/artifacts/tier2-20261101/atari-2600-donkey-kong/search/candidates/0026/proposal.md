# Candidate 0026 — retreat first, then reacquire the route

## Parent and concrete failure hypothesis

**Parent:** `candidates/0007/`. This rewrite also retains two useful later controls: 0012's observed player-edge ladder route, and 0015's proof that zero spare lives does not mean terminal. It deliberately does not inherit 0018's score-farming fallback.

The leading hypothesis is that the archived jump controller confuses *opposing a hazard's velocity* with *escaping the hazard*. On a closing approach, an action opposite the hazard's motion sends Mario toward it; score is awarded for the jump even when the landing is fatal. That generates a convincing but misleading reward loop.

The traces provide direct evidence:

* In the parent `candidates/0007/trace_seed0.jsonl`, controller probes reach phase 2, but around probe `n=450` the state is already `transition`; at `n=500` lives are zero and the controller continues posting `noop` through `n=900`. That is a lifecycle omission: 0007 never resumes the remaining Mario or waits for actual game over. Candidate 0026 retains the parent's code-owned immediate actor but uses sparse, zero-spare-safe restart edges.
* In `candidates/0021/trace_seed2.jsonl`, the route confirms phase 2 at `(47,126)`, then at tick 157 starts `right` toward a hazard on Mario's right moving left, at the `commit` distance bucket. The ramp enters `transition` around 0.6 seconds later. After restarting, the controller explicitly enables `farm=true`, advances only to phase 1, and spends 21 jumps there; it ends with 2,100 points but no screen completion.
* In `candidates/0012/trace_seed2.jsonl`, the same phase-2/right-side/left-moving threat is 21 pixels away and triggers `rightfire`. At tick 163 the jump earns 100 points at `(89,117)`; a later transition ends that life without an upper-screen completion. The trace does show that the nominal route can reach phase 3 at `(105,98)`, so the route itself has a positive control.
* Candidate 0018's lowest-scoring `trace_seed3` confirms the score-lane mechanism: after a single `ramp_advance` at `(105,154)`, `score_fallback_enabled` switches to farming. Snapshots from tick 80 through tick 1600 show phase 1, x≈105, y≈154, and action `up`; the same-side loop earns jump points but does not continue toward the phase-1 x≈46 connector. The evaluation confirms 4,000–4,200 points but `screens_completed=0` and `rivets_removed=0`. This explains why score improvement did not test ascent.
* Candidate 0023's worst trace demonstrates another control hazard: after reaching phase 2, a `jump_unconfirmed` event is followed by repeated `connector_switch` events on the same branch. Its route controller alternates rather than cleanly reacquiring an observed target.

## What changed

This is a structural change to control responsibility and execution, not a jump-threshold tweak to 0007:

1. **Code owns a geometric progress ledger.** It routes over the nominal alternating ladder sequence `(106,46,106,46,106,30)`, permits an alternate only after a measured alignment/climb stall, and advances a phase only after observed y enters the next walkway band. A phase can be credited either by measured climb gain or by two grounded upper-band observations after a jump, so the jump apex alone cannot advance the route and a real landing need not be discarded. Death resets the ledger from the actual new observation. There is no repeated-jump farming mode.
2. **Code changes the default escape objective.** A same-row closing hazard normally triggers a directional jump *away from its current side*, at a 33-pixel hard guard. A jump remains a transaction, synchronized to the documented `$85` RAM byte; after landing, code reacquires the route rather than trusting the old x target. A climb that has visibly gained height is not interrupted except at overlap.
3. **Jev becomes an early tactical vote, not a blocking controller.** It is queried asynchronously in the watch/approach bands, when its measured ~0.14–0.2 s latency can still influence a response before the hard guard. It selects only left jump, right jump, or hold, and must agree across reordered choices. Code's hard collision guard remains authoritative.
4. **Astra becomes a post-death macro reviewer.** It receives only a compact public state after a life boundary and may advise a nominal/short-connector retry, retreat/cross jump style, or first rivet side. A valid policy is applied only at a later active-route boundary; it cannot issue a joystick sequence or override observed progress. Invalid or late text is ignored.
5. **Lifecycle remains deterministic.** FIRE starts idle, sparse FIRE edges handle transitions even at zero spare lives, and the actor continues one post per real-time tick until game termination or the episode deadline. There is no arbitrary step-count return.

No training-seed trace is embedded in the controller. Route coordinates and byte meanings come from the public study and archived observations; alternate edges are used only after a measured failure.

## Falsifiable outcomes

The retreat hypothesis is supported if comparable phase-2 encounters with a hazard approaching from the right use `leftfire`, survive the crossing more often than the archived `rightfire` traces, and retain phase progress rather than restarting into a phase-1 score lane. The main success criterion is a `ramp_advance` sequence continuing beyond phase 2, followed by `stage=rivet`, a changed rivet byte, or a positive `screens_completed` value. Jump points alone do not confirm the hypothesis.

It is refuted if retreat-direction jumps still produce transitions at the same rate, if they repeatedly move Mario away from the only usable next connector without recovery, or if phase advancement is logged without observed y-band evidence. A Jev/Astra call by itself is not evidence of useful model cooperation; any late/low-confidence answer must have no effect.

The run has no candidate-specific cutoff: it waits for terminal game-over or `ctx.done`. A nonterminal transition at zero bonus is therefore allowed to continue, unlike the early harness returns in 0018/0023.

## Tier-2 Jev interface and fallback

### State sent to Jev

Each request uses a small strict-JSON object containing:

* game and event labels (`same_walkway_hazard`),
* stage, player horizontal/vertical bands, climb-committed flag, and observed route phase,
* one code-selected hazard's slot, side, motion, closing flag, distance bucket and urgency,
* the code's escape rule and current deterministic fallback direction.

Coordinates, closing motion, same-row filtering, and all distance arithmetic are calculated in code. Jev does not receive the full RAM or unrelated objects and is never asked to compute a distance.

### Batched questions and thresholds

One asynchronous request batches three independent questions:

* `escape_a`: Choice over `jump_left`, `jump_right`, and `hold`.
* `escape_b`: the same judgment with options reordered, to expose option-order bias.
* `closing`: Noul confirmation that the supplied object is a closing threat on this walkway; this is diagnostic and does not operate the safety guard.

A jump vote is accepted only when both Choice values agree on the same directional jump and each returned probability is at least **0.60**. The signature (stage, observed phase, hazard slot/side/motion) must still match, and the answer must be no more than **5 ticks** old. The matched answer can trigger a jump only while the hazard remains closing and is within 54 pixels. Disagreement, absent probability, low confidence, stale state, timeout, or malformed output discards the vote.

The code fallback runs every tick while Jev works. Independently of Jev, code jumps away from the current hazard side at distance **33 pixels or less** (or overlap); an already progressing ladder climb is protected unless overlap is observed. At most one Jev future is outstanding, with a **0.65 s** timeout. Requests are asynchronous and event-throttled; no model latency is allowed to make the actor wait.

### Astra and interface revision

Astra is called at most once, asynchronously after a life is lost, with stage/route bands, last phase, compact hazard facts, score, and the current macro. It is asked for a compact JSON macro, not an action list. Only whitelisted labels are parsed. A result is staged and can affect a *later* life/screen boundary; it cannot interrupt a live jump or alter a tick. Astra should be revisited in a future iteration only if the trace shows repeated, semantically consistent failures at one route boundary (for example, the same measured connector stall or dangerous landing), not merely because a long call returned.

## Runtime safety

The module imports only the standard library and `metaharness.models.jev`. All generated actions are validated against the literal action set. Both model parsers accept arbitrary garbage safely; all asynchronous futures are polled rather than joined. `ctx.wait_for_tick()` keeps the loop synchronized to the real-time clock, and no controller-defined episode length can forfeit play.
