# Candidate 0008: action-specific safety tests and a phase controller

## Parent and hypothesis

Parent: `candidates/0004/`, with the released-hop bridge retained but its
planner/gating policy replaced.  I also used `0003` (the best candidate) and the
0003/0004 traces as the baseline.

The hypothesis is that the score floor is no longer primarily a joystick
latching problem.  It is an interface failure: Jev is asked to emit one global
`Choice` from a 128-byte dump whose only named spatial fields are explicitly
hypotheses.  Its answer has a strong forward/no-op prior and is never forced to
judge *which particular landing candidate* is safe.  Consequently 0003 executes
many correctly released UP pulses but almost no lateral decisions and never
completes a home.  A controller needs to separate route phase from action
selection, ask independent safety questions for the small legal candidate set,
and let deterministic code combine those judgments.

The trace evidence is unusually direct.  In best candidate 0003, seed 103's
observations reach score 13 and row candidate 6 on one life, but every posted
direction is UP; the episode ends after four deaths with `frogs_home=0`.  The
same episode's Jev distributions at ticks 0/10/21 put roughly 0.41--0.45 on UP,
0.04--0.07 on RIGHT, and 0.06--0.07 on LEFT, so the unconstrained action
question had no useful horizontal recognition.  Across all six 0003 traces the
only actions are UP and NOOP.  0004 confirms that simply adding two Choices,
a risk Score, and an Astra card is not enough: its agreement/risk gates reduced
movement to 1--29 UP pulses per run (seed 107 has one UP amid 9,494 NOOP ticks)
and its mean fell to 5.17.  This candidate therefore changes *what Jev judges*
rather than adding another veto that turns uncertainty into indefinite NOOP.

## What changed

1. **Verified phase bands in code.**  The adaptor's AC candidate is used only
   for the observed reset sentinel (`255`) and forward row progression.  Code
   labels `START`, `ROAD`, `MEDIAN`, `RIVER`, and `HOME_APPROACH`; it does not
   claim that a raw byte is a car or platform decoder.
2. **Candidate judging instead of free action generation.**  For each fresh
   compact state, one Jev request contains five independent `Noul` questions:
   `up_safe`, `left_safe`, `right_safe`, `down_safe`, and `wait_safe`.  Each
   asks whether *that named one-frame candidate* is safe and useful, with
   explicit phase-specific meanings.  This follows the Jev contract's
   code-generates-candidates / Jev-judges pattern and makes LEFT/RIGHT visible
   as actual alternatives instead of low-probability entries in a generic
   Choice.
3. **Temporal, filtered state.**  Jev receives the row/x candidates, action
   consumed on the last tick, row-transition bucket, score/death/timer facts,
   a bounded recent transition history, changed RAM addresses, and labelled
   hex motion bytes.  It receives the Astra card as advisory text.  It does not
   have to rediscover the action map or count raw bytes.  Astra receives the
   fuller live snapshot and writes a short, non-executable route/perception
   card.
4. **Phase-dependent deterministic bridge.**  Code compares the typed
   probabilities.  On ROAD it favors UP only when it beats one-frame waiting;
   lateral road corrections need a stronger margin.  In RIVER it compares
   LEFT/RIGHT support candidates before UP, and after prolonged ambiguity uses a
   bounded alternating side probe rather than consuming the life in NOOP.  On a
   fresh start it preserves the useful forward fallback.  These are modest
   thresholds, not a global agreement veto.
5. **Released pulses and pipelining.**  Jev runs in one background future while
   the main loop posts an explicit action every live tick.  Every directional
   answer is followed by two NOOP cooldown ticks; a late/failed/stale answer is
   discarded and the phase fallback is a single pulse.  Thus stale ticks repeat
   NOOP, never a direction.  The action answer is associated with its tick and
   death count, and it cannot be used after a reset.
6. **Astra revision boundary.**  One opening call analyzes the initial RAM and
   motion evidence.  A second call is launched only after a life loss (or a
   later route transition while the first is unavailable) to write a
   postmortem correction.  Astra never blocks the game and its output is never
   parsed as executable code.  The fast interface is therefore stable while
   the slow model can revise the interpretation after concrete evidence.

## Jev state schema and questions

The state is strict JSON with these fields:

- `tick`, `frame`, `score`, `score_delta`, `deaths`, `reserve_frogs`,
  `timer_s`, `level`, `terminal`, and `terminal_reason`;
- `phase`: a code-derived route band (`START`, `ROAD`, `MEDIAN`, `RIVER`, or
  `HOME_APPROACH`), not an assertion about named objects;
- `frog.row_candidate`, `frog.x_candidate`, `frog.screen_x_candidate`, and
  `frog.state_candidate`;
- `action_consumed`, `last_pulse`, `pulse_cooldown`, and `row_transition`, so
  Jev can tell whether the prior edge advanced the frog or caused a reset;
- `raw_motion.known_fields`, `raw_motion.changed_addresses`, and the recent
  `0x80--0xAB` bytes as labelled hex.  The state explicitly says these are
  temporal evidence and not a complete object decoder;
- `recent_transitions` (at most 18 compact records), and `astra_card` with its
  version; and
- the one-hop/release control rule.

For every state the request asks:

- `up_safe`: is one UP hop safe and useful for the next forward row? In ROAD
  this means a likely free next lane; in RIVER it means supported next-row
  landing;
- `left_safe` / `right_safe`: is that single lateral hop safe and useful for
  alignment or avoiding a recognized threat/platform edge?;
- `down_safe`: is a single retreat genuinely safer from an immediate hazard?;
- `wait_safe`: is one NOOP frame safer than every directional candidate while
  the moving gap changes?

They are independent Nouls, so Jev does not need to compare five numerical
options or generate a route.  Their criteria define `true` as safe *and useful
now*, not merely legal, and require `false` for an unsupported/hallucinated
landing. The code reads `.value` first (a Noul probability), falls back to the `true`
entry in `.probabilities`, and treats malformed answers as undecided.  It never
uses generated text as an action.

## Thresholds and fallback

- Terminal or stale/dead-life answers map to `NOOP`.
- `START`/`ROAD`/`MEDIAN`: UP is accepted at probability at least `0.43` and
  at least `0.02` above `wait_safe`.  A lateral road correction needs at least
  `0.56` and a `0.05` margin over UP.  Otherwise one NOOP is posted while the
  next judgment is requested.
- `RIVER`: LEFT or RIGHT must be at least `0.46`, beat waiting by `0.03`, and
  beat UP by `0.02`; otherwise UP is accepted at `0.46` with a `0.02` wait
  margin.  If all candidates remain ambiguous for 24 river ticks, the code
  alternates one side probe.  This is a bounded search for a supported carrier,
  not a held lateral joystick.
- `HOME_APPROACH`: UP has a lower `0.40` threshold, then a lateral candidate
  can be used at `0.52`; survival is still preferred to a speculative bonus.
- Any missing, timed-out, malformed, too-old (over 28 ticks), or post-death
  answer is dropped.  A fallback is selected from the phase: UP at a fresh
  start/road, an alternating side probe in a stalled river, and NOOP when
  phase/evidence is unknown.  Directional fallback/answers are always released
  and followed by two NOOP ticks.

The initial explicit NOOP prevents an implicit-action race.  The loop uses
`wait_for_tick()` rather than spinning, posts a legal action each frame, and
catches ordinary model failures; budget/deadline/model-unavailable exceptions
retain the harness contract's normal behavior.  Arbitrary Astra text and
stand-in Jev answers are advisory and cannot cause a parse or action error.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a materially higher score with at least one home, traces
showing LEFT/RIGHT pulses in the RIVER or HOME bands, and fewer deaths caused by
blind UP sequences.  The strongest confirmation is a life whose row progresses
past the road band and then reaches a home without a long directional run.

Refutation is a trace in which action-specific Nouls still produce only UP and
NOOP, all episodes remain at zero homes despite river side probes, or the
candidate falls below 0003 while its released pulses and stale-tick behavior
are otherwise correct.  If it crosses the road but fails in the river, the next
iteration should replace the RAM-motion card with a calibrated platform decoder;
if it never crosses ROAD, the postmortem should focus on traffic-gap timing
rather than adding more river actions.
