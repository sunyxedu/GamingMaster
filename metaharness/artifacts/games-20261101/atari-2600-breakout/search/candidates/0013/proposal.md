# Candidate 0013 — turn-gated interception

## Parent and hypothesis

Parent: `candidates/0005/`, the asynchronous code-owned reflected-interceptor with Jev steering advice and Astra tunnel advice (mean 288.0).

The parent is generally fresh, but its point-target servo changes direction at the worst possible time.  In the seed-4 trace, the tick stream immediately before the first replacement serve contains `LEFT`/`RIGHT` reversals at frames 702–715 and again 722–733, mostly one frame at a time.  The same trace later contains 2–5-frame direction cycles at frames 3506–3665 and 4300–4383.  The milestone at frame 918 already has `lives=4`, and the final life is gone at frame 6821.  This is not a stale-model failure: every one of those tick records has `fresh:true`; the parent is oscillating its requested paddle direction while the ball is descending.

For example, the raw parent trace contains the following consecutive records:

```text
{"tick":702,"action":"RIGHT","fresh":true,"score":4.0}
{"tick":703,"action":"LEFT","fresh":true,"score":4.0}
{"tick":704,"action":"RIGHT","fresh":true,"score":4.0}
{"tick":705,"action":"LEFT","fresh":true,"score":4.0}
{"tick":706,"action":"RIGHT","fresh":true,"score":4.0}
```

Hypothesis: the projected landing is useful, but a point bang-bang controller and a four-pixel deadband interpret a small target/position wobble as permission to reverse immediately.  Because NOOP is not an instantaneous brake and sticky actions can preserve the preceding direction, a one-frame reversal near the catch window is worse than briefly neutralizing.  A narrow, stateful turn gate should reduce lethal edge excursions without taking control away from the parent's projection.

## What changed

I keep the parent's architecture: the code posts a deterministic action every native tick; Jev remains a short-lived advisory bias and Astra remains a slow strategic side choice.  The targeted change is in the fast movement bridge:

* measure the paddle's observed one-frame displacement;
* add a wider directional turn band to the parent's projected target (a clearly crossed target can reverse immediately, while a small reversal is given one neutral tick);
* when a reversal is requested in the imminent landing window while the paddle still overlaps the projected catch, post one neutral turn tick rather than immediately swapping directions; and
* allow the original projection to take over again immediately when the projected catch is clearly outside that turn guard, or when serving/termination requires it.

This is deliberately not a new long-lived landing plan, a wall-band phase gate, or a model-policy rewrite.  It tests only whether suppressing the trace's rapid reversal pattern improves catches while retaining parent 0005's aggressive geometry and advisory calls.

## Confirmation/refutation

Confirmation is fewer one-frame opposite-direction pairs near a descending return, longer survival, and a mean above 288 without stale-tick or episode errors.  The hypothesis is refuted if the guard leaves the paddle outside the catch interval (or produces long NOOP runs like candidate 0010), or if direction reversals are reduced but scores/life duration do not improve.  I log turn-gate decisions so a later iteration can distinguish those cases.

## Jev interface and fallback

The strict-JSON Jev state is the parent's compact derived object: frame, lives, score, event, serve flag, flight phase, projected landing lane/x, urgency, paddle relation, speed bucket, code fallback action, strategic side status, brick count, wall index, and walls cleared.  It contains no raw arithmetic task.  Two batched `Choice` questions choose among `TRACK`, `CENTER`, `BIAS_LEFT`, and `BIAS_RIGHT` in reordered option orders.  A value is accepted only if both answers agree, the value is legal, and the weaker selected-option probability is at least 0.45; it expires after 36 frames or a life/brick/wall/strategy version change.  A timeout, malformed answer, disagreement, or stale version leaves the code controller active.

Jev never supplies a raw action.  The code ignores any advisory bias in the imminent landing window and applies the turn gate only when the projected paddle/ball intervals overlap and observed paddle velocity is still in the old command's direction; outside that narrow case the parent's original movement bridge remains in control.  `serve_required` always produces `FIRE`, and a missing ball without a serve request produces `NOOP`.

Astra is launched asynchronously at the parent's strategic boundaries and may return only `LEFT`, `RIGHT`, or `SURVIVE`; malformed or late text is ignored.  Its result is capped to a small far-from-impact target bias.  The real-time loop never waits for either model and catches normal model failures.
