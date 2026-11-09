# Candidate 0007 — outcome-aware attack crossover

## Parents and concrete diagnosis

Parents: `candidates/0003/` and `candidates/0005/` (with their inherited mechanisms from the Jev-only and idle baselines).

Both parents solved the first structural problem: they moved the paddle in local code and posted a fresh action every emulator frame while model calls ran in the background. Their stale fractions were effectively zero (`0003`: `0.000119`; `0005`: `0.000122`), so this proposal keeps that frame-rate ownership and the idle-style no-model fallback.

The remaining failure is not simply missed collisions. It is an outcome failure caused by a permanently defensive supervisory policy. The strongest parent, `0003`, reached `21-2` on seed 2, but its held-out-like training seed 1 ended after 53,999 fresh ticks with `0:0`; its final observation was still `phase="rally"`, with no game loss or win. `0005` shows the same sharper example on seed 0: 53,999 fresh actions, `own_points=0`, `opponent_points=0`, and deadline termination. Its only Astra plan was `{"style":"safe","use_feature":false,"target_bias_px":0,"deadband_px":4}`. The same plan pattern appears in all three `0005` episodes; even the successful seed 2 won `21-1` without testing the feature or an attack policy. The traces therefore do not establish that FIRE or a non-central return is harmful; they establish that the parents never exercised those mechanisms.

This matters because the study explicitly gives Robot Pong 1 Speed and Robot Pong 2 WHAMMY contact features, while the parents hard-disable them. A controller that returns every ball but never scores can survive until the wall deadline, and score is the primary objective. The intervention is a controlled, variation-gated attack trial rather than an unconditional aggressive policy.

## Hypothesis

A safe reflected-intercept controller can preserve the rally but can become scoreless when it always returns centrally and never uses the cartridge's legal contact feature. After the first verified right-side return, a small paddle-edge offset plus one FIRE pulse at a verified, overlapping right-paddle contact should create more decisive returns and improve score, especially in a long scoreless rally. This should retain the parents' survival because the feature is used only in the contact window, only once per rally, and only after a current overlap recheck.

The hypothesis is structural: use model supervision to choose the *return regime* and adapt after reward evidence, not to choose a raw 60-Hz action. A negative point immediately after a feature/edge trial causes a conservative cooldown for the next rally; a long scoreless sequence of verified returns triggers an Astra diagnostic rather than silently remaining safe forever.

## What changed in `harness.py`

* Kept the parent mechanism that owns movement, reflected wall projection, score bookkeeping, action mapping, and one `ctx.act`/`ctx.wait_for_tick` cycle per emulator frame. A pending Jev or Astra call cannot hold a stale action.
* Kept phase precedence: `serve_delay` and `finished` clear old velocity and do not treat a stale point sprite as a rally. Automatic serves never press FIRE.
* Improved local tracking around wall samples: visible consecutive positions and adaptor velocities are combined, the observed extrema update reflection bounds, and the wall sign is corrected before projecting. Paddle velocity is smoothed so NOOP brakes an overshooting paddle instead of immediately reversing on noisy target crossings.
* Changed the supervisory default from `safe/no-feature/center` to a bounded `attack/feature-enabled/alternate-edge` policy. The first contact remains a normal safe return; after a verified contact, the target is moved only 1.5 px for a half paddle or 4 px for a full paddle. This is an offset inside the paddle's collision margin, not a chase outside the paddle.
* Added a one-pulse-per-rally FIRE gate. It requires Robot Pong 1 or 2, a reliable rightward trajectory, at most four projected frames to contact, current paddle/ball overlap, no previous pulse, and no reward-triggered cooldown. FIRE is combined with the already selected movement action; it is never used to start a serve or while uncertain.
* Counts a right-side contact only from a rightward-to-away transition, not from an occlusion or a point-frame sprite. Reward events record whether a feature/edge trial preceded them. A negative reward after a trial starts one conservative rally; it does not disable the mechanism for the whole episode.
* Astra now writes a compact JSON *regime* (`attack`/`safe`, feature permission, edge choice, small bias/deadband). It is launched asynchronously at reset and gets one later diagnostic after three point outcomes or six verified right returns with no score. Astra does not return actions.
* Jev receives batched atomic Choices for direction, feature authorization, and edge shape. Reordered duplicate questions are accepted only when they agree above thresholds. Direction hints affect code only when the local trajectory is uncertain; reliable arithmetic wins. Feature/edge answers are merely reservations and are rechecked against the latest observation. Invalid, late, timed-out, or arbitrary stand-in answers fall back to deterministic control.

## System-1 state schema and questions

Every Jev request receives strict JSON assembled from the current observation and code-derived facts. It contains:

* `rules`: explicit action orientation, automatic-serve rule, and the variation-specific contact-feature rule;
* `scenario`: mode, difficulty, variation, and controlled paddle nominal size;
* `observation`: frame, phase, serve timer and waiting flag;
* `score`: reward-derived own/opponent totals and latest reward;
* `ball`: visible/estimated x/y, signed velocity, `RIGHT`/`AWAY`/`UNKNOWN`, reliability and visibility gap;
* `right_paddle`: y and smoothed measured movement;
* `left_paddle`: only size and visible y as context;
* `control`: code-computed projected contact y, relation (`ABOVE`/`ALIGNED`/`BELOW`/`UNKNOWN`), current code direction, early/current contact windows, feature legality, feature-used/cooldown flags, attack stage, and planned/previous edge vote;
* `plan`: Astra's clamped regime; and
* the last four point/contact records.

The batched questions are:

1. `direction_a` and `direction_b`: independently choose `UP`, `HOLD`, or `DOWN` from the supplied code relation; their option orders are reversed. Accept a hint only when the values agree and each selected option has probability at least `.58`. It can override code only for uncertain perception and expires after six frames.
2. `feature_a` and `feature_b`: choose `USE_FEATURE` or `NO_FEATURE` for a legal contact trial, with reversed option orders. Accept a reservation only when both agree at probability `.58`; code still requires the current four-frame contact, variation, overlap, and one-pulse budget. A missing/ambiguous answer means no feature.
3. `edge_a` and `edge_b`: choose `UPPER`, `CENTER`, or `LOWER`, again with reversed options. Accept agreement at `.55`; otherwise use the Astra plan or alternating edge fallback. Jev never computes a pixel offset.

These are recognition/choice questions, not requests for arithmetic, wall integration, scoring, or action generation. The code computes all numeric facts and translates semantic directions to the calibrated `RIGHT`/`LEFT` names.

## Astra revision trigger and adaptation

The initial Astra call is short and explicitly informed that the local controller is allowed to test Speed/WHAMMY at contact. Its numeric bias/deadband and a positive attack recommendation are accepted, but a generic initial `safe/use_feature=false/center` answer cannot silently suppress the experiment that distinguishes this candidate from the parents. Only the evidence-triggered replan may retreat to safe/no-feature. The harness clamps every field. The second call is not on the critical loop. It is launched after either:

* three observed point outcomes, enough evidence to compare a trial with its reward; or
* six verified right-side contacts without any point, which is the exact failure signature of a scoreless safe rally.

The replan prompt includes the current observation, six recent point/contact records, and the current regime. It asks whether to retain the attack trial or retreat temporarily after negative evidence. A malformed or missing Astra response leaves the bounded local policy intact.

## Fallback and contract safety

The local controller works without either model. All model calls are spawned asynchronously and catch timeout, budget, episode, provider, malformed-answer, and pre-flight stand-in failures. A pending future is never required to post an action. The loop stops when `ctx.done` or the game is done; `ctx.act` is posted before `ctx.wait_for_tick`, so each normal tick is fresh. Invalid action names fall back through the public `action_spec` to NOOP. The harness imports only the standard library and `metaharness.*`, and every Jev state is finite strict JSON.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a higher mean return than `0003`'s `8.667` or `0005`'s `7.333`, especially conversion of their `0:0` deadline episodes into positive points or a match win, while stale ticks remain near zero and no episode errors. Traces should show a bounded edge target after a verified return, at most one feature pulse near contact, and reward records distinguishing attack trials from safe rallies.

Refutation would be repeated negative rewards immediately after correctly overlapped feature/edge contacts, or no scoring improvement despite several long scoreless rallies actually exercising the trial. If the attack raises opponent points without increasing own points, the next iteration should disable the feature or use only the safer edge offset. If scores remain scoreless but no feature is ever accepted, the interface gate—not the attack hypothesis—was not tested and should be audited separately.
