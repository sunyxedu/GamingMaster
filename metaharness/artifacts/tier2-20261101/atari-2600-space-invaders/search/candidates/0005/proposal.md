# Candidate 0005 — bounded quiet periods and asynchronous shoot-and-relocate

## Parents and hypothesis

Crossover of `0000-tier2_jev_only` and `0002-tier2_idle`. The first earns points by firing, but synchronously asks a numeric, raw-observation action question and stops moving for most of each laser flight. The second earns no points but supplies a useful robustness primitive: release controls, wait on the game clock, and never depend on a model returning. This crossover keeps that quiet primitive **bounded and conditional**, combines it with targeted shots and independent movement, and removes model latency from the actuator loop.

**Hypothesis:** the main failure is not merely slow inference. Sparse projectile observations plus the coupling of “laser busy” to “do nothing” creates long, exposed stationary periods. Tick-local geometry and shoot-and-relocate should improve survival-adjusted scoring; Jev should choose among semantic maneuvers rather than repeatedly rediscover coordinates and firing legality. Idle's longer survival is an observational association, not proof that stopping is optimal.

## Concrete archive evidence

- `0000-tier2_jev_only/eval.json`: mean 139.375, stale fraction 0.581969, no errors. `0002-tier2_idle/eval.json`: zero points, no errors, about 693 ticks / 46 seconds survived. The worst Jev episode lasts only 425 ticks / 28.424 seconds for 105 points.
- In `0000-tier2_jev_only/trace_seed1.jsonl`, 179 request observations contain **zero nonempty bomb lists**. Life decreases appear at lines 348 (frame 744, score 75), 506 (frame 1084, score 105), and 791 (frame 1697, zero lives), all with `bombs: []`. Seed 3 also has zero bomb sightings in 158 requests and loses all lives. This does not prove bombs caused each death, but it directly refutes treating an empty bomb list as a certificate of safety.
- Seed 1's ticks contain 224 `noop`, 150 `fire`, and 51 `right_fire`; no left action. Request 14 sees a laser in flight at x=37 and picks `noop` with winning probability 0.54. Requests 14–21 keep choosing `noop` while the laser travels. Later request 111 at frame 1048 chooses `right_fire` with only 0.37 winning probability, immediately before a life loss. The parent never checks those probabilities.
- Seed 1 request 22 takes 0.301 seconds; ordinary calls are about 0.13–0.20 seconds against a 0.0667-second tick. 247/425 ticks are stale. A sub-tick timeout would reject usable answers rather than fix this.
- Player x jumps from the opening value 90 to 37 without a left action, and returns to 90 around deaths. We treat discontinuities as reacquisition, not physically plausible velocity. Observed laser motion in seeds 1 and 3 has median 2 pixels/frame. Movement is learned online rather than treating the opening placeholder as a calibration measurement.
- Initial observed shield masks and observation values override conflicting study prose. No row-value table is hardcoded. Decoder-reported kills are not treated as rewards: the idle parent reports six inferred kills despite zero score.
- No `lessons.md` exists in this archive. No sibling candidate, held-out result, game implementation, model backend, or other experiment is inspected.

## What changes

1. **Tick-paced actuator.** Post one newly checked action per observed frame, then `ctx.wait_for_tick()`. Never block the actuator on either model. Legal-action filtering is always applied.
2. **One persistent Jev worker.** A latest-state mailbox bounds backlog to one pending request and one in flight. Approximately one packet per four ticks; timeout 2 seconds, above measured tails. Late answers expire instead of being executed late. A persistent worker avoids per-request thread/connection churn.
3. **Geometry in code.** Track formation motion by stable row/column lattice, estimate shot and player speeds, predict a short shot intercept, and select the lowest living invader in a chosen column. The single-shot lock controls firing, **not movement**. Detecting a new shot releases the target lock so a new lane can be approached during flight.
4. **Bounded quietness.** Quiet releases movement briefly when aligned or in occupied cover. Exposure expires; after a death with no recently observed bomb, quiet is disabled and exposure limits shorten. Choose nearby relocation rather than a long blind crossing, and commit to that destination for a short window so a late Jev answer cannot make the controller oscillate. Recently observed bomb tracks survive brief decoder gaps. All candidate routes receive the same conservative collision and one-tick sticky-delay checks. Shield occupancy is only a hint, never invulnerability.
5. **Event-driven Astra.** At a death or a prolonged score stall, request a short asynchronous strategic revision. At most two requests, no overlapping Astra calls, and a minimum ten-second frame interval. Astra selects only validated `focus` (`nearest`/`outer`) and `posture` (`mobile`/`cover`) presets. Its plan is copied into subsequent Jev state, changes target ranking / dwell policy, increments a version, and invalidates old advice. It cannot emit executable code or bypass geometry. Malformed text retains the existing plan.

## Jev state and questions

Every request explicitly contains `rules`, the current `plan`, categorical `evidence`, and three `candidates`. Evidence includes visibility uncertainty, quiet permission, overdue/brief exposure, laser availability, progress/stall, formation height, cover condition, and whether a living target exists. Each candidate describes clear/near/collision route, left/right/hold movement, cover, and alignment. Code computes numeric comparisons; Jev is not asked to count aliens, perform arithmetic, infer joystick mappings, or reconstruct earlier requests.

- `maneuver`: Choice over `engage`, `relocate`, `quiet`, `unknown`.
- `crosscheck`: the same independent Choice with reversed option order.
- `quiet_supported`: Noul requiring explicit quiet permission, clear route, and exposure not overdue.

The two Choices must agree with winning probability at least **0.60** each. Present confidence must be at least **0.30**; absent confidence is not invented. Quiet additionally requires Noul at least **0.80**. Unknown, disagreement, malformed answers, and insufficient probability use the fallback.

Packets are tagged with frame, life count, wave, plan version, and player x. Reject responses older than 32 frames or with changed life/wave/version. An accepted semantic intent expires after 36 frames from its original observation. On **every** execution, rebuild candidates and reject advice that increases collision risk, contradicts current quiet permission, or is much worse than the checked fallback. Quiet also requires position drift at most eight pixels. A cached maneuver is not a cached joystick command.

## Fallback and failure handling

Fallback ranks three candidates by predicted collision risk, then progress/exposure costs. It continues targeting and firing while models are late. During opening/discontinuous-position reacquisition, it uses stationary fire; during a known beam flight it releases fire but may move. It does not chase UFOs or assume empty bomb observations imply safety. Bomb speed envelopes, collision margins, conservative screen bounds, and dwell limits are testable control priors, not claimed ROM-exact constants. They are seed-independent.

Timeout/transient/typed-response failures retain the fallback. Budget and backend termination propagate normally; the worker Future is polled to avoid hiding model unavailability. Stop/wake events terminate the persistent Python worker; in-flight model calls remain subject to evaluator cancellation. No private emulator state, nonstandard imports, simulator, direct backend, or cross-episode cache is used.

## Confirmation, refutation, and reporting

Confirm if evaluation raises mean score above 139.375, extends survival over Jev-only, reduces stale ticks substantially, and records real `jev_changed_control` events rather than merely extra calls. Wave clears would be especially strong evidence. Compare score, lives, survival, errors, accepted-versus-effective Jev advice, and movement during shot flight. Events log expired answers, confidence-gated acceptance, effective maneuver changes, unexplained deaths, and Astra policy revisions.

Refute if proactive relocation loses more lives or suppresses firing enough to reduce score; if frequent reacquisition blocks progress; if predicted routes cannot compensate for absent bombs; or if Jev never changes behavior. An Astra answer arriving after the relevant wave is dropped. If Astra usually cannot return before game end, its adaptation hypothesis is unsupported even when the deterministic controller improves. No benchmark score is claimed before the evaluator runs.

## Validation plan

Compile and exercise the actual module with short contract stand-ins, including arbitrary Astra text, first-option Jev responses, delayed answers, life/wave changes, and budget/backend termination. Replay public archived observations only as an offline structural check, not a causal gameplay evaluation. Leave long training-seed experiments to the evaluator.
