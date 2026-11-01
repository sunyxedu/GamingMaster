## `candidates/0005/proposal.md`

# Candidate 0005 — asynchronous Astra + tactical Jev + deterministic guard

## Parents

- `0000-tier2_jev_only`: source of the fast tactical decision mechanism.
- `0002-tier2_idle`: source of the conservative survival principle and the observation that leaving the game idle preserves lives but makes no progress.

## Hypothesis

A Jev-only policy can select plausible actions without maintaining a coherent route. The result can be premature life loss and zero completed screens. An idle policy avoids unnecessary movement but never starts the cartridge.

Combining strategic route selection, frequent tactical recognition, and a deterministic fallback should improve score without sacrificing episode robustness.

Astra should select a small number of strategic preferences. Jev should classify immediate tactical situations. Code should own geometry, timing, legal actions, hazard guards, and the fallback policy.

## What changed

1. Start with exactly one `fire` action when the observation reports `idle`.
2. Maintain an executable deterministic controller for every stage.
3. Start Astra asynchronously and restrict its output to a small JSON route preference.
4. Start Jev asynchronously, batching an immediate-action Choice and a danger Noul into one request.
5. Keep the emulator running while either model is thinking.
6. Associate Jev results with the tick at which their state was captured.
7. Reject invalid, low-probability, stale, or unusable model answers.
8. Override walking and climbing when a nearby same-row hazard is closing.
9. Replan strategically after a screen transition, life loss, or stage change.
10. Preserve the current cumulative ALE reward as the evaluation score rather than reconstructing it from the HUD bonus.

## Jev state schema

Each request receives a compact JSON object:

| Field | Meaning |
|---|---|
| `stage` | idle, ramp, rivet, transition, game_over, or unknown |
| `player.x_region` | left, middle, right |
| `player.y_region` | bottom, lower_middle, upper_middle, top |
| `player.dx`, `player.dy` | observed motion |
| `player.jumping` | whether a jump is in progress |
| `hazard.present` | whether a visible obstacle exists |
| `hazard.relative_position` | semantic position relative to Mario |
| `hazard.motion` | closing, moving_away, uncertain |
| `hazard.distance_bucket` | very_close, near, far |
| `hazard_count` | number of visible hazards |
| `rivet_rows` | four raw rivet status bytes |
| `rivets_removed` | adaptor-derived count |
| `bonus` | current displayed bonus |
| `lives` | remaining spare lives |
| `screens_completed` | completed screens |
| `score` | cumulative unclipped ALE reward |
| `plan` | Astra's validated macro preferences |
| `observation_age_ticks` | age of the summarized observation |

Raw RAM and unrelated object data are deliberately excluded from Jev's request. Code derives the relevant geometry and motion first.

## Questions

### 1. `action`: Choice

Choose one action from the legal set.

Options are described in terms of their actual cartridge use. The options include neutral movement, horizontal movement, climbing, and directional jumps.

Jev is asked to recognize the best immediate tactical action, not invent a long-term route or calculate collision distances.

### 2. `danger`: Noul

Determine whether a hazard poses an immediate collision threat on Mario's current row.

A positive answer causes code to reconsider the nearby hazard geometry. It cannot bypass the deterministic collision guard.

## Confidence and freshness

- Jev timeout: 0.40 seconds.
- A completed answer is used only after checking its value against the legal action set.
- Choice probability below 0.48 is rejected.
- Explicit confidence below 0.35 is rejected.
- Missing or malformed answers use the deterministic controller.
- The actor does not wait synchronously for Jev.
- An outstanding request is not duplicated.
- A completed answer is combined with the latest observation through the hard safety guard.

The timeout is deliberately above the measured Jev p90 of approximately 0.161 seconds, while still allowing the actor to continue during slower responses.

## Astra scheduling

Astra runs in a background thread, initially and after significant strategic interruptions.

The prompt requests only three preferences:

- Ramp route bias.
- Rivet clearing order.
- Risk preference.

The answer is constrained to a small JSON object, reducing latency and avoiding a long narrative that would block subsequent planning.

The timeout is 180 seconds because the archived System-2 measurements show substantial latency variance. Astra never controls the tick loop directly.

At most three Astra requests are started per episode.

Invalid or non-JSON output leaves the existing plan unchanged.

## Deterministic policy

### Ramp

- Find a candidate ladder above Mario.
- Walk toward its horizontal coordinate.
- Climb when aligned.
- Recompute from the current observation instead of assuming a fixed route survives a jump or life loss.
- Prioritize a nearby closing same-row obstacle over route progress.

### Rivet

- Read the four raw row status bytes.
- Prefer the bottom-to-top serpentine route.
- Select a remaining side using the observed rivet mask.
- Do not walk through a known cleared rivet gap.
- Use a jump when the controller detects a likely gap-crossing hazard.

### Idle and transitions

- Start with FIRE once.
- Use NOOP during transitions and game-over.
- Do not repeatedly issue FIRE during active play.

### Fallback

If Jev is pending, times out, fails, returns malformed data, or is uncertain, the deterministic controller selects the next action. The controller does not require Astra to finish before making progress.

## Evidence motivating the design

The leaderboard reports:

- Jev-only: mean score 0.000, 58% stale ticks, 1,907 System-1 calls, and six game-over episodes.
- Idle: mean score 0.000, 100% stale ticks, no model calls, and six deadline endings.
- Astra-only: mean score 0.000, 100% stale ticks, with System-2 calls taking up to 177 seconds.

These outcomes demonstrate that neither model-only control nor idle behavior has established a successful route. The hybrid tests whether asynchronous planning and a local policy can improve actual screen completion.

The archived trace files were not accessible in this execution environment, so this proposal does not attribute specific individual actions or trace lines to either parent.

## Confirmation criteria

The hypothesis is supported if evaluation shows:

1. Positive mean cumulative ALE reward.
2. At least one completed screen on multiple training seeds.
3. Lower life-loss frequency than the unconstrained Jev-only parent.
4. Fewer stale ticks than the idle and Astra-only parents.
5. No candidate errors or malformed-action failures.
6. Jev answers actually influence posted actions.
7. Astra plans are accepted and measurably change route choices.

## Refutation criteria

The hypothesis is refuted if:

- The fallback repeatedly moves toward the wrong ladder.
- Rivet gaps are crossed without jumping.
- Jev's action selections remain inconsistent with the latest observations.
- Astra plans arrive too late to influence meaningful route decisions.
- The deterministic hazard guard causes excessive jumping or prevents climbing.
- The hybrid still loses all lives before completing the first screen.

## Next refinement

Use the evaluation traces to identify the precise first failure state. Calibrate ladder alignment, rivet-row coordinates, hazard closing thresholds, and jump duration against observed trajectories. Do not tune to individual training seeds; use state-derived conditions that transfer to unseen RNG streams.

**Important limitations:** this is a proposed implementation, not a completed candidate submission. I could not inspect the parent traces, `lessons.md`, or pre-flight environment, and therefore cannot claim the code passes evaluation. The ladder and rivet heuristics particularly need validation against actual observations.

No other candidate directories were modified, and no evaluation results have been fabricated.