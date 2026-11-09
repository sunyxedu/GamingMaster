# Candidate 0003: motion-progress feedback for the Jev-only parent

## Parent and one hypothesis

Parent: `candidates/0000-tier2_jev_only/`. I kept its architecture: no Astra calls, one synchronous Jev Choice request per control decision, and the emulator's previous action remains the fallback while a request is running or fails. No `lessons.md` file was present in the archive, so there was no additional refuted lesson to incorporate.

**Hypothesis:** the parent’s worst failure is not merely that Jev is slow. Its action interface gives Jev a large undigested observation and no explicit indication that the currently held direction has stopped moving Ms. Pac-Man. Consequently Jev can repeatedly select a wall-blocked direction. A small motion-progress state plus a deterministic exclusion of a direction that has failed twice should turn local traps into turns without pretending that the unverified maze map is known.

This is deliberately one targeted experiment; I did not add Astra, an imported maze map, or a route planner. The stale-tick behavior remains a useful parent baseline rather than being confounded with asynchronous scheduling.

## Trace evidence

The parent’s worst training episode is seed 2: `eval.json` reports score **110**, 393 ticks, 249 stale ticks, 3 deaths, and only 11 visible collectibles eaten. The aligned `trace_seed2.jsonl` model calls show the local failure:

* During the first calls Jev answered `noop` four times with confidence `0.21`, `0.20`, `0.25`, and `0.18`; the generic question did not establish that neutral input is not a useful movement policy.
* After the opening, the request at frame 340 has player `(29,99)`, direction `left`, score 60, and the answer is `left` (confidence `0.53`). Requests at frames 352, 360, 372, and 380 still report `(29,99)` and keep answering `left` (confidences `0.22`, `0.18`, `0.18`, and `0.22`).
* The same pattern continues through frames 900–956: the player remains `(29,99)` while the answer is repeatedly `left`; at frame 920 the answer is `left` with confidence `0.18`, and at frame 980 a ghost is at `(49,99)` on the same row while Jev still answers `left` with confidence `0.24`. The next recorded life-loss observation is frame 988, score 70, with lives reduced to 2.
* Across the episode Jev chose `left` 96 times out of 145 calls and `noop` 6 times. The parent’s full raw state requests were about 7.7k input tokens, despite not containing a verified wall graph.

This supports adding feedback about observed motion and filtering the known failed direction. It does **not** establish a maze-specific legal move, so the harness does not hard-code one.

## What changed

`harness.py` replaces the raw observation passed directly to Jev with a compact, JSON-only navigation state. It retains score/reward, lives/deaths, player position/direction/velocity, visible ghost positions/velocities and semantic proximity labels, visible power-pill positions, fruit, remaining-pill count, and maze progress. It intentionally omits raw RAM and the full pill list from the Jev request because they are not a wall map and were context-heavy in the parent.

The harness tracks only observations that it has actually seen:

* `motion` is `moving` if the player position changed since the previous decision, otherwise `stationary`.
* The automatic opening is exempt from wall detection until a collectible is eaten or movement is observed.
* `stalled_decisions` increments when the player is stationary across decisions while the same cardinal action is held; it resets on movement or a life change.
* At threshold **2**, `blocked_action` is set to the last cardinal action. That action is removed from the next Choice candidates. This is an observational guard, not a claim about the global maze geometry.

The candidate set contains only `up`, `right`, `left`, and `down`; no-op and diagonals are excluded because the study recommends cardinal movement and the objective is continuous motion. The current player direction/last action is ordered first while it is making progress, preserving momentum and reducing gratuitous reversals. If Jev returns a malformed or excluded value, the bridge uses the previous valid cardinal action when allowed, otherwise the first current candidate. Model failures wait one tick and leave the previous action in force; budget/episode/backend termination exceptions retain the contract’s normal behavior.

## Jev interface

State schema:

```text
{
  game, objective,
  episode: {frame, score, reward_delta, lives, deaths, mode,
            game_over, remaining_pills, mazes_cleared},
  player: {xy, direction, dx, dy},
  ghosts: [{visible, xy, dx, dy, colour,
            distance_band, relation, approach}],
  power_pills: [[x, y], ...],
  fruit: {visible, xy, kind},
  navigation: {last_posted_action, player_direction, motion,
               stalled_decisions, blocked_action, opening_sequence,
               life_changed}
}
```

`distance_band`, `relation`, and `approach` are code-derived qualitative cues; Jev is not asked to calculate distances. Ghost colour is retained as a descriptor, never as a frightened-state flag. `blocked_action` is the only hard action legality signal and is based solely on observed no-progress.

The single question is a `Choice` over the dynamically filtered cardinal candidates. Its instructions say that the game is continuous, forbid noop/diagonals, require respecting `navigation.blocked_action`, and prioritize dot clearing and survival over an unsafe chase. There is no confidence threshold: a low-confidence answer is still a typed directional proposal, and the bridge’s deterministic fallback handles malformed output. A high-stakes future revision should add a separate hazard question or confidence gate, but doing so here would test more than the stated motion hypothesis.

## Confirmation or refutation

The hypothesis is supported if seed-2-like episodes show fewer repeated stationary decisions with the same action, more post-opening position changes, fewer deaths before the first maze clear, and a higher score/collectible count without introducing errors. It is refuted if the player still stalls at the same rate despite `blocked_action` exclusions, or if forced cardinal alternatives cause immediate deaths and reduce the mean score across seeds. Because no Astra or timing architecture changed, a score change can be attributed primarily to the motion-aware Jev interface rather than strategic replanning.
