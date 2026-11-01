# Candidate 0011 — reserve one power pill for the last life

## Parent and diagnosis

Parent: `candidates/0008/`. I retain its observation-derived graph, continuous action posting, asynchronous Jev junction arbitration, and non-blocking Astra policy. The parent improved the mean to **1608.3** and ate **126.5** visible collectibles, but every episode still ended with three deaths and zero maze clears.

The parent’s own traces show that its checkpoint mechanism consumes the entire safety resource before the dangerous end of the run. In `trace_seed2.jsonl`, the harness logs checkpoint events for `[8,147]`, `[8,15]`, `[148,14]`, and `[148,146]` (`power_eaten: 1` through `4`); the first life then ends at frame **2008**, score **1170**, with the player at `[81,99]` and all four ghosts clustered around `[79–80,81]`. The remaining lives end at frames **2804** (score **1310**) and **3225** (score **1460**). In `trace_seed4.jsonl`, the same four checkpoint events precede deaths at frames **2104** (score **940**), **3364** (score **1300**), and **3993** (score **1480**). Thus the pills were successfully reached, but were no longer available when the controller needed an emergency escape; this is not a stale-tick or model-latency failure (`0008` had 0% stale ticks).

## Hypothesis

The checkpoint policy is too eager: it converts every power pill into early progress and leaves no resettable survival resource. Reserving one observed power pill until the final life should reduce repeated lethal exposure and may permit a maze clear, while retaining almost all of 0008’s dot coverage. This is a deliberately narrow test of **resource timing**, not a new ghost model or a rewrite of the route planner.

## One targeted change

`0011` starts with 0008 and adds a per-maze `reserved_power_point` selected deterministically from the observed reachable power-pill candidates (the farthest reachable candidate, with coordinate tie-breaking). While more than one life remains:

* normal checkpoint selection excludes the reserved pill, so the first life can consume the other three but cannot spend the final reserve as routine progress;
* emergency escape selection also excludes the reserve for approaching hazards, but permits it for an immediate collision hazard when it is the only remaining pill resource; and
* when the observation reports the last life (`lives <= 1`), the reserve becomes an ordinary checkpoint and emergency target.

The reserve resets only with a new maze map. Pill disappearance is still verified by the existing code before marking a checkpoint consumed. All graph legality, route execution, Jev arbitration, Astra scheduling, and fallback actions are unchanged.

## Confirmation or refutation

Confirmation would be a trace in which one pill remains visible through at least one life transition, is consumed on the last life or during a genuine immediate escape, and the run survives longer than the parent—especially a maze clear or fewer than three deaths—with comparable 0% stale ticks. A score above 1608 with the reserve still available late would support the resource-timing hypothesis even without a clear.

Refutation is seeing the reserve consumed during routine early routing, all episodes still lose three lives at roughly the same progress, or score/coverage fall substantially while lives and maze clears remain unchanged. If a reserved pill reaches the last life but does not help, the next diagnosis should target the graph’s ghost projection or the timing of emergency turns rather than add more model calls.

## Tier-2 Jev interface and control bridge

The Jev state remains the compact derived JSON from 0008:

```text
observation: {frame, tick, score, lives, deaths, player_xy, direction,
              visible_pills, visible_power_pills, pill_active, maze_clears}
plan: {risk_mode, goal_kind, goal_xy, life_phase, fallback_action,
       legal_actions, graph_node, graph_degree, policy_version}
threats: {summary, nearest_distance_bucket, closing_same_lane_count, immediate}
options: {cardinal: {continues_goal_route, corridor_risk,
                     reaches_checkpoint, branch_degree}}
escape_option: already-computed power-pill branch or "none"
```

Each request batches two `Choice` questions over the same legal cardinal options, with the option order reversed in the second question, plus one `Noul` asking whether the already-computed escape option is safer. Jev only recognizes among code-computed branches; it does not calculate paths, count ghosts, infer frightened colour, or choose a diagonal.

A result is accepted only when both Choices select the same still-legal action, their selected-option probabilities are at least **0.40** on ordinary junctions or **0.55** with an approaching/immediate hazard, and the request is no more than **18 native frames** old with unchanged death, maze, checkpoint, and policy versions. The escape Noul needs probability at least **0.60** and can select only the code-computed escape branch. Missing, malformed, disagreeing, low-confidence, timed-out, or stale answers are ignored.

The deterministic graph route and ghost guard post a cardinal action every tick while Jev/Astra are pending. On the last life the code forces `risk_mode=survival` and allows the reserved pill; otherwise the route continues without spending it unless the immediate hazard fallback says the reserve is the only available escape. Astra remains an asynchronous maze-level risk-policy author and never blocks or posts an action. Arbitrary stand-in model text is harmless because only explicit typed Jev values and an explicit Astra `MODE=` marker are consumed.
