# Tetris strategy prior

## First principle: identify the ruleset

Before selecting a move, code must carry a `ruleset` field.  Guideline play uses seven-bag, SRS, hold, next queue, hard drop, and lock delay; NES play does not.  Mixing these assumptions is a high-impact failure ([Guideline](https://tetris.wiki/Tetris_Guideline), [Random Generator](https://tetris.wiki/Random_Generator), [NES mechanics](https://tetris.wiki/Tetris_(NES))).  The recommendations below target the selected Guideline-compatible marathon engine.

## What strong play looks like

Strong play keeps the stack low, flat, and flexible, avoids buried holes and unnecessary overhangs, and uses the queue/hold to preserve future options.  A practical default is a 9-0 stack: leave one side column open for I-piece Tetrises when the surface is safe.  A 6-3 stack can be easier to keep stable and gives more room for T-spin structures; neither is a commandment.  The [Tetris Effect beginner/community guide](https://tetriseffect.game/beginners-community-guide/) emphasizes clean, accommodating flat stacking and discusses 9-0 and 6-3 approaches.

A hole is an empty cell with occupied cells above it.  Holes are expensive because future pieces must clear the overstack before reaching them.  Downstacking means using the next pieces to remove lines above a hole and expose it; prioritize a placement that both clears and uncovers a hole over a superficially larger clear.  Spread residue rather than building a tall clean-looking tower.  The [Tetris Concept downstacking guide](https://www.tetrisconcept.com/2015/02/downstacking.html) recommends lowering surface height, avoiding blocks above holes, and finding placements within the existing upstack.

A useful placement evaluator should minimize holes, hole depth, aggregate/max height, surface bumpiness, inaccessible wells, row/column transitions, and top-out risk, while rewarding lines cleared, eroded cells, a safe landing height, and flexible future surfaces.  The classic four-feature Tetris AI formulation is `a*aggregate_height + b*complete_lines + c*holes + d*bumpiness`; its published coefficients are only a starting point because they were tuned for one ruleset ([Codemyroad heuristic](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)).  The code should search legal placements and score the resulting board; neither model should do arithmetic that the engine can do exactly.

## Openings and transitions

Openers are conditional plans, not scripts to force regardless of the queue:

* **9-0 / Tetris stacking:** preserve one side well and use I pieces for four-line clears when the stack stays low.
* **6-3 freestyle:** use a three-wide side structure when it leaves a smooth surface and provides T-spin/downstack options.
* **TKI:** an early T-spin opener; attempt only while the required queue and safe placements exist.
* **DT Cannon:** a T-spin double into Tetris route; it is queue- and surface-dependent.
* **4-wide / perfect-clear openers:** high-variance patterns; use only when the seed/queue supports them and a fallback is available.

The [Hard Drop opener reference](https://harddrop.com/wiki/Opener) is a catalog rather than a guarantee.  Astra should choose an opener from the observed queue, estimate whether it remains feasible, and abandon it immediately when it raises top-out risk, creates holes, or consumes the only recovery piece.  Code should enumerate and validate candidate placements; Astra should choose the multi-piece plan and transition point.

## T-spins, B2B, and combos

The engine should recognize a T-spin exactly: the last maneuver of the T was a rotation and three of the four diagonal cells around its center are occupied; pointing-side occupancy distinguishes a proper T-spin from a mini ([T-spin rules](https://tetris.wiki/T-Spin)).  T-spin doubles are often a safer repeatable objective than elaborate triples.  Astra may identify a promising setup, but code must verify legality and scoring.

Track **back-to-back** and **combo** as separate state machines.  A Tetris or line-clearing T-spin is a difficult clear and normally continues B2B; a normal single/double/triple breaks it, while a no-line T-spin is neutral.  Modern Guideline scoring commonly uses a 1.5x B2B action multiplier and `50 * combo * level` for successive line-clearing pieces, but attack tables and combo conventions vary ([TetrisWiki scoring](https://tetris.wiki/Scoring), [Hard Drop B2B](https://harddrop.com/wiki/Back-to-Back)).  Code owns counters and score.  Jev can answer “does this candidate preserve B2B?” or “is this clear difficult?” but should not calculate points.

## Rotation and finesse

Use the actual SRS state/kick tables: four orientation states (0, R, 2, L), five kick tests, a separate I table, and no O kicks ([SRS](https://tetris.wiki/SRS)).  Code should precompute legal placements and shortest input sequences, including wall/floor kicks, rather than asking Jev to invent a rotation sequence.  Finesse is an input-generation/bookkeeping problem: send the minimum legal moves and avoid late corrections.  The [movement finesse reference](https://tetris.wiki/Movement_finesse) and [DAS reference](https://tetris.wiki/DAS) explain why wall-adjacent tapbacks and repeat timing matter.

Astra should reason in placements and board outcomes, not in frame-by-frame keypresses.  Jev should choose among named candidates such as `{"candidate":"B"}` or `{"sequence_id":"p17"}`.  Code should execute and verify every input, rejecting a sequence if the world has advanced or the target is no longer legal.

## Real-time policy for tier 2

The engine’s monotonic clock continues while Astra is planning.  At level 1, a cell takes about one second to fall; by level 10 it is about 0.064 seconds, and at high levels gravity can exceed one cell per frame ([Marathon](https://tetris.wiki/Marathon)).  A grounded piece has roughly 0.5 seconds of lock delay, not a guaranteed model-call budget.  Hard drop removes that slack.  Jev therefore acts on every active piece using a short, typed choice; Astra periodically supplies a longer plan and fallback policy.

A practical control loop is:

1. Code obtains the current symbolic state and enumerates every legal final placement reachable with SRS, hold choices, and the current queue.
2. Code computes exact features and filters placements that top out, create fatal holes, violate the chosen opener, or waste a needed recovery piece.
3. Jev answers independent typed questions: best low-risk candidate, whether a candidate creates a buried hole, whether an opener remains feasible, whether a T-spin/B2B condition is present, and which precomputed sequence is shortest.
4. Code executes the selected sequence immediately, using ghost/lock timing and checking each action result.
5. Astra reviews the resulting board every few pieces or on events (new level, high stack, lost B2B, failed opener, garbage if a future multiplayer mode is enabled), then updates the plan and fallback.

Jev recognizes and chooses; it does not plan a 20-piece strategy or perform arithmetic.  Astra plans openings, transition/recovery, downstack priorities, and risk tradeoffs.  Code owns simulation, search, timing, collision, exact scoring, and bookkeeping.

## What belongs in code, Astra, and Jev

| Knowledge/behavior | Code | Astra prompt | Jev typed question |
|---|---|---|---|
| Board collision, spawn, top-out, line removal | Exact implementation and tests | Never outsource | No |
| Seven-bag, queue, hold permission | Randomizer and state machine | Explain only if diagnosing | “Is hold the safer candidate?” |
| SRS rotations and wall kicks | Complete tables and legality search | No frame-level improvisation | “Which legal sequence reaches C?” |
| Gravity, lock delay, DAS/ARR, elapsed time | Monotonic clock and event loop | Choose a safer fallback at high speed | “Is target still viable?” |
| Aggregate height, holes, hole depth, bumpiness, wells | Exact feature calculation | Interpret tradeoffs | “Which candidate has lower top-out risk?” |
| Score, level, combo, B2B, T-spin classification | Exact counters and variant policy | Decide when to pursue offense | “Full T-spin, mini, or none?” |
| 9-0/6-3 and opener feasibility | Enumerate candidate continuations | Select, abandon, or transition | “Is TKI still feasible: yes/no?” |
| Downstack/recovery | Simulate hole exposure and candidate results | Plan the recovery route | “Which candidate uncovers the next hole?” |
| Finesse/input sequence | Shortest legal sequence and verification | No arithmetic | “Choose sequence A/B/C” |
| Evaluation and replay | Seed, event log, score/lines/survival | Analyze failures | No |

## Common mistakes and safeguards

1. **Ruleset contamination:** applying seven-bag/hold/SRS/T-spins to NES play, or applying NES gravity/scoring to Guideline play. Store and display the ruleset.
2. **Building over holes:** penalize every new hole and its depth; prefer a clear that exposes the hole.
3. **Forcing an opener:** abandon TKI/DT/4-wide/perfect-clear plans when the queue or board no longer supports them.
4. **Rigid well worship:** do not preserve a side well or I piece if the stack becomes unsafe.
5. **Chasing Tetrises while drowning:** downstack and reduce height before optimizing B2B/offense.
6. **Breaking B2B accidentally:** code tracks the chain; Jev only recognizes whether a candidate preserves it.
7. **Ignoring hold reset:** hold is once per falling piece, not unlimited swaps.
8. **Non-finesse late input:** use the ghost and precomputed sequence, then hard-drop before the deadline.
9. **Relying on model time:** never pause gravity while Astra thinks; Jev or a deterministic fallback must act.
10. **Treating a pixel screenshot as ground truth:** prefer exact symbolic state; use screenshots only for diagnostics.

## Sources

* [2009 Tetris Design Guideline PDF](https://archive.org/download/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline.pdf)
* [Guideline OCR](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)
* [Tetris Guideline](https://tetris.wiki/Tetris_Guideline)
* [Random Generator](https://tetris.wiki/Random_Generator)
* [SRS](https://tetris.wiki/SRS)
* [Scoring](https://tetris.wiki/Scoring)
* [T-spin](https://tetris.wiki/T-Spin)
* [Marathon](https://tetris.wiki/Marathon)
* [Tetris Effect beginner/community guide](https://tetriseffect.game/beginners-community-guide/)
* [Downstacking](https://www.tetrisconcept.com/2015/02/downstacking.html)
* [Tetris AI heuristic](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)
* [Movement finesse](https://tetris.wiki/Movement_finesse)
* [DAS](https://tetris.wiki/DAS)
* [Opener reference](https://harddrop.com/wiki/Opener)
