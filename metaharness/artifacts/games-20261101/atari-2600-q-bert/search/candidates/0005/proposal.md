## proposal.md

# Candidate 0005 — Astra-planned, Jev-controlled Q*bert

**Parents:** `0000-tier2_jev_only` and `0002-tier2_idle`.

## Hypothesis

The Jev-only controller demonstrates that fast model judgments can earn meaningful cube points. Its mean score of 229.167 is substantially better than the idle baseline's zero.

However, the Jev-only agent loses all four lives in every evaluated episode and never completes a round. Its weakness is therefore not an inability to make progress, but insufficiently reliable survival and route selection.

The idle controller establishes the opposite baseline: it avoids unnecessary movement decisions but achieves no score.

Candidate 0005 combines the two mechanisms:

- Jev supplies fast local action judgments.
- Astra supplies a longer-horizon coverage strategy asynchronously.
- Deterministic code enforces graph legality and computes routes.
- A conservative local controller continues operating when either model is late or unavailable.

## Changes

### 1. Deterministic board model

Use the 21-node triangular graph rather than the arcade board.

Maintain the five level-specific cube transition systems:

| Level | Cycle |
|---|---|
| 1 | S → D |
| 2 | S → I → D |
| 3 | S → D → S |
| 4 | S → I → D → I |
| 5 | S → I → D → S |

The adaptor's `needed_landings` and `cube_state` are authoritative.

Use breadth-first search to find useful unfinished cubes and generate a short route. Never deliberately select an edge absent from `safe_moves`.

### 2. Astra's responsibility

Astra receives the initial symbolic observation and returns a compact target-order plan.

Astra does not control individual ticks. Its plan is advisory and can be rejected when it conflicts with the current observation.

The first plan is requested asynchronously so that the emulator does not wait for System-2 reasoning.

### 3. Jev's responsibility

Jev chooses one immediate action from the currently safe action set.

It receives:
- Current Q*bert node and motion state.
- Current cube phases and needed landings.
- Current safe and legal movement sets.
- Visible hazards and their risk classifications.
- Astra's target order.
- The deterministic fallback action.

It does not calculate scores, reconstruct the board, or independently infer action legality.

### 4. Typed decision interface

One `Choice` question is used per request.

Its options are dynamically restricted to the current `safe_moves`, with `NOOP` available for deliberate waiting.

Astra's plan and the complete observation summary are supplied explicitly because Jev does not inherit Astra's context.

### 5. Freshness and confidence

Every Jev answer carries the tick at which its observation was collected.

Answers older than two ticks are discarded.

An action must still be legal in the latest observation. Low-confidence judgments are rejected.

A malformed answer, missing field, invalid action, or timeout cannot crash the controller or inject an unsafe movement.

### 6. Fallback

While Jev is working, deterministic route planning supplies actions.

The controller does not block the real-time game waiting for a model response.

If no safe route can be established, it avoids inventing an action.

Astra failures do not disable the local controller.

## Evaluation hypothesis

The expected improvements are:

1. Higher score than the idle baseline.
2. Better survival than the Jev-only parent.
3. More consistent cube coverage.
4. Fewer stale decisions through asynchronous inference.
5. Lower dependence on slow Astra responses.

## Evidence that would confirm the hypothesis

- Mean score exceeds 229.167.
- At least some episodes complete a round.
- Average life losses decline.
- No episode ends with a harness or model-output parsing error.
- Jev decisions remain useful after freshness filtering.
- Astra plans improve route efficiency over the unplanned fallback.

## Evidence that would refute it

- The fallback selects geometrically incorrect actions.
- Jev answers are consistently too late to influence decisions.
- Astra target ordering makes routes less safe.
- The agent continues losing all lives before completing a round.
- Confidence filtering rejects too many otherwise useful judgments.

The next refinement should use episode traces to distinguish incorrect edge mapping, unsafe route selection, and late model decisions rather than changing the strategy blindly.

Repair note: pre-flight exposed an invalid upper-boundary edge (node -1) during BFS; the graph now validates node and neighbor indices and model spawning failures fall back safely.