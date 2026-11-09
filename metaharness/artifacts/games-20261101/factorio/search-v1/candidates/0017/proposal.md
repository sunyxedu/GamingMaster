parent: 0015

## Hypothesis

The remaining ceiling is partly caused by draining assembler output too eagerly.  Candidate 0015's research-aware router still begins every turn by extracting every item from every assembler output.  In its seed-202 late trace, turns 127 and 128 each spend four output-extraction operations for only 1–3 automation or logistic packs per assembler, then spend the remaining transfer budget on lab packs and assembler inputs; the final turn reports `transfer_remaining: 0`.  The same pattern appears in seed 404, where output extraction precedes lab and input moves despite buffered science already being present.  A finite assembler output slot is a useful lossless buffer, so repeatedly moving small amounts to stock can starve higher-value transfers without increasing engine production.

## What changed versus the parent

I retain 0015's map-relative construction, three coal sources, recipe-derived paid crafting, research-aware lab routing, fuel reserve, and bounded exception-only Astra advisor.  The transfer policy now treats assembler science output as a demand buffer: it extracts only the deficit needed to bring combined stock and lab inventories to 12 packs, or flushes the excess when an assembler output inventory reaches 20 items.  Non-science outputs retain the parent's immediate extraction.  This keeps output slots below the engine's capacity while avoiding one or more quota-consuming operations every turn when downstream already has science.  All 100-item and 31-operation guards, recipe accounting, and malformed-advisor handling are unchanged.

## Evidence that would confirm or refute it

Confirmation is fewer turns exhausting the 100-item transfer quota, fewer assembler-input shortages, and equal or higher produced and consumed red/green science.  Output inventories should remain below their flush threshold and should not show `full_output`; lab `missing_science_packs` should not rise.  Refutation is any increase in `full_output`, lower lab consumption because science is trapped in assemblers, or no reduction in transfer exhaustion while input starvation persists.  The controller trace records submitted extraction operations, raw output inventories, machine statuses, transfer counters, and production/consumption ledgers for this diagnosis.

Preflight passed for the submitted harness (64.93 points, 37 turns, no failed operations or contract exception, no model calls in the short dry run).  No full development evaluation was run.
