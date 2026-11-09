# 0011 — Persistent spatial quarantine for failed block interactions

## Parent and hypothesis

Parent: `candidates/0008`. I keep its closed-loop progression controller, compiled
exact commands, emergency overrides, asynchronous Jev choice, and no-Astra policy.

The parent fixed stale target handling only with a timed cooldown. That still leaves
the same spatial transaction eligible after the cooldown, so a single bad support
block or gather target can consume the whole real-time episode. The trace gives
repeated, concrete examples:

* Seed 0 posted `place crafting_table b:-22,67,31 up` at trace line 33,
  invalidated it as `adapter_feedback` at line 36, then posted the identical
  placement again at lines 89/93 and continued doing so through the line-1895
  post / line-1900 invalidation. Inventory remained one crafting table and the
  episode ended at score 3.
* Seed 2 shows the same pattern with
  `place crafting_table b:-113,66,-64 up`: it was posted at line 458 and
  invalidated at line 462, then replayed at lines 515/519, 572/576, and through
  lines 2090/2094. The successful earlier placements prove that this is not a
  general inability to place a table; it is a local target loop.
* The zero-score swamp run also revisited one failed gather coordinate: seed 1
  invalidated `gather b:-12,64,-21 1` at lines 68/71, then retried it at
  lines 655/657, 1249/1253, and 1815/1818 after the 150-second cooldown.

**Hypothesis:** a failed interaction is a fact about the observed block ID during
this episode, not a fact that expires after 25 or 150 seconds. Permanently
quarantining that exact spatial target, while leaving navigation failures on a
short cooldown, will force the existing frontier/alternate-support policy to make
novel progress instead of replaying a known bad transaction. This isolates target
memory rather than adding another long-term strategy.

## One change

`0011` adds `failed_blocks`, a set of candidate IDs. When code invalidates a
`gather BLOCK_ID` or `place ITEM REFERENCE_BLOCK_ID FACE` because of explicit
adapter feedback or a no-progress deadline, it extracts the block ID and adds it
to that set. `blocked()` rejects later commands for that ID regardless of the
old timeout. `support_block()` excludes quarantined placement references, and
`usable_block()`/the compiled menu consequently excludes quarantined gather
nodes. The existing timed cooldown is retained for non-spatial commands and
navigation, so a transient path failure is not made permanent. The Jev state
also labels the quarantined IDs, but Jev cannot invent a replacement: all
commands still come from the current code-compiled menu.

No recipe, combat, scoring, dimension, or action grammar rule is changed. A
successful interaction still requires the parent's observed inventory/movement/
dimension postcondition; quarantine is only entered on the parent's existing
failure paths.

## Confirmation and refutation

Confirmation would be visible in traces as `target_quarantined` followed by no
later post of the same gather or placement target, a growing set of frontier
cells or a different valid support, and inventory/advancement deltas after the
old target fails. A score above the parent's opening-only 2.0 mean—especially
stone, iron, or shield progress on seeds 0/1/2—would support the hypothesis.

It is refuted if the same block ID is posted after quarantine, if the controller
still spends the episode on an unchanged placement phase despite alternate
stands, or if it reaches the same score with no target-loop evidence. If it
removes the local loop but still cannot find wood in the swamp, that would
separate a remaining navigation/frontier weakness from the target-memory
hypothesis rather than justify another change here.

## System-1 interface and real-time fallback

Each Jev request is the parent's small strict-JSON derived state:

* `objective`, code-owned `phase`, `dimension`, health and hunger bands;
* filtered audited personal inventory (food, logs/planks, tools, cobble, iron,
  shield, furnace, bed-related items);
* visible hostile entities reduced to kind, range bucket, and ID;
* `options`, at most seven exact currently legal command strings with their
  code-derived purpose;
* the recent timed failures plus `quarantined_block_targets`; and
* a rule that Jev must choose an exact option and must not invent IDs,
  coordinates, counts, or verbs.

The batched questions are unchanged and have separate roles:

1. `action`: a `Choice` over the exact compiled command strings. Its answer is
   accepted only if the value is still in the freshly compiled menu, has the
   same episode and is no more than ten snapshots old, and its option
   probability is at least `0.55`.
2. `safe`: a `Noul` asking whether no visible close hostile makes the supplied
   work unsafe. It is advisory; code's lava/death/health/threat overrides are
   authoritative. An answer value/probability below `0.35` does not authorize
   an uncompiled command and falls back to code.

There is at most one two-second background Jev request. While it is pending the
harness posts `noop`; after 0.8 seconds it commits the first code-compiled safe
option so the continuously running server is not frozen. Late, stale, malformed,
low-probability, or failed ordinary answers are discarded. Emergency actions,
respawn, stale-observation cancellation, active-job continuation, and the
first-menu-option fallback all remain code-owned. `BudgetExceeded` and
`ModelUnavailable` retain the contract's normal propagation behavior.

There is intentionally no Astra call in this candidate: the parent’s trace showed
that slow macro advice did not break the local retry loop. If this change removes
same-target churn but leaves the bot unable to discover wood or exposed stone,
that is the trigger for a later Astra revision of the frontier/navigation state
and its route options—not for loosening this target quarantine.
