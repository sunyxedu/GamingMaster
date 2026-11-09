parent: 0002

Hypothesis: the leader's remaining low scores are caused by terminal combat risk rather than missing recipes. Once the planner has 15–17 achievements, it still explores or approaches a hostile at night; an enemy can then force a short reversal loop and erase the rest of the episode. A bounded, observation-only panic retreat should increase the minimum score by preserving the achievements already earned and leave the normal progression planner unchanged when health is safe.

Evidence: candidate 0002's archived seed 1108 ends at score 0.682 with 15 achievements. Its final trace steps 216–225 repeatedly posts movement actions while health is 4, daylight is 0.006–0.036, and the same hostile remains nearby; health falls to 2 at step 226 and 0 at step 227. Seed 1103 similarly ends at 0.727: steps 240–251 alternate movement during daylight 0.14–0.25, health 3→1→0, with a zombie beside the player. These are concrete late-run deaths after successful table, furnace, plant, stone-tool and zombie achievements. The leader's safe-sleep repair prevents sleeping hits, but it does not cover an expedition that becomes trapped by a moving mob.

What changed versus 0002:

- Track a short positional trace and the first observed table/furnace as a safe anchor.
- Before ordinary combat or objective selection, enter a 14-tick panic window when a hostile is within six tiles at health ≤4, or within eight tiles at night and health ≤6.
- Route toward the anchor when reachable; otherwise score adjacent legal moves by hostile separation, danger, available exits and reversal status. A last-resort facing attack is used only when no escape tile exists.
- Keep the existing asynchronous Astra adviser, verified recipe bookkeeping, map search, placement targets, survival thresholds and safe-sleep guard. Panic state expires automatically and does not spend a model call.

The hypothesis is confirmed if training episodes show fewer health-zero endings, higher worst-seed score, and the same early achievement breadth with only a small action increase. It is refuted if retreat strands the player against map boundaries, causes missed combat achievements, or leaves the score unchanged because deaths occur before a hostile is visible. Traces should show a retreat toward a station or an increasing hostile separation after the trigger.
