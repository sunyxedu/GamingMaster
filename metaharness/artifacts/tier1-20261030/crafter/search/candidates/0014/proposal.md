parent: 0012

Hypothesis: the remaining late-game loss is a day/night scheduling failure rather than a missing recipe. The controller can reach 15–17 achievements, but its objective remains an exposed mining or return route as daylight collapses. A code-owned dusk mode that returns to a station-adjacent tile and sleeps or holds a low-danger position should preserve the run for another daylight cycle, increasing iron/diamond and skeleton opportunities.

Evidence: candidate 0002 seed 1108 had 15/22 achievements by tick 227, then ended with health 0 at daylight 0.046. The final local map contained a skeleton at [18,3] and zombies at [21,6] and [19,7], while the player was at [18,7]; the last actions were movement near the hostile cluster. Seed 1102 similarly collected iron at tick 536 with health 5, stayed at one iron and no iron tool, and died at tick 577 after alternating a short route. These episodes show useful daytime progress being discarded during the dark return. Existing sleep checks prevent an unsafe sleep start but do not switch the strategic objective when dusk arrives.

What changed versus parent 0012:

- Learn an episode-local anchor only from an observed table/furnace-adjacent, healthy, well-fed, quiet daytime state. No seed coordinate or simulator-private state is used.
- At daylight <= 0.26, enter a bounded night mode. Clear the frontier objective, route to the anchor through the current danger-weighted map, and sleep there when no hostile is visible. If the anchor is disconnected, forget it and choose the least-dangerous local move; release the lock after 150 ticks so a bad shelter hypothesis cannot consume the episode.
- Expose anchor/night status to Astra as context while leaving recipe, crop, combat and inventory-delta verification inherited from the parent. Astra remains advisory and asynchronous.

Evidence that would confirm the hypothesis: traces should show `night_mode` before the first hostile-heavy dark interval, a return to the learned anchor, fewer deaths with health already below 5, and more episodes that craft an iron tool or defeat a skeleton after tick 200. The training score should rise especially on seeds that previously ended around 15–17 achievements. It is refuted if station-adjacent anchors are never learned, sleep at the anchor causes more damage than open travel, or the 150-tick lock delays resource discovery without improving survival.

Development checks: preflight passed after each repair with no model-call or action errors. The first single-seed dev run reached 12/22 and exposed a stale-anchor route (the mode started with an old [25,28] anchor); the nearest-reachable-station repair made the second run survive to tick 255, but its score remained 12/22 and it still died after hostile pressure. This does not confirm the score gain, but it confirms the intended mode is entered and bounded on a real episode and identifies the remaining risk: an observed station is not necessarily a physically safe shelter.
