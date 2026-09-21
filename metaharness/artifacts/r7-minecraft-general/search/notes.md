# Running lab notebook

## Candidate 0002

- No `notes.md` existed when this iteration began. Both prior harnesses were inspected, along with all six trace summaries and the worst-episode model requests/results.
- Check the action bridge before attributing failure to model strategy: Astra seed 0 line 17 returns `gather b:-34,67,41 4`, but line 19 applies bare `gather`. Jev's faster bare-verb loop likewise accumulates `bad_arguments` and zero resource progress.
- Preserve structured observation fields when reducing context. The Astra baseline's text truncation cuts off later inventory/outcome/progress fields; valid per-array summaries retain those fields, while exact execution should use the full observation.
- A model-call budget is not an episode deadline. Astra seeds 1/2 stop before 300 s with `game_done:false`; Jev seed 2 exits on a 5.16 s timeout. Keep ticking through ordinary timeouts and exhausted inference budgets; propagate `ModelUnavailable` and join every spawned future.
- The new experiment uses runtime-generated guarded rules plus Jev recognition of concrete commands. Game-specific arithmetic and paths are learned from public inputs; the fixed interpreter knows no game vocabulary. Compilation latency and incorrect learned guards remain unproven risks.
- Offline contract checks pass (13); no score claim yet. Judge success by completed legal work and authoritative progress, not by fresh neutral heartbeats or high model-to-action fractions alone.

## Candidate 0003

- Candidate 0002's 20-point mean is an idle floor: all 24 Astra calls time out at about 40 seconds, zero menus install, zero Jev calls run, and every applied action is `noop`. Inspect successful model responses and productive commands before crediting freshness or survival scores to an architecture.
- Compilation itself is on the critical path. The parent requests a sizable new DSL program; the earlier Astra baseline produced a complete concrete command in 7.43 seconds. Candidate 0003 tests short grounded command menus with exact simple guards, independent batched Jev readiness checks, and background replenishment instead of another large compiler retry.
- Preserve the entire command. The archive shows `gather b:-34,67,41 4` reduced to `gather`; fast typed verb selection cannot repair discarded arguments. Validate signatures and keep unknown/hold options explicit.
- Reusable semantics and action authorization have different lifetimes. Version and age each judgment, recheck target presence/resource guards/job state against current observations, mark delayed plans' already-dispatched commands used, and expire menus. Retain selected target records separately from broad array samples.
- Ordinary deadlines must not end the episode, but `ModelUnavailable` must propagate, including timeout subclasses. Join every future and distinguish measured inference time from tick-polling response age. Model-call exhaustion is not a terminal game event.
- Thirteen synthetic contract checks pass for 0003; no score experiment was run. Short menus may still time out or become obsolete, and the new thresholds are uncalibrated. The decisive evidence is earlier useful actions and authoritative progress above the 20-point idle floor, not more heartbeat posts.
