parent: 0002

## Hypothesis

The leader's project ranking is capable of wins, but its executor spends on any affordable project in score order rather than protecting a partially funded capital project. The loss traces show the concrete pattern: seed 102 bought four development cards but built only two settlements total and no city, ending at 3 true VP after 43 turns; seed 103 bought three development cards and built four settlements, but no city, ending at 4 VP after 84 turns. This does not prove the purchases caused either loss, but it is consistent with repeated cheap investments competing with the wheat/ore needed to convert a growing hand into fast VP. The very uneven 0002 scores (1.10, .03, .04, 1.10) argue for a failure-aware policy, not a fixed opening script.

## What changed vs. the parent

I kept 0002's public-observation project search, legal-action compiler, transaction checks, and sparse Astra portfolio advice. The executor now reserves one current city or settlement project by ID and recomputes its route, cost, trades, and funding ETA after every observation. It saves while funding that project, only releases a stalled project after 16 own turns without ETA improvement, and reports the stall and `ctx.progress()` evidence to Astra. A completed, blocked, or unavailable target is replaced from the current legal project set. Development-card and award projects do not displace the reservation before 7 VP; at 7 VP or in explicit army mode the reservation is relaxed for tactical alternatives. The model call's foreground wait is raised to 240 seconds, above the archive's 211-second observed maximum, with the existing budget reserve and asynchronous handling retained.

## Evidence that would confirm or refute it

Confirmation would be fewer early development purchases, more city upgrades or timely settlements, and improved worst-seed true VP/win rate without increased illegal actions. Refutation would be resource hoarding, reservations that repeatedly expire without producing a build, or weaker results on boards where development cards or awards are the fastest winning route. The proposal is supported only if held-up capital actually converts into VP before the AlphaBeta opponents; lower card counts alone are not evidence of improvement.
