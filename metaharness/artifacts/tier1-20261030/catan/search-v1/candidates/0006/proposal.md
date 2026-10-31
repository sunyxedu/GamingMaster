parent: 0002

## Hypothesis

Candidate 0003 solved the old interface deadlock, but its fixed action ranking
still has a strategic bottleneck.  In its seed 103 trace, the opening produced
only wood/sheep/wheat (`self.resources` at step 3), then the controller spent
many turns on maritime trades and eventually stopped at 5 VP.  It also bought
and played Monopoly at step 101 while no city path existed.  This suggests that
the remaining failure is resource-flow coherence: a high-pip opening with no
brick/ore complement can be worse than a slightly lower-pip diversified
opening, and arbitrary 4:1 trades can erase the cards needed for the next VP.

## What changed vs. the parent

This harness tests a resource-plan policy.  The second initial settlement gets
explicit bonuses for missing resource types, especially brick and ore.  Normal
settlements and cities are ranked by their production features, while roads get
extra value when an endpoint is an empty productive vertex.  Maritime trades
are simulated against the current hand and receive positive value only when
they reduce the deficit of a concrete city, settlement, road, or development
cost; otherwise ending the turn wins over a card-cycling trade.  VP-producing
builds outrank development plays, and at nine true points they receive an
additional immediate-win bonus.  Robber choices combine production, buildings,
public VP, and hand size.  The loop still commits only current legal IDs and
contains all adaptor exceptions, so it uses no model calls and remains safe
under arbitrary pre-flight model stand-ins.

## Evidence that would confirm or refute it

Confirmation would be a higher floor on maps where the best first intersection
is resource-concentrated: fewer repeated maritime trades, more early brick/ore
access, and more cities or settlements before the hard cap, especially on the
seed-103 pattern.  Refutation would be traces showing that diversified second
placements lose too much production, or that the trade filter waits while a
development-card race is decisive; those traces would show the missed legal
build and the resource deficit immediately before it.
