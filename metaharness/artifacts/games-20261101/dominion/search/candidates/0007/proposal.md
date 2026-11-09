parent: 0006

Hypothesis: candidate 0006's Gardens controller commits to the alternate scoring
route before it has evidence that the route can repay its setup cost. The clearest
failure is its dev seed 3101 (agent seat 1): the final public state has the agent
at 24 VP with 3 Provinces while BMU-1 is at 42 VP with 5 Provinces. The agent
spent the opening on Gardens/Workshop infrastructure and eventually bought an
Estate as the opponent bought its fifth Province. The paired seed 3100 shows the
same controller can win +5 when it races normally, so the issue is seat/race
adaptation rather than a legality error. Candidate 0006's own trace also shows
the controller buying Village at a public -1 margin immediately before BMU-1's
Province ended the game.

What changed versus 0006: I kept its receding-horizon local executor, exact
current-prompt option IDs, trash/discard/reaction handling, and bounded recovery
call. Gardens+Workshop is now a provisional regime. On every observation the
controller compares public Province counts, exact visible VP, turn number, seat,
and Provinces remaining. If by turn four BMU-1 has even one more Province, is
ahead by four VP, or the game is late with fewer than two own Provinces, the
controller permanently switches to money mode and lets Province/Gold/Silver
priorities take over. On second seat it also refuses an early Workshop purchase
until the race provides evidence that a Gardens build is viable. Gardens gains
start at a lower, still explicit card-mass threshold only while the public race
supports them. The switch is logged and all local decisions continue to be
validated against the current prompt; `ctx.progress()` evidence still triggers
the model recovery escape hatch for unfamiliar or stalled states.

Evidence that would confirm the hypothesis: the weak second-seat Gardens episode
should acquire payload or Provinces earlier, finish before BMU-1's fifth Province,
and improve its VP margin without introducing illegal actions or extra model
calls. Traces should contain a `regime_switch` event when public VP/Province
evidence turns against the Gardens plan. Refutation would be a Gardens board where
the early switch loses a race that 0006's build would have won, or ordinary
Chapel/Witch boards regressing; that would mean the threshold needs to account for
deck throughput rather than just the public score gap.
