parent: 0003

## Hypothesis

The leader loses winnable boards because a legal menu label is not a complete
action meaning. The exact label `change position / flip summon` covers both
flipping a Set monster and rotating an already face-up monster. Its substring
dispatch takes the Flip Summon branch first, rewarding a face-up attacker for
turning itself into defense. Prompt advancement and zero illegal actions do
not detect this strategic self-damage.

Concrete archive evidence: in `0003/trace_seed1108.jsonl`, step 46
(`choose r1p46 0`) turns a 2000 ATK / 100 DEF Gene-Warped Warwolf from attack
to defense against a 2000 ATK Warwolf and 1900 ATK Luster Dragon. Step 66
repeats this. In duel 3, step 206 (`choose r3p55 0`) turns Alexandrite Dragon
into defense on an empty opposing monster field, reducing the available
3900 damage to 1900 while the opponent has 4000 LP. This episode loses all
three duels, score -1.048333. A scan of all twelve leader training traces
finds 85 attack-to-defense commands, including 75 on monsters with at least
1700 ATK. These are observed actions, not simulated counterfactual wins.

## Change under test

Keep the leader's legal-action transport, public-information tactical fallback,
subset validation, and native engine authority. Replace ambiguous substring
handling with position-aware semantics: only a face-down monster can receive
Flip Summon value; attack-to-defense and defense-to-attack decisions evaluate
their actual destination, available combat, and defensive purpose. A small
public-board combat allocator compares complete attack assignments, including
stopping, equal trades and remaining threats, rather than maximizing each
attack's immediate LP damage in isolation. Unknown defenders remain explicitly
uncertain. It never reads a simulator or hidden state.

Every executed position command records the expected destination and checks
it against the next observable board. Combat plans are recomputed from the
live menu after every response. Token/progress checks and semantic-cycle
checks remain; a violated consequence or rejected/looping plan sends concrete
evidence to Astra. Astra receives compact current state and tactical summaries,
with bounded waits above observed 90-second inference latency and below each
300-second response deadline. No seed-specific branching is used.

## Confirmation / refutation

Confirm first that fresh native observations match the intended position
transitions, attacks retain useful bodies, and forfeits/truncations remain zero.
Then compare native wins, weakest-seed score and both seats/decks against the
parent, rather than treating LP gain as a win. Refute if corrected positions
merely expose more monsters to Mirror Force, trade away survival, or leave
win rate unchanged. Replay checks establish action legality and repair the
specific archived decisions; they cannot establish counterfactual game wins.



## Development and replay checks

One permitted development episode (seed 3301) completed all three duels in
217 seconds with score **0.3785417**: two wins, one loss, zero forfeits,
truncations, technical errors, or invalid actions. The parent’s corresponding
seed is not a controlled comparison, so this is completion evidence rather
than a claimed score improvement. The local policy replayed 3,929 archived
observations with zero invalid generated actions; parser recovery tests covered
garbage, malformed JSON, empty answers, timeout, backend failure, stale tokens,
duplicate subsets, weighted Tributes, and incomplete orders. The public combat
allocator also passed a visible-board lethal assignment fixture.


The second permitted development episode (seed 3302) completed in 479 seconds
with score **0.3720833**, again two wins and one loss, with no invalid actions,
forfeits, truncations, or technical errors. Across both development seeds this
is **4/6 native wins** and mean episode score **0.3753125**. No full training
evaluation or held-out evaluation was run.

The two runs exercise different inference paths. On 3301 the initial request
hit the 210-second guard and no Astra answer was applied. On 3302 two Astra
plans were applied successfully (reported model latencies 72.49 and 57.61
seconds); a third request hit the guard. Its first request took about 187
seconds end-to-end despite 72.49 seconds reported inference latency. Thus model
queue/transport time remains material. Timeouts are 235 seconds; the separate
wait guard is 210 seconds, with a shared-prompt retry ceiling of 240 seconds.
Ordinary planning is limited to two reviews per duel, with nine total calls
including repairs; initial planning waits for an own Main Phase rather than
spending a call on an opening Draw/chain window.

After those runs were launched, the final semantic-plan compiler was hardened:
a future `flip summon` intention must bind to a currently face-down monster;
a future generic position-change intention must declare its destination. The
same correction is included in Astra's prompt. Explicit fixtures confirm that
an upright attacker cannot satisfy an intended Flip Summon. This last compiler
change was verified by replay/unit checks and the final preflight, but has not
had a third real-model dev run (the two-run allowance is exhausted).

Final preflight: **ok**, native `game_done`, 388 steps, no exception. Stand-in
preflight answered no System-2 call, so garbage/exception paths were additionally
checked with a context stub. `checks.py` and `checks.json` retain the replay and
parser evidence. All 3,929 public-observation replay decisions remain legal;
all 75 identified strong-attack-to-defense decisions choose a different action.

Limitations and follow-up hypotheses: the combat search is a bounded visible
battle assignment evaluator, not a full turn simulator. It does not search
future Spells/Traps, know hidden defender identities, or guarantee damage through
unknown backrow. Astra can still issue strategically conservative plans: its
second 3302 plan called Alexandrite the main attacker but ended the turn after
setting interaction, omitting Battle Phase. That is a separate tactical-plan
quality problem, not evidence that every planned legal action is optimal. The
long flip-control loss on 3301 also shows that fixing positions does not solve
all resource-exchange decisions. Training results, especially worst-seed and
matchup/seat outcomes, must decide whether this experiment generalizes.
