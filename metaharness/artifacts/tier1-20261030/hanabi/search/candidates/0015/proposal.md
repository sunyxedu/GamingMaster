parent: 0010

# Hypothesis

The paired terminal-score search is directionally useful, but it still treats a
sender's uncertain discard as an ordinary expected-loss trade. In the weakest
leader episodes, a low-probability live last copy is erased for a free partner
play or a clue token. The immediate tempo gain is visible to the rollout; the
future chain loss is sparse and can disappear in a noisy posterior sample. A
small, structural risk gate should protect the communication channel from
spending a card that has a material chance of being the only reachable copy.
When the recipient has no safe play and the public chop points at a unique live
card, a literal save clue is similarly more valuable than an optimistic scalar
ranking.

# Evidence from candidate 0010

The leader's seed-14 trace records a 21/25 game with all lives, then a
`lost_live_last_copy` event at turn 53 for **R5**. At turn 53 the acting seat
selected `DISCARD 4` as the rollout parent (`p_dead=0.5`, `p_critical=0.417`,
expected loss `0.811`) over another discard whose paired mean was only 0.2045
higher. The next public event shows that slot 4 was in fact R5. The same
trace loses **Y4** at turn 70 in the final countdown. Other leader episodes
lose W5 (seed 11, turn 26), B5 and B4 (seed 12, turns 10 and 31), and W5
(seed 13, turn 55). Several of these are discard-index play signals, so the
signal's nominal free play can directly trade away a unique card.

# Change from the parent

I retain candidate 0010's exact finite-population beliefs, public discard
signals, paired masked rollouts, legal dictionaries, protocol replay, and
separate Astra sessions. I add two code-level safeguards:

* A discard-index signal is removed from the candidate menu when the sender's
  belief assigns at least 0.16 live-critical probability and less than 0.75
  dead probability to the discarded slot. Because the receiver must decode an
  index deterministically, the risky index is removed entirely and another
  ordinary discard or clue is selected instead; silently treating it as a
  non-signal would violate the public protocol.
* Before rollout/advice, the acting seat checks the public chop convention. If
  the other seat has no certain play and its predicted unprotected chop is a
  unique, still-reachable card, the harness posts a truthful rank save clue
  (color only as a fallback). This is derived from the visible target hand and
  public counts, and never copies the target's private faces into its session.

These are structural admission rules for risky actions, not a retuned rollout
threshold. All fallback paths still choose an action from `legal_actions` and
catch arbitrary or failed model replies.

# Confirmation and refutation

The hypothesis is supported if the evaluator shows fewer `lost_live_last_copy`
events, fewer discarded fives/late prerequisites, and equal or better terminal
score while the number of free-play signals falls only when their sender-side
risk is material. It is refuted if the gate over-saves useful duplicate cards,
starves clues, or leaves the game with more inaccessible chains; a public
save that is not reconstructible from the recipient's masks would also refute
it regardless of score. The preflight stand-in only checks contract safety;
training and held-out scores are the actual test.
