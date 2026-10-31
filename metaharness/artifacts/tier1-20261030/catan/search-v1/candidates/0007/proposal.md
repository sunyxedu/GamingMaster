parent: 0004

## Hypothesis

The economic planner in 0004 reaches 10 on every archived training seed, but its robber objective is still purely local: it rewards the pip value of a blocked hex, a building on that hex, and the victim's hand size. It does not represent the public race. In the traces, this can leave a rival with a near-winning public score untouched: seed 106 has ORANGE at 6 public VP while RED is at 7 (step 82), and the selected robber victim is BLUE; at step 97 ORANGE is still at 6 while RED is at 9 and the policy again chooses WHITE. Those choices happened not to prevent the eventual win, but on an unseen map a leader can convert one production roll into a game-ending turn before the economic plan acts again.

The experiment is whether a bounded public-VP threat term improves the weak tail by robbing the rival who can win soon, without sacrificing the production-aware choice that made 0004 reliable.

## What changed vs. the parent

* Copied 0004's observation-only economic planner, exact legal-action filtering, opening portfolio, route/award search, maritime arithmetic, and defensive turn loop.
* Changed only `_robber`: after its existing hex production, building, and hand-size score, add `0.35 * opponent.public_vp`, plus a small bonus at 7 VP and a larger bonus at 8 VP. The term is bounded and only changes close races; the engine's legal robber/victim list remains authoritative.
* No model calls, hidden state, engine imports, random draws, or cached action IDs were added. The current observation is rescored after every action.

## Evidence that would confirm or refute it

Confirmation would show more robber choices aimed at the public leader when the leader is within one or two points of 10, fewer opponent scoring spikes after our turn, and equal or higher terminal score with no legality errors. Refutation would show the threat bonus repeatedly abandoning a much stronger production block for a low-value leader hex, increasing turns/actions or reducing terminal VP on seeds where the leader cannot actually win soon. Because 0004 already wins its training archive, the useful signal is improved worst-seed behavior and action efficiency on unseen maps rather than merely repeating a 10 VP mean.
