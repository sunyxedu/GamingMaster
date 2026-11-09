- The baseline's one-action-per-call contract is costly on this game: a single slow Astra call can consume the full wall deadline before any move is posted.
- A macro plan must be parsed conservatively; directions in explanatory prose can become accidental moves, so the harness requests and prioritizes an `ACTIONS:` line.
- Local reverse-push dead-square checks and transition prediction are cheap safeguards around irreversible pushes, while the model remains responsible for assignment and ordering.
- Every model call needs catches for timeout, budget, backend, and garbage-output cases; preflight stand-ins deliberately return arbitrary text.
- Leader traces confirm that primitive granularity is fatal: seed 103 made zero moves during a 1,800-second first call, while seed 102 made seven moves after eight calls.
- Reverse-push distances provide a stronger, geometry-aware lower bound than Manhattan distance and can be computed without model latency.
- Search must be bounded and exception-isolated; malformed observations or an expensive frontier cannot prevent an Astra fallback.
- A proposed plan is only useful when its predicted player and box transition matches the returned JSON; clear the suffix after every push.
- Component-minimum player anchors can hide the real post-push side and distort walking costs; retain the actual player square when testing push-state search.
- Dynamic 2x2 pruning is easy to make too aggressive on corridor layouts; static reverse-pull cells and corners are safer first-line pruning.
- A fallback that repeats any legal move can spend the full action cap in a two-cell loop; prefer unseen states and clear model suffixes after every push.
* The naive parent paid one slow Astra call per action; seed 101 used its first call for the whole deadline and made zero moves. Batch primitive plans and use local computation to amortize System-2 latency.
* Sokoban observations expose enough geometry for deterministic flood fill and reverse-push dead-square pruning; keep irreversible safety checks in code rather than trusting prose from the model.
* Replan after pushes but allow a validated walking suffix from a model response, since walking-only calls waste roughly 50 seconds and do not change the push state.
* Parse an explicit compact action line and validate every action against the returned observation; arbitrary stand-in model text must fall back to a legal action without raising.
- Candidate 0001 solved all ten training seeds without Astra calls; local push search can dominate model planning when levels are small.
- Manhattan assignment is a weak search guide in Sokoban; reverse-push BFS distances encode wall and standing-square constraints directly.
- Candidate 0003's seed-103 trace oscillated `left`/`right` for 2,000 actions after model handoffs, so fallback movement must prefer changed, unseen states.
- Keep search exceptions isolated and preserve an Astra fallback: a larger frontier is useful only if malformed observations still survive preflight.
- When all archived training levels are solved, optimize the actual score cost: trace deltas establish `50 * pushes + total_actions` as the integer objective, so walking differences matter after push count is fixed.
- Exact player positions are needed to certify action cost; a player-component key preserves push reachability but loses the position-dependent walking cost.
- A solved-box backward frontier with inverse primitive transitions can meet a forward frontier at an exact `(boxes, player)` state; every proposed join should be replayed before live execution.
- Bidirectional matching potentials can prove minimum cost quickly on these four-box layouts: ten archived starts took at most 0.24 s each and improved the leader by nine integer cost units total.
- Finite, verified partial progress is safer than a movement fallback: the 0003 seed-103 trace spent 1,997 walking steps and hit the action cap for a negative score.
