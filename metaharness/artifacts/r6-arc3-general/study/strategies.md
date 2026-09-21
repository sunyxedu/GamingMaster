# ARC-AGI-3 strategy prior for Astra, code, and optional Jev

Studied 2026-09-21. This document distinguishes published evidence from proposed harness design. The recommendations have not been evaluated as a playing agent in this study. There is no universal build order, fixed opening move, opponent, or metagame counter: the central task is learning unfamiliar rules efficiently.

## What the evidence supports

| Evidence | Finding useful to this harness | Scope / limitation |
|---|---|---|
| [Official preview retrospective](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) | Strong humans move from exploration to efficient execution. Preview games sometimes rewarded brute force, prompting harder future designs | Historical 2025 preview; scores and level counts are not current cache specifications |
| [Graph-based exploration paper](https://arxiv.org/html/2512.24156v1) | Connected components reduce click candidates; a directed graph records tested transitions and routes to unexplored frontiers. The paper reports a median 30 of 52 preview levels solved in its extended runs | Large state spaces and partial observability are limitations. A reset-bookkeeping bug caused repeated fatal exploration |
| [StochasticGoose author's repository](https://github.com/DriesSmit/ARC3-solution) | The preview winner learned which actions change frames. It predicts action type and click position separately, using a state/action experience buffer | Frame change is an exploration signal, not proof of progress. Historical implementation requires extra training dependencies |
| [FrugalExplorer author's repository](https://github.com/shloksah/arc-agi-3-agent) | A lighter effect model uses action/color/region statistics, component-based clicks, and return-to-frontier navigation. It describes frozen UI masks and death attribution | Author-reported prototype, not proof of optimality. Its scoring discussion includes a cutoff that must not be imported into this local level-count evaluator |
| [Official analysis of 160 model replays](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis) | Repeated failures: recognizing a local effect without composing rules, importing an incorrect familiar-game analogy, and winning an easy level without learning a transferable mechanic | Qualitative analysis of specific model runs, not a controlled test of this proposed harness |
| [Workspace Optimization / DreamTeam paper](https://arxiv.org/html/2605.09650v1) | An executable learned model is updated against observed transitions; revisions are checked against earlier evidence. Short action batches can stop at transitions and unexpected events | Authors report public-set gains; architecture and compute budget differ from Astra/Jev. Borrow the evidence discipline, not the entire multi-agent system |
| [Official human study](https://arcprize.org/blog/arc-agi-3-human-dataset) | Public human replays allow study of exploration and recovery; first-run testing measures learning, not memorized solutions | Do not train on a held-out game's replays and then describe its evaluation as unseen |

## Recommended control loop

The following is a proposed adaptation to the two-model system, rather than a claim that these exact settings are established optimal play.

**Code owns exact measurements and execution. Astra owns rule discovery, goal inference, planning, and revision. Jev is optional semantic recognition.** Keep tier 1: no model needs to act while Astra is thinking. A cheaper second model is useful only when it can answer a bounded question more economically than Astra; it is not necessary for real-time safety.

### 1. Inspect before spending a move

**Code:** Parse the current frame into palette counts, connected components, bounding boxes, shape masks, candidate grid spacing, and spatial relations. Preserve raw pixels and intermediate animation frames. Calculate frame differences exactly. Split the display into tentative board, legend/target, and status regions, but keep these labels reversible.

**Astra prompt:** Identify a few possible controllable objects and target relationships. State each as a hypothesis, with observable evidence and an uncertainty. Ask what one interaction would distinguish the leading alternatives. Do not describe an object as a key, avatar, goal, or wall merely because of its color.

**Jev question:** Given already measured objects, choose a tentative visual role from `target_example`, `control`, `movable_piece`, `status_indicator`, `unknown`. Treat this as a suggestion; code cannot infer mechanics from that label.

Why: the official benchmark requires learning without instructions, while its replay analysis shows that misleading analogies can dominate action selection. [Benchmark introduction](https://arcprize.org/blog/arc-agi-3-launch), [model replay analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)

### 2. Use small, discriminating opening probes

**Astra:** Start with one action whose plausible outcomes separate hypotheses. For a directional interface, a single move can identify the affected object, movement step, or rotation. For clicks, select a visually distinct candidate on its actual occupied pixels. In a mixed interface, determine whether a click selects an object for later actions. Probe action 5 near relevant objects only when it is available. There is no justified universal ACTION1 spam opening.

**Code:** Compare before/after positions, shapes, palette changes, selection markers, and status changes. Record no-ops as well as successes. A reverse action is a probe, not an assumed inverse: up/down could rotate a piece or shift a whole board. If a selection or mode changes, previously ineffective actions deserve reconsideration in that new state.

**Jev:** Optionally classify a supplied difference as `translation`, `rotation`, `recolor`, `selection_change`, `multiple_effects`, or `unclear`. Exact displacement and shape transforms should already be computed in code.

This adapts the action-effect learning idea from [StochasticGoose](https://github.com/DriesSmit/ARC3-solution). The proposed opening must remain sensitive to context; it is not a fixed script.

### 3. Compress click exploration into meaningful candidates

**Code:** Generate representatives from visible components, buttons, corners, outlines, and grid cells. Choose an occupied pixel near a component's center, not an arithmetic centroid that might land in a hole. Keep several candidates for a large or multi-part object. Deduplicate coordinates, not every object of one color. Start with high-salience objects; expand to alternate points and a coarse spatial grid if no effect is found. Keep global exploration coverage so a mistaken segmentation does not make part of the screen unreachable.

**Astra:** Decide whether the next uncertainty concerns object identity, click location, an order dependency, or a missing selection mode. Do not keep clicking adjacent pixels when a different action type is the more informative test.

**Jev:** Rank a supplied finite list of candidate object IDs for “looks like a control.” Code maps the chosen ID to coordinates. Jev does not invent x/y values.

The published graph explorer prioritizes connected visual regions over exhaustive pixel search. FrugalExplorer independently describes component-snapped candidates. These support a candidate hierarchy, not a guarantee that each component has one uniform response. [Graph exploration](https://arxiv.org/html/2512.24156v1), [FrugalExplorer](https://github.com/shloksah/arc-agi-3-agent)

### 4. Maintain explicit, falsifiable rules

**Astra:** Maintain a compact rule table: condition, action, predicted effect, supporting observations, counterexamples, and confidence. Distinguish observation (“this press changed orientation”) from inferred mechanism (“orientation determines which face is painted”). Prefer two or three live alternatives to either absolute certainty after one observation or endless enumeration of genres.

**Code:** Store a transition ledger keyed by episode, level, and observation ID. Keep each real transition's before state, action/data, full frame sequence, after state, progress, and prediction error. Once Astra proposes a small executable model in the later implementation phase, evaluate it against stored transitions before trusting its rollouts. A failed prediction should trigger a specific repair: parsing, dynamics, hidden state, or goal predicate.

**Jev:** May flag whether a supplied categorical observation conflicts with a supplied rule. Do not ask it to synthesize dynamics or perform multi-step simulation.

This follows the falsifiable-model approach of [Workspace Optimization](https://arxiv.org/html/2605.09650v1); the ownership split above is specific to this harness.

### 5. Search in the right state space

**Code:** For a small learned deterministic state model, use BFS for unit-cost actions or Dijkstra/A* when transitions carry different costs. Include every feature needed for transitions: position, orientation, color, selected object, inventory, toggles, resource state, and relevant action phase. Search on a learned representation must not read hidden SDK `_game` objects or game source. A model rollout costs computation; a real environment probe still costs a counted action.

For observation-graph fallback, record directed edges, tested actions, failed actions, and paths to frontiers. Return along verified paths. Never assume that inverse-looking controls reverse a transition. Bound graph size and search work; hand Astra the frontier summary when exploration stalls.

**Astra:** Select the representation and goal predicate. A path to a visually attractive target may be useless if the wrong shape, orientation, or mode reaches it. Distinguish a connectivity problem from a transformation sequence or a constraint-satisfaction problem.

**Jev:** No search, distance arithmetic, shortest-path selection, or multi-step planning. It can select among semantic situation labels already derived from the model.

Graph exploration is a useful fallback but has documented failures in large/partially observed spaces. This proposal extends it with explicit state variables and a learned model. [Graph exploration limitations](https://arxiv.org/html/2512.24156v1)

### 6. Treat resources and apparent time carefully

**Code:** Maintain both an exact raw-frame hash and a separately defined semantic state key. Do not automatically mask every changing border pixel. A progress bar can encode fuel, remaining moves, or phase; identical boards with different resources may have different futures. If a display-only mask is justified, freeze its definition for the graph or rebuild keys when it changes. Never change a hash scheme silently mid-level.

**Astra:** Test whether a bar changes with actions and whether its depletion predicts failure. Plan enough resource for both reaching an intermediate object and completing the actual goal. Refill or transform first when the model requires it. An unchanging frame can conceal a counter/phase update, so repeated no-ops are not always equivalent states.

**Jev:** Optional `resource_display`, `target_pattern`, `unknown` classification of a supplied UI region. Resource quantities and route costs belong in code.

Frozen masks are a practical community technique; resource-aware state identity is a necessary caution for this harness. [FrugalExplorer](https://github.com/shloksah/arc-agi-3-agent). Local pacing and action semantics are documented in [interfaces.md](interfaces.md).

### 7. Execute short plans with explicit stop conditions

**Astra:** Once a hypothesis explains observed transitions, select a short plan with predicted outcomes and a stopping condition. Early probes should be single actions. Known safe navigation can use a short batch. A suggested starting batch ceiling is 4–8 actions, subject to tuning; it is a harness choice, not an ARC rule.

**Code:** Submit actions sequentially and validate each response. Stop the remainder on a new level, WIN, GAME_OVER, unavailable next action, unexpected frame/model difference, or a newly relevant object. Do not execute a queue generated from the old level against the next one. Bind every plan to an observation ID so a delayed model response cannot act on stale state.

**Jev:** If semantic event recognition is useful, ask a small independent typed question between actions. If it says `unknown`, return to Astra; do not let it extrapolate the rest of the plan. Metadata-based stops remain deterministic code.

The benefit of short batches is lower model overhead, not beating a real-time clock. Early interruption resembles the batching discipline described by [Workspace Optimization](https://arxiv.org/html/2605.09650v1).

### 8. Learn from failure and from success

**Code:** Attribute a fatal/no-progress action to the exact state and context before RESET. Keep that transition marked tested. Track repeated cycles and failed hypotheses. Undo, when offered, is an ordinary counted action whose scope must be learned; do not assume it refunds resources or reverses every side effect.

**Astra:** After failure, state what prediction failed and which changed condition could make a retry useful. Avoid absolute bans on a move in every context: a dangerous crossing may become safe after a toggle. After success, explain the smallest rule set that accounts for the level and identify what the next level adds. Carry mechanics forward within the game, while rebuilding geometry and local state.

**Jev:** Optional event label on ambiguous visual effects, not a decision to reset or abandon a level.

The graph paper documents a fatal-loop bug from failing to record reset-inducing transitions. The official model audit shows why easy-level success alone is not proof of understanding. [Graph paper](https://arxiv.org/html/2512.24156v1), [official audit](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)

## Applying the prior to the cached games

These are **initial affordances verified by runtime**, not solved rules. The seed manifest in `interfaces.md` should remain the source of truth. Default development games are CD82, LF52, LS20, and SB26; keep the other four free of game-specific tuning.

| Game(s) | Useful initial approach | Ownership |
|---|---|---|
| CD82 | Six controls permit selection, manipulation, and action-5 experiments. Official qualitative analysis identifies a rotate-container / dip-paint / match-target relationship; verify it against the cached version before planning | Astra proposes the relation; code represents orientation and painted faces; Jev may identify candidate target/control objects |
| LS20 | Four directional controls. The preview description involves navigating and transforming a symbol to match another object; reason over both position and transformed symbol state | Astra infers transformations; code searches their combined state. Do not hard-code old level counts, coordinates, or exact routes |
| LF52 | Directions, click, and undo are available; initial RESET yields two frames. Test whether clicking selects what directions affect, while retaining the transient evidence | Astra chooses probes; code tracks selection and transformations; Jev optional object labels |
| SB26 | Only action 5, click, and undo initially. Learn whether clicks configure objects and action 5 commits/executes, but retain that as a hypothesis until tested | Astra distinguishes setup from execution; code records each change and undo scope |
| TN36 and VC33 | Click-only initially. Use the general candidate hierarchy and transition graph; do not assume a fixed click order | Code handles candidates/bookkeeping; Astra interprets rules; Jev optionally ranks candidates |
| SP80 | Mixed actions 1–6. Identify which actions move, select, transform, or commit before constructing a plan | Astra probes dependencies; code tracks the resulting mode/state |
| TU93 | Initially only directions despite a broader metadata tag. Determine their actual effect rather than inferring movement from labels | Astra tests semantics; code measures effects |

CD82 source: [official replay analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis). LS20 source: [preview description](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings). All other rows are proposed probes based on [reset-only observations](runtime-check.json). No version-matched complete walkthrough was verified, and no game-specific source was read. The old graph paper discusses different level totals; the cached LS20 has seven levels, not the older preview's eight.

## Common mistakes and concrete counters

| Mistake | Counter | Implement in |
|---|---|---|
| Treating visual novelty as reward | Keep novelty as a temporary probe priority; evaluate plans against a goal hypothesis and real level progress | Astra + code |
| Treating the first familiar genre as the rules | Require a predicted observable consequence and a discriminating test | Astra prompt |
| All clicks aimed at object centroids | Use occupied pixels, multiple semantic parts, and fallback coverage | Code |
| Same frame assumed to mean same full state | Include resources, mode, recent context; split states when transitions disagree | Code + Astra |
| Ignoring status pixels or treating a timer as goal progress | Retain raw frame; identify what the display measures through interaction | Code + Astra |
| Repeating a reset-causing edge | Record the edge before recovery and change the plan/context | Code |
| Resetting merely to inspect the board | Read the latest cached observation; reserve RESET for deliberate recovery | Code |
| Solving level 1 by coincidence and repeating its path | Summarize causal rules and inspect new level features | Astra prompt |
| Forgetting lessons between model calls | Persistent rule table plus compact evidence references, not an ever-growing unstructured transcript | Code storage + Astra summaries |
| Blind action batching after success | Interrupt immediately on metadata changes and predictions failing | Code |
| Optimizing easy-level replay score | High-water levels only; no duplicate credit; preserve enough budget to reach later levels | Code |
| Jev asked to calculate paths or invent rules | Limit to semantic choices with `unknown`; exact work in code, planning in Astra | Harness prompt design |

These counters combine the evidence above with the installed API constraints; none is a demonstrated optimal parameter setting.

## Astra prompt contract for the next phase

Give Astra the current grid/image, numbered action mask, exact progress/state, recent actions with measured effects, learned rules and counterexamples, and remaining **external** budget. Include a compact object table but allow raw-grid inspection. A useful response contract is:

1. Observed facts, distinct from inferred mechanics.
2. Current goal hypothesis and confidence.
3. Next action or short conditional plan, using validated IDs and supplied click candidates.
4. Expected effects and the evidence that would falsify the plan.
5. Stop conditions and concise memory updates.

Standing prompt principles: the world waits; treat each unfamiliar environment as new; test rules instead of naming genres; retain working explanations until contradicted; revise the specific failed assumption; transfer rules across levels without transferring old coordinates. Ask Astra for decisions at uncertainty or plan boundaries, not automatically for every routine move.

## Jev typed-question examples

Jev is not required for baseline play. If enabled, provide JSON with an `observation_id`, candidate object IDs, measured shape/position/color features, and relevant before/after summaries. Batch independent questions against the same observation. Good questions include:

| Question | Return type | How to use it |
|---|---|---|
| “Which listed region most resembles a target example?” | Choice: supplied region IDs or `unknown` | Astra considers it as a goal cue |
| “Which description fits the supplied transition?” | Choice: `translation`, `rotation`, `recolor`, `selection_change`, `unclear` | Route evidence to the corresponding rule hypothesis |
| “Does this candidate look like an interactive control?” | Choice: `yes`, `no`, `unknown` | Soft candidate priority only |
| “Does the provided observed event match this expected event category?” | Choice: `match`, `mismatch`, `unclear` | Escalate semantic mismatches to Astra |
| “How visually plausible is this component as a status indicator?” | Ordinal score 0–3 with a separate unknown option | Suggest a region label; never discard its pixels automatically |

Do not ask Jev to count pixels, calculate distances, decide whether a path fits an energy budget, infer a new transformation algebra, or pick the strategically best move from raw state. Deterministic checks such as WIN, score changes, bounds, and frame equality need no model. Discard responses with stale observation IDs and validate enum outputs.

## Harness-search priorities

Start with a faithful adaptor and a single Astra loop with persistent structured memory. Next compare exact perception summaries versus full-grid context, one-action probes versus short interrupted batches, and graph fallback versus model-based plans. Add Jev only if it demonstrably reduces total cost while preserving held-out level completion. Evaluate all candidates with identical game-version splits and action/wall-time budgets, reporting per-game levels and actions as well as aggregates.

Do not select a universal action budget from old competition results. More exploration can be worthwhile under the user's level-count objective, but endless no-op search still consumes evaluator resources. Use stagnation-triggered replanning, then controlled expansion or a justified retry. Treat learned within-episode models as temporary state, and do not retain held-out solution traces in later candidates' priors.
