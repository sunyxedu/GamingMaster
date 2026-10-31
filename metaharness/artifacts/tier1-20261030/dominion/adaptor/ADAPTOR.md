# Dominion offline adaptor

`game.py` exposes a Tier 1 `TurnGame` backed by pyminion 0.5.0. The game and both players run locally; no account, GUI, network service, credentials, or runtime download is used. `reset(seed)` selects a fixed scenario and reproducible shuffle stream. Each accepted decision resumes the engine through the next agent prompt or the end of the game.

## Install and provenance

Run `./setup.sh` from this directory. It installs the included wheel offline into the supplied Python (by default the evaluator's Dominion venv) and checks dependencies. `env.example` documents that no environment variables or secrets are needed.

- Engine: pyminion distribution 0.5.0, MIT license.
- Upstream commit: `d51e94604894fd76eecfbbd4fe74d1fe875c0be1`.
- Vendored source and tests: `vendor/pyminion/`; original archive: `vendor/pyminion-source.tar.gz`.
- Wheel: `vendor/wheels/pyminion-0.5.0-py3-none-any.whl`, SHA-256 `15dc35d85f00e986a6638dd7f38370246fc2a42a77371ea40aaf00134465dd79`.
- Adaptor/manifest version: 2.0.0. `game_v1.py` is retained from the previous adaptor. `game.py` is the current public entrypoint; the engine bridge and tested repairs are in `worker.py` and `engine_patches.py`.

The wheel metadata reports 0.5.0. Its package-level `pyminion.__version__` is stale and says 0.3.0; use distribution metadata for the installed version.

## Opponent and measured strength

Every declared game scenario uses `BMU-1`, pyminion's built-in `BigMoneyUltimateDecider` from the pinned source. It plays Smithy when available. Its first affordable, nonempty purchase is Province when total deck Treasure money exceeds $15; Duchy when fewer than five Provinces remain; Estate when fewer than three remain; Gold; Duchy when fewer than seven remain; Smithy when owned Smithies are fewer than owned Treasures divided by eleven; Silver; otherwise it stops. It uses pyminion's default responses to forced choices and has no access to the agent's hidden zones.

We measured it on the actual patched engine over 50 shuffle seeds per prior train/held-out kingdom/policy matchup, pairing both player seats (600 complete games total). Against the previous adaptor's policies, BMU-1 won 299/600 games (49.8%); including half credit for draws, win score was 52.3%, and mean VP margin was +0.19. Results by previous opponent were:

| Previous opponent | Games | Raw wins | Win score (draw = 0.5) | Mean VP margin |
|---|---:|---:|---:|---:|
| BM-1 | 200 | 147 (73.5%) | 75.8% | +4.95 |
| SM-1 | 200 | 102 (51.0%) | 55.3% | +0.70 |
| WM-1 | 200 | 50 (25.0%) | 25.8% | -5.08 |

This shows a clear weakness against early Witch pressure; the combined figure should not hide it. As a cross-check against the vendored example bots, the same BMU-1 policy won 1,701/2,000 games (85.1%) across eligible tested kingdoms against pyminion's `BigMoney`, `BigMoneySmithy`, and `ChapelBot` deciders, with a +7.45 mean VP margin. The batch is a reproducible local strength check, not a claim about expert human play. `OptimizedBotDecider` is an abstract strategy helper without buy/action priorities, not another complete AI. The Bandit example requires an unsupported Bandit pile and is excluded. Among the applicable complete built-ins, BMU-1 beat each alternative individually: win scores of 95.56% versus BigMoney (800 games), 79.44% versus BigMoneySmithy (800), and 80.13% versus ChapelBot (400). These are head-to-head measurements, not a global optimality claim.

Raw per-game reports are `runtime/previous-opponent-benchmark.json` and `runtime/builtins-benchmark.json`. Each records per-kingdom, per-opponent and per-seat aggregates. To reproduce from this directory, run the evaluator Python on `benchmark_opponents.py --pairs 50 --suite previous --output runtime/previous-opponent-benchmark.json` or use `--suite builtins --output runtime/builtins-benchmark.json`. Benchmark seeds start at 900000 plus 10000 times the kingdom's manifest index; each seed is paired across both seats. These seeds are independent of the evaluator splits and cannot be passed to public `reset()`.

## Rules and supported scenarios

Two players start with 7 Copper and 3 Estate, draw five, and use normal two-player piles and exactly ten tested Base kingdom piles. Each turn starts with one Action, Buy, and $0. Ordinary Copper/Silver/Gold are automatically played after Actions. Buying spends money and a Buy; card-effect gains do not. Turns finish before Province/three-pile endings are checked. Final VP counts all owned cards, including Curses and Gardens (`floor(total cards / 10)` per Gardens). Higher VP wins; a tie goes to the player who took fewer turns, otherwise it is a draw.

The supported kingdom cards are Cellar, Chapel, Festival, Gardens, Laboratory, Market, Merchant, Militia, Mine, Moat, Remodel, Sentry, Smithy, Village, Witch, and Workshop, plus basic Supply cards. All four named study fixtures are included exactly. Two held-out kingdom combinations (`heldout_base_draw`, `heldout_chapel_sentry`) add draw/thinning variation while keeping to the tested card whitelist. Unsupported cards and seeds fail explicitly; no kingdom is silently substituted.

Scenario assignments are in `scenarios.py`. For each even seed, the following odd seed uses the same shuffle seed and the opposite agent seat; seat 0 goes first. The pairs and kingdom configuration are:

| Split | Seed pairs | Kingdoms |
|---|---|---|
| train | 1100/1101, 1200/1201, 1300/1301, 1400/1401 | base_intro, chapel_engine, curse_race, gardens_rush |
| heldout | 2100/2101, 2200/2201, 2300/2301, 2400/2401 | heldout_base_draw, heldout_chapel_sentry, heldout_witch_gain, heldout_thin_gardens |
| dev | 3100/3101 | gardens_rush |
| validation | 4100/4101, 4200/4201, 4300/4301, 4400/4401 | base_intro, chapel_engine, curse_race, gardens_rush |

All splits are disjoint. Train and held-out use different kingdom combinations as well as shuffle streams; each has eight seeds and both seats. The engine's process-local Python RNG is initialized from the pair seed before setup and supplies later shuffles in normal engine call order. A seat swap shares the initial seed but is not promised identical later draws after strategies diverge. Replay files record seed, scenario, seat, opponent, engine commit, adaptor version, and accepted actions; the seed is never shown in observations.

## Actions, observations, and scoring

The action vocabulary is `choose`, `pass`, and `wait`. Use `choose DECISION_ID OPTION_ID ...` for ordered selections of distinct option IDs within the prompt's `min` and `max`; `choose DECISION_ID` or `pass` declines only when zero choices are allowed. A JSON object with `decision_id` and `option_ids` is also accepted. `wait` leaves the pending decision unchanged. IDs refer only to options in the current prompt. Invalid, stale, malformed, or duplicate selections do not advance the engine and are reported as `last_error`; no fallback is made.

Prompts include Action play, buy, trash, discard, gain, yes/no reactions, effect ordering, and the nested choices needed by the whitelist. They also occur when the opponent attacks. Forced sole answers resolve automatically. Ordinary Treasures are played automatically; there are no supported Treasure-order interactions.

Observations are compact strict JSON: seat and active player, phase and turn counts, current resources, exact kingdom/Supply card text and counts, the agent's hand/discard, public play zones and ownership/VP ledgers, opponent public counts and ownership, trash, public event changes, derived pile/VP state, and the pending prompt. They do not expose either unrevealed deck order, the opponent's hand, RNG state, reset seed, or private engine logs. Repeated observation is stable. The largest observation in the stress run was under 5 KB.

`score` is the integer VP margin as a float. `info()` separately reports both final VPs, margin, outcome, `win_score`, turns, ending reason, truncation, illegal/fallback counts, scenario and replay metadata. `win_score` is 1/0.5/0 for win/draw/loss. `score_breakdown()` reports the numeric score components. A 400-turn or 20,000-decision safety limit marks a truncation; it is not a Dominion ending and must not count as a win/loss sample. Engine failure marks `valid_trial=False` with a reason.

## Engine repairs and limits

`engine_patches.py` narrowly repairs four upstream edge cases used by the whitelist: duplicate Moat reactions; Remodel/Mine trash-before-gain timing; Sentry's inspected-card ownership and top-first ordering; and Workshop when no eligible pile remains. The engine continues to own turn flow, card resolution, resources, piles, shuffles, termination, and winner calculation. Additional regression tests check starting decks and pile sizes, all enabled cards, nested attacks, Gardens/Curses, gain endings, tie-breaking, hidden-zone privacy, deterministic replay, malformed actions, resets, truncation, and process cleanup.

This is a Base-card simulator, not a live digital client. Throne Room, Bandit, Library, Artisan, Vassal, expansion cards, alternate starting decks, and unsupported kingdom cards are not enabled. Public ownership is tracked from observable engine events; hidden order is intentionally not inferred. `close()` terminates and reaps the worker and closes all streams; it is safe to call repeatedly.

## Verification

Verified with `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python`:

- Generic `validate_adaptor(..., expected_tier=1)`: **PASS**, no warnings; includes two seeds, action checks, malformed action, determinism, and cleanup.
- Vendored pyminion tests plus adaptor tests: **596 passed**.
- Offline money-policy and randomized legal-action batches: 26 games each, all completed without invalid trials, illegal choices, fallbacks, or truncation. Maximum observed JSON was 4,515 bytes for money play and 4,378 bytes for random play.
- Score headroom: the public-observation SM-1/WM-1 smoke policy averaged 48.08% win score and 0.00 VP margin; random play averaged 0% and -70.31. Neither saturates the objective across the scenario suite.
- Opponent comparisons: 600 games against the previous adaptor's policies and 2,000 against the vendored example bots, with both seats paired and no truncations.

To rerun the generic smoke test from the repository root:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/dominion/adaptor', expected_tier=1).to_markdown())"
```

From this directory, reproduce the conformance suite with:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python - <<'PYTEST'
from engine_patches import apply
import pytest
apply()
raise SystemExit(pytest.main(['-q', 'vendor/pyminion/tests', 'test_adaptor.py']))
PYTEST
```

Run `evaluate.py` and `evaluate.py --random` with that interpreter for full public-interface episodes. Replay a current-version episode with `replay.py runtime/episode-.../replay.jsonl`. Replays from the previous adaptor version are explicitly rejected because its manifest and opponent differ. Runtime artifacts are the only files written during play.
