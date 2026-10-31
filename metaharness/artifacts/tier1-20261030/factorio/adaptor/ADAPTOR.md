# Factorio adaptor

This is a Tier 1 remote factory manager running the **real, official Factorio 2.0.72 Linux headless engine**. `game.py` implements `TurnGame`, and `bridge/control.lua` is a trusted control mod. No Factorio Learning Environment, graphical client, account, expansion, Docker, Python dependency or model API is needed. The engine is zero-cost **proprietary** software, not libre software; this follows the user's explicit preference for the free headless server. The adaptor code is MIT licensed. Upstream terms are at https://factorio.com/terms-of-service (staged in `runtime/terms-of-service.html`); third-party engine licenses ship in `runtime/factorio/data/licenses.txt`.

## Setup and offline use

Run `./setup.sh` once with network access, or stage the archive listed in `engine.sha256` under `runtime/` before running it offline. The script downloads an exact version, verifies its SHA-256 and extracts it locally. The archive and executable are already staged on this machine. Subsequent episodes never download anything. Python is `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python`; no packages need installing (`requirements.txt` documents this).

All environment variables are optional and documented in `env.example`. Defaults use the staged executable and this directory's `runtime/`. An optional `FACTORIO_RCON_PASSWORD` is read from the environment; otherwise a fresh cryptographic local secret is generated in memory per episode. There are no hard-coded credentials. Each instance binds RCON and multiplayer exclusively to loopback, chooses ephemeral ports, disables public/LAN listing, online verification, updates, autosaves and empty-server automatic pause. No human client connects. Set no required environment variables to use the default installation.

Each reset creates a new save, force, surface, RNG, inventory, research state, counters and job queue in a unique `runtime/episode-*` directory, then starts one server process. Base and `mh-bridge` are explicitly enabled; space-age, quality and elevated-rails are explicitly disabled and checked by the mod. Close disconnects RCON and terminates/waits for the server, with a kill fallback; it is idempotent. Logs, saves and other generated artifacts remain in `runtime/` for diagnosis. The Lua source is copied into the runtime mod directory, never generated over pinned source files.

## Scenarios

Only **cold-coast-v1** is implemented. There are no pooled repair, later-start, or combat scenarios. All integer seeds are supported; explicit disjoint train/heldout/dev/validation seed lists are in `game.py`. Integers map to engine seeds by the first four bytes (big-endian) of SHA-256 of `mh-factorio-cold-v1:SEED`, with zero replaced by one. Hashing avoids nearby small seed aliases in the engine RNG.

This is a deterministic **custom scenario generator**, not Factorio's default freeplay terrain distribution. The engine's seeded RNG selects coast orientation, position and sinusoidal shoreline, a permutation of ore quadrants, patch-center distances (30–65 tiles), angles, elliptical radii (7–11 by 6–10 tiles), irregular patch edges, amounts (1,800–3,200 units per tile), and tree positions. All deposits, tiles, trees and machines are genuine engine entities. The revealed/buildable square is `[-128,128)` on both axes. Real finite iron, copper, coal and stone each exceed 200,000 units. A clear central area and open corridors make all patches accessible. There is coastal water, wood, and a pump placement checked by `LuaSurface.can_place_entity`. Generator invariants avoid missing resources by construction. On a failed resource/pump validation, reset retries up to three times, adding `0x9E3779B9` modulo `2**32-1` to the previous map seed each time. `map_attempt` and the actual map seed are observed. No retry has been needed for the declared seeds. Seed changes affect physical constraints, not just labels.

Initial stock is fixed: one burner mining drill, one stone furnace, eight iron plates and one wood, in a 200-slot engine inventory. There is no initial factory, research bonus, pollution or enemy generation. This bounded early-game family intentionally supplies no oil or uranium; red/green science and applicable research are its attainable science objectives. Later science weights are retained in the required score formula but have no attainable production chain in this family.

## Turns and actions

The engine starts paused at tick zero. Every accepted gameplay batch executes up to 32 typed operations, then runs **exactly 1,800 ticks**, stopping in the mod's tick handler. There are **128 turns**, totaling 230,400 charged ticks (64 simulated minutes). Simulation speed is 20; wall-clock thinking delay has no effect. No final uncharged interval is run. RCON remains usable while paused. Requests carry monotonically increasing turn IDs; resubmitting an acknowledged turn returns a duplicate acknowledgment without replaying operations. Lost connections or server failures invalidate the trial for evaluator retry.

Actions are strings; join operations with semicolons. All grammar and mechanics are supplied to the model by `describe()` and `action_spec`. Examples:

```
mine_resource 42 10; place stone-furnace 0 0
insert 1200 fuel coal 5; insert 1200 source iron-ore 20
extract 1200 result iron-plate 10; handcraft iron-gear-wheel 2
place burner-mining-drill 25.5 -30.5 4
set_research automation
wait
recipe lab
inspect 100
```

IDs must come from the actual observation. Directions are north/east/south/west = 0/4/8/12. Manual mining/handcrafting is one serial worker using actual prototype mining times and vanilla character mining speed, or recipe duration at crafting speed 1. Its queue is limited to 32 jobs and 3,600 remaining ticks. Crafts reserve real ingredients immediately; only enabled character-craftable recipes with deterministic item products are supported. Outputs blocked by stock capacity wait without loss. Ore mining reduces real deposits; tree mining uses the engine's normal products.

Insert/extract share a quota of **100 actual items per turn**. Placement requires stock and engine collision/terrain checks. Invalid per-operation attempts report errors and still consume that batch's turn. Malformed strings are rejected without advancing. Unknown/stale entities, invalid counts, and unsupported operations never become arbitrary Lua execution. Read-only `inspect OFFSET`, `recipe NAME` and `technology NAME` queries must be alone and charge no turn.

Two conservative differences from the proposed study interface are intentional: buildings must be empty before mining, and assemblers must be empty and idle before changing recipe. Cargo must first be extracted under the transfer quota. Mining loses remaining fluids and burning fuel, matching ordinary engine behavior. This prevents cargo/quota bypass and partially completed recipe duplication. Native craft-item research triggers are mirrored for **paid, completed manual crafts** with prerequisites satisfied (particularly crafting the first lab); no production counters are forged. Other production/research remains engine-driven.

## Observations and score

Observations are strict JSON, checked below 200 kB. They include paused tick/turn/budget; exact stock, jobs, manual outputs, transfers and action results; inclusive water x-runs by y (other terrain is grass); ore tile rows `[id,name,x,y,amount]`, tree rows `[id,x,y]`, patch summaries and a validated pump site; paged buildings with bounding boxes, status codes, inventory roles, recipe/progress, fluids, fuel, stored joules and network ID; enabled recipe ingredients/products; available research including native triggers; completed research and events; raw normal-quality engine production/consumption and audited lab consumption. Building pages hold at most 100 entities, reduced as needed to stay below 190 kB; `entity_page_count` gives the actual page size for choosing the next offset. Read-only queries expose locked recipe/technology details. Observation is a consistent cached snapshot of the paused engine, not an advancing live view.

`P` is cumulative produced items from the player force's **per-surface** `get_item_production_statistics(surface).get_input_count({name,quality='normal'})`, minus the reset baseline. `get_output_count` is consumption. `L` is science consumption sampled from those engine output counters while owned laboratories exist. In this base-only scenario science packs have no alternative allowed recipe sink; script removals/transfers do not affect counters. Powered-lab integration checks verify research advances alongside consumption. Manual output is tracked separately and never passed to `on_flow`. Manually crafted packs earn credit only when genuinely consumed by a lab.

```
B = .01*min(P(iron-plate),1000) + .01*min(P(copper-plate),1000)
  + .02*min(P(iron-gear-wheel),200) + .05*min(P(electronic-circuit),200)
score = B + Σ weight(i) * (P(i) + 4*L(i))
```

Weights for automation/logistic/chemical/production/utility/space science are 1/2/6/12/12/30. Bootstrap caps at 34. This is the requested benchmark score, not an official vanilla or FLE score. Initial stock, transfers, mining/rebuilding and virtual manual outputs do not increase `P`. `score_breakdown()` returns numeric components. `info()` reports validity, infrastructure failures, engine/family, runtime location, charged ticks and wall time. The model sees the raw economic/research ledgers in observations.

## Verification

`verify.py` exercises the actual staged engine, including privileged test fixtures that **are not part of the game action space**. It writes results to `runtime/verification.json` and `runtime/verification.log`. It checks a 60-second paused interval by querying the server twice, exact 1,800-tick steps, furnace production, manual/transfer exclusion, duplicate-step handling, powered assembler and lab production/consumption, malformed and illegal actions, physical differences and feasibility across all declared seeds, deterministic reset, random actions and crash invalidation.

Run the generic check from the repository root:

```
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/factorio/adaptor', expected_tier=1).to_markdown())"
```

Current test evidence is recorded in runtime files. Infrastructure failures are not silently converted into valid zero-score episodes.

### Recorded results on this machine

- Generic validator: **PASS**, including deterministic fresh-instance reset. An additional run with `max_actions=20` covers the entire declared action vocabulary; no warnings were reported.
- All **14** declared scenario seeds have distinct physical-layout SHA-256 fingerprints and validated resources, trees and pump sites (`runtime/verification.json`).
- A fueled stone furnace produces **9 plates in 1,800 ticks**, worth **0.09**. The powered assembler/lab fixture produced two red packs and consumed one in real research; combined with 20 furnace plates, its score was **6.2**.
- The public-action-only bootstrap (`verify_play.py`) reaches **180 iron plates, 30 copper plates, a lab, steam-power components and five manually crafted red packs in 50 turns**, scoring **2.1**. Native smelting triggers and the paid manual lab-craft trigger unlock the expected technologies. The manual packs add no production score.
- Inventory checks (`runtime/economy.log`) verify loaded-building rejection, empty-building recovery, the shared 100-item transfer quota, and exact preservation of a partially used science pack across extraction/reinsertion. Transfers copy engine stack metadata, not just item counts.
- The engine's virtual `science` production counter is included in the raw diagnostic ledger but never scored; only the named science-pack counters enter the formula.

Additional timing/horizon checks **passed** and are recorded in `runtime/clock-verification.json`: speed/inference-delay invariance, exact manual-worker time, and termination after 128 turns / 230,400 ticks with no post-horizon advance. `artifact-manifest.json` records the final source, engine archive and API-document hashes. All verification servers are closed after their checks.
