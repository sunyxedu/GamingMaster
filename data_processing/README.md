# Data pipelines

All data pipelines — fetch, filter, replay/simulate, convert — are Rust. They run
on macOS and Linux with a stable Rust toolchain (`https://rustup.rs`); nothing
else is needed at conversion time. Training reads the output from Python.

```
data_processing/
├── Cargo.toml                 workspace
├── rs/gmshard/                gm-shard.v1 writer / reader / overlay writer (.npy + index.json)
├── rs/gmrules/                rules + observation / candidate encoders of the gmrules games,
│                              shared by converters and simulators (no I/O; see "Adding a game")
├── rs/gmsim/                  PyO3 extension: gmrules simulators / replay for Python (uv sync)
├── rs/gmdata/                 `gmdata <game> fetch|convert|…` for every game below,
│                              `gmdata label <game>` engine teacher labels (src/label/),
│                              `gmdata rules …` generic tools for every gmrules game
├── games/clash_royale/crdata/ `crdata` (Clash Royale replays)
├── games/dataset.py           GameShardDataset: reads gm-shard directories (and .pt)
├── games/clash_royale/        card catalog + load-time observation expansion
├── dataset_seq.py             SeqReplayDataset for generals gio records
├── download/ processing/ parsers/   generals reference decoder/simulator/converter
│                                    (viewers, play_model; the Rust pipeline is
│                                    tested bit-identical to them)
└── games/othello|gomoku/rules.py    rules used by the RL self-play environment
```

```sh
cd data_processing && cargo build --release
G=target/release/gmdata        # add --help to any subcommand
```

## Output format

`gm-shard.v1` directories: one `.npy` per tensor (`obs.*`, `tgt.*`, frame-major,
samples are frame ranges) plus `index.json` (game, converter version, shard meta,
per-sample meta). One-hot planes are stored bit-packed and unpacked on load.
`GameShardDataset(dir, game=...)` memory-maps them directly — no conversion to
`.pt`. It refuses to mix converter versions, shard meta or key sets. generals.io
keeps its historical per-game gio JSON (`SeqReplayDataset`, `training/train_il.py`).
For the multi-game trainers generals reads gm-shards made from them —
bit-packed, memory-mapped, ~10x smaller, ~200x faster to load, records shuffled
across shards (`--seed`, `--keep-order` for path order), and pinned
tensor-for-tensor to `SeqReplayDataset` by `tests/test_rust_generals_parity.py`:

```sh
$G generals shard --input data/generals/training --out data/generals/shards
```

Small corpora can skip the conversion: the gio directory itself is read by
`games/generals.py::GeneralsGioDataset` (same decoding, but a whole-record JSON
parse per draw and a first-run length scan over every record).

## Games, sources and the high-quality recipe

Imitation learning here uses only high-quality play. Every converter has quality
filters and a `--hq` preset; the defaults reproduce the previous (unfiltered)
behaviour, which is what the golden parity tests pin.

| game | source (fetch) | high-quality recipe | notes |
|---|---|---|---|
| generals.io | `$G generals ladder --out D --min-stars 70` (global duel ladder) | winner ≥ 70★ (or `--both`), human accounts only (`isBot` check), ≥ 50 turns | per-account `generals pipeline` (= `scripts/prepare_data.py`) still exists; its default account Human.exe is a bot |
| chess | `$G chess fetch --month 2021-06 --elite` (Lichess Elite DB) or `--month` (full dump) | `convert --hq`: both Elo ≥ 2200, rated, normal termination, no bullet, ≥ 20 plies | elite months are pre-filtered 2400+/2200+ (2500+/2300+ since 2021-12) |
| go | `$G go fetch --url https://dl.u-go.net/gamerecords/KGS-…tar.bz2` | `convert --hq`: both ≥ 6d, no time-loss/forfeit, no handicap, ≥ 50 moves | KGS archives already need one 7d or both 6d |
| shogi | `$G shogi fetch --day 2025/06/01 --ratings 20250601` or `--year 2023` | `convert --hq --ratings D/ratings-….json`: both engines ≥ 3500 on the floodgate table, floodgate-300-10F, no time-up | engine play |
| othello | `$G othello fetch --first-year 2000` (WTHOR) | `convert --hq`: \|recorded − perfect-play score\| ≤ 4 discs, year ≥ 2000 | human tournament games; the score bound measures endgame error |
| gomoku | Gomocup results zips (`https://gomocup.org/results/`) | `convert --top-engines 8` on one league (e.g. Freestyle1) | strength = win rate within the input |
| riichi mahjong | `$G mahjong fetch --shards 0 1 2` (Tenhou Houou mjai, 2.1M games) | `convert --hq`: hanchan only; optional `--top-placements 2` | phoenix room = top players already |
| Pokémon | `$G pokemon fetch --format gen9ou --min-rating 1500` | `convert --hq`: rating ≥ 1500, ≥ 5 turns, no inactivity losses; optional `--winner-only` | forfeits are kept (concessions affect only the last turn) |
| Lux AI S3 | `$G lux fetch --top-teams 10` (Kaggle API) | only episodes where both agents are top-10 leaderboard teams | needs `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` |
| Clash Royale | `crdata fetch --out D` (RoyaleAPI replays, HF) | `crdata convert --hq`: Ranked + Grand Challenge, ≥ 60 s, ≥ 5 own plays; optional `--winner-only` | anonymised (no trophies); see games/clash_royale/README.md |
| Kuhn poker (toy) | `$G kuhn-poker generate` (synthetic hand logs; `scripts/fetch_kuhn_poker.sh`) | `convert --hq`: the scripted expert's seats only | gmrules reference game (below); rules pinned to OpenSpiel |

Measured on real data (kept / seen): chess elite 2021-06 601k / 781k games;
go KGS 2019-04 202 / 1255; shogi 2025-06-01 158 / 447; othello 2019–21
1502 / 3559; gomoku Freestyle1 2023 658 / 2880; Clash Royale 5000 replays
8456 / 9392 player sequences (4186 with `--winner-only`).

## Engine teacher labels

`gmdata label <game>` adds engine soft targets (teacher move distributions per
decoder head) and engine values to existing gm-shards, as overlay shards that
hard-link the base tensors (the base is never modified) and that the trainers
read unchanged (`--soft_target_weight`, `--engine_value_weight`). Engines are
external processes installed outside the repo by
`scripts/engines/fetch_engines.sh` (Stockfish, Egaroucid, YaneuraOu, KataGo;
licences in its header). Columns, options and the per-game extension point are
documented in `rs/gmdata/src/label/mod.rs`; `gmdata label selftest --out DIR`
exercises the whole pipeline with a built-in fake engine.

```sh
scripts/engines/fetch_engines.sh stockfish
$G label chess --input data/chess/train --out data/chess_sf/train \
   --engine ~/gm-engines/bin/stockfish --workers 32     # defaults: nodes=120000, MultiPV 8
scripts/engines/fetch_engines.sh egaroucid
$G label othello --input data/othello/train --out data/othello_ega/train \
   --engine ~/gm-engines/bin/egaroucid --workers 64   # Egaroucid level 15, exact <= 20 empties
```

The teacher policy is `π ∝ exp(Q/τ + w·u)` over the scored moves (`--tau`,
default 0.05; `--tie-weight` w, default 2.0). Q (expected score) saturates in
decided positions — mates and big advantages give Q = 1 (or 0) for every move,
exact othello solves Q ∈ {0, ½, 1} — where `exp(Q/τ)` alone is uniform and a move
that allows mate counts as much as the best defence. u ∈ [−1, 1] is a secondary
utility of the engine's native score (`tgt.teacher_raw`), increasing like Q, so
moves are ordered by Q and, among equal Q, by u:

| game | u |
|---|---|
| chess | mate in n moves: ±(0.5 + 0.5·5/(n + 4)); else 0.5·tanh(cp / 1000) |
| shogi | mate in p plies → n = (p + 1)/2 (mating) or p/2 (mated) moves, same mate formula; else 0.5·tanh(cp / 3000) |
| othello | tanh(discs / 16) (exact and midgame scores) |
| go | none (the default policy is KataGo's visits) |

Every mate ranks above every non-mate score; within one Q class the preference
ratio is at most e^(2w). `--tie-weight 0` gives `π ∝ exp(Q/τ)` exactly; the
teacher meta records `policy.tie_weight` and `policy.secondary`, so resumed runs
with other settings are refused. Details and measurements: `rs/gmdata/src/label/transform.rs`
and the per-game module docs.

Per-game defaults, action-id encodings and measured throughput are in the
module docs of `rs/gmdata/src/label/<game>.rs` (chess: `tests/test_label_chess.py`
runs the CLI end to end with a fake engine, and with Stockfish when installed).

## gmrules games: one rule engine for converter and simulator

New games are written ONCE, in Rust, in `rs/gmrules` (a `Game` trait: turns, chance
events, canonical action ids, outcome, and the observation / candidate encoders).
The same code serves

- the converter (`gmdata <game> convert`): source records are parsed into
  trajectories (chance outcomes + seat actions as canonical ids) and REPLAYED by
  the rules — every event verified, recorded payoffs checked — and every decision
  frame is encoded by `gmrules::encode_frame` (generic driver `rs/gmdata/src/cand.rs`);
- the simulator (`gmsim`, a PyO3 extension built by `uv sync`, see
  `rs/gmsim/README.md`): `env/vec_candidate.py` wraps its batched `VecSim` as the
  RL `VecGameEnv` — it encodes with the same `encode_frame`,

so IL frames and RL frames are identical by construction (and pinned by golden
tests). The toy reference game `kuhn_poker` (`rs/gmrules/src/games/kuhn_poker.rs`,
2-4 players, hidden cards) goes through every layer; `goofspiel` (simultaneous
moves) and `pig` (perfect information + dice, interleaved stream) are test games of
the other two layouts; all three are pinned to OpenSpiel.
`gmdata rules list|check|random|convert` work for every
registered game without per-game code (conformance battery, random-play shards,
trajectory JSONL → shards).

Candidate-action schema (`candidate.v1`; model: `model/games/candidate.py`): the
legal actions of a decision are enumerated as candidates with their own features.

```text
obs.entities    [T, E, Fe] f32      obs.entity_mask [T, E] bool   obs.entity_type [T, E] i32 (optional)
obs.globals     [T, G] f32
obs.actions     [T, A, Fa] f32      obs.action_mask [T, A] bool   (legal candidates, ascending ids)
tgt.action_idx  [T] i64             index of the taken action among the candidates
tgt.value       [T] f32             outcome of the frame's seat in [-1, 1]
```

Shard meta = the game's `FrameSpec` (E, Fe, entity types, G, A, Fa, n_seats,
layout) + the game config + `rules_version` (= `converter_version`); a fresh
training run takes the model's frame dims from it. Layout: hidden-information and
simultaneous games use PER-SEAT streams (one sample / RL row per seat holding only
that seat's own decisions, so no temporal state ever carries another seat's private
information); perfect-information alternating games use one interleaved stream
per game (like the board games). Tensors whose values are all 0/1 can be stored
bit-packed (`FrameSpec::bits_*`).

### Adding a game

1. Rules — `rs/gmrules/src/games/<game>.rs` implementing `gmrules::Game`:
   - canonical `u32` action / chance-outcome ids (take the reference
     implementation's ids when there is one: differential tests compare them
     directly); `legal_actions` strictly ascending (= the candidate order);
   - a `FrameSpec` with fixed maxima (entities, candidates — size `max_actions` for
     the worst legal position of the game — `max_decisions`); keep entity slots
     stable across frames (slot = identity: card, seat, zone);
   - `encode_obs(seat)` reads ONLY what the seat knows now (public state, its own
     private state and actions); `outcome()` in [-1, 1] is both the IL value target
     and the RL reward; `returns()` the raw payoffs; `check_config` validates the
     config; hidden-information games implement `perturb_hidden` (re-deal what
     `seat` cannot see, keep every recorded action legal);
   - register: `pub mod <game>;` in `games/mod.rs` + one line in
     `registry::GAMES`;
   - unit tests: hand-checked rule examples + `invariance::check_game::<G>(&cfg, n,
     seed)` for every variant (causality, privacy, frame conformance, simulator
     replay — see `games/kuhn_poker.rs`).
2. Parity with an independent reference (OpenSpiel, PokerKit, gnubg, DouZero,
   Fairy-Stockfish, ...): a dump script run in a throw-away environment (pattern:
   `scripts/reference/openspiel_dump.py`, `uv run --no-project --with open_spiel`)
   recording per step the turn, every acting seat's legal actions, the chance
   outcomes + probabilities and the final returns; a small committed fixture plus a
   live test (`GM_REFERENCE=1`) in both directions — reference playouts stepped
   through `gmsim.State`, and gmrules' random playouts replayed through the
   reference (`tests/test_candidate_reference.py`). Real records replayed by the
   converter are the second proof: `illegal` / `returns_mismatch` must stay ~0.
3. Converter — `rs/gmdata/src/<game>.rs`: `fetch` (download; or a
   `scripts/fetch_<game>.sh`), a parser from the source format to `cand::Record`s
   (events, meta, optional seat filter for the high-quality recipe, the recorded
   payoffs), `cand::convert::<G>`; flags `--hq`, `--store-events` (golden tests) and
   `--allow-truncated` (causality tests); one `Game` variant in `main.rs`. Template:
   `kuhn_poker.rs` and `scripts/fetch_kuhn_poker.sh`. Record the data licence in the
   table above.
4. Python — `model/games/candidates/<game>.py` with
   `GAME = CandidateGame(name="<game>", sim_config={...})`. Discovery does the rest:
   `build_game_network`, train_il_game `LABEL_KEYS` / `META_ARCH_KEYS`,
   `make_vec_env` (env/vec_candidate.py) and train_rl_joint `GAME_SPECS` (the generic
   `CandidateSpec`). Game-specific encoder / decoder subclasses or an RL spec go in the
   same entry (`encoder=`, `decoder=`, `config_class=`, `rl_spec=`).
5. Tests (template `tests/test_candidate_pipeline.py`): converter frames ==
   `gmsim.replay` frames of each record; simulator frames == converter frames of the
   simulator's own episodes; `tests/invariance.py` — `check_sim_battery` and, on the
   real converter binary, `check_converter_causality` (truncated records) and
   `check_converter_privacy` (opponents' private information changed); a tiny
   `train_multi_game` + `train_rl_joint` run.

Joint pretraining with very large corpora: `train_multi_game --mixture_frame_cap N`
(per game `--game_frame_cap GAME=N`) caps the frames a game counts with in the
mixture weights w_g ∝ m_g·min(N_g, C_g)^α, `--game_weight GAME=W` sets m_g — so a
1e10-frame corpus cannot swamp the mixture.

## Tests

```sh
cargo test --release                                   # Rust unit tests
uv run pytest tests/test_rust_converters_parity.py \
              tests/test_rust_generals_parity.py tests/test_clash_royale_data.py \
              tests/test_label_overlay.py tests/test_fetch_engines.py \
              tests/test_label_chess.py tests/test_label_othello.py \
              tests/test_label_shogi.py tests/test_label_go.py
uv run pytest tests/test_candidate_*.py               # gmrules / gmsim / candidate adapter
GM_REFERENCE=1 uv run pytest tests/test_candidate_reference.py   # + live OpenSpiel diff
```

Real-engine tests run only when the engine is installed (e.g. `STOCKFISH=path`
or `~/gm-engines/bin/stockfish`, `GM_EGAROUCID=path` or `~/gm-engines/bin/egaroucid`,
`GM_YANEURAOU`, `GM_KATAGO`); the label tests otherwise use fake engines.

The parity tests pin every converter to golden SHA-256 digests of the retired
Python reference outputs on committed real-data fixtures (tests/fixtures/).
