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
├── games/pokemon/             exact Showdown move / species vocabulary (showdown_vocab.json,
│                              embedded by `gmdata pokemon convert`; make_vocab.py regenerates it)
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
| generals.io | `$G generals index --out D` saves the whole duel-ladder index (every game's players, stars at the time, result, turns: ~1.75 M games from 2024-08 on; resumable), `$G generals ladder --out D --index D/index.jsonl --each --min-stars 75 --min-opponent-stars 50 --val-every 50 --shards D/shards` fetches and simulates the selected games straight into gm-shards (resumes by `D/manifest.jsonl`); without `--index` it pages the live index | `--each`: every player at ≥ `--min-stars` whose opponent has ≥ `--min-opponent-stars`, winners and losers (default: the winner ≥ 70★, `--both`: both); human accounts only (`isBot`; an unanswered lookup rejects the game), ≥ 50 turns; the thresholds from `scripts/generals_ladder_stats.py` (star difference vs win rate, opening land by star level) | per-account `generals pipeline` (= `scripts/prepare_data.py`) still exists; its default account Human.exe is a bot; a gio JSON record is ~15 MB, its shard sample ~1 MB |
| chess | `$G chess fetch --month 2021-06 --elite` (Lichess Elite DB) or `--month` (full dump); end to end over full monthly dumps `scripts/fetch_chess.sh D` (download, convert, delete; resumable) | `convert --hq`: rated, normal termination, ≥ 20 plies, no BOT accounts, both players at the measured per-speed Elo (blitz ≥ 2400, rapid ≥ 2200, classical ≥ 2000, no bullet: where Stockfish's average centipawn loss of the moves played is ≈ 40, `scripts/quality_by_rating.py`); `--speed-elo` other thresholds; `--val-every N` holds out 1 game in N by its Lichess URL (out/train, out/val); a game its tags reject is skipped unparsed; `--broadcast` reads Lichess's broadcast database (over-the-board events, **CC BY-SA 4.0**, `database.lichess.org/broadcast/`): FIDE ratings, FIDE speed classes of the free-text TimeControl, repeats kept once, `--hq` = both players classical ≥ 2000 / rapid ≥ 2200 / blitz ≥ 2400 FIDE (Stockfish ACPL 29.0 / 22.4 / 37.5), ≥ 20 plies | elite months are pre-filtered 2400+/2200+ (2500+/2300+ since 2021-12) |
| go | `$G go fetch --url https://dl.u-go.net/gamerecords/KGS-…tar.bz2` | `convert --hq`: both ≥ 6d, no time-loss/forfeit, no handicap, ≥ 50 moves | KGS archives already need one 7d or both 6d |
| shogi | `$G shogi fetch --day 2025/06/01 --ratings 20250601` or `--year 2023` | `convert --hq --ratings D/ratings-….json`: both engines ≥ 3500 on the floodgate table, floodgate-300-10F, no time-up | engine play |
| othello | `$G othello fetch --first-year 2000` (WTHOR) | `convert --hq`: \|recorded − perfect-play score\| ≤ 4 discs, year ≥ 2000 | human tournament games; the score bound measures endgame error |
| gomoku | Gomocup results zips (`https://gomocup.org/results/`) | `convert --top-engines 8` on one league (e.g. Freestyle1) | strength = win rate within the input |
| riichi mahjong | `$G mahjong fetch --all` or `--shards 0 1 2` (Tenhou Houou mjai, 2.12M games in 24 parquet shards; Tenhou's log terms: keep data and shards private) | `convert --hq`: hanchan only (by the announced `kyoku_first`), games written twice by two converters dropped, logs that break the rules dropped; optional `--top-placements 2` | phoenix room = top players already; exact Tenhou legality (yaku via RiichiEnv); full-scale command in `rs/gmdata/src/mahjong/mod.rs` |
| Dou Dizhu | no public human data: DouZero self-play, `scripts/fetch_doudizhu.sh D GAMES` (`scripts/generate_doudizhu.py`: the public DouZero ADP / WP agents, Apache-2.0 — weights from the pinned Hugging Face mirror `palemoky/douzero-baselines`, code from kwai/DouZero at a pinned commit, SHA-256 checked — play under the gmrules standard rules) → `$G doudizhu convert` | `convert --hq`: the seats the ADP / WP agents or PerfectDou played (every seat with the default mix; PerfectDou, `--agents adp,wp,perfectdou`, measured as strong as WP) | auction by the documented `heuristic_v1` (marked in `meta.bidding`; RL learns bidding); fixture 150 / 150 games verified, 5554 frames; 1M games ≈ 36M frames ≈ 28 GB at ~22 games/s per CPU worker; records carry DouZero's Q values (`--soft-k`); see `rs/gmdata/src/doudizhu.rs` |
| Pokémon | `$G pokemon fetch --format gen9ou --min-rating 1500` | `convert --hq`: rating ≥ 1500, ≥ 5 turns, no inactivity losses; optional `--winner-only` | forfeits are kept (concessions affect only the last turn); converter v3: one frame per request (lead, every turn, forced switches), move-identity targets, exact move / species ids from Pokémon Showdown's data (`games/pokemon/`) — see `rs/gmdata/src/pokemon.rs` |
| ARC-AGI-3 | recordings in the toolkit's `*.recording.jsonl`: ARC Prize's human baseline set (`arc_agi_3_public_demo_human_testing.zip`, 340 first-time sessions of the 25 public games) and agents' recordings (e.g. ARA Labs' trajectories on Hugging Face, CC BY 4.0) → `$G arc3 convert --input D --out O` | one sample per session (resets and game overs in one stream); frames = env/vec_arc3.py's for the same actions (tests/test_arc3_recordings.py replays human sessions through the env bit-exactly); `--games` / `--exclude-games` (held-out transfer games), `--split train\|val`, `--min-levels` | 339 / 340 human sessions, 180,144 frames, 144 won; the env's RESETs are level resets (competition rules), as in the recordings |
| Lux AI S3 | `$G lux fetch --top-teams 10` (Kaggle API) | only episodes where both agents are top-10 leaderboard teams | needs `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` |
| Clash Royale | `crdata fetch --out D` (RoyaleAPI replays, HF) | `crdata convert --hq`: Ranked + Grand Challenge, ≥ 60 s, ≥ 5 own plays; optional `--winner-only` | anonymised (no trophies); see games/clash_royale/README.md |
| Kuhn poker (toy) | `$G kuhn-poker generate` (synthetic hand logs; `scripts/fetch_kuhn_poker.sh`) | `convert --hq`: the scripted expert's seats only | gmrules reference game (below); rules pinned to OpenSpiel |
| backgammon | `scripts/fetch_backgammon.sh OUT MATCHES WORKERS` = `$G backgammon generate` (gnubg 2-ply self-play, 7-point matches; our engine runs the matches and checks gnubg's legal list at every move) + `convert` | engine play only (no human corpus): gnubg 2-ply both seats; `convert --soft-k 8 --engine-values` (gnubg's equities / match-winning chances) | gmrules game; rules pinned to gnubg (legal moves, its own match loop). Data: generated here, no third-party records; gnubg (GPL-3.0-or-later) runs as an external process in a throw-away uv env, never linked or vendored |
| Skat | `$G skat fetch --out D` (ISS game archive `iss-games-07-2024.sgf.bz2`, 545 MiB, 9.1M games 2007-2024, humans and programs, "available for free to everyone"; + the ISS ratings page); end to end `scripts/fetch_skat.sh D` | `convert --hq --players P --ratings R`: the seats of players with ≥ 2000 games and ≥ 5 money points per game in the corpus (`skat stats`) or ISS Elo ≥ 1050 (± ≤ 60); `--split train\|val` (1% val by game id) | gmrules game (ISS rules; every record replayed, its recorded result reproduced; pinned to the ISS engine by `tests/test_skat_reference.py`); see `rs/gmdata/src/skat.rs` |
| contract bridge | `$G bridge fetch --source openspiel` (OpenSpiel's WBridge5 dataset, GCS: 1.16M deals) and `--source ben` (Ben's WBC play data, GitHub release: 1.89M boards); optional `--source bba` (Ben's BBA bidding sets: ~1.15M EPBot auctions + results per set, no play); `scripts/fetch_bridge.sh` end to end | every board: robot play at world-championship level (WBridge5 at its strongest setting, SAYC; World Computer Bridge Championship robots); `convert --complete-only` drops records that stop without a result | gmrules game (below), rules pinned to OpenSpiel. Licences: the OpenSpiel dataset's README states none (DeepMind research data published with OpenSpiel, Apache-2.0); Ben's data is a release asset of lorserker/ben (GPL-3.0), with no separate data licence. BBO vugraph (LIN) is not fetched (pending a decision); `convert --format lin` reads it |
| heads-up no-limit hold'em | `$G hunl fetch --out D` (ACPC 2009-2017 logs in PHH, Zenodo 10796885, CC-BY-4.0: range requests for the 4.85 GB of heads-up no-limit logs); end to end `scripts/fetch_hunl.sh D` | `convert --hq`: the seats of the year's winning bots (bb/100 ≥ `--min-bb100`); the 200 bb game only (`--rescale`: 2009's 1/2/400 ×50); duplicates dropped; `--keep-per-mille` uniform subsample | 160.6 M unique hands of bot play; gmrules rules pinned to PokerKit; see `rs/gmdata/src/hunl.rs` |
| chess variants (lichess: chess960, crazyhouse, atomic, antichess, three-check, king of the hill, horde, racing kings) | `$G chess-variants fetch --variant all --month 2026-08` (the monthly rated dumps of database.lichess.org, **CC0**; `--max-bytes N`: a sample); end to end `scripts/fetch_chess_variants.sh D` (month by month and variant by variant, resumable, raw dumps deleted unless `KEEP_RAW=1`; `VAL_MONTHS` holds out whole months) | `convert --hq`: rated, normal termination, ≥ 60 s, ≥ 4 plies, top 10 % of each dump by the lower Elo (6.6-8.5 % of the games kept); optional `--min-clock-s S` (cuts time scrambles; clocks are never features) | ONE gmrules game `chess_variants` (a variant chance event, `rs/gmrules/src/games/chess_variants.rs`): shakmaty's variant rules + lichess' automatic ends (scalachess ports), pinned to Fairy-Stockfish by `tests/test_chess_variants_reference.py`; every record replayed, rules ends checked against the PGN result (0 illegal / 0 mismatches on the 2026-08 samples); tests `tests/test_chess_variants_pipeline.py`. Licence: shakmaty (GPL-3.0-or-later) is LINKED into gmrules, so gmdata and the gmsim extension built with it are GPL-3.0-or-later combined works (fine in-house; distributing those binaries means GPL terms); Fairy-Stockfish (GPL-3.0) is only a test reference in a throw-away uv env |
| xiangqi | `$G xiangqi fetch --out D` (the Chinese Chess Practical Dataset, GitHub archive at a pinned commit, 60 MB; **CC BY 4.0**, Yu-Han Tseng and Bo-Nian Chen 2026 — the attribution is written next to the archive and copied next to the shards); end to end `scripts/fetch_xiangqi.sh D` | `convert --hq`: master games (CCPD 大師對局), ≥ 20 plies, known result; `--split train\|val` (2 % val by the moves' hash) | gmrules game (`rs/gmrules/src/games/xiangqi/`: Asian rules as Fairy-Stockfish implements them, pinned by `tests/test_xiangqi_reference.py`); every move resolved against the legal moves, records going on past the rules' end cut there; 26,980 / 567 train / val games (2.23 M / 46 k frames); tests `tests/test_xiangqi_pipeline.py` |

Measured on real data (kept / seen): chess elite 2021-06 601k / 781k games;
go KGS 2019-04 202 / 1255; shogi 2025-06-01 158 / 447; othello 2019–21
1502 / 3559; gomoku Freestyle1 2023 658 / 2880; Clash Royale 5000 replays
8456 / 9392 player sequences (4186 with `--winner-only`); mahjong Houou shards
tenhou-00015 + tenhou-new-00001 + tenhou-release-00005 132,638 / 164,954 games
(29,875 tonpuusen, 2,439 re-spelled duplicates, 2 corrupt logs).

Pokémon replays are spectator logs: a side's own moves, items, abilities and
tera type (all known to its player) appear only once revealed. First-person
trajectories — e.g. Metamon's reconstructed replays
(github.com/UT-Austin-RPL/metamon, MIT), which rebuild each player's own view
of the battle — could replace them later through a converter emitting the same
schema; details in the `pokemon.rs` module docs.
8456 / 9392 player sequences (4186 with `--winner-only`); skat, the full ISS
archive: 9,094,590 / 9,094,699 records verified (0 result mismatches), `--hq` train
11.2 M / 27.0 M seat streams (137 M / 315 M frames, ~57 GB).
8456 / 9392 player sequences (4186 with `--winner-only`); bridge WBridge5 test set
10,000 / 10,000 deals, Ben WBC 1,883,853 / 1,887,139 boards (3,286 rejected by the
verifying replay as data errors; 291 kept as auction-only records), Ben BBA
WBridge5-Sayc set 1,154,937 / 1,154,937 auctions.
8456 / 9392 player sequences (4186 with `--winner-only`); heads-up no-limit
hold'em 72 000 / 72 000 hands of 24 logs (3 per year; 0 illegal actions, 0 payoff
mismatches), 400 k decision frames.

## Engine teacher labels

`gmdata label <game>` adds engine soft targets (teacher move distributions per
decoder head) and engine values to existing gm-shards, as overlay shards that
hard-link the base tensors (the base is never modified) and that the trainers
read unchanged (`--soft_target_weight`, `--engine_value_weight`). Engines are
external processes installed outside the repo by
`scripts/engines/fetch_engines.sh` (Stockfish, Egaroucid, YaneuraOu, KataGo,
Pikafish; licences in its header). Columns, options and the per-game extension point are
documented in `rs/gmdata/src/label/mod.rs`; `gmdata label selftest --out DIR`
exercises the whole pipeline with a built-in fake engine.

```sh
scripts/engines/fetch_engines.sh stockfish
$G label chess --input data/chess/train --out data/chess_sf/train \
   --engine ~/gm-engines/bin/stockfish --workers 32     # defaults: nodes=120000, MultiPV 8
scripts/engines/fetch_engines.sh egaroucid
$G label othello --input data/othello/train --out data/othello_ega/train \
   --engine ~/gm-engines/bin/egaroucid --workers 64   # Egaroucid level 15, exact <= 20 empties
scripts/engines/fetch_engines.sh pikafish
$G label xiangqi --input data/xiangqi/train --out data/xiangqi_pf/train \
   --engine ~/gm-engines/bin/pikafish --workers 32      # defaults: nodes=100000, MultiPV 8
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
| xiangqi | as chess (Pikafish cp and mates in moves) |
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

New games are written ONCE, in Rust, in `rs/gmrules` (the `Game` trait: turns,
chance events, canonical action ids, outcome, and the frame encoders; no I/O). The
same code serves

- the converter: source records are parsed into trajectories (chance outcomes +
  seat actions as canonical ids) and REPLAYED by the rules — every event verified,
  recorded payoffs checked — and every decision frame and its targets are encoded by
  `gmrules::encode_frame` / `encode_target` (generic driver `rs/gmdata/src/traj.rs`);
- the simulator (`gmsim`, a PyO3 abi3 extension built by `uv sync`, see
  `rs/gmsim/README.md`): `env/vec_gmsim.py` wraps its batched `VecSim` as the RL
  `VecGameEnv` — it encodes with the same functions and decodes the policy's action
  keys back into canonical actions,

so IL frames and RL frames are identical by construction (and pinned by golden
tests).

```text
record (any format) ─parse (gmdata/<game>.rs)─► Trajectory ─┐
                                                            ├─ replay ─► frames + targets ─► gm-shards
gmsim VecSim (RL env) ─ chance from seeded Rng, actions ────┘   (encode_frame / encode_target)
```

Frame schema (`FrameSpec`): every decision frame is a fixed set of named, typed,
fixed-shape tensors in three groups — `obs` (the seat's observation: gm-shard
`obs.<name>`, the env's obs), `legal` (the RL decoder's legality masks: the env's
`legal`, never stored) and `target` (IL labels of the taken action: gm-shard
`tgt.<name>`); the spec's `action_keys` (i64 scalar targets) are the policy's action
— what the simulator's `step` takes and the decoder's replay keys. Two kinds:

- CANDIDATE-ACTION schema (`candidate.v1`, `FrameSpec::candidate`; model:
  `model/games/candidate.py`) for games whose legal actions can be enumerated per
  decision (card games, bidding, poker with discretised sizes, backgammon full
  moves): the game writes entity slots / globals (`encode_obs`) and per-candidate
  features (`encode_action`), the framework does the rest:

  ```text
  obs.entities    [T, E, Fe] f32      obs.entity_mask [T, E] bool   obs.entity_type [T, E] i32 (optional)
  obs.globals     [T, G] f32
  obs.actions     [T, A, Fa] f32      obs.action_mask [T, A] bool   (legal candidates, ascending ids)
  tgt.action_idx  [T] i64             index of the taken action among the candidates (action key)
  tgt.value       [T] f32             outcome of the frame's seat in [-1, 1]
  env legal       action [A] bool     (= action_mask)
  ```

- CUSTOM schemas (a `FrameSpec` with `candidate: None`) for games that feed another
  adapter, e.g. chess-like games on the board adapter's from-to decoder: the game
  declares its tensors and implements `encode_custom` (obs + legal), `target_custom`
  (action keys + IL-only labels) and `decode_custom` (action keys → action). The
  reference is `breakthrough` (`fromto.v1`): obs `board_seq [3, R, C]` (bits) /
  `globals_seq [2]`, legal `from [N]` / `to_cond [N, N]`, targets `from_idx` /
  `to_idx` (action keys) + `legal_from` / `legal_to_of` (the chess converter's IL
  masks), model `build_fromto_network`, RL `FromToSpec`.

Shard meta = `FrameSpec::schema_meta()` (schema name, layout, n_seats, the schema's
`meta` facts — candidate dims, board size, ... — which the trainers'
`META_ARCH_KEYS` check and a fresh run takes its model dims from) + the game config +
`rules_version` (= `converter_version`). Layout: hidden-information and simultaneous
games use PER-SEAT streams (one sample / RL row per seat holding only that seat's own
decisions, so no temporal state ever carries another seat's private information);
perfect-information alternating games use one interleaved stream per game (like the
board games). f32 tensors whose values are all 0/1 can be stored bit-packed
(`TensorSpec::bits`).

`doudizhu` (3 players, OpenSpiel's auction, landlord vs two peasants; standard rules =
OpenSpiel's `dou_dizhu` + four-with-two, `rules="openspiel"` = OpenSpiel exactly — pinned
step for step by `tests/test_doudizhu_reference.py`, the play phase against DouZero's
engine by `tests/test_doudizhu_douzero.py`) is the FACTORED-decision example: a
combination with kickers is chosen as its main part, then one kicker per decision (only
completable prefixes are offered), so every frame has at most 92 candidates (exact)
where one hand can hold 521 full plays. Its corpus is DouZero self-play
(`scripts/generate_doudizhu.py`; `gmdata doudizhu convert`, teacher soft targets from
the records' Q values with `--soft-k K --soft-temp AGENT=T`).

Test games (all pinned to OpenSpiel, `tests/test_gmrules_reference.py`):
`kuhn_poker` (the toy reference game through every layer, 2-4 players, hidden
cards: converter from synthetic hand logs, `scripts/fetch_kuhn_poker.sh`),
`goofspiel` (simultaneous moves), `pig` (perfect information + dice, interleaved
stream) and `breakthrough` (the custom-schema board game). `gmdata rules
list|check|random|convert` work for every registered game without per-game code
(conformance battery, random-play shards, trajectory JSONL → shards).

`skat` (3 players, the International Skat Server's rules: bidding incl. jump bids,
pickup / hand, suit / grand / null, schneider / schwarz / ouvert, ISS scoring) is a
real candidate game: decisions factored to ≤ 12 candidates (bid window + MORE, hand
game type + modifier, game + one discard at a time), per-seat streams, reward = ISS
money score / 1260. Converter `gmdata skat` (the ISS archive: every move replayed,
every recorded result reproduced — 9,094,590 of 9,094,699 records, 0 result
mismatches), `scripts/fetch_skat.sh`; parity with the ISS rule engine itself
(its GPLv3 client classes, fetched at test time) by randomized differential tests in
both directions (`tests/test_skat_reference.py`, live with `GM_REFERENCE=1` + a JDK).
Full games: `bridge` (contract bridge: the auction with doubles / redoubles, dummy
exposed after the opening lead and played by declarer, duplicate scoring by
vulnerability; 4 seats in two partnerships, `Game::teams`). Rules and frames are in
`rs/gmrules/src/games/bridge.rs`, pinned to OpenSpiel by `tests/test_bridge_reference.py`.
The converter is `gmdata bridge fetch|check|convert` (OpenSpiel WBridge5 lines, Ben
Jack-format boards, PBN, BBO LIN; `rs/gmdata/src/bridge.rs`), end to end in
`scripts/fetch_bridge.sh`; tests are in `tests/test_bridge_pipeline.py`.
`hunl` (heads-up no-limit hold'em, the ACPC rules, pinned to PokerKit by
`tests/test_hunl_reference.py`) is the ABSTRACTED-action example: exact chip amounts
in the rules and the records, 15 candidates for the policy (fold, check / call,
all-in, min-raise, 11 pot fractions; `candidate_abstract.v1` = the candidate layout,
so the candidate adapter and `CandidateSpec` play it), IL target = the nearest
candidate plus the exact size (`tgt.raise_to`, `tgt.raise_frac`). Converter `gmdata
hunl` (ACPC logs), `scripts/fetch_hunl.sh`, optional evaluation against Slumbot
`scripts/slumbot_client.py`, tests `tests/test_hunl_pipeline.py`.

### Adding a game

1. Rules — `rs/gmrules/src/games/<game>.rs` implementing `gmrules::Game`:
   - canonical `u32` action / chance-outcome ids (take the reference
     implementation's ids when there is one: differential tests compare them
     directly); `legal_actions` strictly ascending (= the candidate order);
   - the schema: `FrameSpec::candidate(CandidateSpec {...})` with fixed maxima
     (entities; candidates `max_actions` = the worst legal position of the game;
     `max_decisions`) — keep entity slots stable across frames (slot = identity:
     card, seat, zone) and prefer dims that do not depend on the config (one network
     plays every variant) — or a custom `FrameSpec` (copy `games/breakthrough.rs`);
     a decision whose legal set can exceed a few hundred (Dou Dizhu leads, backgammon
     doubles) is better FACTORED into consecutive decisions of the same seat (main
     group, then kickers; one checker move per die), each its own frame; an
     ABSTRACTED action space (poker: exact bet amounts in the rules and the records,
     a few discretised sizes for the policy) is a custom schema with the candidate
     layout (the same obs tensors / `action_idx` / meta dims, so the candidate
     adapter and `CandidateSpec` apply) whose `target_custom` maps every legal
     amount to its size candidate and whose `decode_custom` maps a candidate to its
     representative legal amount (the battery checks that the two agree);
   - encoders read ONLY what the seat knows now (public state, its own private
     state and actions); `outcome()` in [-1, 1] is both the IL value target and the
     RL reward; `returns()` the raw payoffs; `check_config` validates the config;
     hidden-information games implement `perturb_hidden` (re-deal what `seat` cannot
     see at the end of the record, keep every recorded action legal — the battery
     also calls it on records cut BEFORE every decision, whose frame is then the
     seat's pending frame: so information a decision's own action reveals, like
     dummy's hand at the opening lead, is checked too; cuts may fall inside a
     factored decision); partnership games return their
     `teams` (teammates share one outcome: the simulator reports the winning team,
     the league assigns whole teams, a team row's `stream_seats` is its team);
   - register: `pub mod <game>;` in `games/mod.rs` + one line in
     `registry::GAMES`;
   - unit tests: hand-checked rule examples + `invariance::check_game::<G>(&cfg, n,
     seed)` for every variant (causality, privacy, frame conformance, the action-key
     round trip decode(target(a)) == a, simulator replays — see
     `games/kuhn_poker.rs`, `games/breakthrough.rs`).
2. Parity with an independent reference (OpenSpiel, PokerKit, gnubg, DouZero,
   Fairy-Stockfish, ...): a dump script run in a throw-away environment
   (pattern: `scripts/reference/openspiel_dump.py`, `uv run --no-project --with
   open_spiel`) recording per step the turn, every acting seat's legal actions, the
   chance outcomes + probabilities and the final returns; a small committed fixture
   plus a live test (`GM_REFERENCE=1`) in both directions — reference playouts
   stepped through `gmsim.State`, and gmrules' random playouts replayed through the
   reference (`tests/test_gmrules_reference.py`). Real records replayed by the
   converter are the second proof: `illegal` / `returns_mismatch` must stay ~0.
3. Converter — `rs/gmdata/src/<game>.rs`: `fetch` (download; or a
   `scripts/fetch_<game>.sh` that downloads AND converts end to end — the full-scale
   runs happen on the cluster), a parser from the source format to `traj::Record`s
   (events, meta, optional seat filter for the high-quality recipe, the recorded
   payoffs), `traj::convert::<G>`; flags `--hq`, `--store-events` (golden tests) and
   `--allow-truncated` (causality tests); one `Game` variant in `main.rs`. Records
   are streamed (the converter consumes an iterator; a parse error is counted, not
   fatal), converted in parallel and written in input order. Engine / solver teacher
   labels (candidate schema): `Record::soft_targets` (canonical action ids per Act
   event) with `ConvertOptions::soft_k` become `tgt.action_probs_idx/val` over
   candidate indices (also `"soft"` in the JSONL of `gmdata rules convert --soft-k
   K`). Template: `kuhn_poker.rs` and `scripts/fetch_kuhn_poker.sh`. Record the data
   licence in the table above.
4. Python — `model/games/gmrules/<game>.py` with `GAME = RulesGame(...)`; discovery
   registers it in `build_game_network`, train_il_game `LABEL_KEYS` /
   `META_ARCH_KEYS` / `LEGAL_FROM_OBS`, `make_vec_env` (env/vec_gmsim.py) and
   train_rl_joint `GAME_SPECS` — no shared file edits.
   - candidate game: `RulesGame(name="<game>", sim_config={...})` (the generic
     adapter and `CandidateSpec`; `config=`, `config_class=`, `encoder=`, `decoder=`,
     `rl_spec=` customise it);
   - custom schema: also `builder=` (the model), `label_keys=` (teacher-forced tgt
     keys), `meta_arch=` (shard meta key → config fields), `sample_keys=` (decoder
     `sample()` key → action key) and `rl_spec=` (e.g.
     `"training.train_rl_joint:FromToSpec"` for from-to games) — see
     `model/games/gmrules/breakthrough.py`.
5. RL: per-seat games are not lockstep (`--actor stream`); the env steps on
   `threads` Rust threads (`--env_kw <game>:threads=N` or `$GM_SIM_THREADS`, default
   1; results never depend on it); `--env_kw <game>:key=value` sets the gmrules
   config and the spec checks it against the network's dims.
6. Tests (templates `tests/test_candidate_pipeline.py`, `tests/test_gmrules_custom.py`):
   converter frames == `gmsim.replay` frames of each record; simulator frames ==
   converter frames of the simulator's own episodes; `tests/invariance.py` —
   `check_sim_battery` and, on the real converter binary, `check_converter_causality`
   (truncated records) and `check_converter_privacy` (opponents' private information
   changed); a tiny `train_multi_game` + `train_rl_joint` run.

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
uv run pytest tests/test_candidate_*.py tests/test_gmrules_*.py   # gmrules / gmsim / adapters
GM_REFERENCE=1 uv run pytest tests/test_gmrules_reference.py   # + live OpenSpiel diff
GM_REFERENCE=1 uv run pytest tests/test_backgammon.py          # + live gnubg diff / match loop
uv run pytest tests/test_hunl_*.py tests/test_slumbot_client.py   # hold'em end to end
GM_REFERENCE=1 uv run pytest tests/test_hunl_reference.py      # + live PokerKit diff
```

Real-engine tests run only when the engine is installed (e.g. `STOCKFISH=path`
or `~/gm-engines/bin/stockfish`, `GM_EGAROUCID=path` or `~/gm-engines/bin/egaroucid`,
`GM_YANEURAOU`, `GM_KATAGO`); the label tests otherwise use fake engines.

The parity tests pin every converter to golden SHA-256 digests of the retired
Python reference outputs on committed real-data fixtures (tests/fixtures/).
