//! Pokémon Showdown replays (observer-view sim protocol) → shards for the pokemon adapter.
//!
//! converter_version 3 (v2 + exact move / species ids, see "Vocabulary" below). One sample
//! per (replay, side) holding one frame per decision the side's player answers, in log
//! order — the requests a live Showdown bot receives:
//!
//!   team preview   phase 2, turn 0 — the lead, when the format has team preview (`|poke|` /
//!                  `|teampreview|` lines): a switch to the lead's team-preview slot.  Both
//!                  leads are chosen simultaneously, so both sides' frames are the state before
//!                  the first lead `|switch|` line (neither sees the other side's lead).
//!   turn           phase 0, turn t ≥ 1 — one frame per side at EVERY `|turn|t` line: the state
//!                  before turn t's simultaneous choices, whatever happens during the turn (so
//!                  which frames exist never depends on the turn's outcome).  Labelled with the
//!                  side's first visible choice of the turn: its `|move|` (also `|cant|..|MOVE`,
//!                  a chosen move that could not be used; `|cant|HOLDER|..|MOVE|[of] USER` is an
//!                  Armor Tail / Dazzling / Queenly Majesty / Damp holder blocking USER's move,
//!                  no choice of HOLDER) or a voluntary `|switch|`.  Turns whose choice is not
//!                  visible — KO'd / flinched / asleep / paralysed before moving, a locked move
//!                  (`[from]lockedmove`) or recharge turn (no choice at all), Struggle, forced
//!                  out first, an Encore landing before the move (Showdown then uses the encored
//!                  move instead of the chosen one, logged without `[from]`), the battle ended —
//!                  keep their frame with the label masked: tgt.action_known 0, action_type −1.
//!   forced switch  phase 1 — every other in-battle `|switch|` of the side: KO replacements,
//!                  pivots (U-turn, Volt Switch, Flip Turn, Parting Shot, Chilly Reception,
//!                  Teleport, Baton Pass, Shed Tail), Eject Button / Eject Pack, Emergency Exit /
//!                  Wimp Out.  The frame is the state when Showdown issues the switch request:
//!                  right before the switch line for mid-turn requests (one side's), and for the
//!                  end-of-turn request (after `|upkeep|`, sent at once to every side with a
//!                  fainted active or a pending eject) the state before the FIRST switch line of
//!                  that block, so a side never sees the other side's simultaneous replacement.
//!                  Server logs mark the answered decision with `|` + `|t:|` (turnLoop): when that
//!                  commit follows the request's cause, the frame is the state AT the commit — the
//!                  switch action's switch-out effects (Natural Cure's cure, the end of
//!                  Neutralizing Gas or a primal weather, ...) are logged between it and the
//!                  `|switch|` line, after the decision.  A request left unanswered (forfeit at the prompt) yields no frame — nor does
//!                  a simultaneous end-of-turn request the OTHER player forfeited at (neither
//!                  switch is logged).  Revival Blessing's choice of a fainted Pokémon logs no
//!                  `|switch|` and has no frame.
//!
//!   obs.species_ids [T,12] i64 (exact species ids)    obs.move_ids [T,4] i64 (exact move ids)
//!   obs.poke_feats [T,12,17] f32   obs.slot_mask [T,12] bool   obs.field_seq [T,22] f32
//!   tgt.action_type [T] i64    0 move / 1 switch / −1 no observed choice
//!   tgt.action_known [T] bool  the frame carries an action label
//!   tgt.move_id [T] i64        exact id of the used move, −1 off move frames — the identity
//!                              target (the policy scores the whole move vocabulary; a live bot
//!                              masks it to the moves of its request)
//!   tgt.move_idx [T] i64       the used move's slot in obs.move_ids when it was revealed BEFORE
//!                              the frame (legacy slot head), else −1 (first use, 5th+ reveal)
//!   tgt.switch_idx / tera / active_idx [T] i64, tgt.value [T] f32
//!   tgt.legal_action_type [T,2] bool   [move, switch]: forced / preview frames [0, 1]; turn
//!                              frames [1, a live own bench Pokémon exists]
//!   tgt.legal_switch [T,6] bool  own slots that may come in (present, not fainted, not active;
//!                              unknown slots of a side without team preview count as legal)
//!
//! obs.field_seq = the v1 layout (weather, terrain, turn/50, both sides' conditions) + [20]
//! forced-switch request, [21] team-preview request.  obs.move_ids stays strictly causal: the
//! first four moves the active Pokémon revealed before the frame.  The legal masks are
//! functions of the frame's state only; a label outside them is counted (stat `mask_mismatch`)
//! and left to the decoder, which gives it no weight under a legality mask.  `|split|` blocks
//! of omniscient logs: only the public line is read.
//!
//! State tracking beyond v1: `-swapsideconditions` (Court Change) swaps the sides' hazards and
//! screens; `-transform` gives the transformed Pokémon the target's revealed moves and boosts
//! for as long as it stays in (moves used meanwhile are not added to its own moveset);
//! Revival Blessing's `-heal` revives the Pokémon: fainted flag, status and tera state cleared
//! (as Showdown does).  Species (v3): a slot shows the forme the log last revealed — the
//! team-preview placeholder of a species whose forme team preview hides (`Urshifu-*`,
//! `Zamazenta-*`, `Greninja-*`, …) until the Pokémon's first switch-in names the forme, and
//! every permanent forme change (`|detailschange|`: Mega Evolution, Primal Reversion,
//! Ogerpon / Terapagos tera formes, Palafin-Hero, Mimikyu-Busted, and the regression when such
//! a Pokémon faints) from its line on; temporary ones (`-formechange`) are not tracked.
//! Replays without a `|win|` yield no samples.
//!
//! Vocabulary (v3; v1 / v2 hashed names into crc32 buckets — v2: 1024 moves / 512 species —
//! where e.g. Swords Dance and Fire Blast, or Dragonite and Hoopa-Unbound, shared an id): every
//! move / species name maps to an exact id of data_processing/games/pokemon/showdown_vocab.json,
//! generated from Pokémon Showdown's data (data/moves.ts, data/pokedex.ts, the team-preview
//! forme hiding of data/rulesets.ts) at a pinned commit by make_vocab.py next to it, and
//! embedded here at build time.  Ids: 0 none (empty name), 1 UNK (a name the vocabulary does
//! not know — counted per name in the convert stats `unk_moves` / `unk_species`, never merged
//! with a known id), 2.. the entries.  Keys are Showdown's toID of the name (species: plus
//! Nidoran♀/♂ → nidoranf/nidoranm and a `*` suffix for a team-preview placeholder); cosmetic
//! formes (Gastrodon-East, Vivillon-Polar, …) and other Unicode spellings are aliases of their
//! entry.  data_processing/games/pokemon/__init__.py is the same lookup in Python (the adapter's
//! deploy request mask).  Shard meta: `{"field_features": 22, "move_vocab": M, "species_vocab":
//! S, "item_vocab": I, "ability_vocab": A, "type_vocab": Y, "vocab_version": V}` (id counts
//! incl. the reserved ids; the vocabulary is append-only, so its size identifies it).
//!
//! converter_version 4: every slot's revealed information, both sides — what the public log
//! has shown by the frame, from either player's view:
//!
//!   obs.slot_moves [T,12,4] i64   the first four moves the Pokémon revealed (the transformed
//!                                 set while transformed); v3 showed only the own active's
//!                                 (obs.move_ids, kept), so the opponent's revealed moves never
//!                                 reached the network
//!   obs.slot_item [T,12] i64      the held item the log revealed (`-item`, `-enditem`, any
//!                                 `[from] item: X`), 0 = not revealed
//!   obs.slot_no_item [T,12] bool  known to hold nothing (`-enditem`: consumed, knocked off,
//!                                 stolen, flung)
//!   obs.slot_ability [T,12] i64   the revealed ability (`-ability`, `[from] ability: X`; a
//!                                 Trace copies onto the tracer), 0 = not revealed
//!   obs.slot_tera_type [T,12] i64 the tera type once terastallized, 0 before
//!
//! A `[from]` effect with an `[of] SOURCE` belongs to SOURCE (Rocky Helmet, Rough Skin,
//! Drizzle), else to the line's Pokémon.  A side's own full sets stay out: the spectator log
//! shows them only once revealed, and filling them in from the whole replay would tell a frame
//! which Pokémon come in and act later (a future leak).
//!
//! converter_version 5: more formats, levels, dumps.
//!
//!   obs.poke_feats [T,12,18]     + [17] level / 100 (details `L<n>`, 100 when not given): 1 in
//!                                the level-100 tiers, 0.05 in Little Cup, per Pokémon in the
//!                                random battles
//!   obs.format_id [T] i64        the battle's format, an id of FORMATS (Gen 9 6v6 singles with
//!                                the standard mechanics: OU, UU, RU, NU, PU, ZU, LC, Ubers,
//!                                Ubers UU, Monotype, Anything Goes, the random battles); 0 = a
//!                                format outside the table.  Shard meta + `poke_features`,
//!                                `format_vocab`, `formats`.
//!
//! Inputs (`--input`, repeatable): directories of the replay API's `.json` objects and / or
//! parquet dumps with its columns (id, log, rating, format / formatid, players, uploadtime:
//! HolidayOugi/pokemon-showdown-replays, milkkarten/pokemon-showdown-replays-merged on
//! Hugging Face, fetched by `fetch-hf`); a replay id is converted once, from the first input
//! that has it.  Selection (`Select`):
//!   * `--formats` keeps those formats;
//!   * `--min-rating R`: a rated battle's rating must reach R.  The rating Showdown stores is
//!     `min(p1, p2)` of the players' Elo after the battle (server/ladders-remote.ts: `room.rated
//!     = Math.min(p1NewElo, p2NewElo)`; at matchmaking the lower pre-battle Elo, ladders.ts),
//!     so R is a bound on BOTH players;
//!   * `--players P --min-peak K [--min-peak-games N]`: both players' highest battle rating in
//!     the format (the `players` table over the same inputs; a lower bound of their peak Elo,
//!     robust to accounts still climbing from 1000) must reach K with ≥ N rated battles — this
//!     also admits unrated battles (tournaments, challenges) between proven players.
//!
//!   $G pokemon fetch-hf --dataset HolidayOugi/pokemon-showdown-replays --match "[Gen 9] OU" --out $P/hf
//!   $G pokemon players --input $P/hf --input $P/raw --formats gen9ou --out $P/players.json
//!   $G pokemon convert --input $P/raw --input $P/hf --formats gen9ou --min-rating 1500 \
//!       --players $P/players.json --min-peak 1700 --min-turns 5 --exclude-inactivity --out $P/shards_v5
//!
//! Re-conversion (v1 / v2 shards are obsolete: hashed, colliding move / species ids) — on the
//! cluster from the checkout root (`cargo build --release -p gmdata` in data_processing/; the
//! fetch is resumable and skips replays already on disk — without internet access on the
//! nodes, fetch elsewhere and rsync the raw directory), then every 5th shard → val:
//!
//!   G=$PWD/data_processing/target/release/gmdata; P=$HOME/gm-data/pokemon
//!   $G pokemon fetch --format gen9ou --min-rating 1500 --max-replays 2000 --sleep-ms 500 \
//!       --out $P/raw
//!   $G pokemon convert --hq --input $P/raw --out $P/shards_v3
//!   mkdir -p $P/split_v3/train $P/split_v3/val; i=0
//!   for s in $P/shards_v3/pokemon_*; do
//!     [ $((i % 5)) -eq 4 ] && v=val || v=train; ln -s $s $P/split_v3/$v/; i=$((i + 1)); done
//!
//! Train on `--data pokemon=$P/split_v3/train --val pokemon=$P/split_v3/val`; fresh runs take
//! pokemon_field_features / pokemon_move_vocab / pokemon_species_vocab from the shard meta.
//! check_meta_arch refuses a hashed-bucket checkpoint (pokemon_move_vocab 0) on v3 shards —
//! its move / species tables index a different id space: rebuild its pokemon adapter (a fresh
//! adapter on the trained core) — and a fresh exact-vocabulary net on v1 / v2 shards, unless
//! asked for the legacy net explicitly (`--override pokemon_move_vocab=0
//! pokemon_species_vocab=0`).  A v1 adapter (20 field features, the slot head) is refused on
//! v2 / v3 shards as before.
//!
//! Spectator logs are third-person: the converter can only show what the log reveals, so a
//! side's own moves, items, abilities, stats and tera type — all known to the player — appear
//! only once revealed, and an Illusion switch is filed under the disguise.  First-person
//! trajectories remove that gap: Metamon (github.com/UT-Austin-RPL/metamon, MIT) rebuilds each
//! player's own view from Showdown replays (the player's team filled in from what the battle
//! reveals, the rest estimated), with the action taken at each step.  A future `gmdata pokemon
//! convert-metamon` could emit this schema from those trajectories: map their steps onto the
//! frames here (turn / forced switch, team preview where present), fill the own-team slots and
//! obs.move_ids with the full sets, and derive legal_* from the known own team.  Only the
//! player's own side may be filled in that way — the opponent stays revealed-only, and
//! estimated fields are a train/deploy gap (a live bot knows its exact set).

use crate::common;
use anyhow::{Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use gmpokemon::*;

/// The samples of a parsed replay (the winner's only with `winner_only`), names mapped to
/// exact vocabulary ids; the names that map to UNK are counted in `unk`.
fn replay_samples(r: &Value, p: &Parse, winner_only: bool, unk: &mut UnkNames) -> Vec<Sample> {
    let mut out = Vec::new();
    let Some(w) = p.winner else { return out };
    let voc = vocab();
    for side in 0..2 {
        let recs = &p.records[side];
        if recs.is_empty() || (winner_only && w != side) {
            continue;
        }
        let t = recs.len();
        let (mut sp, mut mv, mut feats, mut mask, mut field) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut at, mut known, mut mi, mut mid, mut si, mut te, mut ai) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut lt, mut ls) = (vec![], vec![]);
        let (mut smv, mut sit, mut sni, mut sab, mut ste) = (vec![], vec![], vec![], vec![], vec![]);
        for rec in recs {
            let f = &rec.frame;
            // the same mapping as a live request's frame (gmpokemon::frame_ids, gmsim)
            let ids = frame_ids(f, unk);
            sp.extend(ids.species);
            mv.extend(ids.moves);
            smv.extend(ids.slot_moves);
            sit.extend(ids.slot_item);
            sni.extend_from_slice(&f.slot_no_item);
            sab.extend(ids.slot_ability);
            ste.extend(ids.slot_tera);
            feats.extend_from_slice(&f.feats);
            mask.extend_from_slice(&f.mask);
            field.extend_from_slice(&f.field);
            ai.push(f.active);
            at.push(rec.action_type);
            known.push(rec.action_type >= 0);
            mi.push(rec.move_idx);
            mid.push(if rec.action_type == 0 { unk.move_id(voc, &rec.move_name) } else { -1 });
            si.push(rec.switch_idx);
            te.push(rec.tera);
            lt.extend_from_slice(&rec.legal_type);
            ls.extend_from_slice(&rec.legal_switch);
        }
        let v = if w == side { 1.0 } else { -1.0 };
        let fmt = r.get("formatid").or_else(|| r.get("format")).cloned().unwrap_or_else(|| json!("?"));
        let fid = format_id(&record_format(r));
        out.push(Sample::new(json!({"replay_id": r.get("id").cloned().unwrap_or(json!("?")), "side": (if side == 0 { "p1" } else { "p2" }),
                                     "player": p.names[side], "rating": r.get("rating").cloned().unwrap_or(Value::Null),
                                     "format": fmt, "won": w == side}))
            .i64("obs.species_ids", &[2 * TEAM], sp)
            .i64("obs.move_ids", &[MOVES], mv)
            .f32("obs.poke_feats", &[2 * TEAM, FEAT], feats)
            .bool("obs.slot_mask", &[2 * TEAM], mask)
            .f32("obs.field_seq", &[FIELD], field)
            .i64("obs.slot_moves", &[2 * TEAM, MOVES], smv)
            .i64("obs.slot_item", &[2 * TEAM], sit)
            .bool("obs.slot_no_item", &[2 * TEAM], sni)
            .i64("obs.slot_ability", &[2 * TEAM], sab)
            .i64("obs.slot_tera_type", &[2 * TEAM], ste)
            .i64("obs.format_id", &[], vec![fid; t])
            .i64("tgt.action_type", &[], at)
            .bool("tgt.action_known", &[], known)
            .i64("tgt.move_id", &[], mid)
            .i64("tgt.move_idx", &[], mi)
            .i64("tgt.switch_idx", &[], si)
            .i64("tgt.tera", &[], te)
            .i64("tgt.active_idx", &[], ai)
            .bool("tgt.legal_action_type", &[2], lt)
            .bool("tgt.legal_switch", &[TEAM], ls)
            .f32("tgt.value", &[], vec![v; t]));
    }
    out
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download public replays of a format via the replay search API (resumable, polite)
    Fetch {
        #[arg(long, default_value = "gen9ou")]
        format: String,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        #[arg(long, default_value_t = 1000)]
        max_replays: usize,
        #[arg(long, default_value_t = 1000)]
        sleep_ms: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Download replay dumps (parquet) of a Hugging Face dataset: the files whose names
    /// contain one of `--match` (resumable, SHA-256 checked against the Hub's LFS pointers)
    FetchHf {
        /// e.g. HolidayOugi/pokemon-showdown-replays or milkkarten/pokemon-showdown-replays-merged
        #[arg(long)]
        dataset: String,
        /// substrings of the file names to take (e.g. "[Gen 9] OU_part"); none = every parquet file
        #[arg(long = "match")]
        matches: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Per format and player: the rated battles seen and the highest battle rating — a player's
    /// strength estimate (a battle's rating is the lower of both players' Elo, so a player's
    /// highest one is a lower bound of their peak)
    Players {
        /// replay directories (.json) and / or parquet dumps; a replay id counts once
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        /// formats to count (ids as in FORMATS, comma-separated); none = all
        #[arg(long, value_delimiter = ',')]
        formats: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert replays (.json files and / or parquet dumps) to gm-shards
    Convert {
        /// replay directories (.json) and / or parquet dumps (files or directories); a replay
        /// id is converted once, from the first input that has it
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        /// a rated battle's rating (the lower of both players' Elo) must reach this; unrated
        /// battles pass only through `--players` / `--min-peak`
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        /// formats to keep (ids as in FORMATS, comma-separated); none = all
        #[arg(long, value_delimiter = ',')]
        formats: Vec<String>,
        /// the `players` table: with `--min-peak`, both players' highest battle rating in the
        /// format must reach it (rated and unrated battles alike)
        #[arg(long)]
        players: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        min_peak: i64,
        /// with `--min-peak`: both players need at least this many rated battles in the format
        #[arg(long, default_value_t = 1)]
        min_peak_games: i64,
        #[arg(long)]
        max_replays: Option<usize>,
        /// drop replays shorter than this many turns
        #[arg(long, default_value_t = 0)]
        min_turns: i64,
        /// drop replays that end in a forfeit (usually a concession: only the last turn is affected)
        #[arg(long)]
        exclude_forfeit: bool,
        /// drop replays decided by the inactivity timer
        #[arg(long)]
        exclude_inactivity: bool,
        /// imitate only the winning side
        #[arg(long)]
        winner_only: bool,
        /// high-quality preset: rating >= 1500, >= 5 turns, no inactivity losses
        #[arg(long)]
        hq: bool,
    },
}

#[derive(Clone, Copy, Default)]
pub struct Quality {
    pub min_turns: i64,
    pub exclude_forfeit: bool,
    pub exclude_inactivity: bool,
    pub winner_only: bool,
}

fn n_turns(log: &str) -> i64 {
    log.lines().filter_map(|l| l.strip_prefix("|turn|")).filter_map(|t| t.trim().parse::<i64>().ok()).max().unwrap_or(0)
}

const BASE: &str = "https://replay.pokemonshowdown.com";

fn fetch(format: &str, min_rating: i64, max: usize, sleep_ms: u64, out: &Path) -> Result<Value> {
    let client = common::http()?;
    std::fs::create_dir_all(out)?;
    let (mut downloaded, mut listed, mut low, mut existing, mut errors) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut before: Option<i64> = None;
    let pause = || std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    'pages: while downloaded < max {
        let mut url = format!("{BASE}/search.json?format={format}");
        if let Some(b) = before {
            url.push_str(&format!("&before={b}"));
        }
        let page: Vec<Value> = match client.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.json()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[fetch] search page failed ({e}); retrying in 10s");
                std::thread::sleep(std::time::Duration::from_secs(10));
                continue;
            }
        };
        if page.is_empty() {
            break;
        }
        let has_more = page.len() >= 51;
        before = page.last().and_then(|x| x["uploadtime"].as_i64());
        for item in page.iter().take(if has_more { 50 } else { page.len() }) {
            if downloaded >= max {
                break 'pages;
            }
            listed += 1;
            let Some(id) = item["id"].as_str() else { continue };
            if min_rating > 0 && item["rating"].as_i64().unwrap_or(0) < min_rating {
                low += 1;
                continue;
            }
            let dest = out.join(format!("{id}.json"));
            if dest.exists() {
                existing += 1;
                continue;
            }
            pause();
            match client.get(format!("{BASE}/{id}.json")).send().and_then(|r| r.error_for_status()).and_then(|r| r.bytes()) {
                Ok(bts) => {
                    let tmp = dest.with_extension("tmp");
                    std::fs::write(&tmp, &bts)?;
                    std::fs::rename(&tmp, &dest)?;
                    downloaded += 1;
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("[fetch] {id} failed: {e}");
                }
            }
        }
        if !has_more || before.is_none() {
            break;
        }
        pause();
    }
    Ok(json!({"listed": listed, "downloaded": downloaded, "skipped_rating": low, "skipped_existing": existing, "errors": errors}))
}

/// Which battles a conversion keeps beyond the per-replay quality filters.
#[derive(Default)]
pub struct Select {
    pub min_rating: i64,
    /// formats to keep (empty = all)
    pub formats: Vec<String>,
    /// format → player id → (rated battles, highest battle rating)
    pub players: Option<HashMap<String, HashMap<String, (i64, i64)>>>,
    pub min_peak: i64,
    pub min_peak_games: i64,
}

impl Select {
    /// None = keep, Some(reason) = the stat the replay is dropped under
    fn verdict(&self, r: &Value, format: &str) -> Option<&'static str> {
        if !self.formats.is_empty() && !self.formats.iter().any(|f| f == format) {
            return Some("other_format");
        }
        let rating = record_rating(r);
        if self.min_rating > 0 {
            match rating {
                Some(x) if x >= self.min_rating as f64 => {}
                Some(_) => return Some("low_rating"),
                None if self.players.is_none() => return Some("unrated"),
                None => {}
            }
        }
        if let Some(table) = &self.players {
            if self.min_peak > 0 {
                let t = table.get(format);
                let ok = record_players(r).len() == 2 && record_players(r).iter().all(|p| {
                    t.and_then(|t| t.get(&norm(p))).map(|&(n, peak)| n >= self.min_peak_games && peak >= self.min_peak).unwrap_or(false)
                });
                if !ok {
                    return Some("low_peak");
                }
            }
        }
        None
    }
}

/// A replay's rating: None for unrated battles (missing, null or 0).
fn record_rating(r: &Value) -> Option<f64> {
    r.get("rating").and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).filter(|&x| x > 0.0)
}

/// The player names of a replay record: its `players` list, else the log's `|player|` lines.
fn record_players(r: &Value) -> Vec<String> {
    if let Some(a) = r.get("players").and_then(|v| v.as_array()) {
        let v: Vec<String> = a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
        if v.len() == 2 {
            return v;
        }
    }
    let mut out = vec![String::new(), String::new()];
    for l in r.get("log").and_then(|v| v.as_str()).unwrap_or("").lines().take(64) {
        let f: Vec<&str> = l.split('|').collect();
        if f.len() >= 4 && f[1] == "player" && !f[3].is_empty() {
            if let Some(i) = side_idx(f[2]) {
                out[i] = f[3].to_string();
            }
        }
    }
    out.retain(|x| !x.is_empty());
    out
}

/// A string column of a record batch (Utf8 or LargeUtf8), row by row.
fn str_col(batch: &arrow_array::RecordBatch, name: &str) -> Option<Vec<Option<String>>> {
    use arrow_array::{Array, LargeStringArray, StringArray};
    let c = batch.column_by_name(name)?;
    if let Some(a) = c.as_any().downcast_ref::<StringArray>() {
        return Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect());
    }
    if let Some(a) = c.as_any().downcast_ref::<LargeStringArray>() {
        return Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect());
    }
    let c = arrow_cast::cast(c, &arrow_schema::DataType::Utf8).ok()?;
    let a = c.as_any().downcast_ref::<StringArray>()?;
    Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect())
}

/// The `players` column: a list of names (List / LargeList of strings) or a string holding a
/// Python-style list ("['a', 'b']").
fn players_col(batch: &arrow_array::RecordBatch) -> Option<Vec<Option<Value>>> {
    use arrow_array::{Array, LargeListArray, ListArray};
    let c = batch.column_by_name("players")?;
    let names = |v: std::sync::Arc<dyn Array>| -> Option<Value> {
        let s = arrow_cast::cast(&v, &arrow_schema::DataType::Utf8).ok()?;
        let s = s.as_any().downcast_ref::<arrow_array::StringArray>()?;
        Some(json!((0..s.len()).filter(|&i| !s.is_null(i)).map(|i| s.value(i).to_string()).collect::<Vec<_>>()))
    };
    if let Some(a) = c.as_any().downcast_ref::<ListArray>() {
        return Some((0..a.len()).map(|i| if a.is_null(i) { None } else { names(a.value(i)) }).collect());
    }
    if let Some(a) = c.as_any().downcast_ref::<LargeListArray>() {
        return Some((0..a.len()).map(|i| if a.is_null(i) { None } else { names(a.value(i)) }).collect());
    }
    let strs = str_col(batch, "players")?;
    Some(strs.into_iter().map(|s| {
        s.map(|s| json!(s.trim_matches(|c| c == '[' || c == ']').split(',')
            .map(|x| x.trim().trim_matches(|c| c == '\'' || c == '"').to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>()))
    }).collect())
}

fn f64_col(batch: &arrow_array::RecordBatch, name: &str) -> Option<Vec<Option<f64>>> {
    use arrow_array::Array;
    let c = batch.column_by_name(name)?;
    let c = arrow_cast::cast(c, &arrow_schema::DataType::Float64).ok()?;
    let a = c.as_any().downcast_ref::<arrow_array::Float64Array>()?.clone();
    Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect())
}

/// Every replay record of `inputs` — directories / files of replay `.json` (the replay API's
/// objects) and parquet dumps with its columns (id, log, rating, format / formatid, players,
/// uploadtime) — each replay id once, from the first input that has it.  `f` returns false to
/// stop.  Returns (records passed on, duplicates skipped, unreadable entries).
/// A battle's identity across dumps: Showdown numbers battles per format (`gen9ou-2688866368`),
/// and a dump may re-label the id (the merged set: `sethkarten-2383111891`), so the key is the
/// format, the battle number and the two players.
fn canon_key(r: &Value) -> Option<String> {
    let id = r.get("id").and_then(|v| v.as_str())?;
    let number = id.rsplit('-').next().filter(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit()));
    let Some(number) = number else { return Some(id.to_string()) };
    let mut players: Vec<String> = record_players(r).iter().map(|p| norm(p)).collect();
    players.sort();
    Some(format!("{}-{}-{}", record_format(r), number, players.join("|")))
}

/// Duplicates of one battle across dumps whose ratings agree / differ (the first copy is kept).
pub static DUP_RATINGS: std::sync::Mutex<(usize, usize)> = std::sync::Mutex::new((0, 0));

fn iter_records(inputs: &[PathBuf], f: &mut dyn FnMut(Value) -> Result<bool>) -> Result<(usize, usize, usize)> {
    iter_records_cols(inputs, true, f)
}

/// [`iter_records`]; `need_log` false skips the parquet dumps' `log` column (most of their
/// bytes: the players pass reads metadata only; a record then has an empty `log`).
fn iter_records_cols(inputs: &[PathBuf], need_log: bool, f: &mut dyn FnMut(Value) -> Result<bool>) -> Result<(usize, usize, usize)> {
    // canonical key → the first copy's rating (rounded; −1 = unrated)
    let mut seen: HashMap<String, i64> = HashMap::new();
    let (mut n, mut dup, mut bad) = (0usize, 0usize, 0usize);
    let mut take = |r: Value, n: &mut usize, dup: &mut usize| -> Result<bool> {
        if let Some(key) = canon_key(&r) {
            let rating = record_rating(&r).map(|x| x.round() as i64).unwrap_or(-1);
            if let Some(&first) = seen.get(&key) {
                *dup += 1;
                let mut d = DUP_RATINGS.lock().unwrap();
                if first == rating { d.0 += 1 } else { d.1 += 1 }
                return Ok(true);
            }
            seen.insert(key, rating);
        }
        *n += 1;
        f(r)
    };
    for input in inputs {
        let files = common::find_files(input, &[".json", ".parquet"])?;
        for path in files {
            if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
                let bld = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                    .with_context(|| format!("parquet {}", path.display()))?;
                let schema = bld.schema().clone();
                let want = ["id", "log", "rating", "format", "formatid", "players", "uploadtime"];
                let cols: Vec<usize> = want.iter().filter(|&&c| need_log || c != "log")
                    .filter_map(|c| schema.index_of(c).ok()).collect();
                let mask = parquet::arrow::ProjectionMask::roots(bld.parquet_schema(), cols);
                for batch in bld.with_projection(mask).with_batch_size(256).build()? {
                    let batch = batch?;
                    let ids = str_col(&batch, "id");
                    let logs = if need_log {
                        str_col(&batch, "log").context("parquet replays need a `log` column")?
                    } else {
                        vec![Some(String::new()); batch.num_rows()]
                    };
                    let fmt = str_col(&batch, "format");
                    let fid = str_col(&batch, "formatid");
                    let rat = f64_col(&batch, "rating");
                    let up = f64_col(&batch, "uploadtime");
                    let pl = players_col(&batch);
                    for (i, log) in logs.into_iter().enumerate() {
                        let Some(log) = log else {
                            bad += 1;
                            continue;
                        };
                        let pick = |v: &Option<Vec<Option<String>>>| v.as_ref().and_then(|v| v[i].clone());
                        let mut r = json!({"log": log});
                        if let Some(x) = pick(&ids) { r["id"] = json!(x); }
                        if let Some(x) = pick(&fmt) { r["format"] = json!(x); }
                        if let Some(x) = pick(&fid) { r["formatid"] = json!(x); }
                        if let Some(x) = rat.as_ref().and_then(|v| v[i]) { r["rating"] = json!(x); }
                        if let Some(x) = up.as_ref().and_then(|v| v[i]) { r["uploadtime"] = json!(x as i64); }
                        if let Some(x) = pl.as_ref().and_then(|v| v[i].clone()) { r["players"] = x; }
                        if !take(r, &mut n, &mut dup)? {
                            return Ok((n, dup, bad));
                        }
                    }
                }
            } else {
                match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                    Some(v @ Value::Object(_)) if v.get("log").is_some() => {
                        if !take(v, &mut n, &mut dup)? {
                            return Ok((n, dup, bad));
                        }
                    }
                    _ => bad += 1,
                }
            }
        }
    }
    Ok((n, dup, bad))
}

/// The `players` table: per format, per player id (toID of the name), (rated battles, highest
/// battle rating).
pub fn players(inputs: &[PathBuf], formats: &[String]) -> Result<Value> {
    let mut t: BTreeMap<String, HashMap<String, (i64, i64)>> = BTreeMap::new();
    let (mut rated, mut unrated) = (0usize, 0usize);
    // per format: battles by rating (bins of 50), unrated battles
    let mut hist: BTreeMap<String, BTreeMap<i64, usize>> = BTreeMap::new();
    let mut unrated_per: BTreeMap<String, usize> = BTreeMap::new();
    let (n, dup, bad) = iter_records_cols(inputs, false, &mut |r: Value| {
        let fmt = record_format(&r);
        if !formats.is_empty() && !formats.iter().any(|f| *f == fmt) {
            return Ok(true);
        }
        let Some(x) = record_rating(&r) else {
            unrated += 1;
            *unrated_per.entry(fmt).or_default() += 1;
            return Ok(true);
        };
        rated += 1;
        *hist.entry(fmt.clone()).or_default().entry((x as i64).div_euclid(50) * 50).or_default() += 1;
        let m = t.entry(fmt).or_default();
        for p in record_players(&r) {
            let e = m.entry(norm(&p)).or_insert((0, 0));
            e.0 += 1;
            e.1 = e.1.max(x.round() as i64);
        }
        Ok(true)
    })?;
    let table: BTreeMap<&String, BTreeMap<&String, [i64; 2]>> =
        t.iter().map(|(f, m)| (f, m.iter().map(|(p, &(k, peak))| (p, [k, peak])).collect())).collect();
    let (agree, differ) = *DUP_RATINGS.lock().unwrap();
    Ok(json!({"records": n, "duplicates": dup, "duplicate_ratings_agree": agree, "duplicate_ratings_differ": differ,
              "unreadable": bad, "rated": rated, "unrated": unrated,
              "players": t.iter().map(|(f, m)| (f.clone(), m.len())).collect::<BTreeMap<_, _>>(),
              "rating_hist": hist, "unrated_per_format": unrated_per, "table": table}))
}

fn load_players(path: &Path) -> Result<HashMap<String, HashMap<String, (i64, i64)>>> {
    let v: Value = serde_json::from_slice(&std::fs::read(path).with_context(|| format!("read {}", path.display()))?)?;
    let mut out = HashMap::new();
    for (f, m) in v["table"].as_object().context("players file needs a `table`")? {
        let mut pm = HashMap::new();
        for (p, a) in m.as_object().context("table entries are objects")? {
            pm.insert(p.clone(), (a[0].as_i64().unwrap_or(0), a[1].as_i64().unwrap_or(0)));
        }
        out.insert(f.clone(), pm);
    }
    Ok(out)
}

pub fn convert(inputs: &[PathBuf], out: &Path, shard_size: usize, sel: &Select, max_replays: Option<usize>, q: Quality) -> Result<Value> {
    let voc = vocab();
    let meta = json!({"field_features": FIELD, "poke_features": FEAT, "move_vocab": voc.n_moves, "species_vocab": voc.n_species,
                      "item_vocab": voc.n_items, "ability_vocab": voc.n_abilities, "type_vocab": voc.n_types,
                      "format_vocab": FORMATS.len(), "formats": FORMATS, "vocab_version": voc.version});
    let mut w = ShardWriter::new(out, "pokemon", CONVERTER_VERSION, meta, shard_size)?;
    let mut st: BTreeMap<&str, usize> = ["seen", "kept", "samples", "frames", "no_winner", "low_rating", "unrated", "low_peak",
        "other_format", "low_quality"].into_iter().map(|k| (k, 0)).collect();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_format: BTreeMap<String, usize> = BTreeMap::new();
    let mut unk = UnkNames::default();
    let (records, duplicates, unreadable) = iter_records(inputs, &mut |r: Value| {
        if max_replays.map(|m| st["kept"] >= m).unwrap_or(false) {
            return Ok(false);
        }
        *st.get_mut("seen").unwrap() += 1;
        let fmt = record_format(&r);
        if let Some(why) = sel.verdict(&r, &fmt) {
            *st.get_mut(why).unwrap() += 1;
            return Ok(true);
        }
        let log = r["log"].as_str().unwrap_or("");
        if (q.min_turns > 0 && n_turns(log) < q.min_turns)
            || (q.exclude_forfeit && log.contains(" forfeited."))
            || (q.exclude_inactivity && log.contains("due to inactivity")) {
            *st.get_mut("low_quality").unwrap() += 1;
            return Ok(true);
        }
        let p = parse_log(log);
        for (k, v) in &p.unknown {
            *unknown.entry(k.clone()).or_default() += v;
        }
        if p.winner.is_none() {
            *st.get_mut("no_winner").unwrap() += 1;
            return Ok(true);
        }
        let ss = replay_samples(&r, &p, q.winner_only, &mut unk);
        if ss.is_empty() {
            return Ok(true);
        }
        *st.get_mut("kept").unwrap() += 1;
        *per_format.entry(fmt).or_default() += 1;
        for (k, v) in &p.stats {
            *labels.entry(k.clone()).or_default() += v;
        }
        for s in ss {
            *st.get_mut("samples").unwrap() += 1;
            *st.get_mut("frames").unwrap() += s.frames()?;
            w.push(s)?;
        }
        Ok(true)
    })?;
    w.flush()?;
    let mut v = json!(st);
    v["records"] = json!(records);
    v["duplicates"] = json!(duplicates);
    v["not_replay"] = json!(unreadable);
    v["kept_per_format"] = json!(per_format);
    v["shards"] = json!(w.written_shards);
    v["labels"] = json!(labels);
    v["unknown"] = json!(unknown);
    v["vocab"] = json!({"version": voc.version, "move_vocab": voc.n_moves, "species_vocab": voc.n_species});
    // names the vocabulary does not know (UNK ids, never merged with a known id): name → ids
    // written (frames × slots of obs.species_ids / obs.move_ids, plus tgt.move_id labels)
    v["unk_moves"] = json!(unk.moves);
    v["unk_species"] = json!(unk.species);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { format, min_rating, max_replays, sleep_ms, out } => {
            println!("{}", fetch(&format, min_rating, max_replays, sleep_ms, &out)?);
            Ok(())
        }
        Cmd::FetchHf { dataset, matches, out } => {
            println!("{}", common::fetch_hf_files(&dataset, &|name: &str| {
                name.ends_with(".parquet") && (matches.is_empty() || matches.iter().any(|m| name.contains(m.as_str())))
            }, &out)?);
            Ok(())
        }
        Cmd::Players { input, formats, out } => {
            let v = players(&input, &formats)?;
            std::fs::write(&out, serde_json::to_vec(&v)?)?;
            let mut summary = v.clone();
            summary.as_object_mut().unwrap().remove("table");
            println!("{summary}");
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, min_rating, formats, players, min_peak, min_peak_games, max_replays, min_turns,
                       exclude_forfeit, exclude_inactivity, winner_only, hq } => {
            let mut q = Quality { min_turns, exclude_forfeit, exclude_inactivity, winner_only };
            let mut sel = Select { min_rating, formats, players: None, min_peak, min_peak_games };
            if hq {
                sel.min_rating = sel.min_rating.max(1500);
                q.min_turns = q.min_turns.max(5);
                q.exclude_inactivity = true;
            }
            if let Some(p) = players {
                sel.players = Some(load_players(&p)?);
            } else if min_peak > 0 {
                anyhow::bail!("--min-peak needs --players (the table of `gmdata pokemon players`)");
            }
            println!("{}", convert(&input, &out, shard_size, &sel, max_replays, q)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: [&str; 12] = ["|player|p1|Alice|", "|player|p2|Bob|", "|poke|p1|Garchomp, M|", "|poke|p1|Ditto|",
        "|poke|p1|Corviknight, F|", "|poke|p2|Gholdengo|", "|poke|p2|Great Tusk|", "|poke|p2|Kingambit, M|", "|teampreview",
        "|start", "|switch|p1a: Garchomp|Garchomp, M|100/100", "|switch|p2a: Gholdengo|Gholdengo|100/100"];

    fn parse(body: &[&str]) -> Parse {
        let mut lines: Vec<&str> = HEAD.to_vec();
        lines.extend_from_slice(body);
        lines.push("|win|Alice");
        parse_log(&lines.join("\n"))
    }

    fn turn(r: &Rec) -> i64 {
        (r.frame.field[9] * 50.0).round() as i64
    }

    fn forced(r: &Rec) -> bool {
        r.frame.field[FLD_FORCED] == 1.0
    }

    fn preview(r: &Rec) -> bool {
        r.frame.field[FLD_PREVIEW] == 1.0
    }

    /// is the Pokémon in slot j (own 0..6 / opponent 6..12) of the frame active?
    fn active(r: &Rec, j: usize) -> bool {
        r.frame.feats[j * FEAT + F_ACTIVE] == 1.0
    }

    #[test]
    fn lead_frames_are_the_pre_lead_state() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Nasty Plot|p2a: Gholdengo"]);
        for side in 0..2 {
            let r = &p.records[side][0];
            assert!(preview(r) && !forced(r) && turn(r) == 0);
            assert_eq!((r.action_type, r.switch_idx), (1, 0));
            assert!((0..12).all(|j| !active(r, j)), "the lead frame sees no lead");
            assert_eq!(r.legal_type, [false, true]);
            assert_eq!(r.legal_switch, [true, true, true, false, false, false]);
        }
        // without team preview the lead is not a choice: no frame
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|start", "|switch|p1a: Garchomp|Garchomp, M|100/100",
            "|switch|p2a: Gholdengo|Gholdengo|100/100", "|turn|1", "|win|Alice"].join("\n");
        let p = parse_log(&log);
        assert!(p.records.iter().all(|r| r.len() == 1 && turn(&r[0]) == 1));
        assert_eq!(p.records[0][0].legal_switch, [false, true, true, true, true, true], "unknown slots may hold a Pokémon");
    }

    #[test]
    fn move_target_is_the_identity_and_the_slot_only_once_revealed() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp",
            "|upkeep", "|turn|2", "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp",
            "|upkeep", "|turn|3", "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo"]);
        let r = &p.records[0];
        assert_eq!(r.len(), 4);
        assert_eq!((r[1].action_type, r[1].move_idx, r[1].move_name.as_str()), (0, -1, "Earthquake"), "first use: no slot");
        assert!(r[1].frame.moves.iter().all(|m| m.is_empty()), "obs.move_ids is causal");
        assert_eq!((r[2].move_idx, r[2].move_name.as_str()), (-1, "Swords Dance"));
        assert_eq!(r[2].frame.moves[0], "Earthquake");
        assert_eq!((r[3].move_idx, r[3].move_name.as_str()), (0, "Earthquake"));
        assert_eq!(p.records[1][2].move_idx, 0);
    }

    #[test]
    fn unobserved_and_no_choice_turns_keep_an_unlabelled_frame() {
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|cant|p1a: Garchomp|flinch", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Outrage|p2a: Gholdengo", "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|upkeep", "|turn|3",
            "|move|p1a: Garchomp|Outrage|p2a: Gholdengo|[from]lockedmove", "|upkeep", "|turn|4",
            "|cant|p1a: Garchomp|move: Taunt|Swords Dance"]);
        let r = &p.records[0];
        assert_eq!(r.iter().map(|x| x.action_type).collect::<Vec<_>>(), [1, -1, 0, -1, 0]);
        assert_eq!(r[4].move_name, "Swords Dance", "|cant|..|MOVE names the chosen move");
        assert_eq!(p.records[1][3].action_type, -1, "no visible choice");
        assert_eq!(p.stats["unlabelled_cant"], 1);
        assert_eq!(p.stats["unlabelled_locked"], 1);
    }

    #[test]
    fn a_foe_blocked_by_an_ability_is_not_the_holders_choice() {
        // Armor Tail / Dazzling / Queenly Majesty / Damp: `|cant|HOLDER|ability: X|MOVE|[of] USER`
        // names the ability holder; MOVE is the user's (blocked) choice, logged by its own
        // `|move|..|[still]` line.  The holder acts later in the turn (or already did).
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Sucker Punch|p1a: Garchomp|[still]",
            "|cant|p1a: Garchomp|ability: Armor Tail|Sucker Punch|[of] p2a: Gholdengo",
            "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Explosion|p1a: Garchomp|[still]",
            "|cant|p1a: Garchomp|ability: Damp|Explosion|[of] p2a: Gholdengo", "|upkeep", "|turn|3"]);
        let (r, o) = (&p.records[0], &p.records[1]);
        assert_eq!((r[1].action_type, r[1].move_name.as_str()), (0, "Earthquake"), "the holder's own move");
        assert_eq!((o[1].action_type, o[1].move_name.as_str()), (0, "Sucker Punch"));
        assert_eq!((r[2].move_name.as_str(), o[2].move_name.as_str()), ("Swords Dance", "Explosion"));
        let moves = |x: &Rec| x.frame.moves.iter().filter(|m| !m.is_empty()).cloned().collect::<Vec<_>>();
        assert_eq!(moves(&r[3]), ["Earthquake", "Swords Dance"], "no foe move in the holder's moveset");
        assert_eq!(moves(&o[3]), ["Sucker Punch", "Explosion"]);
        assert!(!p.stats.contains_key("unlabelled_cant"));
    }

    #[test]
    fn an_encore_before_the_move_hides_the_choice() {
        // Encore replaces the target's chosen move with the encored one (Showdown's
        // onOverrideAction, logged as a plain |move| line): when it lands before the target
        // moved in the same turn, the logged move is not necessarily the choice — no label.
        // Once encored, the requests offer the encored move only: its uses are choices, and an
        // Encore after the target moved changes nothing about that turn.
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Stealth Rock|p2a: Gholdengo", "|-sidestart|p2: Bob|move: Stealth Rock",
            "|move|p2a: Gholdengo|Nasty Plot|p2a: Gholdengo", "|upkeep", "|turn|2", "|move|p2a: Gholdengo|Encore|p1a: Garchomp",
            "|-start|p1a: Garchomp|Encore", "|move|p1a: Garchomp|Stealth Rock|p2a: Gholdengo|[still]", "|-fail|p1a: Garchomp",
            "|upkeep", "|turn|3", "|move|p1a: Garchomp|Stealth Rock|p2a: Gholdengo|[still]", "|-fail|p1a: Garchomp",
            "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|-end|p1a: Garchomp|Encore", "|upkeep", "|turn|4",
            "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo", "|move|p2a: Gholdengo|Encore|p1a: Garchomp",
            "|-start|p1a: Garchomp|Encore", "|upkeep", "|turn|5"]);
        let r = &p.records[0];
        let labels: Vec<(i64, &str)> = r.iter().map(|x| (x.action_type, x.move_name.as_str())).collect();
        assert_eq!(labels, [(1, ""), (0, "Stealth Rock"), (-1, ""), (0, "Stealth Rock"), (0, "Earthquake"), (-1, "")]);
        assert_eq!(p.stats["unlabelled_encore_override"], 1);
        assert_eq!(r[3].frame.moves[0], "Stealth Rock", "the encored move stays revealed");
        let o = &p.records[1];
        assert_eq!((o[2].move_name.as_str(), o[4].move_name.as_str()), ("Encore", "Encore"), "the Encore user's own choices");
    }

    #[test]
    fn copied_and_swapped_boosts() {
        // |-copyboost|RECEIVER|SOURCE[|STATS] (Psych Up) copies SOURCE's stages onto RECEIVER;
        // |-swapboost|A|B[|STATS] (Guard Swap / Power Swap / Heart Swap) exchanges them — all
        // stats when no list is given (the protocol as the Showdown client reads it)
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Nasty Plot|p2a: Gholdengo", "|-boost|p2a: Gholdengo|spa|2",
            "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|-boost|p1a: Garchomp|atk|2", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Psych Up|p2a: Gholdengo", "|-copyboost|p1a: Garchomp|p2a: Gholdengo|[from] move: Psych Up",
            "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|upkeep", "|turn|3", "|move|p1a: Garchomp|Power Swap|p2a: Gholdengo",
            "|-swapboost|p1a: Garchomp|p2a: Gholdengo|atk, spa|[from] move: Power Swap", "|-boost|p2a: Gholdengo|def|1", "|upkeep",
            "|turn|4", "|move|p2a: Gholdengo|Heart Swap|p1a: Garchomp", "|-swapboost|p2a: Gholdengo|p1a: Garchomp|[from] move: Heart Swap",
            "|upkeep", "|turn|5"]);
        let boosts = |t: i64, j: usize| {
            let r = p.records[0].iter().find(|r| turn(r) == t && !forced(r)).unwrap();
            (0..5).map(|k| (r.frame.feats[j * FEAT + F_BOOSTS + k] * 6.0).round() as i64).collect::<Vec<_>>()
        };
        // [atk, def, spa, spd, spe]
        assert_eq!((boosts(2, 0), boosts(2, TEAM)), (vec![2, 0, 0, 0, 0], vec![0, 0, 2, 0, 0]));
        assert_eq!((boosts(3, 0), boosts(3, TEAM)), (vec![0, 0, 2, 0, 0], vec![0, 0, 2, 0, 0]), "Psych Up copied the stages");
        assert_eq!((boosts(4, 0), boosts(4, TEAM)), (vec![0, 0, 2, 0, 0], vec![0, 1, 2, 0, 0]), "atk and spa swapped (equal here)");
        assert_eq!((boosts(5, 0), boosts(5, TEAM)), (vec![0, 1, 2, 0, 0], vec![0, 0, 2, 0, 0]), "Heart Swap: every stat");
    }

    #[test]
    fn forced_switches_get_request_time_frames() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|U-turn|p2a: Gholdengo", "|-damage|p2a: Gholdengo|60/100",
            "|switch|p1a: Corviknight|Corviknight, F|100/100|[from] U-turn", "|move|p2a: Gholdengo|Make It Rain|p1a: Corviknight",
            "|-damage|p1a: Corviknight|0 fnt", "|faint|p1a: Corviknight", "|upkeep", "|switch|p1a: Ditto|Ditto|100/100",
            "|-transform|p1a: Ditto|p2a: Gholdengo|[from] ability: Imposter", "|turn|2"]);
        let r = &p.records[0];
        let kinds: Vec<(i64, bool, i64, i64)> = r.iter().map(|x| (turn(x), forced(x), x.action_type, x.switch_idx)).collect();
        assert_eq!(kinds, [(0, false, 1, 0), (1, false, 0, -1), (1, true, 1, 2), (1, true, 1, 1), (2, false, -1, -1)]);
        // the pivot frame is after the U-turn hit, before Corviknight comes in
        assert!(active(&r[2], 0) && r[2].frame.feats[TEAM * FEAT + F_HP] == 0.6);
        assert_eq!((r[2].legal_type, r[2].legal_switch), ([false, true], [false, true, true, false, false, false]));
        // the replacement frame sees the fainted active; Garchomp (pivoted out) or Ditto may come in
        assert!(active(&r[3], 2) && r[3].frame.feats[2 * FEAT + F_FAINTED] == 1.0);
        assert_eq!(r[3].legal_switch, [true, true, false, false, false, false]);
        assert_eq!(p.records[1].len(), 3, "the opponent answers no request");
        assert_eq!((p.stats["forced_pivot"], p.stats["forced_replacement"]), (1, 1));
    }

    #[test]
    fn simultaneous_replacements_do_not_see_each_other() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Outrage|p2a: Gholdengo", "|-damage|p2a: Gholdengo|0 fnt", "|faint|p2a: Gholdengo",
            "|-damage|p1a: Garchomp|0 fnt|[from] item: Rocky Helmet", "|faint|p1a: Garchomp", "|upkeep",
            "|switch|p2a: Kingambit|Kingambit, M|100/100", "|switch|p1a: Corviknight|Corviknight, F|100/100", "|turn|2"]);
        let (a, b) = (&p.records[0][2], &p.records[1][2]);
        assert!(forced(a) && forced(b));
        assert_eq!((a.switch_idx, b.switch_idx), (2, 2));
        // p1 answered the same request as p2: its frame does not show Kingambit (opponent slot 2) in
        assert!(!active(a, TEAM + 2) && active(a, TEAM), "p1's frame: Gholdengo is still the (fainted) opposing active");
        assert!(active(b, 0) && !active(b, 2));
    }

    #[test]
    fn pending_causes_are_scoped() {
        // an Eject Button knocked off is not activated; a U-turn blocked by Protect switches nobody
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|U-turn|p2a: Gholdengo", "|-activate|p2a: Gholdengo|move: Protect",
            "|move|p2a: Gholdengo|Knock Off|p1a: Garchomp", "|-enditem|p1a: Garchomp|Eject Button|[from] move: Knock Off|[of] p2a: Gholdengo",
            "|upkeep", "|turn|2", "|switch|p1a: Ditto|Ditto|100/100", "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto"]);
        let r = &p.records[0];
        assert_eq!(r.len(), 3);
        assert_eq!((forced(&r[2]), r[2].action_type, r[2].switch_idx), (false, 1, 1), "a voluntary switch next turn");
        // an activated Eject Button makes the holder's next switch a forced one
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|-damage|p1a: Garchomp|50/100",
            "|-enditem|p1a: Garchomp|Eject Button", "|switch|p1a: Ditto|Ditto|100/100"]);
        let r = &p.records[0];
        assert_eq!(r.iter().map(|x| (forced(x), x.action_type)).collect::<Vec<_>>(), [(false, 1), (false, -1), (true, 1)]);
        assert_eq!(p.stats["forced_eject"], 1);
    }

    #[test]
    fn court_change_and_transform() {
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Stealth Rock|p1a: Garchomp", "|-sidestart|p1: Alice|move: Stealth Rock",
            "|move|p1a: Garchomp|Court Change|p1a: Garchomp", "|-swapsideconditions", "|upkeep", "|turn|2",
            "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|upkeep", "|turn|3", "|switch|p1a: Ditto|Ditto|100/100",
            "|-transform|p1a: Ditto|p2a: Gholdengo", "|upkeep", "|turn|4", "|move|p1a: Ditto|Shadow Ball|p2a: Gholdengo", "|upkeep",
            "|turn|5", "|switch|p1a: Garchomp|Garchomp, M|100/100", "|upkeep", "|turn|6", "|switch|p1a: Ditto|Ditto|100/100", "|upkeep",
            "|turn|7"]);
        let own = |t: i64| p.records[0].iter().find(|r| turn(r) == t && !forced(r)).unwrap();
        assert_eq!((own(2).frame.field[12], own(2).frame.field[17]), (0.0, 1.0), "the rock moved to p2's side");
        let opp = p.records[1].iter().find(|r| turn(r) == 2).unwrap();
        assert_eq!((opp.frame.field[12], opp.frame.field[17]), (1.0, 0.0));
        // transformed: Gholdengo's revealed moves (+ Shadow Ball used meanwhile), then back to Ditto's own moves
        let revealed = ["Stealth Rock".to_string(), "Make It Rain".to_string(), String::new(), String::new()];
        assert_eq!(own(4).frame.moves, revealed);
        assert_eq!((own(4).move_idx, own(4).move_name.as_str()), (-1, "Shadow Ball"));
        assert_eq!(own(5).frame.moves[2], "Shadow Ball");
        assert!(own(7).frame.moves.iter().all(|m| m.is_empty()), "Ditto's own moveset stays unrevealed");
    }

    #[test]
    fn forced_frames_are_the_state_at_the_decision_commit() {
        // Server logs mark every committed decision with `|` + `|t:|` (Battle.turnLoop): the
        // switch request was answered there.  What follows is the switch action — the outgoing
        // Pokémon's switch-out effects (Natural Cure, the end of Neutralizing Gas restarting a
        // foe's Drizzle, ...) are logged BEFORE the |switch| line but after the decision.
        let status = |r: &Rec, j: usize| (0..7).find(|&k| r.frame.feats[j * FEAT + F_STATUS + k] == 1.0).unwrap();
        let rain = |r: &Rec| r.frame.field[1];
        let pivot = ["|turn|1", "|", "|t:|100", "|move|p2a: Gholdengo|Will-O-Wisp|p1a: Garchomp", "|-status|p1a: Garchomp|brn",
            "|move|p1a: Garchomp|U-turn|p2a: Gholdengo", "|-damage|p2a: Gholdengo|70/100", "|", "|t:|101",
            "|-curestatus|p1a: Garchomp|brn|[from] ability: Natural Cure|[silent]",
            "|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Gholdengo", "|switch|p1a: Corviknight|Corviknight, F|100/100",
            "|upkeep", "|turn|2"];
        let p = parse(&pivot);
        let r = &p.records[0];
        assert!(forced(&r[2]) && r[2].switch_idx == 2);
        assert_eq!((status(&r[2], 0), rain(&r[2])), (1, 0.0), "the pivot request's state: burned, no rain yet");
        assert_eq!(r[2].frame.feats[TEAM * FEAT + F_HP], 0.7, "the U-turn hit before the request");
        assert_eq!((status(&r[3], 0), rain(&r[3])), (0, 1.0), "both happened by the next turn");
        // a log without commit markers (synthetic / pre-marker): the state at the switch line
        let bare: Vec<&str> = pivot.iter().copied().filter(|l| *l != "|" && !l.starts_with("|t:|")).collect();
        let q = parse(&bare);
        assert_eq!((status(&q.records[0][2], 0), rain(&q.records[0][2])), (0, 1.0));
        // end of turn: both requests are committed together; the first switch's switch-out effects
        // reach neither frame
        let p = parse(&["|turn|1", "|", "|t:|100", "|move|p1a: Garchomp|Toxic|p2a: Gholdengo", "|-status|p2a: Gholdengo|psn",
            "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|-damage|p1a: Garchomp|0 fnt", "|faint|p1a: Garchomp",
            "|-damage|p2a: Gholdengo|40/100 psn|[from] psn", "|upkeep", "|-activate|p2a: Gholdengo|ability: Emergency Exit", "|",
            "|t:|101", "|-curestatus|p2a: Gholdengo|psn|[from] ability: Natural Cure|[silent]",
            "|switch|p2a: Kingambit|Kingambit, M|100/100", "|switch|p1a: Corviknight|Corviknight, F|100/100", "|turn|2"]);
        let (a, o) = (&p.records[0][2], &p.records[1][2]);
        assert!(forced(a) && forced(o) && a.switch_idx == 2 && o.switch_idx == 2);
        assert_eq!((status(a, TEAM), status(o, 0)), (3, 3), "Gholdengo is still poisoned at the requests");
        assert_eq!(status(&p.records[0][3], TEAM), 0);
    }

    #[test]
    fn revival_blessing_revives_without_status_or_tera() {
        // Showdown's revival (battle.ts 'revivalblessing' action) clears the status, and fainting
        // already dropped the tera state (faintMessages: `delete pokemon.terastallized`): the
        // revived Pokémon comes back at 50% with neither, in both sides' frames
        let p = parse(&["|turn|1", "|-terastallize|p1a: Garchomp|Fire", "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp",
            "|move|p2a: Gholdengo|Will-O-Wisp|p1a: Garchomp", "|-status|p1a: Garchomp|brn", "|upkeep", "|turn|2",
            "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|-damage|p1a: Garchomp|0 fnt", "|faint|p1a: Garchomp", "|upkeep",
            "|switch|p1a: Corviknight|Corviknight, F|100/100", "|turn|3", "|move|p1a: Corviknight|Revival Blessing|p1a: Corviknight",
            "|-heal|p1: Garchomp|50/100|[from] move: Revival Blessing", "|upkeep", "|turn|4"]);
        let feats = |r: &Rec, j: usize| r.frame.feats[j * FEAT..(j + 1) * FEAT].to_vec();
        let (own, opp) = (p.records[0].last().unwrap(), p.records[1].last().unwrap());
        assert_eq!(turn(own), 4);
        for f in [feats(own, 0), feats(opp, TEAM)] {
            assert_eq!((f[F_HP], f[F_FAINTED]), (0.5, 0.0), "revived at 50%");
            assert_eq!(f[F_STATUS], 1.0, "no status after the revival (the burn was cleared)");
            assert_eq!(f[F_TERA], 0.0, "no tera after fainting");
        }
        assert_eq!(own.legal_switch, [true, true, false, false, false, false], "the revived Garchomp may come in");
    }

    #[test]
    fn split_secret_lines_are_not_read() {
        let secret = ["|turn|1", "|split|p1", "|switch|p1a: Ditto|Ditto|250/250", "|switch|p1a: Ditto|Ditto|100/100",
            "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto", "|split|p1", "|-damage|p1a: Ditto|123/250", "|-damage|p1a: Ditto|49/100",
            "|upkeep", "|turn|2"];
        let public = ["|turn|1", "|switch|p1a: Ditto|Ditto|100/100", "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto",
            "|-damage|p1a: Ditto|49/100", "|upkeep", "|turn|2"];
        let (p, q) = (parse(&secret), parse(&public));
        for side in 0..2 {
            assert_eq!(p.records[side].len(), q.records[side].len());
            for (x, y) in p.records[side].iter().zip(&q.records[side]) {
                assert_eq!((&x.frame.feats, x.action_type, x.switch_idx), (&y.frame.feats, y.action_type, y.switch_idx));
            }
        }
        assert_eq!(p.records[0].iter().filter(|r| r.action_type == 1).count(), 2, "the lead + one switch, no phantom frame");
    }

    #[test]
    fn fixture_replays_convert_without_mismatches() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/pokemon");
        let mut frames = 0;
        let mut unk = UnkNames::default();
        for e in std::fs::read_dir(dir).unwrap() {
            let r: Value = serde_json::from_slice(&std::fs::read(e.unwrap().path()).unwrap()).unwrap();
            let p = parse_log(r["log"].as_str().unwrap());
            for k in ["mask_mismatch", "forced_other", "unlabelled_unanswered_request"] {
                assert!(!p.stats.contains_key(k), "{k} in {}", r["id"]);
            }
            for recs in &p.records {
                frames += recs.len();
                for x in recs {
                    assert!(x.action_type < 0 || x.legal_type[x.action_type as usize]);
                    assert!(x.action_type != 1 || x.legal_switch[x.switch_idx as usize]);
                    assert!(x.move_idx < 0 || !x.frame.moves[x.move_idx as usize].is_empty());
                }
            }
            assert_eq!(replay_samples(&r, &p, false, &mut unk).len(), 2);
        }
        assert!(frames > 1500, "{frames}");
        assert!(unk.moves.is_empty() && unk.species.is_empty(), "UNK names: {:?} {:?}", unk.moves, unk.species);
    }

    fn bools(s: &Sample, name: &str) -> Vec<bool> {
        match &s.tensors.iter().find(|t| t.name == name).expect(name).col {
            gmshard::Column::Bool(v) => v.iter().map(|&x| x != 0).collect(),
            _ => panic!("{name} is not a bool tensor"),
        }
    }

    #[test]
    fn revealed_items_abilities_and_tera_types_reach_both_views() {
        // HEAD: p1 Garchomp + Ditto, p2 Gholdengo + ... (team preview); items / abilities by
        // their holders (an [of] source holds its [from] effect), tera type, Knock Off
        let p = parse(&["|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Gholdengo", "|turn|1",
            "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo",
            "|-damage|p1a: Garchomp|80/100|[from] item: Rocky Helmet|[of] p2a: Gholdengo",
            "|-heal|p1a: Garchomp|86/100|[from] item: Leftovers",
            "|-terastallize|p2a: Gholdengo|Fairy", "|move|p2a: Gholdengo|Knock Off|p1a: Garchomp",
            "|-enditem|p1a: Garchomp|Leftovers|[from] move: Knock Off|[of] p2a: Gholdengo",
            "|upkeep", "|turn|2"]);
        let mut unk = UnkNames::default();
        let s = replay_samples(&json!({"id": "t"}), &p, false, &mut unk);
        let v = vocab();
        let (item, ability, tera) = (|n: &str| Vocab::named_id(&v.items, n), |n: &str| Vocab::named_id(&v.abilities, n),
                                     |n: &str| Vocab::named_id(&v.types, n));
        // frames: preview, turn 1, turn 2; slot j of frame f at f * 12 + j
        // (side, g, h): the slots of p1's lead Garchomp / p2's lead Gholdengo in that side's
        // view (own slots 0..6, the opponent's 6..12)
        for (side, g, h) in [(0usize, 0usize, 6usize), (1, 6, 0)] {
            let (it, ni, ab, te) = (i64s(&s[side], "obs.slot_item"), bools(&s[side], "obs.slot_no_item"),
                                    i64s(&s[side], "obs.slot_ability"), i64s(&s[side], "obs.slot_tera_type"));
            let mv = i64s(&s[side], "obs.slot_moves");
            // turn 1: Drizzle was already revealed; nothing else yet
            assert_eq!((ab[12 + h], it[12 + h], it[12 + g], te[12 + h]), (ability("Drizzle"), 0, 0, 0));
            // turn 2: Rocky Helmet on its [of] source, Leftovers knocked off, the tera type
            assert_eq!((it[24 + h], it[24 + g], ni[24 + g], ni[24 + h]), (item("Rocky Helmet"), item("Leftovers"), true, false));
            assert_eq!((te[24 + h], te[24 + g]), (tera("Fairy"), 0));
            assert_eq!((mv[(24 + g) * 4], mv[(24 + h) * 4]), (v.move_id("Earthquake"), v.move_id("Knock Off")));
        }
        assert!(item("Rocky Helmet") > ID_UNK && ability("Drizzle") > ID_UNK && tera("Fairy") > ID_UNK);
    }

    fn i64s(s: &Sample, name: &str) -> Vec<i64> {
        match &s.tensors.iter().find(|t| t.name == name).expect(name).col {
            gmshard::Column::I64(v) => v.clone(),
            _ => panic!("{name} is not an i64 tensor"),
        }
    }

    #[test]
    fn vocabulary_round_trips() {
        let v = vocab();
        let doc: Value = serde_json::from_str(VOCAB_JSON).unwrap();
        let (moves, species) = (doc["moves"].as_array().unwrap(), doc["species"].as_array().unwrap());
        assert_eq!((v.n_moves, v.n_species), (moves.len() + 2, species.len() + 2));
        assert!(v.n_moves > 900 && v.n_species > 1400 && v.version.starts_with("ps-"));
        // every entry's display name and key give its own id (ids = table index + 2, all distinct)
        for (i, e) in moves.iter().enumerate() {
            let (key, name) = (e[0].as_str().unwrap(), e[1].as_str().unwrap());
            assert_eq!((v.move_id(name), v.move_id(key)), (i as i64 + 2, i as i64 + 2), "{name}");
        }
        for (i, e) in species.iter().enumerate() {
            assert_eq!(v.species_id(e[1].as_str().unwrap()), i as i64 + 2, "{}", e[1]);
        }
        for a in doc["species_aliases"].as_array().unwrap() {
            assert_eq!(v.species[a[0].as_str().unwrap()], v.species[a[1].as_str().unwrap()], "{a}");
        }
        // reserved ids: the empty name, a name the vocabulary does not know
        assert_eq!((v.move_id(""), v.move_id("Zz Sentinel Move"), v.species_id(" "), v.species_id("Zzmon")), (0, 1, 0, 1));
        // formes that play differently are distinct (the team-preview placeholder too) ...
        let distinct = |names: &[&str]| names.iter().map(|n| v.species_id(n)).collect::<std::collections::BTreeSet<_>>().len();
        let formes = ["Rotom", "Rotom-Wash", "Rotom-Heat", "Urshifu", "Urshifu-Rapid-Strike", "Urshifu-*", "Ogerpon",
            "Ogerpon-Wellspring", "Ogerpon-Wellspring-Tera", "Ogerpon-Teal-Tera", "Palafin", "Palafin-Hero"];
        assert_eq!(distinct(&formes), formes.len());
        assert!(formes.iter().all(|n| v.species_id(n) > ID_UNK));
        // ... cosmetic formes and other spellings are not
        assert_eq!(v.species_id("Gastrodon-East"), v.species_id("Gastrodon"));
        assert_eq!(v.species_id("Vivillon-Polar"), v.species_id("Vivillon"));
        assert_eq!(v.species_id("Flabe\u{301}be\u{301}"), v.species_id("Flab\u{e9}b\u{e9}"));
        assert_eq!(v.species_id("Nidoran♀"), v.species_id("Nidoran-F"));
        assert_ne!(v.species_id("Nidoran♀"), v.species_id("Nidoran♂"));
        assert_eq!(v.move_id("U-turn"), v.move_id("uturn"));
        // converter v2's crc32 buckets merged these (1024 move / 512 species buckets)
        assert_ne!(v.move_id("Swords Dance"), v.move_id("Fire Blast"));
        assert_eq!(distinct(&["Dragonite", "Hoopa-Unbound", "Primarina", "Rillaboom"]), 4);
        let mv = ["Earthquake", "Tidy Up", "Aura Sphere"].map(|m| v.move_id(m));
        assert!(mv[0] != mv[1] && mv[1] != mv[2] && mv[0] != mv[2]);
    }

    #[test]
    fn species_show_the_forme_the_log_revealed() {
        // team preview hides Urshifu's / Zamazenta's forme; the switch-in names it, and a
        // permanent forme change (the Ogerpon tera forme) holds until the faint regression
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|poke|p1|Urshifu-*, M|", "|poke|p1|Ogerpon-Wellspring, F|",
            "|poke|p2|Zamazenta-*|", "|poke|p2|Gholdengo|", "|teampreview", "|start",
            "|switch|p1a: Urshifu|Urshifu-Rapid-Strike, M|100/100", "|switch|p2a: Gholdengo|Gholdengo|100/100", "|turn|1",
            "|switch|p1a: Blaise|Ogerpon-Wellspring, F|100/100", "|move|p2a: Gholdengo|Make It Rain|p1a: Blaise", "|upkeep",
            "|turn|2", "|-terastallize|p1a: Blaise|Water", "|detailschange|p1a: Blaise|Ogerpon-Wellspring-Tera, F, tera:Water",
            "|move|p1a: Blaise|Ivy Cudgel|p2a: Gholdengo", "|move|p2a: Gholdengo|Make It Rain|p1a: Blaise", "|upkeep", "|turn|3",
            "|move|p2a: Gholdengo|Make It Rain|p1a: Blaise", "|-damage|p1a: Blaise|0 fnt", "|faint|p1a: Blaise",
            "|detailschange|p1: Blaise|Ogerpon-Wellspring, F|[silent]", "|upkeep",
            "|switch|p1a: Urshifu|Urshifu-Rapid-Strike, M|100/100", "|turn|4", "|win|Alice"].join("\n");
        let p = parse_log(&log);
        let sp = |side: usize| p.records[side].iter().map(|r| r.frame.species[..8].to_vec()).collect::<Vec<_>>();
        let (a, b) = (sp(0), sp(1));
        let (u, o, z) = ("Urshifu-Rapid-Strike", "Ogerpon-Wellspring", "Zamazenta-*");
        let want: [[&str; 3]; 6] = [["Urshifu-*", o, z], [u, o, z], [u, o, z], [u, "Ogerpon-Wellspring-Tera", z], [u, o, z], [u, o, z]];
        assert_eq!(a.len(), 6, "lead, turns 1-3, the replacement, turn 4");
        for (f, w) in want.iter().enumerate() {
            assert_eq!([a[f][0].as_str(), a[f][1].as_str(), a[f][6].as_str()], *w, "p1 frame {f}");
        }
        assert_eq!((b[0][6].as_str(), b[1][6].as_str(), b[0][0].as_str()), ("Urshifu-*", u, z), "p2's view");
        // the ids are the vocabulary's, the placeholder its own
        let mut unk = UnkNames::default();
        let s = replay_samples(&json!({"id": "t"}), &p, false, &mut unk);
        let ids = i64s(&s[0], "obs.species_ids");
        let v = vocab();
        assert_eq!(ids[..2], [v.species_id("Urshifu-*"), v.species_id(o)]);
        assert_eq!(ids[12..14], [v.species_id(u), v.species_id(o)]);
        assert_eq!(ids[3 * 12 + 1], v.species_id("Ogerpon-Wellspring-Tera"));
        assert_ne!(v.species_id("Urshifu-*"), v.species_id("Urshifu"));
        assert!(unk.species.is_empty() && unk.moves.is_empty());
    }

    #[test]
    fn unknown_names_map_to_unk_and_are_counted() {
        // a move / species newer than the vocabulary: UNK (id 1), never another move's id
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|poke|p1|Garchomp, M|", "|poke|p2|Zzmon|", "|teampreview",
            "|start", "|switch|p1a: Garchomp|Garchomp, M|100/100", "|switch|p2a: Zzmon|Zzmon|100/100", "|turn|1",
            "|move|p1a: Garchomp|Zz Quake|p2a: Zzmon", "|move|p2a: Zzmon|Splash|p2a: Zzmon", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Zz Quake|p2a: Zzmon", "|upkeep", "|turn|3", "|win|Alice"].join("\n");
        let p = parse_log(&log);
        let mut unk = UnkNames::default();
        let s = replay_samples(&json!({"id": "t"}), &p, false, &mut unk);
        let v = vocab();
        assert_eq!(i64s(&s[0], "tgt.move_id"), [-1, ID_UNK, ID_UNK, -1]);
        assert_eq!(i64s(&s[1], "tgt.move_id")[1], v.move_id("Splash"));
        assert_eq!(i64s(&s[0], "obs.move_ids")[2 * 4..3 * 4], [ID_UNK, 0, 0, 0], "the revealed unknown move");
        let own = i64s(&s[0], "obs.species_ids");
        assert!((0..4).all(|f| own[f * 12] == v.species_id("Garchomp") && own[f * 12 + 6] == ID_UNK));
        // counted per name: 2 labels + 2 revealed-move slots (turns 2, 3) + the per-slot
        // revealed moves (v4: turns 2, 3, both sides' frames); 4 frames × 2 sides
        assert_eq!(unk.moves, BTreeMap::from([("Zz Quake".to_string(), 8)]));
        assert_eq!(unk.species, BTreeMap::from([("Zzmon".to_string(), 8)]));
        // the opponent's revealed move reaches the other side's frames too (v4)
        let opp = i64s(&s[1], "obs.slot_moves");
        assert_eq!(opp[2 * 48 + 6 * 4..2 * 48 + 6 * 4 + 4], [ID_UNK, 0, 0, 0], "p1's move in p2's view");
        // winner_only: only the winner's sample, and only its names are counted
        let mut unk = UnkNames::default();
        assert_eq!(replay_samples(&json!({"id": "t"}), &p, true, &mut unk).len(), 1);
        assert_eq!((unk.moves["Zz Quake"], unk.species["Zzmon"]), (6, 4));
    }

    fn f32s(s: &Sample, name: &str) -> Vec<f32> {
        match &s.tensors.iter().find(|t| t.name == name).expect(name).col {
            gmshard::Column::F32(v) => v.clone(),
            _ => panic!("{name} is not an f32 tensor"),
        }
    }

    #[test]
    fn levels_and_formats() {
        assert_eq!(details_level("Garchomp, L80, M"), Some(80));
        assert_eq!(details_level("Pikachu, L5"), Some(5));
        assert_eq!(details_level("Great Tusk"), None);
        assert_eq!(details_level("Landorus-Therian, M, shiny"), None);
        for (raw, id) in [("[Gen 9] OU", "gen9ou"), ("[Gen 9] Random Battle", "gen9randombattle"),
            ("[Gen 9] RANDOMBATTLE (BLITZ)", "gen9randombattleblitz"), ("[Gen 9] Ubers UU", "gen9ubersuu")] {
            assert_eq!(record_format(&json!({"format": raw})), id);
        }
        assert_eq!(record_format(&json!({"formatid": "gen9uu", "format": "[Gen 9] OU"})), "gen9uu");
        assert_eq!(format_id("gen9ou"), 1);
        assert_eq!(format_id("gen9nationaldex"), 0);
        assert_eq!(format_id(""), 0);
        // a random battle: no team preview, per-Pokémon levels on the switch lines
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|start", "|switch|p1a: Garchomp|Garchomp, L80, M|100/100",
            "|switch|p2a: Snorlax|Snorlax, L88, F|100/100", "|turn|1", "|move|p1a: Garchomp|Earthquake|p2a: Snorlax",
            "|move|p2a: Snorlax|Rest|p2a: Snorlax", "|upkeep", "|turn|2", "|win|Alice"].join("\n");
        let p = parse_log(&log);
        let mut unk = UnkNames::default();
        let s = replay_samples(&json!({"id": "gen9randombattle-1", "formatid": "gen9randombattle"}), &p, false, &mut unk);
        let f = f32s(&s[0], "obs.poke_feats");
        let first = &f[..2 * TEAM * FEAT];
        assert_eq!(first[F_LEVEL], 0.8, "own Garchomp");
        assert_eq!(first[TEAM * FEAT + F_LEVEL], 0.88, "the opponent's Snorlax");
        assert_eq!(first[FEAT + F_LEVEL], 0.0, "an unknown slot");
        assert!(i64s(&s[0], "obs.format_id").iter().all(|&x| x == format_id("gen9randombattle")));
        // team preview levels (Little Cup) reach the unrevealed slots too
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|poke|p1|Pawniard, L5, M|", "|poke|p1|Mudbray, L5|",
            "|poke|p2|Foongus, L5|", "|teampreview", "|start", "|switch|p1a: Pawniard|Pawniard, L5, M|100/100",
            "|switch|p2a: Foongus|Foongus, L5|100/100", "|turn|1", "|win|Bob"].join("\n");
        let p = parse_log(&log);
        let s = replay_samples(&json!({"id": "x", "format": "[Gen 9] LC"}), &p, false, &mut unk);
        let f = f32s(&s[0], "obs.poke_feats");
        assert_eq!((f[F_LEVEL], f[FEAT + F_LEVEL]), (0.05, 0.05));
        assert!(i64s(&s[0], "obs.format_id").iter().all(|&x| x == format_id("gen9lc")));
    }

    fn write_parquet(path: &Path, rows: &[(&str, &str, Option<f64>, &str, [&str; 2])]) {
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
        use std::sync::Arc;
        let mut pl = ListBuilder::new(StringBuilder::new());
        for r in rows {
            for n in r.4 {
                pl.values().append_value(n);
            }
            pl.append(true);
        }
        let cols: Vec<(&str, ArrayRef)> = vec![
            ("id", Arc::new(StringArray::from(rows.iter().map(|r| r.0).collect::<Vec<_>>()))),
            ("log", Arc::new(StringArray::from(rows.iter().map(|r| r.1).collect::<Vec<_>>()))),
            ("rating", Arc::new(Float64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>()))),
            ("format", Arc::new(StringArray::from(rows.iter().map(|r| r.3).collect::<Vec<_>>()))),
            ("players", Arc::new(pl.finish())),
        ];
        let batch = RecordBatch::try_from_iter(cols).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn battle(p1: &str, p2: &str) -> String {
        [format!("|player|p1|{p1}|"), format!("|player|p2|{p2}|"), "|poke|p1|Garchomp, M|".into(), "|poke|p2|Gholdengo|".into(),
         "|teampreview".into(), "|start".into(), "|switch|p1a: Garchomp|Garchomp, M|100/100".into(),
         "|switch|p2a: Gholdengo|Gholdengo|100/100".into(), "|turn|1".into(), "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo".into(),
         "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp".into(), "|upkeep".into(), "|turn|2".into(), format!("|win|{p1}")].join("\n")
    }

    #[test]
    fn parquet_dumps_dedupe_select_and_count_players() {
        let d = std::env::temp_dir().join(format!("gm-pokemon-pq-{}", std::process::id()));
        std::fs::create_dir_all(d.join("b")).unwrap();
        let (ab, cd, ae, ce) = (battle("Alice", "Bob"), battle("Carol", "Dan"), battle("Alice", "Eve"), battle("Carol", "Eve"));
        write_parquet(&d.join("a.parquet"), &[
            ("gen9ou-1", &ab, Some(1720.0), "[Gen 9] OU", ["Alice", "Bob"]),
            ("gen9ou-2", &cd, Some(1400.0), "[Gen 9] OU", ["Carol", "Dan"]),
            ("gen9ou-3", &ae, None, "[Gen 9] OU", ["Alice", "Eve"]),
            ("gen9uu-4", &ce, Some(1650.0), "[Gen 9] UU", ["Carol", "Eve"]),
        ]);
        // a second dump repeating gen9ou-1 (another rating spelling) plus Eve's rated battle
        write_parquet(&d.join("b/c.parquet"), &[
            ("gen9ou-1", &ab, Some(9999.0), "gen9ou", ["Alice", "Bob"]),
            ("gen9ou-5", &battle("Eve", "Bob"), Some(1810.0), "gen9ou", ["Eve", "Bob"]),
        ]);
        let inputs = vec![d.join("a.parquet"), d.join("b")];
        let v = players(&inputs, &["gen9ou".to_string()]).unwrap();
        assert_eq!((v["records"].as_i64(), v["duplicates"].as_i64()), (Some(5), Some(1)));
        assert_eq!((v["rated"].as_i64(), v["unrated"].as_i64()), (Some(3), Some(1)));
        let t = &v["table"]["gen9ou"];
        assert_eq!(t["alice"], json!([1, 1720]));
        assert_eq!(t["bob"], json!([2, 1810]), "the first copy's rating, the duplicate skipped");
        assert_eq!(t["eve"], json!([1, 1810]));
        assert!(v["table"]["gen9uu"].is_null());
        assert_eq!(v["rating_hist"]["gen9ou"], json!({"1700": 1, "1800": 1, "1400": 1}));
        assert_eq!(v["unrated_per_format"]["gen9ou"], json!(1));
        let pf = d.join("players.json");
        std::fs::write(&pf, serde_json::to_vec(&v).unwrap()).unwrap();
        let conv = |sel: &Select| {
            let out = d.join("out");
            std::fs::remove_dir_all(&out).ok();
            convert(&inputs, &out, 64, sel, None, Quality::default()).unwrap()
        };
        // rating only: the rated OU battles >= 1500 (1 and 5); unrated 3 and the 1400 one out
        let st = conv(&Select { min_rating: 1500, formats: vec!["gen9ou".into()], ..Default::default() });
        assert_eq!((st["kept"].as_i64(), st["low_rating"].as_i64(), st["unrated"].as_i64()), (Some(2), Some(1), Some(1)));
        assert_eq!((st["other_format"].as_i64(), st["duplicates"].as_i64()), (Some(1), Some(1)));
        // + peaks >= 1700: the unrated battle 3 passes too (Alice 1720, Eve 1810)
        let sel = Select { min_rating: 1500, formats: vec!["gen9ou".into()], players: Some(load_players(&pf).unwrap()),
                           min_peak: 1700, min_peak_games: 1 };
        let st = conv(&sel);
        assert_eq!((st["kept"].as_i64(), st["low_peak"].as_i64()), (Some(3), Some(0)), "{st}");
        // peaks >= 1750: Alice (1720) is out of battles 1 and 3, rated or not
        let st = conv(&Select { min_peak: 1750, ..Select { players: Some(load_players(&pf).unwrap()), ..sel } });
        assert_eq!((st["kept"].as_i64(), st["low_peak"].as_i64()), (Some(1), Some(2)), "{st}");
        let sel = Select { min_rating: 1500, formats: vec!["gen9ou".into()], players: Some(load_players(&pf).unwrap()),
                           min_peak: 1700, min_peak_games: 1 };
        let sel = Select { min_peak_games: 2, ..sel };
        let st = conv(&sel);
        assert_eq!(st["kept"].as_i64(), Some(0), "only Bob has two rated battles: {st}");
        std::fs::remove_dir_all(&d).ok();
    }
}
