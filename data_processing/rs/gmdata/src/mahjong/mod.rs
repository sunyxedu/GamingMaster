//! Riichi mahjong: mjai event streams (Tenhou logs) → shards for the mahjong adapter.
//!
//! One sample per (game, seat): that seat's decision frames over the game, in log order:
//!   own draw    after each own tsumo (label: dahai / riichi / kan / hora / ryukyoku)
//!   after call  after an own chi / pon (the forced discard)
//!   claim       after another seat's discard when the seat may do anything besides pass
//!               (chi / pon / daiminkan / ron), and after another seat's kakan when it may
//!               rob it (chankan); label: the call or ron it made, else pass
//! State: hands, reds, melds, ponds, riichi, dora, scores, wall.
//!
//!   raw.tiles_q     [T,34,37] u8    obs.players_seq [T,4,12] f32   obs.table_seq [T,12] f32
//!   tgt.action_type [T] i64         tgt.tile_idx [T] i64           tgt.value [T] f32
//!   tgt.legal_action_type [T,8]     tgt.legal_tile_of [T,34]       tgt.il_weight [T]
//!   (the last three: packed bits → f32 {0,1})
//! raw.tiles_q holds the tile planes as exact quarters (every tile feature is a count / 4 or
//! a 0/1 flag: value × 4); the loader expands it to the adapter's float32 obs.tiles_seq
//! (data_processing/games/mahjong.py), a quarter of the disk space.
//! value = final placement 1st +1, 2nd +1/3, 3rd −1/3, 4th −1 (ties → lower seat).
//!
//! converter_version 4 (shard meta {"tile_features": 37}): each tile kind's plane gains
//! every opponent's public discard history (7 flags per opponent, shimocha, toimen,
//! kamicha): it discarded the kind ever (called ones included: its furiten set), passed it
//! after its riichi declaration, passed it since its own last discard (same-turn furiten),
//! it is its latest discard, among its last 6, its latest discard of it was tsumogiri,
//! it is its riichi declaration tile. The per-kind counts of v3 lose the order of the
//! discards, which is what the safety of a tile against a riichi and a hand read rest on.
//!
//! Information (converter_version 3).  A frame — its observation, its masks and whether it
//! exists at all — depends only on the log up to its decision point (the tsumo, the chi /
//! pon, the other seat's dahai / kakan) and never on what another seat keeps hidden; only
//! its label and loss weight look at the response (tests/test_mahjong_leak_audit.py,
//! tests/test_audit_mahjong_frame_selection.py).  Hence, unlike converter_version 2:
//!   * claim frames exist by legality, not by the response (v2 emitted a claim frame
//!     without a chi / pon option only when the seat ronned: the frame was the label);
//!   * the claim frames on a riichi declaration tile precede its reach_accepted (the stick
//!     is deposited only if nobody rons);
//!   * the rinshan-draw frame after a daiminkan / kakan does not show that kan's dora:
//!     Tenhou flips it with the drawer's discard (明槓/加槓は後めくり), and the claim frames
//!     on that discard show it whichever side of the dahai the log writes the dora event
//!     (both spellings of the Houou corpus convert to identical samples);
//!   * tgt.il_weight = 0 on a recorded pass that does not show the seat's choice because a
//!     call it could make was pre-empted by a higher-priority one — another seat's ron
//!     pre-empts chi / pon / kan, a pon / daiminkan pre-empts chi, a ron is never pre-empted
//!     (Tenhou allows the double ron).  The label stays pass; the mahjong decoder's il_loss
//!     honours the weight (older shards have none: weight 1);
//!   * a draw straight after a discard with live-wall tiles left and no abortive draw
//!     pending (or straight after a kakan) can only be the triple ron (三家和), which the
//!     logs write as a bare ryukyoku: the claim frames on that tile get weight 0 (their
//!     outcome is unknown).  After the last discard the draw is the exhaustive one, unless
//!     all three other seats could ron that tile (the triple ron again: three legal rons all
//!     declined is no real alternative).
//!
//! Legality, exact under Tenhou's rules (https://tenhou.net/man/): `legal_action_type` = the
//! action types legal at the frame, `legal_tile_of` = the legal tile arguments of the
//! GROUND-TRUTH action type (all true when that type takes no tile: hora / ryukyoku / pass).
//!   own draw    dahai: any held kind (in riichi only the drawn kind); riichi: closed, not
//!               yet in riichi, ≥ 1000 points, ≥ 4 live-wall tiles, and a discard that leaves
//!               tenpai (tiles = those discards; a wait on a fifth copy counts only while
//!               the concealed tiles do not hold all four); kan: fewer than four kans on the
//!               table and a live-wall tile left — ankan (four in hand) / kakan (own pon +
//!               the fourth in hand), in riichi only an ankan of the drawn tile that keeps
//!               the waits; hora: a winning shape with a yaku (menzen tsumo, haitei, rinshan,
//!               tenhou / chiihou, ...; furiten never blocks a tsumo); ryukyoku: kyushu
//!               kyuhai (own first draw, no call or kan yet, ≥ 9 terminal/honor kinds).
//!   after call  dahai only, minus kuikae (the called kind; for a chi called at an end of
//!               the run also the kind beyond its other end).
//!   claim       pass always.  Unless the seat is in riichi, the discard is the last one
//!               (houtei) or an abortive draw is pending (four winds, four kans by two or
//!               more seats, four riichi): chi (shimocha: run bases with both partners in
//!               hand), pon (≥ 2 held), daiminkan (≥ 3 held, fewer than four kans), each only
//!               if a non-kuikae tile is left to discard.  Hora: a winning shape with a yaku
//!               (riichi, houtei, chankan, yakuhai, tanyao, ...) and no furiten — no wait among
//!               the seat's own discards (called ones included), no winning tile let pass
//!               since its last discard (with or without a yaku: same-turn furiten), none
//!               since its riichi.  A kakan may be robbed (chankan: ron or pass); an ankan by
//!               nobody (Tenhou: not even kokushi).
//! Yaku (the one-han minimum, yakuman included) come from the hand evaluator of RiichiEnv
//! (crate riichienv-core, Apache-2.0); the whole legality is differential-tested against its
//! game engine replaying the fixture games (reference.rs).  A frame whose ground truth falls
//! outside its masks keeps all-true masks and counts as `mask_mismatch` (0 on real logs).
//!
//! Input: a .zip of .mjson(.gz) files, a directory of .mjson(.gz), one file, or Parquet
//! files / directories with an `events` column (one mjai game per row; optional `game_id`,
//! `kyoku_first`); several `--input`s are read in order.  Game length (--hanchan-only,
//! --hq): the `kyoku_first` column, else the log's start_game `kyoku_first` (0 = hanchan,
//! 4 = tonpuusen) — never how far the game went (a hanchan can end in the East round, a
//! tonpuusen extend into the South); a game without it is skipped as `unknown_length`.
//! `--dedup` (in --hq) drops a game whose first kyoku deals the same hands, dora and scores
//! as an earlier one: the same game written by another converter.
//!
//! The Tenhou Houou corpus (`gmdata mahjong fetch`): the Hugging Face dataset
//! hhim8826/tenhou-houou-mjai — 2,120,954 four-player phoenix-room games 2015-01..2026-09
//! (2,091,079 hanchan) in 24 parquet shards, 6.3 GB; columns game_id, source, kyoku_first,
//! aka_flag, n_events, events.  Its sources write an open kan's dora before (nanfeng,
//! tenhou_to_mjai) or after (dataset, tenhou_new) the discard that reveals it — handled
//! above — and 48,196 games appear in both spellings (dropped by --dedup).  Licence `other`:
//! the logs are Tenhou's, whose 牌譜 terms (https://tenhou.net/sc/raw/?old) restrict
//! redistribution and uses unrelated to play on Tenhou, and the dataset grants nothing
//! beyond them — keep the raw data and the shards private.
//!
//! Full-scale conversion (schema v3: every earlier mahjong shard must be regenerated).  One
//! Houou hanchan gives ~660 frames of ~0.8 KB: the whole corpus (~2.04M hanchan after
//! --dedup) is ~1.35e9 frames, ~1.1 TB of shards (~540 KB per game).  On the cluster (gmdata
//! built with `cargo build --release` in data_processing/), holding out the last shard:
//!
//!     G=data_processing/target/release/gmdata R=~/gm-data/mahjong/raw O=~/gm-data/mahjong-v3
//!     $G mahjong fetch --out $R --all
//!     $G mahjong convert --hq --shard-size 1024 --out $O/val --input $R/tenhou-release-00005.parquet
//!     $G mahjong convert --hq --shard-size 1024 --out $O/train \
//!        --input $R/tenhou-0*.parquet $R/tenhou-new-*.parquet $R/tenhou-release-0000[0-4].parquet
//!
//! (val: 21.7k tenhou_to_mjai games that no other shard repeats; 1024 samples per shard keep
//! the file count NFS-friendly, ~8k shard directories; `--max-games N` converts only the
//! first N games of the inputs, e.g. to fit a quota; `--stats-only` previews the counts.)

mod game;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod sim_parity;
use gmmahjong::{rules, state};

use crate::common;
use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, StringArray};
use clap::Subcommand;
use game::{convert_game, FrameStats};
use gmshard::{Sample, ShardWriter};
use rayon::prelude::*;
use rules::{is_red, tile_kind};
use serde_json::{json, Value};
use state::{ACTION_NAMES, TILE_FEATURES};
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};

const CONVERTER_VERSION: u32 = 4;
const HF_DATASET: &str = "hhim8826/tenhou-houou-mjai";

/// One game read from the input: its name, mjai text and the `kyoku_first` column (parquet).
struct RawGame {
    name: String,
    text: String,
    kyoku_first: Option<i64>,
}

/// A game in the conversion queue: to convert, or filtered out by its announced length.
enum Queued {
    Game(RawGame),
    Tonpuusen,
    UnknownLength,
}

/// Every game of `input`, in order; `f` returns false to stop.
fn iter_games(input: &Path, f: &mut dyn FnMut(RawGame) -> Result<bool>) -> Result<bool> {
    let is_parquet = |p: &Path| p.extension().map(|e| e == "parquet").unwrap_or(false);
    if input.is_file() && !is_parquet(input) {
        let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
        if bytes.starts_with(b"PK") {
            let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
            for i in 0..z.len() {
                let mut m = z.by_index(i)?;
                let name = m.name().to_string();
                if m.is_dir() || !(name.ends_with(".mjson") || name.ends_with(".mjson.gz")) {
                    continue;
                }
                let mut buf = Vec::new();
                m.read_to_end(&mut buf)?;
                let text = String::from_utf8_lossy(&common::read_maybe_gz(buf)?).to_string();
                if !f(RawGame { name, text, kyoku_first: None })? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let text = String::from_utf8_lossy(&common::read_maybe_gz(bytes)?).to_string();
        return f(RawGame { name: input.file_name().unwrap().to_string_lossy().to_string(), text, kyoku_first: None });
    }
    let parquet = common::find_files(input, &[".parquet"])?;
    if !parquet.is_empty() {
        for p in parquet {
            let file = std::fs::File::open(&p)?;
            let b = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
            let schema = b.schema().clone();
            let ev_idx = schema.index_of("events").context("parquet needs an `events` column")?;
            let id_idx = schema.index_of("game_id").ok();
            let kf_idx = schema.index_of("kyoku_first").ok();
            let cols: Vec<usize> = std::iter::once(ev_idx).chain(id_idx).chain(kf_idx).collect();
            let mask = parquet::arrow::ProjectionMask::roots(b.parquet_schema(), cols);
            for batch in b.with_projection(mask).with_batch_size(512).build()? {
                let batch = batch?;
                let get = |name: &str| -> Option<Vec<Option<String>>> {
                    let c = batch.column_by_name(name)?;
                    if let Some(a) = c.as_any().downcast_ref::<StringArray>() {
                        return Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect());
                    }
                    let a = c.as_any().downcast_ref::<LargeStringArray>()?;
                    Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect())
                };
                let evs = get("events").context("events must be a string column")?;
                let ids = get("game_id");
                let kf: Option<Vec<Option<i64>>> = batch.column_by_name("kyoku_first").map(|c| {
                    match arrow_cast::cast(c, &arrow_schema::DataType::Int64) {
                        Ok(c) => {
                            let a = c.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap().clone();
                            (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect()
                        }
                        Err(_) => vec![None; batch.num_rows()],
                    }
                });
                for (r, e) in evs.into_iter().enumerate() {
                    let Some(text) = e else { continue };
                    let name = ids.as_ref().and_then(|v| v[r].clone()).unwrap_or_else(|| format!("{}#{r}", p.display()));
                    let kyoku_first = kf.as_ref().and_then(|v| v[r]);
                    if !f(RawGame { name, text, kyoku_first })? {
                        return Ok(false);
                    }
                }
            }
        }
        return Ok(true);
    }
    for p in common::find_files(input, &[".mjson", ".mjson.gz"])? {
        let text = String::from_utf8_lossy(&common::read_maybe_gz(std::fs::read(&p)?)?).to_string();
        if !f(RawGame { name: p.file_name().unwrap().to_string_lossy().to_string(), text, kyoku_first: None })? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The log's own game length: `kyoku_first` of its start_game event (0 = hanchan).
fn start_game_kyoku_first(text: &str) -> Option<i64> {
    let line = text.lines().find(|l| !l.trim().is_empty())?;
    let ev: Value = serde_json::from_str(line).ok()?;
    (ev.get("type")?.as_str()? == "start_game").then(|| ev.get("kyoku_first")?.as_i64()).flatten()
}

/// Identity of a game independent of how a converter spelled it: the first kyoku's dealer,
/// round, counters, scores, dora indicator and the four starting hands (as sorted tiles).
fn deal_key(evs: &[Value]) -> Option<u64> {
    let ev = evs.iter().find(|e| e.get("type").and_then(|v| v.as_str()) == Some("start_kyoku"))?;
    let tile = |v: &Value| -> Option<(usize, bool)> {
        let p = v.as_str()?;
        Some((tile_kind(p).ok()?, is_red(p)))
    };
    let mut hands = Vec::new();
    for t in ev.get("tehais")?.as_array()? {
        let mut h: Vec<(usize, bool)> = t.as_array()?.iter().map(tile).collect::<Option<_>>()?;
        h.sort_unstable();
        hands.push(h);
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for k in ["oya", "kyoku", "honba", "kyotaku"] {
        ev.get(k).and_then(|v| v.as_i64()).hash(&mut hasher);
    }
    ev.get("bakaze").and_then(|v| v.as_str()).hash(&mut hasher);
    ev.get("scores").map(|v| v.to_string()).hash(&mut hasher);
    tile(ev.get("dora_marker")?)?.hash(&mut hasher);
    hands.hash(&mut hasher);
    Some(hasher.finish())
}

pub struct ConvertOpts {
    pub shard_size: usize,
    pub max_games: Option<usize>,
    pub min_frames: usize,
    pub hanchan_only: bool,
    pub top_placements: Option<usize>,
    pub dedup: bool,
    /// drop a game whose log breaks the rules somewhere (a mask mismatch: e.g. a riichi
    /// declared without tenpai, a discard in riichi other than the drawn tile — corrupt
    /// tile identities) instead of keeping it with all-true masks on those frames
    pub drop_inconsistent: bool,
    /// compute the statistics without writing shards
    pub stats_only: bool,
}

/// Conversion of one game in a worker: (deal key, samples, game mismatches) or None when
/// skipped.
type Converted = Option<(Option<u64>, Vec<(Sample, FrameStats)>, usize)>;

fn convert_text(g: &RawGame, min_frames: usize) -> Converted {
    let evs: Vec<Value> = g.text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>().ok()?;
    let (samples, mismatch) = convert_game(&evs, &g.name, min_frames).ok()?;
    Some((deal_key(&evs), samples, mismatch))
}

pub fn convert(inputs: &[PathBuf], out: &Path, o: &ConvertOpts) -> Result<Value> {
    let mut w = if o.stats_only { None } else { Some(ShardWriter::new(out, "mahjong", CONVERTER_VERSION, json!({"tile_features": TILE_FEATURES}), o.shard_size)?) };
    let (mut games, mut skipped, mut samples, mut frames) = (0usize, 0usize, 0usize, 0usize);
    let (mut low_quality, mut unknown_length, mut duplicates, mut inconsistent) = (0usize, 0usize, 0usize, 0usize);
    let mut hist: BTreeMap<&str, usize> = ACTION_NAMES.iter().map(|&n| (n, 0)).collect();
    let mut fs = FrameStats::default();
    let mut seen: HashSet<u64> = HashSet::new();
    // Games are converted in parallel chunks and written in input order, so the output
    // (stats, --max-games and --dedup included) is identical to a sequential pass.
    const CHUNK: usize = 512;
    let mut buf: Vec<Queued> = Vec::with_capacity(CHUNK);
    let mut flush = |buf: &mut Vec<Queued>, games: &mut usize| -> Result<()> {
        let results: Vec<Option<Converted>> = buf
            .par_iter()
            .map(|q| match q {
                Queued::Game(g) => Some(convert_text(g, o.min_frames)),
                _ => None,
            })
            .collect();
        for (q, r) in buf.iter().zip(results) {
            if o.max_games.map(|m| *games >= m).unwrap_or(false) {
                break;
            }
            let (g, r) = match (q, r) {
                (Queued::Game(g), Some(r)) => (g, r),
                (Queued::Tonpuusen, _) => {
                    low_quality += 1;
                    continue;
                }
                _ => {
                    unknown_length += 1;
                    continue;
                }
            };
            let Some((key, ss, mismatch)) = r else {
                skipped += 1;
                if skipped <= 20 {
                    eprintln!("skipped (malformed or unsupported log): {}", g.name);
                }
                continue;
            };
            if o.drop_inconsistent && mismatch > 0 {
                inconsistent += 1;
                if inconsistent <= 20 {
                    eprintln!("dropped (breaks the rules: {mismatch} frame(s) outside the masks): {}", g.name);
                }
                continue;
            }
            if o.dedup {
                if let Some(k) = key {
                    if !seen.insert(k) {
                        duplicates += 1;
                        continue;
                    }
                }
            }
            *games += 1;
            for (smp, st) in ss {
                if let Some(k) = o.top_placements {
                    if smp.meta["placement"].as_u64().unwrap_or(4) as usize > k {
                        continue;
                    }
                }
                samples += 1;
                fs.add(&st);
                if let Some(t) = smp.tensors.iter().find(|t| t.name == "tgt.action_type") {
                    if let gmshard::Column::I64(v) = &t.col {
                        frames += v.len();
                        for &a in v {
                            *hist.get_mut(ACTION_NAMES[a as usize]).unwrap() += 1;
                        }
                    }
                }
                if let Some(w) = w.as_mut() {
                    w.push(smp)?;
                }
            }
        }
        buf.clear();
        Ok(())
    };
    for input in inputs {
        let more = iter_games(input, &mut |g: RawGame| {
            if o.max_games.map(|m| games >= m).unwrap_or(false) {
                return Ok(false);
            }
            // filtered games stay in the queue so that the counts (and --max-games) follow
            // the input order exactly
            let q = if !o.hanchan_only {
                Queued::Game(g)
            } else {
                match g.kyoku_first.or_else(|| start_game_kyoku_first(&g.text)) {
                    Some(0) => Queued::Game(g),
                    Some(_) => Queued::Tonpuusen,
                    None => Queued::UnknownLength,
                }
            };
            buf.push(q);
            if buf.len() >= CHUNK {
                flush(&mut buf, &mut games)?;
            }
            Ok(true)
        })?;
        if !more {
            break;
        }
    }
    flush(&mut buf, &mut games)?;
    let mut shards = 0;
    if let Some(w) = w.as_mut() {
        w.flush()?;
        shards = w.written_shards;
    }
    Ok(json!({"games": games, "skipped": skipped, "low_quality": low_quality, "unknown_length": unknown_length,
              "duplicates": duplicates, "inconsistent": inconsistent, "samples": samples, "frames": frames, "shards": shards,
              "action_hist": hist, "mask_mismatch": fs.mismatch, "claim_frames": fs.claim_frames,
              "chankan_frames": fs.chankan_frames, "preempted_passes": fs.preempted, "hora_legal": fs.hora_legal,
              "hora_declined": fs.hora_declined}))
}

/// The parquet files of the Hugging Face corpus at `revision`, sorted.
fn hf_parquet_files(client: &reqwest::blocking::Client, revision: &str) -> Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/datasets/{HF_DATASET}/tree/{revision}/data");
    let listing: Value = client.get(&url).send()?.error_for_status()?.json()?;
    let mut files: Vec<String> = listing
        .as_array()
        .context("unexpected dataset listing")?
        .iter()
        .filter_map(|e| e.get("path").and_then(|p| p.as_str()))
        .filter(|p| p.ends_with(".parquet"))
        .map(String::from)
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no parquet files at {url}");
    Ok(files)
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download the Tenhou Houou (phoenix room) mjai corpus: parquet shards of the Hugging Face
    /// dataset hhim8826/tenhou-houou-mjai (24 shards, ~88k games / ~300 MB each; licence: see
    /// the module doc — Tenhou's terms, keep the data private)
    Fetch {
        #[arg(long, default_value = "data/mahjong/raw")]
        out: PathBuf,
        /// shard indices into the dataset's sorted parquet list (0..23)
        #[arg(long, num_args = 1.., default_values_t = [0])]
        shards: Vec<usize>,
        /// every shard (the whole 2.12M-game corpus, 6.3 GB)
        #[arg(long)]
        all: bool,
        /// dataset revision (branch, tag or commit) to download
        #[arg(long, default_value = "main")]
        revision: String,
    },
    /// Convert mjai logs to gm-shards
    Convert {
        /// .mjson(.gz) / .zip / .parquet files or directories, read in order
        #[arg(long, num_args = 1.., required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 128)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 8)]
        min_frames: usize,
        /// keep only hanchan (East-South) games, by the parquet `kyoku_first` column or the
        /// log's start_game `kyoku_first` (games without either are skipped)
        #[arg(long)]
        hanchan_only: bool,
        /// imitate only seats that finished in the top N (1..4)
        #[arg(long)]
        top_placements: Option<usize>,
        /// drop games whose first kyoku repeats an earlier game's deal (one game written by
        /// two converters)
        #[arg(long)]
        dedup: bool,
        /// drop games whose log breaks the rules somewhere (counted as `inconsistent`) instead
        /// of keeping them with all-true masks on the offending frames
        #[arg(long)]
        drop_inconsistent: bool,
        /// high-quality preset: hanchan only + dedup + drop-inconsistent (all seats are
        /// phoenix-room players)
        #[arg(long)]
        hq: bool,
        /// compute the statistics only, write no shards
        #[arg(long)]
        stats_only: bool,
        /// worker threads (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, shards, all, revision } => {
            let client = common::http()?;
            let files = hf_parquet_files(&client, &revision)?;
            let picked: Vec<&String> = if all {
                files.iter().collect()
            } else {
                shards
                    .iter()
                    .map(|&i| files.get(i).with_context(|| format!("shard {i} out of range 0..{}", files.len())))
                    .collect::<Result<_>>()?
            };
            for path in picked {
                let name = path.rsplit('/').next().unwrap_or(path);
                let url = format!("https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{path}");
                anyhow::ensure!(common::download(&client, &url, &out.join(name))?, "404: {url}");
            }
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, max_games, min_frames, hanchan_only, top_placements, dedup, drop_inconsistent, hq,
                       stats_only, threads } => {
            if threads > 0 {
                rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
            }
            let o = ConvertOpts { shard_size, max_games, min_frames, hanchan_only: hanchan_only || hq, top_placements,
                                  dedup: dedup || hq, drop_inconsistent: drop_inconsistent || hq, stats_only };
            println!("{}", convert(&input, &out, &o)?);
            Ok(())
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::game::{convert_game, walk_game, Decision};
    use super::state::*;
    use super::*;

    pub(crate) fn fixture_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures").join(name)
    }

    pub(crate) fn read_events(path: &Path) -> Vec<Value> {
        let text = std::fs::read_to_string(path).unwrap();
        text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    /// Every committed real game (golden + audit fixtures).
    pub(crate) fn fixture_games() -> Vec<(String, Vec<Value>)> {
        let mut out = Vec::new();
        for d in ["mahjong", "mahjong_audit"] {
            for p in common::find_files(&fixture_dir(d), &[".mjson"]).unwrap() {
                out.push((p.file_name().unwrap().to_string_lossy().to_string(), read_events(&p)));
            }
        }
        assert!(out.len() >= 16);
        out
    }

    #[test]
    fn fixture_games_have_no_mask_mismatch() {
        let (mut claims, mut rons, mut declined) = (0, 0, 0);
        for (name, evs) in fixture_games() {
            let (out, mismatch) = convert_game(&evs, &name, 1).ok().unwrap();
            assert_eq!((out.len(), mismatch), (4, 0), "{name}");
            for (smp, st) in &out {
                assert_eq!(st.mismatch, 0, "{name}");
                assert!(smp.frames().unwrap() > 0);
                let names: Vec<&str> = smp.tensors.iter().map(|x| x.name.as_str()).collect();
                for k in ["tgt.legal_action_type", "tgt.legal_tile_of", "tgt.il_weight"] {
                    assert!(names.contains(&k), "{k}");
                }
                claims += st.claim_frames;
                rons += st.hora_legal - st.hora_declined;
                declined += st.hora_declined;
            }
        }
        assert!(claims > 1000 && rons > 50 && declined > 0, "{claims} {rons} {declined}");
    }

    /// Every recorded win (ron, chankan, tsumo) is legal under the exact masks: the one-han
    /// minimum and furiten never reject a win Tenhou accepted.
    #[test]
    fn every_recorded_win_is_legal() {
        let mut wins = 0;
        for (name, evs) in fixture_games() {
            walk_game(&evs, &mut |d: &Decision| {
                if d.label.map(|l| l.0) == Some(HORA) {
                    wins += 1;
                    assert!(d.legal.types[HORA as usize], "{name}: event {} seat {}", d.event, d.seat);
                }
            })
            .ok()
            .unwrap();
        }
        assert!(wins > 100, "{wins}");
    }

    /// The same game in the other spelling of the Houou corpus (an open kan's dora written
    /// right after the discard that reveals it instead of right before) converts identically.
    #[test]
    fn both_dora_spellings_convert_identically() {
        let mut moved = 0;
        for (name, evs) in fixture_games() {
            let mut alt = evs.clone();
            let mut i = 0;
            while i + 1 < alt.len() {
                // [tsumo] dora dahai → [tsumo] dahai dora (the dora of the preceding open kan)
                if alt[i]["type"] == "dora" && alt[i + 1]["type"] == "dahai" && i > 0 && alt[i - 1]["type"] == "tsumo" {
                    alt.swap(i, i + 1);
                    moved += 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            let (a, _) = convert_game(&evs, &name, 1).ok().unwrap();
            let (b, _) = convert_game(&alt, &name, 1).ok().unwrap();
            assert_eq!(a.len(), b.len());
            for ((x, _), (y, _)) in a.iter().zip(&b) {
                for (tx, ty) in x.tensors.iter().zip(&y.tensors) {
                    assert_eq!(tx.name, ty.name);
                    assert!(same_col(&tx.col, &ty.col), "{name}: {}", tx.name);
                }
            }
        }
        assert!(moved > 5, "{moved}");
    }

    pub(crate) fn same_col(a: &gmshard::Column, b: &gmshard::Column) -> bool {
        use gmshard::Column::*;
        match (a, b) {
            (F32(x), F32(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            (I64(x), I64(y)) => x == y,
            (U8(x), U8(y)) => x == y,
            (Bits(x, p), Bits(y, q)) => x == y && p == q,
            _ => false,
        }
    }

    #[test]
    fn game_length_and_deal_key() {
        assert_eq!(start_game_kyoku_first("{\"type\":\"start_game\",\"kyoku_first\":4,\"aka_flag\":true}\n{}"), Some(4));
        assert_eq!(start_game_kyoku_first("\n{\"type\":\"start_game\",\"kyoku_first\":0}\n"), Some(0));
        assert_eq!(start_game_kyoku_first("{\"type\":\"start_game\"}\n"), None);
        assert_eq!(start_game_kyoku_first("{\"type\":\"start_kyoku\",\"kyoku_first\":0}\n"), None);
        let (_, evs) = &fixture_games()[0];
        let k = deal_key(evs).unwrap();
        // spelling-independent: a red five written "0m" instead of "5mr", tiles reordered
        let mut alt = evs.clone();
        let sk = alt.iter_mut().find(|e| e["type"] == "start_kyoku").unwrap();
        for hand in sk["tehais"].as_array_mut().unwrap() {
            let h = hand.as_array_mut().unwrap();
            h.reverse();
            for t in h.iter_mut() {
                if let Some(p) = t.as_str().and_then(|p| p.strip_suffix('r')) {
                    *t = json!(format!("0{}", &p[1..]));
                }
            }
        }
        assert_eq!(deal_key(&alt), Some(k));
        // another game deals differently
        let (_, other) = &fixture_games()[1];
        assert_ne!(deal_key(other), Some(k));
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gmdata-mj-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn hanchan_filter_uses_the_announced_length_and_dedup_drops_respellings() {
        let src = tmpdir("src");
        let games = fixture_games();
        let (name, evs) = &games[0];
        let write = |file: &str, evs: &[Value]| {
            let text: String = evs.iter().map(|e| format!("{e}\n")).collect();
            std::fs::write(src.join(file), text).unwrap();
        };
        write(name, evs);
        // the same game again (another spelling / file), and as a tonpuusen / no-length log
        write("copy.mjson", evs);
        let mut ton = evs.clone();
        ton[0]["kyoku_first"] = json!(4);
        write("tonpuu.mjson", &ton);
        let mut nolen = evs.clone();
        nolen[0].as_object_mut().unwrap().remove("kyoku_first");
        write("nolen.mjson", &nolen);
        let run_max = |dedup: bool, hanchan_only: bool, max_games: Option<usize>| {
            let out = tmpdir(&format!("out{dedup}{hanchan_only}{max_games:?}"));
            let o = ConvertOpts { shard_size: 1000, max_games, min_frames: 1, hanchan_only, top_placements: None, dedup,
                                  drop_inconsistent: false, stats_only: false };
            convert(std::slice::from_ref(&src), &out, &o).unwrap()
        };
        let run = |dedup: bool, hanchan_only: bool| run_max(dedup, hanchan_only, None);
        // files in name order: the fixture game, copy, nolen, tonpuu
        let s = run(false, true);
        assert_eq!((s["games"].as_u64(), s["low_quality"].as_u64(), s["unknown_length"].as_u64()), (Some(2), Some(1), Some(1)));
        let s = run(true, true);
        assert_eq!((s["games"].as_u64(), s["duplicates"].as_u64()), (Some(1), Some(1)));
        let s = run(true, false);
        assert_eq!((s["games"].as_u64(), s["duplicates"].as_u64()), (Some(1), Some(3)));
        let s = run(false, false);
        assert_eq!(s["games"].as_u64(), Some(4));
        // --max-games stops where a sequential pass would: nothing after the 2nd game is counted
        let s = run_max(false, true, Some(2));
        assert_eq!((s["games"].as_u64(), s["low_quality"].as_u64(), s["unknown_length"].as_u64()), (Some(2), Some(0), Some(0)));
    }
}
