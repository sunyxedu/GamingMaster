//! Heads-up no-limit Texas hold'em: the Annual Computer Poker Competition 2009-2017
//! heads-up no-limit hand histories (PHH format) → gm-shards through the gmrules
//! engine `gmrules::games::hunl` (replay + verification of every record, recorded
//! payoffs checked; frames = the simulator's).
//!
//! Source: "A Dataset of Poker Hand Histories" (Kim et al., Zenodo record 10796885 →
//! 17136841, `poker-hand-histories.zip`, 20.3 GB, CC-BY-4.0). Its raw ACPC heads-up
//! no-limit logs: 58 380 `.phhs` match logs of 3 000 hands each (4.85 GB compressed,
//! 60.6 GB of text) = 175 M hands, 160.6 M unique (the 2016 logs are archived twice:
//! `logs_2pn_2016` = `logs_2pn_2016_0_99` + `logs_2pn_2016_100_199`, 4 855 logs); the
//! `processed_logs*` / `2pn_processed_logs` copies (the same hands with all-in EV
//! results) and the limit / 3-player logs are never read. No 2015 competition.
//!
//! Unique hands per year (M): 2009 3.6 (1 / 2 / 400: `--rescale`), 2010 16.8, 2011
//! 16.2, 2012 17.4, 2013 19.8, 2014 24.0, 2016 17.0, 2017 45.75. Measured on 72 000
//! hands (3 logs per year): 0 illegal actions, 0 payoff mismatches, 5.56 decision
//! frames per hand (both seats), 1.67 KB per frame on disk (index included), about
//! 45 000 hands / s on 12 cores.
//!
//! Full scale (the cluster; `scripts/fetch_hunl.sh` does fetch + convert end to end,
//! per year, train + val, resumable):
//!
//! ```sh
//! scripts/fetch_hunl.sh ~/gm-data/hunl                    # HQ=1: winning bots' seats
//! #   ≈ 160 M hands → ≈ 420 M frames, ≈ 0.7 TB (train/<year>, val/<year>: 0.5 %)
//! HQ=0 scripts/fetch_hunl.sh ~/gm-data/hunl               # both seats: ≈ 890 M frames, ≈ 1.5 TB
//! KEEP_PER_MILLE=100 scripts/fetch_hunl.sh ~/gm-data/hunl # a uniform tenth: ≈ 42 M frames, ≈ 70 GB
//! ```
//!
//! The tenth is plenty for joint IL pretraining and keeps the trainers' in-memory
//! per-sample index small (GameShardDataset: ~120 B per sample and process, ~20 GB
//! for the full HQ corpus's ~160 M samples).
//!
//! By hand:
//!
//! ```sh
//! gmdata hunl fetch   --out ~/gm-data/hunl/raw              # 4.85 GB of range requests
//! gmdata hunl fetch   --out /tmp/hunl-sample --max-files-per-year 1 --max-hands 50
//! gmdata hunl convert --input ~/gm-data/hunl/raw --years 2017 --out data/hunl/train/2017 --hq --split train
//! gmdata hunl convert --input poker-hand-histories.zip --out data/hunl/train   # the zip itself
//! gmdata hunl stats   --input ~/gm-data/hunl/raw --years 2017                  # bb/100 per bot
//! ```
//!
//! PHH (a TOML subset; `.phhs` = `[n]` sections, one hand each; `.gz` read on the fly):
//! `variant = 'NT'`, `antes`, `blinds_or_straddles = [50, 100]` (heads-up: p1 posts the
//! BIG blind, p2 = the button posts the small blind — PokerKit's convention, the ACPC
//! seats: p1 = seat 0), `starting_stacks`, `actions = ['d dh p1 Ac4s', 'd dh p2 Kd9c',
//! 'p2 cbr 300', 'p1 cc', 'd db QdAd6c', 'p1 f', ...]` (`cbr X` = bet / raise to X in
//! the STREET; the canonical id is the total commitment, i.e. the commitment when the
//! street began + X), `players`, `hand`, `_results` (chips won, checked against the
//! replay). Showdown `sm` actions are checked against the dealt cards and skipped.
//!
//! Filters (all counted in the printed stats): the game definition must be the config's
//! (default 50 / 100 blinds, 20000 stacks = 200 big blinds; `--rescale` also takes
//! hands of the same depth at another chip scale — the 2009 logs, 1 / 2 / 400 —
//! multiplying every amount by the integer factor); duplicates (same players, stacks,
//! blinds, actions and hand number) are dropped; `--split train|val` keeps the hands
//! whose key hash falls outside / inside the `--val-per-mille` bucket;
//! `--keep-per-mille K` a uniform hash-based subsample of K / 1000 of the hands; `--hq`
//! keeps only the seats of WINNING bots (a first pass over the same input measures
//! every player's big blinds per 100 hands; `--min-bb100`), `--player NAME` only the
//! named ones. The hash ([`StableHash`]) is platform- and release-independent, so the
//! split and the subsample are the same on every machine.

use crate::traj::{self, ConvertOptions, Record};
use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::games::hunl::eval::{card_str, parse_cards};
use gmrules::games::hunl::{Hunl, HunlConfig, CALL, FOLD};
use gmrules::{Event, Game, Seat};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The record's download URL (Zenodo record 17136841, the current version of 10796885).
pub const ZENODO_URL: &str = "https://zenodo.org/api/records/17136841/files/poker-hand-histories.zip/content";

// ---------------------------------------------------------------------------
// PHH parsing
// ---------------------------------------------------------------------------

/// One PHH hand (the fields this converter reads) + its de-duplication key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhhHand {
    pub variant: String,
    pub antes: Vec<f64>,
    pub blinds: Vec<f64>,
    pub min_bet: Option<f64>,
    pub stacks: Vec<f64>,
    pub actions: Vec<String>,
    pub players: Vec<String>,
    pub hand: Option<i64>,
    /// `_results` (the ACPC conversion) or `finishing_stacks - starting_stacks`.
    pub results: Option<Vec<f64>>,
    /// Hash of variant, antes, blinds, stacks, players, actions and the hand number (not
    /// the results): the same hand archived twice shares it; two different hands that
    /// happen to repeat a deal and its action line (a few per match: 1.6 M deals,
    /// deterministic bots) differ in their hand numbers. [`StableHash`]: the same on
    /// every machine and Rust release, so the train / val split (`key % 1000`) and the
    /// subsample (`key / 1000 % 1000`) of a hand never change.
    pub key: u64,
}

/// FNV-1a 64 over the fed bytes + the murmur3 finaliser: a platform- and
/// release-independent hash (std's `DefaultHasher` may change between Rust releases).
pub struct StableHash(u64);

impl StableHash {
    pub fn new() -> Self {
        StableHash(0xcbf2_9ce4_8422_2325)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}

impl Default for StableHash {
    fn default() -> Self {
        Self::new()
    }
}

/// A TOML literal of the subset PHH uses: strings, numbers, booleans, arrays.
fn parse_value(s: &str) -> Result<Value> {
    let s = s.trim();
    let (v, rest) = value_prefix(s)?;
    if !rest.trim().is_empty() && !rest.trim_start().starts_with('#') {
        bail!("trailing text {rest:?} after a value");
    }
    Ok(v)
}

fn value_prefix(s: &str) -> Result<(Value, &str)> {
    let s = s.trim_start();
    let Some(c) = s.chars().next() else { bail!("missing value") };
    match c {
        '\'' => {
            let end = s[1..].find('\'').context("unterminated '...' string")?;
            Ok((Value::String(s[1..1 + end].to_string()), &s[end + 2..]))
        }
        '"' => {
            let mut out = String::new();
            let mut it = s[1..].char_indices();
            while let Some((i, ch)) = it.next() {
                match ch {
                    '"' => return Ok((Value::String(out), &s[i + 2..])),
                    '\\' => match it.next() {
                        Some((_, 'n')) => out.push('\n'),
                        Some((_, 't')) => out.push('\t'),
                        Some((_, e)) => out.push(e),
                        None => bail!("dangling escape"),
                    },
                    ch => out.push(ch),
                }
            }
            bail!("unterminated \"...\" string")
        }
        '[' => {
            let mut items = Vec::new();
            let mut rest = &s[1..];
            loop {
                rest = rest.trim_start();
                if let Some(r) = rest.strip_prefix(']') {
                    return Ok((Value::Array(items), r));
                }
                let (v, r) = value_prefix(rest)?;
                items.push(v);
                rest = r.trim_start();
                if let Some(r) = rest.strip_prefix(',') {
                    rest = r;
                } else if !rest.starts_with(']') {
                    bail!("expected ',' or ']' in an array, got {:?}", &rest[..rest.len().min(20)]);
                }
            }
        }
        _ => {
            let end = s.find(|ch: char| ch == ',' || ch == ']' || ch == '#' || ch.is_whitespace()).unwrap_or(s.len());
            let tok = &s[..end];
            let v = match tok {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                t => {
                    let t = t.replace('_', "");
                    if let Ok(i) = t.parse::<i64>() {
                        json!(i)
                    } else {
                        json!(t.parse::<f64>().with_context(|| format!("value {tok:?}"))?)
                    }
                }
            };
            Ok((v, &s[end..]))
        }
    }
}

/// Bracket depth change of a line (outside strings) — multi-line arrays.
fn depth_delta(line: &str) -> i32 {
    let (mut d, mut q) = (0, None);
    for ch in line.chars() {
        match (q, ch) {
            (None, '\'' | '"') => q = Some(ch),
            (Some(o), c) if c == o => q = None,
            (None, '[') => d += 1,
            (None, ']') => d -= 1,
            (None, '#') => break,
            _ => {}
        }
    }
    d
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array()?.iter().map(|x| x.as_str().map(String::from)).collect()
}

fn hand_from_fields(fields: &[(String, String)]) -> Result<PhhHand> {
    let mut h = PhhHand::default();
    let mut hasher = StableHash::new();
    let mut finishing = None;
    for (k, raw) in fields {
        let v = parse_value(raw).with_context(|| format!("key {k}"))?;
        let bad = || anyhow::anyhow!("key {k}: unexpected value {raw}");
        match k.as_str() {
            "variant" => h.variant = v.as_str().ok_or_else(bad)?.to_string(),
            "antes" => h.antes = floats(&v).ok_or_else(bad)?,
            "blinds_or_straddles" => h.blinds = floats(&v).ok_or_else(bad)?,
            "min_bet" => h.min_bet = Some(v.as_f64().ok_or_else(bad)?),
            "starting_stacks" => h.stacks = floats(&v).ok_or_else(bad)?,
            "actions" => h.actions = strings(&v).ok_or_else(bad)?,
            "players" => h.players = strings(&v).ok_or_else(bad)?,
            "hand" => h.hand = v.as_i64(),
            "_results" => h.results = Some(floats(&v).ok_or_else(bad)?),
            "finishing_stacks" => finishing = Some(floats(&v).ok_or_else(bad)?),
            _ => {}
        }
        if matches!(k.as_str(), "variant" | "antes" | "blinds_or_straddles" | "min_bet" | "starting_stacks" | "actions" | "players" | "hand") {
            hasher.write(k.as_bytes());
            hasher.write(&[0xff]);
            hasher.write(v.to_string().as_bytes());
            hasher.write(&[0xfe]);
        }
    }
    if h.results.is_none() {
        if let Some(f) = finishing {
            if f.len() == h.stacks.len() {
                h.results = Some(f.iter().zip(&h.stacks).map(|(a, b)| a - b).collect());
            }
        }
    }
    h.key = hasher.finish();
    Ok(h)
}

/// The hands of a `.phh` (one hand) / `.phhs` (`[n]` sections) text, in order; a hand
/// that does not parse is an `Err` (with its section).
pub fn parse_phh(text: &str) -> Vec<Result<PhhHand>> {
    let mut out = Vec::new();
    let mut fields: Vec<(String, String)> = Vec::new();
    let mut section: Option<String> = None;
    let mut pending: Option<(String, String, i32)> = None;
    let mut errors: Vec<String> = Vec::new();
    let flush = |fields: &mut Vec<(String, String)>, section: &Option<String>, errors: &mut Vec<String>, out: &mut Vec<Result<PhhHand>>| {
        if fields.is_empty() && errors.is_empty() {
            return;
        }
        let r = if errors.is_empty() {
            hand_from_fields(fields)
        } else {
            Err(anyhow::anyhow!("{}", errors.join("; ")))
        };
        out.push(r.with_context(|| format!("section {}", section.as_deref().unwrap_or("(none)"))));
        fields.clear();
        errors.clear();
    };
    for line in text.lines() {
        if let Some((k, mut v, d)) = pending.take() {
            v.push(' ');
            v.push_str(line.trim());
            let d = d + depth_delta(line);
            if d > 0 {
                pending = Some((k, v, d));
            } else {
                fields.push((k, v));
            }
            continue;
        }
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') && !t.contains('=') {
            flush(&mut fields, &section, &mut errors, &mut out);
            section = Some(t.trim_matches(|c| c == '[' || c == ']').to_string());
            continue;
        }
        match t.split_once('=') {
            Some((k, v)) => {
                let (k, v) = (k.trim().to_string(), v.trim().to_string());
                let d = depth_delta(&v);
                if d > 0 {
                    pending = Some((k, v, d));
                } else {
                    fields.push((k, v));
                }
            }
            None => errors.push(format!("unexpected line {t:?}")),
        }
    }
    if let Some((k, _, _)) = pending {
        errors.push(format!("unterminated value of {k}"));
    }
    flush(&mut fields, &section, &mut errors, &mut out);
    out
}

// ---------------------------------------------------------------------------
// PHH hand → gmrules trajectory
// ---------------------------------------------------------------------------

/// Why a parsed hand is not converted (besides replay failures, counted by traj.rs).
#[derive(Clone, Debug, PartialEq)]
pub enum Skip {
    /// Another game definition (variant, player count, blinds, stacks, antes).
    OtherVariant(String),
}

fn int(x: f64, what: &str) -> Result<u32> {
    if x < 0.0 || x.fract() != 0.0 || x > u32::MAX as f64 {
        bail!("{what} {x} is not a chip count");
    }
    Ok(x as u32)
}

/// The chip scale factor mapping the hand's game definition onto `cfg` (1 = the same
/// game), or why it is another game.
pub fn scale_of(h: &PhhHand, cfg: &HunlConfig, rescale: bool) -> std::result::Result<u32, Skip> {
    let other = |m: String| Err(Skip::OtherVariant(m));
    if h.variant != "NT" {
        return other(format!("variant {:?}", h.variant));
    }
    if h.players.len() != 2 || h.stacks.len() != 2 || h.blinds.len() != 2 {
        return other(format!("{} players", h.players.len()));
    }
    if h.antes.iter().any(|&a| a != 0.0) {
        return other(format!("antes {:?}", h.antes));
    }
    let (Ok(sb), Ok(bb), Ok(s0), Ok(s1)) = (int(h.blinds[0], "blind"), int(h.blinds[1], "blind"), int(h.stacks[0], "stack"), int(h.stacks[1], "stack")) else {
        return other("non-integer blinds / stacks".into());
    };
    if s0 != s1 || bb == 0 {
        return other(format!("stacks {s0} / {s1}, blinds {sb} / {bb}"));
    }
    if h.min_bet.is_some_and(|m| m != bb as f64) {
        return other(format!("min_bet {:?} != big blind {bb}", h.min_bet));
    }
    if (sb, bb, s0) == (cfg.small_blind, cfg.big_blind, cfg.stack) {
        return Ok(1);
    }
    if rescale && cfg.big_blind.is_multiple_of(bb) {
        let k = cfg.big_blind / bb;
        if sb.checked_mul(k) == Some(cfg.small_blind) && s0.checked_mul(k) == Some(cfg.stack) {
            return Ok(k);
        }
    }
    other(format!("blinds {sb} / {bb}, stacks {s0} (config {} / {} / {})", cfg.small_blind, cfg.big_blind, cfg.stack))
}

fn seat_of(tok: &str) -> Result<Seat> {
    match tok {
        "p1" => Ok(0),
        "p2" => Ok(1),
        t => bail!("player {t:?} (heads-up: p1 / p2)"),
    }
}

/// The gmrules trajectory of a heads-up hand at chip scale `k` (see the module docs).
pub fn hand_events(h: &PhhHand, cfg: &HunlConfig, k: u32) -> Result<Vec<Event>> {
    let mut hole: [Option<Vec<u8>>; 2] = [None, None];
    let mut events = Vec::new();
    let mut dealt_hole = false;
    // commitments, to turn street amounts into total raise-to amounts
    let mut spent = [cfg.big_blind, cfg.small_blind];
    let mut base = 0u32;
    for (i, a) in h.actions.iter().enumerate() {
        let tok: Vec<&str> = a.split_whitespace().collect();
        let err = || format!("action {i} {a:?}");
        match tok.as_slice() {
            ["d", "dh", p, cards] => {
                let s = seat_of(p).with_context(err)?;
                if dealt_hole || hole[s].is_some() {
                    bail!("{}: hole cards dealt twice / after the betting began", err());
                }
                let c = parse_cards(cards).with_context(|| format!("{}: cards", err()))?;
                if c.len() != 2 {
                    bail!("{}: {} hole cards", err(), c.len());
                }
                hole[s] = Some(c);
            }
            ["d", "db", cards] => {
                let c = parse_cards(cards).with_context(|| format!("{}: cards", err()))?;
                if !dealt_hole {
                    bail!("{}: board before the hole cards", err());
                }
                events.extend(c.iter().map(|&x| Event::Chance(x as u32)));
                base = spent[0].max(spent[1]);
            }
            [p, rest @ ..] if p.starts_with('p') => {
                let s = seat_of(p).with_context(err)?;
                if !dealt_hole {
                    let (Some(a0), Some(a1)) = (&hole[0], &hole[1]) else { bail!("{}: betting before both hole cards", err()) };
                    events.extend(a0.iter().chain(a1.iter()).map(|&x| Event::Chance(x as u32)));
                    dealt_hole = true;
                }
                match rest {
                    ["f"] => events.push(Event::Act(s, FOLD)),
                    ["cc"] => {
                        spent[s] = spent[0].max(spent[1]).min(cfg.stack);
                        events.push(Event::Act(s, CALL));
                    }
                    ["cbr", x] => {
                        let amount: f64 = x.parse().with_context(|| format!("{}: amount", err()))?;
                        let to = int(amount, "amount")
                            .with_context(err)?
                            .checked_mul(k)
                            .and_then(|x| x.checked_add(base))
                            .with_context(|| format!("{}: amount out of range", err()))?;
                        if to < 2 {
                            bail!("{}: raise to {to}", err());
                        }
                        spent[s] = to;
                        events.push(Event::Act(s, to));
                    }
                    ["sm"] | ["sm", "????"] => {}
                    ["sm", cards] => {
                        let mut shown = parse_cards(cards).with_context(|| format!("{}: cards", err()))?;
                        let mut own = hole[s].clone().unwrap_or_default();
                        shown.sort_unstable();
                        own.sort_unstable();
                        if shown != own {
                            bail!("{}: shows {cards}, was dealt {}", err(), own.iter().map(|&c| card_str(c)).collect::<String>());
                        }
                    }
                    _ => bail!("{}: unsupported action", err()),
                }
            }
            _ => bail!("{}: unsupported action", err()),
        }
    }
    if !dealt_hole {
        if let (Some(a0), Some(a1)) = (&hole[0], &hole[1]) {
            events.extend(a0.iter().chain(a1.iter()).map(|&x| Event::Chance(x as u32)));
        }
    }
    Ok(events)
}

// ---------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------

/// A raw ACPC heads-up no-limit log of the Zenodo archive (not the EV-processed copies).
pub fn is_raw_nl_log(path: &str) -> bool {
    let low = path.to_lowercase();
    (low.ends_with(".phhs") || low.ends_with(".phhs.gz"))
        && low.contains("annual-computer-poker-competition")
        && (low.contains("nolimit") || low.contains("2pn"))
        && !low.contains("processed")
}

/// The competition year of a path (its first `20xx` component).
pub fn year_of(path: &str) -> Option<u32> {
    path.split(['/', '\\']).find_map(|c| c.parse::<u32>().ok().filter(|y| (2000..2100).contains(y)))
}

/// One input text: a file, or a member of a zip archive.
#[derive(Clone, Debug)]
pub enum Source {
    File(PathBuf),
    Zip(PathBuf, String),
}

impl Source {
    pub fn name(&self) -> String {
        match self {
            Source::File(p) => p.display().to_string(),
            Source::Zip(z, m) => format!("{}:{m}", z.display()),
        }
    }

    fn short(&self) -> String {
        match self {
            Source::File(p) => p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            Source::Zip(_, m) => m.rsplit('/').next().unwrap_or(m).to_string(),
        }
    }

    pub fn read(&self) -> Result<String> {
        let bytes = match self {
            Source::File(p) => common::read_maybe_gz(std::fs::read(p).with_context(|| format!("read {}", p.display()))?)?,
            Source::Zip(z, m) => ARCHIVES.with(|cell| -> Result<Vec<u8>> {
                // one open archive per thread (its central directory is parsed once)
                let mut map = cell.borrow_mut();
                if !map.contains_key(z) {
                    let f = std::fs::File::open(z).with_context(|| format!("open {}", z.display()))?;
                    map.insert(z.clone(), zip::ZipArchive::new(f)?);
                }
                let a = map.get_mut(z).unwrap();
                let mut e = a.by_name(m)?;
                let mut out = Vec::with_capacity(e.size() as usize);
                e.read_to_end(&mut out)?;
                common::read_maybe_gz(out)
            })?,
        };
        String::from_utf8(bytes).context("not UTF-8")
    }
}

thread_local! {
    static ARCHIVES: RefCell<HashMap<PathBuf, zip::ZipArchive<std::fs::File>>> = RefCell::new(HashMap::new());
}

/// The inputs under `paths`: `.phh` / `.phhs` files (`.gz` too) of directories,
/// files, and the raw ACPC no-limit members of `.zip` archives — in sorted order,
/// `processed` copies skipped, restricted to `years` when given.
pub fn sources(paths: &[PathBuf], years: &[u32]) -> Result<Vec<Source>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_file() && p.to_string_lossy().to_lowercase().ends_with(".zip") {
            let a = zip::ZipArchive::new(std::fs::File::open(p)?)?;
            let mut names: Vec<String> = a.file_names().filter(|n| is_raw_nl_log(n)).map(String::from).collect();
            names.sort();
            out.extend(names.into_iter().map(|n| Source::Zip(p.clone(), n)));
        } else {
            let files = common::find_files(p, &[".phh", ".phhs", ".phh.gz", ".phhs.gz"])?;
            out.extend(files.into_iter().filter(|f| !f.to_string_lossy().to_lowercase().contains("processed")).map(Source::File));
        }
    }
    if !years.is_empty() {
        out.retain(|s| {
            let n = match s {
                Source::File(p) => p.to_string_lossy().to_string(),
                Source::Zip(_, m) => m.clone(),
            };
            year_of(&n).is_some_and(|y| years.contains(&y))
        });
    }
    Ok(out)
}

/// The hands of every source, parsed in parallel batches of files and yielded in
/// input order: `(source, section index, hand)`.
pub fn hands(srcs: Vec<Source>) -> impl Iterator<Item = (String, usize, Result<PhhHand>)> {
    const BATCH: usize = 32;
    let batches: Vec<Vec<Source>> = srcs.chunks(BATCH).map(|c| c.to_vec()).collect();
    batches.into_iter().flat_map(|batch| {
        let parsed: Vec<Vec<(String, usize, Result<PhhHand>)>> = batch
            .par_iter()
            .map(|s| {
                let short = s.short();
                match s.read() {
                    Ok(text) => parse_phh(&text).into_iter().enumerate().map(|(i, h)| (short.clone(), i, h)).collect(),
                    Err(e) => vec![(short, 0, Err(e.context(s.name())))],
                }
            })
            .collect();
        parsed.into_iter().flatten()
    })
}

// ---------------------------------------------------------------------------
// conversion
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Split {
    All,
    Train,
    Val,
}

/// Which hands / seats become samples.
pub struct Filter {
    pub cfg: HunlConfig,
    pub rescale: bool,
    pub dedup: bool,
    pub split: Split,
    pub val_per_mille: u64,
    /// A uniform subsample: keep the hands whose `key / 1000 % 1000` is below this
    /// (1000 = every hand; independent of the split bucket `key % 1000`).
    pub keep_per_mille: u64,
    /// Non-empty: emit only these players' seats (hands without them are dropped).
    pub keep_players: HashSet<String>,
    // counters
    pub other_variant: Cell<usize>,
    pub duplicates: Cell<usize>,
    pub split_skipped: Cell<usize>,
    pub subsampled: Cell<usize>,
    pub no_kept_seat: Cell<usize>,
    pub rescaled: Cell<usize>,
    pub first_other: RefCell<Vec<String>>,
    seen: RefCell<HashSet<u64>>,
}

impl Filter {
    pub fn new(cfg: HunlConfig, rescale: bool, dedup: bool, split: Split, val_per_mille: u64, keep: HashSet<String>) -> Self {
        Filter {
            cfg,
            rescale,
            dedup,
            split,
            val_per_mille,
            keep_per_mille: 1000,
            keep_players: keep,
            other_variant: Cell::new(0),
            duplicates: Cell::new(0),
            split_skipped: Cell::new(0),
            subsampled: Cell::new(0),
            no_kept_seat: Cell::new(0),
            rescaled: Cell::new(0),
            first_other: RefCell::new(Vec::new()),
            seen: RefCell::new(HashSet::new()),
        }
    }

    /// Keep only a uniform `keep_per_mille` / 1000 of the hands (hash-based: the same
    /// hands on every run and machine, all matchups alike).
    pub fn subsample(mut self, keep_per_mille: u64) -> Self {
        self.keep_per_mille = keep_per_mille;
        self
    }

    /// The chip scale of a hand that passes the variant / split / subsample /
    /// de-duplication filters (None: skipped, counted).
    fn admit(&self, h: &PhhHand) -> Option<u32> {
        let k = match scale_of(h, &self.cfg, self.rescale) {
            Ok(k) => k,
            Err(Skip::OtherVariant(m)) => {
                self.other_variant.set(self.other_variant.get() + 1);
                let mut f = self.first_other.borrow_mut();
                if f.len() < 5 {
                    f.push(m);
                }
                return None;
            }
        };
        let val = h.key % 1000 < self.val_per_mille;
        if (self.split == Split::Train && val) || (self.split == Split::Val && !val) {
            self.split_skipped.set(self.split_skipped.get() + 1);
            return None;
        }
        if h.key / 1000 % 1000 >= self.keep_per_mille {
            self.subsampled.set(self.subsampled.get() + 1);
            return None;
        }
        if self.dedup && !self.seen.borrow_mut().insert(h.key) {
            self.duplicates.set(self.duplicates.get() + 1);
            return None;
        }
        Some(k)
    }

    /// The record of an admitted hand (None: filtered, counted).
    pub fn record(&self, src: &str, idx: usize, h: Result<PhhHand>) -> Option<Result<Record>> {
        let h = match h {
            Ok(h) => h,
            Err(e) => return Some(Err(e.context(format!("{src}#{idx}")))),
        };
        let k = self.admit(&h)?;
        let seats: Option<Vec<Seat>> = (!self.keep_players.is_empty())
            .then(|| (0..2).filter(|&s| self.keep_players.contains(&h.players[s])).collect());
        if seats.as_ref().is_some_and(|s| s.is_empty()) {
            self.no_kept_seat.set(self.no_kept_seat.get() + 1);
            return None;
        }
        if k != 1 {
            self.rescaled.set(self.rescaled.get() + 1);
        }
        let id = format!("{src}#{}", h.hand.unwrap_or(idx as i64));
        let events = match hand_events(&h, &self.cfg, k) {
            Ok(e) => e,
            Err(e) => return Some(Err(e.context(id))),
        };
        let mut meta = json!({"players": h.players, "hand": h.hand, "source": src});
        if k != 1 {
            meta["scale"] = json!(k);
        }
        Some(Ok(Record {
            id,
            events,
            meta,
            seats,
            expected_returns: h.results.as_ref().map(|r| r.iter().map(|x| x * k as f64).collect()),
            soft_targets: Vec::new(),
        }))
    }

    pub fn counters(&self) -> Value {
        json!({
            "other_variant": self.other_variant.get(),
            "other_variant_examples": *self.first_other.borrow(),
            "duplicates": self.duplicates.get(),
            "split_skipped": self.split_skipped.get(),
            "subsampled": self.subsampled.get(),
            "no_kept_seat": self.no_kept_seat.get(),
            "rescaled": self.rescaled.get(),
        })
    }
}

/// Per-player results over the admitted (de-duplicated) hands: `(hands, chips won)`.
pub fn player_stats(srcs: Vec<Source>, cfg: &HunlConfig, rescale: bool, split: Split, val_per_mille: u64)
    -> (HashMap<String, (u64, f64)>, Value) {
    let f = Filter::new(cfg.clone(), rescale, true, split, val_per_mille, HashSet::new());
    let mut stats: HashMap<String, (u64, f64)> = HashMap::new();
    let mut no_results = 0usize;
    for (_, _, h) in hands(srcs) {
        let Ok(h) = h else { continue };
        let Some(k) = f.admit(&h) else { continue };
        let Some(r) = h.results.as_ref().filter(|r| r.len() == 2) else {
            no_results += 1;
            continue;
        };
        for (player, won) in h.players.iter().zip(r) {
            let e = stats.entry(player.clone()).or_default();
            e.0 += 1;
            e.1 += won * k as f64;
        }
    }
    let mut c = f.counters();
    c["no_results"] = json!(no_results);
    (stats, c)
}

/// Big blinds won per 100 hands.
pub fn bb100(hands: u64, chips: f64, big_blind: u32) -> f64 {
    if hands == 0 { 0.0 } else { chips / big_blind as f64 / hands as f64 * 100.0 }
}

// ---------------------------------------------------------------------------
// fetch: range requests into the Zenodo zip
// ---------------------------------------------------------------------------

/// A central-directory entry of a zip archive.
#[derive(Clone, Debug)]
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub crc: u32,
    pub csize: u64,
    pub usize: u64,
    pub offset: u64,
}

fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn u64le(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// `(central directory offset, size)` from the archive's last bytes (`tail` = the
/// final `tail.len()` bytes of a `total`-byte archive; zip64 aware).
pub fn zip_cd_location(tail: &[u8], total: u64) -> Result<(u64, u64)> {
    let base = total - tail.len() as u64;
    let eocd = tail.windows(4).rposition(|w| w == [0x50, 0x4b, 0x05, 0x06]).context("no end-of-central-directory record")?;
    let (mut size, mut off) = (u32le(tail, eocd + 12) as u64, u32le(tail, eocd + 16) as u64);
    if off == 0xFFFF_FFFF || size == 0xFFFF_FFFF || u16le(tail, eocd + 10) == 0xFFFF {
        let loc = tail[..eocd].windows(4).rposition(|w| w == [0x50, 0x4b, 0x06, 0x07]).context("no zip64 locator")?;
        let z = u64le(tail, loc + 8);
        anyhow::ensure!(z >= base, "zip64 end record outside the fetched tail");
        let z = (z - base) as usize;
        anyhow::ensure!(u32le(tail, z) == 0x0606_4b50, "bad zip64 end record");
        size = u64le(tail, z + 40);
        off = u64le(tail, z + 48);
    }
    Ok((off, size))
}

/// Parse a central directory.
pub fn zip_entries(cd: &[u8]) -> Result<Vec<ZipEntry>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 46 <= cd.len() && u32le(cd, i) == 0x0201_4b50 {
        let method = u16le(cd, i + 10);
        let crc = u32le(cd, i + 16);
        let (mut csize, mut usize_, mut offset) = (u32le(cd, i + 20) as u64, u32le(cd, i + 24) as u64, u32le(cd, i + 42) as u64);
        let (nlen, elen, clen) = (u16le(cd, i + 28) as usize, u16le(cd, i + 30) as usize, u16le(cd, i + 32) as usize);
        anyhow::ensure!(i + 46 + nlen + elen + clen <= cd.len(), "truncated central directory");
        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + nlen]).to_string();
        let extra = &cd[i + 46 + nlen..i + 46 + nlen + elen];
        let mut j = 0;
        while j + 4 <= extra.len() {
            let (id, sz) = (u16le(extra, j), u16le(extra, j + 2) as usize);
            if id == 1 {
                let mut k = j + 4;
                for v in [&mut usize_, &mut csize, &mut offset] {
                    if *v == 0xFFFF_FFFF && k + 8 <= j + 4 + sz {
                        *v = u64le(extra, k);
                        k += 8;
                    }
                }
            }
            j += 4 + sz;
        }
        out.push(ZipEntry { name, method, crc, csize, usize: usize_, offset });
        i += 46 + nlen + elen + clen;
    }
    Ok(out)
}

fn get_range(client: &reqwest::blocking::Client, url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let mut last = None;
    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(5 << attempt));
        }
        let r = client.get(url).header("Range", format!("bytes={start}-{}", end - 1)).send();
        match r {
            Ok(resp) if resp.status() == reqwest::StatusCode::PARTIAL_CONTENT => {
                let b = resp.bytes()?;
                if b.len() as u64 == end - start {
                    return Ok(b.to_vec());
                }
                last = Some(anyhow::anyhow!("short read: {} of {} bytes", b.len(), end - start));
            }
            Ok(resp) => last = Some(anyhow::anyhow!("HTTP {} for a range request", resp.status())),
            Err(e) => last = Some(e.into()),
        }
    }
    Err(last.unwrap().context(format!("GET {url} bytes {start}-{end}")))
}

/// The raw deflate stream of a zip member as a gzip file (no recompression).
fn gzip_wrap(deflate: &[u8], crc: u32, usize_: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(deflate.len() + 18);
    out.extend_from_slice(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255]);
    out.extend_from_slice(deflate);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(usize_ as u32).to_le_bytes());
    out
}

/// The first `n` hands (`[k]` sections) of a `.phhs` text.
pub fn first_hands(text: &str, n: usize) -> String {
    let mut out = String::new();
    let mut sections = 0;
    for line in text.split_inclusive('\n') {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') && !t.contains('=') {
            sections += 1;
            if sections > n {
                break;
            }
        }
        out.push_str(line);
    }
    out
}

pub struct FetchOptions {
    pub url: String,
    pub years: Vec<u32>,
    /// Non-empty: only members whose path contains one of these strings.
    pub name_contains: Vec<String>,
    pub max_files_per_year: Option<usize>,
    pub max_hands: Option<usize>,
    pub chunk_bytes: u64,
}

/// Download the raw ACPC heads-up no-limit logs of the archive at `url` into `out`
/// with HTTP range requests: `out/<path below competitions/>.phhs.gz` (the members'
/// deflate streams re-wrapped as gzip), or plain `.phhs` truncated to `max_hands`
/// hands. Resumable (existing files are skipped). Returns (files written, bytes).
pub fn fetch(out: &Path, o: &FetchOptions) -> Result<(usize, u64)> {
    let client = common::http()?;
    std::fs::create_dir_all(out)?;
    let head = client.head(&o.url).send()?.error_for_status()?;
    let total: u64 = head
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .context("no Content-Length")?;
    let cache = out.join(".cache");
    std::fs::create_dir_all(&cache)?;
    let tail = get_range(&client, &o.url, total - total.min(1 << 16), total)?;
    let (cd_off, cd_size) = zip_cd_location(&tail, total)?;
    let cd_path = cache.join(format!("zip_cd_{total}_{cd_off}.bin"));
    let cd = match std::fs::read(&cd_path) {
        Ok(cd) if cd.len() as u64 == cd_size => cd,
        _ => {
            eprintln!("central directory: {:.1} MB at {cd_off}", cd_size as f64 / 1e6);
            let cd = get_range(&client, &o.url, cd_off, cd_off + cd_size)?;
            std::fs::write(&cd_path, &cd)?;
            cd
        }
    };
    let mut all = zip_entries(&cd)?;
    all.sort_by_key(|e| e.offset);
    // span of entry i: [offset_i, offset_{i+1})
    let ends: Vec<u64> = (0..all.len()).map(|i| all.get(i + 1).map_or(cd_off, |e| e.offset)).collect();
    let mut per_year: HashMap<u32, usize> = HashMap::new();
    let mut want: Vec<usize> = Vec::new();
    let mut order: Vec<usize> = (0..all.len()).filter(|&i| is_raw_nl_log(&all[i].name)).collect();
    order.sort_by(|&a, &b| all[a].name.cmp(&all[b].name));
    for i in order {
        let Some(y) = year_of(&all[i].name) else { continue };
        if !o.years.is_empty() && !o.years.contains(&y) {
            continue;
        }
        if !o.name_contains.is_empty() && !o.name_contains.iter().any(|s| all[i].name.contains(s.as_str())) {
            continue;
        }
        let n = per_year.entry(y).or_default();
        if o.max_files_per_year.is_some_and(|m| *n >= m) {
            continue;
        }
        *n += 1;
        want.push(i);
    }
    want.sort_by_key(|&i| all[i].offset);
    let dest = |e: &ZipEntry| -> PathBuf {
        let rel = e.name.split("competitions/").nth(1).unwrap_or(&e.name);
        if o.max_hands.is_some() || e.method != 8 { out.join(rel) } else { out.join(format!("{rel}.gz")) }
    };
    let todo: Vec<usize> = want.into_iter().filter(|&i| !dest(&all[i]).exists()).collect();
    eprintln!("{} members to fetch ({:.2} GB)", todo.len(), todo.iter().map(|&i| ends[i] - all[i].offset).sum::<u64>() as f64 / 1e9);
    let (mut files, mut bytes) = (0usize, 0u64);
    let mut k = 0;
    while k < todo.len() {
        // a run of adjacent members, up to chunk_bytes
        let mut j = k + 1;
        while j < todo.len() && all[todo[j]].offset == ends[todo[j - 1]] && ends[todo[j]] - all[todo[k]].offset <= o.chunk_bytes {
            j += 1;
        }
        let start = all[todo[k]].offset;
        let buf = get_range(&client, &o.url, start, ends[todo[j - 1]])?;
        bytes += buf.len() as u64;
        for &i in &todo[k..j] {
            let e = &all[i];
            let at = (e.offset - start) as usize;
            anyhow::ensure!(u32le(&buf, at) == 0x0403_4b50, "{}: bad local header", e.name);
            let data = at + 30 + u16le(&buf, at + 26) as usize + u16le(&buf, at + 28) as usize;
            let raw = &buf[data..data + e.csize as usize];
            let body: Vec<u8> = match (e.method, o.max_hands) {
                (8, None) => gzip_wrap(raw, e.crc, e.usize),
                (0, None) => raw.to_vec(),
                (m, Some(n)) => {
                    let mut text = String::new();
                    if m == 8 {
                        flate2::read::DeflateDecoder::new(raw).read_to_string(&mut text)?;
                    } else {
                        text = String::from_utf8(raw.to_vec())?;
                    }
                    first_hands(&text, n).into_bytes()
                }
                (m, None) => bail!("{}: unsupported compression method {m}", e.name),
            };
            let d = dest(e);
            std::fs::create_dir_all(d.parent().unwrap())?;
            let tmp = d.with_extension("part");
            std::fs::File::create(&tmp)?.write_all(&body)?;
            std::fs::rename(&tmp, &d)?;
            files += 1;
        }
        eprintln!("fetched {files} / {} members, {:.2} GB", todo.len(), bytes as f64 / 1e9);
        k = j;
    }
    Ok((files, bytes))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
pub enum Cmd {
    /// Download the raw ACPC heads-up no-limit logs (HTTP range requests into the
    /// Zenodo zip: only those members, 4.85 GB; resumable)
    Fetch {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = ZENODO_URL)]
        url: String,
        /// competition years (default: all of 2009-2017)
        #[arg(long, value_delimiter = ',')]
        years: Vec<u32>,
        /// only members whose archive path contains this string (repeatable)
        #[arg(long)]
        name_contains: Vec<String>,
        /// a sample: at most this many log files per year
        #[arg(long)]
        max_files_per_year: Option<usize>,
        /// a sample: keep only the first N hands of each file (plain .phhs)
        #[arg(long)]
        max_hands: Option<usize>,
        /// bytes per range request
        #[arg(long, default_value_t = 256)]
        chunk_mb: u64,
    },
    /// Convert PHH hand histories (directories of .phh / .phhs[.gz], or the zip) to
    /// candidate-layout gm-shards (every hand replayed and verified)
    Convert {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// the gmrules hunl config (JSON; default 50 / 100 / 20000)
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long, value_delimiter = ',')]
        years: Vec<u32>,
        /// also convert hands of the same stack depth at another chip scale (2009: 1 / 2 / 400)
        #[arg(long)]
        rescale: bool,
        /// keep only the seats of winning players (bb/100 >= --min-bb100 over the input)
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 0.0)]
        min_bb100: f64,
        /// keep only these players' seats (repeatable)
        #[arg(long)]
        player: Vec<String>,
        #[arg(long, value_enum, default_value_t = Split::All)]
        split: Split,
        #[arg(long, default_value_t = 5)]
        val_per_mille: u64,
        /// a uniform hash-based subsample: keep this many hands per 1000 (all matchups
        /// alike; unlike --max-records, which takes the first hands in file order)
        #[arg(long, default_value_t = 1000)]
        keep_per_mille: u64,
        /// keep duplicate hands
        #[arg(long)]
        no_dedup: bool,
        #[arg(long)]
        max_records: Option<usize>,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep unfinished hands (value_mask false; the causality tests)
        #[arg(long)]
        allow_truncated: bool,
    },
    /// Per-player results (hands, chips, bb/100) of the admitted, de-duplicated hands
    Stats {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long, value_delimiter = ',')]
        years: Vec<u32>,
        #[arg(long)]
        rescale: bool,
    },
}

fn parse_config(s: &str) -> Result<HunlConfig> {
    let j = if s.trim().is_empty() { "{}" } else { s };
    let cfg: HunlConfig = serde_json::from_str(j).with_context(|| format!("hunl config {s:?}"))?;
    Hunl::check_config(&cfg).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(cfg)
}

fn stats_json(stats: &HashMap<String, (u64, f64)>, bb: u32) -> Value {
    let mut v: Vec<(&String, &(u64, f64))> = stats.iter().collect();
    v.sort_by(|a, b| bb100(b.1 .0, b.1 .1, bb).total_cmp(&bb100(a.1 .0, a.1 .1, bb)).then(a.0.cmp(b.0)));
    Value::Array(v.into_iter().map(|(p, &(n, c))| json!({"player": p, "hands": n, "chips": c, "bb100": bb100(n, c, bb)})).collect())
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, url, years, name_contains, max_files_per_year, max_hands, chunk_mb } => {
            let o = FetchOptions { url, years, name_contains, max_files_per_year, max_hands, chunk_bytes: chunk_mb << 20 };
            let (files, bytes) = fetch(&out, &o)?;
            println!("{}", json!({"files": files, "bytes": bytes, "out": out}));
            Ok(())
        }
        Cmd::Stats { input, config, years, rescale } => {
            let cfg = parse_config(&config)?;
            let (stats, counters) = player_stats(sources(&input, &years)?, &cfg, rescale, Split::All, 0);
            println!("{}", json!({"players": stats_json(&stats, cfg.big_blind), "filters": counters}));
            Ok(())
        }
        Cmd::Convert { input, out, config, years, rescale, hq, min_bb100, player, split, val_per_mille, keep_per_mille, no_dedup, max_records, shard_size, store_events, allow_truncated } => {
            let cfg = parse_config(&config)?;
            anyhow::ensure!(val_per_mille <= 1000 && (1..=1000).contains(&keep_per_mille), "--val-per-mille must be <= 1000, --keep-per-mille in 1..=1000");
            let srcs = sources(&input, &years)?;
            anyhow::ensure!(!srcs.is_empty(), "no .phh / .phhs inputs under {input:?} (years {years:?})");
            let mut keep: HashSet<String> = player.into_iter().collect();
            let mut hq_info = Value::Null;
            if hq {
                // the winning players of THIS input (same filters, all splits: a
                // player's strength is measured on every hand it played)
                let (stats, _) = player_stats(srcs.clone(), &cfg, rescale, Split::All, val_per_mille);
                let winners: Vec<&String> = stats.iter().filter(|(_, &(n, c))| bb100(n, c, cfg.big_blind) >= min_bb100).map(|(p, _)| p).collect();
                anyhow::ensure!(!winners.is_empty(), "--hq: no player reaches {min_bb100} bb/100");
                keep.extend(winners.into_iter().cloned());
                hq_info = json!({"min_bb100": min_bb100, "players": stats_json(&stats, cfg.big_blind), "kept": keep.len()});
            }
            let filter = Filter::new(cfg.clone(), rescale, !no_dedup, split, val_per_mille, keep).subsample(keep_per_mille);
            let recs = hands(srcs).filter_map(|(src, i, h)| filter.record(&src, i, h)).take(max_records.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, ..Default::default() };
            let st = traj::convert::<Hunl>(&cfg, recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["filters"] = filter.counters();
            if !hq_info.is_null() {
                v["hq"] = hq_info;
            }
            println!("{v}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrules::replay;

    const SAMPLE: &str = "[1]
variant = 'NT'
ante_trimming_status = false
antes = [0, 0]
blinds_or_straddles = [50, 100]
min_bet = 100
starting_stacks = [20000, 20000]
actions = ['d dh p1 Jc8c', 'd dh p2 Kd5c', 'p2 cbr 200', 'p1 cbr 600', 'p2 cc', 'd db Qs2dTd', 'p1 cbr 835', 'p2 f']
hand = 3
players = ['Rembrant6_2pn_2017', 'Slumbot_2pn_2017']
_results = [600, -600]

[2]
variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [50, 100]
min_bet = 100
starting_stacks = [20000, 20000]
actions = [
  'd dh p1 Ad4s', 'd dh p2 9sAc', 'p2 cbr 200', 'p1 cc', 'd db KhAhKs', 'p1 cbr 250', 'p2 cc',
  'd db 3d', 'p1 cbr 250', 'p2 cc', 'd db 3c', 'p1 cbr 250', 'p2 cc', 'p1 sm Ad4s', 'p2 sm 9sAc',
]
hand = 4
players = ['Slumbot_2pn_2017', 'Rembrant6_2pn_2017']
_results = [-950, 950]
";

    const SAMPLE_2009: &str = "[1]
variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [1, 2]
min_bet = 2
starting_stacks = [400, 400]
actions = ['d dh p1 5cKs', 'd dh p2 KcAs', 'p2 cbr 5', 'p1 cc', 'd db 2hQh4s', 'p1 cc', 'p2 cc', 'd db 6d', 'p1 cc', 'p2 cc', 'd db Ac', 'p1 cc', 'p2 cbr 8', 'p1 f']
hand = 0
players = ['Tartanian3', 'BluffBot4']
_results = [-5, 5]
";

    #[test]
    fn toml_subset_values() {
        assert_eq!(parse_value("'NT'").unwrap(), json!("NT"));
        assert_eq!(parse_value("\"a\\\"b\"").unwrap(), json!("a\"b"));
        assert_eq!(parse_value("[1, 2.5, -3, true, 'x']").unwrap(), json!([1, 2.5, -3, true, "x"]));
        assert_eq!(parse_value("[[1, 2], []]  # c").unwrap(), json!([[1, 2], []]));
        assert_eq!(parse_value("19545.454545").unwrap(), json!(19545.454545));
        assert!(parse_value("[1, 2").is_err() && parse_value("'x").is_err() && parse_value("1 2").is_err());
    }

    #[test]
    fn phh_hands_to_trajectories() {
        let hands = parse_phh(SAMPLE);
        assert_eq!(hands.len(), 2);
        let (a, b) = (hands[0].as_ref().unwrap(), hands[1].as_ref().unwrap());
        assert_eq!(a.actions.len(), 8);
        assert_eq!(b.actions.len(), 15, "multi-line array with a trailing comma");
        assert_eq!((a.hand, b.results.clone()), (Some(3), Some(vec![-950.0, 950.0])));
        assert_ne!(a.key, b.key);
        // the key (dedup, split, subsample) is pinned: the same on every machine / release
        assert_eq!(a.key, 12524262783637424732);
        let mut h = StableHash::new();
        h.write(b"a");
        assert_eq!(h.finish(), 9413272369427828315, "FNV-1a 64 (0xaf63dc4c8601ec8c) + fmix64");
        let cfg = HunlConfig::default();
        assert_eq!(scale_of(a, &cfg, false), Ok(1));
        let ev = hand_events(a, &cfg, 1).unwrap();
        // hole cards (seat 0 then seat 1), SB raises to 200, BB 3-bets to 600, called;
        // the flop bet of 835 is a raise to 600 + 835 = 1435 total
        assert_eq!(ev[..4].iter().filter(|e| matches!(e, Event::Chance(_))).count(), 4);
        assert_eq!(ev[4..7], [Event::Act(1, 200), Event::Act(0, 600), Event::Act(1, CALL)]);
        assert_eq!(ev[10..], [Event::Act(0, 1435), Event::Act(1, FOLD)]);
        for h in [a, b] {
            let r = replay::<Hunl>(&cfg, &hand_events(h, &cfg, 1).unwrap()).unwrap();
            assert!(r.terminal());
            assert_eq!(Some(r.state.returns()), h.results);
        }
        // a showdown claiming other cards is rejected
        let mut bad = b.clone();
        bad.actions[13] = "p1 sm AdAs".into();
        assert!(hand_events(&bad, &cfg, 1).is_err());
        // the 2009 game (1 / 2 / 400): another variant unless rescaled x50
        let h09 = parse_phh(SAMPLE_2009).remove(0).unwrap();
        assert!(matches!(scale_of(&h09, &cfg, false), Err(Skip::OtherVariant(_))));
        assert_eq!(scale_of(&h09, &cfg, true), Ok(50));
        let ev = hand_events(&h09, &cfg, 50).unwrap();
        assert_eq!(ev[4], Event::Act(1, 250));
        let r = replay::<Hunl>(&cfg, &ev).unwrap();
        assert_eq!(r.state.returns(), vec![-250.0, 250.0]);
        // natively, as its own config
        let c09 = HunlConfig { small_blind: 1, big_blind: 2, stack: 400 };
        assert_eq!(replay::<Hunl>(&c09, &hand_events(&h09, &c09, 1).unwrap()).unwrap().state.returns(), vec![-5.0, 5.0]);
        assert!(parse_phh("[1]\nvariant = 'NT'\nactions = ['x'").iter().all(|h| h.is_err()));
        assert!(is_raw_nl_log("data/annual-computer-poker-competition/competitions/2017/logs/logs_2pn_2017/logs_2pn_2017/a.b.1.0.phhs"));
        assert!(!is_raw_nl_log("data/annual-computer-poker-competition/competitions/2017/logs/processed_logs_2pn_2017/x/a.phhs"));
        assert!(!is_raw_nl_log("data/annual-computer-poker-competition/competitions/2012/logs/acpc_2012_2p_limit/a.phhs"));
        assert_eq!(year_of("x/competitions/2016/logs/y.phhs"), Some(2016));
    }

    #[test]
    fn convert_filters_dedup_split_and_hq() {
        let dir = std::env::temp_dir().join(format!("gmdata-hunl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("raw/2017")).unwrap();
        std::fs::create_dir_all(dir.join("raw/2009")).unwrap();
        std::fs::write(dir.join("raw/2017/a.phhs"), SAMPLE).unwrap();
        // the same hands again (an archive copy), gzipped
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(SAMPLE.as_bytes()).unwrap();
        std::fs::write(dir.join("raw/2017/copy.phhs.gz"), gz.finish().unwrap()).unwrap();
        std::fs::write(dir.join("raw/2009/b.phhs"), SAMPLE_2009).unwrap();
        std::fs::write(dir.join("raw/2017/processed_x.phhs"), SAMPLE).unwrap();
        let cfg = HunlConfig::default();
        let srcs = sources(&[dir.join("raw")], &[]).unwrap();
        assert_eq!(srcs.len(), 3, "processed copies are skipped");
        let run = |f: &Filter| {
            let recs = hands(sources(&[dir.join("raw")], &[]).unwrap()).filter_map(|(s, i, h)| f.record(&s, i, h));
            let out = dir.join(format!("out{}", rand_suffix()));
            traj::convert::<Hunl>(&cfg, recs, &out, &ConvertOptions::default()).unwrap()
        };
        let f = Filter::new(cfg.clone(), false, true, Split::All, 5, HashSet::new());
        let st = run(&f);
        assert_eq!((st.converted, st.illegal, st.returns_mismatch, st.samples), (2, 0, 0, 4));
        assert_eq!((f.duplicates.get(), f.other_variant.get()), (2, 1));
        let f = Filter::new(cfg.clone(), true, true, Split::All, 5, HashSet::new());
        assert_eq!(run(&f).converted, 3);
        assert_eq!(f.rescaled.get(), 1);
        // splits partition the hands
        let (tr, va) = (Filter::new(cfg.clone(), true, true, Split::Train, 500, HashSet::new()), Filter::new(cfg.clone(), true, true, Split::Val, 500, HashSet::new()));
        assert_eq!(run(&tr).converted + run(&va).converted, 3);
        // the uniform subsample: all or nothing at the extremes, counted
        let f = Filter::new(cfg.clone(), true, true, Split::All, 5, HashSet::new()).subsample(0);
        assert_eq!((run(&f).converted, f.subsampled.get()), (0, 5));
        let f = Filter::new(cfg.clone(), true, true, Split::All, 5, HashSet::new()).subsample(1000);
        assert_eq!((run(&f).converted, f.subsampled.get()), (3, 0));
        // --hq: Slumbot won 600 - 950 < 0 over two hands, Rembrant6 won
        let (stats, _) = player_stats(sources(&[dir.join("raw")], &[2017]).unwrap(), &cfg, false, Split::All, 0);
        assert_eq!(stats["Slumbot_2pn_2017"], (2, -1550.0));
        assert_eq!(stats["Rembrant6_2pn_2017"], (2, 1550.0));
        let keep: HashSet<String> = ["Rembrant6_2pn_2017".to_string()].into();
        let f = Filter::new(cfg.clone(), false, true, Split::All, 5, keep);
        let st = run(&f);
        assert_eq!((st.converted, st.samples), (2, 2));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn rand_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn zip_directory_parsing_and_gzip_members() {
        // a zip written by the zip crate: locate its central directory from the tail,
        // re-wrap a member's deflate stream as gzip and read it back
        let dir = std::env::temp_dir().join(format!("gmdata-hunl-zip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
            let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            z.start_file("data/annual-computer-poker-competition/competitions/2017/logs/logs_2pn_2017/x.phhs", o).unwrap();
            z.write_all(SAMPLE.as_bytes()).unwrap();
            z.start_file("data/other.txt", o).unwrap();
            z.write_all(b"hello").unwrap();
            z.finish().unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let (off, size) = zip_cd_location(&bytes[bytes.len().saturating_sub(1000)..], bytes.len() as u64).unwrap();
        let entries = zip_entries(&bytes[off as usize..(off + size) as usize]).unwrap();
        assert_eq!(entries.len(), 2);
        let e = &entries[0];
        let at = e.offset as usize;
        let data = at + 30 + u16le(&bytes, at + 26) as usize + u16le(&bytes, at + 28) as usize;
        let gz = gzip_wrap(&bytes[data..data + e.csize as usize], e.crc, e.usize);
        let text = String::from_utf8(common::read_maybe_gz(gz).unwrap()).unwrap();
        assert_eq!(text, SAMPLE);
        assert_eq!(first_hands(&text, 1).matches("variant").count(), 1);
        // the zip as a converter input: only the raw no-limit member
        let srcs = sources(&[path], &[]).unwrap();
        assert_eq!(srcs.len(), 1);
        assert_eq!(hands(srcs).filter(|(_, _, h)| h.is_ok()).count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
