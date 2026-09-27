//! Lichess chess variants — chess960, crazyhouse, atomic, antichess, three-check, king
//! of the hill, horde, racing kings: the monthly per-variant PGN dumps of
//! database.lichess.org (CC0) → verified trajectories → gm-shards of the gmrules game
//! `chess_variants` (rules, frames and the schema: `gmrules::games::chess_variants`).
//!
//! ```sh
//! gmdata chess-variants list    --variant atomic                     # months (+ game counts)
//! gmdata chess-variants fetch   --variant atomic --month 2026-08 --out data/chess_variants/raw
//! gmdata chess-variants fetch   --variant all --month 2026-08 --max-bytes 2000000 --out /tmp/cv  # samples
//! gmdata chess-variants stats   --input data/chess_variants/raw
//! gmdata chess-variants convert --input data/chess_variants/raw/lichess_db_atomic_rated_2026-08.pgn.zst \
//!                               --out data/chess_variants/train/atomic/2026-08 --hq
//! ```
//! (full-scale: `scripts/fetch_chess_variants.sh`, which downloads and converts month by
//! month and variant by variant, resumable, raw dumps deleted unless `KEEP_RAW=1`.)
//!
//! Parsing: the SAN of every move is resolved in the variant's position (shakmaty, the
//! rules library of the gmrules game) into the canonical action id; the record is the
//! trajectory `Chance(variant)`, (chess960) `Chance(Scharnagl start position)`, then
//! the moves (White = seat 0). [`traj::convert`] then REPLAYS every record with the
//! game's rules — every move legal, no move after an automatic end (mate, variant end,
//! stalemate, lichess' insufficient-material / dead-position draws, fifty moves,
//! fivefold repetition) — and checks the PGN result against the rules' result when the
//! game ends on the board; games that end off the board (resignation, time forfeit,
//! agreed or claimed draw) take the PGN result as their value target
//! (`Record::final_outcome`, `adjudicated`). `illegal` and `returns_mismatch` in the
//! output stats are the parity check with lichess' own rules on real games: 0 and 0 on
//! all 333k games of the 2026-08 samples (tests/fixtures/chess_variants holds 49 of them).
//!
//! Skipped before the replay: other / unknown variants, results `*` (kept with
//! `--allow-truncated`), start positions that are not the variant's (lichess "From
//! Position" games; chess960 FENs must be one of the 960 start positions with full
//! castling rights), null moves / unparseable SAN (`unparseable`), and the quality
//! filters:
//!
//! - `--min-elo N` / `--min-elo-variant V=N`: both players' Elo >= N;
//! - `--top-percent P`: per input file and variant, the game's lower Elo is in the top
//!   P % (a first pass over the file's headers computes the threshold) — robust to the
//!   different rating pools of the variants and to rating drift over the years;
//! - `--rated-only`, `--normal-termination` (drops time forfeits, abandoned games, rules
//!   infractions and "Insufficient material" = a flag fell against a bare king),
//!   `--min-duration S` (base + 40·increment >= S seconds), `--min-plies N`, `--no-bots`,
//!   `--min-clock-s S` (below);
//! - `--hq`: rated, normal termination, >= 60 s (no hyperbullet), >= 4 plies, and — unless
//!   an Elo filter is given — the top 10 % of each month and variant (measured on the
//!   2026-08 samples: 6.6-8.1 % of the games kept, 0 illegal, 0 mismatches).
//!
//! Clocks (`[%clk]`) are NOT observation features: frames are functions of the
//! trajectory (the rules state) and the RL simulator has no clock, so a clock input would
//! be real in IL and a constant in RL — the network would learn a time-conditioned policy
//! and play an input it never saw. They serve the data quality instead: `--min-clock-s
//! S` cuts each game before the first move its player made with less than S seconds on
//! the clock (the kept prefix is adjudicated with the PGN result; `clock_cut_ply` in the
//! sample meta). On the 2026-08 `--hq` samples S = 10 drops 6 % (three-check) to 35 %
//! (antichess, much 1+0 / 0+2 bullet) of the frames, S = 5 3 % to 31 %; off by default.

use crate::common;
use crate::traj::{self, ConvertOptions, Record};
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::games::chess_variants::shakmaty::fen::Fen;
use gmrules::games::chess_variants::shakmaty::san::San;
use gmrules::games::chess_variants::shakmaty::variant::VariantPosition;
use gmrules::games::chess_variants::shakmaty::{CastlingMode, EnPassantMode, Position, Setup};
use gmrules::games::chess_variants::{self as cv, ChessVariant, ChessVariants, ChessVariantsConfig};
use gmrules::Event;
use pgn_reader::{RawComment, RawTag, Reader, SanPlus, Visitor};
use serde_json::{json, Value};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufReader, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

/// The PGN tags the converter reads.
const TAGS: [&str; 13] = [
    "Event", "Site", "White", "Black", "Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "Variant",
    "FEN", "WhiteTitle", "BlackTitle",
];

/// The headers of one game.
#[derive(Clone, Debug, Default)]
pub struct Header {
    tags: HashMap<&'static str, String>,
}

impl Header {
    fn get(&self, k: &str) -> Option<&str> {
        self.tags.get(k).map(|s| s.as_str())
    }

    fn elo(&self, k: &str) -> Option<i64> {
        self.get(k).and_then(|v| v.trim().parse().ok())
    }

    /// The lower of the two Elos (None when a player is unrated / missing).
    fn min_elo(&self) -> Option<i64> {
        Some(self.elo("WhiteElo")?.min(self.elo("BlackElo")?))
    }

    fn variant(&self) -> Option<ChessVariant> {
        ChessVariant::from_name(self.get("Variant")?)
    }

    /// Per-seat outcome of the PGN result (None: `*` / unknown).
    fn outcome(&self) -> Option<Vec<f32>> {
        match self.get("Result")? {
            "1-0" => Some(vec![1.0, -1.0]),
            "0-1" => Some(vec![-1.0, 1.0]),
            "1/2-1/2" => Some(vec![0.0, 0.0]),
            _ => None,
        }
    }

    /// Estimated duration base + 40·increment in seconds (None: correspondence / unknown).
    fn duration_s(&self) -> Option<i64> {
        let (b, i) = self.get("TimeControl")?.split_once('+')?;
        Some(b.trim().parse::<i64>().ok()? + 40 * i.trim().parse::<i64>().ok()?)
    }

    fn is_bot_game(&self) -> bool {
        self.get("WhiteTitle") == Some("BOT") || self.get("BlackTitle") == Some("BOT")
    }

    fn id(&self) -> String {
        match self.get("Site") {
            Some(s) => format!("lichess:{}", s.rsplit('/').next().unwrap_or(s)),
            None => String::from("lichess:?"),
        }
    }
}

/// Quality filters (all off by default; `--hq` turns on the preset of the module docs).
#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// variants to keep (empty = all)
    pub variants: Vec<ChessVariant>,
    pub min_elo: i64,
    pub min_elo_variant: HashMap<ChessVariant, i64>,
    /// keep the top P % of each file and variant by the lower Elo (0 = off)
    pub top_percent: f64,
    pub rated_only: bool,
    pub normal_termination: bool,
    pub min_duration_s: i64,
    pub min_plies: usize,
    pub no_bots: bool,
    /// cut each game before the first move its player made with less than this many
    /// seconds on the clock (`[%clk]`; 0 = off): time-scramble moves are not learnt
    pub min_clock_s: i64,
    /// keep unfinished games (result `*`): replayed, written with `tgt.value_mask` false
    /// (`--allow-truncated`, causality tests)
    pub keep_unfinished: bool,
}

/// Counters of the games the parser saw (printed next to the converter stats).
#[derive(Debug, Default)]
pub struct Counts {
    pub games: Cell<usize>,
    pub other_variant: Cell<usize>,
    pub no_result: Cell<usize>,
    pub filtered: Cell<usize>,
    pub unsupported_start: Cell<usize>,
    pub short: Cell<usize>,
    /// games cut before a time-scramble move (`--min-clock-s`)
    pub clock_cut: Cell<usize>,
    pub by_variant: std::cell::RefCell<BTreeMap<&'static str, usize>>,
    /// the first unsupported start positions (for the log)
    pub start_examples: std::cell::RefCell<Vec<String>>,
}

fn bump(c: &Cell<usize>) {
    c.set(c.get() + 1)
}

impl Counts {
    fn json(&self) -> Value {
        json!({
            "games": self.games.get(), "other_variant": self.other_variant.get(), "no_result": self.no_result.get(),
            "filtered": self.filtered.get(), "unsupported_start": self.unsupported_start.get(),
            "short": self.short.get(), "clock_cut": self.clock_cut.get(),
            "records_by_variant": *self.by_variant.borrow(),
            "unsupported_start_examples": *self.start_examples.borrow(),
        })
    }
}

impl Filter {
    /// The header-level filters (not the Elo threshold of `--top-percent`, not plies).
    fn passes_header(&self, h: &Header, v: ChessVariant) -> bool {
        if self.rated_only && !h.get("Event").is_some_and(|e| e.starts_with("Rated")) {
            return false;
        }
        if self.normal_termination && h.get("Termination") != Some("Normal") {
            return false;
        }
        if self.min_duration_s > 0 && h.duration_s().is_none_or(|d| d < self.min_duration_s) {
            return false;
        }
        if self.no_bots && h.is_bot_game() {
            return false;
        }
        let min = *self.min_elo_variant.get(&v).unwrap_or(&self.min_elo);
        if min > 0 && h.min_elo().is_none_or(|e| e < min) {
            return false;
        }
        true
    }
}

/// Per-variant lower-Elo thresholds of `--top-percent` (games passing the header filters).
pub type Thresholds = HashMap<ChessVariant, i64>;

/// What the parser makes of one game.
enum Parsed {
    Skip(Skip),
    /// headers-only pass
    Header(ChessVariant, Header),
    Game {
        header: Header,
        variant: ChessVariant,
        start: Option<u32>,
        moves: Vec<u32>,
        clocks: Vec<Option<f64>>,
        error: Option<String>,
    },
}

enum Skip {
    OtherVariant,
    NoResult,
    Filtered,
    UnsupportedStart(String),
}

struct Movetext {
    header: Header,
    variant: ChessVariant,
    start: Option<u32>,
    pos: VariantPosition,
    moves: Vec<u32>,
    /// `[%clk]` after each move: the mover's clock after it (increment included), seconds
    clocks: Vec<Option<f64>>,
    error: Option<String>,
}

/// The `[%clk H:MM:SS(.f)]` of a comment, in seconds.
pub fn parse_clock(comment: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(comment).ok()?;
    let rest = &s[s.find("[%clk")? + 5..];
    let t = rest[..rest.find(']')?].trim();
    let mut parts = t.split(':');
    let (h, m, sec) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let v = h.parse::<f64>().ok()? * 3600.0 + m.parse::<f64>().ok()? * 60.0 + sec.parse::<f64>().ok()?;
    v.is_finite().then_some(v)
}

/// The number of plies to keep with `--min-clock-s S`: the index of the first move its
/// player made with less than S seconds on the clock (the clock before a move = the
/// player's clock after their previous move; each side's first move is never cut —
/// lichess starts the clocks after it; unknown clocks never cut), else all.
pub fn clock_cut(clocks: &[Option<f64>], n_moves: usize, min_clock_s: i64) -> usize {
    if min_clock_s <= 0 {
        return n_moves;
    }
    (2..n_moves)
        .find(|&k| clocks.get(k - 2).copied().flatten().is_some_and(|c| c < min_clock_s as f64))
        .unwrap_or(n_moves)
}

struct Parser<'a> {
    filter: &'a Filter,
    thresholds: Option<&'a Thresholds>,
    headers_only: bool,
}

/// The start position a game's tags ask for: (Scharnagl number for chess960, the
/// position); an error for a start that is not the variant's.
fn start_of(v: ChessVariant, h: &Header) -> std::result::Result<(Option<u32>, VariantPosition), String> {
    let Some(fen) = h.get("FEN") else {
        if v == ChessVariant::Chess960 {
            return Err("chess960 game without a FEN".into());
        }
        return Ok((None, cv::start_position(v, 0)));
    };
    let setup: Setup = Fen::from_ascii(fen.as_bytes()).map_err(|e| format!("FEN {fen:?}: {e}"))?.into_setup();
    let (sp, want) = if v == ChessVariant::Chess960 {
        let sp = cv::scharnagl_of(&setup.board).ok_or_else(|| format!("FEN {fen:?} is no chess960 start position"))?;
        (Some(sp), cv::start_position(v, sp))
    } else {
        (None, cv::start_position(v, 0))
    };
    // the variant's start: board, turn, castling rights, counters (pockets / checks
    // default) — compared through the positions' setups
    let mode = if v == ChessVariant::Chess960 { CastlingMode::Chess960 } else { CastlingMode::Standard };
    let got = VariantPosition::from_setup(v.shakmaty(), setup, mode).map_err(|e| format!("FEN {fen:?}: {e}"))?;
    if got.to_setup(EnPassantMode::Legal) != want.to_setup(EnPassantMode::Legal) {
        return Err(format!("FEN {fen:?} is not the {} start position", v.name()));
    }
    Ok((sp, want))
}

impl Visitor for Parser<'_> {
    type Tags = Header;
    type Movetext = Movetext;
    type Output = Parsed;

    fn begin_tags(&mut self) -> ControlFlow<Parsed, Header> {
        ControlFlow::Continue(Header::default())
    }

    fn tag(&mut self, tags: &mut Header, name: &[u8], value: RawTag<'_>) -> ControlFlow<Parsed> {
        if let Some(k) = TAGS.iter().find(|k| k.as_bytes() == name) {
            tags.tags.insert(*k, value.decode_utf8_lossy().to_string());
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, h: Header) -> ControlFlow<Parsed, Movetext> {
        let Some(v) = h.variant().filter(|v| self.filter.variants.is_empty() || self.filter.variants.contains(v)) else {
            return ControlFlow::Break(Parsed::Skip(Skip::OtherVariant));
        };
        if h.outcome().is_none() && !self.filter.keep_unfinished {
            return ControlFlow::Break(Parsed::Skip(Skip::NoResult));
        }
        if !self.filter.passes_header(&h, v) {
            return ControlFlow::Break(Parsed::Skip(Skip::Filtered));
        }
        if self.headers_only {
            return ControlFlow::Break(Parsed::Header(v, h));
        }
        if let Some(t) = self.thresholds.and_then(|t| t.get(&v)) {
            if h.min_elo().is_none_or(|e| e < *t) {
                return ControlFlow::Break(Parsed::Skip(Skip::Filtered));
            }
        }
        match start_of(v, &h) {
            Ok((start, pos)) => ControlFlow::Continue(Movetext {
                header: h, variant: v, start, pos, moves: Vec::new(), clocks: Vec::new(), error: None,
            }),
            Err(e) => ControlFlow::Break(Parsed::Skip(Skip::UnsupportedStart(e))),
        }
    }

    fn comment(&mut self, mt: &mut Movetext, comment: RawComment<'_>) -> ControlFlow<Parsed> {
        // the first [%clk] after a move is its player's clock
        if mt.clocks.len() < mt.moves.len() {
            if let Some(c) = parse_clock(comment.as_bytes()) {
                mt.clocks.resize(mt.moves.len() - 1, None);
                mt.clocks.push(Some(c));
            }
        }
        ControlFlow::Continue(())
    }

    fn san(&mut self, mt: &mut Movetext, san_plus: SanPlus) -> ControlFlow<Parsed> {
        if mt.error.is_some() {
            return ControlFlow::Continue(());
        }
        if matches!(san_plus.san, San::Null) {
            mt.error = Some(format!("null move at ply {}", mt.moves.len() + 1));
            return ControlFlow::Continue(());
        }
        match san_plus.san.to_move(&mt.pos) {
            Ok(m) => {
                mt.moves.push(cv::action_id(&m));
                mt.pos.play_unchecked(m);
            }
            Err(_) => mt.error = Some(format!("SAN {} illegal at ply {}", san_plus, mt.moves.len() + 1)),
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, mt: Movetext) -> Parsed {
        Parsed::Game {
            header: mt.header, variant: mt.variant, start: mt.start, moves: mt.moves, clocks: mt.clocks, error: mt.error,
        }
    }
}

/// A `.pgn` / `.pgn.zst` / `.pgn.gz` file, streamed.
fn open(path: &Path) -> Result<Box<dyn Read>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let name = path.to_string_lossy().to_lowercase();
    if name.ends_with(".zst") {
        let mut d = zstd::stream::read::Decoder::new(f)?;
        d.window_log_max(31)?;
        Ok(Box::new(d))
    } else if name.ends_with(".gz") {
        Ok(Box::new(flate2::read::MultiGzDecoder::new(f)))
    } else {
        Ok(Box::new(f))
    }
}

fn reader(path: &Path) -> Result<Reader<BufReader<Box<dyn Read>>>> {
    Ok(Reader::new(BufReader::with_capacity(1 << 20, open(path)?)))
}

/// `--top-percent`: a headers-only pass over `path` → the lower-Elo threshold per variant.
pub fn thresholds(path: &Path, filter: &Filter) -> Result<Thresholds> {
    let mut elos: HashMap<ChessVariant, Vec<i64>> = HashMap::new();
    let mut r = reader(path)?;
    let mut p = Parser { filter, thresholds: None, headers_only: true };
    while let Some(g) = r.read_game(&mut p)? {
        if let Parsed::Header(v, h) = g {
            if let Some(e) = h.min_elo() {
                elos.entry(v).or_default().push(e);
            }
        }
    }
    Ok(elos
        .into_iter()
        .map(|(v, mut e)| {
            e.sort_unstable();
            let keep = ((e.len() as f64) * filter.top_percent / 100.0).ceil() as usize;
            let i = e.len().saturating_sub(keep.max(1));
            (v, e[i])
        })
        .collect())
}

/// The trajectory of a parsed game (`clock_cut`: the moves kept by `--min-clock-s`).
fn record_of(header: Header, variant: ChessVariant, start: Option<u32>, moves: Vec<u32>, clock_cut: Option<usize>) -> Record {
    let mut events = Vec::with_capacity(moves.len() + 2);
    events.push(Event::Chance(variant.index()));
    if let Some(sp) = start {
        events.push(Event::Chance(sp));
    }
    events.extend(moves.iter().enumerate().map(|(i, &a)| Event::Act(i % 2, a)));
    let mut meta = json!({
        "variant": variant.name(),
        "white": header.get("White"),
        "black": header.get("Black"),
        "elo": [header.elo("WhiteElo"), header.elo("BlackElo")],
        "result": header.get("Result"),
        "termination": header.get("Termination"),
        "time_control": header.get("TimeControl"),
    });
    if let Some(k) = clock_cut {
        meta["clock_cut_ply"] = json!(k);
    }
    Record { id: header.id(), events, meta, final_outcome: header.outcome(), ..Default::default() }
}

/// The records of every game in `files`, streamed in order (a PGN syntax error ends a
/// file with an `Err` item); `--top-percent` thresholds are computed per file first.
pub fn records<'a>(files: Vec<PathBuf>, filter: &'a Filter, counts: &'a Counts)
    -> impl Iterator<Item = Result<Record>> + 'a {
    files.into_iter().flat_map(move |path| -> Box<dyn Iterator<Item = Result<Record>> + 'a> {
        let th = if filter.top_percent > 0.0 {
            match thresholds(&path, filter) {
                Ok(t) => {
                    eprintln!("{}: top {} % thresholds {:?}", path.display(), filter.top_percent,
                              t.iter().map(|(v, e)| (v.name(), e)).collect::<BTreeMap<_, _>>());
                    Some(t)
                }
                Err(e) => return Box::new(std::iter::once(Err(e))),
            }
        } else {
            None
        };
        let mut r = match reader(&path) {
            Ok(r) => r,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut done = false;
        Box::new(std::iter::from_fn(move || {
            while !done {
                let mut p = Parser { filter, thresholds: th.as_ref(), headers_only: false };
                let g = match r.read_game(&mut p) {
                    Ok(Some(g)) => g,
                    Ok(None) => {
                        done = true;
                        return None;
                    }
                    Err(e) => {
                        done = true;
                        return Some(Err(anyhow::Error::from(e).context(format!("{name}: PGN syntax"))));
                    }
                };
                bump(&counts.games);
                match g {
                    Parsed::Skip(Skip::OtherVariant) => bump(&counts.other_variant),
                    Parsed::Skip(Skip::NoResult) => bump(&counts.no_result),
                    Parsed::Skip(Skip::Filtered) => bump(&counts.filtered),
                    Parsed::Skip(Skip::UnsupportedStart(e)) => {
                        bump(&counts.unsupported_start);
                        let mut ex = counts.start_examples.borrow_mut();
                        if ex.len() < 5 {
                            ex.push(e);
                        }
                    }
                    Parsed::Header(..) => unreachable!("headers-only parse"),
                    Parsed::Game { header, error: Some(e), .. } => {
                        return Some(Err(anyhow::anyhow!("{}: {e}", header.id())));
                    }
                    Parsed::Game { header, variant, start, mut moves, clocks, error: None } => {
                        let keep = clock_cut(&clocks, moves.len(), filter.min_clock_s);
                        let cut = (keep < moves.len()).then_some(keep);
                        moves.truncate(keep);
                        if moves.len() < filter.min_plies.max(1) {
                            bump(&counts.short);
                            continue;
                        }
                        if cut.is_some() {
                            bump(&counts.clock_cut);
                        }
                        *counts.by_variant.borrow_mut().entry(variant.name()).or_default() += 1;
                        return Some(Ok(record_of(header, variant, start, moves, cut)));
                    }
                }
            }
            None
        }))
    })
}

/// `https://database.lichess.org/<dir>/lichess_db_<dir>_rated_<month>.pgn.zst`
pub fn dump_url(v: ChessVariant, month: &str) -> String {
    let d = v.lichess_dir();
    format!("https://database.lichess.org/{d}/lichess_db_{d}_rated_{month}.pgn.zst")
}

fn dump_name(v: ChessVariant, month: &str) -> String {
    let d = v.lichess_dir();
    format!("lichess_db_{d}_rated_{month}.pgn.zst")
}

/// The first `max_bytes` of a dump, decoded as far as they go and cut after the last
/// complete game (a sample; never a game cut in its movetext).
fn fetch_head(client: &reqwest::blocking::Client, url: &str, max_bytes: u64, dest: &Path) -> Result<usize> {
    let resp = client.get(url).header("Range", format!("bytes=0-{}", max_bytes.saturating_sub(1))).send()?;
    let bytes = resp.error_for_status()?.bytes()?;
    let mut d = zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes.to_vec()))?;
    d.window_log_max(31)?;
    let mut text = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        match d.read(&mut buf) {
            Ok(0) | Err(_) => break, // the prefix ends inside a zstd block
            Ok(n) => text.extend_from_slice(&buf[..n]),
        }
    }
    let cut = text.windows(8).rposition(|w| w == b"\n[Event ").map_or(0, |i| i + 1);
    text.truncate(cut);
    let games = text.windows(7).filter(|w| w == b"[Event ").count();
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = dest.with_extension("part");
    std::fs::File::create(&tmp)?.write_all(&text)?;
    std::fs::rename(&tmp, dest)?;
    Ok(games)
}

fn parse_variants(names: &[String]) -> Result<Vec<ChessVariant>> {
    if names.iter().any(|n| n == "all") {
        return Ok(ChessVariant::ALL.to_vec());
    }
    names.iter().map(|n| ChessVariant::from_name(n).with_context(|| format!("unknown variant {n:?}"))).collect()
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Months with a dump (and their game counts), from database.lichess.org
    List {
        /// variant name, or `all`
        #[arg(long, default_value = "all")]
        variant: String,
    },
    /// Download monthly dumps (kept compressed), or with --max-bytes a sample: the first N
    /// compressed bytes decoded and cut after the last complete game (`.head.pgn`)
    Fetch {
        /// variant names (repeatable), or `all`
        #[arg(long = "variant", required = true)]
        variants: Vec<String>,
        /// YYYY-MM (repeatable)
        #[arg(long = "month", required = true)]
        months: Vec<String>,
        #[arg(long, default_value = "data/chess_variants/raw")]
        out: PathBuf,
        #[arg(long)]
        max_bytes: Option<u64>,
    },
    /// Header statistics per variant (games, rated, terminations, lower-Elo percentiles)
    Stats {
        #[arg(long)]
        input: PathBuf,
    },
    /// Parse, filter, replay-verify and write gm-shards of the gmrules game chess_variants
    Convert {
        /// a .pgn / .pgn.zst / .pgn.gz file or a directory of them
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// variants to keep (repeatable; default all)
        #[arg(long = "variant")]
        variants: Vec<String>,
        #[arg(long, default_value_t = 0)]
        min_elo: i64,
        /// per-variant Elo minimum, VARIANT=N (repeatable)
        #[arg(long)]
        min_elo_variant: Vec<String>,
        /// keep the top P % of each file and variant by the lower Elo
        #[arg(long, default_value_t = 0.0)]
        top_percent: f64,
        #[arg(long)]
        rated_only: bool,
        #[arg(long)]
        normal_termination: bool,
        #[arg(long, default_value_t = 0)]
        min_duration_s: i64,
        #[arg(long, default_value_t = 1)]
        min_plies: usize,
        #[arg(long)]
        no_bots: bool,
        /// cut each game before the first move its player made with less than S seconds
        /// on the clock ([%clk]; the kept prefix takes the PGN result as its value)
        #[arg(long, default_value_t = 0)]
        min_clock_s: i64,
        /// high-quality preset: rated, normal termination, >= 60 s, >= 4 plies and, without
        /// an Elo filter, --top-percent 10
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 512)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep games without a result and games cut short (causality tests)
        #[arg(long)]
        allow_truncated: bool,
    },
}

fn stats(input: &Path) -> Result<Value> {
    let files = common::find_files(input, &[".pgn", ".pgn.zst", ".pgn.gz"])?;
    anyhow::ensure!(!files.is_empty(), "no PGN files under {}", input.display());
    let mut per: BTreeMap<&'static str, (usize, usize, BTreeMap<String, usize>, Vec<i64>, usize)> = BTreeMap::new();
    let everything = Filter::default();
    for path in files {
        let mut r = reader(&path)?;
        let mut p = Parser { filter: &everything, thresholds: None, headers_only: true };
        while let Some(g) = r.read_game(&mut p)? {
            if let Parsed::Header(v, h) = g {
                let e = per.entry(v.name()).or_default();
                e.0 += 1;
                e.1 += h.get("Event").is_some_and(|x| x.starts_with("Rated")) as usize;
                *e.2.entry(h.get("Termination").unwrap_or("?").to_string()).or_default() += 1;
                if let Some(m) = h.min_elo() {
                    e.3.push(m);
                }
                e.4 += h.is_bot_game() as usize;
            }
        }
    }
    let mut out = serde_json::Map::new();
    for (v, (n, rated, term, mut elos, bots)) in per {
        elos.sort_unstable();
        let pct = |p: f64| elos.get(((elos.len() as f64 - 1.0) * p / 100.0).round() as usize).copied();
        out.insert(v.to_string(), json!({
            "games": n, "rated": rated, "bot_games": bots, "terminations": term,
            "lower_elo_percentiles": {"50": pct(50.0), "75": pct(75.0), "90": pct(90.0), "95": pct(95.0), "99": pct(99.0)},
        }));
    }
    Ok(Value::Object(out))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::List { variant } => {
            let client = common::http()?;
            for v in parse_variants(&[variant])? {
                let d = v.lichess_dir();
                let counts = client.get(format!("https://database.lichess.org/{d}/counts.txt")).send()?.error_for_status()?.text()?;
                for line in counts.lines() {
                    let mut it = line.split_whitespace();
                    if let (Some(name), Some(n)) = (it.next(), it.next()) {
                        let month = name.trim_end_matches(".pgn.zst").rsplit('_').next().unwrap_or(name);
                        println!("{} {month} {n}", v.name());
                    }
                }
            }
            Ok(())
        }
        Cmd::Fetch { variants, months, out, max_bytes } => {
            let client = common::http()?;
            for v in parse_variants(&variants)? {
                for m in &months {
                    let url = dump_url(v, m);
                    match max_bytes {
                        Some(n) => {
                            let dest = out.join(dump_name(v, m).replace(".pgn.zst", ".head.pgn"));
                            let games = fetch_head(&client, &url, n, &dest)?;
                            println!("{}", json!({"variant": v.name(), "month": m, "sample": dest, "games": games}));
                        }
                        None => {
                            let dest = out.join(dump_name(v, m));
                            anyhow::ensure!(common::download(&client, &url, &dest)?, "404: {url}");
                            println!("{}", json!({"variant": v.name(), "month": m, "dump": dest}));
                        }
                    }
                }
            }
            Ok(())
        }
        Cmd::Stats { input } => {
            println!("{}", stats(&input)?);
            Ok(())
        }
        Cmd::Convert {
            input, out, variants, min_elo, min_elo_variant, top_percent, rated_only, normal_termination, min_duration_s,
            min_plies, no_bots, min_clock_s, hq, shard_size, max_games, store_events, allow_truncated,
        } => {
            let mut per = HashMap::new();
            for kv in &min_elo_variant {
                let (k, e) = kv.split_once('=').with_context(|| format!("--min-elo-variant {kv:?}: VARIANT=N"))?;
                let v = ChessVariant::from_name(k).with_context(|| format!("unknown variant {k:?}"))?;
                per.insert(v, e.parse::<i64>().with_context(|| format!("--min-elo-variant {kv:?}"))?);
            }
            let mut filter = Filter {
                variants: if variants.is_empty() { Vec::new() } else { parse_variants(&variants)? },
                min_elo,
                min_elo_variant: per,
                top_percent,
                rated_only,
                normal_termination,
                min_duration_s,
                min_plies,
                no_bots,
                min_clock_s,
                keep_unfinished: allow_truncated,
            };
            if hq {
                filter.rated_only = true;
                filter.normal_termination = true;
                filter.min_duration_s = filter.min_duration_s.max(60);
                filter.min_plies = filter.min_plies.max(4);
                if filter.min_elo == 0 && filter.min_elo_variant.is_empty() && filter.top_percent == 0.0 {
                    filter.top_percent = 10.0;
                }
            }
            if !(0.0..=100.0).contains(&filter.top_percent) {
                bail!("--top-percent must be in [0, 100]");
            }
            let files = common::find_files(&input, &[".pgn", ".pgn.zst", ".pgn.gz"])?;
            anyhow::ensure!(!files.is_empty(), "no PGN files under {}", input.display());
            let counts = Counts::default();
            let recs = records(files, &filter, &counts).take(max_games.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, ..Default::default() };
            // one config for every variant: shards of all variants share the shard meta
            let st = traj::convert::<ChessVariants>(&ChessVariantsConfig::default(), recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["parser"] = counts.json();
            println!("{v}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmshard::ShardReader;

    const PGN: &str = r#"[Event "Rated Crazyhouse game"]
[Site "https://lichess.org/aaaa0001"]
[White "w1"]
[Black "b1"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]
[Variant "Crazyhouse"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5 4. P@b4 Qxb4 0-1

[Event "Rated Chess960 game"]
[Site "https://lichess.org/aaaa0002"]
[White "w2"]
[Black "b2"]
[Result "1/2-1/2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "60+0"]
[Termination "Normal"]
[FEN "nbrkbqrn/pppppppp/8/8/8/8/PPPPPPPP/NBRKBQRN w KQkq - 0 1"]
[SetUp "1"]
[Variant "Chess960"]

1. d4 d5 2. Nb3 Nb6 1/2-1/2

[Event "Rated Standard game"]
[Site "https://lichess.org/aaaa0003"]
[White "w3"]
[Black "b3"]
[Result "1-0"]
[Variant "Standard"]

1. e4 e5 1-0

[Event "Rated Atomic game"]
[Site "https://lichess.org/aaaa0004"]
[White "w4"]
[Black "b4"]
[Result "1-0"]
[WhiteElo "1900"]
[BlackElo "1900"]
[TimeControl "60+0"]
[Termination "Normal"]
[Variant "Atomic"]

1. Nf3 d6 2. Ng5 h6 3. Nxf7 1-0

[Event "Rated Atomic game"]
[Site "https://lichess.org/aaaa0005"]
[White "w5"]
[Black "b5"]
[Result "1-0"]
[Termination "Normal"]
[Variant "Atomic"]

1. Nf3 Nf3 1-0

[Event "Rated King of the Hill game"]
[Site "https://lichess.org/aaaa0006"]
[White "w6"]
[Black "b6"]
[Result "*"]
[Variant "King of the Hill"]

1. e4 *

"#;

    #[test]
    fn parses_filters_and_converts_real_format() {
        let dir = std::env::temp_dir().join(format!("gmdata-cv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pgn = dir.join("s.pgn");
        std::fs::write(&pgn, PGN).unwrap();
        let counts = Counts::default();
        let filter = Filter::default();
        let recs: Vec<Result<Record>> = records(vec![pgn.clone()], &filter, &counts).collect();
        // crazyhouse (a drop, resigned), chess960 (agreed draw), standard (skipped), atomic
        // (Nxf7 explodes the black king), atomic with an illegal SAN, king of the hill "*"
        assert_eq!(counts.games.get(), 6);
        assert_eq!((counts.other_variant.get(), counts.no_result.get()), (1, 1));
        assert_eq!(recs.len(), 4);
        assert!(recs[3].is_err(), "Nf3 for black is illegal");
        let zh = recs[0].as_ref().unwrap();
        assert_eq!(zh.events[0], Event::Chance(ChessVariant::Crazyhouse as u32));
        assert_eq!(zh.events.len(), 1 + 8);
        assert_eq!(zh.final_outcome, Some(vec![-1.0, 1.0]));
        let c960 = recs[1].as_ref().unwrap();
        assert_eq!(c960.events[..2], [Event::Chance(0), Event::Chance(cv::scharnagl_of(
            &Fen::from_ascii(b"nbrkbqrn/pppppppp/8/8/8/8/PPPPPPPP/NBRKBQRN w KQkq - 0 1").unwrap().into_setup().board).unwrap())]);
        // the converter: every record replayed; the atomic game ends off the board
        let st = traj::convert::<ChessVariants>(&ChessVariantsConfig::default(),
                                                records(vec![pgn.clone()], &filter, &Counts::default()),
                                                &dir.join("out"), &ConvertOptions { store_events: true, ..Default::default() }).unwrap();
        assert_eq!((st.converted, st.unparseable, st.illegal, st.returns_mismatch, st.adjudicated), (3, 1, 0, 0, 2), "{st:?}");
        let sh = ShardReader::open(&dir.join("out/chess_variants_00000")).unwrap();
        assert_eq!(sh.meta()["schema"], "chessvar.v1");
        assert_eq!(sh.meta()["config"]["variants"].as_array().unwrap().len(), 8);
        assert_eq!(sh.samples[0].meta["variant"], "crazyhouse");
        assert_eq!(sh.samples[0].meta["seat"], -1);
        // the crazyhouse game was resigned (adjudicated): the values are the recorded
        // result from each mover's view, frame by frame
        let values = sh.read_f32("tgt.value").unwrap();
        assert_eq!(values[..4], [-1.0, 1.0, -1.0, 1.0]);
        // the headers-only pass: --top-percent keeps the top of each variant
        let top = Filter { top_percent: 50.0, ..Filter::default() };
        let t = thresholds(&pgn, &top).unwrap();
        assert_eq!(t[&ChessVariant::Crazyhouse], 2000);
        let hq = Filter { rated_only: true, normal_termination: true, min_duration_s: 61, ..Filter::default() };
        let n = records(vec![pgn], &hq, &Counts::default()).filter(|r| r.is_ok()).count();
        assert_eq!(n, 1, "only the crazyhouse game lasts >= 61 s");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn clocks_cut_time_scrambles_and_unfinished_games_are_kept_on_request() {
        assert_eq!(parse_clock(b" [%clk 0:02:03] "), Some(123.0));
        assert_eq!(parse_clock(b"[%eval 0.3] [%clk 1:00:00.5]"), Some(3600.5));
        assert_eq!(parse_clock(b"[%eval 0.3]"), None);
        assert_eq!(parse_clock(b"[%clk 0:0x:03]"), None);
        // plies 0 / 1 never cut (lichess starts the clocks after them); ply k looks at the
        // player's clock after ply k - 2
        let c = [Some(3.0), Some(60.0), Some(9.0), None, Some(4.0), Some(50.0)];
        assert_eq!(clock_cut(&c, 6, 0), 6);
        assert_eq!(clock_cut(&c, 6, 5), 2, "white moved with 3 s left at ply 2");
        assert_eq!(clock_cut(&c[1..], 5, 5), 5, "unknown clocks never cut");
        assert_eq!(clock_cut(&[Some(59.0)], 7, 10), 7);
        let dir = std::env::temp_dir().join(format!("gmdata-cv-clk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pgn = dir.join("c.pgn");
        std::fs::write(&pgn, "[Event \"Rated Atomic game\"]\n[Site \"https://lichess.org/c1\"]\n[Result \"1-0\"]\n\
            [TimeControl \"60+0\"]\n[Termination \"Normal\"]\n[Variant \"Atomic\"]\n\n1. e4 { [%clk 0:01:00] } 1... e5 \
            { [%clk 0:01:00] } 2. Nf3 { [%clk 0:00:04] } 2... Nc6 { [%clk 0:00:50] } 3. Bc4 { [%clk 0:00:02] } 1-0\n\n\
            [Event \"Rated Atomic game\"]\n[Site \"https://lichess.org/c2\"]\n[Result \"*\"]\n[Variant \"Atomic\"]\n\n1. e4 *\n\n")
            .unwrap();
        let (counts, filter) = (Counts::default(), Filter { min_clock_s: 5, ..Filter::default() });
        let recs: Vec<Record> = records(vec![pgn.clone()], &filter, &counts).map(|r| r.unwrap()).collect();
        assert_eq!(recs.len(), 1, "the unfinished game is skipped");
        assert_eq!((recs[0].events.len(), counts.clock_cut.get()), (1 + 4, 1), "Bc4 came with 4 s left");
        assert_eq!(recs[0].meta["clock_cut_ply"], 4);
        let keep = Filter { keep_unfinished: true, ..Filter::default() };
        let recs: Vec<Record> = records(vec![pgn], &keep, &Counts::default()).map(|r| r.unwrap()).collect();
        assert_eq!((recs.len(), recs[1].final_outcome.clone()), (2, None));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The committed sample of real lichess games (tests/fixtures/chess_variants):
    /// every game replays, every rules end agrees with the PGN result.
    #[test]
    fn real_fixture_games_replay_and_agree_with_lichess() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/fixtures/chess_variants/lichess_variants_2026-08_sample.pgn");
        let dir = std::env::temp_dir().join(format!("gmdata-cv-fx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let counts = Counts::default();
        let st = traj::convert::<ChessVariants>(&ChessVariantsConfig::default(),
                                                records(vec![fixture], &Filter::default(), &counts),
                                                &dir, &ConvertOptions::default()).unwrap();
        assert_eq!((st.records, st.converted, st.illegal, st.returns_mismatch, st.unparseable), (49, 49, 0, 0, 0), "{st:?}");
        assert_eq!(counts.by_variant.borrow().len(), 8);
        assert_eq!(st.adjudicated, 27, "resignations, time forfeits, agreed draws");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn start_positions_are_checked() {
        let mut h = Header::default();
        h.tags.insert("FEN", "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1".into());
        assert!(start_of(ChessVariant::Horde, &h).is_ok());
        h.tags.insert("FEN", "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1".into());
        assert!(start_of(ChessVariant::RacingKings, &h).is_ok());
        assert!(start_of(ChessVariant::Horde, &h).is_err());
        h.tags.insert("FEN", "4k3/8/8/8/8/8/8/4K2R w K - 0 1".into());
        assert!(start_of(ChessVariant::Chess960, &h).is_err());
        h.tags.insert("FEN", "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1".into());
        assert_eq!(start_of(ChessVariant::Chess960, &h).unwrap().0, Some(0));
        h.tags.insert("FEN", "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w Kkq - 0 1".into());
        assert!(start_of(ChessVariant::Chess960, &h).is_err(), "castling rights must be complete");
    }
}
