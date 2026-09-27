//! Go: SGF game records → shards for the "go" adapter (BoardEncoder + PlacementDecoder).
//!
//! Frame t = position BEFORE move t, from the side to move (own / opp / empty /
//! simple-ko planes); label = that move. Every move is replayed with captures,
//! suicide rejection and simple ko; a record with an illegal move is skipped.
//! One sample = one game = both colours' frames interleaved in play order.
//!
//! The rules and the frame encoders are gmrules' (`gmrules::games::go`, the go
//! simulator): each record becomes that game's state ([`sgf_config`]: board size,
//! komi, rules flag, HA, AB / AW stones), every frame is its encoder's output for the
//! record's mover and every move its transition — so a gmsim frame of the same
//! position is this frame, bit for bit (the `sim_parity` tests). This module keeps the
//! SGF parsing, the record filters, the targets and the shard layout.
//!
//!   obs.board_seq   [T,4,N,N] f32
//!   obs.globals_seq [T,6] f32     0 mover is black
//!                                 1 komi from the mover's view / 20, clipped to [-1,1]
//!                                   (+KM when white moves, -KM when black moves —
//!                                   KataGo's convention; 0 when KM is missing/unparseable)
//!                                 2 komi known (KM present, parseable, finite)
//!                                 3 area scoring (RU chinese / aga / nz / new zealand /
//!                                   area, case-insensitive; japanese / korean / other /
//!                                   missing → 0)
//!                                 4 min(HA, 9) / 9
//!                                 5 the previous move (either colour) was a pass
//!   tgt.place_idx   [T] i64   row*N+col (0 on pass)
//!   tgt.passed      [T] bool
//!   tgt.value       [T] f32   ±1 from RE for the mover, 0 for draws / unknown
//!
//! Shard meta: {global_features: 6, board_size: N}. Converter versions: 1 = planes
//! only; 2 = + obs.globals_seq (planes and targets unchanged).
//!
//! SGF scope: main line only (first variation), multi-game collections,
//! AB/AW/AE setup, `[]` and `[tt]` (≤19×19) passes, escaped values.

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmrules::games::go::{Colour, Go, GoConfig, Point, PLANES};
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The rules and encoders (gmrules), also for the labeller (`label/go.rs`).
pub(crate) use gmrules::games::go::{encode_globals, encode_planes, globals_consts, opponent, Board, BLACK, EMPTY, GLOBAL_FEATURES,
                                    WHITE};

#[cfg(test)]
mod sim_parity;

/// 2: + obs.globals_seq (side to move, komi, ruleset, handicap, last pass).
pub const CONVERTER_VERSION: u32 = 2;

// ---------------------------------------------------------------- SGF

#[derive(Default, Debug)]
pub struct SgfGame {
    pub props: BTreeMap<String, Vec<String>>,
    pub moves: Vec<(char, Option<Point>)>,
    pub setup_black: Vec<Point>,
    pub setup_white: Vec<Point>,
    pub setup_after_move: bool,
}

impl SgfGame {
    fn first(&self, k: &str) -> Option<&str> {
        self.props.get(k).and_then(|v| v.first()).map(|s| s.as_str())
    }
    /// Err = malformed SZ
    pub fn board_size(&self) -> std::result::Result<i32, ()> {
        match self.first("SZ") {
            None => Ok(19),
            Some(raw) => raw.split(':').next().unwrap_or("").trim().parse::<i32>().map_err(|_| ()),
        }
    }
    pub fn handicap(&self) -> i64 {
        self.first("HA").and_then(|r| r.trim().parse::<i64>().ok()).unwrap_or(0)
    }
    pub fn komi(&self) -> Option<f64> {
        self.first("KM").filter(|r| !r.is_empty()).and_then(|r| py_float(r))
    }
    /// RU names an area-scoring ruleset (Chinese / AGA / New Zealand);
    /// territory rulesets (Japanese, Korean) and unknown / missing → false.
    pub fn area_scoring(&self) -> bool {
        self.first("RU")
            .map(|r| matches!(r.trim().to_lowercase().as_str(), "chinese" | "aga" | "nz" | "new zealand" | "area"))
            .unwrap_or(false)
    }
}

/// The gmrules go config of a record: the game the simulator plays from the record's
/// start — its board size, komi (a non-finite one = unknown), the RU scoring flag, HA,
/// the AB / AW stones (point indices), the first mover and, for a free handicap
/// placement (HA > 0, no AB, the first HA moves black's: what [`game_to_sample`]
/// accepts with `allow_handicap`), the HA placement moves; the simulator's default end
/// rules. The converter builds its frames on this config's state (with `superko` off:
/// no frame depends on how the game ends).
pub fn sgf_config(g: &SgfGame, size: i32) -> GoConfig {
    let pt = |&(r, c): &Point| (r * size + c) as usize;
    let ha = g.handicap();
    let free = ha > 0 && g.setup_black.is_empty() && g.moves.len() as i64 >= ha && g.moves[..ha as usize].iter().all(|(c, _)| *c == 'B');
    GoConfig {
        board_size: size as usize,
        komi: g.komi().filter(|k| k.is_finite()),
        area_scoring: g.area_scoring(),
        handicap: ha,
        setup_black: g.setup_black.iter().map(pt).collect(),
        setup_white: g.setup_white.iter().map(pt).collect(),
        first_player: if !free && matches!(g.moves.first(), Some(('W', _))) { Colour::White } else { Colour::Black },
        free_handicap: if free { ha as usize } else { 0 },
        ..GoConfig::default()
    }
}

/// Python `float()` on a string (surrounding whitespace allowed).
fn py_float(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

enum Tok {
    Open,
    Close,
    Semi,
    Ident(String),
    Value(String),
}

fn tokenize(text: &str) -> std::result::Result<Vec<Tok>, ()> {
    let ch: Vec<char> = text.chars().collect();
    let (mut i, n) = (0usize, ch.len());
    let mut out = Vec::new();
    while i < n {
        let c = ch[i];
        match c {
            '(' => {
                out.push(Tok::Open);
                i += 1;
            }
            ')' => {
                out.push(Tok::Close);
                i += 1;
            }
            ';' => {
                out.push(Tok::Semi);
                i += 1;
            }
            '[' => {
                let mut j = i + 1;
                let mut buf = String::new();
                while j < n && ch[j] != ']' {
                    if ch[j] == '\\' && j + 1 < n {
                        j += 1;
                    }
                    buf.push(ch[j]);
                    j += 1;
                }
                if j >= n {
                    return Err(());
                }
                out.push(Tok::Value(buf));
                i = j + 1;
            }
            c if c.is_alphabetic() => {
                let mut j = i;
                while j < n && ch[j].is_alphabetic() {
                    j += 1;
                }
                let ident: String = ch[i..j].iter().filter(|c| c.is_uppercase()).collect();
                if !ident.is_empty() {
                    out.push(Tok::Ident(ident));
                }
                i = j;
            }
            _ => i += 1,
        }
    }
    Ok(out)
}

fn parse_point(raw: &str, size: i32) -> std::result::Result<Option<Point>, ()> {
    let v = raw.trim();
    if v.is_empty() || (v == "tt" && size <= 19) {
        return Ok(None);
    }
    let b: Vec<char> = v.chars().collect();
    if b.len() != 2 || !('a'..='z').contains(&b[0]) || !('a'..='z').contains(&b[1]) {
        return Err(());
    }
    let (col, row) = (b[0] as i32 - 97, b[1] as i32 - 97);
    if !(0..size).contains(&row) || !(0..size).contains(&col) {
        return Err(());
    }
    Ok(Some((row, col)))
}

fn finish(events: &[(String, Vec<String>)]) -> std::result::Result<SgfGame, ()> {
    let mut size = 19;
    for (ident, vals) in events {
        if ident == "SZ" && !vals.is_empty() {
            size = vals[0].split(':').next().unwrap_or("").trim().parse::<i32>().map_err(|_| ())?;
            break;
        }
    }
    let mut g = SgfGame::default();
    let mut pending_ae: Vec<Point> = Vec::new();
    fn apply_ae(g: &mut SgfGame, pending: &mut Vec<Point>) {
        for p in pending.drain(..) {
            if let Some(i) = g.setup_black.iter().position(|&x| x == p) {
                g.setup_black.remove(i);
            } else if let Some(i) = g.setup_white.iter().position(|&x| x == p) {
                g.setup_white.remove(i);
            }
        }
    }
    for (ident, vals) in events {
        match ident.as_str() {
            ";" => apply_ae(&mut g, &mut pending_ae),
            "B" | "W" => {
                apply_ae(&mut g, &mut pending_ae);
                let raw = vals.first().map(|s| s.as_str()).unwrap_or("");
                g.moves.push((ident.chars().next().unwrap(), parse_point(raw, size)?));
            }
            "AB" | "AW" => {
                if !g.moves.is_empty() {
                    g.setup_after_move = true;
                }
                for v in vals {
                    if let Some(p) = parse_point(v, size)? {
                        if ident == "AB" {
                            g.setup_black.push(p)
                        } else {
                            g.setup_white.push(p)
                        }
                    }
                }
            }
            "AE" => {
                if !g.moves.is_empty() {
                    g.setup_after_move = true;
                } else {
                    for v in vals {
                        if let Some(p) = parse_point(v, size)? {
                            pending_ae.push(p);
                        }
                    }
                }
            }
            _ => g.props.entry(ident.clone()).or_default().extend(vals.iter().cloned()),
        }
    }
    apply_ae(&mut g, &mut pending_ae);
    Ok(g)
}

/// Err = unterminated value (whole file unparseable)
pub fn parse_sgf(text: &str) -> std::result::Result<Vec<SgfGame>, ()> {
    let mut games = Vec::new();
    let mut events: Option<Vec<(String, Vec<String>)>> = None;
    let mut collecting = false;
    let mut depth = 0i32;
    for tok in tokenize(text)? {
        match tok {
            Tok::Open => {
                depth += 1;
                if depth == 1 {
                    events = Some(Vec::new());
                    collecting = true;
                }
            }
            Tok::Close => {
                if depth == 0 {
                    continue;
                }
                collecting = false;
                depth -= 1;
                if depth == 0 {
                    if let Some(ev) = events.take() {
                        if let Ok(g) = finish(&ev) {
                            games.push(g);
                        }
                    }
                }
            }
            _ if !collecting => {}
            Tok::Semi => events.as_mut().unwrap().push((";".into(), Vec::new())),
            Tok::Ident(s) => events.as_mut().unwrap().push((s, Vec::new())),
            Tok::Value(v) => {
                if let Some(last) = events.as_mut().unwrap().last_mut() {
                    last.1.push(v);
                }
            }
        }
    }
    Ok(games)
}

// ---------------------------------------------------------------- records

fn parse_result(re: Option<&str>) -> Option<i32> {
    let s = re?.trim().to_uppercase();
    if s.is_empty() {
        return None;
    }
    if s.starts_with("B+") && s.chars().count() > 2 {
        return Some(1);
    }
    if s.starts_with("W+") && s.chars().count() > 2 {
        return Some(-1);
    }
    if s == "0" || s == "DRAW" || s == "JIGO" {
        return Some(0);
    }
    None
}

/// Rank → comparable strength: 30k..1k → -30..-1, 1d..9d → 1..9, 1p..9p → 11..19.
pub fn rank_value(r: &str) -> Option<i64> {
    let t = r.trim().trim_end_matches('?').trim().to_lowercase();
    let (num, kind) = t.split_at(t.find(|c: char| !c.is_ascii_digit())?);
    let n: i64 = num.parse().ok()?;
    match kind.trim() {
        "k" | "kyu" => Some(-n),
        "d" | "dan" => Some(n),
        "p" | "pro" => Some(10 + n),
        _ => None,
    }
}

/// Quality filters (off by default; `--hq` = both ≥ 6d, no time/forfeit results, ≥ 50 moves).
#[derive(Clone, Copy, Default)]
pub struct Quality {
    pub min_rank: Option<i64>,
    pub no_time_loss: bool,
    pub no_forfeit: bool,
}

fn passes_quality(g: &SgfGame, q: &Quality) -> bool {
    if let Some(min) = q.min_rank {
        let ok = |k: &str| g.first(k).and_then(rank_value).map(|v| v >= min).unwrap_or(false);
        if !ok("BR") || !ok("WR") {
            return false;
        }
    }
    let re = g.first("RE").unwrap_or("").to_uppercase();
    if q.no_time_loss && (re.ends_with("+T") || re.contains("+TIME")) {
        return false;
    }
    if q.no_forfeit && (re.ends_with("+F") || re.contains("+FORFEIT")) {
        return false;
    }
    true
}

fn opt_str(v: Option<&str>) -> Value {
    v.map(Value::from).unwrap_or(Value::Null)
}

fn game_to_sample(g: &SgfGame, size: i32, allow_handicap: bool, allow_no_result: bool, min_moves: usize, source: &str)
    -> std::result::Result<Sample, &'static str> {
    match g.board_size() {
        Ok(s) if s == size => {}
        _ => return Err("wrong_board_size"),
    }
    if g.setup_after_move {
        return Err("handicap");
    }
    let ha = g.handicap();
    if (ha > 0 || !g.setup_black.is_empty() || !g.setup_white.is_empty()) && !allow_handicap {
        return Err("handicap");
    }
    if allow_handicap && ha > 0 && g.setup_black.len() as i64 != ha {
        let free = g.setup_black.is_empty() && g.moves.len() as i64 >= ha && g.moves[..ha as usize].iter().all(|(c, _)| *c == 'B');
        if !free {
            return Err("handicap");
        }
    }
    if g.moves.len() < min_moves.max(1) {
        return Err("too_short");
    }
    let result = match parse_result(g.first("RE")) {
        Some(r) => r,
        None if allow_no_result => 0,
        None => return Err("no_result"),
    };
    // the record's game in the simulator's rules (superko off: no frame depends on how
    // the game ends, and the converter plays on past any end the SGF goes beyond)
    let Some(mut go) = Go::setup(&GoConfig { superko: false, ..sgf_config(g, size) }) else {
        return Err("illegal_move");
    };
    let t = g.moves.len();
    let nn = (size * size) as usize;
    let mut planes = vec![0f32; t * PLANES * nn];
    let mut globals = vec![0f32; t * GLOBAL_FEATURES];
    let (mut idx, mut passed, mut value) = (vec![0i64; t], vec![false; t], vec![0f32; t]);
    for (k, (ch, p)) in g.moves.iter().enumerate() {
        let color = if *ch == 'B' { BLACK } else { WHITE };
        go.encode_obs_as(color, &mut planes[k * PLANES * nn..(k + 1) * PLANES * nn],
                         &mut globals[k * GLOBAL_FEATURES..(k + 1) * GLOBAL_FEATURES]);
        match p {
            None => passed[k] = true,
            Some((r, c)) => idx[k] = (r * size + c) as i64,
        }
        if result != 0 {
            value[k] = if (result > 0) == (color == BLACK) { 1.0 } else { -1.0 };
        }
        if !go.play_as(color, *p) {
            return Err("illegal_move");
        }
    }
    let meta = json!({
        "source": source,
        "players": {"black": opt_str(g.first("PB")), "white": opt_str(g.first("PW"))},
        "ranks": {"black": opt_str(g.first("BR")), "white": opt_str(g.first("WR"))},
        "result": opt_str(g.first("RE")),
        "komi": g.komi(), "handicap": ha, "board_size": size, "n_moves": t,
    });
    let s = size as usize;
    Ok(Sample::new(meta)
        .bits("obs.board_seq", &[4, s, s], &planes)
        .f32("obs.globals_seq", &[GLOBAL_FEATURES], globals)
        .i64("tgt.place_idx", &[], idx)
        .bool("tgt.passed", &[], passed)
        .f32("tgt.value", &[], value))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download a KGS archive (u-go.net) and extract its .sgf files
    Fetch {
        /// archive URL (.tar.bz2), e.g. https://dl.u-go.net/gamerecords/KGS-2019_04-19-1255-.tar.bz2
        #[arg(long)]
        url: String,
        #[arg(long, default_value = "data/go/raw")]
        out: PathBuf,
    },
    /// Convert .sgf files (recursive) to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        #[arg(long, default_value_t = 19)]
        board_size: i32,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 30)]
        min_moves: usize,
        #[arg(long)]
        allow_handicap: bool,
        #[arg(long)]
        allow_no_result: bool,
        /// both players at least this rank (e.g. 6d, 1p)
        #[arg(long)]
        min_rank: Option<String>,
        #[arg(long)]
        no_time_loss: bool,
        #[arg(long)]
        no_forfeit: bool,
        /// high-quality preset: both >= 6d, no time-loss / forfeit results, >= 50 moves, no handicap
        #[arg(long)]
        hq: bool,
    },
}

pub fn fetch(url: &str, out: &Path) -> Result<()> {
    let client = common::http()?;
    let name = url.rsplit('/').next().unwrap_or("archive.tar.bz2");
    let archive = out.join(name);
    anyhow::ensure!(common::download(&client, url, &archive)?, "404: {url}");
    let dec = bzip2::read::BzDecoder::new(std::fs::File::open(&archive)?);
    let mut tar = tar::Archive::new(dec);
    let mut n = 0;
    for e in tar.entries()? {
        let mut e = e?;
        let path = e.path()?.to_path_buf();
        if path.extension().map(|x| x.eq_ignore_ascii_case("sgf")).unwrap_or(false) {
            let dest = out.join(path.components().filter(|c| matches!(c, std::path::Component::Normal(_))).collect::<PathBuf>());
            std::fs::create_dir_all(dest.parent().unwrap())?;
            e.unpack(&dest)?;
            n += 1;
        }
    }
    std::fs::remove_file(&archive)?;
    eprintln!("extracted {n} .sgf files into {}", out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn convert(input: &Path, out: &Path, shard_size: usize, board_size: i32, max_games: Option<usize>, min_moves: usize,
               allow_handicap: bool, allow_no_result: bool, q: Quality) -> Result<Value> {
    let files = common::find_files(input, &[".sgf"])?;
    anyhow::ensure!(!files.is_empty(), "no .sgf files under {:?}", input);
    let mut w = ShardWriter::new(out, "go", CONVERTER_VERSION,
                                 json!({"global_features": GLOBAL_FEATURES, "board_size": board_size}), shard_size)?;
    let mut stats: BTreeMap<&str, usize> = BTreeMap::new();
    for k in ["converted", "unparseable_files", "wrong_board_size", "handicap", "no_result", "too_short", "illegal_move", "low_quality"] {
        stats.insert(k, 0);
    }
    'files: for path in files {
        let text = String::from_utf8_lossy(&std::fs::read(&path)?).to_string();
        let Ok(games) = parse_sgf(&text) else {
            *stats.get_mut("unparseable_files").unwrap() += 1;
            continue;
        };
        let p = path.to_string_lossy().to_string();
        for (gi, g) in games.iter().enumerate() {
            let source = if games.len() == 1 { p.clone() } else { format!("{p}#{gi}") };
            if !passes_quality(g, &q) {
                *stats.get_mut("low_quality").unwrap() += 1;
                continue;
            }
            match game_to_sample(g, board_size, allow_handicap, allow_no_result, min_moves, &source) {
                Ok(s) => {
                    w.push(s)?;
                    *stats.get_mut("converted").unwrap() += 1;
                    if max_games.map(|m| stats["converted"] >= m).unwrap_or(false) {
                        break 'files;
                    }
                }
                Err(r) => *stats.get_mut(r).unwrap() += 1,
            }
        }
    }
    w.flush()?;
    Ok(json!(stats))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { url, out } => fetch(&url, &out),
        Cmd::Convert { input, out, shard_size, board_size, max_games, min_moves, allow_handicap, allow_no_result, min_rank, no_time_loss, no_forfeit, hq } => {
            let mut q = Quality { min_rank: match &min_rank {
                Some(r) => Some(rank_value(r).ok_or_else(|| anyhow::anyhow!("bad rank {r:?} (use e.g. 6d, 3k, 1p)"))?),
                None => None,
            }, no_time_loss, no_forfeit };
            let mut min_moves = min_moves;
            let (mut allow_handicap, mut allow_no_result) = (allow_handicap, allow_no_result);
            if hq {
                q = Quality { min_rank: Some(q.min_rank.unwrap_or(6).max(6)), no_time_loss: true, no_forfeit: true };
                min_moves = min_moves.max(50);
                allow_handicap = false;
                allow_no_result = false;
            }
            println!("{}", convert(&input, &out, shard_size, board_size, max_games, min_moves, allow_handicap, allow_no_result, q)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ko_and_capture() {
        let mut b = Board::new(5);
        // classic ko shape
        for (c, p) in [(BLACK, (0, 1)), (BLACK, (1, 0)), (BLACK, (2, 1)), (WHITE, (0, 2)), (WHITE, (1, 3)), (WHITE, (2, 2)), (WHITE, (1, 1))] {
            assert!(b.play(c, Some(p)));
        }
        assert!(b.play(BLACK, Some((1, 2)))); // captures (1,1)
        assert_eq!(b.at((1, 1)), EMPTY);
        assert_eq!(b.ko, Some((1, 1)));
        assert!(!b.play(WHITE, Some((1, 1)))); // ko
        assert_eq!(parse_result(Some("B+R")), Some(1));
        assert_eq!(parse_result(Some("Void")), None);
        assert_eq!(rank_value("7d"), Some(7));
        assert_eq!(rank_value("3k"), Some(-3));
        assert_eq!(rank_value("1p"), Some(11));
        assert_eq!(rank_value("6d?"), Some(6));
        assert_eq!(rank_value("-"), None);
    }

    fn globals_of(sgf: &str) -> Vec<[f32; GLOBAL_FEATURES]> {
        let games = parse_sgf(sgf).unwrap();
        let s = game_to_sample(&games[0], 9, true, true, 1, "t").unwrap();
        let t = s.tensors.iter().find(|t| t.name == "obs.globals_seq").unwrap();
        assert_eq!(t.frame_shape, vec![GLOBAL_FEATURES]);
        let gmshard::Column::F32(v) = &t.col else { panic!("globals_seq must be f32") };
        v.chunks(GLOBAL_FEATURES).map(|c| c.try_into().unwrap()).collect()
    }

    #[test]
    fn globals_side_komi_rules_pass() {
        // B, W, B pass, W, B, W pass, W (a W move after W's own pass: colours come from
        // the SGF, not from alternation)
        let g = globals_of("(;GM[1]SZ[9]KM[6.5]RU[Chinese]HA[0]RE[W+R];B[cc];W[gg];B[];W[cg];B[gc];W[tt];W[ee])");
        let k = 6.5f32 / 20.0;
        let colour = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let prev_pass = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(g.len(), 7);
        for (i, f) in g.iter().enumerate() {
            assert_eq!(f[0], colour[i], "frame {i} mover");
            assert_eq!(f[1], if colour[i] == 1.0 { -k } else { k }, "frame {i} komi sign");
            assert_eq!((f[2], f[3], f[4]), (1.0, 1.0, 0.0), "frame {i}");
            assert_eq!(f[5], prev_pass[i], "frame {i} previous pass");
        }
        // missing komi, territory rules, handicap (clipped at 9 stones), huge komi clipped
        let g = globals_of("(;SZ[9]RU[Japanese]HA[2]AB[cc][gg];W[ee];B[cg])");
        assert_eq!(g[0], [0.0, 0.0, 0.0, 0.0, 2.0 / 9.0, 0.0]);
        assert_eq!(g[1], [1.0, 0.0, 0.0, 0.0, 2.0 / 9.0, 0.0]);
        let g = globals_of("(;SZ[9]KM[50]RU[ nz ]HA[12]AB[aa][ba][ca][da][ea][fa][ga][ha][ia][ab][bb][cb];W[ee];B[cg])");
        assert_eq!(g[0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(g[1], [1.0, -1.0, 1.0, 1.0, 1.0, 0.0]);
        let g = globals_of("(;SZ[9]KM[?]RU[AGA];B[ee];W[cc])");
        assert_eq!(g[1], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = globals_of("(;SZ[9]KM[-3.5]RU[korean];B[ee];W[cc])");
        assert_eq!((g[0][1], g[1][1], g[0][3]), (3.5 / 20.0, -3.5 / 20.0, 0.0));
    }
}
