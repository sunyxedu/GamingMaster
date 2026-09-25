//! Go: SGF game records → shards for the "go" adapter (BoardEncoder + PlacementDecoder).
//!
//! Frame t = position BEFORE move t, from the side to move (own / opp / empty /
//! simple-ko planes); label = that move. Every move is replayed with captures,
//! suicide rejection and simple ko; a record with an illegal move is skipped.
//!
//!   obs.board_seq [T,4,N,N] f32
//!   tgt.place_idx [T] i64   row*N+col (0 on pass)
//!   tgt.passed    [T] bool
//!   tgt.value     [T] f32   ±1 from RE for the mover, 0 for draws / unknown
//!
//! SGF scope: main line only (first variation), multi-game collections,
//! AB/AW/AE setup, `[]` and `[tt]` (≤19×19) passes, escaped values.

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

type Point = (i32, i32);

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

// ---------------------------------------------------------------- rules

const EMPTY: u8 = 0;
const BLACK: u8 = 1;
const WHITE: u8 = 2;

fn opponent(c: u8) -> u8 {
    if c == BLACK {
        WHITE
    } else {
        BLACK
    }
}

struct Board {
    n: i32,
    g: Vec<u8>,
    ko: Option<Point>,
}

impl Board {
    fn at(&self, p: Point) -> u8 {
        self.g[(p.0 * self.n + p.1) as usize]
    }
    fn set(&mut self, p: Point, v: u8) {
        let n = self.n;
        self.g[(p.0 * n + p.1) as usize] = v;
    }
    fn neighbors(&self, (r, c): Point) -> Vec<Point> {
        let mut v = Vec::with_capacity(4);
        if r > 0 {
            v.push((r - 1, c));
        }
        if r < self.n - 1 {
            v.push((r + 1, c));
        }
        if c > 0 {
            v.push((r, c - 1));
        }
        if c < self.n - 1 {
            v.push((r, c + 1));
        }
        v
    }
    fn group(&self, p: Point) -> (HashSet<Point>, HashSet<Point>) {
        let color = self.at(p);
        let (mut stones, mut libs) = (HashSet::new(), HashSet::new());
        let mut stack = vec![p];
        while let Some(q) = stack.pop() {
            if !stones.insert(q) {
                continue;
            }
            for nb in self.neighbors(q) {
                let v = self.at(nb);
                if v == EMPTY {
                    libs.insert(nb);
                } else if v == color && !stones.contains(&nb) {
                    stack.push(nb);
                }
            }
        }
        (stones, libs)
    }
    fn setup(&mut self, color: u8, pts: &[Point]) -> bool {
        for &(r, c) in pts {
            if !(0..self.n).contains(&r) || !(0..self.n).contains(&c) || self.at((r, c)) != EMPTY {
                return false;
            }
            self.set((r, c), color);
        }
        true
    }
    /// false = illegal (board unchanged)
    fn play(&mut self, color: u8, p: Option<Point>) -> bool {
        let Some((r, c)) = p else {
            self.ko = None;
            return true;
        };
        if !(0..self.n).contains(&r) || !(0..self.n).contains(&c) || self.at((r, c)) != EMPTY || self.ko == Some((r, c)) {
            return false;
        }
        self.set((r, c), color);
        let opp = opponent(color);
        let mut captured: Vec<Point> = Vec::new();
        let mut seen: HashSet<Point> = HashSet::new();
        for q in self.neighbors((r, c)) {
            if self.at(q) == opp && !seen.contains(&q) {
                let (stones, libs) = self.group(q);
                seen.extend(stones.iter().copied());
                if libs.is_empty() {
                    for s in &stones {
                        self.set(*s, EMPTY);
                    }
                    captured.extend(stones);
                }
            }
        }
        let (own, own_libs) = self.group((r, c));
        if own_libs.is_empty() {
            self.set((r, c), EMPTY);
            return false;
        }
        self.ko = if captured.len() == 1 && own.len() == 1 && own_libs.len() == 1 && own_libs.contains(&captured[0]) {
            Some(captured[0])
        } else {
            None
        };
        true
    }
}

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
    let mut b = Board { n: size, g: vec![EMPTY; (size * size) as usize], ko: None };
    if !b.setup(BLACK, &g.setup_black) || !b.setup(WHITE, &g.setup_white) {
        return Err("illegal_move");
    }
    let t = g.moves.len();
    let nn = (size * size) as usize;
    let mut planes = vec![0f32; t * 4 * nn];
    let (mut idx, mut passed, mut value) = (vec![0i64; t], vec![false; t], vec![0f32; t]);
    for (k, (ch, p)) in g.moves.iter().enumerate() {
        let color = if *ch == 'B' { BLACK } else { WHITE };
        let base = k * 4 * nn;
        for (i, &v) in b.g.iter().enumerate() {
            planes[base + i] = (v == color) as u8 as f32;
            planes[base + nn + i] = (v == opponent(color)) as u8 as f32;
            planes[base + 2 * nn + i] = (v == EMPTY) as u8 as f32;
        }
        if let Some((kr, kc)) = b.ko {
            planes[base + 3 * nn + (kr * size + kc) as usize] = 1.0;
        }
        match p {
            None => passed[k] = true,
            Some((r, c)) => idx[k] = (r * size + c) as i64,
        }
        if result != 0 {
            value[k] = if (result > 0) == (color == BLACK) { 1.0 } else { -1.0 };
        }
        if !b.play(color, *p) {
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
        .f32("obs.board_seq", &[4, s, s], planes)
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
               allow_handicap: bool, allow_no_result: bool) -> Result<Value> {
    let files = common::find_files(input, &[".sgf"])?;
    anyhow::ensure!(!files.is_empty(), "no .sgf files under {:?}", input);
    let mut w = ShardWriter::new(out, "go", 1, json!({}), shard_size)?;
    let mut stats: BTreeMap<&str, usize> = BTreeMap::new();
    for k in ["converted", "unparseable_files", "wrong_board_size", "handicap", "no_result", "too_short", "illegal_move"] {
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
        Cmd::Convert { input, out, shard_size, board_size, max_games, min_moves, allow_handicap, allow_no_result } => {
            println!("{}", convert(&input, &out, shard_size, board_size, max_games, min_moves, allow_handicap, allow_no_result)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ko_and_capture() {
        let mut b = Board { n: 5, g: vec![EMPTY; 25], ko: None };
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
    }
}
