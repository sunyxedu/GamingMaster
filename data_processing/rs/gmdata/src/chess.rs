//! Chess: Lichess PGN (`.pgn` / `.pgn.zst`, streamed) → shards for the chess adapter.
//!
//! One sample per game; frame t = position BEFORE move t from the side to move
//! (ranks mirrored for black, files never mirrored — the Maia convention).
//!
//!   obs.board_seq   [T,14,8,8] f32  own P N B R Q K, opp P N B R Q K, ep square, ones
//!   obs.globals_seq [T,8] f32       own K/Q castling, opp K/Q castling, white to move,
//!                                   fullmove/100, halfmove/100 (clipped), in check
//!   tgt.from_idx / tgt.to_idx [T] i64   POV square row*8+col
//!   tgt.promotion   [T] i64         0 none, 1 N, 2 B, 3 R, 4 Q
//!   tgt.value       [T] f32         result for the mover
//!
//! POV mapping: white to move row = 7 − rank; black to move row = rank; col = file.
//! Filters: standard variant, no FEN/SetUp, decisive/drawn result, both Elo ≥ min
//! (untagged skipped when min > 0), ≥ min plies, optional no-bullet
//! (base + 40·inc < 180 s). Games with an unparsable/illegal/null move are skipped.
//! converter_version 2 (rank-only mirror).

use anyhow::{Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use serde_json::{json, Value};
use shakmaty::{san::San, CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};
use std::collections::HashMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

pub const CONVERTER_VERSION: u32 = 2;

struct Game {
    headers: HashMap<String, String>,
    moves: Vec<Move>,
    bad: bool,
}

struct Collector;

impl Visitor for Collector {
    type Tags = HashMap<String, String>;
    type Movetext = (HashMap<String, String>, Chess, Vec<Move>, bool);
    type Output = Game;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(HashMap::new())
    }

    fn tag(&mut self, tags: &mut Self::Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        tags.insert(String::from_utf8_lossy(name).to_string(), value.decode_utf8_lossy().to_string());
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        ControlFlow::Continue((tags, Chess::default(), Vec::new(), false))
    }

    fn san(&mut self, mt: &mut Self::Movetext, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        if mt.3 {
            return ControlFlow::Continue(());
        }
        if matches!(san_plus.san, San::Null) {
            mt.3 = true;
            return ControlFlow::Continue(());
        }
        match san_plus.san.to_move(&mt.1) {
            Ok(m) => {
                mt.1.play_unchecked(m.clone());
                mt.2.push(m);
            }
            Err(_) => mt.3 = true,
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, mt: Self::Movetext) -> Self::Output {
        Game { headers: mt.0, moves: mt.2, bad: mt.3 }
    }
}

fn result_white(r: &str) -> Option<f32> {
    match r {
        "1-0" => Some(1.0),
        "0-1" => Some(-1.0),
        "1/2-1/2" => Some(0.0),
        _ => None,
    }
}

/// Python `int(str)`: optional sign, surrounding whitespace, digits (underscores allowed between digits).
fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let digits: String = body.chars().filter(|&c| c != '_').collect();
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let v: i64 = digits.parse().ok()?;
    Some(if neg { -v } else { v })
}

fn elo(h: &HashMap<String, String>, k: &str) -> Option<i64> {
    h.get(k).and_then(|v| py_int(v))
}

fn is_bullet(h: &HashMap<String, String>) -> bool {
    let tc = h.get("TimeControl").map(|s| s.as_str()).unwrap_or("-");
    let Some((b, i)) = tc.split_once('+') else { return false };
    match (py_int(b), py_int(i)) {
        (Some(b), Some(i)) => b + 40 * i < 180,
        _ => false,
    }
}

fn passes(g: &Game, min_elo: i64, min_plies: usize, skip_bullet: bool) -> bool {
    let h = &g.headers;
    if h.get("Variant").map(|s| s.as_str()).unwrap_or("Standard") != "Standard" {
        return false;
    }
    if h.contains_key("FEN") || h.get("SetUp").map(|s| s.as_str()) == Some("1") {
        return false;
    }
    if result_white(h.get("Result").map(|s| s.as_str()).unwrap_or("")).is_none() {
        return false;
    }
    if min_elo > 0 {
        match (elo(h, "WhiteElo"), elo(h, "BlackElo")) {
            (Some(w), Some(b)) if w >= min_elo && b >= min_elo => {}
            _ => return false,
        }
    }
    if skip_bullet && is_bullet(h) {
        return false;
    }
    g.moves.len() >= min_plies.max(1)
}

fn pov(sq: Square, flip: bool) -> i64 {
    let (file, rank) = (u32::from(sq.file()) as i64, u32::from(sq.rank()) as i64);
    if flip {
        rank * 8 + file
    } else {
        (7 - rank) * 8 + file
    }
}

fn role_idx(r: Role) -> usize {
    match r {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

fn promo_idx(r: Option<Role>) -> i64 {
    match r {
        None => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        Some(_) => 0,
    }
}

fn encode(g: &Game) -> Option<Sample> {
    let wv = result_white(&g.headers["Result"])?;
    let t = g.moves.len();
    let mut planes = vec![0f32; t * 14 * 64];
    let mut globals = Vec::with_capacity(t * 8);
    let (mut from, mut to, mut promo, mut value) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut pos = Chess::default();
    for (k, m) in g.moves.iter().enumerate() {
        if !pos.is_legal(m.clone()) {
            return None;
        }
        let own = pos.turn();
        let flip = own == Color::Black;
        let base = k * 14 * 64;
        for (sq, piece) in pos.board().iter() {
            let plane = if piece.color == own { 0 } else { 6 } + role_idx(piece.role);
            planes[base + plane * 64 + pov(sq, flip) as usize] = 1.0;
        }
        if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
            planes[base + 12 * 64 + pov(ep, flip) as usize] = 1.0;
        }
        for i in 0..64 {
            planes[base + 13 * 64 + i] = 1.0;
        }
        let c = pos.castles();
        globals.extend_from_slice(&[
            c.has(own, CastlingSide::KingSide) as u8 as f32,
            c.has(own, CastlingSide::QueenSide) as u8 as f32,
            c.has(!own, CastlingSide::KingSide) as u8 as f32,
            c.has(!own, CastlingSide::QueenSide) as u8 as f32,
            (own == Color::White) as u8 as f32,
            (u32::from(pos.fullmoves()) as f64 / 100.0).min(1.0) as f32,
            (pos.halfmoves() as f64 / 100.0).min(1.0) as f32,
            pos.is_check() as u8 as f32,
        ]);
        let from_sq = m.from()?;
        from.push(pov(from_sq, flip));
        // python-chess encodes castling as king e1→g1 (standard mode), shakmaty as king→rook
        let to_sq = match m {
            Move::Castle { king, rook } => {
                let side = if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide };
                Square::from_coords(side.king_to_file(), king.rank())
            }
            _ => m.to(),
        };
        to.push(pov(to_sq, flip));
        promo.push(promo_idx(m.promotion()));
        value.push(if own == Color::White { wv } else { -wv });
        pos.play_unchecked(m.clone());
    }
    let h = &g.headers;
    let e = |k: &str| elo(h, k).map(Value::from).unwrap_or(Value::Null);
    let meta = json!({
        "white": h.get("White").cloned().unwrap_or_else(|| "?".into()),
        "black": h.get("Black").cloned().unwrap_or_else(|| "?".into()),
        "elo": [e("WhiteElo"), e("BlackElo")],
        "result": h.get("Result").cloned(),
    });
    Some(Sample::new(meta)
        .bits("obs.board_seq", &[14, 8, 8], &planes)
        .f32("obs.globals_seq", &[8], globals)
        .i64("tgt.from_idx", &[], from)
        .i64("tgt.to_idx", &[], to)
        .i64("tgt.promotion", &[], promo)
        .f32("tgt.value", &[], value))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download a monthly Lichess standard-rated dump (YYYY-MM), kept compressed
    Fetch {
        #[arg(long)]
        month: String,
        #[arg(long, default_value = "data/chess/raw")]
        out: PathBuf,
    },
    /// Stream a .pgn / .pgn.zst file into gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        #[arg(long, default_value_t = 0)]
        min_elo: i64,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 10)]
        min_plies: usize,
        #[arg(long)]
        skip_bullet: bool,
    },
}

fn open(path: &Path) -> Result<Box<dyn Read>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    if path.extension().map(|e| e == "zst").unwrap_or(false) {
        let mut d = zstd::stream::read::Decoder::new(f)?;
        d.window_log_max(31)?;
        Ok(Box::new(d))
    } else {
        Ok(Box::new(f))
    }
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, min_elo: i64, max_games: Option<usize>, min_plies: usize,
               skip_bullet: bool) -> Result<Value> {
    let mut w = ShardWriter::new(out, "chess", CONVERTER_VERSION, json!({}), shard_size)?;
    let mut reader = Reader::new(std::io::BufReader::with_capacity(1 << 20, open(input)?));
    let (mut seen, mut kept) = (0usize, 0usize);
    while let Some(g) = reader.read_game(&mut Collector)? {
        if g.bad {
            continue;
        }
        seen += 1;
        if !passes(&g, min_elo, min_plies, skip_bullet) {
            continue;
        }
        let Some(s) = encode(&g) else { continue };
        w.push(s)?;
        kept += 1;
        if max_games.map(|m| kept >= m).unwrap_or(false) {
            break;
        }
    }
    w.flush()?;
    Ok(json!({"seen": seen, "kept": kept, "shards": w.written_shards}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { month, out } => {
            let name = format!("lichess_db_standard_rated_{month}.pgn.zst");
            let url = format!("https://database.lichess.org/standard/{name}");
            anyhow::ensure!(crate::common::download(&crate::common::http()?, &url, &out.join(&name))?, "404: {url}");
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, min_elo, max_games, min_plies, skip_bullet } => {
            println!("{}", convert(&input, &out, shard_size, min_elo, max_games, min_plies, skip_bullet)?);
            Ok(())
        }
    }
}
