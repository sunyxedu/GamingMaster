//! Chess: Lichess PGN (`.pgn` / `.pgn.zst`, streamed) → shards for the chess adapter.
//!
//! One sample per game; frame t = position BEFORE move t from the side to move
//! (ranks mirrored for black, files never mirrored — the Maia convention).
//!
//!   obs.board_seq   [T,14,8,8] f32  own P N B R Q K, opp P N B R Q K, ep square, ones
//!   obs.globals_seq [T,10] f32      own K/Q castling, opp K/Q castling, white to move,
//!                                   fullmove/100, halfmove/100 (clipped), in check,
//!                                   position occurred ≥ 1 / ≥ 2 times earlier in the game
//!   tgt.from_idx / tgt.to_idx [T] i64   POV square row*8+col
//!   tgt.promotion   [T] i64         0 none, 1 N, 2 B, 3 R, 4 Q
//!   tgt.value       [T] f32         result for the mover
//!   tgt.legal_from  [T,64] f32      {0,1} from-squares of the legal moves (bit-packed)
//!   tgt.legal_to_of [T,64] f32      {0,1} destinations of the legal moves from the GT from-square
//!   tgt.legal_promotion_of [T,5] f32  {0,1} promotion indices of the legal moves with the
//!                                   GT (from, to) — {0} for a non-promotion move
//!
//! POV mapping: white to move row = 7 − rank; black to move row = rank; col = file.
//! Castling is labelled king → g/c file (python-chess standard encoding), in the labels and
//! the masks alike ([`move_target`]). Repetition = FIDE position identity (board, side to
//! move, castling rights, legal en-passant square): globals 8/9 say the frame's position was
//! already seen once / twice before (9 = a threefold repetition is claimable here).
//! Filters: standard variant, no FEN/SetUp, decisive/drawn result, both Elo ≥ min
//! (untagged skipped when min > 0), ≥ min plies, optional no-bullet
//! (base + 40·inc < 180 s). Games with an unparsable/illegal/null move are skipped.
//! converter_version 2: rank-only mirror. converter_version 3: globals 8 → 10 (repetition
//! flags) and the legality masks; shard meta `{"global_features": 10}`.

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

pub const CONVERTER_VERSION: u32 = 3;
/// Width of `obs.globals_seq` (shard meta `global_features`).
pub const GLOBAL_FEATURES: usize = 10;
/// Promotion classes (`tgt.promotion` / `tgt.legal_promotion_of`).
pub const N_PROMOTIONS: usize = 5;

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
                mt.1.play_unchecked(m);
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

/// The result for white of a PGN `Result` tag (`None` = unfinished / unknown).
pub(crate) fn result_white(r: &str) -> Option<f32> {
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

/// Extra quality filters (all off by default; `--hq` turns on a strong preset).
#[derive(Clone, Copy, Default)]
pub struct Quality {
    /// only games whose Event starts with "Rated"
    pub rated_only: bool,
    /// only Termination "Normal" (drops time forfeits, abandoned games, rules infractions)
    pub normal_termination: bool,
    /// estimated duration base + 40·inc must be at least this many seconds
    pub min_duration_s: i64,
}

fn duration_s(h: &HashMap<String, String>) -> Option<i64> {
    let tc = h.get("TimeControl")?;
    let (b, i) = tc.split_once('+')?;
    Some(py_int(b)? + 40 * py_int(i)?)
}

fn passes_quality(g: &Game, q: &Quality) -> bool {
    let h = &g.headers;
    if q.rated_only && !h.get("Event").map(|e| e.starts_with("Rated")).unwrap_or(false) {
        return false;
    }
    if q.normal_termination && h.get("Termination").map(|t| t.as_str()) != Some("Normal") {
        return false;
    }
    if q.min_duration_s > 0 && duration_s(h).map(|d| d < q.min_duration_s).unwrap_or(true) {
        return false;
    }
    true
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

/// POV square index `row·8 + file` of `sq` for the side to move (`flip` = black to move:
/// row = rank; white to move: row = 7 − rank; ranks / files 0-based from a1).
pub(crate) fn pov(sq: Square, flip: bool) -> i64 {
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

/// A move as the chess adapter labels it, in the frame's POV coordinates:
/// (from square, to square, promotion index). Castling → king destination file g/c
/// (python-chess standard mode; shakmaty encodes castling as king → rook). Used for the
/// labels AND the legality masks, so the GT move always lies inside its own masks.
pub(crate) fn move_target(m: &Move, flip: bool) -> Option<(i64, i64, i64)> {
    let from_sq = m.from()?;
    let to_sq = match *m {
        Move::Castle { king, rook } => {
            let side = if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide };
            Square::from_coords(side.king_to_file(), king.rank())
        }
        _ => m.to(),
    };
    Some((pov(from_sq, flip), pov(to_sq, flip), promo_idx(m.promotion())))
}

/// Legality masks of one frame, conditioned on the GT move `gt` = (from, to, promotion):
/// (legal from-squares, legal destinations from `gt.from`, legal promotions of
/// (`gt.from`, `gt.to`)). `None` if `gt` is not inside all three (never for a legal move).
#[allow(clippy::type_complexity)]
pub(crate) fn legal_masks(pos: &Chess, flip: bool, gt: (i64, i64, i64)) -> Option<([f32; 64], [f32; 64], [f32; N_PROMOTIONS])> {
    let (mut from, mut to_of, mut promo_of) = ([0f32; 64], [0f32; 64], [0f32; N_PROMOTIONS]);
    for m in pos.legal_moves() {
        let Some((f, t, p)) = move_target(&m, flip) else { continue };
        from[f as usize] = 1.0;
        if f == gt.0 {
            to_of[t as usize] = 1.0;
            if t == gt.1 {
                promo_of[p as usize] = 1.0;
            }
        }
    }
    let inside = from[gt.0 as usize] == 1.0 && to_of[gt.1 as usize] == 1.0 && promo_of[gt.2 as usize] == 1.0;
    inside.then_some((from, to_of, promo_of))
}

/// Floats of one `obs.board_seq` frame (14 planes of 8×8).
pub(crate) const PLANE_LEN: usize = 14 * 64;

/// Occurrence counts of the positions of one game (Chess Eq/Hash = FIDE repetition
/// identity: board, side to move, castling rights, legal en-passant square).
#[derive(Default)]
pub(crate) struct Repetitions(HashMap<Chess, u32>);

impl Repetitions {
    /// Record `pos`; returns how often it occurred earlier in the game.
    pub(crate) fn see(&mut self, pos: &Chess) -> u32 {
        let n = self.0.entry(pos.clone()).or_insert(0);
        *n += 1;
        *n - 1
    }
}

/// Observation of one frame (the position before the move, from the mover's POV):
/// writes the `obs.board_seq` planes into `planes` (`PLANE_LEN` floats, zero on entry)
/// and appends the `GLOBAL_FEATURES` `obs.globals_seq` values to `globals`; `prior` =
/// earlier occurrences of the position ([`Repetitions::see`]). Shared with
/// `gmdata label chess`, which re-encodes replayed games to verify them.
pub(crate) fn encode_position(pos: &Chess, prior: u32, planes: &mut [f32], globals: &mut Vec<f32>) {
    let own = pos.turn();
    let flip = own == Color::Black;
    for (sq, piece) in pos.board().iter() {
        let plane = if piece.color == own { 0 } else { 6 } + role_idx(piece.role);
        planes[plane * 64 + pov(sq, flip) as usize] = 1.0;
    }
    if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
        planes[12 * 64 + pov(ep, flip) as usize] = 1.0;
    }
    for x in &mut planes[13 * 64..14 * 64] {
        *x = 1.0;
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
        (prior >= 1) as u8 as f32,
        (prior >= 2) as u8 as f32,
    ]);
}

/// Why a game produced no sample.
#[derive(Debug, PartialEq, Eq)]
enum Skip {
    /// illegal move / missing from-square / unknown result
    Invalid,
    /// the GT move fell outside its own legality masks (label/mask convention bug)
    MaskMismatch,
}

/// Per-frame tensors of one game (flat, frame-major).
#[derive(Default)]
struct Frames {
    planes: Vec<f32>,
    globals: Vec<f32>,
    from: Vec<i64>,
    to: Vec<i64>,
    promo: Vec<i64>,
    value: Vec<f32>,
    legal_from: Vec<f32>,
    legal_to_of: Vec<f32>,
    legal_promotion_of: Vec<f32>,
}

/// Replay `moves` from the initial position; `wv` = result for white.
fn encode_moves(moves: &[Move], wv: f32) -> Result<Frames, Skip> {
    let t = moves.len();
    let mut f = Frames {
        planes: vec![0f32; t * PLANE_LEN],
        globals: Vec::with_capacity(t * GLOBAL_FEATURES),
        legal_from: Vec::with_capacity(t * 64),
        legal_to_of: Vec::with_capacity(t * 64),
        legal_promotion_of: Vec::with_capacity(t * N_PROMOTIONS),
        ..Frames::default()
    };
    let mut pos = Chess::default();
    let mut reps = Repetitions::default();
    for (k, m) in moves.iter().enumerate() {
        if !pos.is_legal(*m) {
            return Err(Skip::Invalid);
        }
        let own = pos.turn();
        let flip = own == Color::Black;
        let prior = reps.see(&pos);
        encode_position(&pos, prior, &mut f.planes[k * PLANE_LEN..(k + 1) * PLANE_LEN], &mut f.globals);
        let gt = move_target(m, flip).ok_or(Skip::Invalid)?;
        let (lf, lt, lp) = legal_masks(&pos, flip, gt).ok_or(Skip::MaskMismatch)?;
        f.legal_from.extend_from_slice(&lf);
        f.legal_to_of.extend_from_slice(&lt);
        f.legal_promotion_of.extend_from_slice(&lp);
        f.from.push(gt.0);
        f.to.push(gt.1);
        f.promo.push(gt.2);
        f.value.push(if own == Color::White { wv } else { -wv });
        pos.play_unchecked(*m);
    }
    Ok(f)
}

fn encode(g: &Game) -> Result<Sample, Skip> {
    let wv = result_white(&g.headers["Result"]).ok_or(Skip::Invalid)?;
    let f = encode_moves(&g.moves, wv)?;
    let h = &g.headers;
    let e = |k: &str| elo(h, k).map(Value::from).unwrap_or(Value::Null);
    let meta = json!({
        "white": h.get("White").cloned().unwrap_or_else(|| "?".into()),
        "black": h.get("Black").cloned().unwrap_or_else(|| "?".into()),
        "elo": [e("WhiteElo"), e("BlackElo")],
        "result": h.get("Result").cloned(),
    });
    Ok(Sample::new(meta)
        .bits("obs.board_seq", &[14, 8, 8], &f.planes)
        .f32("obs.globals_seq", &[GLOBAL_FEATURES], f.globals)
        .i64("tgt.from_idx", &[], f.from)
        .i64("tgt.to_idx", &[], f.to)
        .i64("tgt.promotion", &[], f.promo)
        .f32("tgt.value", &[], f.value)
        .bits("tgt.legal_from", &[64], &f.legal_from)
        .bits("tgt.legal_to_of", &[64], &f.legal_to_of)
        .bits("tgt.legal_promotion_of", &[N_PROMOTIONS], &f.legal_promotion_of))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download a monthly Lichess standard-rated dump (YYYY-MM, kept compressed), or with
    /// `--elite` the Lichess Elite Database month (2400+ vs 2200+, 2500+ vs 2300+ since 2021-12,
    /// no bullet)
    Fetch {
        #[arg(long)]
        month: String,
        #[arg(long)]
        elite: bool,
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
        #[arg(long)]
        rated_only: bool,
        #[arg(long)]
        normal_termination: bool,
        #[arg(long, default_value_t = 0)]
        min_duration_s: i64,
        /// high-quality preset: both Elo >= 2200, rated, normal termination, no bullet
        /// (estimated duration >= 180 s), >= 20 plies
        #[arg(long)]
        hq: bool,
    },
}

fn open(path: &Path) -> Result<Box<dyn Read>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    if path.extension().map(|e| e == "zip").unwrap_or(false) {
        // Elite-database months: one .pgn per zip
        let mut z = zip::ZipArchive::new(f)?;
        let mut buf = Vec::new();
        for i in 0..z.len() {
            let mut m = z.by_index(i)?;
            if m.name().ends_with(".pgn") {
                m.read_to_end(&mut buf)?;
            }
        }
        return Ok(Box::new(std::io::Cursor::new(buf)));
    }
    if path.extension().map(|e| e == "zst").unwrap_or(false) {
        let mut d = zstd::stream::read::Decoder::new(f)?;
        d.window_log_max(31)?;
        Ok(Box::new(d))
    } else {
        Ok(Box::new(f))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn convert(input: &Path, out: &Path, shard_size: usize, min_elo: i64, max_games: Option<usize>, min_plies: usize,
               skip_bullet: bool, q: Quality) -> Result<Value> {
    let mut w = ShardWriter::new(out, "chess", CONVERTER_VERSION, json!({"global_features": GLOBAL_FEATURES}),
                                 shard_size)?;
    let mut reader = Reader::new(std::io::BufReader::with_capacity(1 << 20, open(input)?));
    let (mut seen, mut kept, mut mask_mismatch) = (0usize, 0usize, 0usize);
    while let Some(g) = reader.read_game(&mut Collector)? {
        if g.bad {
            continue;
        }
        seen += 1;
        if !passes(&g, min_elo, min_plies, skip_bullet) || !passes_quality(&g, &q) {
            continue;
        }
        let s = match encode(&g) {
            Ok(s) => s,
            Err(Skip::MaskMismatch) => {
                mask_mismatch += 1;
                continue;
            }
            Err(Skip::Invalid) => continue,
        };
        w.push(s)?;
        kept += 1;
        if max_games.map(|m| kept >= m).unwrap_or(false) {
            break;
        }
    }
    w.flush()?;
    Ok(json!({"seen": seen, "kept": kept, "mask_mismatch": mask_mismatch, "shards": w.written_shards}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { month, elite, out } => {
            let (name, url) = if elite {
                let n = format!("lichess_elite_{month}.zip");
                (n.clone(), format!("https://database.nikonoel.fr/{n}"))
            } else {
                let n = format!("lichess_db_standard_rated_{month}.pgn.zst");
                (n.clone(), format!("https://database.lichess.org/standard/{n}"))
            };
            anyhow::ensure!(crate::common::download(&crate::common::http()?, &url, &out.join(&name))?, "404: {url}");
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, min_elo, max_games, min_plies, skip_bullet, rated_only, normal_termination, min_duration_s, hq } => {
            let mut q = Quality { rated_only, normal_termination, min_duration_s };
            let (mut min_elo, mut min_plies, mut skip_bullet) = (min_elo, min_plies, skip_bullet);
            if hq {
                q = Quality { rated_only: true, normal_termination: true, min_duration_s: q.min_duration_s.max(180) };
                min_elo = min_elo.max(2200);
                min_plies = min_plies.max(20);
                skip_bullet = true;
            }
            println!("{}", convert(&input, &out, shard_size, min_elo, max_games, min_plies, skip_bullet, q)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode};

    fn play(sans: &[&str]) -> Vec<Move> {
        let mut pos = Chess::default();
        sans.iter()
            .map(|s| {
                let m = San::from_ascii(s.as_bytes()).unwrap().to_move(&pos).unwrap();
                pos.play_unchecked(m);
                m
            })
            .collect()
    }

    fn position(fen: &str) -> Chess {
        Fen::from_ascii(fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn san(pos: &Chess, s: &str) -> Move {
        San::from_ascii(s.as_bytes()).unwrap().to_move(pos).unwrap()
    }

    fn ones(mask: &[f32]) -> Vec<usize> {
        mask.iter().enumerate().filter(|(_, &x)| x == 1.0).map(|(i, _)| i).collect()
    }

    /// POV square of (file 0..7, rank 1..8) for the side to move (`black` → rank mirror).
    fn sq(file: i64, rank: i64, black: bool) -> usize {
        (if black { (rank - 1) * 8 + file } else { (8 - rank) * 8 + file }) as usize
    }

    #[test]
    fn start_position_masks() {
        let f = encode_moves(&play(&["Nf3", "e5"]), 1.0).unwrap();
        // white: a2..h2 (row 6) + b1, g1 (row 7); black (rank mirror): the same POV squares
        let mut want: Vec<usize> = (0..8).map(|c| sq(c, 2, false)).collect();
        want.extend([sq(1, 1, false), sq(6, 1, false)]);
        want.sort();
        assert_eq!(want.len(), 10);
        assert_eq!(ones(&f.legal_from[..64]), want);
        assert_eq!(ones(&f.legal_from[64..128]), want);
        // knight g1 → {f3, h3}; GT label Nf3 = (g1, f3, none)
        assert_eq!(ones(&f.legal_to_of[..64]), vec![sq(5, 3, false), sq(7, 3, false)]);
        assert_eq!((f.from[0], f.to[0], f.promo[0]), (sq(6, 1, false) as i64, sq(5, 3, false) as i64, 0));
        assert_eq!(ones(&f.legal_promotion_of[..N_PROMOTIONS]), vec![0]);
        // black e7 pawn → {e6, e5} in black's POV
        assert_eq!(ones(&f.legal_to_of[64..128]), vec![sq(4, 5, true), sq(4, 6, true)]);
        assert_eq!(f.globals.len(), 2 * GLOBAL_FEATURES);
    }

    #[test]
    fn repetition_flags() {
        let cycle = ["Nf3", "Nf6", "Ng1", "Ng8"];
        let mut sans: Vec<&str> = cycle.iter().chain(cycle.iter()).copied().collect();
        sans.push("Nf3");
        let f = encode_moves(&play(&sans), 0.0).unwrap();
        let flags: Vec<(f32, f32)> = f.globals.chunks(GLOBAL_FEATURES).map(|g| (g[8], g[9])).collect();
        // frames 0..3 new; 4 = start position again (seen once before); 8 = third time
        assert_eq!(flags[..4], [(0.0, 0.0); 4]);
        assert_eq!(flags[4..8], [(1.0, 0.0); 4]);
        assert_eq!(flags[8], (1.0, 1.0));
        // indices 0..7 unchanged: castling rights all set, white to move on even frames
        assert_eq!(&f.globals[..5], &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(f.globals[GLOBAL_FEATURES + 4], 0.0);
    }

    #[test]
    fn promotion_mask() {
        for (fen, black) in [("8/P6k/8/8/8/8/8/K7 w - - 0 1", false), ("k7/8/8/8/8/8/p6K/8 b - - 0 1", true)] {
            let pos = position(fen);
            let m = san(&pos, if black { "a1=Q" } else { "a8=Q" });
            let gt = move_target(&m, black).unwrap();
            let (from_rank, to_rank) = if black { (2, 1) } else { (7, 8) };
            assert_eq!(gt, (sq(0, from_rank, black) as i64, sq(0, to_rank, black) as i64, 4));
            let (_, to_of, promo_of) = legal_masks(&pos, black, gt).unwrap();
            assert_eq!(ones(&to_of), vec![gt.1 as usize]);
            assert_eq!(ones(&promo_of), vec![1, 2, 3, 4]);
            // under-promotion labels are inside the same mask
            let n = move_target(&san(&pos, if black { "a1=N" } else { "a8=N" }), black).unwrap();
            assert_eq!(n.2, 1);
            assert!(legal_masks(&pos, black, n).is_some());
        }
    }

    #[test]
    fn castling_mask_uses_king_destination() {
        for (fen, black, s) in [("4k3/8/8/8/8/8/8/4K2R w K - 0 1", false, "O-O"),
                                ("4k2r/8/8/8/8/8/8/4K3 b k - 0 1", true, "O-O"),
                                ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", true, "O-O-O")] {
            let pos = position(fen);
            let m = san(&pos, s);
            assert!(matches!(m, Move::Castle { .. }));
            let gt = move_target(&m, black).unwrap();
            let file = if s == "O-O" { 6 } else { 2 };
            let rank = if black { 8 } else { 1 };
            assert_eq!(gt, (sq(4, rank, black) as i64, sq(file, rank, black) as i64, 0));
            let (from, to_of, promo_of) = legal_masks(&pos, black, gt).unwrap();
            assert_eq!(from[sq(4, rank, black)], 1.0);
            assert_eq!(to_of[sq(file, rank, black)], 1.0);
            // the rook square (shakmaty's castling `to`) is not a king destination
            let rook_file = if s == "O-O" { 7 } else { 0 };
            assert_eq!(to_of[sq(rook_file, rank, black)], 0.0);
            assert_eq!(ones(&promo_of), vec![0]);
        }
    }

    #[test]
    fn gt_outside_masks_is_a_mismatch() {
        let pos = Chess::default();
        let e2 = sq(4, 2, false) as i64;
        assert!(legal_masks(&pos, false, (e2, sq(4, 5, false) as i64, 0)).is_none());
        assert!(legal_masks(&pos, false, (e2, sq(4, 4, false) as i64, 4)).is_none());
        assert!(legal_masks(&pos, false, (e2, sq(4, 4, false) as i64, 0)).is_some());
    }
}
