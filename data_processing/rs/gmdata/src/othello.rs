//! Othello: WTHOR `.wtb` databases (Fédération Française d'Othello) → shards
//! for the "othello" adapter (BoardEncoder + PlacementDecoder, 8×8, 3 planes).
//!
//! One sample = one game from the side-to-move perspective at every frame;
//! frame t is the position BEFORE the t-th half-move, including the forced pass
//! frames the replayer inserts (WTHOR stores no passes).
//!
//!   obs.board_seq [T,3,8,8] f32  mover discs / opponent discs / empty
//!   tgt.place_idx [T] i64        row*8+col (0 on pass frames)
//!   tgt.passed    [T] bool
//!   tgt.value     [T] f32        sign(final own − opp discs) for the frame's mover
//!
//! Record layout: 16-byte header (n1 = game count at offset 4, board size at 12),
//! then 68-byte records: tournament u16, black u16, white u16, black real score
//! u8, theoretical score u8, 60 move bytes `10*(row+1)+(col+1)`, 0 = filler.

use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::json;
use std::path::{Path, PathBuf};

const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;
const DIRS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

pub fn opponent(p: u8) -> u8 {
    BLACK + WHITE - p
}

pub fn initial_board() -> [u8; 64] {
    let mut b = [EMPTY; 64];
    b[3 * 8 + 3] = WHITE;
    b[4 * 8 + 4] = WHITE;
    b[3 * 8 + 4] = BLACK;
    b[4 * 8 + 3] = BLACK;
    b
}

pub fn flips(b: &[u8; 64], player: u8, row: i32, col: i32) -> Vec<usize> {
    if !(0..8).contains(&row) || !(0..8).contains(&col) || b[(row * 8 + col) as usize] != EMPTY {
        return Vec::new();
    }
    let opp = opponent(player);
    let mut out = Vec::new();
    for (dr, dc) in DIRS {
        let (mut r, mut c) = (row + dr, col + dc);
        let mut run = Vec::new();
        while (0..8).contains(&r) && (0..8).contains(&c) {
            let i = (r * 8 + c) as usize;
            if b[i] == opp {
                run.push(i);
            } else {
                if b[i] == player {
                    out.extend_from_slice(&run);
                }
                break;
            }
            r += dr;
            c += dc;
        }
    }
    out
}

pub fn has_legal_move(b: &[u8; 64], player: u8) -> bool {
    (0..8).any(|r| (0..8).any(|c| !flips(b, player, r, c).is_empty()))
}

fn planes(b: &[u8; 64], mover: u8, out: &mut Vec<f32>) {
    let opp = opponent(mover);
    for want in [mover, opp, EMPTY] {
        out.extend(b.iter().map(|&v| (v == want) as u8 as f32));
    }
}

pub struct WthorGame {
    pub tournament: u16,
    pub black: u16,
    pub white: u16,
    pub black_real_score: u8,
    pub black_theoretical_score: u8,
    pub moves: Vec<(i32, i32)>,
}

/// Game year recorded in a WTHOR header.
pub fn wtb_year(data: &[u8]) -> Option<u16> {
    (data.len() >= 12).then(|| u16::from_le_bytes([data[10], data[11]]))
}

pub fn parse_wtb(data: &[u8]) -> Result<Vec<WthorGame>> {
    if data.len() < 16 {
        bail!("file too short for WTHOR header: {} bytes", data.len());
    }
    let n1 = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let p1 = data[12];
    if p1 != 0 && p1 != 8 {
        bail!("unsupported WTHOR board size p1={p1} (only 8×8)");
    }
    let expected = 16 + 68 * n1;
    if data.len() != expected {
        bail!("size mismatch: n1={n1} games need {expected} bytes, file has {}", data.len());
    }
    let u16le = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let mut games = Vec::with_capacity(n1);
    for i in 0..n1 {
        let off = 16 + i * 68;
        let mut moves = Vec::new();
        for &byte in &data[off + 8..off + 68] {
            if byte == 0 {
                break;
            }
            let (row, col) = ((byte / 10) as i32 - 1, (byte % 10) as i32 - 1);
            if !(0..8).contains(&row) || !(0..8).contains(&col) {
                bail!("invalid WTHOR 8×8 move byte {byte}");
            }
            moves.push((row, col));
        }
        games.push(WthorGame {
            tournament: u16le(off),
            black: u16le(off + 2),
            white: u16le(off + 4),
            black_real_score: data[off + 6],
            black_theoretical_score: data[off + 7],
            moves,
        });
    }
    Ok(games)
}

/// Replay → (planes, movers, place_idx, passed, final board); Err = illegal/truncated record.
#[allow(clippy::type_complexity)]
fn replay(moves: &[(i32, i32)]) -> Result<(Vec<f32>, Vec<u8>, Vec<i64>, Vec<bool>, [u8; 64])> {
    let mut b = initial_board();
    let mut player = BLACK;
    let (mut pl, mut movers, mut idx, mut passed) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (n, &(r, c)) in moves.iter().enumerate() {
        while !has_legal_move(&b, player) {
            if !has_legal_move(&b, opponent(player)) {
                bail!("move {n}: game already over (double pass) but {} recorded moves remain", moves.len() - n);
            }
            planes(&b, player, &mut pl);
            movers.push(player);
            idx.push(0);
            passed.push(true);
            player = opponent(player);
        }
        let fl = flips(&b, player, r, c);
        if fl.is_empty() {
            bail!("move {n}: ({r}, {c}) is illegal for player {player}");
        }
        planes(&b, player, &mut pl);
        movers.push(player);
        idx.push((r * 8 + c) as i64);
        passed.push(false);
        b[(r * 8 + c) as usize] = player;
        for i in fl {
            b[i] = player;
        }
        player = opponent(player);
    }
    if has_legal_move(&b, player) || has_legal_move(&b, opponent(player)) {
        bail!("record truncated: game not over after last recorded move");
    }
    Ok((pl, movers, idx, passed, b))
}

fn adjusted_black_score(black: i64, white: i64) -> i64 {
    let empties = 64 - black - white;
    if black > white {
        black + empties
    } else if white > black {
        black
    } else {
        black + empties / 2
    }
}

fn game_to_sample(g: &WthorGame, source: &str, index: usize) -> Result<(Sample, bool, i64)> {
    let (pl, movers, idx, passed, fin) = replay(&g.moves)?;
    if movers.is_empty() {
        bail!("empty move list");
    }
    let black = fin.iter().filter(|&&v| v == BLACK).count() as i64;
    let white = fin.iter().filter(|&&v| v == WHITE).count() as i64;
    let sign = ((black > white) as i32 - (black < white) as i32) as f32;
    let value: Vec<f32> = movers.iter().map(|&m| if m == BLACK { sign } else { -sign }).collect();
    let n_pass = passed.iter().filter(|&&p| p).count() as i64;
    let meta = json!({
        "source": source, "game_index": index, "tournament_id": g.tournament,
        "black_player_id": g.black, "white_player_id": g.white,
        "recorded_black_score": g.black_real_score, "theoretical_black_score": g.black_theoretical_score,
        "replayed_black": black, "replayed_white": white, "n_recorded_moves": g.moves.len(), "n_pass_frames": n_pass,
    });
    let s = Sample::new(meta)
        .bits("obs.board_seq", &[3, 8, 8], &pl)
        .i64("tgt.place_idx", &[], idx)
        .bool("tgt.passed", &[], passed)
        .f32("tgt.value", &[], value);
    Ok((s, adjusted_black_score(black, white) == g.black_real_score as i64, n_pass))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download yearly WTHOR databases (WTH_<year>.ZIP → .wtb)
    Fetch {
        #[arg(long, default_value = "data/othello/wthor")]
        out: PathBuf,
        #[arg(long, default_value_t = 1977)]
        first_year: u32,
        #[arg(long)]
        last_year: Option<u32>,
    },
    /// Convert .wtb files to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 512)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        /// keep games whose recorded black score is within E discs of the theoretical
        /// (perfect-play-from-depth) score: E bounds the total endgame error of both players
        #[arg(long)]
        max_endgame_error: Option<i64>,
        #[arg(long)]
        min_year: Option<u16>,
        /// high-quality preset: endgame error <= 4 discs, games from 2000 on
        #[arg(long)]
        hq: bool,
    },
}

fn current_year() -> u32 {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    (1970 + secs / 31_556_952) as u32
}

pub fn fetch(out: &Path, first: u32, last: u32) -> Result<()> {
    let client = common::http()?;
    std::fs::create_dir_all(out)?;
    for year in first..=last {
        let wtb = out.join(format!("WTH_{year}.wtb"));
        if wtb.exists() {
            eprintln!("have {}", wtb.display());
            continue;
        }
        let zip_path = out.join(format!("WTH_{year}.ZIP"));
        let url = format!("https://www.ffothello.org/wthor/base_zip/WTH_{year}.ZIP");
        if !common::download(&client, &url, &zip_path)? {
            eprintln!("  WTH_{year}.ZIP not available — skipped");
            continue;
        }
        let mut z = zip::ZipArchive::new(std::fs::File::open(&zip_path)?)?;
        for i in 0..z.len() {
            let mut f = z.by_index(i)?;
            let name = f.name().to_string();
            if name.to_lowercase().ends_with(".wtb") {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut f, &mut buf)?;
                std::fs::write(out.join(Path::new(&name).file_name().context("zip member name")?), buf)?;
            }
        }
        std::fs::remove_file(&zip_path)?;
    }
    Ok(())
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, max_err: Option<i64>, min_year: Option<u16>)
    -> Result<serde_json::Value> {
    let files = common::find_files(input, &[".wtb"])?;
    anyhow::ensure!(!files.is_empty(), "no .wtb files under {}", input.display());
    let mut w = ShardWriter::new(out, "othello", 1, json!({}), shard_size)?;
    let (mut games, mut converted, mut illegal, mut mismatch, mut pass_frames) = (0usize, 0usize, 0usize, 0usize, 0i64);
    let mut low_quality = 0usize;
    'outer: for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let data = std::fs::read(&path)?;
        let year_ok = min_year.map(|y| wtb_year(&data).map(|fy| fy >= y).unwrap_or(false)).unwrap_or(true);
        for (i, g) in parse_wtb(&data).with_context(|| name.clone())?.iter().enumerate() {
            if max_games.map(|m| games >= m).unwrap_or(false) {
                break 'outer;
            }
            games += 1;
            let err_ok = max_err.map(|e| (g.black_real_score as i64 - g.black_theoretical_score as i64).abs() <= e).unwrap_or(true);
            if !year_ok || !err_ok {
                low_quality += 1;
                continue;
            }
            match game_to_sample(g, &name, i) {
                Ok((s, ok, np)) => {
                    converted += 1;
                    pass_frames += np;
                    mismatch += (!ok) as usize;
                    w.push(s)?;
                }
                Err(e) => {
                    illegal += 1;
                    eprintln!("[warn] {name}#{i}: skipped — {e}");
                }
            }
        }
    }
    w.flush()?;
    Ok(json!({"games": games, "converted": converted, "illegal": illegal, "score_mismatch": mismatch, "low_quality": low_quality,
              "pass_frames": pass_frames, "shards": w.written_shards}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, first_year, last_year } => fetch(&out, first_year, last_year.unwrap_or_else(current_year)),
        Cmd::Convert { input, out, shard_size, max_games, max_endgame_error, min_year, hq } => {
            let (e, y) = if hq {
                (Some(max_endgame_error.unwrap_or(4).min(4)), Some(min_year.unwrap_or(2000).max(2000)))
            } else {
                (max_endgame_error, min_year)
            };
            let s = convert(&input, &out, shard_size, max_games, e, y)?;
            println!("{s}");
            Ok(())
        }
    }
}
