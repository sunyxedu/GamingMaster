//! Shogi: floodgate CSA records → shards for the shogi adapter (converter_version 2).
//!
//! Frame t = position BEFORE move t from the side to move (board rotated 180°
//! for white). Square index = row-major 9×9 with square 0 = 9a (gote's back rank);
//! POV index = sq for black, 80 − sq for white.
//!
//!   obs.board_seq   [T,28,9,9] f32  own P L N S G B R K +P +L +N +S +B +R, then opponent
//!   obs.hand_seq    [T,14,4] f32    own P..R then opponent: [count, count>0, min(count/4,1), 1]
//!   obs.globals_seq [T,7] f32       [min(move_number/512,1), mover is black, in check, 1,
//!                                    rep≥1, rep≥2, rep≥3] — rep = how often this exact position
//!                                    (pieces, hands, side to move) occurred earlier in the game
//!                                    (sennichite = the 4th occurrence)
//!   tgt.dropped [T] i64, tgt.from_idx, tgt.to_idx, tgt.piece_idx, tgt.drop_to_idx [T] i64,
//!   tgt.promoted [T] bool, tgt.value [T] f32 (result for the mover)
//!
//! Legality masks (bit-packed, float32 {0,1} when loaded; POV indices, true shogi rules):
//!
//!   tgt.legal_mode       [T,2]   [any legal board move, any legal drop]
//!   tgt.legal_from       [T,81]  from-squares of legal board moves
//!   tgt.legal_to_of      [T,81]  destinations of legal board moves from tgt.from_idx
//!   tgt.legal_promo_of   [T,2]   [non-promotion legal, promotion legal] for (tgt.from_idx, tgt.to_idx)
//!   tgt.legal_drop_piece [T,14]  own hand slot i (piece type P L N S G B R) has a legal drop;
//!                                slots 7..13 (opponent's hand) are always 0
//!   tgt.legal_drop_to_of [T,81]  legal drop squares for hand slot tgt.piece_idx
//!
//! The `_of` rows are computed for the stored indices on every frame, also on frames of the
//! other mode (from/to = 0 on drop frames, piece_idx = 0 on move frames), where the decoder
//! gives those heads no weight. A game whose recorded move falls outside its masks is skipped
//! (`mask_mismatch`).
//!
//! Shard meta: {"global_features": 7}. converter_version 1 shards had globals [T,4] and no masks.
//!
//! The rules engine and the frame encoders live in `gmrules::games::shogi` (one
//! implementation for this converter, its labeller and the simulator, gmsim). The engine
//! reproduces the reference (python-shogi) semantics exactly, including its legality test:
//! pseudo-legal ∧ ¬(own king attacked after the move) ∧ ¬(pawn-drop mate as approximated
//! there: the king has no unattacked escape square — occupancy unchanged — and no non-king
//! piece attacks the pawn). The move generator ([`Board::legal_moves`]) applies that test to
//! every candidate move; drops go onto empty squares only (the reference `is_pseudo_legal`
//! also accepts a drop onto an opponent piece, which its own generator never produces and
//! real records never contain).

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Shard converter_version: 2 = repetition globals [T,7] + legality masks.
pub(crate) const CONVERTER_VERSION: u32 = 2;

// The rules and the frame encoders are gmrules' (`gmrules::games::shogi`, one
// implementation for this converter, its labeller and the simulator).
pub(crate) use gmrules::games::shogi::{
    board_from_obs, frame_masks, hirate, move_indices, pov, repetition_counts, Board, Mv, BLACK, KING, NONE, OBS_BOARD,
    PAWN, ROOK, WHITE,
};
#[cfg(test)]
pub(crate) use gmrules::games::shogi::BISHOP;
use gmrules::games::shogi::{self as rules, promoted, GLOBAL_FEATURES};

/// One frame's observation, appended: `board` is the frame's zeroed obs.board_seq slice
/// [28·81]; its obs.hand_seq [14·4] and obs.globals_seq [7] values are appended to `hand` /
/// `glb` (the verification of `gmdata label shogi`; [`rules::encode_obs`]).
pub(crate) fn encode_obs(b: &Board, rep: u32, board: &mut [f32], hand: &mut Vec<f32>, glb: &mut Vec<f32>) {
    let (h, g) = (hand.len(), glb.len());
    hand.resize(h + 56, 0.0);
    glb.resize(g + GLOBAL_FEATURES, 0.0);
    rules::encode_obs(b, rep, board, &mut hand[h..], &mut glb[g..]);
}

// ---------------------------------------------------------------- CSA

const CSA_PIECES: [&str; 15] = ["* ", "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU", "TO", "NY", "NK", "NG", "UM", "RY"];

fn csa_piece(s: &str) -> Option<u8> {
    CSA_PIECES.iter().position(|&p| p == s).map(|i| i as u8)
}

/// CSA "91" (file 9, rank 1) → square index.
pub(crate) fn csa_square(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'1'..=b'9').contains(&b[0]) || !(b'1'..=b'9').contains(&b[1]) {
        return None;
    }
    let (file, rnk) = ((b[0] - b'0') as usize, (b[1] - b'0') as usize);
    Some((rnk - 1) * 9 + (9 - file))
}

struct Parsed {
    names: [Option<String>; 2],
    start: Board,
    moves: Vec<Mv>,
}

fn parse_position(lines: &[&str]) -> Option<([Option<(u8, usize)>; 81], [[u32; 9]; 2])> {
    let mut pieces: [Option<(u8, usize)>; 81] = [None; 81];
    let mut hand = [[0u32; 9]; 2];
    for line in lines {
        let b: Vec<char> = line.chars().collect();
        if b.len() < 2 {
            return None;
        }
        match b[1] {
            '+' | '-' => {
                let color = if b[1] == '+' { BLACK } else { WHITE };
                let mut i = 2;
                while i < b.len() {
                    let f = b.get(i)?.to_digit(10)? as usize;
                    let r = b.get(i + 1)?.to_digit(10)? as usize;
                    let pt = csa_piece(&b.get(i + 2..i + 4)?.iter().collect::<String>())?;
                    i += 4;
                    if f == 0 && r == 0 {
                        // only unpromoted P..R can be in hand (a king / promoted piece used to
                        // index past the hand array and panic the whole conversion)
                        if !(PAWN..=ROOK).contains(&pt) {
                            return None;
                        }
                        hand[color][pt as usize] += 1;
                    } else {
                        if r == 0 || f == 0 {
                            return None;
                        }
                        pieces[(r - 1) * 9 + (9 - f)] = Some((pt, color));
                    }
                }
            }
            '1'..='9' => {
                let r = b[1].to_digit(10)? as usize - 1;
                for (c, idx) in (2..29).step_by(3).enumerate() {
                    let s: String = b.get(idx..idx + 3)?.iter().collect();
                    let pt = csa_piece(&s[1..])?;
                    pieces[r * 9 + c] = if pt != 0 {
                        Some((pt, if s.starts_with('+') { BLACK } else if s.starts_with('-') { WHITE } else { return None }))
                    } else {
                        None
                    };
                }
            }
            'I' => {
                pieces = hirate();
                let mut i = 2;
                while i < b.len() {
                    let f = b.get(i)?.to_digit(10)? as usize;
                    let r = b.get(i + 1)?.to_digit(10)? as usize;
                    let pt = csa_piece(&b.get(i + 2..i + 4)?.iter().collect::<String>())?;
                    i += 4;
                    if r == 0 && f == 0 {
                        return None;
                    }
                    let idx = (r - 1) * 9 + (9 - f);
                    if pieces[idx].map(|(p, _)| p) != Some(pt) {
                        return None;
                    }
                    pieces[idx] = None;
                }
            }
            _ => return None,
        }
    }
    Some((pieces, hand))
}

/// Mirrors the reference parser: unknown line kinds / moves before the position → Err.
fn parse_csa(text: &str) -> std::result::Result<Parsed, ()> {
    let mut names: [Option<String>; 2] = [None, None];
    let mut pos_lines: Vec<&str> = Vec::new();
    let mut turn: Option<usize> = None;
    let mut board: Option<Board> = None;
    let mut moves = Vec::new();
    let mut start: Option<Board> = None;
    for line in text.split('\n') {
        let first = line.chars().next();
        match first {
            None | Some('\'') | Some('V') | Some('$') | Some('T') => {}
            Some('N') if matches!(line.chars().nth(1), Some('+') | Some('-')) => {
                names[if line.chars().nth(1) == Some('+') { 0 } else { 1 }] = Some(line.chars().skip(2).collect());
            }
            Some('P') => pos_lines.push(line),
            Some('+') | Some('-') => {
                if line.chars().count() == 1 {
                    turn = Some(if first == Some('+') { BLACK } else { WHITE });
                } else {
                    let b = board.as_mut().ok_or(())?;
                    let cs: Vec<char> = line.chars().collect();
                    let from_s: String = cs.get(1..3).ok_or(())?.iter().collect();
                    let to_s: String = cs.get(3..5).ok_or(())?.iter().collect();
                    let pc_s: String = cs.get(5..7).ok_or(())?.iter().collect();
                    let to = csa_square(&to_s).ok_or(())?;
                    let pt = csa_piece(&pc_s).ok_or(())?;
                    let m = if from_s == "00" {
                        if pt == 0 || pt > 7 {
                            return Err(());
                        }
                        Mv { from: None, to, promotion: false, drop: pt }
                    } else {
                        let from = csa_square(&from_s).ok_or(())?;
                        Mv { from: Some(from), to, promotion: b.piece[from] != pt, drop: NONE }
                    };
                    // reference parser pushes without validation; guard impossible states
                    if m.drop == NONE && b.piece[m.from.unwrap()] == NONE {
                        return Err(());
                    }
                    if m.drop != NONE && b.hand[b.turn][m.drop as usize] == 0 {
                        return Err(());
                    }
                    if m.promotion && promoted(b.piece[m.from.unwrap()]) == 0 {
                        return Err(());
                    }
                    b.push(&m);
                    moves.push(m);
                }
            }
            Some('%') => {
                if board.is_none() {
                    return Err(());
                }
                break;
            }
            _ => return Err(()),
        }
        if board.is_none() {
            if let Some(t) = turn {
                let (pieces, hand) = parse_position(&pos_lines).ok_or(())?;
                // the reference exports hands with count > 1 as a malformed SFEN; refuse them
                if hand.iter().any(|h| h.iter().any(|&n| n > 1)) {
                    return Err(());
                }
                let mut b = Board { piece: [NONE; 81], color: [0; 81], hand, turn: t, king: [None, None], move_number: 1 };
                for (i, p) in pieces.iter().enumerate() {
                    if let Some((pt, c)) = p {
                        b.piece[i] = *pt;
                        b.color[i] = *c as u8;
                        if *pt == KING {
                            b.king[*c] = Some(i);
                        }
                    }
                }
                start = Some(b.clone());
                board = Some(b);
            }
        }
    }
    Ok(Parsed { names, start: start.ok_or(())?, moves })
}

fn game_result(text: &str, start_turn: usize, n_moves: usize) -> &'static str {
    let mut token = None;
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('%') {
            token = l.split_whitespace().next();
            break;
        }
    }
    let Some(t) = token else { return "none" };
    match t {
        "%+ILLEGAL_ACTION" => "w",
        "%-ILLEGAL_ACTION" => "b",
        "%SENNICHITE" | "%JISHOGI" | "%HIKIWAKE" => "draw",
        "%TORYO" | "%TIME_UP" | "%ILLEGAL_MOVE" | "%TSUMI" | "%KACHI" => {
            let stm = (start_turn + n_moves) % 2;
            let winner = if t == "%KACHI" { stm } else { 1 - stm };
            if winner == BLACK {
                "b"
            } else {
                "w"
            }
        }
        _ => "none",
    }
}

pub(crate) fn result_value(result: &str, mover_black: bool) -> f32 {
    match result {
        "b" => if mover_black { 1.0 } else { -1.0 },
        "w" => if mover_black { -1.0 } else { 1.0 },
        _ => 0.0,
    }
}

pub(crate) fn convert_game(text: &str, min_moves: usize, allow_no_result: bool) -> std::result::Result<Sample, &'static str> {
    let p = parse_csa(text).map_err(|_| "parse_error")?;
    if p.moves.len() < min_moves {
        return Err("too_short");
    }
    let mut b = p.start.clone();
    let result = game_result(text, b.turn, p.moves.len());
    if result == "none" && !allow_no_result {
        return Err("no_result");
    }
    let t = p.moves.len();
    let reps = repetition_counts(&p.start, &p.moves);
    let mut board = vec![0f32; t * OBS_BOARD];
    let mut hand = Vec::with_capacity(t * 56);
    let mut glb = Vec::with_capacity(t * GLOBAL_FEATURES);
    let (mut dropped, mut from, mut to, mut promo, mut piece_idx, mut drop_to, mut value) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut lm = Vec::with_capacity(t * 2);
    let mut lf = Vec::with_capacity(t * 81);
    let mut lto = Vec::with_capacity(t * 81);
    let mut lpr = Vec::with_capacity(t * 2);
    let mut ldp = Vec::with_capacity(t * 14);
    let mut ldt = Vec::with_capacity(t * 81);
    let bit = |v: &[bool], out: &mut Vec<f32>| out.extend(v.iter().map(|&x| x as u8 as f32));
    for (k, m) in p.moves.iter().enumerate() {
        if !b.is_legal(m) {
            return Err("illegal_move");
        }
        let mb = b.turn == BLACK;
        encode_obs(&b, reps[k], &mut board[k * OBS_BOARD..(k + 1) * OBS_BOARD], &mut hand, &mut glb);
        let (d, fi, ti, pr, pi, dt) = move_indices(m, mb);
        let f = frame_masks(&b.legal_moves(), mb, fi, ti, pi);
        let inside = if d == 1 {
            f.mode[1] && f.drop_piece[pi as usize] && f.drop_to_of[dt as usize]
        } else {
            f.mode[0] && f.from[fi as usize] && f.to_of[ti as usize] && f.promo_of[pr as usize]
        };
        if !inside {
            return Err("mask_mismatch");
        }
        bit(&f.mode, &mut lm);
        bit(&f.from, &mut lf);
        bit(&f.to_of, &mut lto);
        bit(&f.promo_of, &mut lpr);
        bit(&f.drop_piece, &mut ldp);
        bit(&f.drop_to_of, &mut ldt);
        dropped.push(d);
        from.push(fi);
        to.push(ti);
        promo.push(pr);
        piece_idx.push(pi);
        drop_to.push(dt);
        value.push(result_value(result, mb));
        b.push(m);
    }
    let names: Vec<Value> = p.names.iter().map(|n| n.clone().map(Value::from).unwrap_or(Value::Null)).collect();
    Ok(Sample::new(json!({"players": names, "result": result, "plies": t}))
        .bits("obs.board_seq", &[28, 9, 9], &board)
        .f32("obs.hand_seq", &[14, 4], hand)
        .f32("obs.globals_seq", &[GLOBAL_FEATURES], glb)
        .i64("tgt.dropped", &[], dropped)
        .i64("tgt.from_idx", &[], from)
        .i64("tgt.to_idx", &[], to)
        .bool("tgt.promoted", &[], promo)
        .i64("tgt.piece_idx", &[], piece_idx)
        .i64("tgt.drop_to_idx", &[], drop_to)
        .f32("tgt.value", &[], value)
        .bits("tgt.legal_mode", &[2], &lm)
        .bits("tgt.legal_from", &[81], &lf)
        .bits("tgt.legal_to_of", &[81], &lto)
        .bits("tgt.legal_promo_of", &[2], &lpr)
        .bits("tgt.legal_drop_piece", &[14], &ldp)
        .bits("tgt.legal_drop_to_of", &[81], &ldt))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download floodgate CSA records: one day (`--day 2025/06/01`, individual files)
    /// or a whole year (`--year 2023`, the 7z archive, extracted)
    Fetch {
        #[arg(long)]
        day: Option<String>,
        #[arg(long)]
        year: Option<u32>,
        #[arg(long, default_value = "data/shogi/floodgate")]
        out: PathBuf,
        /// also save the rating table of this date (YYYYMMDD) as <out>/ratings-<date>.json
        #[arg(long)]
        ratings: Option<String>,
    },
    /// Convert .csa files (recursive) to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 30)]
        min_moves: usize,
        #[arg(long)]
        allow_no_result: bool,
        /// ratings JSON written by `fetch --ratings`
        #[arg(long)]
        ratings: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        /// e.g. floodgate-300-10F (drops test channels such as floodgatea-*)
        #[arg(long)]
        game_type: Option<String>,
        #[arg(long)]
        no_time_up: bool,
        /// high-quality preset: both engines >= 3500 (needs --ratings), floodgate-300-10F,
        /// no time-up results, decisive games
        #[arg(long)]
        hq: bool,
    },
}

const FLOODGATE: &str = "https://wdoor.c.u-tokyo.ac.jp/shogi";

/// Parse a floodgate rating page (players-floodgate-YYYYMMDD.html) → name → rating
/// (max over accounts sharing a display name).
pub fn parse_ratings(html: &str) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    let mut rest = html;
    while let Some(i) = rest.find("<td class=\"name\">") {
        rest = &rest[i + 17..];
        let Some(a) = rest.find("\">") else { break };
        let Some(e) = rest[a + 2..].find("</a>") else { break };
        let name = rest[a + 2..a + 2 + e].trim().to_string();
        let Some(r) = rest.find("<td class=\"rate\">") else { break };
        let after = &rest[r..];
        let Some(sp) = after.find("\">") .and_then(|x| after[x + 2..].find('>').map(|y| x + 2 + y + 1)) else { break };
        let val: String = after[sp..].chars().take_while(|c| *c != '<').collect();
        if let Ok(v) = val.trim().parse::<i64>() {
            let e = out.entry(name).or_insert(v);
            *e = (*e).max(v);
        }
        rest = &rest[r..];
    }
    out
}

/// Quality filters (off by default): both engines rated ≥ `min_rating` on the given rating
/// table, main floodgate channel only, time-up results dropped.
#[derive(Clone, Default)]
pub struct Quality {
    pub ratings: Option<BTreeMap<String, i64>>,
    pub min_rating: i64,
    pub game_type: Option<String>,
    pub no_time_up: bool,
}

fn passes_quality(text: &str, file_name: &str, q: &Quality) -> bool {
    if let Some(gt) = &q.game_type {
        if !file_name.contains(&format!("+{gt}+")) {
            return false;
        }
    }
    if q.no_time_up && text.lines().any(|l| l.trim_start().starts_with("%TIME_UP")) {
        return false;
    }
    if let Some(r) = &q.ratings {
        for tag in ["N+", "N-"] {
            let name = text.lines().find_map(|l| l.strip_prefix(tag)).map(|n| n.trim_end_matches('\r'));
            match name.and_then(|n| r.get(n)) {
                Some(&v) if v >= q.min_rating => {}
                _ => return false,
            }
        }
    }
    true
}

fn fetch(day: Option<String>, year: Option<u32>, ratings: Option<String>, out: &Path) -> Result<()> {
    let client = common::http()?;
    if let Some(date) = ratings {
        let html = client.get(format!("{FLOODGATE}/x/rating/players-floodgate-{date}.html")).send()?.error_for_status()?.text()?;
        let table = parse_ratings(&html);
        anyhow::ensure!(!table.is_empty(), "no ratings parsed for {date}");
        std::fs::create_dir_all(out)?;
        std::fs::write(out.join(format!("ratings-{date}.json")), serde_json::to_vec_pretty(&table)?)?;
        eprintln!("{} rated engines saved", table.len());
    }
    if let Some(day) = day {
        let dir = format!("{FLOODGATE}/x/{}/", day.trim_matches('/'));
        let html = client.get(&dir).send()?.error_for_status()?.text()?;
        let mut names: Vec<String> = html.split("href=\"").skip(1).filter_map(|s| s.split('"').next()).filter(|h| h.ends_with(".csa")).map(String::from).collect();
        names.sort();
        names.dedup();
        let dest = out.join(day.trim_matches('/'));
        eprintln!("{} games listed for {day}", names.len());
        for n in names {
            common::download(&client, &format!("{dir}{n}"), &dest.join(&n))?;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    if let Some(y) = year {
        let archive = out.join(format!("wdoor{y}.7z"));
        anyhow::ensure!(common::download(&client, &format!("{FLOODGATE}/archive/wdoor{y}.7z"), &archive)?, "no archive for {y}");
        sevenz_rust::decompress_file(&archive, out.join(y.to_string()))?;
        std::fs::remove_file(&archive)?;
    }
    Ok(())
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, min_moves: usize, allow_no_result: bool,
               q: &Quality) -> Result<Value> {
    let files: Vec<PathBuf> = common::find_files(input, &[".csa"])?.into_iter().filter(|p| p.extension().map(|e| e == "csa").unwrap_or(false)).collect();
    anyhow::ensure!(!files.is_empty(), "no .csa files under {}", input.display());
    let mut w = ShardWriter::new(out, "shogi", CONVERTER_VERSION, json!({"global_features": GLOBAL_FEATURES}), shard_size)?;
    let mut stats: BTreeMap<&str, usize> =
        ["converted", "parse_error", "too_short", "no_result", "illegal_move", "mask_mismatch", "low_quality"].into_iter().map(|k| (k, 0)).collect();
    // Files are converted in parallel chunks and written in input order, so the output (and
    // the stats, max_games included) is identical to a sequential pass.
    const CHUNK: usize = 512;
    let mut i = 0;
    while i < files.len() {
        let left = max_games.map(|m| m.saturating_sub(stats["converted"]));
        if left == Some(0) {
            break;
        }
        let n = left.unwrap_or(CHUNK).clamp(1, CHUNK).min(files.len() - i);
        let chunk = &files[i..i + n];
        i += n;
        let results: Vec<Result<Option<std::result::Result<Sample, &'static str>>>> = chunk
            .par_iter()
            .map(|path| {
                let text = String::from_utf8_lossy(&std::fs::read(path)?).to_string();
                if !passes_quality(&text, &path.file_name().unwrap().to_string_lossy(), q) {
                    return Ok(None);
                }
                Ok(Some(convert_game(&text, min_moves, allow_no_result)))
            })
            .collect();
        for (path, r) in chunk.iter().zip(results) {
            if max_games.map(|m| stats["converted"] >= m).unwrap_or(false) {
                break;
            }
            match r? {
                None => *stats.get_mut("low_quality").unwrap() += 1,
                Some(Ok(mut s)) => {
                    s.meta["source"] = json!(path.file_name().unwrap().to_string_lossy());
                    w.push(s)?;
                    *stats.get_mut("converted").unwrap() += 1;
                }
                Some(Err(r)) => *stats.get_mut(r).unwrap() += 1,
            }
        }
    }
    w.flush()?;
    let mut v = json!(stats);
    v["shards"] = json!(w.written_shards);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { day, year, out, ratings } => fetch(day, year, ratings, &out),
        Cmd::Convert { input, out, shard_size, max_games, min_moves, allow_no_result, ratings, min_rating, game_type, no_time_up, hq } => {
            let table = match &ratings {
                Some(p) => Some(serde_json::from_slice::<BTreeMap<String, i64>>(&std::fs::read(p)?)?),
                None => None,
            };
            let mut q = Quality { ratings: table, min_rating, game_type, no_time_up };
            let mut allow_no_result = allow_no_result;
            if hq {
                anyhow::ensure!(q.ratings.is_some(), "--hq needs --ratings (run `gmdata shogi fetch --ratings YYYYMMDD`)");
                q.min_rating = q.min_rating.max(3500);
                q.game_type.get_or_insert_with(|| "floodgate-300-10F".to_string());
                q.no_time_up = true;
                allow_no_result = false;
            }
            println!("{}", convert(&input, &out, shard_size, max_games, min_moves, allow_no_result, &q)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrules::games::shogi::{col, rank, GOLD, KNIGHT, LANCE, PROM_PAWN, SILVER};

    fn start() -> Board {
        let text = "PI\n+\n";
        parse_csa(text).unwrap().start
    }

    #[test]
    fn opening_moves_and_legality() {
        let b = start();
        let m = Mv { from: csa_square("77"), to: csa_square("76").unwrap(), promotion: false, drop: NONE };
        assert!(b.is_legal(&m));
        let bad = Mv { from: csa_square("77"), to: csa_square("75").unwrap(), promotion: false, drop: NONE };
        assert!(!b.is_legal(&bad));
        assert!(!b.is_check());
        assert_eq!(game_result("%TORYO", BLACK, 1), "b"); // black moved last, white resigns
    }

    fn sq(s: &str) -> usize {
        csa_square(s).unwrap()
    }

    /// Board from (CSA square, piece, colour) triples, hand entries (colour, piece, count).
    fn board(pieces: &[(&str, u8, usize)], hands: &[(usize, u8, u32)], turn: usize) -> Board {
        let mut b = Board { piece: [NONE; 81], color: [0; 81], hand: [[0; 9]; 2], turn, king: [None, None], move_number: 1 };
        for &(s, pt, c) in pieces {
            b.piece[sq(s)] = pt;
            b.color[sq(s)] = c as u8;
            if pt == KING {
                b.king[c] = Some(sq(s));
            }
        }
        for &(c, pt, n) in hands {
            b.hand[c][pt as usize] = n;
        }
        b
    }

    fn drop_squares(moves: &[Mv], pt: u8) -> Vec<usize> {
        moves.iter().filter(|m| m.drop == pt).map(|m| m.to).collect()
    }

    fn mv(from: &str, to: &str) -> Mv {
        Mv { from: Some(sq(from)), to: sq(to), promotion: false, drop: NONE }
    }

    #[test]
    fn start_position_has_30_moves_and_no_drops() {
        let b = start();
        let moves = b.legal_moves();
        assert_eq!(moves.len(), 30);
        assert!(moves.iter().all(|m| b.is_legal(m) && m.drop == NONE && !m.promotion));
        let f = frame_masks(&moves, true, sq("77") as i64, sq("76") as i64, 0);
        assert_eq!(f.mode, [true, false]);
        assert_eq!(f.drop_piece, [false; 14]);
        assert_eq!(f.drop_to_of, [false; 81]);
        // movable: 9 pawns, 2 lances, 2 silvers, 2 golds, the king and the rook
        assert_eq!(f.from.iter().filter(|&&x| x).count(), 17);
        // the 7g pawn has exactly one destination (7f), without promotion
        let to: Vec<usize> = (0..81).filter(|&i| f.to_of[i]).collect();
        assert_eq!(to, vec![sq("76")]);
        assert_eq!(f.promo_of, [true, false]);
    }

    #[test]
    fn white_masks_are_in_white_pov() {
        // after 7g7f white's position is the rotated start position: same POV masks
        let mut b = start();
        let black = frame_masks(&b.legal_moves(), true, 0, 0, 0);
        b.push(&mv("77", "76"));
        let moves = b.legal_moves();
        assert_eq!(moves.len(), 30);
        let white = frame_masks(&moves, false, 0, 0, 0);
        assert_eq!(white.from, black.from);
        // POV 56 (= 3c for white) → POV 47
        let f = frame_masks(&moves, false, 56, 47, 0);
        assert!(f.to_of[47] && f.to_of.iter().filter(|&&x| x).count() == 1);
    }

    #[test]
    fn nifu_and_dead_drops() {
        let b = board(&[("51", KING, WHITE), ("59", KING, BLACK), ("77", PAWN, BLACK)],
                      &[(BLACK, PAWN, 1), (BLACK, LANCE, 1), (BLACK, KNIGHT, 1)], BLACK);
        let moves = b.legal_moves();
        let pawn = drop_squares(&moves, PAWN);
        assert!(pawn.iter().all(|&s| col(s) != col(sq("77"))), "nifu: no pawn drop on file 7");
        assert!(pawn.iter().all(|&s| rank(s) > 0), "no pawn drop on the last rank");
        assert!(pawn.contains(&sq("65")));
        // 81 − last rank (9, incl. the white king) − file 7 below it (8, incl. the pawn) − black king
        assert_eq!(pawn.len(), 81 - 9 - 8 - 1);
        let lance = drop_squares(&moves, LANCE);
        assert!(lance.iter().all(|&s| rank(s) > 0));
        assert_eq!(lance.len(), 81 - 9 - 2);
        let knight = drop_squares(&moves, KNIGHT);
        assert!(knight.iter().all(|&s| rank(s) > 1));
        assert_eq!(knight.len(), 81 - 18 - 2);
        // a promoted pawn on the file does not count for nifu
        let mut b2 = b.clone();
        b2.piece[sq("77")] = PROM_PAWN;
        assert!(drop_squares(&b2.legal_moves(), PAWN).contains(&sq("75")));
        // white: last ranks are 8/9 from black's view (rows 8 / 7..8)
        let w = board(&[("51", KING, WHITE), ("59", KING, BLACK)], &[(WHITE, PAWN, 1), (WHITE, KNIGHT, 1)], WHITE);
        let wm = w.legal_moves();
        assert!(drop_squares(&wm, PAWN).iter().all(|&s| rank(s) < 8));
        assert!(drop_squares(&wm, KNIGHT).iter().all(|&s| rank(s) < 7));
        let f = frame_masks(&wm, false, 0, 0, KNIGHT as i64 - 1);
        assert_eq!(f.mode, [true, true]);
        assert!(f.drop_piece[0] && f.drop_piece[2] && !f.drop_piece[1]);
        assert!(f.drop_piece[7..].iter().all(|&x| !x));
        // POV: white's knight drop rows are its own ranks 3..9 → POV rows 2..8
        assert!((0..81).filter(|&i| f.drop_to_of[i]).all(|i| i / 9 >= 2));
    }

    #[test]
    fn pawn_drop_mate_is_excluded() {
        // white king in the corner, own lance/pawn block 2a/2b, black gold on 1c guards 1b
        let pieces = [("11", KING, WHITE), ("21", LANCE, WHITE), ("22", PAWN, WHITE), ("13", GOLD, BLACK), ("59", KING, BLACK)];
        let b = board(&pieces, &[(BLACK, PAWN, 1), (BLACK, GOLD, 1)], BLACK);
        let moves = b.legal_moves();
        assert!(!drop_squares(&moves, PAWN).contains(&sq("12")), "uchifuzume");
        assert!(drop_squares(&moves, GOLD).contains(&sq("12")), "a gold-drop mate is legal");
        // without the guard the king can take the pawn: legal
        let b2 = board(&[pieces[0], pieces[1], pieces[2], pieces[4]], &[(BLACK, PAWN, 1)], BLACK);
        assert!(drop_squares(&b2.legal_moves(), PAWN).contains(&sq("12")));
    }

    #[test]
    fn forced_and_optional_promotion() {
        let pieces = [
            ("91", KING, WHITE),
            ("19", KING, BLACK),
            ("52", PAWN, BLACK),
            ("32", LANCE, BLACK),
            ("43", KNIGHT, BLACK),
            ("74", KNIGHT, BLACK),
            ("75", KNIGHT, BLACK),
            ("55", SILVER, BLACK),
            ("56", PAWN, BLACK),
        ];
        let b = board(&pieces, &[], BLACK);
        let moves = b.legal_moves();
        let promo = |from: &str, to: &str| frame_masks(&moves, true, sq(from) as i64, sq(to) as i64, 0).promo_of;
        assert_eq!(promo("52", "51"), [false, true]); // pawn to the last rank
        assert_eq!(promo("32", "31"), [false, true]); // lance to the last rank
        assert_eq!(promo("43", "51"), [false, true]); // knight to the last rank
        assert_eq!(promo("74", "82"), [false, true]); // knight to the second-last rank
        assert_eq!(promo("75", "83"), [true, true]); // knight into the zone: optional
        assert_eq!(promo("55", "54"), [true, false]); // outside the zone
        assert_eq!(promo("56", "55"), [false, false]); // own piece on the target: no move
        let f = frame_masks(&moves, true, sq("32") as i64, 0, 0);
        let lance_to: Vec<usize> = (0..81).filter(|&i| f.to_of[i]).collect();
        assert_eq!(lance_to, vec![sq("31")]);
        // the black king in the corner: all 3 neighbours are free and unattacked
        let k = frame_masks(&moves, true, sq("19") as i64, 0, 0);
        assert_eq!(k.to_of.iter().filter(|&&x| x).count(), 3);
    }

    #[test]
    fn self_check_is_excluded() {
        // black silver pinned on the file by a white rook: it may not leave the file
        let b = board(&[("59", KING, BLACK), ("58", SILVER, BLACK), ("51", ROOK, WHITE), ("91", KING, WHITE)], &[], BLACK);
        let moves = b.legal_moves();
        let f = frame_masks(&moves, true, sq("58") as i64, 0, 0);
        let to: Vec<usize> = (0..81).filter(|&i| f.to_of[i]).collect();
        assert_eq!(to, vec![sq("57")]);
    }

    #[test]
    fn repetition_counts_after_king_shuffle() {
        let shuffle = [mv("59", "58"), mv("51", "52"), mv("58", "59"), mv("52", "51")];
        let moves: Vec<Mv> = shuffle.iter().cycle().take(13).copied().collect();
        let reps = repetition_counts(&start(), &moves);
        assert_eq!(reps, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]);
        // same pieces, other side to move: a different position
        let mut b = start();
        let k0 = b.key();
        b.turn ^= 1;
        assert!(b.key() != k0);
    }

    #[test]
    fn promoted_or_king_in_hand_is_a_parse_error() {
        assert!(parse_csa("PI\nP+00TO\n+\n").is_err());
        assert!(parse_csa("PI\nP-00OU\n+\n").is_err());
        assert!(parse_csa("PI\nP+00KI\n+\n").is_ok());
    }

    #[test]
    fn convert_game_writes_globals_and_masks() {
        let text = "PI\n+\n+5958OU\n-5152OU\n+5859OU\n-5251OU\n+7776FU\n%TORYO\n";
        let s = convert_game(text, 0, false).unwrap();
        let g = s.tensors.iter().find(|t| t.name == "obs.globals_seq").unwrap();
        assert_eq!(g.frame_shape, vec![GLOBAL_FEATURES]);
        for name in ["tgt.legal_mode", "tgt.legal_from", "tgt.legal_to_of", "tgt.legal_promo_of", "tgt.legal_drop_piece", "tgt.legal_drop_to_of"] {
            assert!(s.tensors.iter().any(|t| t.name == name), "{name}");
        }
    }
}
