//! Shogi: floodgate CSA records → shards for the shogi adapter.
//!
//! Frame t = position BEFORE move t from the side to move (board rotated 180°
//! for white). Square index = row-major 9×9 with square 0 = 9a (gote's back rank);
//! POV index = sq for black, 80 − sq for white.
//!
//!   obs.board_seq   [T,28,9,9] f32  own P L N S G B R K +P +L +N +S +B +R, then opponent
//!   obs.hand_seq    [T,14,4] f32    own P..R then opponent: [count, count>0, min(count/4,1), 1]
//!   obs.globals_seq [T,4] f32       [min(move_number/512,1), mover is black, in check, 1]
//!   tgt.dropped [T] i64, tgt.from_idx, tgt.to_idx, tgt.piece_idx, tgt.drop_to_idx [T] i64,
//!   tgt.promoted [T] bool, tgt.value [T] f32 (result for the mover)
//!
//! The rules engine reproduces the reference (python-shogi) semantics exactly,
//! including its legality test: pseudo-legal ∧ ¬(own king attacked after the move)
//! ∧ ¬(pawn-drop mate as approximated there: the king has no unattacked escape
//! square — occupancy unchanged — and no non-king piece attacks the pawn).

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const BLACK: usize = 0;
const WHITE: usize = 1;
const NONE: u8 = 0;
const PAWN: u8 = 1;
const LANCE: u8 = 2;
const KNIGHT: u8 = 3;
const SILVER: u8 = 4;
const GOLD: u8 = 5;
const BISHOP: u8 = 6;
const ROOK: u8 = 7;
const KING: u8 = 8;
const PROM_PAWN: u8 = 9;
const PROM_BISHOP: u8 = 13;
const PROM_ROOK: u8 = 14;

fn promoted(p: u8) -> u8 {
    match p {
        1..=4 => p + 8,
        6 => 13,
        7 => 14,
        _ => 0,
    }
}

fn demoted(p: u8) -> u8 {
    match p {
        9..=12 => p - 8,
        13 => 6,
        14 => 7,
        _ => p,
    }
}

fn rank(sq: usize) -> i32 {
    (sq / 9) as i32
}
fn col(sq: usize) -> i32 {
    (sq % 9) as i32
}
fn sq_of(r: i32, c: i32) -> Option<usize> {
    ((0..9).contains(&r) && (0..9).contains(&c)).then(|| (r * 9 + c) as usize)
}

#[derive(Clone)]
struct Board {
    piece: [u8; 81],
    color: [u8; 81], // owner, valid when piece != NONE
    hand: [[u32; 9]; 2],
    turn: usize,
    king: [Option<usize>; 2],
    move_number: u32,
}

#[derive(Clone, Copy, Debug)]
struct Mv {
    from: Option<usize>,
    to: usize,
    promotion: bool,
    drop: u8,
}

impl Board {
    fn occupied(&self, sq: usize) -> bool {
        self.piece[sq] != NONE
    }
    fn owned_by(&self, sq: usize, c: usize) -> bool {
        self.piece[sq] != NONE && self.color[sq] as usize == c
    }

    /// Squares attacked by a `pt` on `sq` moving as `mover` (reference `attacks_from`).
    fn attacks_from(&self, pt: u8, sq: usize, mover: usize) -> Vec<usize> {
        let fwd = if mover == BLACK { -1 } else { 1 };
        let (r, c) = (rank(sq), col(sq));
        let steps = |ds: &[(i32, i32)]| -> Vec<usize> { ds.iter().filter_map(|&(dr, dc)| sq_of(r + dr, c + dc)).collect() };
        let slide = |ds: &[(i32, i32)]| -> Vec<usize> {
            let mut out = Vec::new();
            for &(dr, dc) in ds {
                let (mut rr, mut cc) = (r + dr, c + dc);
                while let Some(s) = sq_of(rr, cc) {
                    out.push(s);
                    if self.occupied(s) {
                        break;
                    }
                    rr += dr;
                    cc += dc;
                }
            }
            out
        };
        let gold = [(fwd, -1), (fwd, 0), (fwd, 1), (0, -1), (0, 1), (-fwd, 0)];
        let king8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
        let diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
        let orth = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        match pt {
            PAWN => steps(&[(fwd, 0)]),
            LANCE => slide(&[(fwd, 0)]),
            KNIGHT => steps(&[(2 * fwd, -1), (2 * fwd, 1)]),
            SILVER => steps(&[(fwd, -1), (fwd, 0), (fwd, 1), (-fwd, -1), (-fwd, 1)]),
            GOLD | 9..=12 => steps(&gold),
            BISHOP => slide(&diag),
            ROOK => slide(&orth),
            KING => steps(&king8),
            PROM_BISHOP => {
                let mut v = steps(&king8);
                v.extend(slide(&diag));
                v
            }
            PROM_ROOK => {
                let mut v = steps(&king8);
                v.extend(slide(&orth));
                v
            }
            _ => Vec::new(),
        }
    }

    fn is_attacked_by(&self, color: usize, sq: Option<usize>, skip_king: bool) -> bool {
        let Some(sq) = sq else { return false };
        for pt in 1..=14u8 {
            if skip_king && pt == KING {
                continue;
            }
            for s in self.attacks_from(pt, sq, color ^ 1) {
                if self.piece[s] == pt && self.color[s] as usize == color {
                    return true;
                }
            }
        }
        false
    }

    fn is_check(&self) -> bool {
        self.is_attacked_by(self.turn ^ 1, self.king[self.turn], false)
    }

    fn can_move_without_promotion(to: usize, pt: u8, color: usize) -> bool {
        let r = rank(to);
        if color == BLACK {
            !matches!(pt, PAWN | LANCE | KNIGHT) || (pt == PAWN && r > 0) || (pt == LANCE && r > 0) || (pt == KNIGHT && r > 1)
        } else {
            !matches!(pt, PAWN | LANCE | KNIGHT) || (pt == PAWN && r < 8) || (pt == LANCE && r < 8) || (pt == KNIGHT && r < 7)
        }
    }

    fn is_double_pawn(&self, to: usize, pt: u8) -> bool {
        pt == PAWN && (0..9).any(|r| {
            let s = (r * 9) as usize + col(to) as usize;
            self.piece[s] == PAWN && self.color[s] as usize == self.turn
        })
    }

    fn is_pseudo_legal(&self, m: &Mv) -> bool {
        if self.owned_by(m.to, self.turn) {
            return false;
        }
        if let Some(from) = m.from {
            let pt = self.piece[from];
            if pt == NONE || !self.owned_by(from, self.turn) {
                return false;
            }
            if m.promotion {
                if pt == GOLD || pt == KING || pt >= PROM_PAWN {
                    return false;
                }
                if self.turn == BLACK && rank(m.to) > 2 && rank(from) > 2 {
                    return false;
                }
                if self.turn == WHITE && rank(m.to) < 6 && rank(from) < 6 {
                    return false;
                }
            }
            if !m.promotion && !Self::can_move_without_promotion(m.to, pt, self.turn) {
                return false;
            }
            self.attacks_from(pt, from, self.turn).contains(&m.to)
        } else if m.drop != NONE {
            if m.promotion || self.hand[self.turn][m.drop as usize] == 0 {
                return false;
            }
            if !Self::can_move_without_promotion(m.to, m.drop, self.turn) || self.is_double_pawn(m.to, m.drop) {
                return false;
            }
            // reference drops onto an occupied square are pseudo-legal only if not own-occupied
            true
        } else {
            false
        }
    }

    fn push(&mut self, m: &Mv) {
        self.move_number += 1;
        let mover = self.turn;
        let pt = if m.drop != NONE {
            self.hand[mover][m.drop as usize] -= 1;
            m.drop
        } else {
            let from = m.from.unwrap();
            let mut pt = self.piece[from];
            if m.promotion {
                pt = promoted(pt);
            }
            self.piece[from] = NONE;
            pt
        };
        // capture into the mover's hand (demoted)
        let captured = self.piece[m.to];
        if captured != NONE {
            self.hand[mover][demoted(captured) as usize] += 1;
        }
        self.piece[m.to] = pt;
        self.color[m.to] = mover as u8;
        if pt == KING {
            self.king[mover] = Some(m.to);
        }
        self.turn ^= 1;
    }

    fn was_check_by_dropping_pawn(&self, m: &Mv) -> bool {
        if m.drop != PAWN {
            return false;
        }
        let pawn_sq = m.to;
        let Some(ksq) = self.king[self.turn] else { return false };
        // pawn (dropper = turn^1) attacks the king square, and that square is not dropper-occupied
        let pawn_attacks: Vec<usize> =
            self.attacks_from(PAWN, pawn_sq, self.turn ^ 1).into_iter().filter(|&s| !self.owned_by(s, self.turn ^ 1)).collect();
        if !pawn_attacks.contains(&ksq) {
            return false;
        }
        for s in self.attacks_from(KING, ksq, self.turn) {
            if self.owned_by(s, self.turn) {
                continue;
            }
            if !self.is_attacked_by(self.turn ^ 1, Some(s), false) {
                return false;
            }
        }
        if self.is_attacked_by(self.turn, Some(pawn_sq), true) {
            return false;
        }
        true
    }

    fn is_legal(&self, m: &Mv) -> bool {
        if !self.is_pseudo_legal(m) {
            return false;
        }
        let mut b = self.clone();
        b.push(m);
        let suicide = b.is_attacked_by(b.turn, b.king[b.turn ^ 1], false);
        !(suicide || b.was_check_by_dropping_pawn(m))
    }
}

// ---------------------------------------------------------------- CSA

const CSA_PIECES: [&str; 15] = ["* ", "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU", "TO", "NY", "NK", "NG", "UM", "RY"];

fn csa_piece(s: &str) -> Option<u8> {
    CSA_PIECES.iter().position(|&p| p == s).map(|i| i as u8)
}

/// CSA "91" (file 9, rank 1) → square index.
fn csa_square(s: &str) -> Option<usize> {
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

fn hirate() -> [Option<(u8, usize)>; 81] {
    let mut p = [None; 81];
    let back = [LANCE, KNIGHT, SILVER, GOLD, KING, GOLD, SILVER, KNIGHT, LANCE];
    for c in 0..9 {
        p[c] = Some((back[c], WHITE));
        p[18 + c] = Some((PAWN, WHITE));
        p[54 + c] = Some((PAWN, BLACK));
        p[72 + c] = Some((back[c], BLACK));
    }
    p[10] = Some((ROOK, WHITE));
    p[16] = Some((BISHOP, WHITE));
    p[64] = Some((BISHOP, BLACK));
    p[70] = Some((ROOK, BLACK));
    p
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

fn result_value(result: &str, mover_black: bool) -> f32 {
    match result {
        "b" => if mover_black { 1.0 } else { -1.0 },
        "w" => if mover_black { -1.0 } else { 1.0 },
        _ => 0.0,
    }
}

fn pov(sq: usize, mover_black: bool) -> i64 {
    if mover_black {
        sq as i64
    } else {
        80 - sq as i64
    }
}

fn convert_game(text: &str, min_moves: usize, allow_no_result: bool) -> std::result::Result<Sample, &'static str> {
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
    let mut board = vec![0f32; t * 28 * 81];
    let mut hand = Vec::with_capacity(t * 56);
    let mut glb = Vec::with_capacity(t * 4);
    let (mut dropped, mut from, mut to, mut promo, mut piece_idx, mut drop_to, mut value) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for (k, m) in p.moves.iter().enumerate() {
        if !b.is_legal(m) {
            return Err("illegal_move");
        }
        let mover = b.turn;
        let mb = mover == BLACK;
        let base = k * 28 * 81;
        for sq in 0..81 {
            let pt = b.piece[sq];
            if pt == NONE {
                continue;
            }
            let plane = (pt - 1) as usize + if b.color[sq] as usize == mover { 0 } else { 14 };
            board[base + plane * 81 + pov(sq, mb) as usize] = 1.0;
        }
        for c in [mover, mover ^ 1] {
            for pt in 1..8usize {
                let n = b.hand[c][pt] as f32;
                hand.extend_from_slice(&[n, (n > 0.0) as u8 as f32, (n as f64 / 4.0).min(1.0) as f32, 1.0]);
            }
        }
        glb.extend_from_slice(&[(b.move_number as f64 / 512.0).min(1.0) as f32, mb as u8 as f32, b.is_check() as u8 as f32, 1.0]);
        if m.drop != NONE {
            dropped.push(1);
            from.push(0);
            to.push(0);
            promo.push(false);
            piece_idx.push(m.drop as i64 - 1);
            drop_to.push(pov(m.to, mb));
        } else {
            dropped.push(0);
            from.push(pov(m.from.unwrap(), mb));
            to.push(pov(m.to, mb));
            promo.push(m.promotion);
            piece_idx.push(0);
            drop_to.push(0);
        }
        value.push(result_value(result, mb));
        b.push(m);
    }
    let names: Vec<Value> = p.names.iter().map(|n| n.clone().map(Value::from).unwrap_or(Value::Null)).collect();
    Ok(Sample::new(json!({"players": names, "result": result, "plies": t}))
        .bits("obs.board_seq", &[28, 9, 9], &board)
        .f32("obs.hand_seq", &[14, 4], hand)
        .f32("obs.globals_seq", &[4], glb)
        .i64("tgt.dropped", &[], dropped)
        .i64("tgt.from_idx", &[], from)
        .i64("tgt.to_idx", &[], to)
        .bool("tgt.promoted", &[], promo)
        .i64("tgt.piece_idx", &[], piece_idx)
        .i64("tgt.drop_to_idx", &[], drop_to)
        .f32("tgt.value", &[], value))
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
    },
}

const FLOODGATE: &str = "https://wdoor.c.u-tokyo.ac.jp/shogi";

fn fetch(day: Option<String>, year: Option<u32>, out: &Path) -> Result<()> {
    let client = common::http()?;
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

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, min_moves: usize, allow_no_result: bool) -> Result<Value> {
    let files: Vec<PathBuf> = common::find_files(input, &[".csa"])?.into_iter().filter(|p| p.extension().map(|e| e == "csa").unwrap_or(false)).collect();
    anyhow::ensure!(!files.is_empty(), "no .csa files under {}", input.display());
    let mut w = ShardWriter::new(out, "shogi", 1, json!({}), shard_size)?;
    let mut stats: BTreeMap<&str, usize> = ["converted", "parse_error", "too_short", "no_result", "illegal_move"].into_iter().map(|k| (k, 0)).collect();
    for path in files {
        if max_games.map(|m| stats["converted"] >= m).unwrap_or(false) {
            break;
        }
        let text = String::from_utf8_lossy(&std::fs::read(&path)?).to_string();
        match convert_game(&text, min_moves, allow_no_result) {
            Ok(mut s) => {
                s.meta["source"] = json!(path.file_name().unwrap().to_string_lossy());
                w.push(s)?;
                *stats.get_mut("converted").unwrap() += 1;
            }
            Err(r) => *stats.get_mut(r).unwrap() += 1,
        }
    }
    w.flush()?;
    let mut v = json!(stats);
    v["shards"] = json!(w.written_shards);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { day, year, out } => fetch(day, year, &out),
        Cmd::Convert { input, out, shard_size, max_games, min_moves, allow_no_result } => {
            println!("{}", convert(&input, &out, shard_size, max_games, min_moves, allow_no_result)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
