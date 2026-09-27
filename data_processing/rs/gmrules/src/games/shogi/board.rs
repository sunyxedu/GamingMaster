//! Shogi rules: the board, move generation and legality — ONE implementation, used by the
//! `gmdata shogi` converter (floodgate CSA records → IL frames), its engine labeller
//! (`gmdata label shogi`) and the simulator ([`super::Shogi`], gmsim).
//!
//! The engine reproduces the python-shogi semantics the converter was pinned to (its
//! legality test: pseudo-legal ∧ ¬(own king attacked after the move) ∧ ¬(pawn-drop mate
//! as python-shogi approximates it, see [`Board::is_legal`])). The move generator
//! ([`Board::legal_moves`]) applies that test to every candidate move; drops go onto
//! empty squares only (python-shogi's `is_pseudo_legal` also accepts a drop onto an
//! opponent piece, which its own generator never produces and real records never
//! contain).
//!
//! Squares: row-major 9×9, square 0 = 9a (gote's back rank, file 9), square 80 = 1i;
//! `rank(sq) = sq / 9` (0 = rank a), `col(sq) = sq % 9` (0 = file 9). Black (sente,
//! seat 0) moves up (towards rank a), White (gote, seat 1) down.
//!
//! Piece types: 1 P, 2 L, 3 N, 4 S, 5 G, 6 B, 7 R, 8 K, 9 +P, 10 +L, 11 +N, 12 +S, 13 +B
//! (horse), 14 +R (dragon); hands are indexed by the unpromoted type (1..=7).

use std::fmt::Write as _;

pub const BLACK: usize = 0;
pub const WHITE: usize = 1;
pub const NONE: u8 = 0;
pub const PAWN: u8 = 1;
pub const LANCE: u8 = 2;
pub const KNIGHT: u8 = 3;
pub const SILVER: u8 = 4;
pub const GOLD: u8 = 5;
pub const BISHOP: u8 = 6;
pub const ROOK: u8 = 7;
pub const KING: u8 = 8;
pub const PROM_PAWN: u8 = 9;
pub const PROM_LANCE: u8 = 10;
pub const PROM_KNIGHT: u8 = 11;
pub const PROM_SILVER: u8 = 12;
pub const PROM_BISHOP: u8 = 13;
pub const PROM_ROOK: u8 = 14;

/// Board squares.
pub const SQUARES: usize = 81;

/// The promoted type of `p` (0 when `p` cannot promote).
pub fn promoted(p: u8) -> u8 {
    match p {
        1..=4 => p + 8,
        6 => 13,
        7 => 14,
        _ => 0,
    }
}

/// The unpromoted type of `p` (what a capture puts into the hand).
pub fn demoted(p: u8) -> u8 {
    match p {
        9..=12 => p - 8,
        13 => 6,
        14 => 7,
        _ => p,
    }
}

/// Row of a square (0 = rank a, gote's back rank).
pub fn rank(sq: usize) -> i32 {
    (sq / 9) as i32
}

/// Column of a square (0 = file 9).
pub fn col(sq: usize) -> i32 {
    (sq % 9) as i32
}

fn sq_of(r: i32, c: i32) -> Option<usize> {
    ((0..9).contains(&r) && (0..9).contains(&c)).then(|| (r * 9 + c) as usize)
}

/// Square index in the mover's point of view: the square for black, the 180°-rotated one
/// (80 − sq) for white (the converter's POV; an involution).
pub fn pov(sq: usize, mover_black: bool) -> i64 {
    if mover_black {
        sq as i64
    } else {
        80 - sq as i64
    }
}

/// Fixed-capacity square list: no heap allocation in the attack / legality hot path (the move
/// generator runs the legality test on every candidate of every frame). The longest attack
/// set is a horse/dragon in the open: 8 king steps + 16 slides.
#[derive(Clone, Copy)]
pub struct Sqs {
    buf: [u8; 32],
    n: usize,
}

impl Sqs {
    fn new() -> Self {
        Sqs { buf: [0; 32], n: 0 }
    }
    fn push(&mut self, s: usize) {
        self.buf[self.n] = s as u8;
        self.n += 1;
    }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.buf[..self.n].iter().map(|&s| s as usize)
    }
    pub fn contains(&self, s: usize) -> bool {
        self.buf[..self.n].contains(&(s as u8))
    }
}

/// A position: pieces with owners, both hands, the side to move and the move number (1 at
/// the start of a record, +1 per ply; the frames' `globals_seq[0]` reads it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub piece: [u8; 81],
    /// owner, valid when piece != NONE
    pub color: [u8; 81],
    /// `hand[colour][piece type]`, types 1..=7 used
    pub hand: [[u32; 9]; 2],
    pub turn: usize,
    pub king: [Option<usize>; 2],
    pub move_number: u32,
}

/// A move: a board move (`from` = Some, `drop` = NONE) or a drop (`from` = None, `drop` = the
/// piece type 1..=7, `promotion` false). Absolute squares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mv {
    pub from: Option<usize>,
    pub to: usize,
    pub promotion: bool,
    pub drop: u8,
}

/// Position identity for the repetition counts: pieces with owners (owner 0 on empty
/// squares), hands (P..R) and the side to move.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PosKey {
    piece: [u8; 81],
    color: [u8; 81],
    hand: [[u32; 7]; 2],
    turn: u8,
}

/// The hirate (even game) placement: `(piece type, owner)` per square.
pub fn hirate() -> [Option<(u8, usize)>; 81] {
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

impl Board {
    /// An empty board, black to move, move 1.
    pub fn empty() -> Board {
        Board { piece: [NONE; 81], color: [0; 81], hand: [[0; 9]; 2], turn: BLACK, king: [None, None], move_number: 1 }
    }

    /// The hirate start position (black to move, move 1).
    pub fn hirate() -> Board {
        let mut b = Board::empty();
        for (sq, p) in hirate().iter().enumerate() {
            if let Some((pt, c)) = *p {
                b.put(sq, pt, c);
            }
        }
        b
    }

    /// Place a piece (`pt` != NONE) of colour `c` on `sq` (the king square is tracked).
    pub fn put(&mut self, sq: usize, pt: u8, c: usize) {
        self.piece[sq] = pt;
        self.color[sq] = c as u8;
        if pt == KING {
            self.king[c] = Some(sq);
        }
    }

    pub fn occupied(&self, sq: usize) -> bool {
        self.piece[sq] != NONE
    }

    pub fn owned_by(&self, sq: usize, c: usize) -> bool {
        self.piece[sq] != NONE && self.color[sq] as usize == c
    }

    /// Repetition identity (sennichite compares pieces with owners, both hands and the side
    /// to move; the move number is not part of it).
    pub fn key(&self) -> PosKey {
        let mut color = [0u8; 81];
        for (sq, c) in color.iter_mut().enumerate() {
            if self.occupied(sq) {
                *c = self.color[sq];
            }
        }
        let mut hand = [[0u32; 7]; 2];
        for (h, full) in hand.iter_mut().zip(self.hand.iter()) {
            h.copy_from_slice(&full[1..8]);
        }
        PosKey { piece: self.piece, color, hand, turn: self.turn as u8 }
    }

    /// Squares attacked by a `pt` on `sq` moving as `mover` (reference `attacks_from`;
    /// a horse/dragon lists its diagonal/orthogonal neighbours twice, as there).
    pub fn attacks_from(&self, pt: u8, sq: usize, mover: usize) -> Sqs {
        let fwd = if mover == BLACK { -1 } else { 1 };
        let (r, c) = (rank(sq), col(sq));
        let steps = |out: &mut Sqs, ds: &[(i32, i32)]| {
            for &(dr, dc) in ds {
                if let Some(s) = sq_of(r + dr, c + dc) {
                    out.push(s);
                }
            }
        };
        let slide = |out: &mut Sqs, ds: &[(i32, i32)]| {
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
        };
        let gold = [(fwd, -1), (fwd, 0), (fwd, 1), (0, -1), (0, 1), (-fwd, 0)];
        let king8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
        let diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
        let orth = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let mut out = Sqs::new();
        match pt {
            PAWN => steps(&mut out, &[(fwd, 0)]),
            LANCE => slide(&mut out, &[(fwd, 0)]),
            KNIGHT => steps(&mut out, &[(2 * fwd, -1), (2 * fwd, 1)]),
            SILVER => steps(&mut out, &[(fwd, -1), (fwd, 0), (fwd, 1), (-fwd, -1), (-fwd, 1)]),
            GOLD | 9..=12 => steps(&mut out, &gold),
            BISHOP => slide(&mut out, &diag),
            ROOK => slide(&mut out, &orth),
            KING => steps(&mut out, &king8),
            PROM_BISHOP => {
                steps(&mut out, &king8);
                slide(&mut out, &diag);
            }
            PROM_ROOK => {
                steps(&mut out, &king8);
                slide(&mut out, &orth);
            }
            _ => {}
        }
        out
    }

    /// Some piece of `color` attacks `sq` (`skip_king`: kings do not count).
    pub fn is_attacked_by(&self, color: usize, sq: Option<usize>, skip_king: bool) -> bool {
        let Some(sq) = sq else { return false };
        for pt in 1..=14u8 {
            if skip_king && pt == KING {
                continue;
            }
            for s in self.attacks_from(pt, sq, color ^ 1).iter() {
                if self.piece[s] == pt && self.color[s] as usize == color {
                    return true;
                }
            }
        }
        false
    }

    /// The side to move is in check.
    pub fn is_check(&self) -> bool {
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
            self.attacks_from(pt, from, self.turn).contains(m.to)
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

    /// Play `m` (no legality check): the capture goes into the mover's hand, demoted.
    pub fn push(&mut self, m: &Mv) {
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

    /// python-shogi's pawn-drop-mate test, on the position AFTER the drop `m`: the pawn
    /// checks the king, every king-step square not held by the defender is attacked by the
    /// dropper — in the current occupancy, the king still on its square — and no non-king
    /// defender attacks the pawn. An approximation of mate that never rejects a legal drop
    /// (every condition implies the exact one) but misses two kinds of true pawn-drop mates:
    /// an escape square attacked only THROUGH the king's square (a slider x-raying the king)
    /// and a pawn attacked only by pinned defenders.
    fn was_check_by_dropping_pawn(&self, m: &Mv) -> bool {
        if m.drop != PAWN {
            return false;
        }
        let pawn_sq = m.to;
        let Some(ksq) = self.king[self.turn] else { return false };
        // pawn (dropper = turn^1) attacks the king square, and that square is not dropper-occupied
        let pawn_checks = self.attacks_from(PAWN, pawn_sq, self.turn ^ 1).iter().any(|s| s == ksq && !self.owned_by(s, self.turn ^ 1));
        if !pawn_checks {
            return false;
        }
        for s in self.attacks_from(KING, ksq, self.turn).iter() {
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

    /// The converter's (python-shogi's) legality test: pseudo-legal (piece movement, own
    /// pieces, the promotion zone, forced promotion of pieces that could not move again,
    /// nifu, drops of pieces in hand onto squares where they can move again), no own king
    /// in check after the move, no pawn-drop mate (python-shogi's approximation, see
    /// [`Self::was_check_by_dropping_pawn`]).
    pub fn is_legal(&self, m: &Mv) -> bool {
        if !self.is_pseudo_legal(m) {
            return false;
        }
        let mut b = self.clone();
        b.push(m);
        let suicide = b.is_attacked_by(b.turn, b.king[b.turn ^ 1], false);
        !(suicide || b.was_check_by_dropping_pawn(m))
    }

    /// Every legal move of the side to move: board moves (each own piece to each attacked
    /// square not held by an own piece, with and without promotion — `is_pseudo_legal` owns
    /// the promotion-zone and forced-promotion rules) and drops of each hand piece onto each
    /// EMPTY square, all filtered by [`Board::is_legal`] (self-check, nifu, dead drops,
    /// pawn-drop mate). Deterministic order, no duplicates.
    pub fn legal_moves(&self) -> Vec<Mv> {
        let t = self.turn;
        let mut out = Vec::new();
        for from in 0..81 {
            if !self.owned_by(from, t) {
                continue;
            }
            let mut seen = [false; 81];
            for to in self.attacks_from(self.piece[from], from, t).iter() {
                if seen[to] || self.owned_by(to, t) {
                    continue;
                }
                seen[to] = true;
                for promotion in [false, true] {
                    let m = Mv { from: Some(from), to, promotion, drop: NONE };
                    if self.is_legal(&m) {
                        out.push(m);
                    }
                }
            }
        }
        for pt in PAWN..=ROOK {
            if self.hand[t][pt as usize] == 0 {
                continue;
            }
            for to in 0..81 {
                if self.occupied(to) {
                    continue;
                }
                let m = Mv { from: None, to, promotion: false, drop: pt };
                if self.is_legal(&m) {
                    out.push(m);
                }
            }
        }
        out
    }

    /// CSA rule 27 entering-king declaration (入玉宣言, "27-point rule", the floodgate /
    /// shogi-server rule): the side to move may declare a win when its king stands in the
    /// enemy camp (the three farthest ranks) and is not in check, it has at least 10 other
    /// pieces in the enemy camp, and those pieces plus its pieces in hand are worth at
    /// least 28 points for black / 27 for white (rook, bishop and their promotions 5, every
    /// other piece 1).
    pub fn can_declare(&self) -> bool {
        let c = self.turn;
        let Some(k) = self.king[c] else { return false };
        let in_camp = |sq: usize| if c == BLACK { rank(sq) <= 2 } else { rank(sq) >= 6 };
        if !in_camp(k) || self.is_check() {
            return false;
        }
        let points = |pt: u8| if matches!(pt, BISHOP | ROOK | PROM_BISHOP | PROM_ROOK) { 5 } else { 1 };
        let (mut n, mut total) = (0u32, 0u32);
        for sq in 0..81 {
            if sq != k && self.owned_by(sq, c) && in_camp(sq) {
                n += 1;
                total += points(self.piece[sq]);
            }
        }
        for pt in PAWN..=ROOK {
            total += self.hand[c][pt as usize] * points(pt);
        }
        n >= 10 && total >= if c == BLACK { 28 } else { 27 }
    }
}

/// Repetition counts: for each frame, how often the position before that move occurred
/// earlier in the game (the start position included).
pub fn repetition_counts(start: &Board, moves: &[Mv]) -> Vec<u32> {
    let mut seen: std::collections::HashMap<PosKey, u32> = std::collections::HashMap::new();
    let mut b = start.clone();
    let mut out = Vec::with_capacity(moves.len());
    for m in moves {
        let c = seen.entry(b.key()).or_insert(0);
        out.push(*c);
        *c += 1;
        b.push(m);
    }
    out
}

// ---------------------------------------------------------------- USI / SFEN

const USI_PIECES: [char; 8] = ['?', 'P', 'L', 'N', 'S', 'G', 'B', 'R'];
const SFEN_PIECES: [&str; 15] = ["", "P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R"];

/// USI square name (`7g`).
pub fn usi_square(sq: usize) -> String {
    format!("{}{}", 9 - sq % 9, (b'a' + (sq / 9) as u8) as char)
}

fn parse_usi_square(file: u8, rank: u8) -> Option<usize> {
    let (f, r) = (file.checked_sub(b'0')?, rank.checked_sub(b'a')?);
    ((1..=9).contains(&f) && r < 9).then(|| r as usize * 9 + (9 - f as usize))
}

/// USI notation of a move (absolute squares): `7g7f`, `8h2b+`, `P*5e`.
pub fn usi_move(m: &Mv) -> String {
    match m.from {
        Some(f) => format!("{}{}{}", usi_square(f), usi_square(m.to), if m.promotion { "+" } else { "" }),
        None => format!("{}*{}", USI_PIECES[m.drop as usize], usi_square(m.to)),
    }
}

/// Parse a USI move (no legality check).
pub fn parse_usi_move(s: &str) -> Option<Mv> {
    let b = s.as_bytes();
    if b.len() == 4 && b[1] == b'*' {
        let drop = USI_PIECES.iter().position(|&c| c as u8 == b[0]).filter(|&p| p >= 1)? as u8;
        return Some(Mv { from: None, to: parse_usi_square(b[2], b[3])?, promotion: false, drop });
    }
    if b.len() == 4 || (b.len() == 5 && b[4] == b'+') {
        let from = parse_usi_square(b[0], b[1])?;
        let to = parse_usi_square(b[2], b[3])?;
        return Some(Mv { from: Some(from), to, promotion: b.len() == 5, drop: NONE });
    }
    None
}

/// SFEN of a position (board, side to move, hands in R B G S N L P order, move number).
pub fn sfen(b: &Board) -> String {
    let mut s = String::new();
    for r in 0..9 {
        let mut empty = 0;
        for c in 0..9 {
            let sq = r * 9 + c;
            if b.piece[sq] == NONE {
                empty += 1;
                continue;
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
                empty = 0;
            }
            let p = SFEN_PIECES[b.piece[sq] as usize];
            s.push_str(&if b.color[sq] as usize == BLACK { p.to_string() } else { p.to_lowercase() });
        }
        if empty > 0 {
            s.push_str(&empty.to_string());
        }
        if r < 8 {
            s.push('/');
        }
    }
    s.push_str(if b.turn == BLACK { " b " } else { " w " });
    let mut hands = String::new();
    for c in [BLACK, WHITE] {
        for pt in (PAWN..=ROOK).rev().map(|p| p as usize) {
            let n = b.hand[c][pt];
            if n == 0 {
                continue;
            }
            if n > 1 {
                hands.push_str(&n.to_string());
            }
            let l = SFEN_PIECES[pt];
            hands.push_str(&if c == BLACK { l.to_string() } else { l.to_lowercase() });
        }
    }
    s.push_str(if hands.is_empty() { "-" } else { &hands });
    let _ = write!(s, " {}", b.move_number);
    s
}

/// Parse SFEN fields (board, side, hands, [move number]); no position checks (see
/// [`check_position`]).
pub fn parse_sfen(fields: &[&str]) -> Result<Board, String> {
    if fields.len() < 3 {
        return Err("SFEN needs board, side and hands".into());
    }
    let mut b = Board::empty();
    let rows: Vec<&str> = fields[0].split('/').collect();
    if rows.len() != 9 {
        return Err(format!("SFEN board {:?} has {} rows", fields[0], rows.len()));
    }
    for (r, row) in rows.iter().enumerate() {
        let (mut c, mut promoted) = (0usize, false);
        for ch in row.chars() {
            if let Some(d) = ch.to_digit(10) {
                c += d as usize;
                continue;
            }
            if ch == '+' {
                promoted = true;
                continue;
            }
            let name = format!("{}{}", if promoted { "+" } else { "" }, ch.to_ascii_uppercase());
            let pt = SFEN_PIECES.iter().position(|&p| p == name).filter(|&p| p > 0).ok_or_else(|| format!("SFEN piece {name:?}"))?;
            if c >= 9 {
                return Err(format!("SFEN row {row:?} is longer than 9 squares"));
            }
            let owner = if ch.is_ascii_uppercase() { BLACK } else { WHITE };
            b.put(r * 9 + c, pt as u8, owner);
            c += 1;
            promoted = false;
        }
        if c != 9 {
            return Err(format!("SFEN row {row:?} covers {c} squares"));
        }
    }
    b.turn = match fields[1] {
        "b" => BLACK,
        "w" => WHITE,
        s => return Err(format!("SFEN side {s:?}")),
    };
    if fields[2] != "-" {
        let mut n = 0u32;
        for ch in fields[2].chars() {
            if let Some(d) = ch.to_digit(10) {
                n = n * 10 + d;
                continue;
            }
            let pt = SFEN_PIECES.iter().position(|&p| p == ch.to_ascii_uppercase().to_string()).filter(|&p| (1..=7).contains(&p));
            let pt = pt.ok_or_else(|| format!("SFEN hand piece {ch:?}"))?;
            b.hand[if ch.is_ascii_uppercase() { BLACK } else { WHITE }][pt] += n.max(1);
            n = 0;
        }
    }
    if let Some(m) = fields.get(3) {
        b.move_number = m.parse().map_err(|_| format!("SFEN move number {m:?}"))?;
    }
    Ok(b)
}

/// A position a game may start from: one king per side, the side NOT to move not in
/// check, no piece that could never move again (pawn / lance on the last rank, knight on
/// the last two), no two unpromoted pawns of one side on a file, at most the 40 pieces of
/// a set (per type), a move number >= 1.
pub fn check_position(b: &Board) -> Result<(), String> {
    for c in [BLACK, WHITE] {
        let kings = (0..81).filter(|&s| b.owned_by(s, c) && b.piece[s] == KING).count();
        if kings != 1 || b.king[c].is_none_or(|k| b.piece[k] != KING || !b.owned_by(k, c)) {
            return Err(format!("{} has {kings} kings", ["black", "white"][c]));
        }
    }
    let mut other = b.clone();
    other.turn ^= 1;
    if other.is_check() {
        return Err("the side not to move is in check".into());
    }
    let mut counts = [0u32; 9]; // per unpromoted type (kings: checked above)
    for sq in 0..81 {
        let pt = b.piece[sq];
        if pt == NONE {
            continue;
        }
        counts[demoted(pt) as usize] += 1;
        let c = b.color[sq] as usize;
        if !Board::can_move_without_promotion(sq, pt, c) {
            return Err(format!("dead piece {pt} on {}", usi_square(sq)));
        }
        if pt == PAWN && (0..81).any(|s| s != sq && s % 9 == sq % 9 && b.piece[s] == PAWN && b.color[s] as usize == c) {
            return Err(format!("two pawns on file {}", 9 - sq % 9));
        }
    }
    for c in [BLACK, WHITE] {
        for pt in 1..8 {
            counts[pt] += b.hand[c][pt];
        }
        if b.hand[c][KING as usize] != 0 || b.hand[c][0] != 0 {
            return Err("only P..R may be in hand".into());
        }
    }
    let max = [0, 18, 4, 4, 4, 4, 2, 2];
    if let Some(pt) = (1..8).find(|&pt| counts[pt] > max[pt]) {
        return Err(format!("{} pieces of type {pt} (a set has {})", counts[pt], max[pt]));
    }
    if b.move_number == 0 {
        return Err("move number 0".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(s: &str) -> usize {
        // USI "7g" → square
        parse_usi_square(s.as_bytes()[0], s.as_bytes()[1]).unwrap()
    }

    /// Board from (USI square, piece, colour) triples, hand entries (colour, piece, count).
    pub(crate) fn board(pieces: &[(&str, u8, usize)], hands: &[(usize, u8, u32)], turn: usize) -> Board {
        let mut b = Board { turn, ..Board::empty() };
        for &(s, pt, c) in pieces {
            b.put(sq(s), pt, c);
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

    fn to_of(moves: &[Mv], from: &str) -> Vec<usize> {
        let mut v: Vec<usize> = moves.iter().filter(|m| m.from == Some(sq(from))).map(|m| m.to).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn promos(moves: &[Mv], from: &str, to: &str) -> [bool; 2] {
        let mut out = [false; 2];
        for m in moves.iter().filter(|m| m.from == Some(sq(from)) && m.to == sq(to)) {
            out[m.promotion as usize] = true;
        }
        out
    }

    #[test]
    fn opening_moves_and_legality() {
        let b = Board::hirate();
        assert!(b.is_legal(&mv("7g", "7f")));
        assert!(!b.is_legal(&mv("7g", "7e")));
        assert!(!b.is_check());
        let moves = b.legal_moves();
        assert_eq!(moves.len(), 30);
        assert!(moves.iter().all(|m| b.is_legal(m) && m.drop == NONE && !m.promotion));
        // movable: 9 pawns, 2 lances, 2 silvers, 2 golds, the king and the rook
        let froms: std::collections::BTreeSet<_> = moves.iter().map(|m| m.from).collect();
        assert_eq!(froms.len(), 17);
        assert_eq!(to_of(&moves, "7g"), vec![sq("7f")]);
    }

    #[test]
    fn nifu_and_dead_drops() {
        let b = board(&[("5a", KING, WHITE), ("5i", KING, BLACK), ("7g", PAWN, BLACK)],
                      &[(BLACK, PAWN, 1), (BLACK, LANCE, 1), (BLACK, KNIGHT, 1)], BLACK);
        let moves = b.legal_moves();
        let pawn = drop_squares(&moves, PAWN);
        assert!(pawn.iter().all(|&s| col(s) != col(sq("7g"))), "nifu: no pawn drop on file 7");
        assert!(pawn.iter().all(|&s| rank(s) > 0), "no pawn drop on the last rank");
        assert!(pawn.contains(&sq("6e")));
        // 81 − last rank (9, incl. the white king) − file 7 below it (8, incl. the pawn) − black king
        assert_eq!(pawn.len(), 81 - 9 - 8 - 1);
        assert_eq!(drop_squares(&moves, LANCE).len(), 81 - 9 - 2);
        assert_eq!(drop_squares(&moves, KNIGHT).len(), 81 - 18 - 2);
        // a promoted pawn on the file does not count for nifu
        let mut b2 = b.clone();
        b2.piece[sq("7g")] = PROM_PAWN;
        assert!(drop_squares(&b2.legal_moves(), PAWN).contains(&sq("7e")));
        // white: its last ranks are i / h..i
        let w = board(&[("5a", KING, WHITE), ("5i", KING, BLACK)], &[(WHITE, PAWN, 1), (WHITE, KNIGHT, 1)], WHITE);
        let wm = w.legal_moves();
        assert!(drop_squares(&wm, PAWN).iter().all(|&s| rank(s) < 8));
        assert!(drop_squares(&wm, KNIGHT).iter().all(|&s| rank(s) < 7));
    }

    #[test]
    fn pawn_drop_mate_is_excluded() {
        // white king in the corner, own lance/pawn block 2a/2b, black gold on 1c guards 1b
        let pieces = [("1a", KING, WHITE), ("2a", LANCE, WHITE), ("2b", PAWN, WHITE), ("1c", GOLD, BLACK), ("5i", KING, BLACK)];
        let b = board(&pieces, &[(BLACK, PAWN, 1), (BLACK, GOLD, 1)], BLACK);
        let moves = b.legal_moves();
        assert!(!drop_squares(&moves, PAWN).contains(&sq("1b")), "uchifuzume");
        assert!(drop_squares(&moves, GOLD).contains(&sq("1b")), "a gold-drop mate is legal");
        // without the guard the king can take the pawn: legal
        let b2 = board(&[pieces[0], pieces[1], pieces[2], pieces[4]], &[(BLACK, PAWN, 1)], BLACK);
        assert!(drop_squares(&b2.legal_moves(), PAWN).contains(&sq("1b")));
        // a pawn-drop CHECK that is not mate is legal (the king escapes to 2a)
        let b3 = board(&[("1a", KING, WHITE), ("1c", GOLD, BLACK), ("5i", KING, BLACK)], &[(BLACK, PAWN, 1)], BLACK);
        assert!(drop_squares(&b3.legal_moves(), PAWN).contains(&sq("1b")));
    }

    /// The two pawn-drop mates python-shogi's approximation (and so the converter's rules)
    /// does not recognise: documented deviations from the JSA rule, pinned here.
    #[test]
    fn pawn_drop_mates_the_approximation_misses() {
        // (1) the only escape (1a) is attacked THROUGH the king's square: black rook 9a
        // x-rays the king on 2a along rank a; 1b / 3b are covered by golds 1c / 3c
        let b = board(&[("2a", KING, WHITE), ("9a", ROOK, BLACK), ("1c", GOLD, BLACK), ("3c", GOLD, BLACK), ("5i", KING, BLACK)],
                      &[(BLACK, PAWN, 1)], BLACK);
        let drop = Mv { from: None, to: sq("2b"), promotion: false, drop: PAWN };
        assert!(b.is_legal(&drop), "python-shogi lets the x-ray pawn-drop mate through");
        let mut after = b.clone();
        after.push(&drop);
        assert!(after.is_check() && after.legal_moves().is_empty(), "it is mate");
        // (2) the pawn is attacked only by a pinned defender: the white gold on 2b guards
        // 1b, but the black bishop on 4d pins it to the king on 1a; 2a / 2b are held by
        // white, the black gold on 1c protects the pawn
        let b = board(&[("1a", KING, WHITE), ("2a", LANCE, WHITE), ("2b", GOLD, WHITE), ("4d", BISHOP, BLACK),
                        ("1c", GOLD, BLACK), ("5i", KING, BLACK)],
                      &[(BLACK, PAWN, 1)], BLACK);
        let drop = Mv { from: None, to: sq("1b"), promotion: false, drop: PAWN };
        assert!(b.is_legal(&drop), "python-shogi counts the pinned gold as a defender");
        let mut after = b.clone();
        after.push(&drop);
        assert!(after.is_check() && after.legal_moves().is_empty(), "it is mate");
    }

    #[test]
    fn forced_and_optional_promotion() {
        let pieces = [
            ("9a", KING, WHITE),
            ("1i", KING, BLACK),
            ("5b", PAWN, BLACK),
            ("3b", LANCE, BLACK),
            ("4c", KNIGHT, BLACK),
            ("7d", KNIGHT, BLACK),
            ("7e", KNIGHT, BLACK),
            ("5e", SILVER, BLACK),
            ("5f", PAWN, BLACK),
        ];
        let b = board(&pieces, &[], BLACK);
        let moves = b.legal_moves();
        assert_eq!(promos(&moves, "5b", "5a"), [false, true]); // pawn to the last rank
        assert_eq!(promos(&moves, "3b", "3a"), [false, true]); // lance to the last rank
        assert_eq!(promos(&moves, "4c", "5a"), [false, true]); // knight to the last rank
        assert_eq!(promos(&moves, "7d", "8b"), [false, true]); // knight to the second-last rank
        assert_eq!(promos(&moves, "7e", "8c"), [true, true]); // knight into the zone: optional
        assert_eq!(promos(&moves, "5e", "5d"), [true, false]); // outside the zone
        assert_eq!(promos(&moves, "5f", "5e"), [false, false]); // own piece on the target: no move
        assert_eq!(to_of(&moves, "3b"), vec![sq("3a")]);
        // the black king in the corner: all 3 neighbours are free and unattacked
        assert_eq!(to_of(&moves, "1i").len(), 3);
    }

    #[test]
    fn self_check_is_excluded() {
        // black silver pinned on the file by a white rook: it may not leave the file
        let b = board(&[("5i", KING, BLACK), ("5h", SILVER, BLACK), ("5a", ROOK, WHITE), ("9a", KING, WHITE)], &[], BLACK);
        assert_eq!(to_of(&b.legal_moves(), "5h"), vec![sq("5g")]);
    }

    #[test]
    fn captures_return_unpromoted_pieces_to_the_hand() {
        let mut b = board(&[("5i", KING, BLACK), ("5a", KING, WHITE), ("2c", PROM_ROOK, WHITE), ("2d", SILVER, BLACK)],
                          &[], BLACK);
        let m = Mv { from: Some(sq("2d")), to: sq("2c"), promotion: true, drop: NONE };
        assert!(b.is_legal(&m));
        b.push(&m);
        assert_eq!(b.hand[BLACK][ROOK as usize], 1, "a dragon is captured as a rook");
        assert_eq!(b.piece[sq("2c")], PROM_SILVER);
        assert_eq!((b.turn, b.move_number), (WHITE, 2));
    }

    #[test]
    fn repetition_counts_after_king_shuffle() {
        let shuffle = [mv("5i", "5h"), mv("5a", "5b"), mv("5h", "5i"), mv("5b", "5a")];
        let moves: Vec<Mv> = shuffle.iter().cycle().take(13).copied().collect();
        let reps = repetition_counts(&Board::hirate(), &moves);
        assert_eq!(reps, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]);
        // same pieces, other side to move: a different position
        let mut b = Board::hirate();
        let k0 = b.key();
        b.turn ^= 1;
        assert!(b.key() != k0);
    }

    #[test]
    fn usi_and_sfen_round_trip() {
        let h = Board::hirate();
        assert_eq!(sfen(&h), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        assert_eq!(parse_sfen(&sfen(&h).split(' ').collect::<Vec<_>>()).unwrap(), h);
        let m = parse_usi_move("8h2b+").unwrap();
        assert_eq!((m.from, m.to, m.promotion), (Some(sq("8h")), sq("2b"), true));
        assert_eq!(usi_move(&m), "8h2b+");
        assert_eq!(usi_move(&parse_usi_move("G*5b").unwrap()), "G*5b");
        for bad in ["", "7g7", "K*5e", "0a1b", "7g7j", "7g7f="] {
            assert!(parse_usi_move(bad).is_none(), "{bad}");
        }
        let s = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1";
        let b = parse_sfen(&s.split(' ').collect::<Vec<_>>()).unwrap();
        assert_eq!(sfen(&b), s);
        check_position(&b).unwrap();
        assert!(parse_sfen(&["9/9/9", "b", "-"]).is_err());
        check_position(&h).unwrap();
        let bad = |s: &str| check_position(&parse_sfen(&s.split(' ').collect::<Vec<_>>()).unwrap()).unwrap_err();
        assert!(bad("4k4/9/9/9/9/9/9/9/9 b - 1").contains("kings"));
        assert!(bad("4k4/9/9/9/9/9/9/9/4K4 b 19p 1").contains("pieces of type 1"));
        assert!(bad("P3k4/9/9/9/9/9/9/9/4K4 b - 1").contains("dead"));
        assert!(bad("4k4/9/9/9/4P4/9/4P4/9/4K4 b - 1").contains("two pawns"));
        assert!(bad("4k4/9/9/9/9/9/9/4r4/4K4 w - 1").contains("not to move"));
    }

    #[test]
    fn declaration_27_points() {
        // black king on 5b, 10 pieces in the camp worth 2·5 + 8 = 18, + hand 2R 2B... (28)
        let mut pieces = vec![("5b", KING, BLACK), ("5i", KING, WHITE)];
        let camp = ["9a", "8a", "7a", "6a", "4a", "3a", "2a", "1a", "9c", "1c"];
        for (i, s) in camp.iter().enumerate() {
            pieces.push((s, if i < 2 { PROM_ROOK } else { GOLD }, BLACK));
        }
        let b = board(&pieces, &[(BLACK, BISHOP, 2)], BLACK);
        // 2 dragons 10 + 8 golds 8 + 2 bishops in hand 10 = 28
        assert!(b.can_declare());
        let short = board(&pieces, &[(BLACK, BISHOP, 1), (BLACK, PAWN, 4)], BLACK); // 27
        assert!(!short.can_declare());
        // white needs 27 (the mirrored position)
        let mut w = Board::empty();
        for sq in 0..81 {
            if b.piece[sq] != NONE {
                w.put(80 - sq, b.piece[sq], b.color[sq] as usize ^ 1);
            }
        }
        w.turn = WHITE;
        w.hand[WHITE] = short.hand[BLACK];
        assert!(w.can_declare());
        // 9 pieces in the camp: no
        let mut nine = b.clone();
        nine.piece[sq("1c")] = NONE;
        nine.hand[BLACK][PAWN as usize] = 5;
        assert!(!nine.can_declare());
        // in check: no; king outside the camp: no
        let mut chk = b.clone();
        chk.put(sq("5a"), PROM_ROOK, WHITE);
        assert!(chk.is_check() && !chk.can_declare());
        let mut out = b.clone();
        out.piece[sq("5b")] = NONE;
        out.put(sq("5d"), KING, BLACK);
        assert!(!out.can_declare());
    }
}
