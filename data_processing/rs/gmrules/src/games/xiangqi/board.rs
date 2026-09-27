//! Xiangqi board primitives: squares, 90-bit bitboards, attack sets, legal move
//! generation and the per-move facts the repetition rules need (checkers, king
//! blockers, AXF "chased" pieces).
//!
//! Every function here is a port of Fairy-Stockfish's `xiangqi` variant
//! (`src/position.cpp` / `bitboard.cpp` / `piece.cpp` of Fairy-Stockfish as shipped in
//! pyffish 0.0.90), including its approximations of the Asian rules: attack sets
//! follow Fairy-Stockfish's `attacks_bb` (NOT restricted to the piece's region) or
//! `attacks_from` (restricted: palace for king / advisors, own half for elephants,
//! file-only for soldiers that have not crossed the river) exactly where
//! Fairy-Stockfish uses the one or the other. `tests/test_xiangqi_reference.py`
//! pins the whole rule engine to pyffish move by move.
//!
//! Squares: `sq = rank * 9 + file`, rank 0 = Red's back rank (UCI rank "1"), file 0 =
//! file "a" (UCI coordinates of Fairy-Stockfish: a1 .. i10). Red = seat 0 moves
//! first ("w" in the FEN), Black = seat 1.

use std::sync::OnceLock;

/// A set of squares (bit `sq`).
pub type Bb = u128;
/// A square, `0..90`.
pub type Sq = usize;

pub const FILES: usize = 9;
pub const RANKS: usize = 10;
pub const SQUARES: usize = 90;

pub const RED: usize = 0;
pub const BLACK: usize = 1;

pub const KING: usize = 0;
pub const ADVISOR: usize = 1;
pub const ELEPHANT: usize = 2;
pub const HORSE: usize = 3;
pub const ROOK: usize = 4;
pub const CANNON: usize = 5;
pub const SOLDIER: usize = 6;
pub const PIECE_TYPES: usize = 7;

/// FEN letters in piece-type order (Fairy-Stockfish's xiangqi letters; upper case =
/// Red): king, advisor, elephant ("b"; "e" accepted), horse ("n"; "h" accepted),
/// rook (chariot), cannon, soldier ("p").
pub const PIECE_CHARS: [char; PIECE_TYPES] = ['k', 'a', 'b', 'n', 'r', 'c', 'p'];

pub const ALL: Bb = (1u128 << SQUARES) - 1;

#[inline]
pub fn bit(sq: Sq) -> Bb {
    1u128 << sq
}

#[inline]
pub fn file_of(sq: Sq) -> usize {
    sq % FILES
}

#[inline]
pub fn rank_of(sq: Sq) -> usize {
    sq / FILES
}

#[inline]
pub fn square(file: usize, rank: usize) -> Sq {
    rank * FILES + file
}

/// Rank of `sq` counted from color `c`'s back rank.
#[inline]
pub fn relative_rank(c: usize, sq: Sq) -> usize {
    if c == RED { rank_of(sq) } else { RANKS - 1 - rank_of(sq) }
}

#[inline]
pub fn more_than_one(b: Bb) -> bool {
    b & b.wrapping_sub(1) != 0
}

#[inline]
pub fn lsb(b: Bb) -> Sq {
    b.trailing_zeros() as Sq
}

/// The squares of a bitboard, ascending.
#[derive(Clone, Copy)]
pub struct Squares(pub Bb);

impl Iterator for Squares {
    type Item = Sq;
    #[inline]
    fn next(&mut self) -> Option<Sq> {
        if self.0 == 0 {
            return None;
        }
        let s = lsb(self.0);
        self.0 &= self.0 - 1;
        Some(s)
    }
}

/// UCI name of a square ("a1" .. "i10").
pub fn square_name(sq: Sq) -> String {
    format!("{}{}", (b'a' + file_of(sq) as u8) as char, rank_of(sq) + 1)
}

/// Parse a UCI square name.
pub fn parse_square(s: &str) -> Option<Sq> {
    let mut ch = s.chars();
    let f = ch.next()?;
    if !('a'..='i').contains(&f) {
        return None;
    }
    let r: usize = ch.as_str().parse().ok()?;
    (1..=RANKS).contains(&r).then(|| square(f as usize - 'a' as usize, r - 1))
}

/// Precomputed step tables.
pub struct Tables {
    pub file: [Bb; FILES],
    pub rank: [Bb; RANKS],
    /// palace of each color (king / advisor region)
    pub palace: [Bb; 2],
    /// own half of each color (elephant region)
    pub half: [Bb; 2],
    /// orthogonal neighbours (Fairy-Stockfish WAZIR)
    pub wazir: [Bb; SQUARES],
    /// diagonal neighbours (FERS)
    pub fers: [Bb; SQUARES],
    /// horse destinations on an empty board, and (destination, leg) pairs
    pub horse_pseudo: [Bb; SQUARES],
    pub horse: Vec<Vec<(Sq, Sq)>>,
    /// elephant destinations on an empty board, and (destination, eye) pairs
    pub elephant_pseudo: [Bb; SQUARES],
    pub elephant: Vec<Vec<(Sq, Sq)>>,
    /// soldier of color c on sq: forward + sideways ("fsW", unrestricted)
    pub soldier: [[Bb; SQUARES]; 2],
    /// the square in front of sq for color c (Fairy-Stockfish SHOGI_PAWN)
    pub soldier_fwd: [[Bb; SQUARES]; 2],
    /// rank + file through sq without sq (PseudoAttacks of rook / cannon)
    pub lines: [Bb; SQUARES],
    /// LineBB: the full rank / file through two aligned squares (0 when not aligned)
    line: Vec<Bb>,
}

impl Tables {
    #[inline]
    pub fn line(&self, a: Sq, b: Sq) -> Bb {
        self.line[a * SQUARES + b]
    }
}

fn build_tables() -> Tables {
    let on = |f: i32, r: i32| (0..FILES as i32).contains(&f) && (0..RANKS as i32).contains(&r);
    let sq_of = |f: i32, r: i32| square(f as usize, r as usize);
    let mut t = Tables {
        file: [0; FILES],
        rank: [0; RANKS],
        palace: [0; 2],
        half: [0; 2],
        wazir: [0; SQUARES],
        fers: [0; SQUARES],
        horse_pseudo: [0; SQUARES],
        horse: vec![Vec::new(); SQUARES],
        elephant_pseudo: [0; SQUARES],
        elephant: vec![Vec::new(); SQUARES],
        soldier: [[0; SQUARES]; 2],
        soldier_fwd: [[0; SQUARES]; 2],
        lines: [0; SQUARES],
        line: vec![0; SQUARES * SQUARES],
    };
    for sq in 0..SQUARES {
        let (f, r) = (file_of(sq), rank_of(sq));
        t.file[f] |= bit(sq);
        t.rank[r] |= bit(sq);
        if (3..=5).contains(&f) {
            if r <= 2 {
                t.palace[RED] |= bit(sq);
            }
            if r >= 7 {
                t.palace[BLACK] |= bit(sq);
            }
        }
        if r <= 4 {
            t.half[RED] |= bit(sq);
        } else {
            t.half[BLACK] |= bit(sq);
        }
    }
    for sq in 0..SQUARES {
        let (f, r) = (file_of(sq) as i32, rank_of(sq) as i32);
        for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if on(f + df, r + dr) {
                t.wazir[sq] |= bit(sq_of(f + df, r + dr));
            }
        }
        for (df, dr) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            if on(f + df, r + dr) {
                t.fers[sq] |= bit(sq_of(f + df, r + dr));
            }
            if on(f + 2 * df, r + 2 * dr) {
                let to = sq_of(f + 2 * df, r + 2 * dr);
                t.elephant_pseudo[sq] |= bit(to);
                t.elephant[sq].push((to, sq_of(f + df, r + dr)));
            }
        }
        // horse: two steps along one axis (the leg is the first of them), one aside
        for (df, dr) in [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)] {
            if on(f + df, r + dr) {
                let to = sq_of(f + df, r + dr);
                let leg = if dr.abs() == 2 { sq_of(f, r + dr / 2) } else { sq_of(f + df / 2, r) };
                t.horse_pseudo[sq] |= bit(to);
                t.horse[sq].push((to, leg));
            }
        }
        for c in [RED, BLACK] {
            let fwd = if c == RED { 1 } else { -1 };
            if on(f, r + fwd) {
                t.soldier[c][sq] |= bit(sq_of(f, r + fwd));
                t.soldier_fwd[c][sq] |= bit(sq_of(f, r + fwd));
            }
            for df in [1, -1] {
                if on(f + df, r) {
                    t.soldier[c][sq] |= bit(sq_of(f + df, r));
                }
            }
        }
        t.lines[sq] = (t.file[f as usize] | t.rank[r as usize]) & !bit(sq);
    }
    for a in 0..SQUARES {
        for b in 0..SQUARES {
            if a != b {
                if file_of(a) == file_of(b) {
                    t.line[a * SQUARES + b] = t.file[file_of(a)];
                } else if rank_of(a) == rank_of(b) {
                    t.line[a * SQUARES + b] = t.rank[rank_of(a)];
                }
            }
        }
    }
    t
}

pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(build_tables)
}

const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Rook (chariot) attacks: every square up to and including the first occupied one.
pub fn rook_attacks(sq: Sq, occ: Bb) -> Bb {
    let mut a = 0;
    let (f, r) = (file_of(sq) as i32, rank_of(sq) as i32);
    for (df, dr) in DIRS {
        let (mut x, mut y) = (f + df, r + dr);
        while (0..FILES as i32).contains(&x) && (0..RANKS as i32).contains(&y) {
            let s = square(x as usize, y as usize);
            a |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            x += df;
            y += dr;
        }
    }
    a
}

/// Cannon captures as Fairy-Stockfish's hopper attack set: the squares after the first
/// occupied square (the screen) up to and including the next occupied one.
pub fn cannon_attacks(sq: Sq, occ: Bb) -> Bb {
    let mut a = 0;
    let (f, r) = (file_of(sq) as i32, rank_of(sq) as i32);
    for (df, dr) in DIRS {
        let (mut x, mut y) = (f + df, r + dr);
        let mut hurdle = false;
        while (0..FILES as i32).contains(&x) && (0..RANKS as i32).contains(&y) {
            let s = square(x as usize, y as usize);
            if hurdle {
                a |= bit(s);
            }
            if occ & bit(s) != 0 {
                if hurdle {
                    break;
                }
                hurdle = true;
            }
            x += df;
            y += dr;
        }
    }
    a
}

/// Horse destinations whose leg is empty.
pub fn horse_attacks(sq: Sq, occ: Bb) -> Bb {
    tables().horse[sq].iter().filter(|&&(_, leg)| occ & bit(leg) == 0).fold(0, |a, &(to, _)| a | bit(to))
}

/// Elephant destinations whose eye is empty (NOT restricted to the own half).
pub fn elephant_attacks(sq: Sq, occ: Bb) -> Bb {
    tables().elephant[sq].iter().filter(|&&(_, eye)| occ & bit(eye) == 0).fold(0, |a, &(to, _)| a | bit(to))
}

/// Fairy-Stockfish `attacks_bb(c, pt, sq, occ)`: the piece's attack pattern, NOT
/// restricted to its region (the king attacks like its move type, the wazir).
pub fn attacks_bb(c: usize, pt: usize, sq: Sq, occ: Bb) -> Bb {
    let t = tables();
    match pt {
        KING => t.wazir[sq],
        ADVISOR => t.fers[sq],
        ELEPHANT => elephant_attacks(sq, occ),
        HORSE => horse_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        CANNON => cannon_attacks(sq, occ),
        _ => t.soldier[c][sq],
    }
}

/// The region a piece type of color `c` may stand on (Fairy-Stockfish `board_bb(c, pt)`).
#[inline]
pub fn region(c: usize, pt: usize) -> Bb {
    let t = tables();
    match pt {
        KING | ADVISOR => t.palace[c],
        ELEPHANT => t.half[c],
        _ => ALL,
    }
}

/// A soldier of color `c` on `sq` has crossed the river (Fairy-Stockfish's
/// "promoted soldier", soldierPromotionRank = 6).
#[inline]
pub fn soldier_crossed(c: usize, sq: Sq) -> bool {
    relative_rank(c, sq) >= 5
}

/// Fairy-Stockfish `attacks_from(c, pt, sq)`: attacks restricted to the piece's region,
/// soldiers on their own side of the river attack straight ahead only.
pub fn attacks_from(c: usize, pt: usize, sq: Sq, occ: Bb) -> Bb {
    let mut b = attacks_bb(c, pt, sq, occ);
    if pt == SOLDIER && !soldier_crossed(c, sq) {
        b &= tables().file[file_of(sq)];
    }
    b & region(c, pt)
}

/// Fairy-Stockfish `moves_from(c, pt, sq)` (quiet-move pattern; callers mask empty
/// squares): rooks and cannons slide, every other piece moves like it attacks.
pub fn moves_from(c: usize, pt: usize, sq: Sq, occ: Bb) -> Bb {
    match pt {
        ROOK | CANNON => rook_attacks(sq, occ),
        _ => attacks_from(c, pt, sq, occ),
    }
}

/// Zobrist keys (fixed seed: identical on every platform).
struct Zobrist {
    psq: [[u64; SQUARES]; 2 * PIECE_TYPES],
    side: u64,
}

fn zobrist() -> &'static Zobrist {
    static Z: OnceLock<Zobrist> = OnceLock::new();
    Z.get_or_init(|| {
        let mut s: u64 = 0x5851_F42D_4C95_7F2D;
        let mut next = || {
            // splitmix64
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut psq = [[0u64; SQUARES]; 2 * PIECE_TYPES];
        for row in psq.iter_mut() {
            for k in row.iter_mut() {
                *k = next();
            }
        }
        Zobrist { psq, side: next() }
    })
}

/// A position: piece placement and the side to move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    /// 0 = empty, else `1 + color * 7 + piece type`
    cells: [u8; SQUARES],
    by_color: [Bb; 2],
    by_type: [Bb; PIECE_TYPES],
    pub side: usize,
}

/// The standard initial position.
pub const START_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

impl Board {
    pub fn empty() -> Board {
        Board { cells: [0; SQUARES], by_color: [0; 2], by_type: [0; PIECE_TYPES], side: RED }
    }

    #[inline]
    pub fn piece_at(&self, sq: Sq) -> Option<(usize, usize)> {
        let c = self.cells[sq];
        (c != 0).then(|| (((c - 1) / 7) as usize, ((c - 1) % 7) as usize))
    }

    #[inline]
    pub fn type_at(&self, sq: Sq) -> usize {
        ((self.cells[sq] - 1) % 7) as usize
    }

    #[inline]
    pub fn occupied(&self) -> Bb {
        self.by_color[RED] | self.by_color[BLACK]
    }

    #[inline]
    pub fn color_bb(&self, c: usize) -> Bb {
        self.by_color[c]
    }

    #[inline]
    pub fn type_bb(&self, pt: usize) -> Bb {
        self.by_type[pt]
    }

    #[inline]
    pub fn pieces(&self, c: usize, pt: usize) -> Bb {
        self.by_color[c] & self.by_type[pt]
    }

    #[inline]
    pub fn king_sq(&self, c: usize) -> Sq {
        lsb(self.pieces(c, KING))
    }

    pub fn put(&mut self, sq: Sq, c: usize, pt: usize) {
        debug_assert!(self.cells[sq] == 0);
        self.cells[sq] = (1 + c * 7 + pt) as u8;
        self.by_color[c] |= bit(sq);
        self.by_type[pt] |= bit(sq);
    }

    fn remove(&mut self, sq: Sq) -> (usize, usize) {
        let (c, pt) = self.piece_at(sq).expect("a piece to remove");
        self.cells[sq] = 0;
        self.by_color[c] &= !bit(sq);
        self.by_type[pt] &= !bit(sq);
        (c, pt)
    }

    /// Move the piece on `from` to `to` (capturing what stands there) and pass the
    /// turn; returns the captured piece.
    pub fn make_move(&mut self, from: Sq, to: Sq) -> Option<(usize, usize)> {
        let captured = self.piece_at(to).map(|_| self.remove(to));
        let (c, pt) = self.remove(from);
        self.put(to, c, pt);
        self.side = 1 - self.side;
        captured
    }

    /// Zobrist key of the placement and the side to move (Fairy-Stockfish's
    /// repetition identity for xiangqi).
    pub fn key(&self) -> u64 {
        let z = zobrist();
        let mut k = if self.side == BLACK { z.side } else { 0 };
        for sq in Squares(self.occupied()) {
            k ^= z.psq[self.cells[sq] as usize - 1][sq];
        }
        k
    }

    /// Soldiers of color `c` that have crossed the river.
    pub fn crossed_soldiers(&self, c: usize) -> Bb {
        let zone = if c == RED { ALL & !tables().half[RED] } else { tables().half[RED] };
        self.pieces(c, SOLDIER) & zone
    }

    /// Fairy-Stockfish `attackers_to(s, occupied, c)`: pieces of color `c` attacking
    /// `s` with occupancy `occ` (region-aware; the horse by its forward leg rule;
    /// soldiers attack a square on `c`'s own side of the river only from behind it).
    pub fn attackers_to(&self, s: Sq, occ: Bb, c: usize) -> Bb {
        let t = tables();
        let them = 1 - c;
        let mut b = 0;
        if t.palace[c] & bit(s) != 0 {
            b |= t.wazir[s] & self.pieces(c, KING);
            b |= t.fers[s] & self.pieces(c, ADVISOR);
        }
        if t.half[c] & bit(s) != 0 {
            b |= elephant_attacks(s, occ) & self.pieces(c, ELEPHANT);
        }
        for s2 in Squares(t.horse_pseudo[s] & self.pieces(c, HORSE)) {
            if horse_attacks(s2, occ) & bit(s) != 0 {
                b |= bit(s2);
            }
        }
        b |= rook_attacks(s, occ) & self.pieces(c, ROOK);
        b |= cannon_attacks(s, occ) & self.pieces(c, CANNON);
        b |= t.soldier[them][s] & self.pieces(c, SOLDIER);
        if b & self.by_type[SOLDIER] != 0 && relative_rank(c, s) < 5 {
            b ^= b & self.by_type[SOLDIER] & !t.soldier_fwd[them][s];
        }
        b
    }

    /// Pieces giving check to the side to move.
    pub fn checkers(&self) -> Bb {
        self.attackers_to(self.king_sq(self.side), self.occupied(), 1 - self.side)
    }

    /// Fairy-Stockfish `slider_blockers(pieces(a), king of !a, ...)`: pieces of both
    /// colors whose removal would expose the king of color `1 - a` to an attack by a
    /// rook, cannon (two screens), horse (the leg) or elephant (the eye) of color `a`.
    pub fn king_blockers(&self, a: usize) -> Bb {
        let t = tables();
        let k = 1 - a;
        let s = self.king_sq(k);
        let occ = self.occupied();
        let mut snipers: Bb = 0;
        let mut sliding: Bb = 0;
        // rooks / cannons on the king's lines that do not attack it now
        let r = t.lines[s] & self.pieces(a, ROOK);
        if r != 0 {
            snipers |= r & !rook_attacks(s, occ);
            sliding |= snipers & self.by_type[ROOK];
        }
        let c = t.lines[s] & self.pieces(a, CANNON);
        if c != 0 {
            snipers |= c & !cannon_attacks(s, occ);
        }
        // horses a horse move away whose leg is blocked (asymmetric: the forward rule)
        let h = t.horse_pseudo[s] & self.pieces(a, HORSE);
        if h != 0 {
            for s2 in Squares(h) {
                if attacks_from(a, HORSE, s2, occ) & bit(s) == 0 {
                    snipers |= bit(s2);
                }
            }
            sliding |= snipers & self.by_type[HORSE];
        }
        let e = t.elephant_pseudo[s] & self.pieces(a, ELEPHANT);
        if e != 0 {
            snipers |= e & !elephant_attacks(s, occ);
            sliding |= snipers & self.by_type[ELEPHANT];
        }
        let occupancy = occ ^ sliding;
        let mut blockers = 0;
        for sn in Squares(snipers) {
            let pt = self.type_at(sn);
            let (between, hopper) = match pt {
                HORSE => (t.wazir[sn] & t.fers[s], false),
                CANNON => (between_bb(s, sn), true),
                _ => (between_bb(s, sn), false),
            };
            let b = between & if hopper { occ ^ bit(sn) } else { occupancy };
            if b != 0 && (!more_than_one(b) || (hopper && b.count_ones() == 2)) {
                blockers |= b;
            }
        }
        blockers
    }

    /// Fairy-Stockfish `legal(m)` for a pseudo-legal move: the mover's king is not
    /// attacked afterwards and the two kings do not face each other on an open file.
    pub fn is_legal(&self, from: Sq, to: Sq) -> bool {
        let us = self.side;
        let them = 1 - us;
        let pt = self.type_at(from);
        let occupied = (self.occupied() ^ bit(from)) | bit(to);
        let s = if pt == KING { to } else { self.king_sq(us) };
        if rook_attacks(s, occupied) & self.pieces(them, KING) & !bit(to) != 0 {
            return false;
        }
        if pt == KING {
            return self.attackers_to(to, occupied, them) == 0;
        }
        self.attackers_to(self.king_sq(us), occupied, them) & !bit(to) == 0
    }

    /// Legal moves of the side to move as action ids `from * 90 + to`, ascending.
    pub fn legal_moves(&self, out: &mut Vec<u32>) {
        let us = self.side;
        let occ = self.occupied();
        let theirs = self.by_color[1 - us];
        for from in Squares(self.by_color[us]) {
            let pt = self.type_at(from);
            let targets = (moves_from(us, pt, from, occ) & !occ) | (attacks_from(us, pt, from, occ) & theirs);
            for to in Squares(targets) {
                if self.is_legal(from, to) {
                    out.push((from * SQUARES + to) as u32);
                }
            }
        }
        out.sort_unstable();
    }

    /// Neither side can win on material (Fairy-Stockfish `has_insufficient_material`
    /// for xiangqi): color `c` has no rook, horse, cannon or soldier.
    pub fn insufficient_material(&self, c: usize) -> bool {
        self.by_color[c] & (self.by_type[ROOK] | self.by_type[HORSE] | self.by_type[CANNON] | self.by_type[SOLDIER]) == 0
    }

    /// Fairy-Stockfish `Position::chased()` (AXF chasing rule): the pieces of the side
    /// to move that the LAST move (`from` → `to`, `captured` = it captured) chases —
    /// newly attacked (directly, by discovery, or by undermining their protection
    /// through new pins / discovered checks) and unprotected or attacked by a weaker
    /// piece; attacks on the king and on soldiers that have not crossed the river do
    /// not count, nor do mutual attacks of equal pieces. `blockers` = king blockers of
    /// this position per color ([`Board::king_blockers`] of the opponent), `prev` =
    /// those of the position before the move (None at the first position).
    pub fn chased(&self, from: Sq, to: Sq, captured: bool, blockers: &[Bb; 2], prev: Option<&[Bb; 2]>) -> Bb {
        let t = tables();
        let stm = self.side;
        let mover = 1 - stm;
        let occ = self.occupied();
        let mut b: Bb = 0;
        let mut pins = blockers[stm];
        let file_pieces = t.file[file_of(self.king_sq(mover))] & self.by_color[stm];
        if file_pieces & self.by_type[KING] != 0 && !more_than_one(file_pieces & !self.by_type[KING]) {
            pins |= file_pieces & !self.by_type[KING];
        }
        // direct attacks of the moved piece (rooks / cannons: not along their line)
        let moved = self.type_at(to);
        if moved != KING && moved != SOLDIER {
            let mut direct = attacks_from(mover, moved, to, occ) & self.by_color[stm];
            if moved == ROOK || moved == CANNON {
                direct &= !t.line(from, to);
            }
            self.add_chased(&mut b, pins, to, moved, direct);
        }
        // discovered attacks: horses / elephants unblocked at `from`, rooks / cannons on
        // the lines through `from`, cannons given a screen on `to`
        let cand = (t.wazir[from] & self.pieces(mover, HORSE))
            | (t.fers[from] & self.pieces(mover, ELEPHANT))
            | (t.lines[from] & (self.pieces(mover, CANNON) | self.pieces(mover, ROOK)))
            | (t.lines[to] & self.pieces(mover, CANNON));
        let before = (if captured { occ } else { occ ^ bit(to) }) ^ bit(from);
        for s in Squares(cand) {
            let pt = self.type_at(s);
            let disc = self.by_color[stm] & attacks_bb(mover, pt, s, occ) & !attacks_bb(mover, pt, s, before);
            self.add_chased(&mut b, pins, s, pt, disc);
        }
        if let Some(prev) = prev {
            // fake roots: defenders that the move pinned to their king
            let unpromoted = (self.pieces(stm, KING) | self.pieces(stm, SOLDIER)) ^ self.crossed_soldiers(stm);
            for s in Squares(blockers[stm] & !prev[stm] & self.by_color[stm]) {
                let pt = self.type_at(s);
                let fake = self.by_color[stm] & !unpromoted & attacks_bb(stm, pt, s, occ);
                for s2 in Squares(fake) {
                    if self.attackers_to(s2, occ, mover) & !blockers[mover] != 0 {
                        b |= bit(s2);
                    }
                }
            }
            // discovered-check threats: a piece of the mover that now blocks one of its
            // own attacks on the king captures with check
            let ksq = self.king_sq(stm);
            let king_reach = attacks_from(stm, KING, ksq, occ);
            for s in Squares(blockers[stm] & !prev[stm] & self.by_color[mover]) {
                let pt = self.type_at(s);
                let mut da = attacks_from(mover, pt, s, occ) & self.by_color[stm];
                b |= da & !king_reach;
                da &= king_reach;
                for s2 in Squares(da) {
                    if self.attackers_to(s2, occ ^ bit(s) ^ bit(ksq), mover) & !bit(s) != 0 {
                        b |= bit(s2);
                    }
                }
            }
        }
        b
    }

    /// Fairy-Stockfish's `addChased` lambda of `Position::chased()`.
    fn add_chased(&self, b: &mut Bb, pins: Bb, attacker_sq: Sq, attacker: usize, mut attacks: Bb) {
        if attacks & !*b == 0 {
            return;
        }
        let stm = self.side;
        let occ = self.occupied();
        // no chase of the king (checks) or of soldiers on their own side of the river
        attacks &= !((self.pieces(stm, KING) | self.pieces(stm, SOLDIER)) ^ self.crossed_soldiers(stm));
        // attacks on stronger pieces are chases whatever the protection
        if attacker == HORSE || attacker == CANNON {
            *b |= attacks & self.pieces(stm, ROOK);
        }
        if attacker == ELEPHANT || attacker == ADVISOR {
            *b |= attacks & (self.pieces(stm, ROOK) | self.pieces(stm, CANNON) | self.pieces(stm, HORSE));
        }
        // mutual attacks of equal pieces are no chase (exceptions: the lame horse, pins)
        if attacker == HORSE && tables().fers[attacker_sq] & occ != 0 {
            for s in Squares(attacks & self.pieces(stm, HORSE)) {
                if attacks_bb(stm, HORSE, s, occ) & bit(attacker_sq) != 0 {
                    attacks ^= bit(s);
                }
            }
        } else {
            attacks &= !self.pieces(stm, attacker) | pins;
        }
        // attacks on (possibly) unprotected pieces
        for s in Squares(attacks) {
            let roots = self.attackers_to(s, occ ^ bit(attacker_sq), stm) & !pins;
            if roots == 0
                || (roots == self.pieces(stm, KING)
                    && rook_attacks(self.king_sq(1 - stm), occ ^ bit(attacker_sq)) & bit(s) != 0)
            {
                *b |= bit(s);
            }
        }
    }

    /// Parse a Fairy-Stockfish xiangqi FEN (board, side to move, halfmove clock);
    /// returns the board and the halfmove clock. A pocket suffix `[...]` is ignored.
    pub fn from_fen(fen: &str) -> Result<(Board, u32), String> {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().ok_or("empty FEN")?;
        let placement = placement.split('[').next().unwrap_or(placement);
        let side = parts.next().unwrap_or("w");
        let _castling = parts.next();
        let _ep = parts.next();
        let halfmove: u32 = match parts.next() {
            Some(h) => h.parse().map_err(|_| format!("halfmove clock {h:?}"))?,
            None => 0,
        };
        let mut b = Board::empty();
        let rows: Vec<&str> = placement.split('/').collect();
        if rows.len() != RANKS {
            return Err(format!("{} ranks in {placement:?} (need 10)", rows.len()));
        }
        for (i, row) in rows.iter().enumerate() {
            let rank = RANKS - 1 - i;
            let mut file = 0usize;
            let mut digits = String::new();
            let flush = |digits: &mut String, file: &mut usize| -> Result<(), String> {
                if !digits.is_empty() {
                    *file += digits.parse::<usize>().map_err(|_| format!("bad run {digits:?}"))?;
                    digits.clear();
                }
                Ok(())
            };
            for ch in row.chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                    continue;
                }
                flush(&mut digits, &mut file)?;
                let lower = ch.to_ascii_lowercase();
                let pt = match lower {
                    'e' => ELEPHANT,
                    'h' => HORSE,
                    _ => PIECE_CHARS.iter().position(|&p| p == lower).ok_or_else(|| format!("piece {ch:?}"))?,
                };
                if file >= FILES {
                    return Err(format!("rank {} of {placement:?} is too long", rank + 1));
                }
                let c = if ch.is_ascii_uppercase() { RED } else { BLACK };
                b.put(square(file, rank), c, pt);
                file += 1;
            }
            flush(&mut digits, &mut file)?;
            if file != FILES {
                return Err(format!("rank {} of {placement:?} has {file} files", rank + 1));
            }
        }
        b.side = match side {
            "w" | "r" => RED,
            "b" => BLACK,
            s => return Err(format!("side to move {s:?}")),
        };
        b.validate()?;
        Ok((b, halfmove))
    }

    /// Sanity of a position: one king per side in its palace, advisors in the palace,
    /// elephants on their side, soldiers never behind their start rank, kings not
    /// facing, the side that just moved not in check.
    pub fn validate(&self) -> Result<(), String> {
        let t = tables();
        for c in [RED, BLACK] {
            let name = if c == RED { "red" } else { "black" };
            if self.pieces(c, KING).count_ones() != 1 {
                return Err(format!("{name} needs exactly one king"));
            }
            if self.pieces(c, KING) & !t.palace[c] != 0 || self.pieces(c, ADVISOR) & !t.palace[c] != 0 {
                return Err(format!("{name} king / advisor outside the palace"));
            }
            if self.pieces(c, ELEPHANT) & !t.half[c] != 0 {
                return Err(format!("{name} elephant across the river"));
            }
            if Squares(self.pieces(c, SOLDIER)).any(|s| relative_rank(c, s) < 3) {
                return Err(format!("{name} soldier behind its start rank"));
            }
            let limits = [1, 2, 2, 2, 2, 2, 5];
            for (pt, &n) in limits.iter().enumerate() {
                if self.pieces(c, pt).count_ones() > n {
                    return Err(format!("{name} has more than {n} {}", PIECE_CHARS[pt]));
                }
            }
        }
        if rook_attacks(self.king_sq(RED), self.occupied()) & self.pieces(BLACK, KING) != 0 {
            return Err("the kings face each other".into());
        }
        let waiting = 1 - self.side;
        if self.attackers_to(self.king_sq(waiting), self.occupied(), self.side) != 0 {
            return Err("the side not to move is in check".into());
        }
        Ok(())
    }

    /// The placement + side part of the FEN (Fairy-Stockfish letters).
    pub fn fen_board(&self) -> String {
        let mut s = String::new();
        for rank in (0..RANKS).rev() {
            let mut empty = 0;
            for file in 0..FILES {
                match self.piece_at(square(file, rank)) {
                    None => empty += 1,
                    Some((c, pt)) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let ch = PIECE_CHARS[pt];
                        s.push(if c == RED { ch.to_ascii_uppercase() } else { ch });
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.side == RED { 'w' } else { 'b' });
        s
    }
}

/// Squares strictly between two aligned squares, plus `b` (Fairy-Stockfish
/// `between_bb`: the semi-open segment (a, b]; just `b` when not aligned).
pub fn between_bb(a: Sq, b: Sq) -> Bb {
    let t = tables();
    let line = t.line(a, b);
    if line == 0 {
        return bit(b);
    }
    let (lo, hi) = (a.min(b), a.max(b));
    let step = if file_of(a) == file_of(b) { FILES } else { 1 };
    let mut r = 0;
    let mut s = lo + step;
    while s < hi {
        r |= bit(s);
        s += step;
    }
    r | bit(b)
}

/// Fairy-Stockfish `undo_move_board`: map a set through a move backwards (a piece on the
/// move's destination goes back to its origin).
#[inline]
pub fn undo_move_board(b: Bb, mv: Option<(Sq, Sq)>) -> Bb {
    match mv {
        Some((from, to)) if b & bit(to) != 0 => (b ^ bit(to)) | bit(from),
        _ => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(b: &Board, depth: usize) -> u64 {
        let mut moves = Vec::new();
        b.legal_moves(&mut moves);
        if depth == 1 {
            return moves.len() as u64;
        }
        moves
            .iter()
            .map(|&m| {
                let mut c = b.clone();
                c.make_move(m as usize / SQUARES, m as usize % SQUARES);
                perft(&c, depth - 1)
            })
            .sum()
    }

    #[test]
    fn perft_from_the_start_position() {
        let (b, _) = Board::from_fen(START_FEN).unwrap();
        // the published xiangqi perft numbers (chessprogramming.org / Fairy-Stockfish)
        assert_eq!(perft(&b, 1), 44);
        assert_eq!(perft(&b, 2), 1920);
        assert_eq!(perft(&b, 3), 79666);
    }

    #[test]
    fn fen_round_trip_and_validation() {
        let (b, h) = Board::from_fen(START_FEN).unwrap();
        assert_eq!(h, 0);
        assert_eq!(b.fen_board(), "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w");
        assert_eq!(parse_square("i10"), Some(89));
        assert_eq!(square_name(0), "a1");
        // facing kings, missing king, elephant across the river
        assert!(Board::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1").unwrap_err().contains("face"));
        assert!(Board::from_fen("3k5/9/9/9/9/9/9/9/9/9 w - - 0 1").is_err());
        assert!(Board::from_fen("3k5/9/9/9/2B6/9/9/9/9/5K3 w - - 0 1").unwrap_err().contains("elephant"));
        // pocket suffix of some Fairy-Stockfish FENs
        assert!(Board::from_fen("3k2b2/4P4/4b4/9/8p/6Bc1/6P1P/3AB4/4pp3/1p1K3R1[] w - - 0 1").is_ok());
    }

    fn move_names(b: &Board) -> Vec<String> {
        let mut m = Vec::new();
        b.legal_moves(&mut m);
        m.iter().map(|&a| format!("{}{}", square_name(a as usize / SQUARES), square_name(a as usize % SQUARES))).collect()
    }

    #[test]
    fn piece_rules() {
        // cannon: slides without capturing, captures over exactly one screen
        let (b, _) = Board::from_fen("5k3/9/4r4/9/4p4/9/9/4C4/9/3K5 w - - 0 1").unwrap();
        let m = move_names(&b);
        for (mv, legal) in [("e3e5", true), ("e3e6", false), ("e3e7", false), ("e3e8", true), ("e3a3", true)] {
            assert_eq!(m.contains(&mv.to_string()), legal, "{mv}");
        }
        // horse leg blocking
        let (b, _) = Board::from_fen("4k4/9/9/9/9/9/9/9/1R7/1N1K5 w - - 0 1").unwrap();
        let b1 = parse_square("b1").unwrap();
        assert_eq!(horse_attacks(b1, b.occupied()) & bit(parse_square("a3").unwrap()), 0);
        assert_ne!(horse_attacks(b1, b.occupied()) & bit(parse_square("d2").unwrap()), 0);
        // soldiers move sideways only after crossing the river
        let (b, _) = Board::from_fen("4k4/9/9/9/2P6/4P4/9/9/9/3K5 w - - 0 1").unwrap();
        let m = move_names(&b);
        assert!(m.contains(&"c6b6".to_string()) && m.contains(&"c6c7".to_string()));
        assert!(m.contains(&"e5e6".to_string()) && !m.contains(&"e5d5".to_string()));
        // elephants: blocked eye, never across the river
        let (b, _) = Board::from_fen("3k5/9/9/9/9/9/9/9/3A5/2B1K4 w - - 0 1").unwrap();
        let m = move_names(&b);
        assert!(m.contains(&"c1a3".to_string()) && !m.contains(&"c1e3".to_string()));
        // flying general: the cannon between the kings may only move along the file
        let (b, _) = Board::from_fen("4k4/9/9/9/9/4C4/9/9/9/4K4 w - - 0 1").unwrap();
        let m = move_names(&b);
        let from_e5: Vec<&String> = m.iter().filter(|s| s.starts_with("e5")).collect();
        assert!(!from_e5.is_empty() && from_e5.iter().all(|s| s.as_bytes()[2] == b'e'), "{from_e5:?}");
        // the king may not step onto an open file facing the other king
        let (b, _) = Board::from_fen("3k5/9/9/9/9/9/9/9/9/4K4 w - - 0 1").unwrap();
        let m = move_names(&b);
        assert!(!m.contains(&"e1d1".to_string()) && m.contains(&"e1f1".to_string()) && m.contains(&"e1e2".to_string()));
    }
}
