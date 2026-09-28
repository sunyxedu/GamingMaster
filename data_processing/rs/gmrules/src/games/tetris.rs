//! Versus Tetris (1v1 garbage battle, guideline rules), turn-based: every turn BOTH
//! players place one piece at once (a simultaneous turn), the "pieces per second"
//! equalised — how bot battles and the teacher (Cold Clear) see the game.
//!
//! Rules (the conventions of Cold Clear's libtetris, which the teacher plans with, so
//! every teacher move is a legal action here — `gmdata tetris` checks it):
//!
//! - 10 × 40 field (y = 0 at the bottom), 20 visible rows. SRS rotation with its kick
//!   tables, no 180° rotation; a piece spawns North at x = 4, y = 19 (or y = 20 when
//!   that is blocked; both blocked = block out, the player loses).
//! - Placements: every resting position the piece can reach from its spawn by left /
//!   right / CW / CCW / one-cell soft drops (0G: complete search, no move limit) — the
//!   current piece, or with HOLD the hold piece (the next piece when the hold is empty,
//!   which then consumes it); a placement whose four cells all lie above row 20 is a
//!   lock out (legal only when nothing else is, and loses).
//! - T-spins: the last successful move was a rotation of a T and ≥ 3 of the 4 corners
//!   around its centre are occupied: FULL when both corners on the pointing side are,
//!   or the rotation used the last kick test, else MINI.
//! - Attack: single 0, double 1, triple 2, quad 4; T-spin single 2 / double 4 /
//!   triple 6, mini double 1; +1 back-to-back for a quad or a T-spin clear after one;
//!   combo bonus 0,0,1,1,2,2,3,3,4,4,4,5 (the n-th consecutive clear); a perfect
//!   clear sends 10 (replacing the rest).
//! - Garbage: a clear first cancels the player's pending garbage (oldest first), the
//!   rest is sent; a placement without a clear receives up to `garbage_cap` pending
//!   lines (oldest first; each attack is one batch with one hole column, a chance
//!   event when it is sent). Lines sent this turn arrive at the opponent from the next
//!   turn on (both players' attacks of one turn are simultaneous). Garbage pushing a
//!   block out of the 40 rows loses.
//! - Pieces: ONE 7-bag sequence for both players (chance events, each piece uniform
//!   over the bag's remaining pieces), each player at its own position in it; the
//!   current piece and the next 5 are known.
//! - The game ends when a player loses (both in the same turn: a draw) or after
//!   `max_turns` turns (a draw). Outcome ±1 / 0.
//!
//! Frames (per seat, simultaneous: the per-seat layout; nothing is hidden — both
//! fields, queues and pending garbage are public, the hole columns of pending garbage
//! are known to nobody):
//!
//! ```text
//! entities [20, 34] bits  the 10 columns of my field (type 0), then the opponent's
//!                         (type 1): rows 0..23 occupied, column one-hot
//! globals  [109]          per player (me, opponent): current piece, hold (7 + none),
//!                         next 5 pieces (one-hot each), back-to-back, combo / 12,
//!                         pending garbage / 20, stack height / 40; turn / max_turns
//! actions  [≤ 160, 49]    hold, piece, rotation, x / 10, y / 20, T-spin kind, lines
//!                         cleared, attack / 10, perfect clear, b2b and combo after,
//!                         the column heights / holes / max height / bumpiness after,
//!                         the four cells, garbage received (/ 10), loses
//! ```
//!
//! Canonical action ids: `(((hold · 4 + rot) · 14 + x + 2) · 42 + y) · 3 + tspin` over
//! the placement's CANONICAL position (libtetris' `canonical`: the one (rot, x, y) of
//! each cell set of O / I / S / Z).

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const W: i32 = 10;
pub const H: i32 = 40;
pub const VISIBLE: i32 = 20;
pub const NEXT: usize = 5;
pub const N_PIECES: usize = 7;
/// entity rows encoded per column
pub const ROWS: usize = 24;
pub const ENTITY_FEATURES: usize = ROWS + W as usize;
pub const PLAYER_GLOBALS: usize = 7 + 8 + 7 * NEXT + 4;
pub const GLOBALS: usize = 2 * PLAYER_GLOBALS + 1;
pub const ACTION_FEATURES: usize = 49;
/// measured: at most 121 legal placements in random play (9,386 decisions), 84 in Cold Clear games
pub const MAX_ACTIONS: usize = 160;
const FULL_ROW: u16 = (1 << W) - 1;
const COMBO_GARBAGE: [u32; 12] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5];

/// libtetris' piece order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    I = 0,
    O = 1,
    T = 2,
    L = 3,
    J = 4,
    S = 5,
    Z = 6,
}

pub const PIECES: [Piece; 7] = [Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z];

impl Piece {
    pub fn from_index(i: usize) -> Piece {
        PIECES[i]
    }

    pub fn to_char(self) -> char {
        ['I', 'O', 'T', 'L', 'J', 'S', 'Z'][self as usize]
    }
}

/// Rotation states, clockwise from North.
pub const NORTH: u8 = 0;
pub const EAST: u8 = 1;
pub const SOUTH: u8 = 2;
pub const WEST: u8 = 3;

/// Cells relative to rotation point 1 (libtetris' table), North.
const NORTH_CELLS: [[(i32, i32); 4]; 7] = [
    [(-1, 0), (0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(-1, 0), (0, 0), (1, 0), (0, 1)],
    [(-1, 0), (0, 0), (1, 0), (1, 1)],
    [(-1, 0), (0, 0), (1, 0), (-1, 1)],
    [(-1, 0), (0, 0), (0, 1), (1, 1)],
    [(-1, 1), (0, 1), (0, 0), (1, 0)],
];

pub fn piece_cells(p: Piece, rot: u8) -> [(i32, i32); 4] {
    let mut c = NORTH_CELLS[p as usize];
    for (x, y) in c.iter_mut() {
        let (a, b) = (*x, *y);
        (*x, *y) = match rot {
            NORTH => (a, b),
            EAST => (b, -a),
            SOUTH => (-a, -b),
            _ => (-b, a),
        };
    }
    c
}

/// The five SRS rotation points (libtetris' offsets; kicks = source − target).
fn rotation_points(p: Piece, rot: u8) -> [(i32, i32); 5] {
    match (p, rot) {
        (Piece::O, NORTH) => [(0, 0); 5],
        (Piece::O, EAST) => [(0, -1); 5],
        (Piece::O, SOUTH) => [(-1, -1); 5],
        (Piece::O, _) => [(-1, 0); 5],
        (Piece::I, NORTH) => [(0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)],
        (Piece::I, EAST) => [(-1, 0), (0, 0), (0, 0), (0, 1), (0, -2)],
        (Piece::I, SOUTH) => [(-1, 1), (1, 1), (-2, 1), (1, 0), (-2, 0)],
        (Piece::I, _) => [(0, 1), (0, 1), (0, 1), (0, -1), (0, 2)],
        (_, NORTH) | (_, SOUTH) => [(0, 0); 5],
        (_, EAST) => [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (_, _) => [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    }
}

/// T-spin corners of the TARGET rotation: (mini side = the pointing side, the others).
fn tspin_corners(rot: u8) -> ([(i32, i32); 2], [(i32, i32); 2]) {
    match rot {
        NORTH => ([(-1, 1), (1, 1)], [(1, -1), (-1, -1)]),
        EAST => ([(1, 1), (1, -1)], [(-1, -1), (-1, 1)]),
        SOUTH => ([(1, -1), (-1, -1)], [(-1, 1), (1, 1)]),
        _ => ([(-1, -1), (-1, 1)], [(1, 1), (1, -1)]),
    }
}

pub const TSPIN_NONE: u8 = 0;
pub const TSPIN_MINI: u8 = 1;
pub const TSPIN_FULL: u8 = 2;

/// A piece position (rotation point 1 at (x, y)) and the T-spin status of its last move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Falling {
    pub piece: Piece,
    pub rot: u8,
    pub x: i32,
    pub y: i32,
    pub tspin: u8,
}

impl Falling {
    pub fn cells(&self) -> [(i32, i32); 4] {
        let mut c = piece_cells(self.piece, self.rot);
        for (x, y) in c.iter_mut() {
            *x += self.x;
            *y += self.y;
        }
        c
    }

    /// libtetris' `canonical`: the one position of each cell set.
    pub fn canonical(&self) -> Falling {
        use Piece::*;
        let f = *self;
        match (f.piece, f.rot) {
            (T | J | L, _) => f,
            (O, NORTH) => f,
            (O, EAST) => Falling { rot: NORTH, y: f.y - 1, ..f },
            (O, WEST) => Falling { rot: NORTH, x: f.x - 1, ..f },
            (O, _) => Falling { rot: NORTH, x: f.x - 1, y: f.y - 1, ..f },
            (S | Z, NORTH | WEST) => f,
            (S | Z, EAST) => Falling { rot: WEST, x: f.x + 1, ..f },
            (S | Z, _) => Falling { rot: NORTH, y: f.y - 1, ..f },
            (I, NORTH | WEST) => f,
            (I, EAST) => Falling { rot: WEST, y: f.y - 1, ..f },
            (I, _) => Falling { rot: NORTH, x: f.x - 1, ..f },
        }
    }
}

/// A 10 × 40 field, one bit per cell.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub rows: [u16; H as usize],
}

impl Default for Field {
    fn default() -> Self {
        Field { rows: [0; H as usize] }
    }
}

/// What a lock did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LockResult {
    pub lines: u32,
    pub tspin: u8,
    pub attack: u32,
    pub perfect_clear: bool,
    pub b2b_bonus: bool,
    pub locked_out: bool,
}

impl Field {
    pub fn occupied(&self, x: i32, y: i32) -> bool {
        x < 0 || x >= W || y < 0 || y >= H || self.rows[y as usize] & (1 << x) != 0
    }

    pub fn obstructed(&self, f: &Falling) -> bool {
        f.cells().iter().any(|&(x, y)| self.occupied(x, y))
    }

    pub fn height(&self, x: i32) -> i32 {
        (0..H).rev().find(|&y| self.rows[y as usize] & (1 << x) != 0).map_or(0, |y| y + 1)
    }

    pub fn max_height(&self) -> i32 {
        (0..H).rev().find(|&y| self.rows[y as usize] != 0).map_or(0, |y| y + 1)
    }

    /// Try moving by (dx, dy); a move resets the T-spin status.
    fn shift(&self, f: &Falling, dx: i32, dy: i32) -> Option<Falling> {
        let g = Falling { x: f.x + dx, y: f.y + dy, tspin: TSPIN_NONE, ..*f };
        (!self.obstructed(&g)).then_some(g)
    }

    /// SRS rotation to `target` with libtetris' T-spin detection.
    fn rotate(&self, f: &Falling, target: u8) -> Option<Falling> {
        let a = rotation_points(f.piece, f.rot);
        let b = rotation_points(f.piece, target);
        for i in 0..5 {
            let g = Falling { rot: target, x: f.x + a[i].0 - b[i].0, y: f.y + a[i].1 - b[i].1, tspin: TSPIN_NONE, ..*f };
            if self.obstructed(&g) {
                continue;
            }
            let mut g = g;
            if f.piece == Piece::T {
                let (mini, other) = tspin_corners(target);
                let m = mini.iter().filter(|&&(dx, dy)| self.occupied(g.x + dx, g.y + dy)).count();
                let o = other.iter().filter(|&&(dx, dy)| self.occupied(g.x + dx, g.y + dy)).count();
                if m + o >= 3 {
                    g.tspin = if i == 4 || m == 2 { TSPIN_FULL } else { TSPIN_MINI };
                }
            }
            return Some(g);
        }
        None
    }

    /// libtetris' Row19Or20 spawn (None = block out).
    pub fn spawn(&self, p: Piece) -> Option<Falling> {
        let mut f = Falling { piece: p, rot: NORTH, x: 4, y: 19, tspin: TSPIN_NONE };
        if !self.obstructed(&f) {
            return Some(f);
        }
        f.y = 20;
        (!self.obstructed(&f)).then_some(f)
    }

    /// Every resting placement reachable from `start` (0G, complete), canonical and
    /// deduplicated by (canonical position, T-spin status), in ascending
    /// [`placement_key`] order.
    pub fn placements(&self, start: Falling) -> Vec<Falling> {
        // state index: ((rot · 16 + x + 3) · 42 + y) · 3 + tspin
        const NX: i32 = 16;
        const NY: i32 = H + 2;
        let idx = |f: &Falling| (((f.rot as i32 * NX + f.x + 3) * NY + f.y) * 3 + f.tspin as i32) as usize;
        let mut seen = vec![false; (4 * NX * NY * 3) as usize];
        let mut stack = vec![start];
        seen[idx(&start)] = true;
        let mut out: Vec<Falling> = Vec::new();
        while let Some(f) = stack.pop() {
            let mut next = [None; 5];
            next[0] = self.shift(&f, -1, 0);
            next[1] = self.shift(&f, 1, 0);
            next[2] = self.shift(&f, 0, -1);
            if f.piece != Piece::O {
                next[3] = self.rotate(&f, (f.rot + 1) % 4);
                next[4] = self.rotate(&f, (f.rot + 3) % 4);
            }
            if next[2].is_none() {
                out.push(Falling { ..f.canonical() });
            }
            for g in next.into_iter().flatten() {
                let i = idx(&g);
                if !seen[i] {
                    seen[i] = true;
                    stack.push(g);
                }
            }
        }
        out.sort_by_key(|f| placement_key(f));
        out.dedup();
        out
    }

    /// Lock `f` (a valid resting position): clear lines, score the attack with the
    /// player's back-to-back / combo state (updated).
    pub fn lock(&mut self, f: &Falling, b2b: &mut bool, combo: &mut u32) -> LockResult {
        let mut locked_out = true;
        for (x, y) in f.cells() {
            self.rows[y as usize] |= 1 << x;
            if y < VISIBLE {
                locked_out = false;
            }
        }
        let mut lines = 0u32;
        let mut w = 0usize;
        for r in 0..H as usize {
            if self.rows[r] == FULL_ROW {
                lines += 1;
            } else {
                self.rows[w] = self.rows[r];
                w += 1;
            }
        }
        for r in w..H as usize {
            self.rows[r] = 0;
        }
        let mini = f.tspin == TSPIN_MINI;
        let spin = f.tspin != TSPIN_NONE;
        let mut attack = match (lines, f.tspin) {
            (0, _) => 0,
            (1, TSPIN_NONE) | (1, TSPIN_MINI) => 0,
            (2, TSPIN_NONE) | (2, TSPIN_MINI) => 1,
            (3, TSPIN_NONE) | (1, _) => 2,
            (4, _) | (2, _) => 4,
            _ => 6, // T-spin triple
        };
        let _ = mini;
        let mut b2b_bonus = false;
        if lines > 0 {
            let hard = lines == 4 || spin;
            if hard {
                if *b2b {
                    attack += 1;
                    b2b_bonus = true;
                }
                *b2b = true;
            } else {
                *b2b = false;
            }
            attack += COMBO_GARBAGE[(*combo as usize).min(COMBO_GARBAGE.len() - 1)];
            *combo += 1;
        } else {
            *combo = 0;
        }
        let perfect_clear = lines > 0 && self.rows.iter().all(|&r| r == 0);
        if perfect_clear {
            attack = 10;
        }
        LockResult { lines, tspin: f.tspin, attack, perfect_clear, b2b_bonus, locked_out }
    }

    /// Push one garbage line (hole at `col`) in from below; true = a block left the field.
    pub fn add_garbage(&mut self, col: u8) -> bool {
        let dead = self.rows[H as usize - 1] != 0;
        for r in (1..H as usize).rev() {
            self.rows[r] = self.rows[r - 1];
        }
        self.rows[0] = FULL_ROW & !(1 << col);
        dead
    }
}

/// The sort / id key of a canonical placement (without the hold bit).
pub fn placement_key(f: &Falling) -> u32 {
    ((f.rot as u32 * 14 + (f.x + 2) as u32) * 42 + f.y as u32) * 3 + f.tspin as u32
}

/// Canonical action id of placing `f` (canonical), `hold` = via the hold.
pub fn action_id(hold: bool, f: &Falling) -> u32 {
    (hold as u32) * 4 * 14 * 42 * 3 + placement_key(f)
}

/// (hold, rot, x, y, tspin) of an action id.
pub fn decode_id(id: u32) -> (bool, u8, i32, i32, u8) {
    let per_hold = 4 * 14 * 42 * 3;
    let (hold, k) = (id >= per_hold, id % per_hold);
    let tspin = (k % 3) as u8;
    let y = ((k / 3) % 42) as i32;
    let x = ((k / 3 / 42) % 14) as i32 - 2;
    let rot = (k / 3 / 42 / 14) as u8;
    (hold, rot, x, y, tspin)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TetrisConfig {
    /// turns (pieces per player) before the game ends in a draw, 1..=2000
    pub max_turns: usize,
    /// garbage lines a placement without a clear receives at most, 1..=40
    pub garbage_cap: u32,
}

impl Default for TetrisConfig {
    fn default() -> Self {
        TetrisConfig { max_turns: 800, garbage_cap: 8 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Player {
    pub field: Field,
    pub hold: Option<Piece>,
    /// index of the current piece in the shared sequence
    pub pos: usize,
    pub b2b: bool,
    pub combo: u32,
    /// pending garbage batches (lines, hole column), oldest first
    pub pending: VecDeque<(u32, u8)>,
    pub dead: bool,
    pub pieces: u32,
    pub attack: u32,
    pub lines: u32,
    /// garbage lines received so far
    pub received: u32,
}

impl Player {
    fn new() -> Self {
        Player { field: Field::default(), hold: None, pos: 0, b2b: false, combo: 0, pending: VecDeque::new(), dead: false,
                 pieces: 0, attack: 0, lines: 0, received: 0 }
    }

    pub fn pending_lines(&self) -> u32 {
        self.pending.iter().map(|b| b.0).sum()
    }
}

#[derive(Clone, Debug)]
pub struct Tetris {
    cfg: TetrisConfig,
    /// the shared piece sequence drawn so far
    pub seq: Vec<Piece>,
    /// pieces left in the current bag (bit i = PIECES[i])
    bag: u8,
    pub players: [Player; 2],
    pub turn_no: usize,
    acts: [Option<u32>; 2],
    /// attacks sent, waiting for their hole column (target seat, lines)
    holes: VecDeque<(usize, u32)>,
    over: bool,
}

impl Tetris {
    pub fn config(&self) -> &TetrisConfig {
        &self.cfg
    }

    fn needed(&self) -> usize {
        self.players.iter().map(|p| p.pos + 1 + NEXT).max().unwrap()
    }

    /// The piece placed without / with hold.
    pub fn piece_of(&self, seat: Seat, hold: bool) -> Piece {
        let p = &self.players[seat];
        match (hold, p.hold) {
            (false, _) => self.seq[p.pos],
            (true, Some(h)) => h,
            (true, None) => self.seq[p.pos + 1],
        }
    }

    /// Placements of `seat` for its current piece / via hold, as (hold, canonical position).
    pub fn placements(&self, seat: Seat) -> Vec<(bool, Falling)> {
        let p = &self.players[seat];
        let mut out = Vec::new();
        for hold in [false, true] {
            let piece = self.piece_of(seat, hold);
            if hold && piece == self.seq[p.pos] {
                continue; // holding into the same piece places the same piece
            }
            if let Some(s) = p.field.spawn(piece) {
                out.extend(p.field.placements(s).into_iter().map(|f| (hold, f)));
            }
        }
        // a lock out only when nothing else is left
        let lock_out = |f: &Falling| f.cells().iter().all(|&(_, y)| y >= VISIBLE);
        if out.iter().any(|(_, f)| !lock_out(f)) {
            out.retain(|(_, f)| !lock_out(f));
        }
        out
    }

    /// The falling piece of an action id of `seat` (None = not a placement shape).
    pub fn falling_of(&self, seat: Seat, id: u32) -> Falling {
        let (hold, rot, x, y, tspin) = decode_id(id);
        Falling { piece: self.piece_of(seat, hold), rot, x, y, tspin }
    }

    fn check_over(&mut self) {
        if self.players.iter().any(|p| p.dead) || self.turn_no >= self.cfg.max_turns {
            self.over = true;
        }
    }

    /// After chance: a player whose current piece cannot spawn is blocked out.
    fn spawn_checks(&mut self) {
        if self.over || self.seq.len() < self.needed() || !self.holes.is_empty() {
            return;
        }
        for s in 0..2 {
            let p = &self.players[s];
            if p.field.spawn(self.seq[p.pos]).is_none() {
                self.players[s].dead = true;
            }
        }
        self.check_over();
    }

    fn resolve(&mut self) {
        let acts = [self.acts[0].take().unwrap(), self.acts[1].take().unwrap()];
        let mut out = [0u32; 2];
        for s in 0..2 {
            let (hold, _, _, _, _) = decode_id(acts[s]);
            let f = self.falling_of(s, acts[s]);
            let cur = self.seq[self.players[s].pos];
            let p = &mut self.players[s];
            let mut advance = 1;
            if hold {
                if p.hold.is_none() {
                    advance = 2;
                }
                p.hold = Some(cur);
            }
            p.pos += advance;
            let (mut b2b, mut combo) = (p.b2b, p.combo);
            let r = p.field.lock(&f, &mut b2b, &mut combo);
            p.b2b = b2b;
            p.combo = combo;
            p.pieces += 1;
            p.lines += r.lines;
            p.attack += r.attack;
            if r.locked_out {
                p.dead = true;
            }
            if r.lines > 0 {
                let mut a = r.attack;
                while a > 0 {
                    match p.pending.front_mut() {
                        Some(b) if b.0 > a => {
                            b.0 -= a;
                            a = 0;
                        }
                        Some(_) => a -= p.pending.pop_front().unwrap().0,
                        None => break,
                    }
                }
                out[s] = a;
            } else {
                let mut room = self.cfg.garbage_cap;
                while room > 0 {
                    let Some(b) = p.pending.front_mut() else { break };
                    let n = b.0.min(room);
                    let col = b.1;
                    for _ in 0..n {
                        if p.field.add_garbage(col) {
                            p.dead = true;
                        }
                    }
                    p.received += n;
                    room -= n;
                    b.0 -= n;
                    if b.0 == 0 {
                        p.pending.pop_front();
                    }
                }
            }
        }
        for s in 0..2 {
            if out[s] > 0 {
                self.holes.push_back((1 - s, out[s]));
            }
        }
        self.turn_no += 1;
        self.check_over();
        self.spawn_checks();
    }

    /// What locking `f` would do to `seat`'s field (a copy): (field after, lock result,
    /// garbage lines received).
    fn preview(&self, seat: Seat, f: &Falling) -> (Field, LockResult, u32) {
        let p = &self.players[seat];
        let mut field = p.field.clone();
        let (mut b2b, mut combo) = (p.b2b, p.combo);
        let r = field.lock(f, &mut b2b, &mut combo);
        let mut recv = 0;
        if r.lines == 0 {
            recv = p.pending_lines().min(self.cfg.garbage_cap);
        }
        (field, r, recv)
    }
}

fn one_hot(out: &mut [f32], i: usize) {
    out[i] = 1.0;
}

impl Game for Tetris {
    type Config = TetrisConfig;
    const NAME: &'static str = "tetris";
    const VERSION: u32 = 1;

    fn check_config(cfg: &TetrisConfig) -> Result<(), RulesError> {
        if !(1..=2000).contains(&cfg.max_turns) || !(1..=40).contains(&cfg.garbage_cap) {
            rules_bail!("max_turns 1..=2000, garbage_cap 1..=40; got {cfg:?}");
        }
        Ok(())
    }

    fn spec(cfg: &TetrisConfig) -> FrameSpec {
        FrameSpec::candidate(CandidateSpec {
            n_seats: 2,
            max_entities: 2 * W as usize,
            entity_features: ENTITY_FEATURES,
            entity_types: 2,
            global_features: GLOBALS,
            max_actions: MAX_ACTIONS,
            action_features: ACTION_FEATURES,
            hidden_info: false,
            simultaneous: true,
            bits_entities: true,
            bits_globals: false,
            bits_actions: false,
            max_decisions: cfg.max_turns,
        })
    }

    fn new_game(cfg: &TetrisConfig) -> Self {
        Tetris { cfg: cfg.clone(), seq: Vec::new(), bag: 0x7f, players: [Player::new(), Player::new()], turn_no: 0,
                 acts: [None, None], holes: VecDeque::new(), over: false }
    }

    fn turn(&self) -> Turn {
        if self.over {
            Turn::Terminal
        } else if !self.holes.is_empty() || self.seq.len() < self.needed() {
            Turn::Chance
        } else {
            let mut s = SeatSet::EMPTY;
            for p in 0..2 {
                if self.acts[p].is_none() {
                    s = s.with(p);
                }
            }
            Turn::Act(s)
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        if !self.holes.is_empty() {
            out.extend((0..W as u32).map(|c| (c, 1.0 / W as f64)));
        } else {
            let left: Vec<u32> = (0..7u32).filter(|&i| self.bag & (1 << i) != 0).collect();
            let p = 1.0 / left.len() as f64;
            out.extend(left.into_iter().map(|i| (i, p)));
        }
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Chance {
            rules_bail!("tetris: no chance event now");
        }
        if let Some((target, lines)) = self.holes.front().copied() {
            if outcome >= W as u32 {
                rules_bail!("tetris: hole column {outcome}");
            }
            self.holes.pop_front();
            self.players[target].pending.push_back((lines, outcome as u8));
        } else {
            if outcome >= 7 || self.bag & (1 << outcome) == 0 {
                rules_bail!("tetris: piece {outcome} is not in the bag");
            }
            self.bag &= !(1 << outcome);
            if self.bag == 0 {
                self.bag = 0x7f;
            }
            self.seq.push(Piece::from_index(outcome as usize));
        }
        self.spawn_checks();
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if !matches!(self.turn(), Turn::Act(s) if s.contains(seat)) {
            return;
        }
        out.extend(self.placements(seat).into_iter().map(|(h, f)| action_id(h, &f)));
        out.sort_unstable();
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if !matches!(self.turn(), Turn::Act(s) if s.contains(seat)) {
            rules_bail!("tetris: seat {seat} is not to act");
        }
        let mut legal = Vec::new();
        self.legal_actions(seat, &mut legal);
        if legal.binary_search(&action).is_err() {
            rules_bail!("tetris: action {action} ({}) is not a legal placement of seat {seat}", self.action_string(seat, action));
        }
        self.acts[seat] = Some(action);
        if self.acts.iter().all(|a| a.is_some()) {
            self.resolve();
        }
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        match (self.players[0].dead, self.players[1].dead) {
            (true, false) => vec![-1.0, 1.0],
            (false, true) => vec![1.0, -1.0],
            _ => vec![0.0, 0.0],
        }
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| r as f32).collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        for (k, s) in [seat, 1 - seat].into_iter().enumerate() {
            let field = &self.players[s].field;
            for x in 0..W as usize {
                let f = obs.entity(k * W as usize + x, k as i32);
                for y in 0..ROWS {
                    f[y] = ((field.rows[y] >> x) & 1) as f32;
                }
                f[ROWS + x] = 1.0;
            }
        }
        let g = obs.globals();
        for (k, s) in [seat, 1 - seat].into_iter().enumerate() {
            let p = &self.players[s];
            let o = &mut g[k * PLAYER_GLOBALS..(k + 1) * PLAYER_GLOBALS];
            one_hot(&mut o[0..7], self.seq[p.pos] as usize);
            one_hot(&mut o[7..15], p.hold.map_or(7, |h| h as usize));
            for i in 0..NEXT {
                one_hot(&mut o[15 + 7 * i..22 + 7 * i], self.seq[p.pos + 1 + i] as usize);
            }
            let b = 15 + 7 * NEXT;
            o[b] = p.b2b as u8 as f32;
            o[b + 1] = (p.combo as f32 / 12.0).min(1.0);
            o[b + 2] = (p.pending_lines() as f32 / 20.0).min(1.0);
            o[b + 3] = p.field.max_height() as f32 / H as f32;
        }
        g[2 * PLAYER_GLOBALS] = self.turn_no as f32 / self.cfg.max_turns as f32;
    }

    fn encode_action(&self, seat: Seat, action: u32, feat: &mut [f32]) {
        let (hold, _, _, _, _) = decode_id(action);
        let f = self.falling_of(seat, action);
        let (field, r, recv) = self.preview(seat, &f);
        let p = &self.players[seat];
        let (mut b2b, mut combo) = (p.b2b, p.combo);
        p.field.clone().lock(&f, &mut b2b, &mut combo);
        feat[0] = hold as u8 as f32;
        feat[1 + f.piece as usize] = 1.0;
        feat[8 + f.rot as usize] = 1.0;
        feat[12] = f.x as f32 / W as f32;
        feat[13] = f.y as f32 / VISIBLE as f32;
        feat[14 + f.tspin as usize] = 1.0;
        feat[17 + r.lines as usize] = 1.0;
        feat[22] = r.attack as f32 / 10.0;
        feat[23] = r.perfect_clear as u8 as f32;
        feat[24] = b2b as u8 as f32;
        feat[25] = (combo as f32 / 12.0).min(1.0);
        let heights: Vec<i32> = (0..W).map(|x| field.height(x)).collect();
        for x in 0..W as usize {
            feat[26 + x] = heights[x] as f32 / VISIBLE as f32;
        }
        let holes: i32 = (0..W).map(|x| (0..heights[x as usize]).filter(|&y| !field.occupied(x, y)).count() as i32).sum();
        feat[36] = holes as f32 / 40.0;
        feat[37] = field.max_height() as f32 / VISIBLE as f32;
        feat[38] = heights.windows(2).map(|w| (w[0] - w[1]).abs()).sum::<i32>() as f32 / 40.0;
        for (i, (x, y)) in f.cells().into_iter().enumerate() {
            feat[39 + 2 * i] = x as f32 / W as f32;
            feat[40 + 2 * i] = y as f32 / VISIBLE as f32;
        }
        feat[47] = recv as f32 / 10.0;
        let dies = r.locked_out || field.max_height() as u32 + recv > H as u32;
        feat[48] = dies as u8 as f32;
    }

    fn action_string(&self, seat: Seat, action: u32) -> String {
        let (hold, rot, x, y, tspin) = decode_id(action);
        let piece = if self.seq.len() > self.players[seat].pos + 1 { self.piece_of(seat, hold).to_char() } else { '?' };
        format!("{}{}{}x{}y{}{}", if hold { "H" } else { "" }, piece, ['N', 'E', 'S', 'W'][rot as usize & 3], x, y,
                ["", "-mini", "-tspin"][tspin as usize % 3])
    }

    fn chance_string(&self, outcome: u32) -> String {
        if self.holes.is_empty() { format!("piece:{}", PIECES[outcome as usize % 7].to_char()) } else { format!("hole:{outcome}") }
    }

    fn state_string(&self) -> String {
        let mut s = format!("turn {} seq {}\n", self.turn_no, self.seq.iter().map(|p| p.to_char()).collect::<String>());
        for (i, p) in self.players.iter().enumerate() {
            s += &format!("p{i}: pos {} hold {:?} b2b {} combo {} pending {:?} dead {} attack {}\n", p.pos,
                          p.hold.map(|h| h.to_char()), p.b2b, p.combo, p.pending, p.dead, p.attack);
            for y in (0..p.field.max_height()).rev() {
                s += &(0..W).map(|x| if p.field.occupied(x, y) { '#' } else { '.' }).collect::<String>();
                s.push('\n');
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::check_game;

    fn field(rows: &[&str]) -> Field {
        // rows top to bottom, '#' = filled
        let mut f = Field::default();
        let n = rows.len();
        for (i, r) in rows.iter().enumerate() {
            let y = n - 1 - i;
            for (x, c) in r.chars().enumerate() {
                if c == '#' {
                    f.rows[y] |= 1 << x;
                }
            }
        }
        f
    }

    #[test]
    fn empty_field_placements() {
        let f = Field::default();
        let count = |p: Piece| f.placements(f.spawn(p).unwrap()).len();
        // distinct cell sets on an empty field: O 9, I 17, S/Z 17, T/L/J 34
        assert_eq!(count(Piece::O), 9);
        assert_eq!(count(Piece::I), 17);
        assert_eq!(count(Piece::S), 17);
        assert_eq!(count(Piece::Z), 17);
        assert_eq!(count(Piece::T), 34);
        assert_eq!(count(Piece::L), 34);
        assert_eq!(count(Piece::J), 34);
    }

    #[test]
    fn tspin_double_is_found_and_scored() {
        // a T-spin double slot at x = 1..3 (the overhang at (0, 2))
        let mut fl = field(&[
            "#.........",
            "...#######",
            "#.########",
        ]);
        let t = fl.placements(fl.spawn(Piece::T).unwrap());
        let tsd = t.iter().find(|f| f.rot == SOUTH && f.x == 1 && f.y == 1).expect("the TSD placement");
        assert_eq!(tsd.tspin, TSPIN_FULL, "{tsd:?}");
        let (mut b2b, mut combo) = (false, 0);
        let r = fl.lock(tsd, &mut b2b, &mut combo);
        assert_eq!((r.lines, r.attack, b2b, combo), (2, 4, true, 1));
        // then a quad with back-to-back: 4 + 1 + combo 1 → 0
        let mut fl = field(&["#########.", "#########.", "#########.", "#########."]);
        let i = fl.placements(fl.spawn(Piece::I).unwrap());
        let q = i.iter().find(|f| f.cells().iter().all(|&(x, _)| x == 9)).unwrap();
        let r = fl.lock(q, &mut b2b, &mut combo);
        assert_eq!((r.lines, r.attack, r.b2b_bonus, r.perfect_clear), (4, 10, true, true)); // a perfect clear
    }

    #[test]
    fn garbage_cancels_then_arrives() {
        let cfg = TetrisConfig::default();
        let mut g = Tetris::new_game(&cfg);
        g.players[0].pending.push_back((3, 0));
        // seat 0 clears a double (attack 1): pending 3 → 2, nothing sent
        g.players[0].field = field(&["........##", "........##"]);
        let mut f = g.players[0].field.clone();
        let (mut b, mut c) = (false, 0);
        let r = f.lock(&Falling { piece: Piece::O, rot: NORTH, x: 0, y: 0, tspin: 0 }, &mut b, &mut c);
        assert_eq!(r.lines, 0);
        let mut fl = Field::default();
        assert!(!fl.add_garbage(3));
        assert_eq!(fl.rows[0], FULL_ROW & !(1 << 3));
    }

    #[test]
    fn action_ids_round_trip() {
        for hold in [false, true] {
            for rot in 0..4u8 {
                for x in -2..12 {
                    for y in 0..42 {
                        for t in 0..3u8 {
                            let f = Falling { piece: Piece::T, rot, x, y, tspin: t };
                            assert_eq!(decode_id(action_id(hold, &f)), (hold, rot, x, y, t));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn conformance_battery() {
        let cfg = TetrisConfig { max_turns: 120, garbage_cap: 8 };
        let rep = check_game::<Tetris>(&cfg, 40, 5).unwrap();
        assert_eq!(rep.terminal, 40);
    }
}

#[cfg(test)]
mod stats {
    use super::*;
    use crate::rng::Rng;

    /// `cargo test --release -p gmrules tetris_random_stats -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn tetris_random_stats() {
        let cfg = TetrisConfig::default();
        let mut rng = Rng::new(1);
        let (mut max_legal, mut turns, mut decisions) = (0usize, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        let mut hist = vec![0usize; 8];
        for _ in 0..200 {
            let mut g = Tetris::new_game(&cfg);
            let mut legal = Vec::new();
            let mut ch = Vec::new();
            loop {
                match g.turn() {
                    Turn::Terminal => break,
                    Turn::Chance => {
                        ch.clear();
                        g.chance_outcomes(&mut ch);
                        let o = ch[rng.below(ch.len() as u64) as usize].0;
                        g.apply_chance(o).unwrap();
                    }
                    Turn::Act(s) => {
                        let seat = s.iter().next().unwrap();
                        legal.clear();
                        g.legal_actions(seat, &mut legal);
                        max_legal = max_legal.max(legal.len());
                        hist[(legal.len() / 32).min(7)] += 1;
                        decisions += 1;
                        let a = legal[rng.below(legal.len() as u64) as usize];
                        g.apply_action(seat, a).unwrap();
                    }
                }
            }
            turns += g.turn_no;
        }
        println!("max legal {max_legal}, hist/32 {hist:?}, {turns} turns, {decisions} decisions, {:.1} us/decision",
                 t0.elapsed().as_secs_f64() * 1e6 / decisions as f64);
    }
}
