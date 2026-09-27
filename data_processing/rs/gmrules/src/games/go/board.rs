//! The Go board: the rules of the IL converter (`gmdata go convert`, moved here so the
//! converter, its KataGo labeller and the simulator share ONE implementation) plus what
//! the simulator adds on top of them — the legal mask of a whole frame, the
//! Tromp-Taylor area count and a position hash.
//!
//! Rules ([`Board::play`], unchanged from the converter): a stone is placed on an empty
//! point, adjacent opponent groups left without liberties are captured, a move whose own
//! group then has no liberty (suicide, single or multi-stone) is illegal, and so is the
//! simple-ko point [`Board::ko`] — set by a move that captures exactly one stone with a
//! single stone that is left with exactly one liberty, the captured point, and cleared
//! (or replaced) by the next move of either colour, a pass included. The ko point binds
//! whoever moves next (the converter replays SGF colours, which need not alternate).
//!
//! Coordinates: `(row, col)`, row 0 = the top = SGF row `a`, point index `row * n + col`.

use std::collections::HashSet;

/// `(row, col)`.
pub type Point = (i32, i32);

pub const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;

pub fn opponent(c: u8) -> u8 {
    if c == BLACK {
        WHITE
    } else {
        BLACK
    }
}

/// A Go position with the converter's rules (captures, suicide rejected, simple ko):
/// `g` = N² points row-major (row 0 = SGF row `a` = the top), `ko` = the point the
/// side to move may not play (set by a single-stone capture that could be retaken).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub n: i32,
    pub g: Vec<u8>,
    pub ko: Option<Point>,
}

impl Board {
    pub fn new(n: i32) -> Self {
        Board { n, g: vec![EMPTY; (n * n) as usize], ko: None }
    }
    pub fn at(&self, p: Point) -> u8 {
        self.g[(p.0 * self.n + p.1) as usize]
    }
    pub fn set(&mut self, p: Point, v: u8) {
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
    /// Place setup stones (SGF AB / AW); false = a point off the board or occupied.
    pub fn setup(&mut self, color: u8, pts: &[Point]) -> bool {
        for &(r, c) in pts {
            if !(0..self.n).contains(&r) || !(0..self.n).contains(&c) || self.at((r, c)) != EMPTY {
                return false;
            }
            self.set((r, c), color);
        }
        true
    }
    /// Play `color`'s move (None = pass); false = illegal (board unchanged).
    pub fn play(&mut self, color: u8, p: Option<Point>) -> bool {
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

    // ---- simulator additions (the converter's rules above are untouched) ------------

    /// The up-to-4 neighbours of point index `i`: `(ids, how many)`.
    fn neighbour_ids(&self, i: usize) -> ([usize; 4], usize) {
        let n = self.n as usize;
        let (r, c) = (i / n, i % n);
        let mut out = [0usize; 4];
        let mut k = 0;
        if r > 0 {
            out[k] = i - n;
            k += 1;
        }
        if r + 1 < n {
            out[k] = i + n;
            k += 1;
        }
        if c > 0 {
            out[k] = i - 1;
            k += 1;
        }
        if c + 1 < n {
            out[k] = i + 1;
            k += 1;
        }
        (out, k)
    }

    /// The legal mask of `color`'s placements (`out`: N² entries, 1 = legal) — exactly
    /// the points where [`play`](Board::play) succeeds, computed in one pass: a point is
    /// legal iff it is empty, not the ko point, and has an empty neighbour, or an
    /// opponent neighbour group in atari (the move captures it), or an own neighbour
    /// group with a second liberty (the joined group keeps one). Otherwise no neighbour
    /// group is captured and every own one had this point as its last liberty: suicide.
    pub fn legal_points(&self, color: u8, out: &mut [u8]) {
        let nn = self.g.len();
        assert_eq!(out.len(), nn, "legal_points: {} entries for {nn} points", out.len());
        // group id of every stone + the number of distinct liberties of every group
        let mut gid = vec![u32::MAX; nn];
        let mut libs: Vec<u32> = Vec::new();
        let mut mark = vec![u32::MAX; nn];
        let mut stack = Vec::new();
        for s in 0..nn {
            if self.g[s] == EMPTY || gid[s] != u32::MAX {
                continue;
            }
            let (id, col) = (libs.len() as u32, self.g[s]);
            let mut count = 0u32;
            gid[s] = id;
            stack.push(s);
            while let Some(q) = stack.pop() {
                let (nb, k) = self.neighbour_ids(q);
                for &x in &nb[..k] {
                    let v = self.g[x];
                    if v == EMPTY {
                        if mark[x] != id {
                            mark[x] = id;
                            count += 1;
                        }
                    } else if v == col && gid[x] == u32::MAX {
                        gid[x] = id;
                        stack.push(x);
                    }
                }
            }
            libs.push(count);
        }
        let n = self.n;
        let ko = self.ko.map(|(r, c)| (r * n + c) as usize);
        let opp = opponent(color);
        for (p, o) in out.iter_mut().enumerate() {
            *o = 0;
            if self.g[p] != EMPTY || ko == Some(p) {
                continue;
            }
            let (nb, k) = self.neighbour_ids(p);
            let legal = nb[..k].iter().any(|&x| match self.g[x] {
                EMPTY => true,
                v if v == opp => libs[gid[x] as usize] == 1,
                _ => libs[gid[x] as usize] >= 2,
            });
            *o = legal as u8;
        }
    }

    /// Tromp-Taylor area `(black, white)`: a colour's stones plus the empty points from
    /// which only that colour's stones can be reached through empty points (an empty
    /// region bordering both colours, or none, counts for nobody). Every stone on the
    /// board counts as alive: the game is played out, dead stones are captured first.
    pub fn area(&self) -> (u32, u32) {
        let nn = self.g.len();
        let (mut black, mut white) = (0u32, 0u32);
        for &v in &self.g {
            match v {
                BLACK => black += 1,
                WHITE => white += 1,
                _ => {}
            }
        }
        let mut seen = vec![false; nn];
        let mut stack = Vec::new();
        for s in 0..nn {
            if self.g[s] != EMPTY || seen[s] {
                continue;
            }
            let (mut size, mut borders) = (0u32, 0u8);
            seen[s] = true;
            stack.push(s);
            while let Some(q) = stack.pop() {
                size += 1;
                let (nb, k) = self.neighbour_ids(q);
                for &x in &nb[..k] {
                    match self.g[x] {
                        EMPTY if !seen[x] => {
                            seen[x] = true;
                            stack.push(x);
                        }
                        EMPTY => {}
                        v => borders |= v,
                    }
                }
            }
            match borders {
                BLACK => black += size,
                WHITE => white += size,
                _ => {}
            }
        }
        (black, white)
    }

    /// Zobrist hash of the stones (positional: the ko point and the side to move are
    /// not part of it) — the positional-superko history of the simulator. 64 bits: two
    /// distinct positions of one game collide with probability ~ moves² / 2⁶⁵.
    pub fn position_hash(&self) -> u64 {
        self.g
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != EMPTY)
            .fold(0u64, |h, (p, &v)| h ^ splitmix64(((p as u64) << 2) | v as u64))
    }
}

/// SplitMix64 finaliser (the Zobrist keys: a pure function of point and colour).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
