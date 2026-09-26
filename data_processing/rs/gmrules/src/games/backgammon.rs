//! Backgammon with the doubling cube — match play (default: 7-point matches, Crawford
//! rule; one episode = one whole match) or money play (one game per episode). The
//! rules are gnubg's (GNU Backgammon), pinned by randomized differential tests of the
//! legal move sets against gnubg itself and by replaying gnubg-played matches
//! (tests/test_backgammon.py).
//!
//! Why match play by default: its result is exactly a win or a loss (outcome ±1, the
//! framework's [-1, 1] contract with no scaling), it is the format of tournament play,
//! and it contains everything money play has (cube, gammons, backgammons) plus the
//! score-dependent strategy and the Crawford rule. Money play (`match_length = 0`) is
//! the other real format; its payoff (points × cube) is unbounded, so its outcome is
//! `clamp(points / money_scale, -1, 1)` (`returns` = the exact points).
//!
//! Rules (seat 0 and seat 1; every seat counts points from its own side: its home
//! board is points 1..6, it moves from 24 towards 1 and bears off from its home board).
//! - A game starts with the opening roll: each seat throws one die (ties re-thrown —
//!   the chance outcome is the ordered pair with d0 != d1, probability 1/30 each); the
//!   higher die moves first, playing both dice. The cube is not offered before the
//!   opening move.
//! - A turn: the player on roll may double first (cube centred or owned by them, not
//!   the Crawford game, cube < `max_cube`); the opponent takes (cube value doubles,
//!   the taker owns it) or drops (the doubler wins the current cube value). Then the
//!   dice (21 distinct rolls; doubles are played four times).
//! - Checker play: checkers on the bar enter first (opponent's home board, points
//!   25 − die); a point held by two or more opposing checkers is blocked; landing on a
//!   single opposing checker hits it (to the bar). Bearing off once all checkers are
//!   home: a die equal to the point, or a larger die from the highest occupied point.
//!   As many dice as possible must be played; if only one of two different dice can
//!   be played, the larger one must be played when possible. No legal move = pass.
//! - The game ends when a player has borne off all 15 checkers: they win cube value ×
//!   1 (single), × 2 (gammon: the loser has borne off none) or × 3 (backgammon: none
//!   borne off and a checker on the bar or in the winner's home board). Money play
//!   with the Jacoby rule: gammons count only once the cube has been turned. A drop
//!   ends the game for the current cube value (no gammons).
//! - Match play: points go to the winner's score; the match ends at `match_length`.
//!   The game after a player first reaches `match_length − 1` points is the Crawford
//!   game (no doubling); the games after it are post-Crawford (doubling allowed).
//!   Doubling is legal whenever the cube is accessible, also when it is "dead" (the
//!   doubler already wins the match with the current cube) — legal, never useful.
//!
//! Decisions and canonical ids (strictly ascending = the candidate order):
//! ```text
//! 0 no double (roll)   1 double   2 take   3 drop   4 pass (no legal checker move)
//! 16 + code             one checker move (from, die) of a SPLIT turn (see below)
//! >= 9 << 24            a full move: the multiset of its checker moves, codes
//!                       (from·8 + die, from 1..=25 with 25 = the bar) sorted
//!                       descending, packed big-endian into the four bytes
//! ```
//! A checker-play decision's candidates are the DISTINCT resulting positions (like
//! gnubg's move list); each is represented by its smallest packed id. A roll with
//! more than [`MAX_ACTIONS`] distinct moves (only big doubles in spread positions) is
//! SPLIT: the player picks one checker move at a time (candidates = the single checker
//! moves that keep a maximal full move reachable) until the remaining completions fit,
//! then picks among the completions. Every legal move stays reachable, one decision
//! per frame. Chance ids: roll = 6·(hi−1) + (lo−1) (hi >= lo, p = 1/36 doubles, 2/36
//! otherwise), opening roll = 36 + 6·(d0−1) + (d1−1) (d0 = seat 0's die, p = 1/30).
//!
//! Frames (perfect information → one interleaved stream per episode, every frame from
//! the acting seat's view; the taker acts on a take decision):
//! ```text
//! entities [28, 13]  slots 0..23 = the actor's points 1..24 (type 0), 24 own bar,
//!                    25 opponent's bar (type 1), 26 own borne-off, 27 opponent's
//!                    borne-off (type 2): own count ≥1 ≥2 ≥3 ≥4 (n−4)/11, opponent
//!                    count (same five), point/25, own home board, opponent's home board
//! globals  [45]      decision kind (cube / take / move / split move), the roll (hi and
//!                    lo one-hot, double), dice left, checker moves played, cube (log2 /
//!                    12, owner me / centre / opponent, who can double, dead cube),
//!                    actor on roll, money / Jacoby, match length, scores, away, 1-away,
//!                    Crawford / post-Crawford, games played, pip counts, contact
//! actions  [128, 119] (bits) kind (pass / full / split move / no double / double /
//!                    take / drop), the actor's checkers per point AFTER the move
//!                    (≥1 ≥2 ≥3), opposing blots hit per point, own bar after,
//!                    checkers borne off, hits, dice left (split moves), wins the game,
//!                    gammon saved
//! ```
//! Outcome (IL value target, RL reward): match play ±1 (match won / lost); money play
//! clamp(points / money_scale, −1, 1).

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const NO_DOUBLE: u32 = 0;
pub const DOUBLE: u32 = 1;
pub const TAKE: u32 = 2;
pub const DROP: u32 = 3;
pub const PASS: u32 = 4;
/// Id of the split-turn checker move with code c = from·8 + die: `SUBMOVE_BASE + c`.
pub const SUBMOVE_BASE: u32 = 16;
/// Smallest full-move id (one checker move from point 1 with a 1).
pub const FULL_MOVE_MIN: u32 = 9 << 24;
/// Chance id of the opening roll (d0, d1): `OPENING_BASE + 6·(d0−1) + (d1−1)`.
pub const OPENING_BASE: u32 = 36;
/// The bar as a "from" point.
pub const BAR: usize = 25;
pub const CHECKERS: u8 = 15;
/// Candidate slots of a frame: the largest checker-play decision presented whole.
pub const MAX_ACTIONS: usize = 128;
/// gnubg's MAX_CUBE.
pub const MAX_CUBE: u32 = 4096;

pub const ENTITIES: usize = 28;
pub const ENTITY_FEATURES: usize = 13;
pub const GLOBALS: usize = 45;
pub const ACTION_FEATURES: usize = 119;

/// One seat's checkers in its own numbering: `[i]` = point i+1 (i < 24), `[24]` = bar;
/// borne off = 15 − sum.
pub type Side = [u8; 25];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BackgammonConfig {
    /// Points to win the match, 1..=25; 0 = money play (one game per episode).
    pub match_length: u32,
    /// Crawford rule (match play).
    pub crawford: bool,
    /// Money play: gammons / backgammons count only once the cube was turned.
    pub jacoby: bool,
    /// The doubling cube is in play.
    pub cube: bool,
    /// Largest cube value (a power of two, 1..=4096): no double beyond it.
    pub max_cube: u32,
    /// Money play: outcome = clamp(points / money_scale, −1, 1).
    pub money_scale: f64,
    /// Match play: the score the match starts at (seat 0, seat 1), each below
    /// `match_length` (default 0-0). With a player at `match_length − 1` the first game is
    /// the Crawford game unless `post_crawford`.
    pub score: [u32; 2],
    /// Match play: the Crawford game has already been played (a player is at
    /// `match_length − 1`).
    pub post_crawford: bool,
}

impl Default for BackgammonConfig {
    fn default() -> Self {
        BackgammonConfig {
            match_length: 7,
            crawford: true,
            jacoby: true,
            cube: true,
            max_cube: MAX_CUBE,
            money_scale: 4.0,
            score: [0, 0],
            post_crawford: false,
        }
    }
}

/// A position from the point of view of one seat ("me"): both sides in their own
/// numbering (the opponent's point j is my point 25 − j).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pos {
    pub me: Side,
    pub opp: Side,
}

pub fn off(s: &Side) -> u8 {
    CHECKERS - s.iter().sum::<u8>()
}

pub fn pips(s: &Side) -> u32 {
    (0..24).map(|i| (i as u32 + 1) * s[i] as u32).sum::<u32>() + 25 * s[24] as u32
}

fn all_home(s: &Side) -> bool {
    s[6..25].iter().all(|&c| c == 0)
}

/// Is there contact (some checker of mine can still meet an opposing one)?
pub fn contact(p: &Pos) -> bool {
    if p.me[24] > 0 || p.opp[24] > 0 {
        return true;
    }
    // my rearmost checker vs the opponent's rearmost checker (in my numbering)
    let my_back = (0..24).rev().find(|&i| p.me[i] > 0).map_or(0, |i| i + 1);
    let opp_back = (0..24).rev().find(|&j| p.opp[j] > 0).map_or(25, |j| 24 - j); // my point of it
    my_back > opp_back
}

impl Pos {
    /// From (point, count) lists in each side's own numbering (25 = bar).
    pub fn from_points(me: &[(usize, u8)], opp: &[(usize, u8)]) -> Pos {
        let mut p = Pos { me: [0; 25], opp: [0; 25] };
        for &(pt, n) in me {
            p.me[pt - 1] += n;
        }
        for &(pt, n) in opp {
            p.opp[pt - 1] += n;
        }
        p
    }

    /// Checks: at most 15 checkers a side, no point held by both sides.
    pub fn validate(&self) -> Result<(), String> {
        let (a, b) = (self.me.iter().map(|&c| c as u32).sum::<u32>(), self.opp.iter().map(|&c| c as u32).sum::<u32>());
        if a > 15 || b > 15 {
            return Err(format!("more than 15 checkers ({a}, {b})"));
        }
        for q in 1..=24 {
            if self.me[q - 1] > 0 && self.opp[24 - q] > 0 {
                return Err(format!("point {q} held by both sides"));
            }
        }
        Ok(())
    }

    /// The same position from the opponent's point of view.
    pub fn flip(&self) -> Pos {
        Pos { me: self.opp, opp: self.me }
    }
}

/// The starting position of a side.
pub fn start_side() -> Side {
    let mut s = [0u8; 25];
    s[23] = 2;
    s[12] = 5;
    s[7] = 3;
    s[5] = 5;
    s
}

/// Can I move a checker from `from` (1..=25, 25 = bar) with `die`?
pub fn can_play(p: &Pos, from: usize, die: u8) -> bool {
    if from == 0 || from > 25 || p.me[from - 1] == 0 {
        return false;
    }
    if p.me[24] > 0 && from != BAR {
        return false;
    }
    let to = from as i32 - die as i32;
    if to >= 1 {
        p.opp[24 - to as usize] < 2
    } else {
        if !all_home(&p.me) {
            return false;
        }
        // exact, or a larger die from the highest occupied point
        to == 0 || p.me[from..6].iter().all(|&c| c == 0)
    }
}

/// Move a checker from `from` with `die` (must be playable); returns whether it hit.
pub fn apply_sub(p: &mut Pos, from: usize, die: u8) -> bool {
    p.me[from - 1] -= 1;
    let to = from as i32 - die as i32;
    if to >= 1 {
        let q = to as usize;
        let hit = p.opp[24 - q] == 1;
        if hit {
            p.opp[24 - q] = 0;
            p.opp[24] += 1;
        }
        p.me[q - 1] += 1;
        hit
    } else {
        false
    }
}

/// Apply a move given as (from, to) checker moves in my numbering (from 25 = bar, to
/// 0 = off, the format of gnubg's move tuples), checking each step against the
/// board (not the dice). Landing on a single opposing checker hits it.
pub fn apply_steps(p: &Pos, steps: &[(usize, usize)]) -> Result<Pos, String> {
    let mut q = *p;
    for &(from, to) in steps {
        if !(1..=25).contains(&from) || to >= from || to > 24 {
            return Err(format!("bad checker move {from}/{to}"));
        }
        if q.me[from - 1] == 0 {
            return Err(format!("no checker on {from} for {from}/{to}"));
        }
        q.me[from - 1] -= 1;
        if to >= 1 {
            let oi = 24 - to;
            match q.opp[oi] {
                0 => {}
                1 => {
                    q.opp[oi] = 0;
                    q.opp[24] += 1;
                }
                n => return Err(format!("{from}/{to} lands on a point held by {n} opposing checkers")),
            }
            q.me[to - 1] += 1;
        }
    }
    Ok(q)
}

fn code(from: usize, die: u8) -> u32 {
    from as u32 * 8 + die as u32
}

/// Pack checker-move codes (the order is irrelevant) into a full-move id.
pub fn pack(codes: &[u32]) -> u32 {
    let mut c: Vec<u32> = codes.to_vec();
    c.sort_unstable_by(|a, b| b.cmp(a));
    c.iter().enumerate().fold(0, |acc, (i, &x)| acc | x << (24 - 8 * i))
}

/// The (from, die) checker moves of a full-move id.
pub fn unpack(id: u32) -> Vec<(usize, u8)> {
    (0..4).map(|i| (id >> (24 - 8 * i)) & 0xFF).filter(|&c| c != 0).map(|c| ((c / 8) as usize, (c % 8) as u8)).collect()
}

/// Dice left to play this turn (sorted descending).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Dice {
    d: [u8; 4],
    n: u8,
}

impl Dice {
    pub fn roll(a: u8, b: u8) -> Dice {
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        if hi == lo {
            Dice { d: [hi; 4], n: 4 }
        } else {
            Dice { d: [hi, lo, 0, 0], n: 2 }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.d[..self.n as usize]
    }

    pub fn len(&self) -> usize {
        self.n as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn without(&self, j: usize) -> Dice {
        let mut out = Dice::default();
        for (k, &d) in self.as_slice().iter().enumerate() {
            if k != j {
                out.d[out.n as usize] = d;
                out.n += 1;
            }
        }
        out
    }

    fn index_of(&self, die: u8) -> Option<usize> {
        self.as_slice().iter().position(|&d| d == die)
    }
}

/// A complete checker-move sequence of a turn (from one state).
struct Seq {
    result: Pos,
    codes: [u32; 4],
    n: u8,
}

fn dfs(pos: &Pos, dice: Dice, codes: &mut [u32; 4], depth: usize, out: &mut Vec<Seq>, seen: &mut HashSet<(Pos, Dice)>) {
    let mut any = false;
    let ds = dice.as_slice();
    for j in 0..ds.len() {
        let d = ds[j];
        if j > 0 && ds[j - 1] == d {
            continue; // this die value was just tried
        }
        for from in (1..=25).rev() {
            if !can_play(pos, from, d) {
                continue;
            }
            any = true;
            let mut np = *pos;
            apply_sub(&mut np, from, d);
            codes[depth] = code(from, d);
            let nd = dice.without(j);
            if nd.is_empty() {
                out.push(Seq { result: np, codes: *codes, n: depth as u8 + 1 });
            } else if seen.insert((np, nd)) {
                // a state reached again (another order of the same checker moves: with
                // one die value, or after one of two different dice, the state fixes the
                // multiset of moves played) has already been expanded
                dfs(&np, nd, codes, depth + 1, out, seen);
            }
        }
    }
    if !any {
        out.push(Seq { result: *pos, codes: *codes, n: depth as u8 });
    }
}

/// Kind of a checker-play candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pass,
    Full,
    Sub,
}

/// A checker-play candidate: its id, the position after it, and move facts.
#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub id: u32,
    pub kind: Kind,
    pub result: Pos,
    /// bit q−1: an opposing blot on my point q was hit
    pub hit_mask: u32,
    pub hits: u8,
    pub borne: u8,
    /// split moves: the die used
    pub die: u8,
}

fn make_cand(start: &Pos, result: Pos, id: u32, kind: Kind, die: u8) -> Cand {
    let mut hit_mask = 0u32;
    for q in 1..=24 {
        if start.opp[24 - q] == 1 && result.opp[24 - q] == 0 {
            hit_mask |= 1 << (q - 1);
        }
    }
    Cand { id, kind, result, hit_mask, hits: hit_mask.count_ones() as u8, borne: off(&result.me) - off(&start.me), die }
}

/// The legal moves of a (partial) turn from `start` with `dice` left: the distinct
/// resulting positions of the sequences playing the most dice (at the start of a
/// turn with two different dice: the larger die when only one can be played), each
/// with its smallest packed id, ascending; a lone PASS when nothing can be played.
/// Also returns the number of dice those moves play.
pub fn legal_moves(start: &Pos, dice: Dice) -> (Vec<Cand>, u8) {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut codes = [0u32; 4];
    dfs(start, dice, &mut codes, 0, &mut out, &mut seen);
    let max = out.iter().map(|s| s.n).max().unwrap_or(0);
    if max == 0 {
        return (vec![make_cand(start, *start, PASS, Kind::Pass, 0)], 0);
    }
    let mut keep: Vec<&Seq> = out.iter().filter(|s| s.n == max).collect();
    if max == 1 && dice.n == 2 && dice.d[0] != dice.d[1] {
        let hi = dice.d[0] as u32;
        if keep.iter().any(|s| s.codes[0] % 8 == hi) {
            keep.retain(|s| s.codes[0] % 8 == hi);
        }
    }
    let mut best: HashMap<Pos, u32> = HashMap::with_capacity(keep.len());
    for s in keep {
        let id = pack(&s.codes[..s.n as usize]);
        best.entry(s.result).and_modify(|b| *b = (*b).min(id)).or_insert(id);
    }
    let mut cands: Vec<Cand> = best.into_iter().map(|(r, id)| make_cand(start, r, id, Kind::Full, 0)).collect();
    cands.sort_unstable_by_key(|c| c.id);
    (cands, max)
}

/// Most dice playable from a (partial) turn state (memoised search).
fn max_playable(pos: &Pos, dice: Dice, memo: &mut HashMap<(Pos, Dice), u8>) -> u8 {
    if dice.is_empty() {
        return 0;
    }
    if let Some(&m) = memo.get(&(*pos, dice)) {
        return m;
    }
    let mut best = 0u8;
    let ds = dice.as_slice();
    'outer: for j in 0..ds.len() {
        let d = ds[j];
        if j > 0 && ds[j - 1] == d {
            continue;
        }
        for from in (1..=25).rev() {
            if can_play(pos, from, d) {
                let mut np = *pos;
                apply_sub(&mut np, from, d);
                best = best.max(1 + max_playable(&np, dice.without(j), memo));
                if best as usize == dice.len() {
                    break 'outer;
                }
            }
        }
    }
    memo.insert((*pos, dice), best);
    best
}

/// Split-turn candidates: the single checker moves after which `need − 1` more dice
/// can still be played (so a maximal move stays reachable), ascending ids.
fn split_moves(pos: &Pos, dice: Dice, need: u8) -> Vec<Cand> {
    let mut memo = HashMap::new();
    let mut out = Vec::new();
    let ds = dice.as_slice();
    for j in 0..ds.len() {
        let d = ds[j];
        if j > 0 && ds[j - 1] == d {
            continue;
        }
        for from in 1..=25 {
            if can_play(pos, from, d) {
                let mut np = *pos;
                apply_sub(&mut np, from, d);
                if max_playable(&np, dice.without(j), &mut memo) + 1 >= need {
                    out.push(make_cand(pos, np, SUBMOVE_BASE + code(from, d), Kind::Sub, d));
                }
            }
        }
    }
    out.sort_unstable_by_key(|c| c.id);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// chance: the opening roll of a game
    Opening,
    /// the player on roll may double
    Cube,
    /// the opponent answers a double
    Take,
    /// chance: the player on roll throws the dice
    Roll,
    /// the player on roll plays checkers
    Move,
    /// the episode is over
    Over,
}

/// How a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GameResult {
    pub winner: Seat,
    /// cube value × multiplier
    pub points: u32,
    /// 1 single, 2 gammon, 3 backgammon (before the Jacoby rule)
    pub multiplier: u8,
    pub dropped: bool,
    pub cube: u32,
}

#[derive(Clone, Debug)]
pub struct Backgammon {
    cfg: BackgammonConfig,
    board: [Side; 2],
    phase: Phase,
    on_roll: Seat,
    /// the roll (hi, lo); zeros before the dice are thrown
    roll: (u8, u8),
    /// dice left to play in this turn
    dice: Dice,
    /// checker moves played in this turn (split turns)
    played: u8,
    /// dice this turn's legal moves play
    total: u8,
    /// the current checker-play candidates (Move phase)
    cands: Vec<Cand>,
    /// the candidates are split-turn single checker moves
    split: bool,
    cube: u32,
    /// −1 centred
    owner: i8,
    score: [u32; 2],
    crawford_game: bool,
    post_crawford: bool,
    results: Vec<GameResult>,
    match_winner: Option<Seat>,
}

impl Backgammon {
    pub fn config(&self) -> &BackgammonConfig {
        &self.cfg
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn on_roll(&self) -> Seat {
        self.on_roll
    }

    pub fn roll(&self) -> (u8, u8) {
        self.roll
    }

    pub fn dice_left(&self) -> Dice {
        self.dice
    }

    pub fn cube(&self) -> (u32, i8) {
        (self.cube, self.owner)
    }

    pub fn score(&self) -> [u32; 2] {
        self.score
    }

    pub fn crawford(&self) -> (bool, bool) {
        (self.crawford_game, self.post_crawford)
    }

    pub fn results(&self) -> &[GameResult] {
        &self.results
    }

    pub fn match_winner(&self) -> Option<Seat> {
        self.match_winner
    }

    /// `seat`'s checkers in its own numbering.
    pub fn side(&self, seat: Seat) -> &Side {
        &self.board[seat]
    }

    /// The position from `seat`'s point of view.
    pub fn pos(&self, seat: Seat) -> Pos {
        Pos { me: self.board[seat], opp: self.board[1 - seat] }
    }

    /// The current checker-play candidates (Move phase).
    pub fn candidates(&self) -> &[Cand] {
        &self.cands
    }

    /// The current checker-play candidates are single checker moves of a split turn.
    pub fn is_split(&self) -> bool {
        self.split
    }

    fn money(&self) -> bool {
        self.cfg.match_length == 0
    }

    /// The seat to act at a decision.
    pub fn actor(&self) -> Option<Seat> {
        match self.phase {
            Phase::Cube | Phase::Move => Some(self.on_roll),
            Phase::Take => Some(1 - self.on_roll),
            _ => None,
        }
    }

    /// May `seat` double at the start of its turn?
    pub fn may_double(&self, seat: Seat) -> bool {
        self.cfg.cube
            && !self.crawford_game
            && (self.owner < 0 || self.owner as usize == seat)
            && self.cube < self.cfg.max_cube
    }

    /// Match play: would winning a single game at the current cube already give
    /// `seat` the match (a dead cube for it)?
    pub fn cube_dead_for(&self, seat: Seat) -> bool {
        !self.money() && self.score[seat] + self.cube >= self.cfg.match_length
    }

    fn start_turn(&mut self) {
        self.roll = (0, 0);
        self.dice = Dice::default();
        self.cands.clear();
        self.split = false;
        self.phase = if self.may_double(self.on_roll) { Phase::Cube } else { Phase::Roll };
    }

    fn start_move(&mut self, hi: u8, lo: u8) {
        self.roll = (hi, lo);
        self.dice = Dice::roll(hi, lo);
        self.played = 0;
        let pos = self.pos(self.on_roll);
        let (full, total) = legal_moves(&pos, self.dice);
        self.total = total;
        if full.len() <= MAX_ACTIONS {
            self.cands = full;
            self.split = false;
        } else {
            self.cands = split_moves(&pos, self.dice, total);
            self.split = true;
        }
        self.phase = Phase::Move;
    }

    fn new_game_board(&mut self) {
        self.board = [start_side(), start_side()];
        self.cube = 1;
        self.owner = -1;
        self.roll = (0, 0);
        self.dice = Dice::default();
        self.cands.clear();
        self.split = false;
        self.phase = Phase::Opening;
    }

    fn end_game(&mut self, winner: Seat, multiplier: u8, dropped: bool) {
        let counted = if self.money() && self.cfg.jacoby && self.cube == 1 { 1 } else { multiplier };
        let points = self.cube * counted as u32;
        self.results.push(GameResult { winner, points, multiplier, dropped, cube: self.cube });
        self.cands.clear();
        self.split = false;
        if self.money() {
            self.match_winner = Some(winner);
            self.phase = Phase::Over;
            return;
        }
        let n = self.cfg.match_length;
        self.score[winner] += points;
        if self.score[winner] >= n {
            self.match_winner = Some(winner);
            self.phase = Phase::Over;
            return;
        }
        if self.crawford_game {
            self.crawford_game = false;
            self.post_crawford = true;
        } else if self.cfg.crawford && n > 1 && !self.post_crawford && self.score.contains(&(n - 1)) {
            self.crawford_game = true;
        }
        self.new_game_board();
    }

    fn finish_turn(&mut self) {
        let me = self.on_roll;
        if off(&self.board[me]) == CHECKERS {
            let loser = &self.board[1 - me];
            let mult = if off(loser) > 0 {
                1
            } else if loser[24] > 0 || loser[18..24].iter().any(|&c| c > 0) {
                3
            } else {
                2
            };
            self.end_game(me, mult, false);
        } else {
            self.on_roll = 1 - me;
            self.start_turn();
        }
    }

    /// The actions of the current turn that play the move ending in `target` (the
    /// position from the mover's view after the whole move): one full-move id, or the
    /// split-turn checker moves (smallest id first among those that can still reach
    /// it) and then the completion. Error when `target` is not a legal result.
    pub fn plan_move(&self, target: &Pos) -> Result<Vec<u32>, String> {
        if self.phase != Phase::Move {
            return Err(format!("not at a checker-play decision ({:?})", self.phase));
        }
        let mut g = self.clone();
        let mut out = Vec::new();
        loop {
            if !g.split {
                let c = g.cands.iter().find(|c| c.result == *target).ok_or_else(|| {
                    format!("no legal move of roll {}-{} reaches the recorded position", g.roll.0, g.roll.1)
                })?;
                out.push(c.id);
                return Ok(out);
            }
            let mut next = None;
            for c in &g.cands {
                let j = g.dice.index_of(c.die).expect("split move die");
                let (moves, _) = legal_moves(&c.result, g.dice.without(j));
                if moves.iter().any(|m| m.result == *target) {
                    next = Some(c.id);
                    break;
                }
            }
            let id = next.ok_or_else(|| format!("no split move of roll {}-{} reaches the recorded position", g.roll.0, g.roll.1))?;
            out.push(id);
            let seat = g.on_roll;
            g.apply_action(seat, id).map_err(|e| e.0)?;
        }
    }

    /// Mover's position after the full-move id `id` from `pos` (no legality check).
    pub fn result_of(pos: &Pos, id: u32) -> Pos {
        let mut p = *pos;
        let mut hit_points = Vec::new();
        let subs = unpack(id);
        for &(from, _) in &subs {
            p.me[from - 1] = p.me[from - 1].wrapping_sub(1);
        }
        for &(from, die) in &subs {
            let to = from as i32 - die as i32;
            if to >= 1 {
                let q = to as usize;
                p.me[q - 1] = p.me[q - 1].wrapping_add(1);
                if pos.opp[24 - q] == 1 && !hit_points.contains(&q) {
                    hit_points.push(q);
                }
            }
        }
        for q in hit_points {
            p.opp[24 - q] = 0;
            p.opp[24] += 1;
        }
        p
    }

    fn step_string(from: usize, to: i32) -> String {
        let f = if from == BAR { "bar".to_string() } else { from.to_string() };
        let t = if to <= 0 { "off".to_string() } else { to.to_string() };
        format!("{f}/{t}")
    }
}

fn thermo(f: &mut [f32], n: u8) {
    for (k, x) in f.iter_mut().enumerate().take(4) {
        *x = (n as usize > k) as u8 as f32;
    }
    f[4] = (n.saturating_sub(4)) as f32 / 11.0;
}

impl Game for Backgammon {
    type Config = BackgammonConfig;
    const NAME: &'static str = "backgammon";
    const VERSION: u32 = 1;

    fn check_config(cfg: &BackgammonConfig) -> Result<(), RulesError> {
        if cfg.match_length > 25 {
            rules_bail!("match_length must be 0 (money play) or 1..=25, got {}", cfg.match_length);
        }
        if cfg.max_cube == 0 || cfg.max_cube > MAX_CUBE || !cfg.max_cube.is_power_of_two() {
            rules_bail!("max_cube must be a power of two in 1..={MAX_CUBE}, got {}", cfg.max_cube);
        }
        if !(cfg.money_scale.is_finite() && cfg.money_scale > 0.0) {
            rules_bail!("money_scale must be > 0, got {}", cfg.money_scale);
        }
        let n = cfg.match_length;
        if n == 0 {
            if cfg.score != [0, 0] || cfg.post_crawford {
                rules_bail!("money play has no score / Crawford state");
            }
        } else {
            if cfg.score.iter().any(|&s| s >= n) {
                rules_bail!("score {:?} must be below match_length {n}", cfg.score);
            }
            let away1 = cfg.score.iter().filter(|&&s| s == n - 1).count();
            if cfg.post_crawford && (away1 == 0 || !cfg.crawford) {
                rules_bail!("post_crawford needs the Crawford rule and a player at {} (score {:?})", n - 1, cfg.score);
            }
            if cfg.crawford && n > 1 && away1 == 2 && !cfg.post_crawford {
                rules_bail!("both players at {} happens only after the Crawford game: set post_crawford", n - 1);
            }
        }
        Ok(())
    }

    fn spec(cfg: &BackgammonConfig) -> FrameSpec {
        // a game rarely takes more than ~150 decisions (random play ~400); matches
        // last at most 2N − 1 games
        let games = if cfg.match_length == 0 { 1 } else { 2 * cfg.match_length as usize - 1 };
        FrameSpec::candidate(CandidateSpec {
            n_seats: 2,
            max_entities: ENTITIES,
            entity_features: ENTITY_FEATURES,
            entity_types: 3,
            global_features: GLOBALS,
            max_actions: MAX_ACTIONS,
            action_features: ACTION_FEATURES,
            hidden_info: false,
            simultaneous: false,
            bits_entities: false,
            bits_globals: false,
            bits_actions: true,
            max_decisions: 2000 * games,
        })
    }

    fn new_game(cfg: &BackgammonConfig) -> Self {
        let mut g = Backgammon {
            cfg: cfg.clone(),
            board: [start_side(), start_side()],
            phase: Phase::Opening,
            on_roll: 0,
            roll: (0, 0),
            dice: Dice::default(),
            played: 0,
            total: 0,
            cands: Vec::new(),
            split: false,
            cube: 1,
            owner: -1,
            score: cfg.score,
            crawford_game: false,
            post_crawford: cfg.post_crawford,
            results: Vec::new(),
            match_winner: None,
        };
        let n = cfg.match_length;
        g.crawford_game = cfg.crawford && n > 1 && !cfg.post_crawford && cfg.score.contains(&(n - 1));
        g.new_game_board();
        g
    }

    fn turn(&self) -> Turn {
        match self.phase {
            Phase::Opening | Phase::Roll => Turn::Chance,
            Phase::Over => Turn::Terminal,
            _ => Turn::Act(SeatSet::single(self.actor().unwrap())),
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        match self.phase {
            Phase::Opening => {
                for d0 in 1..=6u32 {
                    for d1 in 1..=6u32 {
                        if d0 != d1 {
                            out.push((OPENING_BASE + 6 * (d0 - 1) + (d1 - 1), 1.0 / 30.0));
                        }
                    }
                }
            }
            Phase::Roll => {
                for lo in 1..=6u32 {
                    for hi in lo..=6u32 {
                        let id = 6 * (hi - 1) + (lo - 1);
                        out.push((id, if hi == lo { 1.0 / 36.0 } else { 2.0 / 36.0 }));
                    }
                }
                out.sort_unstable_by_key(|&(id, _)| id);
            }
            _ => {}
        }
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        match self.phase {
            Phase::Opening => {
                if !(OPENING_BASE..OPENING_BASE + 36).contains(&outcome) {
                    rules_bail!("backgammon: opening roll id {outcome} outside {OPENING_BASE}..{}", OPENING_BASE + 36);
                }
                let (d0, d1) = ((outcome - OPENING_BASE) / 6 + 1, (outcome - OPENING_BASE) % 6 + 1);
                if d0 == d1 {
                    rules_bail!("backgammon: the opening roll cannot be a double ({d0}-{d1})");
                }
                self.on_roll = if d0 > d1 { 0 } else { 1 };
                self.start_move(d0.max(d1) as u8, d0.min(d1) as u8);
                Ok(())
            }
            Phase::Roll => {
                let (hi, lo) = (outcome / 6 + 1, outcome % 6 + 1);
                if outcome >= 36 || hi < lo {
                    rules_bail!("backgammon: roll id {outcome} is not 6·(hi−1) + (lo−1) with hi >= lo");
                }
                self.start_move(hi as u8, lo as u8);
                Ok(())
            }
            p => rules_bail!("backgammon: no chance event in phase {p:?}"),
        }
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.actor() != Some(seat) {
            return;
        }
        match self.phase {
            Phase::Cube => out.extend([NO_DOUBLE, DOUBLE]),
            Phase::Take => out.extend([TAKE, DROP]),
            Phase::Move => out.extend(self.cands.iter().map(|c| c.id)),
            _ => {}
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.actor() != Some(seat) {
            rules_bail!("backgammon: seat {seat} is not to act ({:?})", self.phase);
        }
        match (self.phase, action) {
            (Phase::Cube, NO_DOUBLE) => self.phase = Phase::Roll,
            (Phase::Cube, DOUBLE) => self.phase = Phase::Take,
            (Phase::Take, TAKE) => {
                self.cube *= 2;
                self.owner = seat as i8;
                self.phase = Phase::Roll;
            }
            (Phase::Take, DROP) => self.end_game(self.on_roll, 1, true),
            (Phase::Move, a) => {
                let Ok(i) = self.cands.binary_search_by_key(&a, |c| c.id) else {
                    rules_bail!("backgammon: {a} is not a legal checker play of roll {}-{}", self.roll.0, self.roll.1);
                };
                let c = self.cands[i];
                let me = self.on_roll;
                self.board[me] = c.result.me;
                self.board[1 - me] = c.result.opp;
                match c.kind {
                    Kind::Pass | Kind::Full => self.finish_turn(),
                    Kind::Sub => {
                        let j = self.dice.index_of(c.die).expect("split move die");
                        self.dice = self.dice.without(j);
                        self.played += 1;
                        let need = self.total - self.played;
                        let pos = self.pos(me);
                        if need == 0 {
                            self.finish_turn();
                        } else {
                            let (comp, n) = legal_moves(&pos, self.dice);
                            debug_assert_eq!(n, need, "completions of a split turn");
                            if comp.len() <= MAX_ACTIONS {
                                self.cands = comp;
                                self.split = false;
                            } else {
                                self.cands = split_moves(&pos, self.dice, need);
                                self.split = true;
                            }
                        }
                    }
                }
            }
            (p, a) => rules_bail!("backgammon: action {a} is illegal in phase {p:?}"),
        }
        Ok(())
    }

    fn outcome(&self) -> Vec<f32> {
        let Some(w) = self.match_winner else { return vec![0.0, 0.0] };
        if self.money() {
            let v = (self.results.last().map_or(0, |r| r.points) as f64 / self.cfg.money_scale).min(1.0) as f32;
            let mut o = vec![-v, -v];
            o[w] = v;
            o
        } else {
            let mut o = vec![-1.0, -1.0];
            o[w] = 1.0;
            o
        }
    }

    fn returns(&self) -> Vec<f64> {
        let Some(w) = self.match_winner else { return vec![0.0, 0.0] };
        let v = if self.money() { self.results.last().map_or(0, |r| r.points) as f64 } else { 1.0 };
        let mut r = vec![-v, -v];
        r[w] = v;
        r
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let p = self.pos(seat);
        for i in 0..24 {
            let q = i + 1;
            let f = obs.entity(i, 0);
            thermo(&mut f[0..5], p.me[i]);
            thermo(&mut f[5..10], p.opp[24 - q]);
            f[10] = q as f32 / 25.0;
            f[11] = (q <= 6) as u8 as f32;
            f[12] = (q >= 19) as u8 as f32;
        }
        let f = obs.entity(24, 1);
        thermo(&mut f[0..5], p.me[24]);
        f[10] = 1.0;
        let f = obs.entity(25, 1);
        thermo(&mut f[5..10], p.opp[24]);
        f[10] = 1.0;
        thermo(&mut obs.entity(26, 2)[0..5], off(&p.me));
        thermo(&mut obs.entity(27, 2)[5..10], off(&p.opp));

        let opp = 1 - seat;
        let n = self.cfg.match_length;
        let g = obs.globals();
        match self.phase {
            Phase::Cube => g[0] = 1.0,
            Phase::Take => g[1] = 1.0,
            Phase::Move if self.split => g[3] = 1.0,
            Phase::Move => g[2] = 1.0,
            _ => {}
        }
        if self.phase == Phase::Move {
            let (hi, lo) = self.roll;
            g[4 + hi as usize - 1] = 1.0;
            g[10 + lo as usize - 1] = 1.0;
            g[16] = (hi == lo) as u8 as f32;
            if hi != lo {
                g[17] = self.dice.index_of(hi).is_some() as u8 as f32;
                g[18] = self.dice.index_of(lo).is_some() as u8 as f32;
            }
            g[19] = self.dice.len() as f32 / 4.0;
            g[20] = self.played as f32 / 4.0;
        }
        g[21] = (self.cube as f32).log2() / 12.0;
        match self.owner {
            -1 => g[23] = 1.0,
            o if o as usize == seat => g[22] = 1.0,
            _ => g[24] = 1.0,
        }
        g[25] = self.may_double(seat) as u8 as f32;
        g[26] = self.may_double(opp) as u8 as f32;
        g[27] = (self.on_roll == seat) as u8 as f32;
        if self.money() {
            g[28] = 1.0;
            g[29] = (self.cfg.jacoby && self.cube == 1) as u8 as f32;
        } else {
            let nf = n as f32;
            g[30] = nf / 25.0;
            g[31] = self.score[seat] as f32 / nf;
            g[32] = self.score[opp] as f32 / nf;
            g[33] = (n - self.score[seat]) as f32 / nf;
            g[34] = (n - self.score[opp]) as f32 / nf;
            g[35] = self.crawford_game as u8 as f32;
            g[36] = self.post_crawford as u8 as f32;
            g[37] = (n - self.score[seat] == 1) as u8 as f32;
            g[38] = (n - self.score[opp] == 1) as u8 as f32;
            g[39] = self.cube_dead_for(seat) as u8 as f32;
            g[40] = self.cube_dead_for(opp) as u8 as f32;
        }
        g[41] = (self.results.len() as f32 / 16.0).min(1.0);
        g[42] = pips(&p.me) as f32 / 167.0;
        g[43] = pips(&p.opp) as f32 / 167.0;
        g[44] = contact(&p) as u8 as f32;
    }

    fn encode_action(&self, _seat: Seat, action: u32, feat: &mut [f32]) {
        match action {
            NO_DOUBLE => feat[3] = 1.0,
            DOUBLE => feat[4] = 1.0,
            TAKE => feat[5] = 1.0,
            DROP => feat[6] = 1.0,
            _ => {
                let Ok(i) = self.cands.binary_search_by_key(&action, |c| c.id) else { return };
                let c = &self.cands[i];
                feat[match c.kind {
                    Kind::Pass => 0,
                    Kind::Full => 1,
                    Kind::Sub => 2,
                }] = 1.0;
                let r = &c.result;
                for i in 0..24 {
                    for k in 0..3 {
                        feat[7 + 3 * i + k] = (r.me[i] as usize > k) as u8 as f32;
                    }
                    feat[79 + i] = (c.hit_mask >> i & 1) as f32;
                }
                for k in 0..3 {
                    feat[103 + k] = (r.me[24] as usize > k) as u8 as f32;
                }
                for k in 0..4 {
                    feat[106 + k] = (c.borne as usize > k) as u8 as f32;
                    feat[110 + k] = (c.hits as usize > k) as u8 as f32;
                }
                if c.kind == Kind::Sub {
                    let left = self.dice.len() - 1;
                    for k in 0..3 {
                        feat[114 + k] = (left > k) as u8 as f32;
                    }
                }
                let o = off(&r.me);
                feat[117] = (o == CHECKERS) as u8 as f32;
                feat[118] = (o > 0) as u8 as f32;
            }
        }
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        match action {
            NO_DOUBLE => "no double".into(),
            DOUBLE => "double".into(),
            TAKE => "take".into(),
            DROP => "drop".into(),
            PASS => "pass".into(),
            a if (SUBMOVE_BASE..SUBMOVE_BASE + 256).contains(&a) => {
                let c = a - SUBMOVE_BASE;
                let (from, die) = ((c / 8) as usize, c % 8);
                format!("({})", Self::step_string(from, from as i32 - die as i32))
            }
            a if a >= FULL_MOVE_MIN => {
                let mut subs = unpack(a);
                subs.sort_by(|x, y| y.cmp(x));
                subs.iter().map(|&(f, d)| Self::step_string(f, f as i32 - d as i32)).collect::<Vec<_>>().join(" ")
            }
            a => format!("?{a}"),
        }
    }

    fn chance_string(&self, outcome: u32) -> String {
        if outcome >= OPENING_BASE {
            let o = outcome - OPENING_BASE;
            format!("opening {}-{}", o / 6 + 1, o % 6 + 1)
        } else {
            format!("roll {}-{}", outcome / 6 + 1, outcome % 6 + 1)
        }
    }

    fn state_string(&self) -> String {
        let side = |s: &Side| {
            let pts: Vec<String> = (0..24).filter(|&i| s[i] > 0).map(|i| format!("{}:{}", i + 1, s[i])).collect();
            format!("[{}] bar {} off {}", pts.join(" "), s[24], off(s))
        };
        format!(
            "phase={:?} on_roll={} roll={}-{} dice_left={:?} cube={} owner={} score={:?} crawford={} post={} games={}\n  seat0 {}\n  seat1 {}",
            self.phase, self.on_roll, self.roll.0, self.roll.1, self.dice.as_slice(), self.cube, self.owner, self.score,
            self.crawford_game, self.post_crawford, self.results.len(), side(&self.board[0]), side(&self.board[1])
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::check_game;
    use crate::trajectory::{replay, Event};

    fn moves(p: &Pos, a: u8, b: u8) -> Vec<Cand> {
        legal_moves(p, Dice::roll(a, b)).0
    }

    fn results(p: &Pos, a: u8, b: u8) -> Vec<Pos> {
        moves(p, a, b).iter().map(|c| c.result).collect()
    }

    fn start() -> Pos {
        Pos { me: start_side(), opp: start_side() }
    }

    #[test]
    fn opening_moves() {
        let p = start();
        // 6-5: 24/13, 24/18 13/8, 24/18 8/3, 13/2 (13/7/2 = 13/8/2), 13/7 13/8, 13/7 8/3, 8/2 8/3
        let m = moves(&p, 6, 5);
        assert_eq!(m.len(), 7);
        assert!(m.windows(2).all(|w| w[0].id < w[1].id));
        assert!(results(&p, 6, 5).contains(&apply_steps(&p, &[(24, 18), (18, 13)]).unwrap()));
        // 13/2 is one move, represented by its smallest id
        let r = apply_steps(&p, &[(13, 8), (8, 2)]).unwrap();
        let c = m.iter().find(|c| c.result == r).unwrap();
        assert_eq!(c.id, pack(&[code(13, 5), code(8, 6)]).min(pack(&[code(13, 6), code(7, 5)])));
        assert_eq!(Backgammon::result_of(&p, c.id), r);
        for c in &m {
            assert_eq!(Backgammon::result_of(&p, c.id), c.result);
            assert_eq!(c.kind, Kind::Full);
            assert_eq!(pips(&c.result.me), 167 - 11);
        }
        // doubles play four checkers; every result moves 4·d pips
        for d in 1..=6u8 {
            for c in moves(&p, d, d) {
                assert_eq!(pips(&c.result.me), 167 - 4 * d as u32, "{d}");
                assert_eq!(unpack(c.id).len(), 4);
            }
        }
        // 6-6 from the start: 24/18(2) 13/7(2), 24/18(2) 13/1? no: 1 is blocked...
        assert!(results(&p, 6, 6).contains(&apply_steps(&p, &[(24, 18), (24, 18), (13, 7), (13, 7)]).unwrap()));
    }

    #[test]
    fn bar_entry_blocking_and_hits() {
        // me: one checker on the bar, the rest on my 6-point; the opponent holds its
        // 1..5 points (my 24..20): only a 6 enters (my 19 = their 6)
        let p = mk(&[(25, 1), (6, 14)], &[(24, 2), (23, 2), (22, 2), (21, 2), (20, 2), (12, 5)]);
        let m = moves(&p, 5, 3);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].id, PASS);
        assert_eq!(m[0].kind, Kind::Pass);
        let m = moves(&p, 6, 3);
        // bar/19 then 3: 19/16 or 6/3
        let rs = results(&p, 6, 3);
        assert_eq!(rs.len(), 2);
        assert!(rs.contains(&apply_steps(&p, &[(25, 19), (19, 16)]).unwrap()));
        assert!(rs.contains(&apply_steps(&p, &[(25, 19), (6, 3)]).unwrap()));
        assert!(m.iter().all(|c| c.hits == 0));
        // two on the bar with 6-3: only one enters, nothing else may move
        let p2 = mk(&[(25, 2), (6, 13)], &[(24, 2), (23, 2), (22, 2), (21, 2), (20, 2), (12, 5)]);
        let rs = results(&p2, 6, 3);
        assert_eq!(rs, vec![apply_steps(&p2, &[(25, 19)]).unwrap()]);
        // hitting a blot sends it to the bar
        let p3 = Pos::from_points(&[(13, 15)], &[(15, 1), (6, 14)]); // their 15 = my 10
        let m = moves(&p3, 3, 1);
        let hit = m.iter().find(|c| c.result.me[9] == 1).unwrap();
        assert_eq!((hit.hits, hit.hit_mask, hit.result.opp[24], hit.result.opp[14]), (1, 1 << 9, 1, 0));
    }

    /// A position with the opponent's checkers given in MY numbering.
    fn mk(me: &[(usize, u8)], opp_my: &[(usize, u8)]) -> Pos {
        let opp: Vec<(usize, u8)> = opp_my.iter().map(|&(q, n)| (if q == BAR { BAR } else { 25 - q }, n)).collect();
        let p = Pos::from_points(me, &opp);
        p.validate().unwrap();
        p
    }

    #[test]
    fn larger_die_and_max_dice_rules() {
        // me: one checker on 13, fourteen on my 1-point (not all home: they cannot
        // move). Opponent holds 12 and 7: 13/12 and 13/7 blocked → pass
        let p = mk(&[(13, 1), (1, 14)], &[(12, 2), (7, 2), (20, 11)]);
        assert_eq!(moves(&p, 6, 1)[0].id, PASS);
        // holds 12 and 6: 13/7 then 7/6 blocked, 13/12 blocked → only the 6: 13/7
        let p = mk(&[(13, 1), (1, 14)], &[(12, 2), (6, 2), (20, 11)]);
        assert_eq!(results(&p, 6, 1), vec![apply_steps(&p, &[(13, 7)]).unwrap()]);
        // holds 6 only: 13/7 (then 7/6 blocked) or 13/12 (then 12/6 blocked) — either
        // die alone, never both: the larger one must be played
        let p = mk(&[(13, 1), (1, 14)], &[(6, 2), (20, 11)]);
        assert_eq!(results(&p, 6, 1), vec![apply_steps(&p, &[(13, 7)]).unwrap()]);
        // holds 7 and 6: only the 1 can be played (13/12, then 12/6 blocked)
        let p = mk(&[(13, 1), (1, 14)], &[(7, 2), (6, 2), (20, 11)]);
        assert_eq!(results(&p, 6, 1), vec![apply_steps(&p, &[(13, 12)]).unwrap()]);
        // both dice are played whenever possible
        let p = mk(&[(13, 2), (1, 13)], &[(20, 13)]);
        for c in moves(&p, 6, 1) {
            assert_eq!(pips(&p.me) - pips(&c.result.me), 7);
        }
        // doubles: as many as possible — 24/19/14, then 14/9 is blocked (two of four)
        let p = mk(&[(24, 1), (1, 14)], &[(9, 2), (10, 13)]);
        let (m, total) = legal_moves(&p, Dice::roll(5, 5));
        assert_eq!((total, m.len()), (2, 1));
        assert_eq!(m[0].result, apply_steps(&p, &[(24, 19), (19, 14)]).unwrap());
        // once home mid-roll, the rest may bear off: 13/8/3 then 3/off and 1/off
        let p = mk(&[(13, 1), (1, 14)], &[(20, 15)]);
        let (m, total) = legal_moves(&p, Dice::roll(5, 5));
        assert_eq!((total, m.len()), (4, 1));
        assert_eq!(off(&m[0].result.me), 2);
    }

    #[test]
    fn bearing_off() {
        // the opponent's checkers far away (its 1-point = my 24)
        let home = |me: &[(usize, u8)]| mk(me, &[(24, 15)]);
        // checkers on 5 and 3: a 6 bears off from the highest (5) only
        let p = home(&[(5, 1), (3, 1)]);
        let rs = results(&p, 6, 6);
        // 6-6: 5/off, 3/off → both off (then the game is over mid-roll: fine)
        assert!(rs.iter().all(|r| off(&r.me) == 15));
        let rs = results(&p, 6, 1);
        // 6 bears off the 5; 1: 3/2 (or 5/4 first then 6 bears off 4? 5/4 then 4/off with 6:
        // also legal) → results: {3/2 + 5/off}, {5/4 then 4/off = 5/off... same}, {3/2, 5/off}
        for r in &rs {
            assert_eq!(off(&r.me), 14);
        }
        // not all home: no bearing off
        let p = home(&[(7, 1), (2, 14)]);
        let rs = results(&p, 6, 2);
        // 7/1 (6) then 2/off (2) is legal (all home after the first move)
        assert!(rs.contains(&apply_steps(&p, &[(7, 1), (2, 0)]).unwrap()));
        // 2 then 6: 7/5, then a 6 bears off the 5 (highest)
        assert!(rs.contains(&apply_steps(&p, &[(7, 5), (5, 0)]).unwrap()));
        // a larger die may not bear off from a lower point while a higher one is occupied
        let p = home(&[(6, 1), (2, 1)]);
        let rs = results(&p, 5, 3);
        // 5: 6/1 or 2/? (2 with a 5: 6 is occupied → no); 3: 6/3, 2/? no (6 occupied)
        assert!(!rs.iter().any(|r| r.me[1] == 0 && r.me[5] == 1));
    }

    fn play(g: &mut Backgammon, ev: &[Event]) {
        for e in ev {
            match *e {
                Event::Chance(o) => g.apply_chance(o).unwrap(),
                Event::Act(s, a) => g.apply_action(s, a).unwrap(),
            }
        }
    }

    fn roll_id(hi: u32, lo: u32) -> u32 {
        6 * (hi - 1) + (lo - 1)
    }

    #[test]
    fn scoring_cube_and_crawford() {
        // money, Jacoby: a gammon with the cube centred counts 1
        let mut cfg = BackgammonConfig { match_length: 0, ..Default::default() };
        let mut g = Backgammon::new_game(&cfg);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap(); // seat 0: 6, seat 1: 2 → seat 0 starts 6-2
        assert_eq!((g.on_roll, g.roll), (0, (6, 2)));
        // put seat 0 one move from winning, seat 1 with nothing off
        g.board[0] = [0; 25];
        g.board[0][0] = 1; // one checker on my 1-point: any move bears it off
        g.board[1] = start_side();
        g.start_move(6, 2);
        let a = g.cands[0].id;
        g.apply_action(0, a).unwrap();
        assert_eq!(g.phase, Phase::Over);
        assert_eq!(g.results[0].multiplier, 3, "a checker in the winner's home board: backgammon");
        assert_eq!(g.results[0].points, 1, "Jacoby: cube never turned");
        assert_eq!(g.returns(), vec![1.0, -1.0]);
        assert_eq!(g.outcome(), vec![0.25, -0.25]);
        // without Jacoby, after a take: cube 2 × backgammon 3 = 6 → clamp(6 / 4) = 1
        cfg.jacoby = false;
        let mut g = Backgammon::new_game(&cfg);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap();
        g.board[0] = [0; 25];
        g.board[0][0] = 2;
        g.board[1] = start_side();
        g.start_move(1, 1);
        // any legal move: bears both off? 1-1 plays four: 1/off twice → wins
        let a = g.cands[0].id;
        g.apply_action(0, a).unwrap();
        assert_eq!(g.results[0].points, 3);
        // drops: the doubler wins the current cube
        let mut g = Backgammon::new_game(&cfg);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap(); // seat 0 moves first
        let a = g.cands[0].id;
        g.apply_action(0, a).unwrap();
        assert_eq!((g.phase, g.on_roll), (Phase::Cube, 1));
        g.apply_action(1, DOUBLE).unwrap();
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)));
        g.apply_action(0, TAKE).unwrap();
        assert_eq!((g.cube, g.owner, g.phase), (2, 0, Phase::Roll));
        g.apply_chance(roll_id(3, 1)).unwrap();
        let a = g.cands[0].id;
        g.apply_action(1, a).unwrap();
        assert_eq!(g.phase, Phase::Cube, "seat 0 owns the cube");
        g.apply_action(0, DOUBLE).unwrap();
        g.apply_action(1, DROP).unwrap();
        assert_eq!((g.phase, g.results[0].points, g.results[0].winner, g.results[0].dropped), (Phase::Over, 2, 0, true));
        assert_eq!(g.returns(), vec![2.0, -2.0]);

        // match to 3: seat 0 wins a gammon (2 points) → 2-0, the next game is Crawford
        let cfg = BackgammonConfig { match_length: 3, ..Default::default() };
        let mut g = Backgammon::new_game(&cfg);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap();
        g.board[0] = [0; 25];
        g.board[0][0] = 1;
        g.board[1] = start_side();
        g.board[1][23] = 0; // nothing in seat 0's home board: a gammon, not a backgammon
        g.board[1][12] += 2;
        g.start_move(6, 2);
        let a = g.cands[0].id;
        g.apply_action(0, a).unwrap();
        assert_eq!((g.score, g.results[0].multiplier, g.phase), ([2, 0], 2, Phase::Opening));
        assert!(g.crawford_game && !g.post_crawford);
        g.apply_chance(OPENING_BASE + 6 + 4).unwrap(); // seat 0: 2, seat 1: 5 → seat 1 starts
        let a = g.cands[0].id;
        g.apply_action(1, a).unwrap();
        assert_eq!(g.phase, Phase::Roll, "no cube in the Crawford game");
        // seat 1 wins the Crawford game with a gammon: 2-2, post-Crawford
        g.board[1] = [0; 25];
        g.board[1][0] = 1;
        g.board[0] = start_side();
        g.board[0][23] = 0;
        g.board[0][5] += 2;
        g.on_roll = 1;
        g.start_move(6, 2);
        let a = g.cands[0].id;
        g.apply_action(1, a).unwrap();
        assert_eq!(g.score, [2, 2]);
        assert!(!g.crawford_game && g.post_crawford);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap();
        let a = g.cands[0].id;
        g.apply_action(0, a).unwrap();
        assert_eq!(g.phase, Phase::Cube, "post-Crawford: doubling is allowed again");
        assert!(g.cube_dead_for(0) && g.cube_dead_for(1));

        // matches starting at a given score
        let at = |score: [u32; 2], post: bool| BackgammonConfig { match_length: 3, score, post_crawford: post, ..Default::default() };
        let g = Backgammon::new_game(&at([2, 0], false));
        assert!(g.crawford_game && !g.post_crawford);
        let g = Backgammon::new_game(&at([2, 1], true));
        assert!(!g.crawford_game && g.post_crawford);
        assert!(Backgammon::check_config(&at([2, 2], false)).is_err());
        assert!(Backgammon::check_config(&at([1, 1], true)).is_err());
        assert!(Backgammon::check_config(&at([3, 0], false)).is_err());
        assert!(Backgammon::check_config(&BackgammonConfig { match_length: 0, score: [1, 0], ..Default::default() }).is_err());
        // a 1-point match has no Crawford game (gnubg: the rule needs a match of 2+)
        let g = Backgammon::new_game(&BackgammonConfig { match_length: 1, ..Default::default() });
        assert!(!g.crawford_game && g.may_double(0));
    }

    #[test]
    fn split_turns_reach_every_move() {
        // a spread position with 1-1: more distinct moves than MAX_ACTIONS
        let me: Vec<(usize, u8)> = vec![(24, 1), (22, 1), (20, 1), (18, 1), (16, 1), (14, 1), (12, 1), (10, 1), (8, 1), (7, 1),
                                         (6, 1), (5, 1), (4, 1), (3, 1), (2, 1)];
        let p = Pos::from_points(&me, &[]);
        let (full, total) = legal_moves(&p, Dice::roll(1, 1));
        assert_eq!(total, 4);
        assert!(full.len() > MAX_ACTIONS, "{}", full.len());
        let cfg = BackgammonConfig { match_length: 0, ..Default::default() };
        let mut g = Backgammon::new_game(&cfg);
        g.apply_chance(OPENING_BASE + 6 * 5 + 1).unwrap();
        g.board[0] = p.me;
        g.board[1] = p.opp;
        g.start_move(1, 1);
        assert!(g.is_split());
        // every legal full move has a plan; the plans' first steps are split moves and
        // replaying a plan lands exactly on the move
        let mut firsts = HashSet::new();
        for (k, m) in full.iter().enumerate() {
            if k % 7 != 0 {
                continue;
            }
            let plan = g.plan_move(&m.result).unwrap();
            assert!(plan.len() >= 2 && plan[0] < FULL_MOVE_MIN);
            firsts.insert(plan[0]);
            let mut h = g.clone();
            for &a in &plan {
                h.apply_action(0, a).unwrap();
            }
            assert_eq!(h.on_roll, 1);
            assert_eq!(h.pos(0), m.result);
        }
        assert!(firsts.len() > 1);
        assert!(g.plan_move(&p).is_err());
    }

    #[test]
    fn ids_and_strings() {
        let id = pack(&[code(13, 5), code(24, 6)]);
        assert!(id >= FULL_MOVE_MIN);
        assert_eq!(unpack(id), vec![(24, 6), (13, 5)]);
        let g = Backgammon::new_game(&BackgammonConfig::default());
        assert_eq!(g.action_string(0, id), "24/18 13/8");
        assert_eq!(g.action_string(0, pack(&[code(25, 3), code(3, 5)])), "bar/22 3/off");
        assert_eq!(g.action_string(0, SUBMOVE_BASE + code(8, 2)), "(8/6)");
        let mut out = Vec::new();
        g.chance_outcomes(&mut out);
        assert_eq!(out.len(), 30);
        assert!((out.iter().map(|&(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn replay_and_conformance() {
        for cfg in [
            BackgammonConfig { match_length: 0, ..Default::default() },
            BackgammonConfig { match_length: 0, jacoby: false, max_cube: 8, ..Default::default() },
            BackgammonConfig { match_length: 3, ..Default::default() },
            BackgammonConfig { match_length: 1, ..Default::default() },
            BackgammonConfig { match_length: 5, cube: false, ..Default::default() },
        ] {
            let rep = check_game::<Backgammon>(&cfg, 12, 7).unwrap();
            assert_eq!(rep.terminal, 12, "{cfg:?}");
            assert!(rep.max_legal <= MAX_ACTIONS);
        }
        // a full random match replays from its events
        let cfg = BackgammonConfig::default();
        let mut rng = crate::rng::Rng::new(3);
        let ev = crate::invariance::random_trajectory::<Backgammon>(&cfg, &mut rng, 1 << 20);
        let r = replay::<Backgammon>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        let s = r.state.score();
        assert!(s[0] >= 7 || s[1] >= 7, "{s:?}");
        let mut g = Backgammon::new_game(&cfg);
        play(&mut g, &ev);
        assert_eq!(g.match_winner(), r.state.match_winner());
    }
}

#[cfg(test)]
mod measure {
    use super::*;
    use crate::rng::Rng;

    /// Distribution of the number of distinct legal moves per roll in random play:
    /// `cargo test --release -p gmrules measure_move_counts -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure_move_counts() {
        let mut rng = Rng::new(1);
        let mut hist_d = vec![0usize; 4000];
        let mut hist_n = vec![0usize; 4000];
        let t = std::time::Instant::now();
        let mut calls = 0usize;
        for _ in 0..400 {
            let mut p = Pos { me: start_side(), opp: start_side() };
            for _ply in 0..400 {
                let (a, b) = (rng.below(6) as u8 + 1, rng.below(6) as u8 + 1);
                let (m, _) = legal_moves(&p, Dice::roll(a, b));
                calls += 1;
                if a == b {
                    hist_d[m.len()] += 1
                } else {
                    hist_n[m.len()] += 1
                }
                let c = m[rng.below(m.len() as u64) as usize];
                p = c.result.flip();
                if off(&p.opp) == 15 {
                    break;
                }
            }
        }
        let el = t.elapsed();
        let q = |h: &Vec<usize>, f: f64| {
            let tot: usize = h.iter().sum();
            let mut acc = 0;
            for (i, &c) in h.iter().enumerate() {
                acc += c;
                if acc as f64 >= f * tot as f64 {
                    return i;
                }
            }
            0
        };
        let mx = |h: &Vec<usize>| h.iter().rposition(|&c| c > 0).unwrap_or(0);
        let over = |h: &Vec<usize>, a: usize| h[a + 1..].iter().sum::<usize>() as f64 / h.iter().sum::<usize>() as f64;
        eprintln!("calls {calls} in {el:?} ({:.1} us/call)", el.as_secs_f64() * 1e6 / calls as f64);
        eprintln!("non-doubles: p50 {} p99 {} p999 {} max {}", q(&hist_n, 0.5), q(&hist_n, 0.99), q(&hist_n, 0.999), mx(&hist_n));
        eprintln!("doubles: p50 {} p90 {} p99 {} p999 {} max {}", q(&hist_d, 0.5), q(&hist_d, 0.9), q(&hist_d, 0.99), q(&hist_d, 0.999),
                  mx(&hist_d));
        for a in [64usize, 96, 128, 192, 256] {
            eprintln!("A={a}: doubles over {:.4}, non-doubles over {:.5}", over(&hist_d, a), over(&hist_n, a));
        }
    }
}
