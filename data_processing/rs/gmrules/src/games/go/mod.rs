//! Go — the simulator of the KGS imitation data: the rules and the frame encoders of
//! `gmdata go convert` (converter v2, the frozen pre-training schema of the "go" adapter
//! `model/games/registry.py::_build_go`: BoardEncoder + PlacementDecoder), moved here so
//! the converter, its KataGo labeller and the simulator run ONE implementation. A
//! simulator frame is bit for bit the converter's frame of the same position.
//!
//! Rules ([`board`]): the converter's — captures, suicide illegal (single and
//! multi-stone), simple ko (a single-stone capture by a single stone left with one
//! liberty bans the captured point for the next move; any move or pass clears it); a
//! pass is always legal. Seats: 0 = Black, 1 = White; Black moves first unless the
//! config says otherwise (`first_player`: fixed-handicap games start with White;
//! `free_handicap` = k: Black's first k moves in a row, free placement).
//!
//! End of the game and scoring (the simulator's; the IL records end by resignation /
//! time / their own count, the converter takes the value target from RE):
//!
//! - two consecutive passes (any colours) end the game;
//! - `max_moves` (default 2·N², AlphaGo Zero's 722 on 19×19) moves, passes included,
//!   end it;
//! - cycles: legality stays simple ko (the IL legal set), so long cycles (double / triple
//!   ko, sending-two-returning-one) are legal; with `superko` (default on) a move that
//!   recreates an earlier whole-board position of the game (positional superko, stones
//!   only; the initial position included; passes never repeat anything) ENDS the game
//!   instead of being illegal — an early move cap, scored alike. Off: cycles run into
//!   `max_moves`;
//! - every end is scored by Tromp-Taylor area ([`Board::area`]: stones + empty regions
//!   reaching only that colour; every stone on the board counts as alive) with `komi`
//!   for White: margin = area(B) − area(W) − komi; outcome ±1 (0: a tie, possible with
//!   an integral komi), `returns` = ±margin (points).
//!
//! Komi convention. obs.globals_seq encodes the komi from the mover's view (+komi/20 for
//! White, −komi/20 for Black, KataGo's convention, clipped to ±1), whether it is known
//! and whether the rules score by area. The committed KGS 4d+ sample
//! (tests/fixtures/go, 2019-04: 30 even games — 28 RU[Chinese] KM[7.5], 1 AGA 7.5, 1
//! Japanese 6.5; the handicap games KM[0.5]) is essentially Chinese area scoring with
//! 7.5, so the default config is komi 7.5, area scoring (globals: ±0.375, known 1,
//! area 1, handicap 0) — the frames of the corpus' even games under the rules the
//! simulator actually scores by. `area_scoring = false` only relabels the frames as
//! territory rules (replaying such records); the simulator always counts area.
//!
//! Frames (interleaved: one stream per game, every frame from the mover's view):
//!
//! ```text
//! obs     board_seq    [4, N, N] bits  own / opponent / empty stones, the simple-ko point
//!         globals_seq  [6] f32         mover is black, komi (mover's view) / 20 clipped,
//!                                      komi known, area scoring, min(HA, 9) / 9 (HA =
//!                                      `handicap`, max(0, ·)), the previous move was a pass
//! legal   place        [N·N] bool      legal points (empty, not the ko point, no suicide)
//!         pass         [] bool         always true
//! target  place_idx    [] i64          action key: row·N + col, 0 on a pass
//!         passed       [] bool         action key: the move is a pass
//! ```
//!
//! `board_seq` / `globals_seq` / `place_idx` / `passed` are the converter's obs / tgt
//! tensors (names, dtypes, values); the converter adds `tgt.value` (the record's RE)
//! and writes no legal mask. Action ids: point row·N + col (0..N²), pass = N² (the
//! labeller's `teacher_idx` convention); the keys (place_idx, passed) decode to the
//! pass whenever passed = 1 (the PlacementDecoder's pass-gated action). Spec meta = the
//! converter's shard meta: `{global_features: 6, board_size: N}` (train_il_game
//! META_ARCH_KEYS["go"] → board_global_features, board_h, board_w).

pub mod board;
#[cfg(test)]
mod tests;

pub use board::{opponent, Board, Point, BLACK, EMPTY, WHITE};

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

/// Width of obs.globals_seq (shard meta `global_features`).
pub const GLOBAL_FEATURES: usize = 6;
/// Planes of obs.board_seq.
pub const PLANES: usize = 4;
/// Board sizes the simulator accepts (the KataGo labeller's GTP range).
pub const SIZES: std::ops::RangeInclusive<usize> = 2..=25;

/// Legal / target tensor indices (see [`Go::spec`]; obs: board_seq, globals_seq).
const LEGAL_PLACE: usize = 0;
const LEGAL_PASS: usize = 1;
const TGT_PLACE: usize = 0;
const TGT_PASSED: usize = 1;

/// The per-game constants of obs.globals_seq: (komi known, komi / 20 from White's view
/// clipped to ±1, area scoring flag, min(max(HA, 0), 9) / 9).
pub type GlobalConsts = (bool, f32, f32, f32);

/// [`GlobalConsts`] from a komi (None / non-finite = unknown: encoded as 0, "known" 0),
/// the HA value and the area-scoring flag — the converter's and the labeller's too.
pub fn globals_consts(komi: Option<f64>, handicap: i64, area: bool) -> GlobalConsts {
    let komi = komi.filter(|k| k.is_finite());
    let white_komi = komi.map(|k| (k / 20.0).clamp(-1.0, 1.0) as f32).unwrap_or(0.0);
    let ha = handicap.clamp(0, 9) as f32 / 9.0;
    (komi.is_some(), white_komi, area as u8 as f32, ha)
}

/// One obs.globals_seq frame (`out`: [`GLOBAL_FEATURES`] values) for `color` to move.
/// (Black's komi feature is the negated White one: −0.0 when the komi is 0 or unknown.)
pub fn encode_globals(color: u8, consts: GlobalConsts, prev_pass: bool, out: &mut [f32]) {
    let (komi_known, white_komi, area, ha_feat) = consts;
    out[0] = (color == BLACK) as u8 as f32;
    out[1] = if color == BLACK { -white_komi } else { white_komi };
    out[2] = komi_known as u8 as f32;
    out[3] = area;
    out[4] = ha_feat;
    out[5] = prev_pass as u8 as f32;
}

/// One obs.board_seq frame (`out`: 4·N² values, all written) of `b` with `color` to
/// move: own / opponent / empty stones and the simple-ko point.
pub fn encode_planes(b: &Board, color: u8, out: &mut [f32]) {
    let nn = (b.n * b.n) as usize;
    for (i, &v) in b.g.iter().enumerate() {
        out[i] = (v == color) as u8 as f32;
        out[nn + i] = (v == opponent(color)) as u8 as f32;
        out[2 * nn + i] = (v == EMPTY) as u8 as f32;
    }
    out[3 * nn..4 * nn].fill(0.0);
    if let Some((kr, kc)) = b.ko {
        out[3 * nn + (kr * b.n + kc) as usize] = 1.0;
    }
}

/// A colour (config values `"black"` / `"white"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Colour {
    #[default]
    Black,
    White,
}

impl Colour {
    pub fn stone(self) -> u8 {
        match self {
            Colour::Black => BLACK,
            Colour::White => WHITE,
        }
    }
}

/// The game (JSON config; every field optional).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoConfig {
    /// N (19; 9 / 13 for small boards), within [`SIZES`].
    pub board_size: usize,
    /// White's komi (7.5; see the module docs). `null` = unknown: frames say so
    /// (globals 1 / 2 = 0) and the count uses 0.
    pub komi: Option<f64>,
    /// obs.globals_seq[3], the ruleset's scoring the network is told (true = area:
    /// Chinese / AGA / NZ records). The simulator itself always counts area.
    pub area_scoring: bool,
    /// The SGF HA value: obs.globals_seq[4] only (the stones are `setup_black` or a free
    /// placement).
    pub handicap: i64,
    /// Setup stones (SGF AB / AW) as point indices row·N + col, placed as given.
    pub setup_black: Vec<usize>,
    pub setup_white: Vec<usize>,
    /// Who moves first (fixed-handicap games: `"white"`).
    pub first_player: Colour,
    /// Free handicap placement: Black makes the first k moves in a row (then White).
    pub free_handicap: usize,
    /// Moves (passes included) after which the game ends and is counted; null = 2·N².
    pub max_moves: Option<usize>,
    /// A move recreating an earlier whole-board position ends the game (see the module
    /// docs); legality stays simple ko either way.
    pub superko: bool,
}

impl Default for GoConfig {
    fn default() -> Self {
        GoConfig {
            board_size: 19,
            komi: Some(7.5),
            area_scoring: true,
            handicap: 0,
            setup_black: Vec::new(),
            setup_white: Vec::new(),
            first_player: Colour::Black,
            free_handicap: 0,
            max_moves: None,
            superko: true,
        }
    }
}

impl GoConfig {
    /// The move cap: `max_moves`, default 2·N².
    pub fn move_cap(&self) -> usize {
        self.max_moves.unwrap_or(2 * self.board_size * self.board_size)
    }
}

/// Why a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// Two consecutive passes.
    DoublePass,
    /// A move recreated an earlier position (`superko`).
    Superko,
    /// `max_moves` moves were played.
    MoveCap,
}

/// A Go game state (see the module docs).
#[derive(Clone, Debug)]
pub struct Go {
    board: Board,
    /// colour to move ([`BLACK`] / [`WHITE`])
    to_move: u8,
    free_handicap: usize,
    consts: GlobalConsts,
    /// komi of the count (0 when unknown)
    komi: f64,
    /// the last move was a pass (obs.globals_seq[5])
    prev_pass: bool,
    /// consecutive passes
    passes: u32,
    /// moves played, passes included
    moves: usize,
    max_moves: usize,
    superko: bool,
    /// position hashes of the game so far (superko)
    history: HashSet<u64>,
    end: Option<End>,
}

fn seat_of(color: u8) -> Seat {
    if color == BLACK {
        0
    } else {
        1
    }
}

impl Go {
    /// The initial state of `cfg` (setup stones placed like the converter's
    /// `Board::setup`: black first, then white), or None when a setup stone is off the
    /// board or on an occupied point. [`Game::new_game`] = this on a checked config.
    pub fn setup(cfg: &GoConfig) -> Option<Go> {
        let n = cfg.board_size as i32;
        let pt = |&i: &usize| if i < (n * n) as usize { ((i as i32) / n, (i as i32) % n) } else { (-1, -1) };
        let mut board = Board::new(n);
        let black: Vec<Point> = cfg.setup_black.iter().map(pt).collect();
        let white: Vec<Point> = cfg.setup_white.iter().map(pt).collect();
        if !board.setup(BLACK, &black) || !board.setup(WHITE, &white) {
            return None;
        }
        let mut history = HashSet::new();
        if cfg.superko {
            history.insert(board.position_hash());
        }
        Some(Go {
            to_move: if cfg.free_handicap > 0 { BLACK } else { cfg.first_player.stone() },
            free_handicap: cfg.free_handicap,
            consts: globals_consts(cfg.komi, cfg.handicap, cfg.area_scoring),
            komi: cfg.komi.filter(|k| k.is_finite()).unwrap_or(0.0),
            prev_pass: false,
            passes: 0,
            moves: 0,
            max_moves: cfg.move_cap(),
            superko: cfg.superko,
            history,
            end: None,
            board,
        })
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Colour to move ([`BLACK`] / [`WHITE`]).
    pub fn to_move(&self) -> u8 {
        self.to_move
    }

    /// Moves played (passes included).
    pub fn moves(&self) -> usize {
        self.moves
    }

    /// Why the game ended (None: it goes on).
    pub fn end(&self) -> Option<End> {
        self.end
    }

    /// The Tromp-Taylor margin area(B) − area(W) − komi of the current position.
    pub fn score_margin(&self) -> f64 {
        let (b, w) = self.board.area();
        b as f64 - w as f64 - self.komi
    }

    /// `color`'s frame of this position (the obs tensors of [`Go::spec`]): the one
    /// observation encoder of the simulator ([`Game::encode_custom`], `color` = the side
    /// to move) and of the converter (`color` = the record's mover).
    pub fn encode_obs_as(&self, color: u8, board_seq: &mut [f32], globals_seq: &mut [f32]) {
        encode_planes(&self.board, color, board_seq);
        encode_globals(color, self.consts, self.prev_pass, globals_seq);
    }

    /// Play `color`'s move (None = pass) with the converter's rules; false = illegal
    /// (nothing changes). The one transition of the simulator ([`Game::apply_action`],
    /// `color` = the side to move) and of the converter (the record's colours, which
    /// may break alternation; it also plays on past the end of the game, which only
    /// records the first reason in [`Go::end`]).
    pub fn play_as(&mut self, color: u8, p: Option<Point>) -> bool {
        if !self.board.play(color, p) {
            return false;
        }
        self.moves += 1;
        self.prev_pass = p.is_none();
        let mut repeated = false;
        if p.is_none() {
            self.passes += 1;
        } else {
            self.passes = 0;
            if self.superko {
                repeated = !self.history.insert(self.board.position_hash());
            }
        }
        self.to_move = if self.moves < self.free_handicap { BLACK } else { opponent(color) };
        if self.end.is_none() {
            self.end = if self.passes >= 2 {
                Some(End::DoublePass)
            } else if repeated {
                Some(End::Superko)
            } else if self.moves >= self.max_moves {
                Some(End::MoveCap)
            } else {
                None
            };
        }
        true
    }

    fn n(&self) -> usize {
        self.board.n as usize
    }

    fn point_name(&self, p: usize) -> String {
        let n = self.n();
        const COLS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";
        format!("{}{}", COLS[p % n] as char, n - p / n)
    }
}

impl Game for Go {
    type Config = GoConfig;
    const NAME: &'static str = "go";
    /// = the version of `gmdata go convert` whose frames this reproduces (2: planes +
    /// obs.globals_seq); bump both together when the frame layout changes.
    const VERSION: u32 = 2;

    fn check_config(cfg: &GoConfig) -> Result<(), RulesError> {
        if !SIZES.contains(&cfg.board_size) {
            rules_bail!("board_size {} outside {}..={}", cfg.board_size, SIZES.start(), SIZES.end());
        }
        if cfg.komi.is_some_and(|k| !k.is_finite()) {
            rules_bail!("komi must be finite (null = unknown)");
        }
        let nn = cfg.board_size * cfg.board_size;
        let mut seen = HashSet::new();
        for &p in cfg.setup_black.iter().chain(&cfg.setup_white) {
            if p >= nn {
                rules_bail!("setup stone {p} off the {n}x{n} board", n = cfg.board_size);
            }
            if !seen.insert(p) {
                rules_bail!("setup point {p} given twice");
            }
        }
        if cfg.free_handicap > 0 && cfg.first_player != Colour::Black {
            rules_bail!("free handicap placement is black's: first_player must be black");
        }
        if cfg.max_moves == Some(0) {
            rules_bail!("max_moves must be >= 1 (null = 2·N²)");
        }
        Ok(())
    }

    fn spec(cfg: &GoConfig) -> FrameSpec {
        let n = cfg.board_size;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: cfg.move_cap(),
            schema: "go.v2".into(),
            obs: vec![TensorSpec::bits("board_seq", &[PLANES, n, n]), TensorSpec::f32("globals_seq", &[GLOBAL_FEATURES])],
            legal: vec![TensorSpec::bool("place", &[n * n]), TensorSpec::bool("pass", &[])],
            target: vec![TensorSpec::i64("place_idx", &[]), TensorSpec::bool("passed", &[])],
            action_keys: vec!["place_idx".into(), "passed".into()],
            meta: json!({"global_features": GLOBAL_FEATURES, "board_size": n}),
            candidate: None,
        }
    }

    fn new_game(cfg: &GoConfig) -> Self {
        Go::setup(cfg).expect("go: check_config guarantees a valid setup")
    }

    fn turn(&self) -> Turn {
        if self.end.is_some() {
            Turn::Terminal
        } else {
            Turn::Act(SeatSet::single(seat_of(self.to_move)))
        }
    }

    fn chance_outcomes(&self, _out: &mut Vec<(u32, f64)>) {}

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        rules_bail!("go has no chance events (got {outcome})")
    }

    /// The legal points (ascending), then the pass (N², always legal).
    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            return;
        }
        let nn = self.n() * self.n();
        let mut mask = vec![0u8; nn];
        self.board.legal_points(self.to_move, &mut mask);
        out.extend(mask.iter().enumerate().filter(|&(_, &m)| m == 1).map(|(p, _)| p as u32));
        out.push(nn as u32);
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("go: seat {seat} is not to act");
        }
        let (n, a) = (self.n(), action as usize);
        let p = match a {
            _ if a == n * n => None,
            _ if a < n * n => Some(((a / n) as i32, (a % n) as i32)),
            _ => rules_bail!("go: action {action} outside 0..={}", n * n),
        };
        if !self.play_as(self.to_move, p) {
            rules_bail!("go: {} is illegal for seat {seat}", self.action_string(seat, action));
        }
        Ok(())
    }

    /// ±1 by the Tromp-Taylor margin at the end (+0.0 for both on a tie, and while the
    /// game goes on).
    fn outcome(&self) -> Vec<f32> {
        let m = self.returns()[0];
        if m > 0.0 {
            vec![1.0, -1.0]
        } else if m < 0.0 {
            vec![-1.0, 1.0]
        } else {
            vec![0.0, 0.0]
        }
    }

    /// The margin in points, ±(area(B) − area(W) − komi), at the end (0 before).
    fn returns(&self) -> Vec<f64> {
        if self.end.is_none() {
            return vec![0.0, 0.0];
        }
        let m = self.score_margin();
        vec![m, 0.0 - m]
    }

    fn encode_custom(&self, seat: Seat, f: &mut FrameMut<'_>) {
        debug_assert_eq!(seat, seat_of(self.to_move));
        {
            let (b, g) = match f.obs.bufs_mut() {
                [crate::schema::BufMut::F32(b), crate::schema::BufMut::F32(g)] => (b, g),
                _ => unreachable!("go obs layout"),
            };
            self.encode_obs_as(self.to_move, b, g);
        }
        let nn = self.n() * self.n();
        for &a in f.legal_ids.iter() {
            if (a as usize) < nn {
                f.legal.bool(LEGAL_PLACE)[a as usize] = 1;
            } else {
                f.legal.bool(LEGAL_PASS)[0] = 1;
            }
        }
    }

    fn target_custom(&self, _seat: Seat, _legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let nn = (self.n() * self.n()) as u32;
        let pass = action == nn;
        t.i64(TGT_PLACE)[0] = if pass { 0 } else { action as i64 };
        t.bool(TGT_PASSED)[0] = pass as u8;
    }

    /// passed = 1 → the pass (whatever place_idx says); passed = 0 → the point place_idx.
    fn decode_custom(&self, _seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let nn = (self.n() * self.n()) as i64;
        let a: i64 = match keys[1] {
            1 => nn,
            0 if (0..nn).contains(&keys[0]) => keys[0],
            _ => return None,
        };
        let a = a as u32;
        legal.binary_search(&a).is_ok().then_some(a)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        let nn = self.n() * self.n();
        match action as usize {
            a if a == nn => "pass".into(),
            a if a < nn => self.point_name(a),
            a => format!("a{a}"),
        }
    }

    fn state_string(&self) -> String {
        let n = self.n();
        let mut s = String::from("   ");
        for c in 0..n {
            s.push_str(&self.point_name(c)[..1]);
            s.push(' ');
        }
        s.push('\n');
        for r in 0..n {
            s.push_str(&format!("{:2} ", n - r));
            for c in 0..n {
                let ko = self.board.ko == Some((r as i32, c as i32));
                s.push(match self.board.g[r * n + c] {
                    BLACK => 'X',
                    WHITE => 'O',
                    _ if ko => '*',
                    _ => '.',
                });
                s.push(' ');
            }
            s.push('\n');
        }
        let colour = |c: u8| if c == BLACK { "black" } else { "white" };
        format!("{s}to move: {} moves: {} passes: {} end: {:?} margin (B-W-komi): {}", colour(self.to_move), self.moves,
                self.passes, self.end, self.score_margin())
    }
}
