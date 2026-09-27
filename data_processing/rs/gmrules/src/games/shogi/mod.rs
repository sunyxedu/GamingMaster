//! Shogi (floodgate / CSA rules) — the rules and frames of the IL converter
//! `gmdata shogi convert` (floodgate CSA records, converter_version 2: a FROZEN schema)
//! as a gmrules game, so the simulator (gmsim → `env/vec_gmsim.py`) emits exactly the
//! converter's frames. Network: the existing shogi adapter (`model/games/shogi.py`:
//! ShogiEncoder + the move-or-drop ShogiDecoder); Python registration
//! `model/games/gmrules/shogi.py`; RL: train_rl_joint's `ShogiSpec`.
//!
//! # Rules
//!
//! [`board`]: the converter's own engine, moved here (one implementation for converter,
//! labeller and simulator) — python-shogi's semantics, including its pawn-drop-mate
//! approximation (it never forbids a legal drop, but lets two rare kinds of pawn-drop
//! mate through, see [`Board::is_legal`]). Moves, drops, promotion (optional in / into /
//! out of the zone, forced for pieces that could not move again), nifu, dead drops,
//! self-check.
//!
//! Game end, checked after every move (and at the start) in this order ([`EndReason`]):
//!
//! 1. sennichite: the 4th occurrence of a position (pieces with owners, both hands, side
//!    to move — the converter's repetition key [`PosKey`], counted from the start of the
//!    game) ends it: a draw, unless every move of one side from the first occurrence of
//!    that position to the fourth gave check — that side loses (連続王手の千日手, JSA
//!    rule; a draw if both sides checked throughout);
//! 2. no legal move: the side to move loses (checkmate; also the practically impossible
//!    stalemate, which shogi scores as a loss too);
//! 3. `max_plies` plies played (the CSA / floodgate `Max_Moves` draw; default 256 —
//!    floodgate's 2025 records say `Max_Moves:512`, set `max_plies` to 512 for that);
//! 4. entering-king declaration (`entering_king = "csa27"`, the default): when the side to
//!    move meets CSA rule 27 ([`Board::can_declare`]: king in the enemy camp and not in
//!    check, ≥ 10 other pieces there, ≥ 28 / 27 points with the pieces in hand) it
//!    declares and wins. The declaration is automatic — a win is a win, so a player never
//!    loses anything by declaring at once, and the action space stays the adapter's (no
//!    "declare" action). `"none"` turns it off.
//!
//! Resignation, time forfeits and the server's rulings are not moves: a record ending
//! that way stops before a terminal state; the converter takes the value target from the
//! CSA result line and replays records as recorded ([`Shogi::replaying`]: no automatic
//! game end — floodgate players may go on past a position where they could declare,
//! and records may run to 512 plies), so its frames do not depend on these rules.
//!
//! Validated against the floodgate server's own rulings (`tests/fixtures/shogi_endings/`,
//! gmdata shogi.rs tests): sennichite draws, perpetual-check losses, the 512-ply
//! `max_moves` draw, successful and illegal declarations end exactly where shogi-server
//! ended those games.
//!
//! Actions (absolute squares, the ids of a game-wide id space; [`Game::action_string`]
//! is USI, e.g. `7g7f`, `8h2b+`, `P*5e`):
//!
//! ```text
//! board move   (from · 81 + to) · 2 + promote     ∈ [0, 13122)
//! drop         13122 + (piece − 1) · 81 + to      ∈ [13122, 13689), piece 1..7 = P L N S G B R
//! ```
//!
//! # Frames (interleaved: one stream per game, every frame from the mover's view)
//!
//! Exactly the converter_version 2 shard tensors (`gmdata shogi convert`; the `tgt.value`
//! the converter adds is the recorded result, not part of the frame). POV square = the
//! square for black, 80 − square for white (the board rotated 180°); hand slots 0..6 =
//! the mover's P L N S G B R, 7..13 the opponent's.
//!
//! ```text
//! obs     board_seq     [28, 9, 9] bits  own P L N S G B R K +P +L +N +S +B +R, then opponent
//!         hand_seq      [14, 4] f32      per hand slot: [count, count > 0, min(count / 4, 1), 1]
//!         globals_seq   [7] f32          min(move number / 512, 1), mover is black, in check, 1,
//!                                        rep ≥ 1, rep ≥ 2, rep ≥ 3 (rep = earlier occurrences of
//!                                        this position in the game: sennichite = the 4th)
//! legal   mode          [2] bool         [a board move is legal, a drop is legal]
//!         from          [81] bool        from-squares of legal board moves
//!         to_cond       [81, 81] bool    destinations per from-square
//!         promo_cond    [81, 81, 2] bool [non-promotion legal, promotion legal] per (from, to)
//!         drop_piece    [14] bool        hand slots with a legal drop (slots 7..13 never)
//!         drop_to_cond  [14, 81] bool    drop squares per hand slot
//! target  dropped       [] i64           action key: 1 = drop
//!         from_idx      [] i64           action key: POV from-square (0 on drops)
//!         to_idx        [] i64           action key: POV to-square (0 on drops)
//!         promoted      [] bool          action key: the move promotes (false on drops)
//!         piece_idx     [] i64           action key: hand slot of a drop (0 on board moves)
//!         drop_to_idx   [] i64           action key: POV drop square (0 on board moves)
//!         legal_mode / legal_from / legal_to_of / legal_promo_of / legal_drop_piece /
//!         legal_drop_to_of  bits         the IL masks: the legal tables' rows at the stored
//!                                        indices (legal_to_of = to_cond[from_idx], ...)
//! ```
//!
//! The simulator's step decodes the action keys by `dropped`: a drop reads piece_idx /
//! drop_to_idx, a board move from_idx / to_idx / promoted; the other branch's keys are
//! ignored (the ShogiDecoder samples both branches on every frame). Shard meta:
//! {"global_features": 7} (the model's `shogi_global_features`).

mod board;

pub use board::{
    col, check_position, demoted, hirate, parse_sfen, parse_usi_move, pov, promoted, rank, repetition_counts, sfen,
    usi_move, usi_square, Board, Mv, PosKey, Sqs, BISHOP, BLACK, GOLD, KING, KNIGHT, LANCE, NONE, PAWN, PROM_BISHOP,
    PROM_KNIGHT, PROM_LANCE, PROM_PAWN, PROM_ROOK, PROM_SILVER, ROOK, SILVER, SQUARES, WHITE,
};

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{BufMut, TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Schema name (the converter_version 2 frame layout).
pub const SCHEMA: &str = "shogi.v2";
/// `board_seq` planes: 14 piece types × 2 sides.
pub const PIECE_CHANNELS: usize = 28;
/// Hand slots: the mover's P..R, then the opponent's.
pub const HAND_SLOTS: usize = 14;
/// Features per hand slot.
pub const HAND_FEATURES: usize = 4;
/// `globals_seq` width (shard meta `global_features`).
pub const GLOBAL_FEATURES: usize = 7;
/// Floats per `board_seq` frame (28 planes × 81 squares).
pub const OBS_BOARD: usize = PIECE_CHANNELS * SQUARES;
/// Board-move action ids: (from · 81 + to) · 2 + promote.
pub const N_BOARD_ACTIONS: u32 = 81 * 81 * 2;
/// All action ids (board moves, then drops).
pub const N_ACTIONS: u32 = N_BOARD_ACTIONS + 7 * 81;
/// Default ply cap (the CSA `Max_Moves` draw).
pub const DEFAULT_MAX_PLIES: usize = 256;
/// Largest accepted `max_plies`.
pub const MAX_MAX_PLIES: usize = 100_000;

const LEGAL_MODE: usize = 0;
const LEGAL_FROM: usize = 1;
const LEGAL_TO_COND: usize = 2;
const LEGAL_PROMO_COND: usize = 3;
const LEGAL_DROP_PIECE: usize = 4;
const LEGAL_DROP_TO_COND: usize = 5;
const TGT_DROPPED: usize = 0;
const TGT_FROM: usize = 1;
const TGT_TO: usize = 2;
const TGT_PROMOTED: usize = 3;
const TGT_PIECE: usize = 4;
const TGT_DROP_TO: usize = 5;
const TGT_LEGAL_MODE: usize = 6;
const TGT_LEGAL_FROM: usize = 7;
const TGT_LEGAL_TO_OF: usize = 8;
const TGT_LEGAL_PROMO_OF: usize = 9;
const TGT_LEGAL_DROP_PIECE: usize = 10;
const TGT_LEGAL_DROP_TO_OF: usize = 11;

// ---------------------------------------------------------------- frame encoders

/// One frame's observation — the converter's encoding: `board` = the zeroed
/// `obs.board_seq` frame [28·81], `hand` = `obs.hand_seq` [14·4], `glb` = `obs.globals_seq`
/// [7] (both fully written). `rep` = earlier occurrences of the position
/// ([`repetition_counts`]).
pub fn encode_obs(b: &Board, rep: u32, board: &mut [f32], hand: &mut [f32], glb: &mut [f32]) {
    let mover = b.turn;
    let mb = mover == BLACK;
    for sq in 0..81 {
        let pt = b.piece[sq];
        if pt == NONE {
            continue;
        }
        let plane = (pt - 1) as usize + if b.color[sq] as usize == mover { 0 } else { 14 };
        board[plane * 81 + pov(sq, mb) as usize] = 1.0;
    }
    let mut i = 0;
    for c in [mover, mover ^ 1] {
        for pt in 1..8usize {
            let n = b.hand[c][pt] as f32;
            hand[i..i + 4].copy_from_slice(&[n, (n > 0.0) as u8 as f32, (n as f64 / 4.0).min(1.0) as f32, 1.0]);
            i += 4;
        }
    }
    glb.copy_from_slice(&[
        (b.move_number as f64 / 512.0).min(1.0) as f32,
        mb as u8 as f32,
        b.is_check() as u8 as f32,
        1.0,
        (rep >= 1) as u8 as f32,
        (rep >= 2) as u8 as f32,
        (rep >= 3) as u8 as f32,
    ]);
}

/// Per-frame legality masks in POV indices (the IL targets `tgt.legal_*`); the `_of` rows
/// are the ones selected by the frame's stored target indices.
#[derive(Debug)]
pub struct FrameMasks {
    pub mode: [bool; 2],
    pub from: [bool; 81],
    pub to_of: [bool; 81],
    pub promo_of: [bool; 2],
    pub drop_piece: [bool; 14],
    pub drop_to_of: [bool; 81],
}

/// The IL masks of a frame with legal `moves` (any order) and stored indices `from_idx` /
/// `to_idx` / `piece_idx` (POV).
pub fn frame_masks(moves: &[Mv], mover_black: bool, from_idx: i64, to_idx: i64, piece_idx: i64) -> FrameMasks {
    let mut f = FrameMasks {
        mode: [false; 2],
        from: [false; 81],
        to_of: [false; 81],
        promo_of: [false; 2],
        drop_piece: [false; 14],
        drop_to_of: [false; 81],
    };
    for m in moves {
        let to = pov(m.to, mover_black);
        match m.from {
            Some(sq) => {
                let from = pov(sq, mover_black);
                f.mode[0] = true;
                f.from[from as usize] = true;
                if from == from_idx {
                    f.to_of[to as usize] = true;
                    if to == to_idx {
                        f.promo_of[m.promotion as usize] = true;
                    }
                }
            }
            None => {
                let slot = m.drop as i64 - 1;
                f.mode[1] = true;
                f.drop_piece[slot as usize] = true;
                if slot == piece_idx {
                    f.drop_to_of[to as usize] = true;
                }
            }
        }
    }
    f
}

/// (dropped, from, to, promoted, piece, drop_to) targets of `m` in POV indices (from / to /
/// promoted are 0 / false on drops, piece / drop_to 0 on board moves).
pub fn move_indices(m: &Mv, mover_black: bool) -> (i64, i64, i64, bool, i64, i64) {
    if m.drop != NONE {
        (1, 0, 0, false, m.drop as i64 - 1, pov(m.to, mover_black))
    } else {
        (0, pov(m.from.unwrap(), mover_black), pov(m.to, mover_black), m.promotion, 0, 0)
    }
}

/// The position of an encoded frame (the inverse of [`encode_obs`]): pieces from the 28
/// planes, hands from the counts, the side to move and the move number from the globals.
/// Rejects frames no position encodes to (two pieces on a square, fractional counts, …).
pub fn board_from_obs(board: &[f32], hand: &[f32], glb: &[f32]) -> Result<Board, String> {
    if board.len() != OBS_BOARD || hand.len() != 56 || glb.len() < 2 {
        return Err(format!("frame sizes {} / {} / {} are not 28·81 / 14·4 / ≥2", board.len(), hand.len(), glb.len()));
    }
    let mover = if glb[1] == 1.0 {
        BLACK
    } else if glb[1] == 0.0 {
        WHITE
    } else {
        return Err(format!("globals[1] (mover is black) = {}", glb[1]));
    };
    let mb = mover == BLACK;
    let mut b = Board { turn: mover, ..Board::empty() };
    for plane in 0..28 {
        for p in 0..81 {
            let x = board[plane * 81 + p];
            if x == 0.0 {
                continue;
            }
            if x != 1.0 {
                return Err(format!("board plane {plane} square {p} = {x}"));
            }
            let sq = if mb { p } else { 80 - p };
            if b.piece[sq] != NONE {
                return Err(format!("two pieces on square {sq}"));
            }
            let (pt, owner) = ((plane % 14 + 1) as u8, if plane < 14 { mover } else { mover ^ 1 });
            b.piece[sq] = pt;
            b.color[sq] = owner as u8;
            if pt == KING {
                if b.king[owner].is_some() {
                    return Err(format!("two kings of colour {owner}"));
                }
                b.king[owner] = Some(sq);
            }
        }
    }
    for (row, (c, pt)) in [mover, mover ^ 1].iter().flat_map(|&c| (1..8usize).map(move |pt| (c, pt))).enumerate() {
        let n = hand[row * 4];
        if !(n >= 0.0 && n.fract() == 0.0 && n <= 18.0) {
            return Err(format!("hand row {row} count {n}"));
        }
        b.hand[c][pt] = n as u32;
    }
    let g = glb[0] as f64 * 512.0;
    if !(1.0..512.0).contains(&g) {
        return Err(format!("globals[0] = {} is not a move number in [1, 512)", glb[0]));
    }
    b.move_number = g.round() as u32;
    Ok(b)
}

// ---------------------------------------------------------------- actions

/// The action id of a move (see the module docs).
pub fn action_of(m: &Mv) -> u32 {
    match m.from {
        Some(f) => ((f * 81 + m.to) * 2 + m.promotion as usize) as u32,
        None => N_BOARD_ACTIONS + (m.drop as u32 - 1) * 81 + m.to as u32,
    }
}

/// The move of an action id (< [`N_ACTIONS`]).
pub fn move_of(action: u32) -> Mv {
    if action < N_BOARD_ACTIONS {
        let ft = (action / 2) as usize;
        Mv { from: Some(ft / 81), to: ft % 81, promotion: action % 2 == 1, drop: NONE }
    } else {
        let r = action - N_BOARD_ACTIONS;
        Mv { from: None, to: (r % 81) as usize, promotion: false, drop: (r / 81 + 1) as u8 }
    }
}

/// Inverse of [`pov`].
fn unpov(p: i64, mover_black: bool) -> usize {
    if mover_black {
        p as usize
    } else {
        (80 - p) as usize
    }
}

// ---------------------------------------------------------------- the game

/// Entering-king declaration rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnteringKing {
    /// CSA rule 27 (floodgate), declared automatically ([`Board::can_declare`]).
    #[default]
    Csa27,
    /// No declaration.
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShogiConfig {
    /// Start position, an SFEN ("" = hirate); its move number starts the frames' move count.
    pub sfen: String,
    /// Draw after this many plies (CSA `Max_Moves`; 1..=100000).
    pub max_plies: usize,
    /// `"csa27"` (automatic CSA rule-27 declaration) or `"none"`.
    pub entering_king: EnteringKing,
}

impl Default for ShogiConfig {
    fn default() -> Self {
        ShogiConfig { sfen: String::new(), max_plies: DEFAULT_MAX_PLIES, entering_king: EnteringKing::Csa27 }
    }
}

impl ShogiConfig {
    /// The start position of the config.
    pub fn start(&self) -> Result<Board, String> {
        if self.sfen.trim().is_empty() {
            return Ok(Board::hirate());
        }
        let fields: Vec<&str> = self.sfen.split_whitespace().collect();
        let fields = if fields.first() == Some(&"sfen") { &fields[1..] } else { &fields[..] };
        let b = parse_sfen(fields)?;
        check_position(&b)?;
        Ok(b)
    }
}

/// Why a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    /// the side to move is checkmated: it loses
    Checkmate,
    /// the side to move has no legal move but is not in check: it loses
    NoLegalMove,
    /// the 4th occurrence of a position: a draw
    Sennichite,
    /// the 4th occurrence, one side checked throughout: that side loses
    PerpetualCheck,
    /// `max_plies` plies: a draw
    MaxPlies,
    /// the side to move declares a win (entering king)
    Declaration,
}

impl EndReason {
    pub fn name(self) -> &'static str {
        match self {
            EndReason::Checkmate => "checkmate",
            EndReason::NoLegalMove => "no_legal_move",
            EndReason::Sennichite => "sennichite",
            EndReason::PerpetualCheck => "perpetual_check",
            EndReason::MaxPlies => "max_plies",
            EndReason::Declaration => "declaration",
        }
    }
}

/// The automatic game-end rules of a [`Shogi`] state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rules {
    /// off: replay a record as recorded (no automatic end at all)
    adjudicate: bool,
    max_plies: usize,
    declaration: bool,
}

/// A shogi game (see the module docs): the board, the repetition history, the legal
/// moves of the side to move and the end of the game.
#[derive(Clone, Debug)]
pub struct Shogi {
    board: Board,
    rules: Rules,
    /// every position of the game so far: (occurrences, ply of the first occurrence)
    occ: HashMap<PosKey, (u32, u32)>,
    /// `checks[k]`: the side to move after k plies is in check (move k gave check)
    checks: Vec<bool>,
    /// earlier occurrences of the current position
    rep: u32,
    /// legal moves of the side to move (the rules' generation order)
    legal: Vec<Mv>,
    /// their action ids, ascending
    legal_ids: Vec<u32>,
    /// terminal: (result for black in {-1, 0, 1}, why)
    end: Option<(i8, EndReason)>,
}

impl Shogi {
    /// A game from `start` with the config's game-end rules (`cfg.sfen` is not read).
    pub fn from_board(start: Board, cfg: &ShogiConfig) -> Shogi {
        let rules = Rules { adjudicate: true, max_plies: cfg.max_plies, declaration: cfg.entering_king == EnteringKing::Csa27 };
        Shogi::with_rules(start, rules)
    }

    /// A game that REPLAYS A RECORD from `start` as recorded: none of the automatic game
    /// ends applies (records end where their source says: resignations, the server's
    /// rulings, undeclared entering-king positions, 512-ply floodgate games). The frames
    /// are those of any other state (they never depend on the game-end rules): the
    /// converter builds its frames through this state.
    pub fn replaying(start: Board) -> Shogi {
        Shogi::with_rules(start, Rules { adjudicate: false, max_plies: 0, declaration: false })
    }

    fn with_rules(start: Board, rules: Rules) -> Shogi {
        let mut occ = HashMap::with_capacity(256);
        occ.insert(start.key(), (1, 0));
        let mut g = Shogi {
            checks: vec![start.is_check()],
            board: start,
            rules,
            occ,
            rep: 0,
            legal: Vec::new(),
            legal_ids: Vec::new(),
            end: None,
        };
        g.refresh_legal();
        if g.rules.adjudicate {
            g.end = g.evaluate_end(0);
        }
        g
    }

    fn refresh_legal(&mut self) {
        self.legal = self.board.legal_moves();
        self.legal_ids.clear();
        self.legal_ids.extend(self.legal.iter().map(action_of));
        self.legal_ids.sort_unstable();
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn side_to_move(&self) -> Seat {
        self.board.turn
    }

    /// Plies played since the start position.
    pub fn plies(&self) -> usize {
        self.checks.len() - 1
    }

    /// Earlier occurrences of the current position (the frames' repetition feature).
    pub fn repetitions(&self) -> u32 {
        self.rep
    }

    /// The side to move is in check.
    pub fn in_check(&self) -> bool {
        *self.checks.last().expect("history")
    }

    /// Legal moves of the side to move (empty when the game is over).
    pub fn legal_moves(&self) -> &[Mv] {
        if self.end.is_some() { &[] } else { &self.legal }
    }

    /// Their action ids, ascending (empty when the game is over).
    pub fn legal_actions_ids(&self) -> &[u32] {
        if self.end.is_some() { &[] } else { &self.legal_ids }
    }

    /// The end of the game: (result for black in {-1, 0, 1}, reason); None = running.
    pub fn end(&self) -> Option<(i8, EndReason)> {
        self.end
    }

    /// The game end at the current position (see the module docs for the order); `first`
    /// = the ply of the current position's first occurrence.
    fn evaluate_end(&self, first: usize) -> Option<(i8, EndReason)> {
        let stm = self.board.turn;
        // result for black of a win of `seat`
        let win = |seat: usize| if seat == BLACK { 1 } else { -1 };
        if self.rep >= 3 {
            // moves first+1 ..= n: the last mover's are n, n-2, ..., first+2
            let n = self.plies();
            let last_mover = stm ^ 1;
            let last_all = (first + 2..=n).step_by(2).all(|k| self.checks[k]);
            let other_all = (first + 1..n).step_by(2).all(|k| self.checks[k]);
            return Some(match (last_all, other_all) {
                (true, false) => (win(stm), EndReason::PerpetualCheck),
                (false, true) => (win(last_mover), EndReason::PerpetualCheck),
                _ => (0, EndReason::Sennichite),
            });
        }
        if self.legal.is_empty() {
            let why = if self.in_check() { EndReason::Checkmate } else { EndReason::NoLegalMove };
            return Some((win(stm ^ 1), why));
        }
        if self.rules.max_plies > 0 && self.plies() >= self.rules.max_plies {
            return Some((0, EndReason::MaxPlies));
        }
        if self.rules.declaration && self.board.can_declare() {
            return Some((win(stm), EndReason::Declaration));
        }
        None
    }

    /// Play a legal move of the side to move.
    pub fn play(&mut self, m: &Mv) -> Result<(), String> {
        if let Some((_, why)) = self.end {
            return Err(format!("the game is over ({})", why.name()));
        }
        let a = action_of(m);
        if self.legal_ids.binary_search(&a).is_err() {
            return Err(format!("{} is not a legal move in {}", usi_move(m), sfen(&self.board)));
        }
        self.board.push(m);
        self.checks.push(self.board.is_check());
        let ply = self.plies() as u32;
        let e = self.occ.entry(self.board.key()).or_insert((0, ply));
        self.rep = e.0;
        e.0 += 1;
        let first = e.1 as usize;
        self.refresh_legal();
        if self.rules.adjudicate {
            self.end = self.evaluate_end(first);
        }
        Ok(())
    }

    fn check_seat(&self, seat: Seat) -> Result<(), RulesError> {
        if self.end.is_some() || seat != self.board.turn {
            rules_bail!("shogi: seat {seat} is not to act");
        }
        Ok(())
    }
}

/// Count the leaf nodes of the game tree `depth` plies deep through the game's action
/// enumeration (`legal_actions` / `apply_action`; game ends included).
pub fn perft(g: &Shogi, depth: usize) -> u64 {
    let ids = g.legal_actions_ids();
    if depth <= 1 {
        return if depth == 0 { 1 } else { ids.len() as u64 };
    }
    let seat = g.side_to_move();
    ids.iter()
        .map(|&a| {
            let mut c = g.clone();
            c.apply_action(seat, a).expect("legal action");
            perft(&c, depth - 1)
        })
        .sum()
}

impl Game for Shogi {
    type Config = ShogiConfig;
    const NAME: &'static str = "shogi";
    /// The frame layout of the `gmdata shogi` converter_version 2 (its frozen schema).
    const VERSION: u32 = 2;

    fn check_config(cfg: &ShogiConfig) -> Result<(), RulesError> {
        if !(1..=MAX_MAX_PLIES).contains(&cfg.max_plies) {
            rules_bail!("max_plies {} outside 1..={MAX_MAX_PLIES}", cfg.max_plies);
        }
        cfg.start().map_err(|e| RulesError(format!("sfen {:?}: {e}", cfg.sfen)))?;
        Ok(())
    }

    fn spec(cfg: &ShogiConfig) -> FrameSpec {
        let n = SQUARES;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: cfg.max_plies.max(1),
            schema: SCHEMA.into(),
            obs: vec![
                TensorSpec::bits("board_seq", &[PIECE_CHANNELS, 9, 9]),
                TensorSpec::f32("hand_seq", &[HAND_SLOTS, HAND_FEATURES]),
                TensorSpec::f32("globals_seq", &[GLOBAL_FEATURES]),
            ],
            legal: vec![
                TensorSpec::bool("mode", &[2]),
                TensorSpec::bool("from", &[n]),
                TensorSpec::bool("to_cond", &[n, n]),
                TensorSpec::bool("promo_cond", &[n, n, 2]),
                TensorSpec::bool("drop_piece", &[HAND_SLOTS]),
                TensorSpec::bool("drop_to_cond", &[HAND_SLOTS, n]),
            ],
            target: vec![
                TensorSpec::i64("dropped", &[]),
                TensorSpec::i64("from_idx", &[]),
                TensorSpec::i64("to_idx", &[]),
                TensorSpec::bool("promoted", &[]),
                TensorSpec::i64("piece_idx", &[]),
                TensorSpec::i64("drop_to_idx", &[]),
                TensorSpec::bits("legal_mode", &[2]),
                TensorSpec::bits("legal_from", &[n]),
                TensorSpec::bits("legal_to_of", &[n]),
                TensorSpec::bits("legal_promo_of", &[2]),
                TensorSpec::bits("legal_drop_piece", &[HAND_SLOTS]),
                TensorSpec::bits("legal_drop_to_of", &[n]),
            ],
            action_keys: ["dropped", "from_idx", "to_idx", "promoted", "piece_idx", "drop_to_idx"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            meta: json!({"global_features": GLOBAL_FEATURES}),
            candidate: None,
        }
    }

    fn new_game(cfg: &ShogiConfig) -> Self {
        Shogi::from_board(cfg.start().expect("shogi: the config was checked"), cfg)
    }

    fn turn(&self) -> Turn {
        if self.end.is_some() { Turn::Terminal } else { Turn::Act(SeatSet::single(self.board.turn)) }
    }

    fn chance_outcomes(&self, _out: &mut Vec<(u32, f64)>) {}

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        rules_bail!("shogi has no chance events (got {outcome})")
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.end.is_none() && seat == self.board.turn {
            out.extend_from_slice(&self.legal_ids);
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        self.check_seat(seat)?;
        if action >= N_ACTIONS {
            rules_bail!("shogi: action {action} out of range");
        }
        self.play(&move_of(action)).map_err(|e| RulesError(format!("shogi: {e}")))
    }

    fn outcome(&self) -> Vec<f32> {
        let r = self.end.map_or(0.0, |(v, _)| v as f32);
        vec![r, -r]
    }

    fn returns(&self) -> Vec<f64> {
        self.outcome().iter().map(|&v| v as f64).collect()
    }

    fn encode_custom(&self, seat: Seat, f: &mut FrameMut<'_>) {
        let mb = seat == BLACK;
        match f.obs.bufs_mut() {
            [BufMut::F32(board), BufMut::F32(hand), BufMut::F32(glb)] => encode_obs(&self.board, self.rep, board, hand, glb),
            _ => unreachable!("shogi obs layout"),
        }
        let n = SQUARES;
        for &a in f.legal_ids.iter() {
            let m = move_of(a);
            let to = pov(m.to, mb) as usize;
            match m.from {
                Some(sq) => {
                    let from = pov(sq, mb) as usize;
                    f.legal.bool(LEGAL_MODE)[0] = 1;
                    f.legal.bool(LEGAL_FROM)[from] = 1;
                    f.legal.bool(LEGAL_TO_COND)[from * n + to] = 1;
                    f.legal.bool(LEGAL_PROMO_COND)[(from * n + to) * 2 + m.promotion as usize] = 1;
                }
                None => {
                    let slot = m.drop as usize - 1;
                    f.legal.bool(LEGAL_MODE)[1] = 1;
                    f.legal.bool(LEGAL_DROP_PIECE)[slot] = 1;
                    f.legal.bool(LEGAL_DROP_TO_COND)[slot * n + to] = 1;
                }
            }
        }
    }

    fn target_custom(&self, seat: Seat, legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let mb = seat == BLACK;
        let (d, fi, ti, pr, pi, dt) = move_indices(&move_of(action), mb);
        t.i64(TGT_DROPPED)[0] = d;
        t.i64(TGT_FROM)[0] = fi;
        t.i64(TGT_TO)[0] = ti;
        t.bool(TGT_PROMOTED)[0] = pr as u8;
        t.i64(TGT_PIECE)[0] = pi;
        t.i64(TGT_DROP_TO)[0] = dt;
        let moves: Vec<Mv> = legal.iter().map(|&a| move_of(a)).collect();
        let f = frame_masks(&moves, mb, fi, ti, pi);
        let bits = |t: &mut TensorsMut<'_>, i: usize, v: &[bool]| {
            for (x, &b) in t.f32(i).iter_mut().zip(v) {
                *x = b as u8 as f32;
            }
        };
        bits(t, TGT_LEGAL_MODE, &f.mode);
        bits(t, TGT_LEGAL_FROM, &f.from);
        bits(t, TGT_LEGAL_TO_OF, &f.to_of);
        bits(t, TGT_LEGAL_PROMO_OF, &f.promo_of);
        bits(t, TGT_LEGAL_DROP_PIECE, &f.drop_piece);
        bits(t, TGT_LEGAL_DROP_TO_OF, &f.drop_to_of);
    }

    fn decode_custom(&self, seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let mb = seat == BLACK;
        let &[dropped, from, to, promoted, piece, drop_to] = keys else { return None };
        let sq = |p: i64| (0..SQUARES as i64).contains(&p).then(|| unpov(p, mb));
        let m = match dropped {
            1 => {
                if !(0..7).contains(&piece) {
                    return None;
                }
                Mv { from: None, to: sq(drop_to)?, promotion: false, drop: (piece + 1) as u8 }
            }
            0 => {
                if !(0..=1).contains(&promoted) {
                    return None;
                }
                Mv { from: Some(sq(from)?), to: sq(to)?, promotion: promoted == 1, drop: NONE }
            }
            _ => return None,
        };
        let a = action_of(&m);
        legal.binary_search(&a).is_ok().then_some(a)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        if action >= N_ACTIONS {
            return format!("a{action}");
        }
        usi_move(&move_of(action))
    }

    /// The SFEN on the first line; the end of the game on the second.
    fn state_string(&self) -> String {
        match self.end {
            Some((r, why)) => format!("{}\nend: {} (black {r:+})", sfen(&self.board), why.name()),
            None => sfen(&self.board),
        }
    }
}

#[cfg(test)]
mod tests;
