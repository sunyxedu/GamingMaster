//! Xiangqi (Chinese chess): the rules (Asian / AXF rules as Fairy-Stockfish implements
//! them) and the from-to frames of the board adapter
//! (`model/games/board.py`: BoardEncoder + FromToDecoder, `fromto.v1` like
//! `breakthrough`). Converter: `gmdata xiangqi` (`rs/gmdata/src/xiangqi.rs`); simulator:
//! gmsim → `env/vec_gmsim.py`; Python registration: `model/games/gmrules/xiangqi.py`.
//!
//! # Rules
//!
//! Standard xiangqi on the 9 x 10 board: Red (seat 0, "w" in the FEN) moves first. King
//! (general) and advisors stay in the palace, elephants on their side of the river
//! (their eye blocks them), the horse is blocked by its leg, the cannon moves like a
//! rook and captures over exactly one screen, soldiers move forward and, once across the
//! river, also sideways; the two kings may never face each other on an open file
//! ("flying general"); a move may not leave the own king in check. Game end, checked
//! after every move in Fairy-Stockfish's search order:
//!
//! 1. natural move limit (Fairy-Stockfish `nMoveRule = 50` with its AXF extension): 100
//!    plies without a capture draw — the checks of the side that gave more checks in
//!    that span count only up to 10 (each further check extends the limit by 2 plies);
//!    not when the side to move is checkmated;
//! 2. repetition (`nFoldRule = 3`, `perpetualCheckIllegal`, `chasingRule = AXF`): the
//!    third occurrence of a position (placement + side to move, since the last capture)
//!    ends the game — the side that gave check with every move of the cycle loses
//!    (both: draw); otherwise the side that chased (AXF chase: attacked an unprotected
//!    piece or a stronger piece, not the king or an uncrossed soldier; Fairy-Stockfish
//!    `Position::chased()`, see `board.rs`) with every move loses (both: draw);
//!    otherwise a draw;
//! 3. no legal move: the side to move loses (checkmate AND stalemate);
//! 4. neither side has a rook, horse, cannon or soldier: draw (Fairy-Stockfish
//!    `has_insufficient_material` for both sides, the automatic draw of Fairy-Stockfish
//!    GUIs such as pychess).
//!
//! Resignation, draw agreement, time forfeits and arbiter rulings are not moves: a
//! record ending that way stops before a terminal state and the converter labels it
//! with the declared result (`traj::Record::adjudicated`). Pikafish differs only in the
//! move limit (60 moves = 120 plies, same 10-check rule); its chase detection follows
//! the same AXF approximation.
//!
//! Actions: `from * 90 + to` over absolute squares (`sq = rank * 9 + file`, rank 0 =
//! Red's back rank = UCI rank 1, file 0 = "a"); [`Game::action_string`] is the
//! Fairy-Stockfish / UCI coordinate move ("h3e3", "a10a9").
//!
//! # Frames (interleaved: one stream per game, every frame from the mover's view)
//!
//! The board is seen from the mover's side: Red to move row = 9 − rank, col = file;
//! Black to move rotated by 180° (row = rank, col = 8 − file) — own pieces at the
//! bottom, the mover's left file in column 0. POV square = row · 9 + col.
//!
//! ```text
//! obs     board_seq    [19, 10, 9] bits  0-6 own K A B N R C P, 7-13 opponent's,
//!                                        14 / 15 the opponent's last move from / to,
//!                                        16 own pieces the last move chases (AXF),
//!                                        17 opponent pieces the mover's previous move
//!                                        chases (followed through the reply), 18 ones
//!         globals_seq  [7] f32           mover is Red, plies / 1000, plies since the
//!                                        last capture / 100 (≤ 2), mover in check,
//!                                        position seen before (a repetition), the
//!                                        opponent's / the mover's run of consecutive
//!                                        checks / 10 (≤ 1)
//! legal   from         [90] bool         squares with a legal move
//!         to_cond      [90, 90] bool     destinations per from-square
//! target  from_idx     [] i64            action key: POV from-square
//!         to_idx       [] i64            action key: POV to-square
//!         legal_from   [90] bits         IL legality (= legal.from)
//!         legal_to_of  [90] bits         destinations of the taken move's from-square
//! ```
//!
//! Shard meta: board_h 10, board_w 9, piece_channels 19, global_features 7.

mod board;

pub use board::{
    parse_square, square_name, Bb, Board, Sq, Squares, ADVISOR, BLACK, CANNON, ELEPHANT, FILES, HORSE, KING,
    PIECE_CHARS, PIECE_TYPES, RANKS, RED, ROOK, SOLDIER, SQUARES, START_FEN,
};
use board::undo_move_board;

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Hard bound on plies per game (records longer than this are rejected, simulations
/// truncated as draws). Master games reach ~430 plies.
pub const MAX_PLIES: usize = 1000;
/// Planes of `obs.board_seq`.
pub const PIECE_CHANNELS: usize = 19;
/// Width of `obs.globals_seq`.
pub const GLOBAL_FEATURES: usize = 7;

const OBS_BOARD: usize = 0;
const OBS_GLOBALS: usize = 1;
const LEGAL_FROM: usize = 0;
const LEGAL_TO_COND: usize = 1;
const TGT_FROM: usize = 0;
const TGT_TO: usize = 1;
const TGT_LEGAL_FROM: usize = 2;
const TGT_LEGAL_TO_OF: usize = 3;

const PLANE_LAST_FROM: usize = 14;
const PLANE_LAST_TO: usize = 15;
const PLANE_CHASED_OWN: usize = 16;
const PLANE_CHASED_THEIRS: usize = 17;
const PLANE_ONES: usize = 18;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct XiangqiConfig {
    /// Start position, a Fairy-Stockfish xiangqi FEN ("" = the standard initial
    /// position). The frames do not depend on it.
    pub fen: String,
}

/// Why a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    Checkmate,
    Stalemate,
    /// third occurrence of a position, nobody checked / chased throughout: draw
    Repetition,
    /// repetition with perpetual check: the checking side loses (both: draw)
    PerpetualCheck,
    /// repetition with a perpetual chase: the chasing side loses (both: draw)
    PerpetualChase,
    /// 100 plies without a capture (natural move limit): draw
    MoveLimit,
    /// neither side has an attacking piece: draw
    Material,
}

impl EndReason {
    pub fn name(self) -> &'static str {
        match self {
            EndReason::Checkmate => "checkmate",
            EndReason::Stalemate => "stalemate",
            EndReason::Repetition => "repetition",
            EndReason::PerpetualCheck => "perpetual_check",
            EndReason::PerpetualChase => "perpetual_chase",
            EndReason::MoveLimit => "move_limit",
            EndReason::Material => "material",
        }
    }
}

/// Per-position facts of the game so far (Fairy-Stockfish `StateInfo`).
#[derive(Clone, Debug)]
struct Hist {
    key: u64,
    /// the move that led here (None at the start position)
    mv: Option<(Sq, Sq)>,
    /// the side to move is in check
    in_check: bool,
    /// pieces of the side to move the last move chases
    chased: Bb,
    /// king blockers per king color
    blockers: [Bb; 2],
    /// plies since the last capture (Fairy-Stockfish `rule50`)
    rule50: u32,
}

#[derive(Clone, Debug)]
pub struct Xiangqi {
    board: Board,
    hist: Vec<Hist>,
    /// legal moves of the side to move, ascending
    legal: Vec<u32>,
    /// terminal: (result for Red in {-1, 0, 1}, why)
    end: Option<(i8, EndReason)>,
    /// Fairy-Stockfish gamePly of the start position (FEN move numbers)
    start_ply: u32,
}

/// POV square of `sq` for `seat` (see the module docs).
#[inline]
pub fn pov(seat: Seat, sq: Sq) -> usize {
    let (f, r) = (sq % FILES, sq / FILES);
    if seat == RED { (RANKS - 1 - r) * FILES + f } else { r * FILES + (FILES - 1 - f) }
}

/// Inverse of [`pov`].
#[inline]
pub fn unpov(seat: Seat, p: usize) -> Sq {
    let (row, col) = (p / FILES, p % FILES);
    if seat == RED { (RANKS - 1 - row) * FILES + col } else { row * FILES + (FILES - 1 - col) }
}

/// The action id of a move.
#[inline]
pub fn action_of(from: Sq, to: Sq) -> u32 {
    (from * SQUARES + to) as u32
}

/// (from, to) of an action id.
#[inline]
pub fn move_of(action: u32) -> (Sq, Sq) {
    (action as usize / SQUARES, action as usize % SQUARES)
}

/// UCI coordinate string of an action ("h3e3").
pub fn uci(action: u32) -> String {
    let (f, t) = move_of(action);
    format!("{}{}", square_name(f), square_name(t))
}

/// Parse a UCI coordinate move ("h3e3", "a10a9"); None when malformed.
pub fn parse_uci(s: &str) -> Option<u32> {
    let s = s.trim();
    // the second square starts at the second letter
    let split = s.char_indices().skip(1).find(|(_, c)| c.is_ascii_alphabetic())?.0;
    Some(action_of(parse_square(&s[..split])?, parse_square(&s[split..])?))
}

impl Xiangqi {
    /// A game starting from `fen` (Fairy-Stockfish xiangqi FEN).
    pub fn from_fen(fen: &str) -> Result<Xiangqi, String> {
        let (board, rule50) = Board::from_fen(fen)?;
        let fullmove: u32 = fen.split_whitespace().nth(5).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
        let start_ply = 2 * (fullmove - 1) + board.side as u32;
        let blockers = [board.king_blockers(BLACK), board.king_blockers(RED)];
        let root = Hist {
            key: board.key(),
            mv: None,
            in_check: board.checkers() != 0,
            chased: 0,
            blockers,
            rule50,
        };
        let mut g = Xiangqi { board, hist: Vec::with_capacity(256), legal: Vec::with_capacity(64), end: None, start_ply };
        g.hist.push(root);
        g.board.legal_moves(&mut g.legal);
        g.end = g.evaluate_end();
        Ok(g)
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn side_to_move(&self) -> Seat {
        self.board.side
    }

    /// Legal moves of the side to move (ascending action ids; empty when the game is
    /// over).
    pub fn legal_moves(&self) -> &[u32] {
        if self.end.is_some() { &[] } else { &self.legal }
    }

    /// Moves played since the start position.
    pub fn plies(&self) -> usize {
        self.hist.len() - 1
    }

    /// The end of the game: (result for Red in {-1, 0, 1}, reason); None = running.
    pub fn end(&self) -> Option<(i8, EndReason)> {
        self.end
    }

    /// The side to move is in check.
    pub fn in_check(&self) -> bool {
        self.hist.last().expect("history").in_check
    }

    /// Plies since the last capture.
    pub fn rule50(&self) -> u32 {
        self.hist.last().expect("history").rule50
    }

    /// Play a legal move of the side to move.
    pub fn play(&mut self, action: u32) -> Result<(), String> {
        if let Some((_, why)) = self.end {
            return Err(format!("the game is over ({})", why.name()));
        }
        self.play_through(action)?;
        self.end = self.evaluate_end();
        Ok(())
    }

    /// Play a legal move WITHOUT adjudicating the game end (Fairy-Stockfish's
    /// position semantics: the repetition / move-limit state is then read with
    /// [`Xiangqi::optional_game_end`]; tests of the reference's own examples).
    pub fn play_through(&mut self, action: u32) -> Result<(), String> {
        if self.legal.binary_search(&action).is_err() {
            return Err(format!("{} is not a legal move in {}", uci(action), self.fen()));
        }
        let (from, to) = move_of(action);
        let prev = self.hist.last().expect("history").clone();
        let captured = self.board.make_move(from, to).is_some();
        let blockers = [self.board.king_blockers(BLACK), self.board.king_blockers(RED)];
        let chased = self.board.chased(from, to, captured, &blockers, Some(&prev.blockers));
        self.hist.push(Hist {
            key: self.board.key(),
            mv: Some((from, to)),
            in_check: self.board.checkers() != 0,
            chased,
            blockers,
            rule50: if captured { 0 } else { prev.rule50 + 1 },
        });
        self.legal.clear();
        self.board.legal_moves(&mut self.legal);
        self.end = None;
        Ok(())
    }

    /// Fairy-Stockfish `is_optional_game_end` at the current position (at the root of a
    /// search, `ply = 0`): the natural move limit and the repetition rules; the value
    /// is for the SIDE TO MOVE (+1 win, 0 draw, −1 loss).
    pub fn optional_game_end(&self) -> Option<(i8, EndReason)> {
        let h = &self.hist;
        let n = h.len() - 1;
        let st = &h[n];
        if st.rule50 > 99 && (!st.in_check || !self.legal.is_empty()) {
            let mut offset = 0i64;
            if n >= 20 {
                let end = (st.rule50 as usize).min(n);
                let mut check_them = st.in_check as i64;
                let mut check_us = h[n - 1].in_check as i64;
                let mut i = 2;
                while i < end {
                    check_them += h[n - i].in_check as i64;
                    check_us += h[n - i - 1].in_check as i64;
                    i += 2;
                }
                offset = 2 * (check_them.max(check_us) - 10).max(0);
            }
            if st.rule50 as i64 - offset > 99 {
                return Some((0, EndReason::MoveLimit));
            }
        }
        let end = (st.rule50 as usize).min(n);
        if end >= 4 {
            let mut stp = n - 2;
            let mut cnt = 0;
            let mut perp_them = st.in_check && h[stp].in_check;
            let mut perp_us = h[n - 1].in_check && h[stp - 1].in_check;
            let mut chase_them = undo_move_board(st.chased, h[n - 1].mv) & h[stp].chased;
            let mut chase_us = undo_move_board(h[n - 1].chased, h[stp].mv) & h[stp - 1].chased;
            let mut i = 4;
            while i <= end {
                // chased pieces are empty when there is no previous move
                if i != n {
                    chase_them = undo_move_board(chase_them, h[stp - 1].mv) & h[stp - 2].chased;
                }
                stp -= 2;
                perp_them &= h[stp].in_check;
                if h[stp].key == st.key {
                    cnt += 1;
                    if cnt + 1 >= 3 {
                        let pick = |them: bool, us: bool| if !us { 1 } else if !them { -1 } else { 0 };
                        return Some(if perp_them || perp_us {
                            (pick(perp_them, perp_us), EndReason::PerpetualCheck)
                        } else if chase_them != 0 || chase_us != 0 {
                            (pick(chase_them != 0, chase_us != 0), EndReason::PerpetualChase)
                        } else {
                            (0, EndReason::Repetition)
                        });
                    }
                }
                if i < end {
                    perp_us &= h[stp - 1].in_check;
                    chase_us = undo_move_board(chase_us, h[stp].mv) & h[stp - 1].chased;
                }
                i += 2;
            }
        }
        None
    }

    /// The game end at the current position (see the module docs for the order).
    fn evaluate_end(&self) -> Option<(i8, EndReason)> {
        let red = |v: i8| if self.board.side == RED { v } else { -v };
        if let Some((v, why)) = self.optional_game_end() {
            return Some((red(v), why));
        }
        if self.legal.is_empty() {
            return Some((red(-1), if self.in_check() { EndReason::Checkmate } else { EndReason::Stalemate }));
        }
        if self.board.insufficient_material(RED) && self.board.insufficient_material(BLACK) {
            return Some((0, EndReason::Material));
        }
        None
    }

    /// Earlier occurrences of the current position within the repetition window.
    pub fn repetitions(&self) -> usize {
        let h = &self.hist;
        let n = h.len() - 1;
        let end = (h[n].rule50 as usize).min(n);
        (4..=end).step_by(2).filter(|&i| h[n - i].key == h[n].key).count()
    }

    /// Consecutive moves that gave check, ending with the last move (`own` = false) or
    /// with the move before it (`own` = true: the side to move's own run).
    fn check_run(&self, own: bool) -> usize {
        let n = self.hist.len() - 1;
        let mut k = if own { n.wrapping_sub(1) } else { n };
        let mut run = 0;
        while k >= 1 && k <= n && self.hist[k].in_check {
            run += 1;
            if k < 2 {
                break;
            }
            k -= 2;
        }
        run
    }

    /// Fairy-Stockfish FEN of the current position (board, side, "- -", plies since the
    /// last capture, move number).
    pub fn fen(&self) -> String {
        let ply = self.start_ply + self.plies() as u32;
        let fullmove = 1 + (ply - self.board.side as u32) / 2;
        format!("{} - - {} {}", self.board.fen_board(), self.rule50(), fullmove)
    }

    fn check_seat(&self, seat: Seat) -> Result<(), RulesError> {
        if self.end.is_some() || seat != self.board.side {
            rules_bail!("xiangqi: seat {seat} is not to act");
        }
        Ok(())
    }
}

impl Game for Xiangqi {
    type Config = XiangqiConfig;
    const NAME: &'static str = "xiangqi";
    const VERSION: u32 = 1;

    fn check_config(cfg: &XiangqiConfig) -> Result<(), RulesError> {
        if !cfg.fen.is_empty() {
            Board::from_fen(&cfg.fen).map_err(|e| RulesError(format!("fen {:?}: {e}", cfg.fen)))?;
        }
        Ok(())
    }

    fn spec(_cfg: &XiangqiConfig) -> FrameSpec {
        let n = SQUARES;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: MAX_PLIES,
            schema: "fromto.v1".into(),
            obs: vec![
                TensorSpec::bits("board_seq", &[PIECE_CHANNELS, RANKS, FILES]),
                TensorSpec::f32("globals_seq", &[GLOBAL_FEATURES]),
            ],
            legal: vec![TensorSpec::bool("from", &[n]), TensorSpec::bool("to_cond", &[n, n])],
            target: vec![
                TensorSpec::i64("from_idx", &[]),
                TensorSpec::i64("to_idx", &[]),
                TensorSpec::bits("legal_from", &[n]),
                TensorSpec::bits("legal_to_of", &[n]),
            ],
            action_keys: vec!["from_idx".into(), "to_idx".into()],
            meta: json!({
                "board_h": RANKS,
                "board_w": FILES,
                "piece_channels": PIECE_CHANNELS,
                "global_features": GLOBAL_FEATURES,
            }),
            candidate: None,
        }
    }

    fn new_game(cfg: &XiangqiConfig) -> Self {
        let fen = if cfg.fen.is_empty() { START_FEN } else { cfg.fen.as_str() };
        Xiangqi::from_fen(fen).expect("xiangqi: the config was checked")
    }

    fn turn(&self) -> Turn {
        if self.end.is_some() { Turn::Terminal } else { Turn::Act(SeatSet::single(self.board.side)) }
    }

    fn chance_outcomes(&self, _out: &mut Vec<(u32, f64)>) {}

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        rules_bail!("xiangqi has no chance events (got {outcome})")
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.end.is_none() && seat == self.board.side {
            out.extend_from_slice(&self.legal);
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        self.check_seat(seat)?;
        self.play(action).map_err(|e| RulesError(format!("xiangqi: {e}")))
    }

    fn outcome(&self) -> Vec<f32> {
        let r = self.end.map_or(0.0, |(v, _)| v as f32);
        vec![r, -r]
    }

    fn returns(&self) -> Vec<f64> {
        self.outcome().iter().map(|&v| v as f64).collect()
    }

    fn encode_custom(&self, seat: Seat, f: &mut FrameMut<'_>) {
        let n = SQUARES;
        let b = &self.board;
        let h = self.hist.last().expect("history");
        {
            let planes = f.obs.f32(OBS_BOARD);
            for sq in Squares(b.occupied()) {
                let (c, pt) = b.piece_at(sq).expect("occupied");
                let ch = if c == seat { pt } else { PIECE_TYPES + pt };
                planes[ch * n + pov(seat, sq)] = 1.0;
            }
            if let Some((from, to)) = h.mv {
                planes[PLANE_LAST_FROM * n + pov(seat, from)] = 1.0;
                planes[PLANE_LAST_TO * n + pov(seat, to)] = 1.0;
            }
            for sq in Squares(h.chased) {
                planes[PLANE_CHASED_OWN * n + pov(seat, sq)] = 1.0;
            }
            if self.hist.len() >= 2 {
                // the mover's previous move chased these opponent pieces; follow them
                // through the opponent's reply
                let mut theirs = self.hist[self.hist.len() - 2].chased;
                if let Some((from, to)) = h.mv {
                    if theirs & board::bit(from) != 0 {
                        theirs = (theirs ^ board::bit(from)) | board::bit(to);
                    }
                }
                for sq in Squares(theirs) {
                    planes[PLANE_CHASED_THEIRS * n + pov(seat, sq)] = 1.0;
                }
            }
            planes[PLANE_ONES * n..(PLANE_ONES + 1) * n].fill(1.0);
        }
        let g = f.obs.f32(OBS_GLOBALS);
        g[0] = (seat == RED) as u8 as f32;
        g[1] = (self.plies() as f32 / MAX_PLIES as f32).min(1.0);
        g[2] = (h.rule50 as f32 / 100.0).min(2.0);
        g[3] = h.in_check as u8 as f32;
        g[4] = (self.repetitions() > 0) as u8 as f32;
        g[5] = (self.check_run(false) as f32 / 10.0).min(1.0);
        g[6] = (self.check_run(true) as f32 / 10.0).min(1.0);
        for &a in f.legal_ids.iter() {
            let (from, to) = move_of(a);
            let (pf, pt) = (pov(seat, from), pov(seat, to));
            f.legal.bool(LEGAL_FROM)[pf] = 1;
            f.legal.bool(LEGAL_TO_COND)[pf * n + pt] = 1;
        }
    }

    fn target_custom(&self, seat: Seat, legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let (from, to) = move_of(action);
        let pf = pov(seat, from);
        t.i64(TGT_FROM)[0] = pf as i64;
        t.i64(TGT_TO)[0] = pov(seat, to) as i64;
        for &a in legal {
            let (f2, t2) = move_of(a);
            let p2 = pov(seat, f2);
            t.f32(TGT_LEGAL_FROM)[p2] = 1.0;
            if p2 == pf {
                t.f32(TGT_LEGAL_TO_OF)[pov(seat, t2)] = 1.0;
            }
        }
    }

    fn decode_custom(&self, seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let n = SQUARES as i64;
        if !(0..n).contains(&keys[0]) || !(0..n).contains(&keys[1]) {
            return None;
        }
        let a = action_of(unpov(seat, keys[0] as usize), unpov(seat, keys[1] as usize));
        legal.binary_search(&a).is_ok().then_some(a)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        uci(action)
    }

    /// The FEN on the first line; the end of the game on the second.
    fn state_string(&self) -> String {
        match self.end {
            Some((r, why)) => format!("{}\nend: {} (red {r:+})", self.fen(), why.name()),
            None => self.fen(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{decode_action, encode_frame, encode_target, new_target, Frame};
    use crate::game::Layout;
    use crate::invariance::check_game;
    use crate::trajectory::{replay, Event};

    fn play_all(g: &mut Xiangqi, moves: &[&str]) {
        for m in moves {
            let a = parse_uci(m).unwrap();
            g.play(a).unwrap_or_else(|e| panic!("{m}: {e}"));
        }
    }

    #[test]
    fn start_position_moves_and_uci() {
        let g = Xiangqi::new_game(&XiangqiConfig::default());
        assert_eq!(g.legal_moves().len(), 44);
        assert!(g.legal_moves().windows(2).all(|w| w[0] < w[1]));
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(RED)));
        assert_eq!(g.fen(), START_FEN);
        let a = parse_uci("h3e3").unwrap();
        assert!(g.legal_moves().contains(&a));
        assert_eq!(uci(a), "h3e3");
        assert_eq!(parse_uci("a10a9"), Some(action_of(81, 72)));
        assert_eq!(Xiangqi::spec(&XiangqiConfig::default()).layout(), Layout::Interleaved);
        let mut g2 = g.clone();
        play_all(&mut g2, &["h3e3", "h10g8"]);
        assert_eq!(g2.fen(), "rnbakab1r/9/1c4nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR w - - 2 2");
    }

    #[test]
    fn checkmate_stalemate_and_material() {
        // Rb1-b10 mates the lone king (a9 rook holds rank 9)
        let mut g = Xiangqi::from_fen("4k4/R8/9/9/9/9/9/9/9/1R1K5 w - - 0 1").unwrap();
        play_all(&mut g, &["b1b10"]);
        assert_eq!(g.end(), Some((1, EndReason::Checkmate)));
        assert_eq!(g.outcome(), vec![1.0, -1.0]);
        assert_eq!(g.turn(), Turn::Terminal);
        // stalemate loses: black king d10 has no move (e-file rook, rank-9 rook)
        let g = Xiangqi::from_fen("3k5/R8/9/9/9/9/9/9/9/4RK3 b - - 0 1").unwrap();
        assert_eq!(g.end(), Some((1, EndReason::Stalemate)));
        // kings (and defenders) only: draw
        let g = Xiangqi::from_fen("3k5/9/4b4/9/9/9/9/4B4/9/5K3 w - - 0 1").unwrap();
        assert_eq!(g.end(), Some((0, EndReason::Material)));
        assert!(Xiangqi::from_fen("3k5/9/9/9/9/9/9/9/9/4KN3 w - - 0 1").unwrap().end().is_none());
    }

    #[test]
    fn move_limit_and_halfmove_clock() {
        // a rook shuttles without capturing: 100 plies after the start (no checks) draw
        let mut g = Xiangqi::from_fen("3k5/9/9/9/9/9/9/9/R8/4K4 w - - 0 1").unwrap();
        let cycle = ["a2b2", "d10d9", "b2c2", "d9d10", "c2b2", "d10d9", "b2a2", "d9d10"];
        let mut i = 0;
        while g.end().is_none() {
            let a = parse_uci(cycle[i % cycle.len()]).unwrap();
            if g.legal_moves().contains(&a) {
                g.play(a).unwrap();
            } else {
                g.play(g.legal_moves()[0]).unwrap();
            }
            i += 1;
        }
        // the shuttle repeats positions: the repetition rule ends it first
        assert!(matches!(g.end(), Some((0, EndReason::Repetition)) | Some((0, EndReason::MoveLimit))), "{:?}", g.end());
        // the FEN's halfmove clock counts: 99 + 1 quiet move = the limit
        let mut g = Xiangqi::from_fen("3k5/9/9/9/9/9/9/9/R8/4K4 w - - 99 60").unwrap();
        assert!(g.end().is_none());
        play_all(&mut g, &["a2b2"]);
        assert_eq!(g.end(), Some((0, EndReason::MoveLimit)));
        let mut g = Xiangqi::from_fen("3k5/9/9/9/9/9/9/9/R8/4K4 w - - 98 60").unwrap();
        play_all(&mut g, &["a2a3"]);
        assert!(g.end().is_none());
    }

    /// Fairy-Stockfish's own AXF tests (pyffish test.py `test_is_optional_game_end`;
    /// examples of http://www.asianxiangqi.org/English/AXF_rules_Eng.pdf): the value
    /// after the moves is for the side to move (+1 = Fairy-Stockfish VALUE_MATE).
    const AXF_CASES: &[(&str, &str, &[&str], i8)] = &[
        ("direct chase by cannon", "2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1",
         &["c3a3", "a8b8", "a3b3", "b8a8", "b3a3", "a8b8", "a3b3", "b8a8", "b3a3"], 1),
        ("chase with chasing side to move", "2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1",
         &["c3a3", "a8b8", "a3b3", "b8a8", "b3a3", "a8b8", "a3b3", "b8a8", "b3a3", "a8b8", "a3b3", "b8a8"], -1),
        ("discovered chase by cannon", "2bakabr1/9/9/r1p1p1p2/p7R/P8/9/9/9/CC1AKA3 w - - 0 1",
         &["a5a6", "a7b7", "a6b6", "b7a7", "b6a6", "a7b7", "a6b6", "b7a7", "b6a6"], 1),
        ("chase by soldier", "2bakabr1/9/9/r1p1p1p2/p7R/P8/9/9/9/1C1AKA3 w - - 0 1",
         &["a5a6", "a7b7", "a6b6", "b7a7", "b6a6", "a7b7", "a6b6", "b7a7", "b6a6"], 0),
        ("discovered and anti-discovered chase", "5k3/9/9/5C3/5c3/5C3/9/9/5p3/4K4 w - - 0 1",
         &["f5d5", "f6d6", "d5f5", "d6f6", "f5d5", "f6d6", "d5f5", "d6f6"], -1),
        ("perpetual check vs intermittent checks", "9/3kc4/3a5/3P5/9/4p4/9/4K4/9/3C5 w - - 0 1",
         &["d7e7", "e5d5", "e7d7", "d5e5", "d7e7", "e5d5", "e7d7", "d5e5"], 1),
        ("perpetual check by soldier", "3k5/9/9/9/9/5p3/9/5p3/5K3/5C3 w - - 0 1",
         &["f2e2", "f3e3", "e2f2", "e3f3", "f2e2", "f3e3", "e2f2", "e3f3"], 1),
        ("perpetual check 3", "3k5/4P4/4b4/3C5/4c4/9/9/9/9/5K3 w - - 0 1",
         &["d7e7", "e8g6", "e7d7", "g6e8", "d7e7", "e8g6", "e7d7", "g6e8"], 1),
        ("perpetual check 4", "3k5/9/9/9/9/9/9/9/cr1CAK3/9 w - - 0 1",
         &["d2d4", "b2b4", "d4d2", "b4b2", "d2d4", "b2b4", "d4d2", "b4b2"], 1),
        ("perpetual check 5", "4k4/9/4b4/2c2nR2/9/9/9/9/9/3K5 w - - 0 1",
         &["g7g6", "f7g9", "g6g7", "g9f7", "g7g6", "f7g9", "g6g7", "g9f7"], 1),
        ("perpetual check 6", "3P5/3k5/3nn4/9/9/9/9/9/9/5K3 w - - 0 1",
         &["d10e10", "d9e9", "e10d10", "e9d9", "d10e10", "d9e9", "e10d10", "e9d9"], 1),
        ("perpetual check 7", "4ck3/9/9/9/9/2r1R4/9/9/4A4/3AK4 w - - 0 1",
         &["e5e4", "c5c4", "e4e5", "c4c5", "e5e4", "c5c4", "e4e5", "c4c5"], 1),
        ("perpetual check 8", "4k4/9/9/c1c6/9/r8/9/9/C8/3K5 w - - 0 1",
         &["a2c2", "a5c5", "c2a2", "c5a5", "a2c2", "a5c5", "c2a2", "c5a5"], 1),
        ("mutual perpetual check", "9/4c4/3k5/3r5/9/9/4C4/9/4K4/3R5 w - - 0 1",
         &["e4d4", "d7e7", "d4e4", "e7d7", "e4d4", "d7e7", "d4e4", "e7d7"], 0),
        ("mutual perpetual check 2", "3k5/6c2/9/7P1/6c2/6P2/9/9/9/5K3 w - - 0 1",
         &["h7g7", "g6h6", "g7h7", "h6g6", "h7g7", "g6h6", "g7h7", "h6g6"], 0),
        ("mutual perpetual check 3", "4ck3/9/9/9/9/2r1R1N2/6N2/9/4A4/3AK4 w - - 0 1",
         &["e5e4", "c5c4", "e4e5", "c4c5", "e5e4", "c5c4", "e4e5", "c4c5"], 0),
        ("mutual perpetual check 4", "5k3/9/9/c8/9/P1P6/9/2C6/9/3K5 w - - 0 1",
         &["c3a3", "a7c7", "a3c3", "c7a7", "c3a3", "a7c7", "a3c3", "c7a7"], 0),
        ("mutual chase", "4k4/7n1/9/4pR3/9/9/4P4/9/9/4K4 w - - 0 1",
         &["f7h7", "h9f8", "h7h8", "f8g6", "h8g8", "g6i7", "g8g7", "i7h9", "g7h7",
           "h9f8", "h7h8", "f8g6", "h8g8", "g6i7", "g8g7", "i7h9", "g7h7"], 0),
        ("mutual chase 2", "4k4/9/r1r6/9/PPPP5/9/9/9/1C7/5K3 w - - 0 1",
         &["b2a2", "a8b8", "a2c2", "c8d8", "c2b2", "b8a8", "b2d2", "d8c8", "d2a2",
           "a8b8", "a2c2", "c8d8", "c2b2", "b8a8", "b2d2", "d8c8", "d2a2"], 0),
        ("D106 attack does not change", "3k2b2/4P4/4b4/9/8p/6Bc1/6P1P/3AB4/4pp3/1p1K3R1[] w - - 0 1",
         &["h1h2", "h5h4", "h2h1", "h4h5", "h1h2", "h5h4", "h2h1", "h4h5"], 0),
        ("D39 pinned chariot + mutual chase", "2baka1r1/C4rN2/9/1Rp1p4/9/9/4P4/9/4A4/4KA3 w - - 0 1",
         &["b7b9", "f10e9", "b9b10", "e9f10", "b10b9", "f10e9", "b9b10", "e9f10", "b10b9"], 1),
        ("D39 2", "5k3/9/9/9/9/9/7r1/9/2nRA3c/4K4 w - - 0 1",
         &["e2f1", "h4h2", "f1e2", "h2h4", "e2f1", "h4h2", "f1e2", "h2h4"], 1),
        ("creating pins to undermine root", "4k4/4c4/9/4p4/9/9/3rn4/3NR4/4K4/9 b - - 0 1",
         &["e4g5", "e2f2", "g5e4", "f2e2", "e4g5", "e2f2", "g5e4", "f2e2"], -1),
        ("discovered check capture threat by rook", "5k3/9/9/9/9/1N2P1C2/9/4BC3/9/cr1RK4 w - - 0 1",
         &["b5c3", "b1c1", "c3b5", "c1b1", "b5c3", "b1c1", "c3b5", "c1b1"], 1),
        ("pin + discovered check threat by horse", "5k3/9/9/9/9/4c4/3n5/3NBA3/4A4/4K4 w - - 0 1",
         &["e1d1", "e5d5", "d1e1", "d5e5", "e1d1", "e5d5", "d1e1", "d5e5"], 1),
        ("pin + discovered check threat by rook", "5k3/9/9/9/9/4c4/3r5/3NB4/4A4/4K4 w - - 0 1",
         &["e1d1", "e5d5", "d1e1", "d5e5", "e1d1", "e5d5", "d1e1", "d5e5"], 1),
        ("x-ray protected discovered check", "5k3/9/9/9/9/9/9/9/9/3NK1cr1 w - - 0 1",
         &["d1c3", "h1h3", "c3d1", "h3h1", "d1c3", "h1h3", "c3d1", "h3h1"], 1),
        ("no overprotection by king", "3k5/9/9/3n5/9/9/3r5/9/9/3NK4 w - - 0 1",
         &["d1c3", "d4c4", "c3d1", "c4d4", "d1c3", "d4c4", "c3d1", "c4d4"], 0),
        ("overprotection by king", "3k5/9/9/9/9/9/3r5/9/9/3NK4 w - - 0 1",
         &["d1c3", "d4c4", "c3d1", "c4d4", "d1c3", "d4c4", "c3d1", "c4d4"], 1),
        ("fake protection by cannon", "5k3/9/9/9/9/1C7/1r7/9/1C7/4K4 w - - 0 1",
         &["b5c5", "b4c4", "c5b5", "c4b4", "b5c5", "b4c4", "c5b5", "c4b4"], 1),
        ("fake protection by cannon + mutual chase", "4ka3/c2R1R2c/4b4/9/9/9/9/9/9/4K4 w - - 0 1",
         &["f9f7", "f10e9", "f7f9", "e9f10", "f9f7", "f10e9", "f7f9", "e9f10"], 0),
    ];

    #[test]
    fn repetition_rules_match_fairy_stockfish_axf_examples() {
        for &(name, fen, moves, want) in AXF_CASES {
            let mut g = Xiangqi::from_fen(fen).unwrap();
            for m in moves {
                g.play_through(parse_uci(m).unwrap()).unwrap_or_else(|e| panic!("{name}: {m}: {e}"));
            }
            let got = g.optional_game_end();
            assert!(got.is_some(), "{name}: no game end");
            assert_eq!(got.unwrap().0, want, "{name}: {got:?}");
            // the adjudicating game ends at the third occurrence with that result
            let mut g = Xiangqi::from_fen(fen).unwrap();
            for m in moves {
                if g.end().is_some() {
                    break;
                }
                g.play(parse_uci(m).unwrap()).unwrap();
            }
            assert!(g.end().is_some(), "{name}: the game did not end");
        }
        // "mutual pins by flying generals": a game end, Fairy-Stockfish leaves the value open
        let mut g = Xiangqi::from_fen("4k4/9/9/9/4n4/9/5C3/9/4N4/4K4 w - - 0 1").unwrap();
        for m in ["e2g1", "e10f10", "g1e2", "f10e10", "e2g1", "e10f10", "g1e2", "f10e10"] {
            g.play_through(parse_uci(m).unwrap()).unwrap();
        }
        assert!(g.optional_game_end().is_some());
    }

    #[test]
    fn frames_rotate_for_black_and_decode() {
        let cfg = XiangqiConfig::default();
        let spec = Xiangqi::spec(&cfg);
        spec.validate().unwrap();
        let red = Xiangqi::new_game(&cfg);
        let black = Xiangqi::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1").unwrap();
        let mut fr = Frame::new(&spec);
        let mut fb = Frame::new(&spec);
        encode_frame(&red, RED, &spec, &mut fr.as_mut(&spec)).unwrap();
        encode_frame(&black, BLACK, &spec, &mut fb.as_mut(&spec)).unwrap();
        // the start position is symmetric under the rotation: the same planes
        assert_eq!(fr.obs[0], fb.obs[0]);
        assert_eq!(fr.legal, fb.legal);
        let (gr, gb) = (fr.obs[1].as_f32().unwrap(), fb.obs[1].as_f32().unwrap());
        assert_eq!((gr[0], gb[0]), (1.0, 0.0));
        // own king at the bottom centre of the POV board (row 9, col 4)
        let planes = fr.obs[0].as_f32().unwrap();
        assert_eq!(planes[KING * 90 + 9 * 9 + 4], 1.0);
        assert_eq!(planes[(PIECE_TYPES + KING) * 90 + 4], 1.0);
        assert!(planes[PLANE_ONES * 90..].iter().all(|&x| x == 1.0));
        // action keys round trip, IL masks = legal tables
        let a = parse_uci("b3e3").unwrap();
        let mut t = new_target(&spec);
        encode_target(&black, BLACK, &spec, &fb.legal_ids, action_of(89 - parse_square("b3").unwrap(), 89 - parse_square("e3").unwrap()),
                      &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
        let keys = [t[0].as_i64().unwrap()[0], t[1].as_i64().unwrap()[0]];
        let mut t2 = new_target(&spec);
        encode_target(&red, RED, &spec, &fr.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t2)).unwrap();
        assert_eq!(keys, [t2[0].as_i64().unwrap()[0], t2[1].as_i64().unwrap()[0]], "the rotated move has the same POV keys");
        assert_eq!(decode_action(&red, RED, &spec, &fr.legal_ids, &keys).unwrap(), a);
        assert!(decode_action(&red, RED, &spec, &fr.legal_ids, &[0, 0]).is_err());
        let from_row = &fr.legal[1].as_bool().unwrap()[keys[0] as usize * 90..(keys[0] as usize + 1) * 90];
        let to_of: Vec<u8> = t2[3].as_f32().unwrap().iter().map(|&x| x as u8).collect();
        assert_eq!(from_row, &to_of[..]);
    }

    #[test]
    fn frames_carry_last_move_checks_and_repetitions() {
        let cfg = XiangqiConfig { fen: "3k5/9/9/9/9/9/9/9/R8/4K4 w - - 0 1".into() };
        let spec = Xiangqi::spec(&cfg);
        let mut ev = Vec::new();
        let mut g = Xiangqi::new_game(&cfg);
        for m in ["a2a10", "d10d9", "a10a9", "d9d10", "a9a10", "d10d9"] {
            let a = parse_uci(m).unwrap();
            ev.push(Event::Act(g.side_to_move(), a));
            g.play(a).unwrap();
        }
        let r = replay::<Xiangqi>(&cfg, &ev).unwrap();
        let glob = |d: usize| r.decisions[d].frame.obs[1].as_f32().unwrap().to_vec();
        // after a2a10+ (check) black is in check; the check run of the opponent is 1
        assert_eq!(glob(1)[3], 1.0);
        assert_eq!(glob(1)[5], 0.1);
        // red's third move (a10a9, decision 2) follows its own check: own run 1
        assert_eq!(glob(2)[6], 0.1);
        // decision 5: black to move in the position of decision 1 again → seen before
        assert_eq!(glob(5)[4], 1.0);
        assert_eq!(glob(3)[4], 0.0);
        // last move planes: decision 1 sees a2 → a10 from black's side
        let planes = r.decisions[1].frame.obs[0].as_f32().unwrap();
        let (from, to) = (parse_square("a2").unwrap(), parse_square("a10").unwrap());
        assert_eq!(planes[PLANE_LAST_FROM * 90 + pov(BLACK, from)], 1.0);
        assert_eq!(planes[PLANE_LAST_TO * 90 + pov(BLACK, to)], 1.0);
        assert_eq!(planes[PLANE_LAST_FROM * 90..PLANE_LAST_TO * 90].iter().sum::<f32>(), 1.0);
        assert_eq!(spec.max_decisions, MAX_PLIES);
    }

    #[test]
    fn conformance_battery() {
        let rep = check_game::<Xiangqi>(&XiangqiConfig::default(), 6, 1).unwrap();
        assert_eq!(rep.terminal, 6);
        // endgame starts: shorter games, every end rule reachable
        for (i, fen) in ["3k5/9/9/9/9/9/9/9/R8/4K4 w - - 0 1", "4k4/4a4/4b4/9/2p6/6P2/9/4B4/4A4/2C1K4 b - - 0 1",
                         "3akab2/9/4b4/4n4/2p1c1p2/9/2P1C1P2/4B4/4A4/2BAK4 w - - 0 1"].iter().enumerate() {
            let rep = check_game::<Xiangqi>(&XiangqiConfig { fen: fen.to_string() }, 6, i as u64).unwrap();
            assert_eq!(rep.terminal, 6);
        }
    }
}
