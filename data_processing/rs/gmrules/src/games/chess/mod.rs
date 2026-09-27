//! Standard chess — the gmrules game `chess`: FIDE rules by shakmaty (the move generator
//! the `gmdata chess` converter replays every Lichess record with) and the chess
//! converter's FROZEN frames (converter_version 3, the schema the running pretraining
//! trains on, model: the board adapter's `BoardEncoder` + `FromToDecoder`,
//! model/games/registry.py `_build_chess`).
//!
//! This module IS the converter's per-frame code: `gmdata chess convert` (gmdata
//! chess.rs) replays every record through [`crate::trajectory::replay`] of this game
//! (config [`ChessConfig::records`]) and writes the replay's obs / target tensors, and
//! `gmdata label chess` re-encodes shards with [`encode_position`] / [`legal_masks`] /
//! [`Repetitions`]; the simulator (gmsim, `env/vec_gmsim.py`) encodes its frames with
//! the same [`Game`] methods — an IL frame and the RL frame of the same position are the
//! same bytes (pinned: gmdata's `chess_sim_parity` tests, tests/test_chess_sim.py).
//!
//! Frames (interleaved: one stream per game, every frame the position BEFORE the move
//! from the mover's view — ranks mirrored for Black, files never; POV square
//! `row · 8 + file`, row = 7 − rank for White, rank for Black, see [`pov`]):
//!
//! ```text
//! obs     board_seq     [14, 8, 8] bits  own P N B R Q K, opponent P N B R Q K, the en passant
//!                                        square after EVERY double pawn push (shakmaty's
//!                                        EnPassantMode::Always: capturable or not), ones
//!         globals_seq   [10] f32         own K / Q castling right, opponent K / Q castling right,
//!                                        White to move, fullmove / 100, halfmove clock / 100
//!                                        (both computed in f64, clipped at 1, then f32), in
//!                                        check, the position occurred >= 1 / >= 2 times
//!                                        earlier in the game (FIDE identity, [`Repetitions`])
//! legal   from          [64] bool         squares with a legal move
//!         to_cond       [64, 64] bool     destinations per from-square
//!         promotion_cond [64, 64, 5] bool promotions per (from, to): 0 none, 1 N, 2 B, 3 R, 4 Q
//! target  from_idx / to_idx / promotion [] i64   the action keys (POV squares, promotion)
//!         legal_from          [64] bits  IL legality: from-squares of the legal moves (= legal.from)
//!         legal_to_of         [64] bits  destinations of the legal moves from the taken from-square
//!         legal_promotion_of  [5] bits   promotions of the legal moves with the taken (from, to)
//! ```
//!
//! Castling is labelled king → g / c file (UCI / python-chess standard notation), in the
//! keys and in every mask alike ([`move_target`]); shakmaty's own castling move is king →
//! rook. The shards (`gmdata chess convert`) store obs.board_seq, obs.globals_seq,
//! tgt.from_idx / to_idx / promotion / legal_from / legal_to_of / legal_promotion_of and
//! tgt.value (the mover's result); the legal tensors are the simulator's (the RL
//! decoder samples under them). Shard meta `{"global_features": 10}` = the spec's meta.
//!
//! Actions: the canonical id of a move is its UCI (standard castling e1g1 / e1c1 / e8g8 /
//! e8c8) on absolute squares, `(promotion · 64 + from) · 64 + to` (squares a1 = 0 … h8 =
//! 63, promotion 0 none, 1 N, 2 B, 3 R, 4 Q; [`action_id`], [`uci`]); seat 0 = White,
//! seat 1 = Black, alternating, no chance events. The action keys are the move's
//! [`move_target`], one-to-one with the legal moves of a position.
//!
//! Game ends (checked after every move, in this order; [`End`]):
//!
//! - checkmate (the mover's opponent wins) and stalemate (a draw) — the rules;
//! - insufficient material (a draw; shakmaty's `is_insufficient_material`: neither side
//!   can mate — K vs K, a lone minor piece, bishops all on one square colour — FIDE's
//!   dead position restricted to the material cases, as lichess rules it);
//! - the fifty-move rule: a draw as soon as the halfmove clock reaches 100 — AUTOMATIC
//!   here (FIDE: a claim at 50 moves, automatic at 75), and checkmate on the 100th
//!   halfmove still wins (FIDE 9.6: mate takes precedence);
//! - threefold repetition: a draw as soon as a position occurs for the third time — also
//!   AUTOMATIC (FIDE: a claim at three occurrences, automatic at five; the side that
//!   profits always claims, so for self-play the automatic draw is the same game);
//! - the ply cap `max_plies` (default 1024 = the spec's `max_decisions`): an adjudicated
//!   draw.
//!
//! Each automatic draw is a config switch ([`ChessConfig`]); the RL env keeps them on (the
//! defaults). Records continue past threefold repetitions and 50-move clocks (Lichess:
//! claims), so the converter replays with [`ChessConfig::records`] (only mate and
//! stalemate end a game, cap 100 000 plies) — its frames are the same function of the
//! position and history, only fewer games end. Hence globals 9 (seen twice before) and
//! a halfmove clock >= 100 occur in IL frames but never in the default env's frames.
//!
//! Validation: perft through this game's own action enumeration (start position,
//! Kiwipete and the standard positions 3-6), the invariance battery, parity with the
//! converter (byte-identical frames of real records, gmdata `chess.rs` tests) and the
//! python-chess cross-check of the converter (tests/test_chess_converter.py).

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{BufMut, TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, CastlingSide, Chess, Color, EnPassantMode, KnownOutcome, Move, Position, Role, Square};
use std::collections::HashMap;

/// The frame schema generation: `gmdata chess convert`'s converter_version (= [`Game::VERSION`]).
pub const CONVERTER_VERSION: u32 = 3;
/// Width of `obs.globals_seq` (shard meta `global_features`).
pub const GLOBAL_FEATURES: usize = 10;
/// Promotion classes (`promotion`, `legal_promotion_of`): 0 none, 1 N, 2 B, 3 R, 4 Q.
pub const N_PROMOTIONS: usize = 5;
/// `obs.board_seq` planes.
pub const PIECE_CHANNELS: usize = 14;
/// Floats of one `obs.board_seq` frame (14 planes of 8×8).
pub const PLANE_LEN: usize = PIECE_CHANNELS * 64;
/// Board squares (the from-to decoder's tokens).
pub const N_SQUARES: usize = 64;
/// Size of the action id space (`(promotion · 64 + from) · 64 + to`).
pub const N_ACTIONS: u32 = (N_PROMOTIONS * 64 * 64) as u32;
/// Default ply cap (an adjudicated draw; the spec's `max_decisions`).
pub const DEFAULT_MAX_PLIES: usize = 1024;
/// The ply cap of [`ChessConfig::records`] (and the largest `max_plies`).
pub const RECORD_MAX_PLIES: usize = 100_000;
/// Schema name (the simulator's spec; the converter's shard meta predates schema names).
pub const SCHEMA: &str = "chess.v3";

/// A move's (from, to, promotion) in the frame's POV coordinates — the action keys.
pub type Target = (i64, i64, i64);

/// POV square index `row·8 + file` of `sq` for the side to move (`flip` = Black to move:
/// row = rank; White to move: row = 7 − rank; ranks / files 0-based from a1).
pub fn pov(sq: Square, flip: bool) -> i64 {
    let (file, rank) = (u32::from(sq.file()) as i64, u32::from(sq.rank()) as i64);
    if flip {
        rank * 8 + file
    } else {
        (7 - rank) * 8 + file
    }
}

/// The inverse of [`pov`] (`p` in 0..64).
pub fn unpov(p: i64, flip: bool) -> Square {
    let (row, file) = ((p / 8) as u32, (p % 8) as u32);
    let rank = if flip { row } else { 7 - row };
    Square::new(rank * 8 + file)
}

fn role_idx(r: Role) -> usize {
    match r {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

/// Promotion class of a promotion piece (0 none, 1 N, 2 B, 3 R, 4 Q).
pub fn promo_idx(r: Option<Role>) -> i64 {
    match r {
        None => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        Some(_) => 0,
    }
}

const PROMOTION_CHARS: [&str; N_PROMOTIONS] = ["", "n", "b", "r", "q"];

/// The destination of a move as UCI (and the converter's labels) name it: castling =
/// the king's destination on the g / c file (shakmaty's `to` is the rook's square).
pub fn uci_to(m: &Move) -> Square {
    match *m {
        Move::Castle { king, rook } => {
            let side = if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide };
            Square::from_coords(side.king_to_file(), king.rank())
        }
        _ => m.to(),
    }
}

/// A move as the chess adapter labels it, in the frame's POV coordinates:
/// (from square, to square, promotion index). Castling → king destination file g/c
/// (python-chess standard mode; shakmaty encodes castling as king → rook). Used for the
/// labels AND the legality masks, so the GT move always lies inside its own masks.
pub fn move_target(m: &Move, flip: bool) -> Option<Target> {
    let from_sq = m.from()?;
    Some((pov(from_sq, flip), pov(uci_to(m), flip), promo_idx(m.promotion())))
}

/// The canonical action id of a move: its UCI on absolute squares,
/// `(promotion · 64 + from) · 64 + to` (see the module docs).
pub fn action_id(m: &Move) -> u32 {
    let from = m.from().expect("standard chess moves have a from-square");
    (promo_idx(m.promotion()) as u32 * 64 + u32::from(from)) * 64 + u32::from(uci_to(m))
}

/// An action id's (from square, to square, promotion class); None outside the id space.
pub fn action_parts(id: u32) -> Option<(Square, Square, u32)> {
    (id < N_ACTIONS).then(|| (Square::new((id / 64) % 64), Square::new(id % 64), id / 4096))
}

/// The action keys ([`move_target`]) of an action id for the mover (`flip` = Black to move).
pub fn action_keys(id: u32, flip: bool) -> Option<Target> {
    action_parts(id).map(|(f, t, p)| (pov(f, flip), pov(t, flip), p as i64))
}

/// The action id with action keys `keys` for the mover (None: keys out of range).
pub fn action_of_keys(keys: Target, flip: bool) -> Option<u32> {
    let (f, t, p) = keys;
    if !(0..64).contains(&f) || !(0..64).contains(&t) || !(0..N_PROMOTIONS as i64).contains(&p) {
        return None;
    }
    Some((p as u32 * 64 + u32::from(unpov(f, flip))) * 64 + u32::from(unpov(t, flip)))
}

/// The UCI of an action id (`e2e4`, `e7e8q`, castling `e1g1`).
pub fn uci(id: u32) -> String {
    match action_parts(id) {
        Some((f, t, p)) => format!("{f}{t}{}", PROMOTION_CHARS[p as usize]),
        None => format!("?{id}"),
    }
}

/// IL legality masks of one frame: (from-squares of the legal moves, destinations from
/// `gt.0`, promotions of (`gt.0`, `gt.1`)) given the legal moves' targets.
#[allow(clippy::type_complexity)]
fn masks_of(targets: impl Iterator<Item = Target>, gt: Target) -> Option<([f32; 64], [f32; 64], [f32; N_PROMOTIONS])> {
    let (mut from, mut to_of, mut promo_of) = ([0f32; 64], [0f32; 64], [0f32; N_PROMOTIONS]);
    for (f, t, p) in targets {
        from[f as usize] = 1.0;
        if f == gt.0 {
            to_of[t as usize] = 1.0;
            if t == gt.1 {
                promo_of[p as usize] = 1.0;
            }
        }
    }
    let inside = from[gt.0 as usize] == 1.0 && to_of[gt.1 as usize] == 1.0 && promo_of[gt.2 as usize] == 1.0;
    inside.then_some((from, to_of, promo_of))
}

/// Legality masks of one frame, conditioned on the GT move `gt` = (from, to, promotion):
/// (legal from-squares, legal destinations from `gt.from`, legal promotions of
/// (`gt.from`, `gt.to`)). `None` if `gt` is not inside all three (never for a legal move).
#[allow(clippy::type_complexity)]
pub fn legal_masks(pos: &Chess, flip: bool, gt: Target) -> Option<([f32; 64], [f32; 64], [f32; N_PROMOTIONS])> {
    masks_of(pos.legal_moves().iter().filter_map(|m| move_target(m, flip)), gt)
}

/// Occurrence counts of the positions of one game (Chess Eq/Hash = FIDE repetition
/// identity: board, side to move, castling rights, legal en-passant square).
#[derive(Clone, Debug, Default)]
pub struct Repetitions(HashMap<Chess, u32>);

impl Repetitions {
    /// Record `pos`; returns how often it occurred earlier in the game.
    pub fn see(&mut self, pos: &Chess) -> u32 {
        let n = self.0.entry(pos.clone()).or_insert(0);
        *n += 1;
        *n - 1
    }

    /// Forget every position (after an irreversible move: no earlier position can occur
    /// again, so the counts of every later position stay exact — see [`StandardChess`]).
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Positions remembered.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The observation of one frame (the position before the move, from the mover's POV)
/// into `planes` (`PLANE_LEN` floats, zero on entry) and `globals` (`GLOBAL_FEATURES`
/// floats); `prior` = earlier occurrences of the position ([`Repetitions::see`]).
pub fn encode_obs(pos: &Chess, prior: u32, planes: &mut [f32], globals: &mut [f32]) {
    let own = pos.turn();
    let flip = own == Color::Black;
    for (sq, piece) in pos.board().iter() {
        let plane = if piece.color == own { 0 } else { 6 } + role_idx(piece.role);
        planes[plane * 64 + pov(sq, flip) as usize] = 1.0;
    }
    if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
        planes[12 * 64 + pov(ep, flip) as usize] = 1.0;
    }
    for x in &mut planes[13 * 64..14 * 64] {
        *x = 1.0;
    }
    let c = pos.castles();
    globals.copy_from_slice(&[
        c.has(own, CastlingSide::KingSide) as u8 as f32,
        c.has(own, CastlingSide::QueenSide) as u8 as f32,
        c.has(!own, CastlingSide::KingSide) as u8 as f32,
        c.has(!own, CastlingSide::QueenSide) as u8 as f32,
        (own == Color::White) as u8 as f32,
        (u32::from(pos.fullmoves()) as f64 / 100.0).min(1.0) as f32,
        (pos.halfmoves() as f64 / 100.0).min(1.0) as f32,
        pos.is_check() as u8 as f32,
        (prior >= 1) as u8 as f32,
        (prior >= 2) as u8 as f32,
    ]);
}

/// [`encode_obs`] appending the `GLOBAL_FEATURES` globals to `globals` (the frame-major
/// column builders of `gmdata label chess`).
pub fn encode_position(pos: &Chess, prior: u32, planes: &mut [f32], globals: &mut Vec<f32>) {
    let n = globals.len();
    globals.resize(n + GLOBAL_FEATURES, 0.0);
    encode_obs(pos, prior, planes, &mut globals[n..]);
}

/// Why a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum End {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    /// halfmove clock 100 (config `fifty_move_draw`)
    FiftyMoves,
    /// a position's third occurrence (config `threefold_draw`)
    Threefold,
    /// `max_plies` plies played (an adjudicated draw)
    PlyCap,
}

/// Rule switches of the `chess` game (JSON config; the defaults are the RL env's).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChessConfig {
    /// A position's third occurrence ends the game, a draw (FIDE: a claim).
    pub threefold_draw: bool,
    /// Halfmove clock 100 (fifty moves each without a capture or pawn move) ends the game,
    /// a draw (FIDE: a claim); checkmate on that move still wins.
    pub fifty_move_draw: bool,
    /// Insufficient mating material on both sides ends the game, a draw.
    pub insufficient_material_draw: bool,
    /// Plies after which the game is adjudicated a draw (= the spec's `max_decisions`).
    pub max_plies: usize,
}

impl Default for ChessConfig {
    fn default() -> Self {
        ChessConfig { threefold_draw: true, fifty_move_draw: true, insufficient_material_draw: true, max_plies: DEFAULT_MAX_PLIES }
    }
}

impl ChessConfig {
    /// The converter's config: records continue past threefold repetitions, 50-move
    /// clocks and (old records) dead material, so only checkmate and stalemate end a game;
    /// cap [`RECORD_MAX_PLIES`].
    pub fn records() -> Self {
        ChessConfig { threefold_draw: false, fifty_move_draw: false, insufficient_material_draw: false, max_plies: RECORD_MAX_PLIES }
    }
}

/// Obs / legal / target tensor indices (see [`StandardChess::spec`]).
const LEGAL_FROM: usize = 0;
const LEGAL_TO_COND: usize = 1;
const LEGAL_PROMOTION_COND: usize = 2;
const TGT_FROM: usize = 0;
const TGT_TO: usize = 1;
const TGT_PROMOTION: usize = 2;
const TGT_LEGAL_FROM: usize = 3;
const TGT_LEGAL_TO_OF: usize = 4;
const TGT_LEGAL_PROMOTION_OF: usize = 5;

/// A standard chess game (see the module docs).
///
/// Repetitions: the occurrence counts of the positions since the last irreversible move
/// (shakmaty's `is_irreversible`: pawn moves, captures, castling, moves that lose a
/// castling right or an en passant chance). A position before an irreversible move never
/// occurs again — captures and promotions only ever lower the material, pawns only
/// advance, castling rights only disappear, an en passant chance exists only right after
/// its double push — so these counts equal the converter's whole-game counts
/// ([`Repetitions`] of every position so far; tested on random games).
#[derive(Clone, Debug)]
pub struct StandardChess {
    cfg: ChessConfig,
    pos: Chess,
    /// legal moves of the current position with their action ids, ascending (empty when over)
    legal: Vec<(u32, Move)>,
    reps: Repetitions,
    /// earlier occurrences of the current position
    prior: u32,
    plies: usize,
    over: Option<(KnownOutcome, End)>,
}

impl StandardChess {
    /// A game in progress from `pos` (it counts as the first position of the game).
    pub fn from_position(pos: Chess, cfg: &ChessConfig) -> Self {
        let mut reps = Repetitions::default();
        let prior = reps.see(&pos);
        let mut g = StandardChess { cfg: cfg.clone(), pos, legal: Vec::new(), reps, prior, plies: 0, over: None };
        g.refresh();
        g
    }

    /// [`from_position`](Self::from_position) of a FEN (standard castling).
    pub fn from_fen(fen: &str, cfg: &ChessConfig) -> Result<Self, String> {
        let setup = Fen::from_ascii(fen.as_bytes()).map_err(|e| format!("{fen}: {e}"))?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|e| format!("{fen}: {e}"))?;
        Ok(Self::from_position(pos, cfg))
    }

    pub fn position(&self) -> &Chess {
        &self.pos
    }

    pub fn config(&self) -> &ChessConfig {
        &self.cfg
    }

    /// Earlier occurrences of the current position (globals 8 / 9 say >= 1 / >= 2).
    pub fn prior_occurrences(&self) -> u32 {
        self.prior
    }

    /// Positions in the repetition memory (since the last irreversible move).
    pub fn remembered_positions(&self) -> usize {
        self.reps.len()
    }

    pub fn plies(&self) -> usize {
        self.plies
    }

    /// The game's result and why (None while it runs).
    pub fn end(&self) -> Option<(KnownOutcome, End)> {
        self.over
    }

    /// Legal moves with their action ids, ascending (empty when the game is over).
    pub fn legal_moves(&self) -> &[(u32, Move)] {
        &self.legal
    }

    /// The legal move of action id `id`.
    pub fn move_of(&self, id: u32) -> Option<Move> {
        self.legal.binary_search_by_key(&id, |&(a, _)| a).ok().map(|i| self.legal[i].1)
    }

    /// The position as a FEN.
    pub fn fen(&self) -> String {
        Fen::from_position(&self.pos, EnPassantMode::Legal).to_string()
    }

    fn mover_seat(&self) -> Seat {
        match self.pos.turn() {
            Color::White => 0,
            Color::Black => 1,
        }
    }

    fn flip(&self) -> bool {
        self.pos.turn() == Color::Black
    }

    /// Legal moves and the game status of the current position (see the module docs for
    /// the order of the ends).
    fn refresh(&mut self) {
        let moves = self.pos.legal_moves();
        let cfg = &self.cfg;
        self.over = if moves.is_empty() {
            Some(if self.pos.is_check() {
                (KnownOutcome::Decisive { winner: !self.pos.turn() }, End::Checkmate)
            } else {
                (KnownOutcome::Draw, End::Stalemate)
            })
        } else if cfg.insufficient_material_draw && self.pos.is_insufficient_material() {
            Some((KnownOutcome::Draw, End::InsufficientMaterial))
        } else if cfg.fifty_move_draw && self.pos.halfmoves() >= 100 {
            Some((KnownOutcome::Draw, End::FiftyMoves))
        } else if cfg.threefold_draw && self.prior >= 2 {
            Some((KnownOutcome::Draw, End::Threefold))
        } else if self.plies >= cfg.max_plies {
            Some((KnownOutcome::Draw, End::PlyCap))
        } else {
            None
        };
        self.legal.clear();
        if self.over.is_none() {
            self.legal.extend(moves.iter().map(|m| (action_id(m), *m)));
            self.legal.sort_unstable_by_key(|&(id, _)| id);
        }
    }
}

impl Game for StandardChess {
    type Config = ChessConfig;
    const NAME: &'static str = "chess";
    const VERSION: u32 = CONVERTER_VERSION;

    fn check_config(cfg: &ChessConfig) -> Result<(), RulesError> {
        if !(1..=RECORD_MAX_PLIES).contains(&cfg.max_plies) {
            rules_bail!("chess: max_plies must be 1..={RECORD_MAX_PLIES}, got {}", cfg.max_plies);
        }
        Ok(())
    }

    fn spec(cfg: &ChessConfig) -> FrameSpec {
        let n = N_SQUARES;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: cfg.max_plies,
            schema: SCHEMA.into(),
            obs: vec![TensorSpec::bits("board_seq", &[PIECE_CHANNELS, 8, 8]), TensorSpec::f32("globals_seq", &[GLOBAL_FEATURES])],
            legal: vec![
                TensorSpec::bool("from", &[n]),
                TensorSpec::bool("to_cond", &[n, n]),
                TensorSpec::bool("promotion_cond", &[n, n, N_PROMOTIONS]),
            ],
            target: vec![
                TensorSpec::i64("from_idx", &[]),
                TensorSpec::i64("to_idx", &[]),
                TensorSpec::i64("promotion", &[]),
                TensorSpec::bits("legal_from", &[n]),
                TensorSpec::bits("legal_to_of", &[n]),
                TensorSpec::bits("legal_promotion_of", &[N_PROMOTIONS]),
            ],
            action_keys: vec!["from_idx".into(), "to_idx".into(), "promotion".into()],
            // = the converter's shard meta (train_il_game META_ARCH_KEYS["chess"])
            meta: json!({"global_features": GLOBAL_FEATURES}),
            candidate: None,
        }
    }

    fn new_game(cfg: &ChessConfig) -> Self {
        Self::from_position(Chess::default(), cfg)
    }

    fn turn(&self) -> Turn {
        if self.over.is_some() {
            Turn::Terminal
        } else {
            Turn::Act(SeatSet::single(self.mover_seat()))
        }
    }

    fn chance_outcomes(&self, _out: &mut Vec<(u32, f64)>) {}

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        rules_bail!("chess: no chance events (got {outcome})")
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.over.is_none() && seat == self.mover_seat() {
            out.extend(self.legal.iter().map(|&(id, _)| id));
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.over.is_some() || seat != self.mover_seat() {
            rules_bail!("chess: seat {seat} is not to move");
        }
        let Some(m) = self.move_of(action) else {
            rules_bail!("chess: action {action} ({}) is illegal in {}", uci(action), self.fen());
        };
        if self.pos.is_irreversible(m) {
            self.reps.clear();
        }
        self.pos.play_unchecked(m);
        self.plies += 1;
        self.prior = self.reps.see(&self.pos);
        self.refresh();
        Ok(())
    }

    fn outcome(&self) -> Vec<f32> {
        match self.over {
            Some((KnownOutcome::Decisive { winner: Color::White }, _)) => vec![1.0, -1.0],
            Some((KnownOutcome::Decisive { winner: Color::Black }, _)) => vec![-1.0, 1.0],
            _ => vec![0.0, 0.0],
        }
    }

    fn returns(&self) -> Vec<f64> {
        self.outcome().iter().map(|&x| x as f64).collect()
    }

    fn encode_custom(&self, _seat: Seat, f: &mut FrameMut<'_>) {
        match f.obs.bufs_mut() {
            [BufMut::F32(planes), BufMut::F32(globals)] => encode_obs(&self.pos, self.prior, planes, globals),
            _ => unreachable!("chess obs layout"),
        }
        let flip = self.flip();
        let n = N_SQUARES;
        for (_, m) in &self.legal {
            let (fr, to, p) = move_target(m, flip).expect("a board move");
            let (fr, to, p) = (fr as usize, to as usize, p as usize);
            f.legal.bool(LEGAL_FROM)[fr] = 1;
            f.legal.bool(LEGAL_TO_COND)[fr * n + to] = 1;
            f.legal.bool(LEGAL_PROMOTION_COND)[(fr * n + to) * N_PROMOTIONS + p] = 1;
        }
    }

    fn target_custom(&self, _seat: Seat, _legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let flip = self.flip();
        let m = self.move_of(action).expect("encode_target passes a legal action");
        let gt = move_target(&m, flip).expect("a board move");
        t.i64(TGT_FROM)[0] = gt.0;
        t.i64(TGT_TO)[0] = gt.1;
        t.i64(TGT_PROMOTION)[0] = gt.2;
        let (lf, lt, lp) = masks_of(self.legal.iter().filter_map(|(_, m)| move_target(m, flip)), gt)
            .expect("the taken move lies inside its own legality masks");
        t.f32(TGT_LEGAL_FROM).copy_from_slice(&lf);
        t.f32(TGT_LEGAL_TO_OF).copy_from_slice(&lt);
        t.f32(TGT_LEGAL_PROMOTION_OF).copy_from_slice(&lp);
    }

    fn decode_custom(&self, _seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let id = action_of_keys((keys[0], keys[1], keys[2]), self.flip())?;
        legal.binary_search(&id).is_ok().then_some(id)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        uci(action)
    }

    fn state_string(&self) -> String {
        format!("{} | ply {} seen before {} | {:?}", self.fen(), self.plies, self.prior, self.over)
    }
}

#[cfg(test)]
mod tests;
