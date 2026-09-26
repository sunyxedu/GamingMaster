//! Chess variants as played on Lichess — chess960, crazyhouse, atomic, antichess,
//! three-check, king of the hill, horde and racing kings — as ONE gmrules game
//! (`chess_variants`): one frame schema, one network (a variant one-hot tells it which
//! rules apply), one mixed-variant corpus (the shard meta does not depend on the
//! variant), one RL env that draws the variant per episode. Standard chess stays the
//! separate `chess` game (gmdata chess.rs); chess960 from Scharnagl position 518 IS
//! standard chess with king → rook castling.
//!
//! Rules: shakmaty's variant positions (`shakmaty::variant`, the move generator of
//! lichess' analysis stack) for moves, captures, explosions, drops, promotions,
//! checks and the variant ends; on top, lichess' AUTOMATIC game ends as scalachess
//! (lichess' rules library) defines them — `Position.status`: checkmate > variant end
//! > stalemate > automatic draw ([`rules_outcome`]):
//!
//! - racing kings: after White reaches the 8th rank Black may still draw by reaching
//!   it with a LEGAL move (scalachess `specialEnd`; shakmaty only asks for an
//!   unattacked square);
//! - insufficient material / dead positions, per variant ([`lichess_insufficient_material`],
//!   ports of scalachess `isInsufficientMaterial`): chess960 the standard rule, three-check
//!   bare kings, atomic its winning-material table + closed pawn structures (kings,
//!   blocked pawns, "pawnitized" bishops), antichess opposite-colour bishops with
//!   unattackable blocked pawns, horde a stalemated horde (after every black move), never
//!   in crazyhouse / king of the hill / racing kings;
//! - fifty-move rule: a draw once the halfmove clock reaches 100 (automatic on lichess,
//!   not a claim), in every variant but crazyhouse (which has none);
//! - fivefold repetition: a draw;
//! - threefold repetition is a CLAIM on lichess (auto-claimed by a user preference),
//!   so records continue past it: a draw only with `auto_threefold` (the RL env's
//!   setting, model/games/gmrules/chess_variants.py; the converter's is off — in rational
//!   play the side that profits claims, so an automatic draw is the same game);
//! - `max_plies` (default 1024): a draw — the one deviation from lichess, a hard bound
//!   the framework needs (`max_decisions`); longer real games are rejected anyway.
//!
//! Games that end OFF the rules (resignation, time forfeit, agreed / claimed draws)
//! stop before a terminal state; the converter takes their result from the PGN
//! (`traj::Record::final_outcome`). Parity: every lichess record replays (the gmdata
//! converter verifies every move and every rules end against the PGN result; 0
//! illegal / 0 mismatches on 333k games of 2026-08); differential tests against
//! Fairy-Stockfish (tests/test_chess_variants_reference.py) on random playouts both
//! ways and on real lichess games.
//!
//! Trajectory: `Chance(variant)` (the configured variants, `weights`), for chess960
//! `Chance(start position)` (Scharnagl 0..960, uniform over `chess960_positions`, default
//! all), then the moves — seat 0 =
//! White (the horde), seat 1 = Black, alternating. Action ids (absolute squares, the
//! UCI of Fairy-Stockfish with `UCI_Chess960`):
//!
//! ```text
//! board move   ((promotion · 64) + from) · 64 + to       promotion 0 none, 1 N, 2 B, 3 R, 4 Q, 5 K
//!              castling = king square → own rook square  (unambiguous in chess960)
//! drop         24576 + role · 64 + to                    role 0 P, 1 N, 2 B, 3 R, 4 Q  (crazyhouse)
//! ```
//!
//! Frames (`chessvar.v1`, interleaved: one stream per game, every frame from the
//! mover's view — ranks mirrored for Black, files never, the chess converter's
//! convention; POV square = row · 8 + file, row = 7 − rank for White, rank for Black):
//!
//! ```text
//! obs     board_seq    [17, 8, 8] bits  own P N B R Q K, opponent P N B R Q K, legal en passant
//!                                       square, rooks with castling rights, promoted pieces
//!                                       (crazyhouse), goal squares (king of the hill: centre;
//!                                       racing kings: 8th rank), ones
//!         hand_seq     [10, 8] bits     pocket slots own P N B R Q, opponent P N B R Q
//!                                       (crazyhouse; zeros elsewhere): count >= 1 .. >= 8
//!         globals_seq  [24] f32         variant one-hot (8: chess960, crazyhouse, atomic, antichess,
//!                                       threecheck, kingofthehill, horde, racingkings), mover is
//!                                       White, in check, position seen >= 1/2/3/4 times before,
//!                                       checks given by mover >= 1/2, by opponent >= 1/2
//!                                       (three-check), castling rights own K/Q, opponent K/Q,
//!                                       halfmove clock / 100, fullmove / 100 (both clipped at 1)
//! legal   from            [74] bool          tokens with a legal move: 64 board squares, then the
//!                                            pocket slots (a drop starts at the mover's pocket
//!                                            slot 64 + role)
//!         to_cond         [74, 74] bool      destinations per from-token
//!         promotion_cond  [74, 74, 6] bool   promotions per (from, to): 0 none, 1 N, 2 B, 3 R, 4 Q, 5 K
//! target  from_idx / to_idx / promotion  [] i64   the action keys (POV tokens, promotion class)
//!         legal_from      [74] bits      IL legality (= legal.from)
//!         legal_to_of     [74] bits      destinations of the taken move's from-token
//!         legal_promotion_of [6] bits    promotions of the taken (from, to)
//! ```
//!
//! The model is the board adapter's from-to decoder over the 74 tokens (board squares +
//! pocket slots: a drop is "from the pocket slot to the square") — see
//! model/games/chess_variants.py; RL: train_rl_joint's FromToSpec.

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;
/// The rules library (re-exported: the gmdata converter parses SAN with the same crate).
pub use shakmaty;
use shakmaty::fen::Fen;
use shakmaty::variant::{Variant as ShakmatyVariant, VariantPosition};
use shakmaty::zobrist::Zobrist64;
use shakmaty::{
    attacks, Bitboard, Board, CastlingMode, CastlingSide, Color, EnPassantMode, KnownOutcome, Move, MoveList, Outcome,
    Position, Rank, Role, Setup, Square,
};

/// Board squares.
pub const N_SQUARES: usize = 64;
/// Droppable roles per side (P N B R Q).
pub const N_POCKET: usize = 5;
/// Decoder tokens: 64 board squares + own and opponent pocket slots.
pub const N_TOKENS: usize = N_SQUARES + 2 * N_POCKET;
/// Promotion classes: none, N, B, R, Q, K (antichess).
pub const N_PROMOTIONS: usize = 6;
/// `board_seq` planes.
pub const PIECE_CHANNELS: usize = 17;
/// Features per pocket slot (count >= 1 .. >= 8).
pub const HAND_FEATURES: usize = 8;
/// `globals_seq` width.
pub const GLOBAL_FEATURES: usize = 24;
/// First drop action id.
pub const DROP_BASE: u32 = (N_PROMOTIONS * 4096) as u32;
/// Chess960 start positions (Scharnagl numbering; 518 = the standard position).
pub const N_START_960: u32 = 960;
/// Default ply cap (a draw; the spec's `max_decisions`).
pub const DEFAULT_MAX_PLIES: usize = 1024;

/// Schema name (shard meta `schema`).
pub const SCHEMA: &str = "chessvar.v1";

/// A lichess variant (the one-hot / chance-outcome order is fixed forever).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChessVariant {
    Chess960 = 0,
    Crazyhouse = 1,
    Atomic = 2,
    Antichess = 3,
    ThreeCheck = 4,
    KingOfTheHill = 5,
    Horde = 6,
    RacingKings = 7,
}

impl ChessVariant {
    pub const ALL: [ChessVariant; 8] = [
        ChessVariant::Chess960,
        ChessVariant::Crazyhouse,
        ChessVariant::Atomic,
        ChessVariant::Antichess,
        ChessVariant::ThreeCheck,
        ChessVariant::KingOfTheHill,
        ChessVariant::Horde,
        ChessVariant::RacingKings,
    ];

    /// Canonical name (configs, shard meta, the one-hot order).
    pub const fn name(self) -> &'static str {
        match self {
            ChessVariant::Chess960 => "chess960",
            ChessVariant::Crazyhouse => "crazyhouse",
            ChessVariant::Atomic => "atomic",
            ChessVariant::Antichess => "antichess",
            ChessVariant::ThreeCheck => "threecheck",
            ChessVariant::KingOfTheHill => "kingofthehill",
            ChessVariant::Horde => "horde",
            ChessVariant::RacingKings => "racingkings",
        }
    }

    /// The name of the monthly Lichess dump directory
    /// (`https://database.lichess.org/<dir>/lichess_db_<dir>_rated_<YYYY-MM>.pgn.zst`).
    pub const fn lichess_dir(self) -> &'static str {
        match self {
            ChessVariant::Chess960 => "chess960",
            ChessVariant::Crazyhouse => "crazyhouse",
            ChessVariant::Atomic => "atomic",
            ChessVariant::Antichess => "antichess",
            ChessVariant::ThreeCheck => "threeCheck",
            ChessVariant::KingOfTheHill => "kingOfTheHill",
            ChessVariant::Horde => "horde",
            ChessVariant::RacingKings => "racingKings",
        }
    }

    /// Fairy-Stockfish variant name (the differential reference).
    pub const fn fairy_stockfish(self) -> &'static str {
        match self {
            ChessVariant::Chess960 => "chess",
            ChessVariant::Crazyhouse => "crazyhouse",
            ChessVariant::Atomic => "atomic",
            ChessVariant::Antichess => "antichess",
            ChessVariant::ThreeCheck => "3check",
            ChessVariant::KingOfTheHill => "kingofthehill",
            ChessVariant::Horde => "horde",
            ChessVariant::RacingKings => "racingkings",
        }
    }

    /// Our names, the lichess PGN `Variant` tags / dump directories and the
    /// Fairy-Stockfish names (case-insensitive, ignoring spaces, '-' and '_').
    pub fn from_name(s: &str) -> Option<ChessVariant> {
        let k: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        Some(match k.as_str() {
            "chess960" | "fischerandom" | "fischerrandom" => ChessVariant::Chess960,
            "crazyhouse" | "zh" => ChessVariant::Crazyhouse,
            "atomic" => ChessVariant::Atomic,
            "antichess" => ChessVariant::Antichess,
            "threecheck" | "3check" => ChessVariant::ThreeCheck,
            "kingofthehill" | "koth" => ChessVariant::KingOfTheHill,
            "horde" => ChessVariant::Horde,
            "racingkings" => ChessVariant::RacingKings,
            _ => return None,
        })
    }

    pub fn from_index(i: u32) -> Option<ChessVariant> {
        Self::ALL.get(i as usize).copied()
    }

    pub const fn index(self) -> u32 {
        self as u32
    }

    /// shakmaty's variant (chess960 = standard chess rules, chess960 castling mode).
    pub const fn shakmaty(self) -> ShakmatyVariant {
        match self {
            ChessVariant::Chess960 => ShakmatyVariant::Chess,
            ChessVariant::Crazyhouse => ShakmatyVariant::Crazyhouse,
            ChessVariant::Atomic => ShakmatyVariant::Atomic,
            ChessVariant::Antichess => ShakmatyVariant::Antichess,
            ChessVariant::ThreeCheck => ShakmatyVariant::ThreeCheck,
            ChessVariant::KingOfTheHill => ShakmatyVariant::KingOfTheHill,
            ChessVariant::Horde => ShakmatyVariant::Horde,
            ChessVariant::RacingKings => ShakmatyVariant::RacingKings,
        }
    }

    /// The fifty-move rule ends the game automatically (lichess: every variant but
    /// crazyhouse).
    pub const fn fifty_move_rule(self) -> bool {
        !matches!(self, ChessVariant::Crazyhouse)
    }
}

/// The start position of `variant` (`sp` = the Scharnagl number for chess960, ignored
/// otherwise).
pub fn start_position(variant: ChessVariant, sp: u32) -> VariantPosition {
    match variant {
        ChessVariant::Chess960 => {
            let board = Board::chess960(sp % N_START_960);
            let setup = Setup { castling_rights: board.rooks(), board, ..Setup::initial() };
            VariantPosition::from_setup(ShakmatyVariant::Chess, setup, CastlingMode::Chess960)
                .unwrap_or_else(|_| panic!("chess960 start position {sp} is valid"))
        }
        v => VariantPosition::new(v.shakmaty()),
    }
}

/// The Scharnagl number of a chess960 start board (None: not a start position).
pub fn scharnagl_of(board: &Board) -> Option<u32> {
    (0..N_START_960).find(|&sp| &Board::chess960(sp) == board)
}

fn role_index(r: Role) -> usize {
    match r {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

const POCKET_ROLES: [Role; N_POCKET] = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];

fn promotion_class(p: Option<Role>) -> u32 {
    match p {
        None | Some(Role::Pawn) => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        Some(Role::King) => 5,
    }
}

const PROMOTION_ROLES: [Option<Role>; N_PROMOTIONS] =
    [None, Some(Role::Knight), Some(Role::Bishop), Some(Role::Rook), Some(Role::Queen), Some(Role::King)];

/// The canonical action id of a move (see the module docs).
pub fn action_id(m: &Move) -> u32 {
    match *m {
        Move::Normal { from, to, promotion, .. } => (promotion_class(promotion) * 64 + u32::from(from)) * 64 + u32::from(to),
        Move::EnPassant { from, to } => u32::from(from) * 64 + u32::from(to),
        Move::Castle { king, rook } => u32::from(king) * 64 + u32::from(rook),
        Move::Put { role, to } => DROP_BASE + role_index(role) as u32 * 64 + u32::from(to),
    }
}

/// An action id's parts: (from square, or the dropped role), to square, promotion class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionParts {
    pub from: Option<Square>,
    pub drop: Option<Role>,
    pub to: Square,
    pub promotion: u32,
}

/// Decode an action id (None: outside the id space).
pub fn action_parts(id: u32) -> Option<ActionParts> {
    if id < DROP_BASE {
        let to = Square::new(id % 64);
        let from = Square::new((id / 64) % 64);
        Some(ActionParts { from: Some(from), drop: None, to, promotion: id / 4096 })
    } else if id < DROP_BASE + (N_POCKET * 64) as u32 {
        let k = id - DROP_BASE;
        Some(ActionParts { from: None, drop: Some(POCKET_ROLES[(k / 64) as usize]), to: Square::new(k % 64), promotion: 0 })
    } else {
        None
    }
}

/// The action's UCI (Fairy-Stockfish with `UCI_Chess960`: castling king → rook, drops
/// `P@e4`, promotions `e7e8q` / `e7e8k`).
pub fn uci(id: u32) -> String {
    match action_parts(id) {
        Some(ActionParts { drop: Some(r), to, .. }) => format!("{}@{}", r.upper_char(), to),
        Some(ActionParts { from: Some(f), to, promotion, .. }) => match PROMOTION_ROLES[promotion as usize] {
            Some(r) => format!("{f}{to}{}", r.char()),
            None => format!("{f}{to}"),
        },
        _ => format!("?{id}"),
    }
}

/// POV square index of `sq` for the mover (`flip` = Black to move: row = rank; White:
/// row = 7 − rank; col = file) — the chess converter's convention.
pub fn pov(sq: Square, flip: bool) -> usize {
    let (file, rank) = (sq.file().to_usize(), sq.rank().to_usize());
    if flip { rank * 8 + file } else { (7 - rank) * 8 + file }
}

/// The action keys (from token, to token, promotion class) of an action id for the
/// mover (`flip` = Black to move). Drops start at the mover's pocket slot 64 + role.
pub fn action_keys(id: u32, flip: bool) -> (usize, usize, usize) {
    match action_parts(id) {
        Some(ActionParts { drop: Some(r), to, .. }) => (N_SQUARES + role_index(r), pov(to, flip), 0),
        Some(ActionParts { from: Some(f), to, promotion, .. }) => (pov(f, flip), pov(to, flip), promotion as usize),
        _ => (usize::MAX, usize::MAX, usize::MAX),
    }
}

/// Why a game is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum End {
    /// checkmate, stalemate, a variant win / draw (explosion, hill, third check, race,
    /// horde captured, antichess)
    Rules,
    /// lichess' insufficient-material / dead-position draw of the variant
    /// ([`lichess_insufficient_material`])
    InsufficientMaterial,
    FiftyMoves,
    Fivefold,
    Threefold,
    PlyCap,
}

// ---- lichess' automatic game ends ------------------------------------------------
//
// Ports of scalachess (lichess' rules library, `chess.variant.*`): shakmaty generates
// the moves, these decide the automatic ends where the two libraries differ — the
// per-variant insufficient-material / dead-position draws and the racing-kings
// catch-up rule. The citations name the scalachess definitions.

fn bishops_on_opposite_colors(b: &Board) -> bool {
    (b.bishops() & Bitboard::LIGHT_SQUARES).any() && (b.bishops() & Bitboard::DARK_SQUARES).any()
}

/// `InsufficientMatingMaterial(board)`: only kings and minor pieces, and at most one
/// minor or only bishops on one square colour (standard chess / chess960).
fn standard_insufficient(b: &Board) -> bool {
    let occ = b.occupied();
    occ == b.kings() | b.knights() | b.bishops()
        && (occ.count() <= 3 || (occ == b.kings() | b.bishops() && !bishops_on_opposite_colors(b)))
}

/// `Atomic.insufficientAtomicWinningMaterial`.
fn atomic_insufficient_material(b: &Board) -> bool {
    let occ = b.occupied();
    let n = occ.count();
    if b.white().count() >= 2 && b.black().count() >= 2 {
        // opposite-colour bishops (one each) can never explode each other
        occ == b.kings() | b.bishops() && n <= 4 && bishops_on_opposite_colors(b)
    } else if occ == b.kings() | b.knights() {
        n <= 4
    } else {
        occ == b.kings() | b.rooks() | b.knights() | b.bishops() && !bishops_on_opposite_colors(b) && n <= 3
    }
}

/// `InsufficientMatingMaterial.pawnBlockedByPawn`: a pawn whose square ahead holds a
/// pawn and that has no legal move (`has_move(square, color)`: its only candidates
/// are captures, the square ahead being taken).
fn pawn_blocked_by_pawn(b: &Board, sq: Square, has_move: &impl Fn(Square, Color) -> bool) -> bool {
    let Some(p) = b.piece_at(sq).filter(|p| p.role == Role::Pawn) else {
        return false;
    };
    let ahead = sq.offset(if p.color == Color::White { 8 } else { -8 });
    ahead.is_some_and(|a| b.pawns().contains(a)) && !has_move(sq, p.color)
}

/// `Atomic.kingSafety` of the capture `from` × `to` by `color` (explosions included):
/// never explode the own king, always allowed to explode the opponent's, else the own
/// king must not be attacked afterwards (kings next to each other cannot check).
fn atomic_capture_is_legal(b: &Board, color: Color, from: Square, to: Square) -> bool {
    let blast = attacks::king_attacks(to);
    if b.king_of(color).is_some_and(|k| blast.contains(k)) {
        return false;
    }
    if b.king_of(!color).is_some_and(|k| blast.contains(k)) {
        return true;
    }
    let mut after = b.clone();
    after.discard_piece_at(from);
    after.discard_piece_at(to); // the victim; the capturer explodes on `to`
    for s in blast & after.occupied() & !after.pawns() {
        after.discard_piece_at(s);
    }
    let Some(k) = after.king_of(color) else { return true };
    if (attacks::king_attacks(k) & after.kings() & after.by_color(!color)).any() {
        return true;
    }
    let occ = after.occupied();
    let attackers = (attacks::rook_attacks(k, occ) & (after.rooks() | after.queens()))
        | (attacks::bishop_attacks(k, occ) & (after.bishops() | after.queens()))
        | (attacks::knight_attacks(k) & after.knights())
        | (attacks::pawn_attacks(color, k) & after.pawns());
    (attackers & after.by_color(!color)).is_empty()
}

/// `Atomic.atomicClosedPosition`: every piece is a king, a bishop or a pawn blocked by
/// a pawn, and the bishops (if any) are "pawnitized" — all of one side and one square
/// colour, every enemy pawn on the other colour.
fn atomic_closed_position(pos: &VariantPosition, moves: &MoveList) -> bool {
    let b = pos.board();
    let turn = pos.turn();
    // the side to move: its legal moves; the other side (scalachess `withColor`, no en
    // passant): its pawns' captures that pass the king-safety rule
    let has_move = |sq: Square, color: Color| {
        if color == turn {
            moves.iter().any(|m| m.from() == Some(sq))
        } else {
            (attacks::pawn_attacks(color, sq) & b.by_color(!color)).into_iter().any(|t| atomic_capture_is_legal(b, color, sq, t))
        }
    };
    let closed = b.iter().all(|(sq, p)| matches!(p.role, Role::King | Role::Bishop) || pawn_blocked_by_pawn(b, sq, &has_move));
    if !closed {
        return false;
    }
    match b.bishops().first() {
        None => true,
        Some(bs) => {
            let side = b.color_at(bs).expect("bishop");
            let light = bs.is_light();
            b.iter().all(|(sq, p)| match p.role {
                Role::Pawn => p.color == side || sq.is_light() != light,
                Role::Bishop => p.color == side && sq.is_light() == light,
                Role::King => true,
                _ => false,
            })
        }
    }
}

/// `Antichess.isInsufficientMaterial`: only bishops (each side's on one square colour,
/// the two sides' on different colours) and pawns blocked by pawns that can never
/// attack or be attacked by the opposing bishops.
fn antichess_insufficient_material(pos: &VariantPosition, moves: &MoveList) -> bool {
    let b = pos.board();
    if b.bishops() | b.pawns() != b.occupied() {
        return false;
    }
    let one_colour = |x: Bitboard| x.any() && ((x & Bitboard::LIGHT_SQUARES).is_empty() || (x & Bitboard::DARK_SQUARES).is_empty());
    let (wb, bb) = (b.bishops() & b.white(), b.bishops() & b.black());
    if !one_colour(wb) || !one_colour(bb) {
        return false;
    }
    let (w_light, b_light) = ((wb & Bitboard::LIGHT_SQUARES).any(), (bb & Bitboard::LIGHT_SQUARES).any());
    if w_light == b_light {
        return false;
    }
    let turn = pos.turn();
    // antichess has no king safety: the other side's pawn moves are its captures
    let has_move = |sq: Square, color: Color| {
        if color == turn {
            moves.iter().any(|m| m.from() == Some(sq))
        } else {
            (attacks::pawn_attacks(color, sq) & b.by_color(!color)).any()
        }
    };
    let safe = |color: Color, opposing_light: bool| {
        (b.pawns() & b.by_color(color)).into_iter().all(|p| pawn_blocked_by_pawn(b, p, &has_move) && p.is_light() != opposing_light)
    };
    safe(Color::White, b_light) && safe(Color::Black, w_light)
}

/// `Horde.hordeClosedPosition`: the horde is (or, after every black move, will be)
/// stalemated — unless its last piece can be captured at once.
fn horde_closed_position(pos: &VariantPosition, moves: &MoveList) -> bool {
    let b = pos.board();
    let horde = b.white();
    if horde.count() == 1 && horde.first().is_some_and(|s| b.attacks_to(s, Color::Black, b.occupied()).any()) {
        return false;
    }
    if pos.turn() == Color::White {
        moves.is_empty()
    } else {
        moves.iter().all(|m| {
            let mut after = pos.clone();
            after.play_unchecked(*m);
            after.legal_moves().is_empty()
        })
    }
}

/// Lichess' automatic draw by insufficient material / dead position of `variant`
/// (scalachess `Variant.isInsufficientMaterial`), given the side to move's legal
/// `moves`: standard rules in chess960, kings only in three-check, the atomic /
/// antichess / horde rules above, never in crazyhouse, king of the hill and racing
/// kings.
pub fn lichess_insufficient_material(variant: ChessVariant, pos: &VariantPosition, moves: &MoveList) -> bool {
    let b = pos.board();
    match variant {
        ChessVariant::Chess960 => standard_insufficient(b),
        ChessVariant::ThreeCheck => b.occupied() == b.kings(),
        ChessVariant::Atomic => atomic_insufficient_material(b) || atomic_closed_position(pos, moves),
        ChessVariant::Antichess => antichess_insufficient_material(pos, moves),
        ChessVariant::Horde => horde_closed_position(pos, moves),
        ChessVariant::Crazyhouse | ChessVariant::KingOfTheHill | ChessVariant::RacingKings => false,
    }
}

/// Racing kings' end (scalachess `RacingKings.specialEnd` / `specialDraw`): White to
/// move — exactly one king on the 8th rank wins, both draw; Black to move after White
/// reached it — White wins unless a LEGAL black move reaches the 8th rank too.
fn racing_kings_end(pos: &VariantPosition, moves: &MoveList) -> Option<KnownOutcome> {
    let b = pos.board();
    let goal = Bitboard::from_rank(Rank::Eighth);
    let reached = |c: Color| (b.kings() & b.by_color(c) & goal).any();
    let (w, bl) = (reached(Color::White), reached(Color::Black));
    match pos.turn() {
        Color::White if w && bl => Some(KnownOutcome::Draw),
        Color::White if w || bl => Some(KnownOutcome::Decisive { winner: if w { Color::White } else { Color::Black } }),
        Color::White => None,
        Color::Black => {
            let king = b.king_of(Color::Black);
            let reaches_goal = |m: &Move| match m {
                Move::Normal { role: Role::King, to, .. } => goal.contains(*to),
                _ => king.is_some_and(|k| goal.contains(k)),
            };
            (w && !moves.iter().any(reaches_goal)).then_some(KnownOutcome::Decisive { winner: Color::White })
        }
    }
}

/// The rules' result of `pos` (the side to move's legal `moves`) in lichess' order
/// (scalachess `Position.status`: checkmate, variant end, stalemate, automatic
/// draws); the automatic draws that depend on the history (fifty moves, repetitions)
/// are the caller's.
pub fn rules_outcome(variant: ChessVariant, pos: &VariantPosition, moves: &MoveList) -> Option<(KnownOutcome, End)> {
    let end = match variant {
        ChessVariant::RacingKings => racing_kings_end(pos, moves),
        _ => match pos.variant_outcome() {
            Outcome::Known(o) => Some(o),
            Outcome::Unknown => None,
        },
    };
    if let Some(o) = end {
        return Some((o, End::Rules));
    }
    if moves.is_empty() {
        // checkmate or stalemate (antichess / racing kings: no checks exist)
        let o = if pos.is_check() { KnownOutcome::Decisive { winner: !pos.turn() } } else { KnownOutcome::Draw };
        return Some((o, End::Rules));
    }
    lichess_insufficient_material(variant, pos, moves).then_some((KnownOutcome::Draw, End::InsufficientMaterial))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChessVariantsConfig {
    /// Variants the first chance event draws from (canonical names, see
    /// [`ChessVariant::name`]); converters keep the default (all) so that shards of
    /// every variant share one shard meta.
    pub variants: Vec<String>,
    /// Draw weights of `variants` (empty = uniform).
    pub weights: Vec<f64>,
    /// The chess960 start positions the second chance event draws from, uniformly
    /// (Scharnagl numbers; empty = all 960) — RL curricula, replaying a given game.
    pub chess960_positions: Vec<u32>,
    /// Threefold repetition ends the game (a draw). Lichess: a claim, auto-claimed by a
    /// user preference — records may continue past it, so converters keep it off.
    pub auto_threefold: bool,
    /// Hard ply cap (a draw) = the spec's `max_decisions`.
    pub max_plies: usize,
}

impl Default for ChessVariantsConfig {
    fn default() -> Self {
        ChessVariantsConfig {
            variants: ChessVariant::ALL.iter().map(|v| v.name().to_string()).collect(),
            weights: Vec::new(),
            chess960_positions: Vec::new(),
            auto_threefold: false,
            max_plies: DEFAULT_MAX_PLIES,
        }
    }
}

impl ChessVariantsConfig {
    /// The variant chance outcomes `(variant index, probability)`, ascending.
    fn choices(&self) -> Vec<(u32, f64)> {
        let mut out: Vec<(u32, f64)> = self
            .variants
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                let w = if self.weights.is_empty() { 1.0 } else { self.weights[i] };
                ChessVariant::from_name(n).map(|v| (v.index(), w))
            })
            .collect();
        let total: f64 = out.iter().map(|&(_, w)| w).sum();
        out.iter_mut().for_each(|(_, w)| *w /= total);
        out.sort_by_key(|&(i, _)| i);
        out
    }
}

/// Obs / legal / target tensor indices (see [`ChessVariants::spec`]).
const OBS_BOARD: usize = 0;
const OBS_HAND: usize = 1;
const OBS_GLOBALS: usize = 2;
const LEGAL_FROM: usize = 0;
const LEGAL_TO_COND: usize = 1;
const LEGAL_PROMOTION_COND: usize = 2;
const TGT_FROM: usize = 0;
const TGT_TO: usize = 1;
const TGT_PROMOTION: usize = 2;
const TGT_LEGAL_FROM: usize = 3;
const TGT_LEGAL_TO_OF: usize = 4;
const TGT_LEGAL_PROMOTION_OF: usize = 5;

/// `board_seq` plane indices beyond the 12 piece planes.
const PLANE_EP: usize = 12;
const PLANE_CASTLING_ROOKS: usize = 13;
const PLANE_PROMOTED: usize = 14;
const PLANE_GOAL: usize = 15;
const PLANE_ONES: usize = 16;

#[derive(Clone, Debug)]
pub struct ChessVariants {
    /// variant chance outcomes (index, probability)
    choices: Vec<(u32, f64)>,
    /// chess960 start positions (empty = all 960)
    starts: Vec<u32>,
    auto_threefold: bool,
    max_plies: usize,
    variant: Option<ChessVariant>,
    /// chess960 chosen, start position not yet drawn
    pending_start: bool,
    pos: VariantPosition,
    /// legal moves of the current position, ascending ids (empty unless to move)
    legal: Vec<(u32, Move)>,
    /// positions since the last irreversible move, the current one included (repetitions)
    window: Vec<(u64, VariantPosition)>,
    /// earlier occurrences of the current position
    prior: u32,
    plies: usize,
    over: Option<(KnownOutcome, End)>,
}

impl ChessVariants {
    pub fn variant(&self) -> Option<ChessVariant> {
        self.variant
    }

    pub fn position(&self) -> &VariantPosition {
        &self.pos
    }

    /// Earlier occurrences of the current position (repetition count − 1).
    pub fn prior_occurrences(&self) -> u32 {
        self.prior
    }

    pub fn plies(&self) -> usize {
        self.plies
    }

    /// The game's result and why (None while it runs).
    pub fn end(&self) -> Option<(KnownOutcome, End)> {
        self.over
    }

    /// Legal moves with their action ids, ascending.
    pub fn legal_moves(&self) -> &[(u32, Move)] {
        &self.legal
    }

    fn started(&self) -> bool {
        self.variant.is_some() && !self.pending_start
    }

    fn start(&mut self, variant: ChessVariant, sp: u32) {
        self.variant = Some(variant);
        self.pending_start = false;
        self.pos = start_position(variant, sp);
        self.window = vec![(hash(&self.pos), self.pos.clone())];
        self.prior = 0;
        self.refresh();
    }

    /// Legal moves and the game status of the current position (lichess' order:
    /// the rules' outcome — mate / variant end / stalemate / insufficient material,
    /// [`rules_outcome`] — then the automatic draws of the history).
    fn refresh(&mut self) {
        let variant = self.variant.expect("started");
        let moves = self.pos.legal_moves();
        self.over = match rules_outcome(variant, &self.pos, &moves) {
            Some(r) => Some(r),
            None if variant.fifty_move_rule() && self.pos.halfmoves() >= 100 => Some((KnownOutcome::Draw, End::FiftyMoves)),
            None if self.prior + 1 >= 5 => Some((KnownOutcome::Draw, End::Fivefold)),
            None if self.auto_threefold && self.prior + 1 >= 3 => Some((KnownOutcome::Draw, End::Threefold)),
            None if self.plies >= self.max_plies => Some((KnownOutcome::Draw, End::PlyCap)),
            None => None,
        };
        self.legal.clear();
        if self.over.is_none() {
            self.legal.extend(moves.iter().map(|m| (action_id(m), *m)));
            self.legal.sort_unstable_by_key(|&(id, _)| id);
        }
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

    /// A game in progress from an arbitrary position of `variant` (hand-checked rule
    /// tests; the position counts as the first of the repetition window).
    pub fn from_position(variant: ChessVariant, pos: VariantPosition, cfg: &ChessVariantsConfig) -> Self {
        let mut g = ChessVariants::new_game(cfg);
        g.variant = Some(variant);
        g.pending_start = false;
        g.window = vec![(hash(&pos), pos.clone())];
        g.pos = pos;
        g.refresh();
        g
    }

    /// [`from_position`](Self::from_position) from a FEN (chess960 FENs use chess960
    /// castling). Tolerates contrived material (rule tests: five bishops, ...) and
    /// impossible checks, not rule violations (missing kings, pawns on the back rank).
    pub fn from_fen(variant: ChessVariant, fen: &str, cfg: &ChessVariantsConfig) -> Result<Self, String> {
        let setup: Setup = Fen::from_ascii(fen.as_bytes()).map_err(|e| format!("{fen}: {e}"))?.into_setup();
        let mode = if variant == ChessVariant::Chess960 { CastlingMode::Chess960 } else { CastlingMode::Standard };
        let pos = VariantPosition::from_setup(variant.shakmaty(), setup, mode)
            .or_else(shakmaty::PositionError::ignore_too_much_material)
            .or_else(shakmaty::PositionError::ignore_impossible_check)
            .map_err(|e| format!("{fen}: {e}"))?;
        Ok(Self::from_position(variant, pos, cfg))
    }

    /// The position as a FEN (pockets / promoted pieces / remaining checks included).
    pub fn fen(&self) -> String {
        Fen::from_position(&self.pos, EnPassantMode::Legal).to_string()
    }
}

fn hash(pos: &VariantPosition) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

fn set_bits(plane: &mut [f32], bb: Bitboard, flip: bool) {
    for sq in bb {
        plane[pov(sq, flip)] = 1.0;
    }
}

impl Game for ChessVariants {
    type Config = ChessVariantsConfig;
    const NAME: &'static str = "chess_variants";
    const VERSION: u32 = 1;

    fn check_config(cfg: &ChessVariantsConfig) -> Result<(), RulesError> {
        if cfg.variants.is_empty() {
            rules_bail!("chess_variants: no variant configured");
        }
        let mut seen = std::collections::HashSet::new();
        for n in &cfg.variants {
            let Some(v) = ChessVariant::from_name(n) else {
                rules_bail!("chess_variants: unknown variant {n:?} (known: {:?})",
                            ChessVariant::ALL.iter().map(|v| v.name()).collect::<Vec<_>>());
            };
            if !seen.insert(v) {
                rules_bail!("chess_variants: variant {n:?} listed twice");
            }
        }
        if !cfg.weights.is_empty() {
            if cfg.weights.len() != cfg.variants.len() {
                rules_bail!("chess_variants: {} weights for {} variants", cfg.weights.len(), cfg.variants.len());
            }
            if cfg.weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
                rules_bail!("chess_variants: weights must be positive, got {:?}", cfg.weights);
            }
        }
        if !(1..=100_000).contains(&cfg.max_plies) {
            rules_bail!("chess_variants: max_plies must be 1..=100000, got {}", cfg.max_plies);
        }
        let mut starts = std::collections::HashSet::new();
        for &sp in &cfg.chess960_positions {
            if sp >= N_START_960 || !starts.insert(sp) {
                rules_bail!("chess_variants: chess960_positions must be distinct Scharnagl numbers < 960, got {:?}",
                            cfg.chess960_positions);
            }
        }
        Ok(())
    }

    fn spec(cfg: &ChessVariantsConfig) -> FrameSpec {
        let n = N_TOKENS;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: cfg.max_plies,
            schema: SCHEMA.into(),
            obs: vec![
                TensorSpec::bits("board_seq", &[PIECE_CHANNELS, 8, 8]),
                TensorSpec::bits("hand_seq", &[2 * N_POCKET, HAND_FEATURES]),
                TensorSpec::f32("globals_seq", &[GLOBAL_FEATURES]),
            ],
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
            meta: json!({
                "board_h": 8, "board_w": 8, "piece_channels": PIECE_CHANNELS,
                "global_features": GLOBAL_FEATURES, "hand_piece_types": N_POCKET,
                "hand_features": HAND_FEATURES, "n_promotions": N_PROMOTIONS, "n_tokens": N_TOKENS,
                "variants": ChessVariant::ALL.iter().map(|v| v.name()).collect::<Vec<_>>(),
            }),
            candidate: None,
        }
    }

    fn new_game(cfg: &ChessVariantsConfig) -> Self {
        let mut starts = cfg.chess960_positions.clone();
        starts.sort_unstable();
        ChessVariants {
            choices: cfg.choices(),
            starts,
            auto_threefold: cfg.auto_threefold,
            max_plies: cfg.max_plies,
            variant: None,
            pending_start: false,
            pos: VariantPosition::default(),
            legal: Vec::new(),
            window: Vec::new(),
            prior: 0,
            plies: 0,
            over: None,
        }
    }

    fn turn(&self) -> Turn {
        if !self.started() {
            Turn::Chance
        } else if self.over.is_some() {
            Turn::Terminal
        } else {
            Turn::Act(SeatSet::single(self.mover_seat()))
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        if self.variant.is_none() {
            out.extend_from_slice(&self.choices);
        } else if self.pending_start {
            if self.starts.is_empty() {
                let p = 1.0 / N_START_960 as f64;
                out.extend((0..N_START_960).map(|sp| (sp, p)));
            } else {
                let p = 1.0 / self.starts.len() as f64;
                out.extend(self.starts.iter().map(|&sp| (sp, p)));
            }
        }
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.variant.is_none() {
            let Some(v) = ChessVariant::from_index(outcome).filter(|v| self.choices.iter().any(|&(i, _)| i == v.index()))
            else {
                rules_bail!("chess_variants: variant {outcome} is not configured");
            };
            if v == ChessVariant::Chess960 {
                self.variant = Some(v);
                self.pending_start = true;
            } else {
                self.start(v, 0);
            }
            Ok(())
        } else if self.pending_start {
            if outcome >= N_START_960 || !(self.starts.is_empty() || self.starts.binary_search(&outcome).is_ok()) {
                rules_bail!("chess_variants: chess960 start position {outcome} is not configured");
            }
            self.start(ChessVariant::Chess960, outcome);
            Ok(())
        } else {
            rules_bail!("chess_variants: no chance event now (got {outcome})")
        }
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.started() && self.over.is_none() && seat == self.mover_seat() {
            out.extend(self.legal.iter().map(|&(id, _)| id));
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if !self.started() || self.over.is_some() || seat != self.mover_seat() {
            rules_bail!("chess_variants: seat {seat} is not to move");
        }
        let Ok(i) = self.legal.binary_search_by_key(&action, |&(id, _)| id) else {
            rules_bail!("chess_variants: action {action} ({}) is illegal", uci(action));
        };
        let m = self.legal[i].1;
        let irreversible = self.pos.is_irreversible(m);
        self.pos.play_unchecked(m);
        self.plies += 1;
        if irreversible {
            self.window.clear();
        }
        let h = hash(&self.pos);
        let earlier = self.window.iter().filter(|(hh, p)| *hh == h && *p == self.pos).count();
        self.window.push((h, self.pos.clone()));
        self.prior = earlier as u32;
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
        let variant = self.variant.expect("started");
        let us = self.pos.turn();
        let flip = us == Color::Black;
        {
            let planes = f.obs.f32(OBS_BOARD);
            for (sq, piece) in self.pos.board().iter() {
                let plane = if piece.color == us { 0 } else { 6 } + role_index(piece.role);
                planes[plane * 64 + pov(sq, flip)] = 1.0;
            }
            if let Some(ep) = self.pos.legal_ep_square() {
                planes[PLANE_EP * 64 + pov(ep, flip)] = 1.0;
            }
            set_bits(&mut planes[PLANE_CASTLING_ROOKS * 64..(PLANE_CASTLING_ROOKS + 1) * 64],
                     self.pos.castles().castling_rights(), flip);
            set_bits(&mut planes[PLANE_PROMOTED * 64..(PLANE_PROMOTED + 1) * 64], self.pos.promoted(), flip);
            let goal = match variant {
                ChessVariant::KingOfTheHill => Bitboard::CENTER,
                ChessVariant::RacingKings => Bitboard::from_rank(Rank::Eighth),
                _ => Bitboard::EMPTY,
            };
            set_bits(&mut planes[PLANE_GOAL * 64..(PLANE_GOAL + 1) * 64], goal, flip);
            planes[PLANE_ONES * 64..(PLANE_ONES + 1) * 64].fill(1.0);
        }
        if let Some(pockets) = self.pos.pockets() {
            let hand = f.obs.f32(OBS_HAND);
            for (side, color) in [us, !us].into_iter().enumerate() {
                for (r, role) in POCKET_ROLES.into_iter().enumerate() {
                    let count = *pockets.get(color).get(role) as usize;
                    let slot = side * N_POCKET + r;
                    for k in 0..HAND_FEATURES.min(count) {
                        hand[slot * HAND_FEATURES + k] = 1.0;
                    }
                }
            }
        }
        {
            let g = f.obs.f32(OBS_GLOBALS);
            g[variant.index() as usize] = 1.0;
            g[8] = (us == Color::White) as u8 as f32;
            g[9] = self.pos.is_check() as u8 as f32;
            for k in 0..4 {
                g[10 + k] = (self.prior > k as u32) as u8 as f32;
            }
            if let Some(rc) = self.pos.remaining_checks() {
                let given = |c: Color| 3 - u32::from(*rc.get(c));
                let (mine, theirs) = (given(us), given(!us));
                g[14] = (mine >= 1) as u8 as f32;
                g[15] = (mine >= 2) as u8 as f32;
                g[16] = (theirs >= 1) as u8 as f32;
                g[17] = (theirs >= 2) as u8 as f32;
            }
            let c = self.pos.castles();
            g[18] = c.has(us, CastlingSide::KingSide) as u8 as f32;
            g[19] = c.has(us, CastlingSide::QueenSide) as u8 as f32;
            g[20] = c.has(!us, CastlingSide::KingSide) as u8 as f32;
            g[21] = c.has(!us, CastlingSide::QueenSide) as u8 as f32;
            g[22] = (self.pos.halfmoves() as f32 / 100.0).min(1.0);
            g[23] = (u32::from(self.pos.fullmoves()) as f32 / 100.0).min(1.0);
        }
        let n = N_TOKENS;
        for &(id, _) in &self.legal {
            let (fr, to, p) = action_keys(id, flip);
            f.legal.bool(LEGAL_FROM)[fr] = 1;
            f.legal.bool(LEGAL_TO_COND)[fr * n + to] = 1;
            f.legal.bool(LEGAL_PROMOTION_COND)[(fr * n + to) * N_PROMOTIONS + p] = 1;
        }
    }

    fn target_custom(&self, _seat: Seat, legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let flip = self.flip();
        let (fr, to, p) = action_keys(action, flip);
        t.i64(TGT_FROM)[0] = fr as i64;
        t.i64(TGT_TO)[0] = to as i64;
        t.i64(TGT_PROMOTION)[0] = p as i64;
        for &a in legal {
            let (f2, t2, p2) = action_keys(a, flip);
            t.f32(TGT_LEGAL_FROM)[f2] = 1.0;
            if f2 == fr {
                t.f32(TGT_LEGAL_TO_OF)[t2] = 1.0;
                if t2 == to {
                    t.f32(TGT_LEGAL_PROMOTION_OF)[p2] = 1.0;
                }
            }
        }
    }

    fn decode_custom(&self, _seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let (n, np) = (N_TOKENS as i64, N_PROMOTIONS as i64);
        let (fr, to, p) = (keys[0], keys[1], keys[2]);
        if !(0..n).contains(&fr) || !(0..n).contains(&to) || !(0..np).contains(&p) {
            return None;
        }
        let want = (fr as usize, to as usize, p as usize);
        let flip = self.flip();
        legal.iter().copied().find(|&a| action_keys(a, flip) == want)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        uci(action)
    }

    fn chance_string(&self, outcome: u32) -> String {
        if self.variant.is_none() {
            ChessVariant::from_index(outcome).map_or_else(|| format!("variant{outcome}"), |v| v.name().to_string())
        } else {
            format!("chess960 start {outcome}")
        }
    }

    fn state_string(&self) -> String {
        match self.variant {
            None => "chess_variants: variant not drawn yet".into(),
            Some(_) if self.pending_start => "chess960: start position not drawn yet".into(),
            Some(v) => format!("{} {} | ply {} seen before {} | {:?}", v.name(), self.fen(), self.plies, self.prior, self.over),
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

    fn cfg() -> ChessVariantsConfig {
        ChessVariantsConfig::default()
    }

    fn game(v: ChessVariant, fen: &str) -> ChessVariants {
        ChessVariants::from_fen(v, fen, &cfg()).unwrap()
    }

    fn id_of(g: &ChessVariants, u: &str) -> u32 {
        g.legal_moves().iter().map(|&(id, _)| id).find(|&id| uci(id) == u).unwrap_or_else(|| {
            panic!("{u} not legal in {}: {:?}", g.fen(), g.legal_moves().iter().map(|&(id, _)| uci(id)).collect::<Vec<_>>())
        })
    }

    fn play(g: &mut ChessVariants, moves: &[&str]) {
        for u in moves {
            let seat = g.mover_seat();
            let id = id_of(g, u);
            g.apply_action(seat, id).unwrap();
        }
    }

    fn ucis(g: &ChessVariants) -> Vec<String> {
        g.legal_moves().iter().map(|&(id, _)| uci(id)).collect()
    }

    #[test]
    fn names_ids_and_start_positions() {
        for v in ChessVariant::ALL {
            assert_eq!(ChessVariant::from_name(v.name()), Some(v));
            assert_eq!(ChessVariant::from_name(v.lichess_dir()), Some(v));
            if v != ChessVariant::Chess960 {
                assert_eq!(ChessVariant::from_name(v.fairy_stockfish()), Some(v));
            }
            assert_eq!(ChessVariant::from_index(v.index()), Some(v));
        }
        for tag in ["Chess960", "Crazyhouse", "Atomic", "Antichess", "Three-check", "King of the Hill", "Horde", "Racing Kings"] {
            assert!(ChessVariant::from_name(tag).is_some(), "{tag}");
        }
        assert_eq!(ChessVariant::from_name("Standard"), None);
        // Scharnagl 518 is the standard position; the inverse covers all 960
        assert_eq!(Board::chess960(518), Board::new());
        for sp in [0, 1, 518, 959] {
            assert_eq!(scharnagl_of(&Board::chess960(sp)), Some(sp));
        }
        assert_eq!(scharnagl_of(&Board::horde()), None);
        let g = ChessVariants::from_position(ChessVariant::Chess960, start_position(ChessVariant::Chess960, 518), &cfg());
        assert_eq!(g.fen().split(' ').next().unwrap(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR");
        assert_eq!(g.legal_moves().len(), 20);
        // the horde start (36 white pawns) and racing kings (21 moves)
        let h = ChessVariants::from_position(ChessVariant::Horde, start_position(ChessVariant::Horde, 0), &cfg());
        assert_eq!(h.position().board().white().count(), 36);
        assert_eq!(h.legal_moves().len(), 8);
        let rk = ChessVariants::from_position(ChessVariant::RacingKings, start_position(ChessVariant::RacingKings, 0), &cfg());
        assert_eq!(rk.legal_moves().len(), 21);
    }

    #[test]
    fn action_ids_are_unique_and_decode() {
        let mut rng = crate::rng::Rng::new(7);
        for v in ChessVariant::ALL {
            let mut g = ChessVariants::from_position(v, start_position(v, rng.below(960) as u32), &cfg());
            for _ in 0..200 {
                if g.end().is_some() {
                    break;
                }
                let ids: Vec<u32> = g.legal_moves().iter().map(|&(id, _)| id).collect();
                assert!(ids.windows(2).all(|w| w[0] < w[1]), "{v:?} ids ascending and unique");
                let flip = g.flip();
                let mut keys = std::collections::HashSet::new();
                for &(id, m) in g.legal_moves() {
                    assert_eq!(action_id(&m), id);
                    let p = action_parts(id).unwrap();
                    assert_eq!(p.to, match m { Move::Castle { rook, .. } => rook, _ => m.to() });
                    assert!(keys.insert(action_keys(id, flip)), "{v:?}: keys unique");
                }
                let a = ids[rng.below(ids.len() as u64) as usize];
                let seat = g.mover_seat();
                g.apply_action(seat, a).unwrap();
            }
        }
        assert_eq!(uci(DROP_BASE + 64 + 28), "N@e4");
        assert_eq!(action_parts(DROP_BASE + 5 * 64), None);
    }

    #[test]
    fn crazyhouse_drops_and_promoted_pieces_demote() {
        let mut g = ChessVariants::from_position(ChessVariant::Crazyhouse, start_position(ChessVariant::Crazyhouse, 0), &cfg());
        play(&mut g, &["e2e4", "d7d5", "e4d5"]); // white captures a pawn → pocket
        let p = g.position().pockets().unwrap();
        assert_eq!(p.white.pawn, 1);
        play(&mut g, &["d8d5"]); // black recaptures
        assert!(ucis(&g).iter().any(|u| u == "P@e4"), "white may drop its pawn");
        assert!(!ucis(&g).iter().any(|u| u.starts_with("P@") && (u.ends_with('1') || u.ends_with('8'))),
                "no pawn drops on the back ranks");
        // a promoted piece goes back to the pocket as a pawn
        let mut g = game(ChessVariant::Crazyhouse, "4k3/1P6/8/8/8/8/8/4K3[] w - - 0 1");
        play(&mut g, &["b7b8q"]);
        assert!(g.position().promoted().contains(Square::B8));
        let mut g2 = game(ChessVariant::Crazyhouse, "1Q~4r1/8/8/8/8/8/8/k3K3[] b - - 0 1");
        play(&mut g2, &["g8b8"]);
        let pk = g2.position().pockets().unwrap();
        assert_eq!((pk.black.pawn, pk.black.queen), (1, 0), "captured promoted queen = a pawn");
        // frame: the drop starts at the mover's pocket slot, the hand planes count
        let spec = ChessVariants::spec(&cfg());
        let mut g = game(ChessVariant::Crazyhouse, "4k3/8/8/8/8/8/8/4K3[NNp] w - - 0 1");
        let mut f = Frame::new(&spec);
        encode_frame(&g, 0, &spec, &mut f.as_mut(&spec)).unwrap();
        let hand = f.obs[1].as_f32().unwrap();
        assert_eq!(&hand[8..16], &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "own knights: 2");
        assert_eq!(hand[5 * 8], 1.0, "opponent pawn: 1");
        let drop = id_of(&g, "N@e4");
        let mut t = new_target(&spec);
        encode_target(&g, 0, &spec, &f.legal_ids, drop, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
        assert_eq!(t[0].as_i64().unwrap()[0], 64 + 1);
        assert_eq!(t[1].as_i64().unwrap()[0], pov(Square::E4, false) as i64);
        play(&mut g, &["N@e4"]);
        assert_eq!(g.position().pockets().unwrap().white.knight, 1);
    }

    #[test]
    fn atomic_explosions() {
        // Nxc7 explodes the capturer, the captured pawn and every non-pawn piece around c7
        let mut g = game(ChessVariant::Atomic, "rn1qkbnr/pppbpppp/8/1N6/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
        play(&mut g, &["b5c7"]);
        assert_eq!(g.position().board().piece_at(Square::D7), None, "neighbouring bishop exploded");
        assert_eq!(g.position().board().piece_at(Square::C7), None, "the capturer explodes too");
        assert_eq!(g.position().board().piece_at(Square::D8), None, "neighbouring queen exploded");
        assert!(g.position().board().piece_at(Square::B7).is_some(), "pawns survive explosions");
        let mut g = game(ChessVariant::Atomic, "4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1");
        play(&mut g, &["d1d7"]);
        assert_eq!(g.end(), Some((KnownOutcome::Decisive { winner: Color::White }, End::Rules)));
        assert_eq!(g.outcome(), vec![1.0, -1.0]);
        // connected kings: no check through adjacency, and a king never captures
        let g = game(ChessVariant::Atomic, "8/8/8/3kK3/8/8/8/8 w - - 0 1");
        assert!(!g.position().is_check());
        let g = game(ChessVariant::Atomic, "8/8/8/3kKp2/8/8/8/8 w - - 0 1");
        assert!(!ucis(&g).iter().any(|u| u == "e5f5"), "kings cannot capture in atomic");
    }

    #[test]
    fn three_check_counts_checks() {
        let mut g = ChessVariants::from_position(ChessVariant::ThreeCheck, start_position(ChessVariant::ThreeCheck, 0), &cfg());
        play(&mut g, &["e2e4", "e7e5", "f1c4", "a7a6", "c4f7"]); // Bxf7+ : check 1
        let rc = *g.position().remaining_checks().unwrap();
        assert_eq!((u32::from(rc.white), u32::from(rc.black)), (2, 3));
        let spec = ChessVariants::spec(&cfg());
        let mut f = Frame::new(&spec);
        encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
        let gl = f.obs[2].as_f32().unwrap();
        assert_eq!(gl[ChessVariant::ThreeCheck as usize], 1.0);
        assert_eq!((gl[9], gl[14], gl[16], gl[17]), (1.0, 0.0, 1.0, 0.0), "black in check, opponent gave 1");
        play(&mut g, &["e8f7", "d1h5", "g7g6", "h5e5"]); // Qh5+ is check 2, Qxe5 no check
        assert_eq!(u32::from(g.position().remaining_checks().unwrap().white), 1);
        assert!(g.end().is_none());
        // FEN remaining checks "white+black": white needs one more check
        let mut g = game(ChessVariant::ThreeCheck, "4k3/8/8/8/8/8/8/R3K3 w - - 1+3 0 1");
        play(&mut g, &["a1a8"]);
        assert_eq!(g.end(), Some((KnownOutcome::Decisive { winner: Color::White }, End::Rules)));
    }

    #[test]
    fn king_of_the_hill_racing_kings_horde_antichess() {
        let mut g = game(ChessVariant::KingOfTheHill, "4k3/8/8/8/8/3K4/8/8 w - - 0 1");
        play(&mut g, &["d3d4"]);
        assert_eq!(g.outcome(), vec![1.0, -1.0]);
        // racing kings: no checks may be given; a race to the 8th rank, black may equalise
        let g = game(ChessVariant::RacingKings, "8/k7/8/8/8/8/8/1R5K w - - 0 1");
        assert!(!ucis(&g).iter().any(|u| u == "b1a1" || u == "b1b7"), "giving check is illegal");
        assert!(ucis(&g).iter().any(|u| u == "b1b6"));
        let mut g = game(ChessVariant::RacingKings, "8/k6K/8/8/8/8/8/8 w - - 0 1");
        play(&mut g, &["h7h8"]);
        assert!(g.end().is_none(), "black can still reach the 8th rank");
        play(&mut g, &["a7a8"]);
        assert_eq!(g.end(), Some((KnownOutcome::Draw, End::Rules)));
        let mut g = game(ChessVariant::RacingKings, "8/7K/8/k7/8/8/8/8 w - - 0 1");
        play(&mut g, &["h7h8"]);
        assert_eq!(g.outcome(), vec![1.0, -1.0], "black cannot catch up");
        // horde: first-rank pawns may double-step; black wins by capturing everything
        let g = game(ChessVariant::Horde, "4k3/8/8/8/8/8/8/P7 w - - 0 1");
        assert!(ucis(&g).iter().any(|u| u == "a1a3"));
        let mut g = game(ChessVariant::Horde, "4k3/8/8/8/8/8/1r6/P7 b - - 0 1");
        play(&mut g, &["b2b1"]);
        assert!(g.end().is_none());
        let mut g = game(ChessVariant::Horde, "4k3/8/8/8/8/8/8/Pr6 b - - 0 1");
        play(&mut g, &["b1a1"]);
        assert_eq!(g.outcome(), vec![-1.0, 1.0]);
        // antichess: captures are compulsory; losing everything or being stalemated wins
        let g = game(ChessVariant::Antichess, "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1");
        assert_eq!(ucis(&g), vec!["e1d2"]);
        let mut g = game(ChessVariant::Antichess, "8/8/8/8/8/8/3p4/4K3 w - - 0 1");
        play(&mut g, &["e1d2"]);
        assert_eq!(g.outcome(), vec![-1.0, 1.0], "black has no piece left: black wins");
        let g = game(ChessVariant::Antichess, "8/8/8/8/8/1p6/1P6/8 w - - 0 1");
        assert_eq!(g.outcome(), vec![1.0, -1.0], "white is stalemated: white wins");
        let g = game(ChessVariant::Antichess, "8/P7/8/8/8/8/8/k7 w - - 0 1");
        assert!(ucis(&g).iter().any(|u| u == "a7a8k"), "promotion to king");
    }

    #[test]
    fn chess960_castling_is_king_to_rook() {
        // king b1, rooks a1 / h1: O-O-O puts the king on c1, the rook on d1
        let mut g = game(ChessVariant::Chess960, "1r4kr/pppppppp/8/8/8/8/PPPPPPPP/RK5R w KQkq - 0 1");
        assert!(ucis(&g).iter().any(|u| u == "b1a1"));
        play(&mut g, &["b1a1"]);
        let b = g.position().board();
        assert_eq!((b.piece_at(Square::C1).map(|p| p.role), b.piece_at(Square::D1).map(|p| p.role)),
                   (Some(Role::King), Some(Role::Rook)));
        // king already on g1 with the rook on h1: castling moves only the rook
        let mut g = game(ChessVariant::Chess960, "6kr/pppppppp/8/8/8/8/PPPPPPPP/R5KR w KQk - 0 1");
        assert!(ucis(&g).iter().any(|u| u == "g1h1"));
        play(&mut g, &["g1h1"]);
        let b = g.position().board();
        assert_eq!((b.piece_at(Square::G1).map(|p| p.role), b.piece_at(Square::F1).map(|p| p.role)),
                   (Some(Role::King), Some(Role::Rook)));
    }

    #[test]
    fn automatic_draws_fifty_fivefold_threefold_cap() {
        // 50 moves: halfmove clock 99 → one quiet move ends the game (not in crazyhouse)
        let mut g = game(ChessVariant::Chess960, "4k3/8/8/8/8/8/8/R3K3 w - - 99 80");
        play(&mut g, &["a1a2"]);
        assert_eq!(g.end(), Some((KnownOutcome::Draw, End::FiftyMoves)));
        let mut g = game(ChessVariant::Chess960, "4k3/R7/8/8/8/8/8/1R2K3 w - - 99 80");
        play(&mut g, &["b1b8"]); // mate on the 100th halfmove: mate wins
        assert_eq!(g.end(), Some((KnownOutcome::Decisive { winner: Color::White }, End::Rules)));
        let mut g = game(ChessVariant::Crazyhouse, "4k3/8/8/8/8/8/8/R3K3[] w - - 99 80");
        play(&mut g, &["a1a2"]);
        assert!(g.end().is_none(), "crazyhouse has no fifty-move rule");
        // repetitions: fivefold ends the game; threefold only with auto_threefold
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let mut g = ChessVariants::from_position(ChessVariant::KingOfTheHill, start_position(ChessVariant::KingOfTheHill, 0), &cfg());
        for i in 0..4 {
            play(&mut g, &shuffle);
            assert_eq!(g.prior_occurrences(), i + 1);
            assert_eq!(g.end().is_some(), i + 1 == 4, "start position seen {} times", i + 2);
        }
        assert_eq!(g.end(), Some((KnownOutcome::Draw, End::Fivefold)));
        let auto = ChessVariantsConfig { auto_threefold: true, ..cfg() };
        let mut g = ChessVariants::from_position(ChessVariant::Atomic, start_position(ChessVariant::Atomic, 0), &auto);
        play(&mut g, &shuffle);
        assert!(g.end().is_none());
        play(&mut g, &shuffle);
        assert_eq!(g.end(), Some((KnownOutcome::Draw, End::Threefold)));
        // an irreversible move resets the repetition window
        let mut g = ChessVariants::from_position(ChessVariant::Horde, start_position(ChessVariant::Horde, 0), &cfg());
        play(&mut g, &["a4a5", "g8f6", "b5b6", "f6g8"]);
        assert_eq!(g.prior_occurrences(), 0);
        // the ply cap
        let capped = ChessVariantsConfig { max_plies: 4, ..cfg() };
        let mut g = ChessVariants::from_position(ChessVariant::RacingKings, start_position(ChessVariant::RacingKings, 0), &capped);
        play(&mut g, &["h2h3", "a2a3", "h3h4", "a3a4"]);
        assert_eq!(g.end(), Some((KnownOutcome::Draw, End::PlyCap)));
    }

    /// The rule vectors of scalachess' own tests (test-kit AtomicVariantTest,
    /// AntichessVariantTest, HordeVariantTest, RacingKingsVariantTest): lichess'
    /// automatic ends, move by move.
    #[test]
    fn lichess_automatic_ends_match_scalachess() {
        let draw = Some((KnownOutcome::Draw, End::InsufficientMaterial));
        let after = |v: ChessVariant, fen: &str, moves: &[&str]| {
            let mut g = game(v, fen);
            play(&mut g, moves);
            g
        };
        // atomic: winning material (K+B vs K+B on opposite colours, bare kings, K+minor);
        // scalachess' K+B vs K+B position has White in check with Black to move (c6), here e6
        assert_eq!(game(ChessVariant::Atomic, "4K3/8/4b3/8/8/8/5B2/3k4 b - -").end(), draw);
        for (fen, m) in [("8/1n6/8/8/8/8/k7/2K1b2R w - -", "h1e1"), ("8/1b6/8/8/8/8/k7/2K1n2R w - -", "h1e1"),
                         ("8/8/8/8/8/8/N4r2/5k1K b - -", "f2a2"), ("6r1/8/8/1k6/8/8/2K5/6R1 w - -", "g1g8")] {
            assert_eq!(after(ChessVariant::Atomic, fen, &[m]).end(), draw, "{fen} {m}");
        }
        // ... not with helpmate material
        for (fen, m) in [("B2BBBB1/7P/8/8/8/8/3kb3/4K3 w - - 1 53", "h7h8b"), ("8/5k2/8/8/8/8/4pK2/5b2 b - - 1 44", "e2e1b"),
                         ("8/5k2/8/8/8/8/4pK2/5b2 b - - 1 44", "e2e1n"), ("8/5k2/8/8/8/8/4pKB1/5B2 b - - 1 44", "e2e1b"),
                         ("8/5k2/8/8/8/8/4pKB1/6B1 b - - 1 44", "e2e1b"), ("8/5k2/8/8/8/8/4pKB1/6N1 b - - 1 44", "e2e1b"),
                         ("8/5k2/8/8/8/8/4pKN1/6B1 b - - 1 44", "e2e1b"), ("8/5k2/8/8/8/8/4pKN1/6N1 b - - 1 44", "e2e1b"),
                         ("8/5k2/8/8/8/8/4pKN1/6N1 b - - 1 44", "e2e1n"), ("8/5k2/8/8/8/8/4pKN1/6n1 b - - 1 44", "e2e1n"),
                         ("8/5k2/8/8/8/8/4pKn1/6n1 b - - 1 44", "e2e1n"), ("3Q4/2b2k2/5P2/8/8/8/6K1/8 b - - 0 57", "c7d8")] {
            assert_eq!(after(ChessVariant::Atomic, fen, &[m]).end(), None, "{fen} {m}");
        }
        // atomic closed positions: kings + immobile pawns, a pawnitized bishop
        assert_eq!(after(ChessVariant::Atomic, "8/8/6p1/3K4/6P1/2k5/8/8 w - -", &["g4g5"]).end(), draw);
        assert_eq!(after(ChessVariant::Atomic, "8/8/2k1p3/5p2/4PP2/1b6/4K3/8 w - - 0 1", &["e4e5"]).end(), draw);
        assert_eq!(after(ChessVariant::Atomic, "8/8/2k5/5p2/8/2b2P2/8/3K4 w - - 0 1", &["f3f4"]).end(), None);
        assert_eq!(after(ChessVariant::Atomic, "6bk/4B2p/8/7P/4K3/8/8/8 w - - 0 1", &["h5h6"]).end(), None);
        // checkmate overrides a closed position (the mating pawn could "capture" the king)
        assert_eq!(after(ChessVariant::Atomic, "8/8/b1p5/kpP5/p3K3/PP6/8/8 w - - 0 1", &["b3b4"]).end(),
                   Some((KnownOutcome::Decisive { winner: Color::White }, End::Rules)));
        // the closed pawn structure of a real game lichess drew (lichess.org/risWjxVl)
        assert_eq!(game(ChessVariant::Atomic, "8/7p/7P/8/1p6/pP6/P7/1kK5 w - - 0 49").end(), draw);
        // antichess: opposite-colour bishops, with unattackable blocked pawns
        let anti = ChessVariant::Antichess;
        assert_eq!(after(anti, "8/2b5/8/8/8/6Q1/4B3/8 b - -", &["c7g3"]).end(), draw);
        assert_eq!(after(anti, "8/6P1/8/8/1b6/8/8/5B2 w - -", &["g7g8b"]).end(), draw);
        assert_eq!(after(anti, "8/6p1/4B1P1/4p3/4P3/8/2p5/8 b - - 1 28", &["c2c1b"]).end(), draw);
        assert_eq!(after(anti, "7b/8/1p6/8/8/8/5B2/8 w - -", &["f2b6"]).end(), None);
        assert_eq!(after(anti, "8/6p1/1B4P1/4p3/4P3/8/3p4/8 b - -", &["d2d1b"]).end(), None);
        assert_eq!(after(anti, "5b2/1P4p1/4B1P1/4p3/4P3/8/8/8 w - -", &["b7b8b"]).end(), None);
        assert_eq!(after(anti, "8/8/4BbP1/4p3/4P3/8/8/8 b - -", &["f6g7"]).end(), None);
        assert_eq!(game(anti, "4K3/8/1b6/8/8/8/5B2/3k4 b - -").end(), None, "antichess kings are ordinary pieces");
        // horde: a stalemated horde (now, or after every black move)
        for fen in ["8/p7/pk6/P7/P7/8/8/8 b - -", "QNBRRBNQ/PPpPPpPP/P1P2PkP/8/8/8/8/8 b - -", "b7/pk6/P7/P7/8/8/8/8 b - - 0 1",
                    "8/p7/P7/P7/8/2q5/8/7k b - - 0 1", "krb5/pb1p4/P2Pp3/P3Pp2/5Pp1/6Pp/7P/8 b - - 0 1"] {
            assert_eq!(game(ChessVariant::Horde, fen).end(), draw, "{fen}");
        }
        for fen in ["k7/p2p4/P2Pp1p1/P3PpPp/5PpP/6Pp/7P/7n b - - 0 1", "8/1b5r/1P6/1Pk3q1/1PP5/r1P5/P1P5/2P5 b - - 0 52",
                    "8/8/8/7k/7P/7P/8/8 b - - 0 58", "8/p7/P7/P7/8/8/8/6qk b - - 0 1", "7B/6k1/8/8/8/8/8/8 b - -",
                    "7B/7k/8/8/8/8/8/8 w - - 0 1", "r7/2Bb4/q3k3/8/8/3q4/8/5qqr b - -"] {
            assert_eq!(game(ChessVariant::Horde, fen).end(), None, "{fen}");
        }
        // racing kings
        let rk = |fen: &str| game(ChessVariant::RacingKings, fen).end();
        for fen in ["4krn1/K2b4/8/8/8/8/8/8 w - - 4 3", "4brk1/8/5n2/K7/8/8/8/8 w - - 6 4", "3kbrn1/8/8/K7/8/8/8/8 w - - 4 3",
                    "4brk1/4n3/8/K7/8/8/8/8 w - - 4 3"] {
            assert_eq!(rk(fen), Some((KnownOutcome::Decisive { winner: Color::Black }, End::Rules)), "{fen}");
        }
        for fen in ["K3br2/5k2/8/8/6n1/8/8/8 w - - 4 3", "K3b2r/5k2/5n2/8/8/8/8/8 w - - 4 3"] {
            assert_eq!(rk(fen), Some((KnownOutcome::Decisive { winner: Color::White }, End::Rules)), "{fen}");
        }
        assert_eq!(rk("K3brk1/8/5n2/8/8/8/8/8 w - - 4 3"), Some((KnownOutcome::Draw, End::Rules)));
        assert_eq!(rk("1K2br2/5k2/5n2/8/8/8/8/8 b - - 3 2"), None);
        // chess960: the standard rule (kings + one minor, or same-coloured bishops only)
        let std = |fen: &str| game(ChessVariant::Chess960, fen).end();
        // (dark squares: c1, e3, d6; light: c6)
        for fen in ["8/8/8/4k3/8/8/8/4K3 w - - 0 1", "8/8/8/4k3/8/8/8/2B1K3 w - - 0 1", "8/8/8/4k3/8/8/8/1N2K3 w - - 0 1",
                    "8/8/3b4/4k3/8/8/8/2B1K3 w - - 0 1", "8/8/3b4/4k3/8/4B3/8/2B1K3 w - - 0 1"] {
            assert_eq!(std(fen), draw, "{fen}");
        }
        for fen in ["8/8/2b5/4k3/8/8/8/2B1K3 w - - 0 1", "8/8/8/4k3/8/8/8/NN2K3 w - - 0 1", "8/8/2n5/4k3/8/8/8/2B1K3 w - - 0 1",
                    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"] {
            assert_eq!(std(fen), None, "{fen}");
        }
        // three-check: bare kings only; crazyhouse / king of the hill: never
        assert_eq!(game(ChessVariant::ThreeCheck, "8/8/8/4k3/8/8/8/4K3 w - - 0 1").end(), draw);
        assert_eq!(game(ChessVariant::ThreeCheck, "8/8/8/4k3/8/8/8/2B1K3 w - - 0 1").end(), None);
        assert_eq!(game(ChessVariant::KingOfTheHill, "8/8/8/8/8/8/8/k3K3 w - - 0 1").end(), None);
    }

    #[test]
    fn chess960_start_positions_can_be_restricted() {
        let c = ChessVariantsConfig { variants: vec!["chess960".into()], chess960_positions: vec![7, 518], ..cfg() };
        ChessVariants::check_config(&c).unwrap();
        let mut g = ChessVariants::new_game(&c);
        g.apply_chance(ChessVariant::Chess960 as u32).unwrap();
        let mut out = Vec::new();
        g.chance_outcomes(&mut out);
        assert_eq!(out, vec![(7, 0.5), (518, 0.5)]);
        assert!(g.clone().apply_chance(8).is_err());
        g.apply_chance(518).unwrap();
        for bad in [vec![960], vec![3, 3]] {
            assert!(ChessVariants::check_config(&ChessVariantsConfig { chess960_positions: bad, ..cfg() }).is_err());
        }
    }

    #[test]
    fn trajectory_chance_events_and_frames() {
        let c = cfg();
        let spec = ChessVariants::spec(&c);
        spec.validate().unwrap();
        assert_eq!(spec.layout(), Layout::Interleaved);
        let mut g = ChessVariants::new_game(&c);
        let mut out = Vec::new();
        g.chance_outcomes(&mut out);
        assert_eq!(out.len(), 8);
        assert!((out.iter().map(|&(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
        g.apply_chance(ChessVariant::Chess960 as u32).unwrap();
        assert_eq!(g.turn(), Turn::Chance);
        out.clear();
        g.chance_outcomes(&mut out);
        assert_eq!(out.len(), 960);
        g.apply_chance(518).unwrap();
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)));
        // a variant outside the config is impossible
        let only = ChessVariantsConfig { variants: vec!["atomic".into(), "horde".into()], weights: vec![3.0, 1.0], ..c.clone() };
        let mut h = ChessVariants::new_game(&only);
        out.clear();
        h.chance_outcomes(&mut out);
        assert_eq!(out, vec![(2, 0.75), (6, 0.25)]);
        assert!(h.apply_chance(ChessVariant::Crazyhouse as u32).is_err());
        // black's frame is rank-mirrored: its pawns sit on POV row 6
        play(&mut g, &["e2e4"]);
        let mut f = Frame::new(&spec);
        encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
        let planes = f.obs[0].as_f32().unwrap();
        assert!((48..56).all(|sq| planes[sq] == 1.0), "own pawns on POV row 6");
        assert_eq!(planes[6 * 64 + pov(Square::E4, true)], 1.0, "white's e4 pawn, mirrored");
        assert_eq!(planes[PLANE_EP * 64..PLANE_EP * 64 + 64].iter().sum::<f32>(), 0.0, "no legal en passant");
        assert_eq!(planes[PLANE_CASTLING_ROOKS * 64..(PLANE_CASTLING_ROOKS + 1) * 64].iter().sum::<f32>(), 4.0);
        let gl = f.obs[2].as_f32().unwrap();
        assert_eq!((gl[0], gl[8], gl[18], gl[21]), (1.0, 0.0, 1.0, 1.0));
        // targets and decoding round-trip; wrong keys decode to nothing
        let a = id_of(&g, "g8f6");
        let mut t = new_target(&spec);
        encode_target(&g, 1, &spec, &f.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
        let keys: Vec<i64> = (0..3).map(|i| t[i].as_i64().unwrap()[0]).collect();
        assert_eq!(keys, vec![pov(Square::G8, true) as i64, pov(Square::F6, true) as i64, 0]);
        assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &keys).unwrap(), a);
        assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[0, 0, 0]).is_err());
        assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[keys[0], keys[1], 4]).is_err());
        // a replayed trajectory reproduces it
        let ev = vec![Event::Chance(0), Event::Chance(518), Event::Act(0, id_of(&ChessVariants::from_position(
            ChessVariant::Chess960, start_position(ChessVariant::Chess960, 518), &c), "e2e4"))];
        let r = replay::<ChessVariants>(&c, &ev).unwrap();
        assert_eq!(r.decisions.len(), 1);
        assert!(!r.terminal());
    }

    #[test]
    fn conformance_battery() {
        // each variant alone (few long random games: causality replays are quadratic)
        for v in ChessVariant::ALL {
            let c = ChessVariantsConfig { variants: vec![v.name().into()], max_plies: 300, ..cfg() };
            let rep = check_game::<ChessVariants>(&c, 3, v.index() as u64).unwrap();
            assert_eq!(rep.terminal, 3, "{v:?}");
        }
        let rep = check_game::<ChessVariants>(&ChessVariantsConfig { max_plies: 200, auto_threefold: true, ..cfg() }, 8, 11).unwrap();
        assert_eq!(rep.games, 8);
    }
}
