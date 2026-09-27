//! TEST-ONLY reference: the chess converter's per-frame code as it was before the frames
//! moved into gmrules (gmdata `chess.rs` at 39b4290, converter_version 3), verbatim but
//! for the visibility of the items. It shares no code with `gmrules::games::chess`
//! (only shakmaty), so the parity tests ([`super::parity`]) compare two implementations:
//! the converter must stay bit-identical to this one on every record.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};
use std::collections::HashMap;

pub const GLOBAL_FEATURES: usize = 10;
pub const N_PROMOTIONS: usize = 5;

/// POV square index `row·8 + file` of `sq` for the side to move (`flip` = black to move:
/// row = rank; white to move: row = 7 − rank; ranks / files 0-based from a1).
pub fn pov(sq: Square, flip: bool) -> i64 {
    let (file, rank) = (u32::from(sq.file()) as i64, u32::from(sq.rank()) as i64);
    if flip {
        rank * 8 + file
    } else {
        (7 - rank) * 8 + file
    }
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

fn promo_idx(r: Option<Role>) -> i64 {
    match r {
        None => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        Some(_) => 0,
    }
}

/// A move as the chess adapter labels it, in the frame's POV coordinates:
/// (from square, to square, promotion index). Castling → king destination file g/c
/// (python-chess standard mode; shakmaty encodes castling as king → rook). Used for the
/// labels AND the legality masks, so the GT move always lies inside its own masks.
pub fn move_target(m: &Move, flip: bool) -> Option<(i64, i64, i64)> {
    let from_sq = m.from()?;
    let to_sq = match *m {
        Move::Castle { king, rook } => {
            let side = if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide };
            Square::from_coords(side.king_to_file(), king.rank())
        }
        _ => m.to(),
    };
    Some((pov(from_sq, flip), pov(to_sq, flip), promo_idx(m.promotion())))
}

/// Legality masks of one frame, conditioned on the GT move `gt` = (from, to, promotion):
/// (legal from-squares, legal destinations from `gt.from`, legal promotions of
/// (`gt.from`, `gt.to`)). `None` if `gt` is not inside all three (never for a legal move).
#[allow(clippy::type_complexity)]
pub fn legal_masks(pos: &Chess, flip: bool, gt: (i64, i64, i64)) -> Option<([f32; 64], [f32; 64], [f32; N_PROMOTIONS])> {
    let (mut from, mut to_of, mut promo_of) = ([0f32; 64], [0f32; 64], [0f32; N_PROMOTIONS]);
    for m in pos.legal_moves() {
        let Some((f, t, p)) = move_target(&m, flip) else { continue };
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

/// Floats of one `obs.board_seq` frame (14 planes of 8×8).
pub const PLANE_LEN: usize = 14 * 64;

/// Occurrence counts of the positions of one game (Chess Eq/Hash = FIDE repetition
/// identity: board, side to move, castling rights, legal en-passant square).
#[derive(Default)]
pub struct Repetitions(HashMap<Chess, u32>);

impl Repetitions {
    /// Record `pos`; returns how often it occurred earlier in the game.
    pub fn see(&mut self, pos: &Chess) -> u32 {
        let n = self.0.entry(pos.clone()).or_insert(0);
        *n += 1;
        *n - 1
    }
}

/// Observation of one frame (the position before the move, from the mover's POV):
/// writes the `obs.board_seq` planes into `planes` (`PLANE_LEN` floats, zero on entry)
/// and appends the `GLOBAL_FEATURES` `obs.globals_seq` values to `globals`; `prior` =
/// earlier occurrences of the position ([`Repetitions::see`]).
pub fn encode_position(pos: &Chess, prior: u32, planes: &mut [f32], globals: &mut Vec<f32>) {
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
    globals.extend_from_slice(&[
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

/// Why a game produced no sample.
#[derive(Debug, PartialEq, Eq)]
pub enum Skip {
    /// illegal move / missing from-square / unknown result
    Invalid,
    /// the GT move fell outside its own legality masks (label/mask convention bug)
    MaskMismatch,
}

/// Per-frame tensors of one game (flat, frame-major).
#[derive(Default)]
pub struct Frames {
    pub planes: Vec<f32>,
    pub globals: Vec<f32>,
    pub from: Vec<i64>,
    pub to: Vec<i64>,
    pub promo: Vec<i64>,
    pub value: Vec<f32>,
    pub legal_from: Vec<f32>,
    pub legal_to_of: Vec<f32>,
    pub legal_promotion_of: Vec<f32>,
}

/// Replay `moves` from the initial position; `wv` = result for white.
pub fn encode_moves(moves: &[Move], wv: f32) -> Result<Frames, Skip> {
    let t = moves.len();
    let mut f = Frames {
        planes: vec![0f32; t * PLANE_LEN],
        globals: Vec::with_capacity(t * GLOBAL_FEATURES),
        legal_from: Vec::with_capacity(t * 64),
        legal_to_of: Vec::with_capacity(t * 64),
        legal_promotion_of: Vec::with_capacity(t * N_PROMOTIONS),
        ..Frames::default()
    };
    let mut pos = Chess::default();
    let mut reps = Repetitions::default();
    for (k, m) in moves.iter().enumerate() {
        if !pos.is_legal(*m) {
            return Err(Skip::Invalid);
        }
        let own = pos.turn();
        let flip = own == Color::Black;
        let prior = reps.see(&pos);
        encode_position(&pos, prior, &mut f.planes[k * PLANE_LEN..(k + 1) * PLANE_LEN], &mut f.globals);
        let gt = move_target(m, flip).ok_or(Skip::Invalid)?;
        let (lf, lt, lp) = legal_masks(&pos, flip, gt).ok_or(Skip::MaskMismatch)?;
        f.legal_from.extend_from_slice(&lf);
        f.legal_to_of.extend_from_slice(&lt);
        f.legal_promotion_of.extend_from_slice(&lp);
        f.from.push(gt.0);
        f.to.push(gt.1);
        f.promo.push(gt.2);
        f.value.push(if own == Color::White { wv } else { -wv });
        pos.play_unchecked(*m);
    }
    Ok(f)
}
