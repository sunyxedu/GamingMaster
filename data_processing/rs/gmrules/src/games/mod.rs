//! Game rule engines (one module per game; register each in `crate::registry::GAMES`).
//!
//! - [`kuhn_poker`]: the toy reference game (hidden information, chance, 2-4 players),
//!   implemented through every layer (converter, shards, adapter, simulator, trainers).
//! - [`goofspiel`]: the simultaneous-move test game (no converter).
//! - [`pig`]: the perfect-information dice test game (interleaved layout, no converter).
//! - [`breakthrough`]: the CUSTOM-schema reference game (a from-to board game for the
//!   board adapter; the pattern for chess-like games).
//! - [`chess_variants`]: the Lichess chess variants (shakmaty's rules), one game with
//!   a variant chance event; from-to decoder over board squares + pocket slots.

pub mod breakthrough;
pub mod goofspiel;
pub mod kuhn_poker;
pub mod pig;
pub mod chess_variants;
