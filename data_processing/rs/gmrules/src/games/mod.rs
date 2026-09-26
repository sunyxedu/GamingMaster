//! Game rule engines (one module per game; register each in `crate::registry::GAMES`).
//!
//! - [`kuhn_poker`]: the toy reference game (hidden information, chance, 2-4 players),
//!   implemented through every layer (converter, shards, adapter, simulator, trainers).
//! - [`goofspiel`]: the simultaneous-move test game (no converter).

pub mod goofspiel;
pub mod kuhn_poker;
