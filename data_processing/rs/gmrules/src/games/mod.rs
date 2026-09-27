//! Game rule engines (one module per game; register each in `crate::registry::GAMES`).
//!
//! - [`kuhn_poker`]: the toy reference game (hidden information, chance, 2-4 players),
//!   implemented through every layer (converter, shards, adapter, simulator, trainers).
//! - [`goofspiel`]: the simultaneous-move test game (no converter).
//! - [`pig`]: the perfect-information dice test game (interleaved layout, no converter).
//! - [`breakthrough`]: the CUSTOM-schema reference game (a from-to board game for the
//!   board adapter; the pattern for chess-like games).
//! - [`chess`]: standard chess (shakmaty's FIDE rules) with the `gmdata chess` converter's
//!   frozen v3 frames (the converter builds its frames through this game).
//! - [`chess_variants`]: the Lichess chess variants (shakmaty's rules), one game with
//!   a variant chance event; from-to decoder over board squares + pocket slots.
//! - [`xiangqi`]: Chinese chess (Asian rules as Fairy-Stockfish implements them, pinned to
//!   pyffish), from-to frames; converter `gmdata xiangqi`.
//! - [`backgammon`]: backgammon with the doubling cube (match / money play; full-move
//!   candidates; gnubg-pinned).
//! - [`doudizhu`]: Dou Dizhu (3 players, auction, landlord vs peasants; OpenSpiel's
//!   `dou_dizhu` + four-with-two), candidate schema with factored kicker decisions.
//! - [`skat`]: Skat (3 players, ISS rules; converter from the ISS game archive).
//! - [`bridge`]: contract bridge (auction + card play, duplicate scoring; a team game).
//! - [`hunl`]: heads-up no-limit Texas hold'em (ACPC rules; exact bet amounts in the
//!   rules, abstracted bet-size candidates for the policy: a custom schema with the
//!   candidate layout).

pub mod backgammon;
pub mod breakthrough;
pub mod chess;
pub mod doudizhu;
pub mod bridge;
pub mod goofspiel;
pub mod hunl;
pub mod kuhn_poker;
pub mod pig;
pub mod chess_variants;
pub mod xiangqi;
pub mod skat;
