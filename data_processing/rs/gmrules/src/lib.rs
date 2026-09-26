//! `gmrules` — game rules and observation / legal-action encoders, implemented ONCE
//! and shared by the converters (`gmdata`: replay + verification of every record → IL
//! frames) and the simulators (`gmsim`: the RL environment). IL observations and RL
//! observations are therefore identical by construction. No I/O in this crate.
//!
//! ```text
//! record (any source format) ──parse (gmdata/<game>.rs)──► Trajectory ─┐
//!                                                                      ├─ replay ─► Decision frames ─► gm-shards
//! VecSim (RL env, gmsim) ── chance from a seeded Rng, actions ─────────┘   (encode_frame, the same code)
//! ```
//!
//! Pieces:
//! - [`game::Game`]: the trait a rule engine implements (turns, chance, canonical
//!   action ids, outcome, `encode_obs` / `encode_action`, `perturb_hidden`), plus
//!   [`game::FrameSpec`] (fixed frame shapes = the candidate-action gm-shard schema).
//! - [`frame`]: frame buffers and [`frame::encode_frame`], the single encoding routine.
//! - [`trajectory`]: [`trajectory::Event`] sequences (the universal record format)
//!   and the verifying [`trajectory::replay`].
//! - [`sim::VecSim`]: N episodes stepped together (rows / streams / seats).
//! - [`invariance`]: causality / privacy / conformance checks every game ships
//!   ([`invariance::check_game`]).
//! - [`registry`]: type-erased access by game name (gmsim, the Python side).
//! - [`rng::Rng`]: the deterministic RNG.
//!
//! Games: `kuhn_poker` (the toy reference game: hidden cards, chance, 2-4 players,
//! implemented through every layer), `goofspiel` (simultaneous moves) and `pig`
//! (perfect information + dice: the interleaved layout) — all pinned to OpenSpiel.
//!
//! Adding a game: `src/games/<game>.rs` implementing [`game::Game`] (+ its unit test
//! calling `invariance::check_game`), `pub mod <game>;` in `games/mod.rs`, one line in
//! [`registry::GAMES`]. See data_processing/README.md "Adding a game" for the
//! converter, Python and RL sides.

pub mod frame;
pub mod game;
pub mod games;
pub mod invariance;
pub mod registry;
pub mod rng;
pub mod sim;
pub mod trajectory;

pub use frame::{encode_frame, Frame, FrameMut, ObsView};
pub use game::{FrameSpec, Game, Layout, RulesError, Seat, SeatSet, Turn};
pub use rng::Rng;
pub use sim::{SimOptions, StepInfo, VecSim};
pub use trajectory::{replay, Decision, Event, Replay, ReplayError, Stream};
