//! `gmrules` — game rules and frame encoders, implemented ONCE and shared by the
//! converters (`gmdata`: replay + verification of every record → IL frames) and the
//! simulators (`gmsim`: the RL environment). IL observations and RL observations are
//! therefore identical by construction. No I/O in this crate.
//!
//! ```text
//! record (any source format) ──parse (gmdata/<game>.rs)──► Trajectory ─┐
//!                                                                      ├─ replay ─► frames + targets ─► gm-shards
//! VecSim (RL env, gmsim) ── chance from a seeded Rng, actions ─────────┘   (encode_frame / encode_target)
//! ```
//!
//! Pieces:
//! - [`game::Game`]: the trait a rule engine implements (turns, chance, canonical
//!   action ids, outcome, the encoders, `perturb_hidden`), and [`game::FrameSpec`]:
//!   the frame schema — named obs / legal / target tensors ([`schema`]) plus the
//!   action keys; [`game::FrameSpec::candidate`] builds the candidate-action schema
//!   (`encode_obs` / `encode_action`), custom schemas use `encode_custom` /
//!   `target_custom` / `decode_custom`.
//! - [`frame`]: frame buffers and the single encoding path: [`frame::encode_frame`],
//!   [`frame::encode_target`], [`frame::decode_action`].
//! - [`trajectory`]: [`trajectory::Event`] sequences (the universal record format)
//!   and the verifying [`trajectory::replay`].
//! - [`sim::VecSim`]: N episodes stepped together (rows / streams / seats; parallel
//!   across envs with `SimOptions::threads`).
//! - [`invariance`]: causality / privacy / conformance checks every game ships
//!   ([`invariance::check_game`]).
//! - [`registry`]: type-erased access by game name (gmsim, the Python side).
//! - [`rng::Rng`]: the deterministic RNG.
//!
//! Games: `kuhn_poker` (the toy reference game: hidden cards, chance, 2-4 players,
//! implemented through every layer), `goofspiel` (simultaneous moves), `pig`
//! (perfect information + dice: the interleaved layout) and `breakthrough` (the
//! custom-schema board game on the from-to decoder) — all pinned to OpenSpiel.
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
pub mod schema;
pub mod sim;
pub mod trajectory;

pub use frame::{action_keys_of, decode_action, encode_frame, encode_target, new_target, Frame, FrameMut, ObsView};
pub use game::{CandidateSpec, FrameSpec, Game, Layout, RulesError, Seat, SeatSet, Turn, CANDIDATE_SCHEMA};
pub use rng::Rng;
pub use schema::{Buf, BufMut, BufRef, DType, TensorSpec, TensorsMut};
pub use sim::{SimOptions, StepInfo, VecSim};
pub use trajectory::{replay, Decision, Event, Replay, ReplayError, Stream};
