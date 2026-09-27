//! Riichi mahjong under Tenhou's rules, shared by the converter and the simulator so that an
//! RL frame is the IL frame of the same position:
//!
//! - [`rules`]: tiles, hand shapes, waits and the hora rules (yaku via riichienv-core's hand
//!   evaluator);
//! - [`state`]: the per-kyoku state tracked from mjai events, the exact legality of every
//!   decision and the observation encoding (gmdata's `mahjong convert`, see its module docs);
//! - [`sim`]: whole hanchan games on riichienv-core's engine, observed through [`state`]
//!   (gmsim's `MahjongSim`, env/vec_mahjong.py);
//! - [`vec`]: N games stepped together with gmsim's per-seat row conventions.

pub mod rules;
pub mod sim;
pub mod state;
pub mod vec;
