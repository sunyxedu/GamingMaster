//! Clash Royale replay data: the catalog, the elixir and hand-cycle reconstructions, the
//! per-side decision frames (`frames::SideFrames` — the converter's and the simulator's) and
//! the RoyaleAPI replay types; with the `io` feature (default) also the Parquet readers, the
//! Hugging Face fetch and the gm-shard output of the `crdata` binary (crate docs in main.rs).

#![allow(dead_code)]
pub mod catalog;
pub mod elixir;
#[cfg(feature = "io")]
pub mod fetch;
pub mod frames;
pub mod hand;
#[cfg(all(test, feature = "io"))]
mod leak_audit;
pub mod replay;
