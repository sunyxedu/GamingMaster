//! Othello teacher labels (Egaroucid console) — extension point, not implemented yet.
//!
//! To implement, replace this file (keep `Args` and `run`): an `OthelloLabeller`
//! implementing [`super::game::Labeller`] that replays each WTHOR game from
//! tgt.place_idx / tgt.passed (forced pass frames included, see `crate::othello`),
//! re-encodes obs.board_seq for [`super::game::Labeller::verify`], scores every legal
//! move with Egaroucid (native unit: final disc difference → `raw_unit() = "discs"`,
//! Q from the engine's win probability or a documented logistic of the disc score),
//! skips pass / single-move frames, and maps the policy onto the PlacementDecoder heads
//! (place [64] = p(place | not pass); pass = P(pass) when the adapter has a pass head).
//! Suggested action id: `row·8 + col`, pass = 64. Then `run` becomes
//! `super::run_labeller(OthelloLabeller::new(&args)?, &args.common)`.

use super::LabelArgs;
use anyhow::Result;

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::not_implemented("othello", &args.common)
}
