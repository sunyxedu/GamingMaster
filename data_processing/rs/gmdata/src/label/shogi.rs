//! Shogi teacher labels (USI engine, e.g. YaneuraOu NNUE) — extension point, not
//! implemented yet.
//!
//! To implement, replace this file (keep `Args` and `run`): a `ShogiLabeller`
//! implementing [`super::game::Labeller`] that replays each CSA game from
//! tgt.dropped / from_idx / to_idx / piece_idx / drop_to_idx / promoted (POV squares,
//! see `crate::shogi`), re-encodes obs.board_seq / obs.hand_seq / obs.globals_seq /
//! tgt.legal_* for [`super::game::Labeller::verify`], queries the engine through
//! [`super::uci::UciSession`] with `Flavor::Usi` (MultiPV = `--max-moves`,
//! `usinewgame` per position; Q = logistic(cp / scale) with a documented scale →
//! `raw_unit() = "cp"`), and maps the policy onto the ShogiDecoder heads
//! (mode [2], from [81] | move, to [81] | move, from = label, promo [2] | from, to =
//! labels, piece [14] | drop, drop_to [81] | drop, piece = label — see
//! `ShogiDecoder.il_head_terms`). Suggested action id: board moves
//! `(from·81 + to)·2 + promote`, drops `81·81·2 + piece·81 + to`. Then `run` becomes
//! `super::run_labeller(ShogiLabeller::new(&args)?, &args.common)`.

use super::LabelArgs;
use anyhow::Result;

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::not_implemented("shogi", &args.common)
}
