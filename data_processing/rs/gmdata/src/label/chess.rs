//! Chess teacher labels (UCI engine, e.g. Stockfish) — extension point, not implemented yet.
//!
//! To implement, replace this file (keep `Args` and `run`): a `ChessLabeller`
//! implementing [`super::game::Labeller`] that replays each game from
//! tgt.from_idx / to_idx / promotion (POV squares, castling = king to g/c file, see
//! `crate::chess`), re-encodes obs.board_seq / obs.globals_seq / tgt.legal_* with the
//! converter's logic for [`super::game::Labeller::verify`], queries the engine through
//! [`super::uci::UciSession`] (defaults `Threads=1`, a small `Hash`, overridable by
//! `--engine-option`; MultiPV = `--max-moves`, `UCI_ShowWDL`, `ucinewgame` per position
//! for reproducibility), and maps the policy onto the FromToDecoder heads
//! (from [64] marginal, to [64] | from = label, promotion [5] | from, to = labels).
//! Suggested action id: `(from·64 + to)·5 + promotion`. Then `run` becomes
//! `super::run_labeller(ChessLabeller::new(&args)?, &args.common)`.

use super::LabelArgs;
use anyhow::Result;

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::not_implemented("chess", &args.common)
}
