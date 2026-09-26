//! Go teacher labels (KataGo analysis engine) — extension point, not implemented yet.
//!
//! To implement, replace this file (keep `Args` and `run`): a `GoLabeller`
//! implementing [`super::game::Labeller`] that replays each SGF game from
//! tgt.place_idx / tgt.passed plus the setup stones / komi / rules in the sample meta
//! (see `crate::go`: captures, suicide, simple ko), re-encodes obs.board_seq /
//! obs.globals_seq for [`super::game::Labeller::verify`], queries KataGo's JSON
//! analysis engine (`maxVisits` from `--limit visits=N`; Q = winrate + drawrate/2 for
//! the mover, raw = scoreLead → `raw_unit() = "score_lead"`, visits →
//! `has_visits() = true`, `default_policy() = Visits`), and maps the policy onto the
//! PlacementDecoder heads (place [N²] = p(place | not pass), pass = P(pass)). The cache
//! key must include the ko state, komi, rules and side to move. Suggested action id:
//! `row·N + col`, pass = N². Then `run` becomes
//! `super::run_labeller(GoLabeller::new(&args)?, &args.common)`.

use super::LabelArgs;
use anyhow::Result;

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::not_implemented("go", &args.common)
}
