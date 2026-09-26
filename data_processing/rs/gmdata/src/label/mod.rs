//! `gmdata label <game>` — engine "teacher" labels for existing gm-shards.
//!
//! ```text
//! scripts/engines/fetch_engines.sh stockfish        # engines live outside the repo
//! gmdata label chess --input data/chess/train --out data/chess_sf/train \
//!     --engine ~/gm-engines/bin/stockfish --workers 32 --limit nodes=1000000 --max-moves 32
//! python -m training.train_il_game --game chess --data_dir data/chess_sf/train \
//!     --soft_target_weight 1.0 --engine_value_weight 0.5 ...
//! ```
//!
//! For every input shard an overlay shard is written under `--out` (same relative path):
//! the base tensor files are hard-linked (symlink / copy fallback; the base is never
//! modified), teacher columns are added, and `index.json` gains `meta.teacher` (engine,
//! limits, transform, labeller versions) and `teacher_stats` (per-shard counters). The
//! Python loader (`data_processing/games/dataset.py`) and trainer
//! (`training/train_il_game.py`, `--soft_target_weight` / `--engine_value_weight`) read
//! them unchanged. Train and validation data must be labelled with the same settings
//! (the loader requires identical shard meta across a directory and between them).
//!
//! Teacher columns (`[T, …]`, T = the shard's frames; K = `--max-moves`):
//!
//! ```text
//! tgt.teacher_idx        [T,K] i64   flat action ids of the scored moves, best first (by the
//!                                    policy's score: softmax Q/τ + w·u, i.e. by Q and among
//!                                    equal Q by u; visits), −1 pad (encoding documented per game)
//! tgt.teacher_q          [T,K] f32   mover-POV expected score W + D/2 ∈ [0,1] per move, NaN pad
//! tgt.teacher_raw        [T,K] f32   engine-native score (cp, discs, score lead), NaN pad —
//!                                    games whose native unit is not Q (meta.teacher.raw_unit)
//! tgt.teacher_visits     [T,K] f32   search visits, NaN pad — MCTS engines
//! tgt.teacher_prior      [T,K] f32   the engine network's raw policy prior per move, NaN
//!                                    pad — engines that report it (KataGo; meta.teacher.prior)
//! tgt.engine_value       [T]   f32   2·Q(position) − 1 ∈ [−1,1] (0 where unlabelled)
//! tgt.engine_value_mask  [T]   bool  frame has a teacher
//! tgt.<h>_probs_idx/_val [T,k]       sparse soft target of categorical head h (k < n classes):
//!                                    i64 class ids (−1 pad) + f32 probabilities (0 pad)
//! tgt.<h>_probs          [T,n] f32   dense soft target of categorical head h (k ≥ n)
//! tgt.<h>_probs          [T]   f32   bernoulli head h: P(class 1), with
//! tgt.<h>_probs_mask     [T]   bool  its presence
//! ```
//!
//! Heads and their meaning follow the decoder's `il_head_terms` docstring (marginals,
//! conditionals on the ground-truth prefix; an all-zero row = no teacher). The policy
//! is `π ∝ exp(Q/τ + w·u)` over the scored moves (`--tau`, default 0.05) or normalised
//! visits (`--policy visits`). `w·u` is the secondary tie-break (`--tie-weight` w, default
//! 2.0; details in [`transform`]): Q saturates in decided positions (every move mates or
//! wins: Q = 1; exact othello solves: Q ∈ {0, ½, 1}), where `exp(Q/τ)` alone is uniform —
//! a move that allows mate counts as much as the best defence. u ∈ [−1, 1] is the game's
//! secondary utility of the engine's native score (`tgt.teacher_raw`: chess / shogi mate
//! distance and centipawns, othello discs; [`game::Labeller::secondary`]), increasing
//! like Q, so moves are ordered by Q and, among equal Q, by u; within a class of equal Q
//! the preference ratio is at most e^(2w). w = 0, and games without a secondary utility
//! (go), give `π ∝ exp(Q/τ)` exactly. The meta records it (`policy.tie_weight`,
//! `policy.secondary`). The raw columns allow re-deriving π later without relabelling
//! ([`transform::softmax_q_tie`] with the game's `secondary` of `teacher_raw`).
//! Frames without a teacher (sampling, forced moves, engine failures, samples failing
//! replay verification) carry pad rows and mask false.
//!
//! Engines are external processes (never linked or vendored), one per worker
//! (`--workers`, each single-threaded with a small hash — set via `--engine-option`),
//! long-lived and restarted after a crash / timeout, with a per-worker position cache
//! (`--cache-size`). Progress (positions/s, cache hit rate, errors) goes to stderr;
//! a JSON summary to stdout. Runs are resumable (existing outputs are skipped) and
//! splittable across jobs (`--part i/n`).
//!
//! Module map: [`game`] (the per-game [`game::Labeller`] trait — the extension point),
//! [`driver`] (shards, workers, output), [`engine`] (processes), [`uci`] (UCI/USI),
//! [`transform`] (Q, policies, row packing), [`pool`], [`cache`], [`fake`] (fake
//! engines), [`toy`] (a worked example game). Adding a game: implement `Labeller` in
//! `label/<game>.rs`, put its flags in that file's `Args`, and call [`run_labeller`]
//! from its `run`.
//!
//! Self-test without any engine binary (the gmdata binary doubles as a fake UCI engine):
//!
//! ```text
//! gmdata label selftest --out /tmp/label-selftest     # toy shards → labelled overlays
//! ```

// Shared API for the per-game labellers: parts of it are only used once they land.
#![allow(dead_code)]

pub mod cache;
pub mod chess;
pub mod driver;
pub mod engine;
pub mod fake;
pub mod game;
pub mod go;
pub mod othello;
pub mod pool;
pub mod shogi;
#[cfg(test)]
pub mod testkit;
pub mod toy;
pub mod transform;
pub mod uci;

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use driver::RunOpts;
use engine::EngineCommand;
use game::{Labeller, Limits};
use gmshard::LinkPolicy;
use std::path::PathBuf;
use std::time::Duration;
use transform::PolicyKind;

/// Flags shared by every `gmdata label <game>`.
#[derive(Args, Debug, Clone)]
pub struct LabelArgs {
    /// directory of gm-shards (searched recursively) or one shard directory
    #[arg(long)]
    pub input: PathBuf,
    /// output root: one overlay shard per input shard, same relative path
    #[arg(long)]
    pub out: PathBuf,
    /// engine executable (see scripts/engines/fetch_engines.sh)
    #[arg(long)]
    pub engine: PathBuf,
    /// extra engine command-line argument (repeatable, in order)
    #[arg(long = "engine-arg", allow_hyphen_values = true)]
    pub engine_args: Vec<String>,
    /// engine option NAME=VALUE (repeatable; UCI/USI setoption, config override, …)
    #[arg(long = "engine-option")]
    pub engine_options: Vec<String>,
    /// worker threads, one engine process each
    #[arg(long, default_value_t = default_workers())]
    pub workers: usize,
    /// search budget KEY=VALUE (repeatable; per game, e.g. nodes=1000000, visits=800, level=21)
    #[arg(long = "limit")]
    pub limits: Vec<String>,
    /// softmax temperature over the expected score Q ∈ [0,1]
    #[arg(long, default_value_t = 0.05)]
    pub tau: f64,
    /// weight w of the secondary tie-break π ∝ exp(Q/τ + w·u), u ∈ [−1, 1] = the game's
    /// utility of the engine's native score (mate distance / cp, discs), which orders the
    /// moves of decided positions where Q saturates [default 2.0 for chess, shogi, othello;
    /// 0 = off: π ∝ exp(Q/τ)]
    #[arg(long)]
    pub tie_weight: Option<f64>,
    /// teacher policy (default: the game's — softmax for alpha-beta engines, visits for KataGo)
    #[arg(long, value_enum)]
    pub policy: Option<PolicyKind>,
    /// moves scored per position (MultiPV / top visits); the width K of the teacher columns
    #[arg(long)]
    pub max_moves: Option<usize>,
    /// label only the first N samples of each shard (the rest get no teacher)
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// label every n-th frame (per-sample phase, so both sides are covered)
    #[arg(long, default_value_t = 1)]
    pub every_nth_frame: usize,
    /// stop after N shards
    #[arg(long)]
    pub max_shards: Option<usize>,
    /// I/N: label only shards I, I+N, I+2N, … (split one corpus over N jobs)
    #[arg(long, value_parser = parse_part)]
    pub part: Option<(usize, usize)>,
    /// timeout of one engine command (seconds)
    #[arg(long, default_value_t = 120.0)]
    pub timeout_s: f64,
    /// cached positions per worker (0 = off)
    #[arg(long, default_value_t = 20_000)]
    pub cache_size: usize,
    /// seconds between progress lines
    #[arg(long, default_value_t = 30.0)]
    pub progress_s: f64,
    /// fail the run if more than this fraction of samples fail replay verification
    #[arg(long, default_value_t = 0.001)]
    pub max_verify_fail_frac: f64,
    /// fail the run if more than this fraction of engine queries fail
    #[arg(long, default_value_t = 0.01)]
    pub max_engine_error_frac: f64,
    /// how base tensor files enter the overlay
    #[arg(long, value_enum, default_value_t = LinkArg::Auto)]
    pub link: LinkArg,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkArg {
    /// hard link, else symlink, else copy
    Auto,
    Symlink,
    Copy,
}

fn default_workers() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn parse_part(s: &str) -> Result<(usize, usize), String> {
    let (i, n) = s.split_once('/').ok_or_else(|| format!("{s:?} is not I/N"))?;
    let (i, n) = (i.trim().parse::<usize>().map_err(|e| e.to_string())?, n.trim().parse::<usize>().map_err(|e| e.to_string())?);
    if n == 0 || i >= n {
        return Err(format!("{s}: need 0 <= I < N"));
    }
    Ok((i, n))
}

impl LabelArgs {
    /// Driver options (engine = `--engine` + `--engine-arg`).
    pub fn to_opts(&self) -> Result<RunOpts> {
        if !self.engine.is_file() {
            bail!("--engine {} does not exist (install engines with scripts/engines/fetch_engines.sh)", self.engine.display());
        }
        let mut o = RunOpts::new(&self.input, &self.out, EngineCommand::process(&self.engine, &self.engine_args));
        o.engine_options = self
            .engine_options
            .iter()
            .map(|kv| kv.split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).with_context(|| format!("--engine-option {kv:?} is not NAME=VALUE")))
            .collect::<Result<_>>()?;
        o.limits = Limits::parse(&self.limits)?;
        o.workers = self.workers.max(1);
        o.tau = self.tau;
        o.tie_weight = self.tie_weight;
        o.policy = self.policy;
        o.max_moves = self.max_moves;
        o.max_samples = self.max_samples;
        o.every_nth_frame = self.every_nth_frame;
        o.max_shards = self.max_shards;
        o.part = self.part;
        o.timeout = Duration::from_secs_f64(self.timeout_s.max(0.001));
        o.cache_size = self.cache_size;
        o.progress = Duration::from_secs_f64(self.progress_s.max(0.0));
        o.max_verify_fail_frac = self.max_verify_fail_frac;
        o.max_engine_error_frac = self.max_engine_error_frac;
        o.link = match self.link {
            LinkArg::Auto => LinkPolicy::Auto,
            LinkArg::Symlink => LinkPolicy::Symlink,
            LinkArg::Copy => LinkPolicy::Copy,
        };
        Ok(o)
    }
}

/// Run `lab` with the CLI flags and print the JSON summary (what each game's `run` calls).
pub fn run_labeller<L: Labeller>(lab: L, args: &LabelArgs) -> Result<()> {
    let summary = driver::run(lab, &args.to_opts()?)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn not_implemented(game: &str, _args: &LabelArgs) -> Result<()> {
    bail!("gmdata label {game} is not implemented yet (extension point: data_processing/rs/gmdata/src/label/{game}.rs)")
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Chess shards via a UCI engine (Stockfish)
    Chess(chess::Args),
    /// Othello shards via Egaroucid
    Othello(othello::Args),
    /// Go shards via KataGo
    Go(go::Args),
    /// Shogi shards via a USI engine (YaneuraOu NNUE)
    Shogi(shogi::Args),
    /// Label toy shards with the built-in fake engine (no engine binary needed)
    #[command(hide = true)]
    Selftest {
        /// empty output directory (gets base/ and labelled/)
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 2)]
        workers: usize,
        /// run the fake engine in-process instead of as a child process
        #[arg(long)]
        in_process: bool,
    },
    /// Serve a fake engine on stdin/stdout (tests)
    #[command(hide = true)]
    FakeEngine {
        #[command(subcommand)]
        kind: FakeKind,
    },
}

#[derive(Subcommand)]
pub enum FakeKind {
    /// the toy game's UCI engine
    Toy,
    /// a chess UCI engine scoring every legal move (label/chess.rs `fake_engine`)
    Chess,
    /// rule-table engine (see label/fake.rs for the script format)
    Canned {
        #[arg(long)]
        script: PathBuf,
    },
    /// an Egaroucid-console look-alike for othello (label/othello.rs; engine flags ignored)
    Othello {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        engine_flags: Vec<String>,
    },
    /// KataGo analysis-engine (JSON) fake (see label/go/fake.rs)
    Katago {
        /// exit on the n-th analysis query
        #[arg(long)]
        crash_on: Option<usize>,
        /// KataGo's own arguments (`analysis -config C -model M -override-config K=V,…`);
        /// only reportAnalysisWinratesAs is read
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        katago_args: Vec<String>,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Chess(a) => chess::run(a),
        Cmd::Othello(a) => othello::run(a),
        Cmd::Go(a) => go::run(a),
        Cmd::Shogi(a) => shogi::run(a),
        Cmd::Selftest { out, workers, in_process } => {
            if out.exists() && std::fs::read_dir(&out)?.next().is_some() {
                bail!("--out {} is not empty", out.display());
            }
            toy::write_shards(&out.join("base"), 2, 6, 7)?;
            let cmd = if in_process {
                toy::fake_engine_command()
            } else {
                let exe = std::env::current_exe().context("locate the gmdata binary")?;
                EngineCommand::process(exe, &["label".into(), "fake-engine".into(), "toy".into()])
            };
            let mut o = RunOpts::new(&out.join("base"), &out.join("labelled"), cmd);
            o.workers = workers.max(1);
            o.progress = Duration::from_secs(1);
            let summary = driver::run(toy::ToyLabeller::default(), &o)?;
            println!("{}", serde_json::to_string(&summary)?);
            Ok(())
        }
        Cmd::FakeEngine { kind } => match kind {
            FakeKind::Toy => fake::run_stdio(&mut toy::fake_engine()),
            FakeKind::Chess => fake::run_stdio(&mut chess::fake_engine()),
            FakeKind::Canned { script } => {
                let text = std::fs::read_to_string(&script).with_context(|| format!("read {}", script.display()))?;
                fake::run_stdio(&mut fake::Canned::parse(&text)?)
            }
            FakeKind::Othello { .. } => fake::run_stdio(&mut othello::FakeEgaroucid::new()),
            FakeKind::Katago { crash_on, katago_args } => {
                let mut f = go::fake::KataFake::from_katago_args(&katago_args).map_err(anyhow::Error::msg)?;
                if let Some(n) = crash_on {
                    f = f.crash_on(n);
                }
                fake::run_stdio(&mut f)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        cmd: Cmd,
    }

    #[test]
    fn cli_parses_common_flags() {
        let cli = Cli::try_parse_from([
            "gmdata", "chess", "--input", "in", "--out", "out", "--engine", "/bin/sh", "--engine-arg", "-x",
            "--engine-option", "Hash=16", "--limit", "nodes=1000", "--workers", "3", "--tau", "0.1", "--policy",
            "visits", "--max-moves", "8", "--every-nth-frame", "2", "--part", "1/4", "--link", "copy", "--tie-weight", "0.5",
        ])
        .unwrap();
        let Cmd::Chess(a) = cli.cmd else { panic!() };
        let o = a.common.to_opts().unwrap();
        assert_eq!((o.workers, o.tau, o.max_moves, o.every_nth_frame, o.part), (3, 0.1, Some(8), 2, Some((1, 4))));
        assert_eq!(o.tie_weight, Some(0.5));
        // othello --disc-tau replaces the softmax: it takes no --tau / --tie-weight
        let oth = |extra: &[&str]| {
            let mut v = vec!["gmdata", "othello", "--input", "i", "--out", "o", "--engine", "/bin/sh"];
            v.extend_from_slice(extra);
            Cli::try_parse_from(v).map(|c| match c.cmd { Cmd::Othello(a) => a, _ => unreachable!() })
        };
        assert_eq!(oth(&["--tie-weight", "0"]).unwrap().common.tie_weight, Some(0.0));
        assert_eq!(oth(&[]).unwrap().common.tie_weight, None, "default: the driver's 2.0 where it applies");
        assert!(oth(&["--disc-tau", "2", "--tie-weight", "1"]).is_err() && oth(&["--disc-tau", "2", "--tau", "0.1"]).is_err());
        assert!(oth(&["--disc-tau", "2"]).is_ok());
        assert_eq!(o.engine_options, vec![("Hash".to_string(), "16".to_string())]);
        assert_eq!(o.limits.get("nodes"), Some("1000"));
        assert_eq!((o.policy, o.link), (Some(PolicyKind::Visits), LinkPolicy::Copy));
        assert_eq!(o.command.describe(), "/bin/sh -x");
        assert!(Cli::try_parse_from(["gmdata", "go", "--input", "i", "--out", "o", "--engine", "e", "--part", "4/4"]).is_err());
        let Cmd::Go(g) = Cli::try_parse_from(["gmdata", "go", "--input", "i", "--out", "o", "--engine", "/nonexistent"]).unwrap().cmd
        else { panic!() };
        assert!(g.common.to_opts().is_err(), "missing engine binary");
        assert!(go::run(Cli::try_parse_from(["gmdata", "go", "--input", "i", "--out", "o", "--engine", "/bin/sh"])
            .map(|c| match c.cmd { Cmd::Go(a) => a, _ => unreachable!() }).unwrap()).is_err());
    }

    #[test]
    fn fake_katago_takes_katago_arguments() {
        // the go labeller appends `-override-config …` to whatever --engine-arg gives
        let cli = Cli::try_parse_from([
            "gmdata", "fake-engine", "katago", "--crash-on", "3", "analysis", "-config", "c.cfg", "-override-config",
            "reportAnalysisWinratesAs=SIDETOMOVE",
        ])
        .unwrap();
        let Cmd::FakeEngine { kind: FakeKind::Katago { crash_on, katago_args } } = cli.cmd else { panic!() };
        assert_eq!(crash_on, Some(3));
        assert_eq!(katago_args, ["analysis", "-config", "c.cfg", "-override-config", "reportAnalysisWinratesAs=SIDETOMOVE"]);
        let cli = Cli::try_parse_from(["gmdata", "fake-engine", "katago", "-override-config", "a=1"]).unwrap();
        let Cmd::FakeEngine { kind: FakeKind::Katago { katago_args, .. } } = cli.cmd else { panic!() };
        assert_eq!(katago_args, ["-override-config", "a=1"]);
    }
}
