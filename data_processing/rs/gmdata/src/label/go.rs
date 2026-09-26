//! `gmdata label go` — KataGo teacher labels for go shards (converter v2, `crate::go`).
//!
//! ```text
//! scripts/engines/fetch_engines.sh katago     # engine + kata1-b18c384nbt network, outside the repo
//! gmdata label go --input data/go/train --out data/go_kata/train --engine ~/gm-engines/bin/katago
//!     # = --limit visits=100 --max-moves 16 --workers 1 --analysis-threads 16 --policy visits
//! python -m training.train_il_game --game go --data_dir data/go_kata/train \
//!     --soft_target_weight 1.0 --engine_value_weight 0.5 ...
//! ```
//!
//! Label train and validation shards with the same settings and KataGo installation:
//! the engine id (`KataGo <version> <network>`) is part of `meta.teacher`, and the loader
//! refuses to mix metas (Homebrew's macOS KataGo is 1.18.2; fetch_engines.sh pins every
//! Linux build — CPU, OpenCL, CUDA, TensorRT — to 1.18.1).
//!
//! Without `--engine-arg`, the engine runs as
//! `katago analysis -config <generated> -model <--model>`; the generated config (a
//! content-addressed file in the temp dir, rewritten if missing) sets
//! `reportAnalysisWinratesAs = SIDETOMOVE`, `wideRootNoise = 0.0`,
//! `ignorePreRootHistory = false`, `analysisPVLen = 1` and the thread / batch / cache
//! flags below. `--engine-option K=V` becomes `-override-config K=V,…` (any KataGo
//! config key, e.g. `wideRootNoise=0.04`); with explicit `--engine-arg`s the command is
//! exactly `--engine` + those args + `-override-config` (an `-override-config` among the
//! args is extended). The override always carries `reportAnalysisWinratesAs=SIDETOMOVE`
//! (KataGo has no per-query switch, and its analysis_example.cfg reports BLACK, which would
//! flip every white-to-move Q / score lead / engine value), and every new process must pass
//! a perspective probe (komi 150: black to move ≈ 0, white to move ≈ 1) before it labels.
//! An `--engine-option` key KataGo reports as unused at startup (a typo) stops the run
//! (the driver records the options in `meta.teacher`); `--perf-option` takes only speed /
//! device / logging keys (`katago::is_perf_key`), which stay out of the meta.
//!
//! # What is labelled
//!
//! One KataGo query per game (`analyzeTurns` = every labelled, uncached turn;
//! `--turns-per-query` caps it): `initialStones` = the frame-0 stones (handicap / setup),
//! `initialPlayer` = the frame-0 mover, `moves` = the recorded moves with their SGF
//! colours, `rules` = `--area-rules` (default `chinese`) when obs.globals_seq[3] (area
//! scoring) is set, else `--territory-rules` (default `japanese`) — both are KataGo's
//! simple-ko, no-suicide rule sets, like the converter — and `komi` = the sample meta's
//! komi (rounded to a multiple of ½; when the record has none: `--komi-area` 7.5 /
//! `--komi-territory` 6.5, the usual values of those rule sets), `maxVisits` =
//! `--limit visits=N` (default 100).
//!
//! Columns (K = `--max-moves`, default 16; see `label/mod.rs` for the common ones):
//!
//! ```text
//! tgt.teacher_idx      [T,K] i64  searched moves, most visited first: row·N + col, pass = N²
//!                                 (row 0 = the top = SGF row a = GTP row N; col 0 = GTP A)
//! tgt.teacher_visits   [T,K] f32  moveInfos[].edgeVisits (root visits into the move)
//! tgt.teacher_q        [T,K] f32  moveInfos[].winrate, mover POV (KataGo counts draws ½)
//! tgt.teacher_raw      [T,K] f32  moveInfos[].scoreLead, points, mover POV (raw_unit score_lead)
//! tgt.teacher_prior    [T,K] f32  moveInfos[].prior: the raw network policy
//! tgt.engine_value     [T]   f32  2·rootInfo.winrate − 1 (mover POV)
//! tgt.place_probs_idx/_val [T,K]  p(place | not pass) (PlacementDecoder `place` head)
//! tgt.pass_probs       [T]   f32  P(pass) + tgt.pass_probs_mask [T] bool (`pass` head)
//! ```
//!
//! The policy is the AlphaZero target — normalised visits over the kept moves
//! (`--policy visits`, the default here; `--policy softmax --tau τ` uses the Q column
//! instead). Its `place` row is the policy conditioned on not passing and `pass` is its
//! pass mass, exactly the `PlacementDecoder.il_head_terms` soft targets; a policy that
//! is all pass leaves `place` without a teacher. Symmetric moves (KataGo's root
//! symmetry pruning) carry copied visits and so share the mass evenly.
//!
//! Frames without a teacher (counted in `teacher_stats.skipped`): `turn_order` — the
//! recorded mover is not the opponent of the previous move's colour (e.g. a move after
//! the same colour's pass), where KataGo would search for the other side;
//! `after_double_pass` — every frame after two consecutive passes (the game is over for
//! KataGo). Handicap placement moves (consecutive black moves) are `turn_order` frames.
//!
//! # Replay verification
//!
//! Positions are rebuilt from the shard alone: frame 0's own / opponent planes plus its
//! mover (obs.globals_seq[0]) give the absolute stones, then tgt.place_idx / tgt.passed
//! are played with the converter's rules (`crate::go::Board`: captures, suicide, simple
//! ko) and the colours of obs.globals_seq[0]. Every frame is re-encoded with the
//! converter's own encoders (planes + all six globals, from the sample meta's komi /
//! handicap and the area flag) and compared bit-exactly; each move KataGo reports is
//! checked to be legal in the replayed position (a coordinate mix-up fails loudly).
//!
//! # Cache
//!
//! Per worker, keyed by stones, ko point, mover, komi, rules, the last 5 moves (KataGo's
//! network sees them — `ignorePreRootHistory = false` — and they carry the pass state)
//! and the handicap-stone count (KataGo's `whiteHandicapBonus` under Chinese rules).
//! Hits are mostly shared openings.
//!
//! # Defaults and throughput
//!
//! * `--limit visits=100`: b18c384nbt with 100 visits plays far above the strongest
//!   records in the corpus (KGS 6d+; its most visited move matches theirs ~45% of the
//!   time), and 100 visits is where the visit distribution says something beyond the
//!   top move. Cost is linear in visits.
//! * `--max-moves 16`: measured on 646 KGS positions at 100 visits (`--max-moves 64`),
//!   KataGo reports 6.2 searched moves per position (p90 10, max 21); the moves beyond
//!   the 16 most visited hold 0.07% of the visits (beyond 8: 1.3%). The 16 × 5 teacher
//!   columns cost ≈ 0.6 KB per frame next to the base shard's ≈ 0.2 KB.
//! * `--workers` = KataGo processes, each loading the network (default 1 when the flag
//!   is not given — not the one-per-core default of the single-threaded engines), and
//!   `--analysis-threads 16` = positions each process searches in parallel (one search
//!   thread each, `numSearchThreadsPerAnalysisThread = 1`: parallel positions batch
//!   the network as well as parallel threads on one tree, without the virtual-loss
//!   weakening of multi-threaded search at low visits). GPUs: 1–2 processes with 16–64
//!   analysis threads; CPUs (Eigen): one
//!   process per 16–32 cores with `--analysis-threads` = those cores.
//! * Measured on an Apple M4 Pro (Homebrew KataGo 1.18.2, Metal, b18c384nbt, 100
//!   visits, 4 KGS 19×19 games = 909 positions, wall clock including start-up):
//!   ```text
//!   --workers 1 --analysis-threads 16 (defaults; Metal mux)        6.9 positions/s
//!   --workers 1 --analysis-threads 32                              6.4 positions/s
//!   --workers 2 --analysis-threads 16                              6.6 positions/s
//!   GPU only (--perf-option numNNServerThreadsPerModel=1), 32      3.4 positions/s
//!   ```
//!   ≈ 700 visits/s: the machine's network throughput (`katago benchmark`: ~550
//!   evaluations/s mux, ~200 GPU only; KataGo's evaluation cache serves the rest,
//!   neighbouring turns sharing subtrees). A 256-game shard takes ~2 h; the whole KGS
//!   corpus (~31k games, ~6.5M positions) ~11 days. With `--every-nth-frame 8` a whole
//!   shard ran at 5 positions/s (fewer turns per query: less cache reuse and a larger
//!   idle tail per query; `--workers 2` overlaps queries, untested for this case).
//! * Expected on the cluster (not measured here): an EPYC 9355 node without a GPU
//!   (32 Zen 5 cores, Eigen AVX2 build; b18c384nbt ≈ 10 GFLOP per evaluation, ~4–5
//!   evaluations/s per core) ≈ 150 evaluations/s ≈ 1.5–2 positions/s at 100 visits —
//!   too slow for the full corpus (≈ 40 node-days): on CPU nodes label a subsample
//!   (`--every-nth-frame 4 --limit visits=50` ≈ 5 node-days). One datacenter GPU
//!   (CUDA / TensorRT build, several thousand evaluations/s for b18c384nbt) ≈ 30–60
//!   positions/s, i.e. the corpus in ~1.5–2.5 GPU-days (`--workers 2
//!   --analysis-threads 32` per GPU; several GPUs: one run per GPU with `--part i/n`
//!   and `CUDA_VISIBLE_DEVICES`).
//!
//! `--timeout-s` is per round of parallel searches: a query of n turns may take
//! `timeout · ⌈n / --analysis-threads⌉`. `--startup-timeout-s` bounds the handshake
//! (network load; OpenCL tunes itself on the first run, which can take minutes).

pub mod fake;
pub mod katago;
#[cfg(test)]
mod tests;

use super::engine::EngineCommand;
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove};
use super::transform::{conditional, PolicyKind};
use super::LabelArgs;
use crate::go::{encode_globals, encode_planes, globals_consts, Board, BLACK, EMPTY, GLOBAL_FEATURES, WHITE};
use anyhow::{bail, Context, Result};
use gmshard::ShardReader;
use katago::{KataSession, QueryGame};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Bump when the meaning of a go label changes.
pub const LABELLER_VERSION: u32 = 1;
/// The shards this labeller replays (`crate::go::CONVERTER_VERSION`).
pub const CONVERTER_VERSION: u32 = 2;
/// Moves of history in the cache key (KataGo's network input).
pub const HISTORY: usize = 5;
pub const DEFAULT_MAX_MOVES: usize = 16;
pub const DEFAULT_VISITS: u64 = 100;
/// KataGo processes when `--workers` is not given on the command line.
pub const DEFAULT_PROCESSES: usize = 1;
const SKIP_TURN_ORDER: &str = "turn_order";
const SKIP_DOUBLE_PASS: &str = "after_double_pass";

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
    /// KataGo network (.bin.gz) [default: ${GM_ENGINES:-$HOME/gm-engines}/nets/katago/default.bin.gz]
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// board size of the shards (the converter's --board-size)
    #[arg(long, default_value_t = 19)]
    pub board_size: usize,
    /// KataGo rules for area-scoring records (a shorthand or a JSON rules object)
    #[arg(long, default_value = "chinese")]
    pub area_rules: String,
    /// KataGo rules for territory-scoring / unknown-rules records
    #[arg(long, default_value = "japanese")]
    pub territory_rules: String,
    /// komi of area-scoring records without KM
    #[arg(long, default_value_t = 7.5)]
    pub komi_area: f64,
    /// komi of territory-scoring records without KM
    #[arg(long, default_value_t = 6.5)]
    pub komi_territory: f64,
    /// positions each KataGo process searches in parallel (numAnalysisThreads)
    #[arg(long, default_value_t = 16)]
    pub analysis_threads: usize,
    /// search threads per position (numSearchThreadsPerAnalysisThread)
    #[arg(long, default_value_t = 1)]
    pub search_threads: usize,
    /// nnMaxBatchSize (0 = analysis threads × search threads)
    #[arg(long, default_value_t = 0)]
    pub nn_batch: usize,
    /// nnCacheSizePowerOfTwo (network evaluations cached per process)
    #[arg(long, default_value_t = 18)]
    pub nn_cache_pow2: u32,
    /// speed-only KataGo config KEY=VALUE (device placement, batching, logging; repeatable;
    /// other keys are refused — they go through --engine-option): written into the
    /// generated config, not recorded in the teacher meta. macOS arm64 default:
    /// KataGo's Metal "mux" layout (2 GPU + 2 Neural Engine server threads) unless
    /// numNNServerThreadsPerModel is given
    #[arg(long = "perf-option")]
    pub perf_options: Vec<String>,
    /// turns per KataGo query (0 = a whole game per query)
    #[arg(long, default_value_t = 0)]
    pub turns_per_query: usize,
    /// seconds to wait for KataGo to load its network (the handshake)
    #[arg(long, default_value_t = 600.0)]
    pub startup_timeout_s: f64,
}

/// `${GM_ENGINES:-$HOME/gm-engines}/nets/katago/default.bin.gz`.
pub fn default_model() -> Option<PathBuf> {
    let root = std::env::var_os("GM_ENGINES")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("gm-engines")))?;
    Some(root.join("nets").join("katago").join("default.bin.gz"))
}

fn rules_value(s: &str) -> Result<Value> {
    let s = s.trim();
    if s.starts_with('{') {
        serde_json::from_str(s).with_context(|| format!("rules {s:?} is not a JSON object"))
    } else if s.is_empty() {
        bail!("empty rules")
    } else {
        Ok(Value::String(s.to_string()))
    }
}

fn file_name(p: &Path) -> Option<String> {
    let resolved = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    resolved.file_name().map(|n| n.to_string_lossy().to_string())
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// KataGo komi: a multiple of ½ within ±150.
fn katago_komi(k: f64) -> f64 {
    ((k * 2.0).round() / 2.0).clamp(-150.0, 150.0)
}

/// A replayed game (shared by its positions).
#[derive(Debug)]
pub struct GoGame {
    pub q: QueryGame,
    pub area: bool,
    /// black stones placed before white's first move (initial stones, then leading black moves)
    pub initial_black: usize,
    pub leading_black: usize,
}

/// The position before move `t` of a sample.
#[derive(Debug)]
pub struct GoPos {
    pub t: usize,
    /// absolute colours (row 0 = top) with the simple-ko point
    pub board: Board,
    pub mover: u8,
    pub skip: Option<&'static str>,
    pub game: Arc<GoGame>,
}

/// Cache key: everything KataGo's answer depends on (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GoKey {
    stones: Vec<u8>,
    ko: Option<(i32, i32)>,
    mover: u8,
    komi2: i64,
    rules: String,
    history: Vec<(u8, i64)>,
    handicap: usize,
}

/// The go labeller (see the module docs).
#[derive(Debug, Clone)]
pub struct GoLabeller {
    pub n: usize,
    pub max_moves: usize,
    pub area_rules: Value,
    pub territory_rules: Value,
    pub komi_area: f64,
    pub komi_territory: f64,
    pub turns_per_query: usize,
    pub analysis_threads: usize,
    pub startup_timeout: Duration,
    /// the generated KataGo config (path, text), written by `open_session` when missing
    pub config: Option<(PathBuf, String)>,
    /// (key, value, semantic) of the generated config
    pub settings: Vec<(String, String, bool)>,
    /// network file name (engine id fallback when KataGo lacks `query_models`)
    pub model_name: Option<String>,
}

impl Default for GoLabeller {
    fn default() -> Self {
        GoLabeller {
            n: 19,
            max_moves: DEFAULT_MAX_MOVES,
            area_rules: json!("chinese"),
            territory_rules: json!("japanese"),
            komi_area: 7.5,
            komi_territory: 6.5,
            turns_per_query: 0,
            analysis_threads: 16,
            startup_timeout: Duration::from_secs(600),
            config: None,
            settings: katago::config_settings(16, 1, 16, 18),
            model_name: None,
        }
    }
}

impl GoLabeller {
    pub fn new(args: &Args) -> Result<Self> {
        if !(2..=25).contains(&args.board_size) {
            bail!("--board-size {} outside 2..=25 (GTP coordinates)", args.board_size);
        }
        for (flag, k) in [("--komi-area", args.komi_area), ("--komi-territory", args.komi_territory)] {
            if !k.is_finite() || (k * 2.0).fract() != 0.0 || k.abs() > 150.0 {
                bail!("{flag} {k}: KataGo needs an integer or half-integer komi within ±150");
            }
        }
        let (at, st) = (args.analysis_threads.max(1), args.search_threads.max(1));
        let mut settings = katago::config_settings(at, st, if args.nn_batch == 0 { at * st } else { args.nn_batch }, args.nn_cache_pow2);
        let mut perf: Vec<(String, String)> = Vec::new();
        for kv in &args.perf_options {
            let (k, v) = kv.split_once('=').with_context(|| format!("--perf-option {kv:?} is not KEY=VALUE"))?;
            if settings.iter().any(|s| s.2 && s.0 == k.trim()) || !katago::is_perf_key(k.trim()) {
                bail!("--perf-option {kv}: {} is not a KataGo speed / device / logging setting and may change the teacher; \
                       pass it as --engine-option (recorded in the teacher meta)", k.trim());
            }
            perf.push((k.trim().to_string(), v.trim().to_string()));
        }
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) && !perf.iter().any(|(k, _)| k == "numNNServerThreadsPerModel") {
            perf.extend(katago::METAL_MUX.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        }
        katago::apply_perf(&mut settings, &perf);
        let text = katago::config_text(&settings);
        let auto = args.common.engine_args.is_empty();
        let config = auto.then(|| (std::env::temp_dir().join(format!("gmdata-katago-{:016x}.cfg", fnv1a(&text))), text));
        let model_name = if auto {
            args.model.clone().or_else(default_model).as_deref().and_then(file_name)
        } else {
            let a = &args.common.engine_args;
            a.iter().position(|x| x == "-model").and_then(|i| a.get(i + 1)).and_then(|m| file_name(Path::new(m)))
        };
        let n = args.board_size;
        Ok(GoLabeller {
            n,
            max_moves: args.common.max_moves.unwrap_or(DEFAULT_MAX_MOVES).max(1),
            area_rules: rules_value(&args.area_rules)?,
            territory_rules: rules_value(&args.territory_rules)?,
            komi_area: args.komi_area,
            komi_territory: args.komi_territory,
            turns_per_query: args.turns_per_query,
            analysis_threads: at,
            startup_timeout: Duration::from_secs_f64(args.startup_timeout_s.max(1.0)),
            config,
            settings,
            model_name,
        })
    }

    fn nn(&self) -> usize {
        self.n * self.n
    }

    /// Engine answer for frame `pos` → the driver's [`Analysis`] (validated against the
    /// replayed position).
    fn to_analysis(&self, pos: &GoPos, r: katago::TurnResult) -> Result<Analysis> {
        if r.current_player != pos.mover {
            bail!("turn {}: KataGo searched for {} but the record's mover is {}", pos.t,
                  katago::colour_str(r.current_player), katago::colour_str(pos.mover));
        }
        let (n, nn) = (self.n as i32, self.nn() as i64);
        let mut moves = Vec::with_capacity(r.moves.len());
        for m in r.moves {
            if m.action != nn {
                let p = ((m.action / n as i64) as i32, (m.action % n as i64) as i32);
                if pos.board.at(p) != EMPTY || !pos.board.clone().play(pos.mover, Some(p)) {
                    bail!("turn {}: KataGo move {} is illegal in the replayed position (coordinate mismatch?)",
                          pos.t, katago::gtp_of(m.action as usize, self.n));
                }
            }
            if !(m.prior.is_nan() || (0.0..=1.0).contains(&m.prior)) || !m.visits.is_finite() || m.visits < 0.0 {
                bail!("turn {}: bad prior / visits in {m:?}", pos.t);
            }
            moves.push(ScoredMove::new(m.action, m.winrate.clamp(0.0, 1.0)).raw(m.score_lead).visits(m.visits).prior(m.prior));
        }
        Ok(Analysis { moves, value: Some(r.root_winrate.clamp(0.0, 1.0)) })
    }

    fn analyse(&self, session: &mut KataSession, frames: &[(usize, &GoPos)], cfg: &EngineConfig) -> Result<Vec<Analysis>> {
        let Some((_, first)) = frames.first() else { return Ok(Vec::new()) };
        let game = &first.game;
        let turns: Vec<usize> = frames.iter().map(|(t, _)| *t).collect();
        let visits = cfg.limits.u64_or("visits", DEFAULT_VISITS)?;
        let rounds = turns.len().div_ceil(self.analysis_threads.max(1)).max(1) as u32;
        let results = session.analyze(&game.q, &turns, visits, cfg.timeout.saturating_mul(rounds))?;
        frames.iter().zip(results).map(|((_, pos), r)| self.to_analysis(pos, r)).collect()
    }

    fn write_config(&self) -> Result<()> {
        let Some((path, text)) = &self.config else { return Ok(()) };
        if std::fs::read_to_string(path).ok().as_deref() == Some(text.as_str()) {
            return Ok(());
        }
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("write {}", path.display()))
    }
}

impl Labeller for GoLabeller {
    type Position = GoPos;
    type Key = GoKey;
    type Session = KataSession;

    fn game(&self) -> &'static str {
        "go"
    }

    fn version(&self) -> u32 {
        LABELLER_VERSION
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        let d = shard.dir.display();
        if shard.game != "go" || shard.converter_version != CONVERTER_VERSION {
            bail!("{d}: not a go v{CONVERTER_VERSION} shard (game {:?}, converter v{})", shard.game, shard.converter_version);
        }
        let size = shard.meta()["board_size"].as_u64();
        if size != Some(self.n as u64) {
            bail!("{d}: board_size {size:?}, labelling --board-size {}", self.n);
        }
        let n = self.n;
        for (name, shape) in [("obs.board_seq", vec![4, n, n]), ("obs.globals_seq", vec![GLOBAL_FEATURES]),
                              ("tgt.place_idx", vec![]), ("tgt.passed", vec![])] {
            let info = shard.tensor_info(name)?;
            if info.frame_shape != shape {
                bail!("{d}: {name} frame shape {:?}, expected {shape:?}", info.frame_shape);
            }
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        vec![
            HeadSpec { name: "place", kind: HeadKind::Categorical { n: self.nn(), k: self.max_moves.min(self.nn()) } },
            HeadSpec { name: "pass", kind: HeadKind::Bernoulli },
        ]
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("score_lead")
    }

    fn has_visits(&self) -> bool {
        true
    }

    fn has_prior(&self) -> bool {
        true
    }

    fn default_policy(&self) -> PolicyKind {
        PolicyKind::Visits
    }

    fn default_max_moves(&self) -> usize {
        DEFAULT_MAX_MOVES
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        let l = given.with_defaults(&[("visits", "100")], &["visits"])?;
        if l.u64_or("visits", DEFAULT_VISITS)? < 2 {
            bail!("--limit visits must be ≥ 2 (with one visit KataGo searches no move)");
        }
        Ok(l)
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<KataSession> {
        self.write_config()?;
        let cmd = match &cfg.command {
            // --engine-option + the forced side-to-move perspective (whatever config the
            // process runs with, e.g. KataGo's analysis_example.cfg says BLACK)
            EngineCommand::Process { program, args, cwd, env } => EngineCommand::Process {
                program: program.clone(),
                args: katago::override_args(args, &cfg.options)?,
                cwd: cwd.clone(),
                env: env.clone(),
            },
            c => c.clone(),
        };
        let mut session = KataSession::open(cmd, self.startup_timeout.max(cfg.timeout), cfg.timeout)?;
        session.check_side_to_move(self.n, cfg.timeout)?;
        // an --engine-option KataGo does not use (a typo, or a key of another version /
        // backend) would sit in meta.teacher without having changed anything
        let unused: Vec<String> =
            session.unused_config_keys().into_iter().filter(|k| cfg.options.iter().any(|(o, _)| o == k)).collect();
        if !unused.is_empty() {
            bail!("KataGo {} ignores --engine-option key(s) {unused:?} (\"Unused key\" at startup): a typo, or unknown to this version / backend",
                  session.version);
        }
        Ok(session)
    }

    fn engine_id(&self, session: &KataSession) -> String {
        session.engine_id(self.model_name.as_deref())
    }

    fn teacher_meta(&self, _cfg: &EngineConfig) -> Value {
        let semantic: serde_json::Map<String, Value> =
            self.settings.iter().filter(|s| s.2).map(|(k, v, _)| (k.clone(), Value::String(v.clone()))).collect();
        json!({
            "protocol": "katago analysis engine (JSON)",
            "action_id": "row*N+col (row 0 = top = SGF row a = GTP row N; col 0 = GTP column A); pass = N*N",
            "board_size": self.n,
            "rules": {"area": self.area_rules, "territory": self.territory_rules},
            "komi_when_unknown": {"area": self.komi_area, "territory": self.komi_territory},
            "katago_config": if self.config.is_some() { Value::Object(semantic) } else { json!("from --engine-arg") },
            "q": "moveInfos[].winrate, side to move (draws count 1/2)",
            "raw": "moveInfos[].scoreLead (points, side to move)",
            "visits": "moveInfos[].edgeVisits",
            "prior": "moveInfos[].prior",
            "value": "rootInfo.winrate",
            "heads": {"place": "p(place | not pass)", "pass": "P(pass)"},
            "skipped": [SKIP_TURN_ORDER, SKIP_DOUBLE_PASS],
            "cache_key": format!("stones, ko, mover, komi, rules, last {HISTORY} moves, handicap stones"),
        })
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<GoPos>> {
        let (n, nn) = (self.n, self.nn());
        let planes = s.f32s("obs.board_seq")?;
        let gl = s.f32s("obs.globals_seq")?;
        let place = s.i64s("tgt.place_idx")?;
        let passed = s.bools("tgt.passed")?;
        let meta = s.meta();
        let komi = meta.get("komi").and_then(Value::as_f64);
        if meta.get("handicap").and_then(Value::as_i64).is_none() {
            bail!("sample meta has no handicap");
        }
        // a {0, 1} flag of obs.globals_seq (exactly, as the converter writes it)
        let flag = |t: usize, j: usize| -> Result<bool> {
            let x = gl.get(t * GLOBAL_FEATURES + j).copied().unwrap_or(f32::NAN);
            if x.to_bits() == 1f32.to_bits() {
                Ok(true)
            } else if x.to_bits() == 0f32.to_bits() {
                Ok(false)
            } else {
                bail!("frame {t}: obs.globals_seq[{j}] = {x} is not a {{0, 1}} flag")
            }
        };
        let mover = |t: usize| -> Result<u8> { Ok(if flag(t, 0)? { BLACK } else { WHITE }) };
        let area = flag(0, 3)?;
        // frame 0: absolute stones from the mover-relative planes
        let c0 = mover(0)?;
        let mut board = Board::new(n as i32);
        let mut initial = Vec::new();
        for i in 0..nn {
            let (own, opp) = (planes[i] == 1.0, planes[nn + i] == 1.0);
            let c = match (own, opp) {
                (true, true) => bail!("frame 0 point {i}: stone of both colours"),
                (true, false) => c0,
                (false, true) => crate::go::opponent(c0),
                (false, false) => continue,
            };
            board.set(((i / n) as i32, (i % n) as i32), c);
            initial.push((c, i));
        }
        let kos: Vec<usize> = (0..nn).filter(|&i| planes[3 * nn + i] == 1.0).collect();
        match kos.as_slice() {
            [] => {}
            [k] => board.ko = Some(((k / n) as i32, (k % n) as i32)),
            _ => bail!("frame 0 has {} ko points", kos.len()),
        }
        let mut moves = Vec::with_capacity(s.len);
        for t in 0..s.len {
            let p = if passed[t] {
                if place[t] != 0 {
                    bail!("frame {t}: pass with place_idx {}", place[t]);
                }
                None
            } else if (0..nn as i64).contains(&place[t]) {
                Some(place[t] as usize)
            } else {
                bail!("frame {t}: place_idx {} outside the board", place[t]);
            };
            moves.push((mover(t)?, p));
        }
        let initial_black = if initial.iter().all(|&(c, _)| c == BLACK) { initial.len() } else { 0 };
        let leading_black = moves.iter().take_while(|m| m.0 == BLACK).count();
        let komi_eff = katago_komi(komi.filter(|k| k.is_finite()).unwrap_or(if area { self.komi_area } else { self.komi_territory }));
        let game = Arc::new(GoGame {
            q: QueryGame {
                n,
                initial,
                initial_player: c0,
                moves: moves.clone(),
                rules: if area { self.area_rules.clone() } else { self.territory_rules.clone() },
                komi: komi_eff,
            },
            area,
            initial_black,
            leading_black,
        });
        let double_pass = (1..moves.len()).find(|&j| moves[j].1.is_none() && moves[j - 1].1.is_none());
        let mut out = Vec::with_capacity(s.len);
        for (t, &(c, p)) in moves.iter().enumerate() {
            let skip = if double_pass.is_some_and(|j| t > j) {
                Some(SKIP_DOUBLE_PASS)
            } else if t > 0 && moves[t - 1].0 == c {
                Some(SKIP_TURN_ORDER)
            } else {
                None
            };
            out.push(GoPos { t, board: board.clone(), mover: c, skip, game: game.clone() });
            if !board.play(c, p.map(|i| ((i / n) as i32, (i % n) as i32))) {
                bail!("frame {t}: recorded move {} by {} is illegal", p.map(|i| katago::gtp_of(i, n)).unwrap_or("pass".into()),
                      katago::colour_str(c));
            }
        }
        Ok(out)
    }

    fn verify(&self, s: &SampleView, positions: &[GoPos]) -> Result<()> {
        let nn = self.nn();
        let meta = s.meta();
        let Some(first) = positions.first() else { return Ok(()) };
        let consts = globals_consts(meta.get("komi").and_then(Value::as_f64), meta["handicap"].as_i64().unwrap_or(0), first.game.area);
        let mut planes = vec![0f32; positions.len() * 4 * nn];
        let mut gl = vec![0f32; positions.len() * GLOBAL_FEATURES];
        let moves = &first.game.q.moves;
        for (t, pos) in positions.iter().enumerate() {
            encode_planes(&pos.board, pos.mover, &mut planes[t * 4 * nn..(t + 1) * 4 * nn]);
            encode_globals(pos.mover, consts, t > 0 && moves[t - 1].1.is_none(), &mut gl[t * GLOBAL_FEATURES..(t + 1) * GLOBAL_FEATURES]);
        }
        s.check_f32("obs.board_seq", &planes)?;
        s.check_f32("obs.globals_seq", &gl)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &GoPos) -> Option<&'static str> {
        pos.skip
    }

    fn key(&self, pos: &GoPos) -> GoKey {
        let g = &pos.game;
        let nn = self.nn() as i64;
        GoKey {
            stones: pos.board.g.clone(),
            ko: pos.board.ko,
            mover: pos.mover,
            komi2: (g.q.komi * 2.0) as i64,
            rules: g.q.rules.to_string(),
            history: g.q.moves[pos.t.saturating_sub(HISTORY)..pos.t].iter().map(|&(c, p)| (c, p.map(|i| i as i64).unwrap_or(nn))).collect(),
            handicap: g.initial_black + g.leading_black.min(pos.t),
        }
    }

    fn query(&self, session: &mut KataSession, pos: &GoPos, cfg: &EngineConfig) -> Result<Analysis> {
        self.analyse(session, &[(pos.t, pos)], cfg)?.pop().context("no analysis")
    }

    fn batch_size(&self) -> usize {
        if self.turns_per_query == 0 {
            1 << 20
        } else {
            self.turns_per_query
        }
    }

    fn query_batch(&self, session: &mut KataSession, _s: &SampleView, frames: &[(usize, &GoPos)], cfg: &EngineConfig) -> Result<Vec<Analysis>> {
        self.analyse(session, frames, cfg)
    }

    fn head_rows(&self, _s: &SampleView, _t: usize, _pos: &GoPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        let pass = self.nn() as i64;
        let p_pass: f32 = policy.iter().filter(|(a, _)| *a == pass).map(|(_, p)| *p).sum();
        Ok(vec![
            HeadRow::Categorical(conditional(policy, |a| *a != pass, |a| *a)),
            HeadRow::Bernoulli(p_pass.clamp(0.0, 1.0)),
        ])
    }
}

pub fn run(mut args: Args) -> Result<()> {
    let lab = GoLabeller::new(&args)?;
    if let Some((path, _)) = &lab.config {
        let model = args.model.clone().or_else(default_model).context("no --model and no $HOME for the default network")?;
        if !model.is_file() {
            bail!("--model {} not found (install it with scripts/engines/fetch_engines.sh katago)", model.display());
        }
        args.common.engine_args =
            vec!["analysis".into(), "-config".into(), path.display().to_string(), "-model".into(), model.display().to_string()];
    }
    // the shared --workers default (one per core) suits one-thread engines; one KataGo
    // process already searches --analysis-threads positions in parallel
    let given = std::env::args().any(|a| a == "--workers" || a.starts_with("--workers="));
    if !given {
        args.common.workers = DEFAULT_PROCESSES;
    } else if args.common.workers > 4 {
        eprintln!("[label go] note: --workers {} starts {} KataGo processes, each loading the network; \
                   on one GPU use 1-2 and raise --analysis-threads instead", args.common.workers, args.common.workers);
    }
    super::run_labeller(lab, &args.common)
}
