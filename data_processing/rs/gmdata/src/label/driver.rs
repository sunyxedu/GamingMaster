//! The shard-level driver: input directory of gm-shards → output directory of overlay
//! shards (one per input shard, same relative path), with a worker pool, per-worker
//! engine sessions and position caches, resumability, counters and a progress log.
//!
//! * Discovery: every `gm-shard.v1` directory under `--input` (recursive, sorted,
//!   symlinked directories followed, hidden directories and the output tree skipped),
//!   or `--input` itself when it is a shard. `--part i/n` keeps shards `i, i+n, …` so
//!   several jobs can label one corpus into one output tree; `--max-shards` caps the run.
//! * Resume: a shard whose output `index.json` exists is skipped — after checking that
//!   it was labelled with the same teacher meta (a different setting is an error: use a
//!   new `--out`). Outputs are written atomically ([`gmshard::OverlayWriter`]), so an
//!   interrupted run leaves only complete shards (plus `.name.tmp-*` debris, removed when
//!   that shard is redone).
//! * Per frame (see [`super::game`]): replay + verify per sample, frame selection
//!   (`--max-samples` per shard, `--every-nth-frame` with a per-sample phase so both
//!   sides' moves are covered), skip reasons, cache, engine query with one
//!   restart-and-retry, policy transform, head rows.
//! * Frames without a teacher (sampling, skips, engine failures, failed samples, frames
//!   no sample covers) get pad rows: `teacher_idx` −1, `teacher_q/raw/visits` NaN,
//!   sparse `_probs_idx` −1 / `_probs_val` 0, dense `_probs` zeros, bernoulli
//!   `_probs` 0 with `_probs_mask` false, `engine_value` 0 with `engine_value_mask` false.
//! * Health: the run stops before committing a shard when the engine fails
//!   systematically (≥ 20 failed positions and more than half of all queries) or when
//!   more than 10% of a shard's samples fail replay/verification; it fails at the end
//!   (and early, once 10 000 samples / queries are in) when more than
//!   `--max-verify-fail-frac` of the samples or `--max-engine-error-frac` of the
//!   queries failed. Written shards stay valid overlays; delete the output tree before
//!   rerunning after a failure (existing outputs are skipped).
//! * Output: per shard, `index.json` gets `meta.teacher` (run-level settings, identical
//!   across the run — `GameShardDataset` requires uniform shard meta, so label train and
//!   validation data with the same settings) and a top-level `teacher_stats` (this
//!   shard's counters). The run prints a JSON summary on stdout; progress goes to stderr.

use super::cache::PositionCache;
use super::engine::{is_timeout, EngineCommand};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove, ShardData};
use super::pool::Pool;
use super::transform::{clean_row, normalise, softmax_q, value_from_q, DenseRows, Padded, PolicyKind, SparseRows};
use anyhow::{bail, Context, Result};
use gmshard::{LinkPolicy, OverlayWriter, ShardReader};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Bump when the driver's output layout / semantics change (recorded in the teacher meta).
pub const CORE_VERSION: u32 = 1;

/// Error messages kept per shard in `teacher_stats.errors`.
const MAX_NOTES: usize = 8;

/// Everything a run needs besides the labeller (the CLI builds it from its flags).
#[derive(Debug, Clone)]
pub struct RunOpts {
    pub input: PathBuf,
    pub out: PathBuf,
    pub command: EngineCommand,
    pub engine_options: Vec<(String, String)>,
    pub limits: Limits,
    pub workers: usize,
    pub tau: f64,
    pub policy: Option<PolicyKind>,
    pub max_moves: Option<usize>,
    pub max_samples: Option<usize>,
    pub every_nth_frame: usize,
    pub max_shards: Option<usize>,
    pub part: Option<(usize, usize)>,
    pub timeout: Duration,
    pub cache_size: usize,
    pub progress: Duration,
    pub max_verify_fail_frac: f64,
    pub max_engine_error_frac: f64,
    pub link: LinkPolicy,
    /// no progress lines on stderr
    pub quiet: bool,
}

impl RunOpts {
    /// Defaults matching the CLI's.
    pub fn new(input: &Path, out: &Path, command: EngineCommand) -> Self {
        RunOpts {
            input: input.to_path_buf(),
            out: out.to_path_buf(),
            command,
            engine_options: Vec::new(),
            limits: Limits::default(),
            workers: 1,
            tau: 0.05,
            policy: None,
            max_moves: None,
            max_samples: None,
            every_nth_frame: 1,
            max_shards: None,
            part: None,
            timeout: Duration::from_secs(120),
            cache_size: 20_000,
            progress: Duration::from_secs(30),
            max_verify_fail_frac: 0.001,
            max_engine_error_frac: 0.01,
            link: LinkPolicy::Auto,
            quiet: false,
        }
    }
}

/// Counters of a sample / shard / run.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Stats {
    pub samples: u64,
    pub frames: u64,
    /// frames with a teacher
    pub labelled: u64,
    /// frames not selected (--max-samples / --every-nth-frame)
    pub sampled_out: u64,
    /// frames the labeller skipped, by reason
    pub skipped: BTreeMap<String, u64>,
    pub replay_failed: u64,
    pub verify_failed: u64,
    pub cancelled: u64,
    /// positions sent to the engine (cache misses)
    pub queries: u64,
    pub cache_hits: u64,
    /// failed attempts followed by a session restart
    pub retries: u64,
    pub timeouts: u64,
    /// positions left unlabelled after the retry failed
    pub engine_errors: u64,
    /// engine answers the driver rejected (no moves, bad ids / Q, head-row errors)
    pub bad_analysis: u64,
    /// probability mass dropped by head top-k truncation (sum over rows)
    pub truncated_mass: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

impl Stats {
    fn note(&mut self, msg: String) {
        if self.errors.len() < MAX_NOTES {
            self.errors.push(msg);
        }
    }

    pub fn merge(&mut self, o: &Stats) {
        self.samples += o.samples;
        self.frames += o.frames;
        self.labelled += o.labelled;
        self.sampled_out += o.sampled_out;
        for (k, v) in &o.skipped {
            *self.skipped.entry(k.clone()).or_default() += v;
        }
        self.replay_failed += o.replay_failed;
        self.verify_failed += o.verify_failed;
        self.cancelled += o.cancelled;
        self.queries += o.queries;
        self.cache_hits += o.cache_hits;
        self.retries += o.retries;
        self.timeouts += o.timeouts;
        self.engine_errors += o.engine_errors;
        self.bad_analysis += o.bad_analysis;
        self.truncated_mass += o.truncated_mass;
        for e in &o.errors {
            self.note(e.clone());
        }
    }

    pub fn failed_samples(&self) -> u64 {
        self.replay_failed + self.verify_failed
    }
}

/// Result of a run (printed as JSON by the CLI).
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub game: String,
    pub shards_total: usize,
    pub shards_written: usize,
    pub shards_existing: usize,
    pub elapsed_s: f64,
    /// labelled frames per second (wall clock, this run)
    pub positions_per_s: f64,
    /// engine queries per second
    pub searches_per_s: f64,
    pub cache_hit_rate: f64,
    pub workers: usize,
    pub stats: Stats,
    /// the teacher meta written into every shard
    pub teacher: Value,
}

/// A labelled frame.
#[derive(Debug, Clone)]
struct FrameOut {
    moves: Vec<ScoredMove>,
    value: f32,
    heads: Vec<HeadRow>,
}

/// A labelled sample.
struct SampleOut {
    frames: Vec<Option<FrameOut>>,
    stats: Stats,
}

struct Job {
    data: Arc<ShardData>,
    sample: usize,
    phase_key: u64,
}

struct Worker<L: Labeller> {
    lab: Arc<L>,
    cfg: Arc<EngineConfig>,
    heads: Arc<Vec<HeadSpec>>,
    session: Option<L::Session>,
    cache: PositionCache<L::Key, Analysis>,
    policy: PolicyKind,
    tau: f32,
    every: usize,
    max_samples: Option<usize>,
    stop: Arc<AtomicBool>,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// `--every-nth-frame` phase of a sample: frames with (t + phase) % n == 0 are labelled
/// (`shard_rel` = the shard's path relative to `--input`).
pub fn frame_phase(shard_rel: &str, sample: usize, every: usize) -> usize {
    phase_of(fnv1a(shard_rel), sample, every)
}

fn phase_of(phase_key: u64, sample: usize, every: usize) -> usize {
    if every <= 1 {
        return 0;
    }
    (splitmix(phase_key ^ splitmix(sample as u64)) % every as u64) as usize
}

impl<L: Labeller> Worker<L> {
    fn run(&mut self, job: Job) -> SampleOut {
        let v = job.data.view(job.sample);
        let mut out = SampleOut { frames: vec![None; v.len], stats: Stats { samples: 1, frames: v.len as u64, ..Stats::default() } };
        if self.stop.load(Ordering::Relaxed) {
            out.stats.cancelled = 1;
            return out;
        }
        let positions = match self.lab.replay(&v) {
            Ok(p) if p.len() == v.len => p,
            Ok(p) => {
                out.stats.replay_failed = 1;
                out.stats.note(format!("sample {}: replay gave {} positions for {} frames", v.index, p.len(), v.len));
                return out;
            }
            Err(e) => {
                out.stats.replay_failed = 1;
                out.stats.note(format!("sample {}: replay: {e:#}", v.index));
                return out;
            }
        };
        if let Err(e) = self.lab.verify(&v, &positions) {
            out.stats.verify_failed = 1;
            out.stats.note(format!("sample {}: verify: {e:#}", v.index));
            return out;
        }
        let selected = self.max_samples.map(|m| job.sample < m).unwrap_or(true);
        let phase = phase_of(job.phase_key, job.sample, self.every);
        // frames that need an analysis, resolved from the cache where possible; the
        // misses are queried below (a position repeated within the sample only once)
        let mut analyses: Vec<Option<Analysis>> = vec![None; v.len];
        let mut wanted: Vec<usize> = Vec::new();
        let mut misses: Vec<(usize, L::Key)> = Vec::new();
        let mut first_miss: std::collections::HashMap<L::Key, usize> = std::collections::HashMap::new();
        let mut repeats: Vec<(usize, usize)> = Vec::new(); // (frame, frame it repeats)
        for (t, pos) in positions.iter().enumerate() {
            if !selected || !(t + phase).is_multiple_of(self.every) {
                out.stats.sampled_out += 1;
                continue;
            }
            if let Some(r) = self.lab.skip_reason(&v, t, pos) {
                *out.stats.skipped.entry(r.to_string()).or_default() += 1;
                continue;
            }
            wanted.push(t);
            let key = self.lab.key(pos);
            if let Some(a) = self.cache.get(&key) {
                out.stats.cache_hits += 1;
                analyses[t] = Some(a);
            } else if let Some(&t0) = first_miss.get(&key) {
                out.stats.cache_hits += 1;
                repeats.push((t, t0));
            } else {
                first_miss.insert(key.clone(), t);
                misses.push((t, key));
            }
        }
        for chunk in misses.chunks(self.lab.batch_size().max(1)) {
            let frames: Vec<(usize, &L::Position)> = chunk.iter().map(|(t, _)| (*t, &positions[*t])).collect();
            if let Some(got) = self.query(&v, &frames, &mut out.stats) {
                for ((t, key), a) in chunk.iter().zip(got) {
                    self.cache.insert(key.clone(), a.clone());
                    analyses[*t] = Some(a);
                }
            }
        }
        for (t, t0) in repeats {
            analyses[t] = analyses[t0].clone();
        }
        for t in wanted {
            let Some(a) = analyses[t].take() else { continue }; // engine failure (counted)
            match self.finish(&v, t, &positions[t], a) {
                Ok(f) => {
                    out.stats.labelled += 1;
                    out.frames[t] = Some(f);
                }
                Err(e) => {
                    out.stats.bad_analysis += 1;
                    out.stats.note(format!("sample {} frame {t}: {e:#}", v.index));
                }
            }
        }
        out
    }

    /// One engine request (a single position, or a chunk for batching labellers) with
    /// one restart-and-retry; `None` = failed twice (every frame counted as an error).
    fn query(&mut self, v: &SampleView, frames: &[(usize, &L::Position)], stats: &mut Stats) -> Option<Vec<Analysis>> {
        stats.queries += frames.len() as u64;
        let mut last = None;
        for _ in 0..2 {
            if self.session.is_none() {
                match self.lab.open_session(&self.cfg) {
                    Ok(s) => self.session = Some(s),
                    Err(e) => {
                        last = Some(e.context("start engine session"));
                        continue;
                    }
                }
            }
            let session = self.session.as_mut().unwrap();
            let r = if frames.len() == 1 && self.lab.batch_size() <= 1 {
                self.lab.query(session, frames[0].1, &self.cfg).map(|a| vec![a])
            } else {
                self.lab.query_batch(session, v, frames, &self.cfg)
            };
            match r {
                Ok(got) if got.len() == frames.len() => return Some(got),
                Ok(got) => {
                    // a labeller bug, not an engine failure: no retry
                    stats.engine_errors += frames.len() as u64;
                    stats.note(format!("sample {}: query_batch returned {} analyses for {} frames", v.index, got.len(), frames.len()));
                    return None;
                }
                Err(e) => {
                    self.session = None; // drop = kill; reopened on the next attempt
                    stats.retries += 1;
                    if is_timeout(&e) {
                        stats.timeouts += 1;
                    }
                    last = Some(e);
                }
            }
        }
        stats.engine_errors += frames.len() as u64;
        if let Some(e) = last {
            let ts: Vec<usize> = frames.iter().map(|f| f.0).collect();
            stats.note(format!("sample {} frame(s) {ts:?}: engine: {e:#}", v.index));
        }
        None
    }

    /// Analysis → kept moves, policy, value, head rows (validated).
    fn finish(&self, v: &SampleView, t: usize, pos: &L::Position, a: Analysis) -> Result<FrameOut> {
        let mut moves = a.moves;
        if moves.is_empty() {
            bail!("engine scored no move");
        }
        let mut ids = std::collections::HashSet::new();
        for m in &moves {
            if m.action < 0 || !ids.insert(m.action) {
                bail!("bad or repeated action id {}", m.action);
            }
            if !(m.q.is_finite() && (0.0..=1.0).contains(&m.q)) {
                bail!("Q {} of action {} is not in [0, 1]", m.q, m.action);
            }
        }
        match self.policy {
            PolicyKind::Softmax => moves.sort_by(|x, y| y.q.total_cmp(&x.q).then(x.action.cmp(&y.action))),
            PolicyKind::Visits => moves.sort_by(|x, y| {
                let (vx, vy) = (if x.visits.is_nan() { -1.0 } else { x.visits }, if y.visits.is_nan() { -1.0 } else { y.visits });
                vy.total_cmp(&vx).then(y.q.total_cmp(&x.q)).then(x.action.cmp(&y.action))
            }),
        }
        moves.truncate(self.cfg.max_moves);
        let probs = match self.lab.custom_policy(&moves) {
            Some(w) if w.len() != moves.len() => bail!("custom policy gave {} weights for {} moves", w.len(), moves.len()),
            Some(w) => normalise(&w).context("custom policy has no positive weight")?,
            None => match self.policy {
                PolicyKind::Softmax => softmax_q(&moves.iter().map(|m| m.q).collect::<Vec<_>>(), self.tau),
                PolicyKind::Visits => normalise(&moves.iter().map(|m| m.visits).collect::<Vec<_>>()).context("no move has visits")?,
            },
        };
        let value = a.value.unwrap_or_else(|| moves.iter().map(|m| m.q).fold(0.0, f32::max));
        if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
            bail!("position value {value} is not in [0, 1]");
        }
        let policy: Vec<(i64, f32)> = moves.iter().zip(&probs).map(|(m, &p)| (m.action, p)).collect();
        let rows = self.lab.head_rows(v, t, pos, &policy)?;
        if rows.len() != self.heads.len() {
            bail!("head_rows returned {} rows for {} heads", rows.len(), self.heads.len());
        }
        let mut heads = Vec::with_capacity(rows.len());
        for (spec, row) in self.heads.iter().zip(rows) {
            heads.push(match (spec.kind, row) {
                (_, HeadRow::Missing) => HeadRow::Missing,
                (HeadKind::Categorical { n, .. }, HeadRow::Categorical(r)) => {
                    HeadRow::Categorical(clean_row(&r, n).with_context(|| format!("head {}", spec.name))?)
                }
                (HeadKind::Bernoulli, HeadRow::Bernoulli(p)) if p.is_finite() && (0.0..=1.0).contains(&p) => HeadRow::Bernoulli(p),
                (k, r) => bail!("head {}: row {r:?} does not fit {k:?}", spec.name),
            });
        }
        Ok(FrameOut { moves, value: value_from_q(value), heads })
    }
}

/// Column builders of one head.
enum HeadCols {
    Sparse(SparseRows),
    Dense(DenseRows),
    Bern(Vec<f32>, Vec<bool>),
}

fn head_cols(spec: &HeadSpec) -> HeadCols {
    match spec.kind {
        HeadKind::Categorical { n, k } if k >= n => HeadCols::Dense(DenseRows::new(n)),
        HeadKind::Categorical { k, .. } => HeadCols::Sparse(SparseRows::new(k)),
        HeadKind::Bernoulli => HeadCols::Bern(Vec::new(), Vec::new()),
    }
}

/// Column names a run adds (for docs / tests).
pub fn teacher_columns(heads: &[HeadSpec], raw: bool, visits: bool) -> Vec<String> {
    let mut v = vec!["tgt.teacher_idx".to_string(), "tgt.teacher_q".into()];
    if raw {
        v.push("tgt.teacher_raw".into());
    }
    if visits {
        v.push("tgt.teacher_visits".into());
    }
    v.push("tgt.engine_value".into());
    v.push("tgt.engine_value_mask".into());
    for h in heads {
        match h.kind {
            HeadKind::Categorical { n, k } if k >= n => v.push(format!("tgt.{}_probs", h.name)),
            HeadKind::Categorical { .. } => {
                v.push(format!("tgt.{}_probs_idx", h.name));
                v.push(format!("tgt.{}_probs_val", h.name));
            }
            HeadKind::Bernoulli => {
                v.push(format!("tgt.{}_probs", h.name));
                v.push(format!("tgt.{}_probs_mask", h.name));
            }
        }
    }
    v
}

/// Assemble the teacher columns of a shard from its labelled samples.
fn build_overlay(
    reader: &ShardReader,
    outs: &[SampleOut],
    heads: &[HeadSpec],
    k: usize,
    raw: bool,
    visits: bool,
    stats: &mut Stats,
) -> Result<OverlayWriter> {
    let n = reader.n_frames;
    let mut frames: Vec<Option<&FrameOut>> = vec![None; n];
    let mut covered = vec![false; n];
    for (s, out) in reader.samples.iter().zip(outs) {
        for (t, f) in out.frames.iter().enumerate() {
            let at = s.start + t;
            if covered[at] {
                bail!("{}: samples overlap at frame {at}", reader.dir.display());
            }
            covered[at] = true;
            frames[at] = f.as_ref();
        }
    }
    let mut idx = Padded::new(k, -1i64);
    let mut q = Padded::new(k, f32::NAN);
    let mut rawc = Padded::new(k, f32::NAN);
    let mut vis = Padded::new(k, f32::NAN);
    let (mut ev, mut evm) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut cols: Vec<HeadCols> = heads.iter().map(head_cols).collect();
    for f in &frames {
        match f {
            Some(f) => {
                idx.push(f.moves.iter().map(|m| m.action));
                q.push(f.moves.iter().map(|m| m.q));
                rawc.push(f.moves.iter().map(|m| m.raw));
                vis.push(f.moves.iter().map(|m| m.visits));
                ev.push(f.value);
                evm.push(true);
                for (c, row) in cols.iter_mut().zip(&f.heads) {
                    match (c, row) {
                        (HeadCols::Sparse(s), HeadRow::Categorical(r)) => s.push(r),
                        (HeadCols::Dense(d), HeadRow::Categorical(r)) => d.push(r),
                        (HeadCols::Bern(p, m), HeadRow::Bernoulli(x)) => {
                            p.push(*x);
                            m.push(true);
                        }
                        (HeadCols::Sparse(s), _) => s.push_empty(),
                        (HeadCols::Dense(d), _) => d.push_empty(),
                        (HeadCols::Bern(p, m), _) => {
                            p.push(0.0);
                            m.push(false);
                        }
                    }
                }
            }
            None => {
                idx.push_empty();
                q.push_empty();
                rawc.push_empty();
                vis.push_empty();
                ev.push(0.0);
                evm.push(false);
                for c in cols.iter_mut() {
                    match c {
                        HeadCols::Sparse(s) => s.push_empty(),
                        HeadCols::Dense(d) => d.push_empty(),
                        HeadCols::Bern(p, m) => {
                            p.push(0.0);
                            m.push(false);
                        }
                    }
                }
            }
        }
    }
    let mut ov = OverlayWriter::new(reader);
    ov.add_i64("tgt.teacher_idx", &[k], idx.data)?;
    ov.add_f32("tgt.teacher_q", &[k], q.data)?;
    if raw {
        ov.add_f32("tgt.teacher_raw", &[k], rawc.data)?;
    }
    if visits {
        ov.add_f32("tgt.teacher_visits", &[k], vis.data)?;
    }
    ov.add_f32("tgt.engine_value", &[], ev)?;
    ov.add_bool("tgt.engine_value_mask", &[], &evm)?;
    for (h, c) in heads.iter().zip(cols) {
        match c {
            HeadCols::Sparse(s) => {
                stats.truncated_mass += s.truncated_mass;
                ov.add_i64(&format!("tgt.{}_probs_idx", h.name), &[s.k], s.idx)?;
                ov.add_f32(&format!("tgt.{}_probs_val", h.name), &[s.k], s.val)?;
            }
            HeadCols::Dense(d) => {
                ov.add_f32(&format!("tgt.{}_probs", h.name), &[d.n], d.data)?;
            }
            HeadCols::Bern(p, m) => {
                ov.add_f32(&format!("tgt.{}_probs", h.name), &[], p)?;
                ov.add_bool(&format!("tgt.{}_probs_mask", h.name), &[], &m)?;
            }
        }
    }
    Ok(ov)
}

/// Every shard under `input` (see the module docs) as (shard dir, path relative to input).
pub fn find_shards(input: &Path, out: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    if ShardReader::is_shard(input) {
        let name = input.file_name().context("input shard has no directory name")?;
        return Ok(vec![(input.to_path_buf(), PathBuf::from(name))]);
    }
    if !input.is_dir() {
        bail!("--input {} is not a directory", input.display());
    }
    let out_abs = std::fs::canonicalize(out).ok();
    let mut found = Vec::new();
    fn walk(dir: &Path, rel: &Path, depth: usize, out_abs: &Option<PathBuf>, found: &mut Vec<(PathBuf, PathBuf)>) -> Result<()> {
        if depth > 12 {
            return Ok(());
        }
        let mut entries: Vec<_> = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            if out_abs.is_some() && std::fs::canonicalize(&p).ok() == *out_abs {
                continue;
            }
            let r = rel.join(&name);
            if ShardReader::is_shard(&p) {
                found.push((p, r));
            } else {
                walk(&p, &r, depth + 1, out_abs, found)?;
            }
        }
        Ok(())
    }
    walk(input, Path::new(""), 0, &out_abs, &mut found)?;
    Ok(found)
}

/// Path-like option values are reduced to their file name in the teacher meta (the
/// meta must match across machines).
fn meta_value(v: &str) -> String {
    if v.contains('/') || v.contains('\\') {
        Path::new(v).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| v.to_string())
    } else {
        v.to_string()
    }
}

fn rate(n: u64, secs: f64) -> f64 {
    if secs > 0.0 { n as f64 / secs } else { 0.0 }
}

fn hms(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{}h{:02}m{:02}s", s / 3600, s / 60 % 60, s % 60)
}

/// Label every shard under `opts.input` with `lab` (see the module docs).
pub fn run<L: Labeller>(lab: L, opts: &RunOpts) -> Result<Summary> {
    let lab = Arc::new(lab);
    let game = lab.game();
    let heads = lab.heads();
    let mut names = std::collections::HashSet::new();
    for h in &heads {
        if !names.insert(h.name) {
            bail!("head {} declared twice", h.name);
        }
        if let HeadKind::Categorical { n, k } = h.kind {
            if n == 0 || k == 0 {
                bail!("head {}: n and k must be positive", h.name);
            }
        }
    }
    if opts.every_nth_frame == 0 {
        bail!("--every-nth-frame must be ≥ 1");
    }
    let limits = lab.resolve_limits(&opts.limits)?;
    let max_moves = opts.max_moves.unwrap_or_else(|| lab.default_max_moves()).max(1);
    let policy = opts.policy.unwrap_or_else(|| lab.default_policy());
    if policy == PolicyKind::Visits && !lab.has_visits() {
        bail!("--policy visits: the {game} labeller reports no visits");
    }
    let (raw, visits) = (lab.raw_unit().is_some(), lab.has_visits());
    let cfg = Arc::new(EngineConfig {
        command: opts.command.clone(),
        options: opts.engine_options.clone(),
        limits: limits.clone(),
        timeout: opts.timeout,
        max_moves,
    });

    let mut shards = find_shards(&opts.input, &opts.out)?;
    let total_found = shards.len();
    if let Some((i, n)) = opts.part {
        if n == 0 || i >= n {
            bail!("--part {i}/{n}: need 0 ≤ i < n");
        }
        shards = shards.into_iter().enumerate().filter(|(j, _)| j % n == i).map(|(_, s)| s).collect();
    }
    if let Some(m) = opts.max_shards {
        shards.truncate(m);
    }
    if shards.is_empty() {
        bail!("no gm-shards to label under {} ({total_found} found before --part / --max-shards)", opts.input.display());
    }

    // fail fast on a bad engine, and learn its identity for the meta
    let probe = lab.open_session(&cfg).with_context(|| format!("start the engine ({})", cfg.command.describe()))?;
    let engine_id = lab.engine_id(&probe);
    drop(probe);
    let head_meta: serde_json::Map<String, Value> = heads
        .iter()
        .map(|h| {
            let m = match h.kind {
                HeadKind::Categorical { n, k } => {
                    json!({"kind": "categorical", "n": n, "k": k.min(n), "storage": if k >= n { "dense" } else { "sparse" }})
                }
                HeadKind::Bernoulli => json!({"kind": "bernoulli"}),
            };
            (h.name.to_string(), m)
        })
        .collect();
    let policy_meta = lab.custom_policy_meta().unwrap_or_else(|| match policy {
        PolicyKind::Softmax => json!({"kind": "softmax", "tau": opts.tau}),
        PolicyKind::Visits => json!({"kind": "visits"}),
    });
    let teacher = json!({
        "labeller": format!("gmdata label {game}"),
        "labeller_version": lab.version(),
        "core_version": CORE_VERSION,
        "engine": engine_id,
        "engine_options": opts.engine_options.iter().map(|(k, v)| (k.clone(), Value::String(meta_value(v)))).collect::<serde_json::Map<_, _>>(),
        "limits": limits.to_json(),
        "policy": policy_meta,
        "max_moves": max_moves,
        "every_nth_frame": opts.every_nth_frame,
        "max_samples": opts.max_samples,
        "heads": head_meta,
        "raw_unit": lab.raw_unit(),
        "visits": visits,
        "game": lab.teacher_meta(&cfg),
    });

    let stop = Arc::new(AtomicBool::new(false));
    let heads_arc = Arc::new(heads.clone());
    let workers: Vec<_> = (0..opts.workers.max(1))
        .map(|_| {
            let mut w = Worker::<L> {
                lab: lab.clone(),
                cfg: cfg.clone(),
                heads: heads_arc.clone(),
                session: None,
                cache: PositionCache::new(opts.cache_size),
                policy,
                tau: opts.tau as f32,
                every: opts.every_nth_frame,
                max_samples: opts.max_samples,
                stop: stop.clone(),
            };
            move |job: Job| w.run(job)
        })
        .collect();
    let pool = Pool::new(workers);
    let log = |msg: String| {
        if !opts.quiet {
            eprintln!("[label {game}] {msg}");
        }
    };
    log(format!(
        "{} shard(s) to do ({} found), {} worker(s), engine {engine_id:?}, limits {:?}, policy {}, max {max_moves} moves",
        shards.len(),
        total_found,
        pool.workers(),
        limits.0,
        teacher["policy"]
    ));

    let t0 = Instant::now();
    let mut totals = Stats::default();
    let (mut written, mut existing) = (0usize, 0usize);
    let n_shards = shards.len();
    for (si, (src, rel)) in shards.iter().enumerate() {
        let dst = opts.out.join(rel);
        if dst.join("index.json").exists() {
            let have: Value = serde_json::from_slice(&std::fs::read(dst.join("index.json"))?)
                .with_context(|| format!("parse {}", dst.join("index.json").display()))?;
            if have["meta"]["teacher"] != teacher {
                bail!("{} was labelled with different settings (meta.teacher {} vs {}); use a new --out", dst.display(),
                      have["meta"]["teacher"], teacher);
            }
            existing += 1;
            continue;
        }
        if dst.exists() {
            bail!("{} exists but has no index.json; remove it", dst.display());
        }
        if let (Some(parent), Some(name)) = (dst.parent(), dst.file_name()) {
            let prefix = format!(".{}.tmp-", name.to_string_lossy());
            if let Ok(rd) = std::fs::read_dir(parent) {
                for e in rd.flatten() {
                    if e.file_name().to_string_lossy().starts_with(&prefix) {
                        let _ = std::fs::remove_dir_all(e.path());
                    }
                }
            }
        }
        let ts = Instant::now();
        let reader = ShardReader::open(src)?;
        lab.check_shard(&reader)?;
        let data = Arc::new(ShardData::load(reader)?);
        let n_samples = data.reader.n_samples();
        let rel_s = rel.to_string_lossy().to_string();
        let phase_key = fnv1a(&rel_s);
        let jobs: Vec<Job> = (0..n_samples).map(|j| Job { data: data.clone(), sample: j, phase_key }).collect();
        let mut shard = Stats::default();
        let mut last_log = Instant::now();
        let outs = pool.map(jobs, |done, r: &SampleOut| {
            shard.merge(&r.stats);
            let (q, e) = (totals.queries + shard.queries, totals.engine_errors + shard.engine_errors);
            if e >= 20 && e * 2 > q {
                stop.store(true, Ordering::Relaxed);
            }
            if last_log.elapsed() >= opts.progress {
                last_log = Instant::now();
                let secs = t0.elapsed().as_secs_f64();
                let hits = totals.cache_hits + shard.cache_hits;
                log(format!(
                    "shard {}/{n_shards} {rel_s}: samples {done}/{n_samples} | {:.0} pos/s, {:.0} searches/s | cache {:.1}% | engine errors {e} | verify failed {} | {}",
                    si + 1,
                    rate(totals.labelled + shard.labelled, secs),
                    rate(q, secs),
                    100.0 * hits as f64 / (hits + q).max(1) as f64,
                    totals.failed_samples() + shard.failed_samples(),
                    hms(secs)
                ));
            }
        });
        if stop.load(Ordering::Relaxed) {
            let mut all = totals.clone();
            all.merge(&shard);
            bail!("engine failing systematically ({} of {} positions failed); last errors: {:#?}", all.engine_errors, all.queries, all.errors);
        }
        let failed = shard.failed_samples();
        if failed >= 3 && failed as f64 > 0.1 * n_samples as f64 {
            bail!("{rel_s}: {failed} of {n_samples} samples failed replay/verification (systematic?); first errors: {:#?}", shard.errors);
        }
        let mut ov = build_overlay(&data.reader, &outs, &heads, max_moves, raw, visits, &mut shard)?;
        let secs = ts.elapsed().as_secs_f64();
        ov.set_teacher(teacher.clone());
        let mut stats_json = serde_json::to_value(&shard)?;
        stats_json["elapsed_s"] = json!((secs * 1000.0).round() / 1000.0);
        stats_json["workers"] = json!(pool.workers());
        ov.set_field("teacher_stats", stats_json)?;
        let mode = ov.link_policy(opts.link).commit(&dst)?;
        written += 1;
        totals.merge(&shard);
        log(format!(
            "wrote {} ({n_samples} samples, {}/{} frames labelled, files {}) in {secs:.1}s | run: {:.0} pos/s, cache {:.1}%",
            dst.display(),
            shard.labelled,
            shard.frames,
            mode.as_str(),
            rate(totals.labelled, t0.elapsed().as_secs_f64()),
            100.0 * totals.cache_hits as f64 / (totals.cache_hits + totals.queries).max(1) as f64,
        ));
        check_health(&totals, opts, false)?;
    }
    check_health(&totals, opts, true)?;
    let secs = t0.elapsed().as_secs_f64();
    Ok(Summary {
        game: game.to_string(),
        shards_total: n_shards,
        shards_written: written,
        shards_existing: existing,
        elapsed_s: (secs * 1000.0).round() / 1000.0,
        positions_per_s: rate(totals.labelled, secs),
        searches_per_s: rate(totals.queries, secs),
        cache_hit_rate: totals.cache_hits as f64 / (totals.cache_hits + totals.queries).max(1) as f64,
        workers: pool.workers(),
        stats: totals,
        teacher,
    })
}

/// Run-level failure thresholds (see the module docs).
fn check_health(s: &Stats, opts: &RunOpts, last: bool) -> Result<()> {
    let failed = s.failed_samples();
    if (last || s.samples >= 10_000) && failed as f64 > opts.max_verify_fail_frac * s.samples as f64 {
        bail!("{failed} of {} samples failed replay/verification (> {:.3}%); first errors: {:#?}", s.samples,
              100.0 * opts.max_verify_fail_frac, s.errors);
    }
    if (last || s.queries >= 10_000) && s.engine_errors as f64 > opts.max_engine_error_frac * s.queries as f64 {
        bail!("{} of {} engine queries failed (> {:.2}%); first errors: {:#?}", s.engine_errors, s.queries,
              100.0 * opts.max_engine_error_frac, s.errors);
    }
    Ok(())
}
