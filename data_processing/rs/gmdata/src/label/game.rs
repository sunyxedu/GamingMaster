//! The extension point: one [`Labeller`] per game. The driver ([`super::driver`])
//! owns shards, workers, caching, transforms, packing and output; a game supplies
//! replay, verification, the engine conversation and the mapping of the teacher
//! policy onto its decoder heads.
//!
//! Per sample (one worker thread, its own engine session):
//!
//! 1. [`Labeller::replay`] — positions of every frame, reconstructed from the shard
//!    (targets + sample meta / first frame), exactly as the converter walked them.
//! 2. [`Labeller::verify`] — re-encode the observation planes (and any other
//!    replay-derived columns, e.g. legality masks) with the converter's logic and
//!    compare bit-exactly ([`SampleView::check_f32`]). A failing sample is skipped
//!    and counted; the run fails if more than `--max-verify-fail-frac` (0.1%) fail.
//! 3. per selected frame: [`Labeller::skip_reason`] (forced move, terminal, pass
//!    frame, …) → cache lookup by [`Labeller::key`] (a position repeated within the
//!    sample is searched once) → [`Labeller::query`] (engine search → [`Analysis`]:
//!    scored moves as flat action ids with mover-POV Q), or [`Labeller::query_batch`]
//!    over chunks of [`Labeller::batch_size`] frames for engines that analyse many
//!    positions per request.
//! 4. the driver keeps the top `--max-moves` moves, computes the policy (softmax of
//!    Q/τ, or normalised visits), and [`Labeller::head_rows`] turns that policy over
//!    full actions into one row per decoder head, exactly as the decoder's
//!    `il_head_terms` docstring defines the head's soft target (marginals and
//!    conditionals on the ground-truth prefix: [`super::transform::marginal`] /
//!    [`super::transform::conditional`]).
//!
//! Flat action ids (`tgt.teacher_idx`) are per game and must be documented in the
//! game's module docs. Chess (implemented, [`super::chess`]; also recorded in
//! `meta.teacher.game.action_id`): `promotion·4096 + from·64 + to` in the frame's POV
//! squares (the FromToDecoder labels). Suggested for the others: othello / go
//! `row·N + col` with pass = N², shogi board moves `(from·81 + to)·2 + promote` and
//! drops `81·81·2 + piece·81 + to`.

use super::engine::EngineCommand;
use super::transform::PolicyKind;
use anyhow::{bail, Context, Result};
use gmshard::{ShardReader, Tensor};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hash::Hash;
use std::time::Duration;

/// One move the engine scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredMove {
    /// the game's flat action id (≥ 0)
    pub action: i64,
    /// mover-POV expected score in [0, 1]
    pub q: f32,
    /// the engine's native score (cp, discs, score lead, …); NaN when not applicable
    pub raw: f32,
    /// search visits (MCTS engines); NaN when not applicable
    pub visits: f32,
}

impl ScoredMove {
    pub fn new(action: i64, q: f32) -> Self {
        ScoredMove { action, q, raw: f32::NAN, visits: f32::NAN }
    }

    pub fn raw(mut self, raw: f32) -> Self {
        self.raw = raw;
        self
    }

    pub fn visits(mut self, visits: f32) -> Self {
        self.visits = visits;
        self
    }
}

/// What the engine said about one position.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Analysis {
    /// scored moves, any order (the driver sorts by Q — by visits under
    /// `--policy visits` — and keeps the top `--max-moves`)
    pub moves: Vec<ScoredMove>,
    /// mover-POV expected score of the position in [0, 1] (the engine's root
    /// evaluation); `None` = the best scored move's Q
    pub value: Option<f32>,
}

/// A decoder head's soft-target shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    /// `n` classes; rows keep the top `k` classes (stored dense `[n]` when `k ≥ n`,
    /// else sparse `_probs_idx` / `_probs_val` `[k]`)
    Categorical { n: usize, k: usize },
    /// scalar P(class 1) (`_probs` [T] + `_probs_mask` [T])
    Bernoulli,
}

/// One head of the game's decoder (`name` = the `il_head_terms` key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadSpec {
    pub name: &'static str,
    pub kind: HeadKind,
}

/// A head's teacher row for one frame.
#[derive(Debug, Clone, PartialEq)]
pub enum HeadRow {
    /// (class, probability) pairs; normalised by the driver; empty = no teacher
    Categorical(Vec<(i64, f32)>),
    /// P(class 1) in [0, 1]
    Bernoulli(f32),
    /// no teacher for this head on this frame
    Missing,
}

/// `--limit KEY=VALUE` search budgets (interpreted per game).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Limits(pub BTreeMap<String, String>);

impl Limits {
    pub fn parse(items: &[String]) -> Result<Self> {
        let mut m = BTreeMap::new();
        for it in items {
            let (k, v) = it.split_once('=').with_context(|| format!("--limit {it:?} is not KEY=VALUE"))?;
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
        Ok(Limits(m))
    }

    /// `key` as u64 (`default` when absent).
    pub fn u64_or(&self, key: &str, default: u64) -> Result<u64> {
        match self.0.get(key) {
            None => Ok(default),
            Some(v) => v.replace('_', "").parse().with_context(|| format!("--limit {key}={v}: not an integer")),
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Fill defaults and reject keys outside `allowed`.
    pub fn with_defaults(&self, defaults: &[(&str, &str)], allowed: &[&str]) -> Result<Self> {
        for k in self.0.keys() {
            if !allowed.contains(&k.as_str()) {
                bail!("unknown --limit {k:?} (this game accepts {allowed:?})");
            }
        }
        let mut m = self.0.clone();
        for (k, v) in defaults {
            m.entry(k.to_string()).or_insert_with(|| v.to_string());
        }
        Ok(Limits(m))
    }

    pub fn to_json(&self) -> Value {
        Value::Object(self.0.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect())
    }
}

/// Everything a session needs to start and query its engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub command: EngineCommand,
    /// `--engine-option NAME=VALUE` in order (UCI/USI setoption, config overrides, …)
    pub options: Vec<(String, String)>,
    /// resolved budgets ([`Labeller::resolve_limits`])
    pub limits: Limits,
    /// per-command timeout (`--timeout-s`)
    pub timeout: Duration,
    /// moves to score per position (`--max-moves`; e.g. the UCI MultiPV)
    pub max_moves: usize,
}

/// All columns of one shard, loaded once and shared by the workers.
pub struct ShardData {
    pub reader: ShardReader,
    pub tensors: BTreeMap<String, Tensor>,
}

impl ShardData {
    pub fn load(reader: ShardReader) -> Result<Self> {
        let tensors = reader.read_all()?;
        Ok(ShardData { reader, tensors })
    }

    pub fn view(&self, index: usize) -> SampleView<'_> {
        let r = self.reader.sample_range(index);
        SampleView { shard: self, index, start: r.start, len: r.len() }
    }
}

/// One sample of a loaded shard (frames `start..start+len` of every tensor).
#[derive(Clone, Copy)]
pub struct SampleView<'a> {
    pub shard: &'a ShardData,
    pub index: usize,
    pub start: usize,
    pub len: usize,
}

impl<'a> SampleView<'a> {
    /// Per-sample meta from index.json.
    pub fn meta(&self) -> &'a Value {
        &self.shard.reader.samples[self.index].meta
    }

    /// Shard-level converter meta.
    pub fn shard_meta(&self) -> &'a Value {
        self.shard.reader.meta()
    }

    pub fn tensor(&self, name: &str) -> Result<&'a Tensor> {
        self.shard.tensors.get(name).with_context(|| format!("shard has no tensor {name:?}"))
    }

    /// This sample's frames of an integer / bool column as i64.
    pub fn i64s(&self, name: &str) -> Result<Cow<'a, [i64]>> {
        self.tensor(name)?.rows_i64(self.start, self.len)
    }

    /// This sample's frames of an F32 / Bits column as f32 (T × frame elements).
    pub fn f32s(&self, name: &str) -> Result<Cow<'a, [f32]>> {
        self.tensor(name)?.rows_f32(self.start, self.len)
    }

    pub fn bools(&self, name: &str) -> Result<Vec<bool>> {
        self.tensor(name)?.rows_bool(self.start, self.len)
    }

    /// Bit-exact check of re-encoded frames (all T frames of `name`, frame-major; F32
    /// compares `f32::to_bits`, Bits the unpacked {0,1} values). The error names the
    /// first mismatching frame (sample-relative) and element.
    pub fn check_f32(&self, name: &str, expected: &[f32]) -> Result<()> {
        let t = self.tensor(name)?;
        let per = t.frame_len();
        if expected.len() != self.len * per {
            bail!("{name}: re-encoded {} values, the sample has {} frames of {per}", expected.len(), self.len);
        }
        let stored = t.rows_f32(self.start, self.len)?;
        if let Some(i) = stored.iter().zip(expected).position(|(a, b)| a.to_bits() != b.to_bits()) {
            bail!("{name}: frame {} element {}: stored {} but re-encoded {}", i / per, i % per, stored[i], expected[i]);
        }
        Ok(())
    }

    /// Bit-exact check of a re-encoded integer / bool column (all T frames).
    pub fn check_i64(&self, name: &str, expected: &[i64]) -> Result<()> {
        let t = self.tensor(name)?;
        let per = t.frame_len();
        let stored = self.i64s(name)?;
        if expected.len() != stored.len() {
            bail!("{name}: re-encoded {} values, the sample has {}", expected.len(), stored.len());
        }
        if let Some(i) = stored.iter().zip(expected).position(|(a, b)| a != b) {
            bail!("{name}: frame {} element {}: stored {} but re-encoded {}", i / per, i % per, stored[i], expected[i]);
        }
        Ok(())
    }
}

/// A game's labelling logic (see the module docs for the flow).
pub trait Labeller: Send + Sync + 'static {
    /// A replayed position (whatever query / key / head mapping need).
    type Position: Send;
    /// Cache key: the full identity the engine's answer depends on.
    type Key: Hash + Eq + Clone + Send;
    /// A live engine conversation (process + protocol state), one per worker.
    type Session: Send;

    /// The shards' `index.json` "game".
    fn game(&self) -> &'static str;

    /// Labeller version (bump whenever the meaning of a label changes); recorded in
    /// the teacher meta.
    fn version(&self) -> u32;

    /// Reject shards this labeller cannot replay (default: game name mismatch).
    /// Check converter_version / shard meta / required tensors here.
    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        if shard.game != self.game() {
            bail!("{}: shard game {:?}, labeller {:?}", shard.dir.display(), shard.game, self.game());
        }
        Ok(())
    }

    /// The decoder's heads, in the order [`Labeller::head_rows`] returns rows.
    fn heads(&self) -> Vec<HeadSpec>;

    /// Unit of [`ScoredMove::raw`] (`"cp"`, `"discs"`, `"score_lead"`, …) — adds the
    /// `tgt.teacher_raw` column; `None` when Q is the engine's native unit.
    fn raw_unit(&self) -> Option<&'static str> {
        None
    }

    /// Whether [`ScoredMove::visits`] is filled (adds `tgt.teacher_visits`).
    fn has_visits(&self) -> bool {
        false
    }

    /// Policy transform when `--policy` is not given.
    fn default_policy(&self) -> PolicyKind {
        PolicyKind::Softmax
    }

    /// A game-specific teacher policy replacing `--policy` / `--tau` (e.g. othello
    /// `--disc-tau`: a softmax over the engine's native disc scores). Called with the
    /// kept moves (sorted, truncated to `--max-moves`); returns one non-negative weight
    /// per move (normalised by the driver), or `None` for the default transform.
    fn custom_policy(&self, _moves: &[ScoredMove]) -> Option<Vec<f32>> {
        None
    }

    /// The teacher meta `policy` entry when [`Labeller::custom_policy`] is active
    /// (`None` = the driver's `{"kind": "softmax", "tau": …}` / `{"kind": "visits"}`).
    fn custom_policy_meta(&self) -> Option<Value> {
        None
    }

    /// Moves scored per position when `--max-moves` is not given.
    fn default_max_moves(&self) -> usize;

    /// Validate `--limit` and fill the game's defaults (e.g. `nodes=1000000`).
    fn resolve_limits(&self, given: &Limits) -> Result<Limits>;

    /// Start an engine session (spawn + handshake + options).
    fn open_session(&self, cfg: &EngineConfig) -> Result<Self::Session>;

    /// Engine name + version as reported by the session (teacher meta "engine").
    fn engine_id(&self, session: &Self::Session) -> String;

    /// Extra run-level teacher meta (network file name, rule set, score scale, …).
    /// Must be identical for every shard of a run and free of machine-specific paths.
    fn teacher_meta(&self, _cfg: &EngineConfig) -> Value {
        Value::Null
    }

    /// Positions of every frame of the sample (`len` entries).
    fn replay(&self, s: &SampleView) -> Result<Vec<Self::Position>>;

    /// Re-encode the sample from `positions` and compare bit-exactly with the shard.
    fn verify(&self, s: &SampleView, positions: &[Self::Position]) -> Result<()>;

    /// Why frame `t` gets no teacher (e.g. `"forced"`, `"terminal"`, `"pass"`); counted.
    fn skip_reason(&self, _s: &SampleView, _t: usize, _pos: &Self::Position) -> Option<&'static str> {
        None
    }

    fn key(&self, pos: &Self::Position) -> Self::Key;

    /// Ask the engine. Errors (timeouts, crashes, protocol violations) make the driver
    /// restart the session and retry once; a second failure leaves the frame unlabelled.
    fn query(&self, session: &mut Self::Session, pos: &Self::Position, cfg: &EngineConfig) -> Result<Analysis>;

    /// Positions per engine request (default 1 = [`Labeller::query`] per position).
    /// Engines that analyse many positions per request (KataGo `analyzeTurns` over a
    /// whole game) return more and implement [`Labeller::query_batch`]; the driver then
    /// sends a sample's uncached frames in chunks of this size, retrying a failed chunk
    /// once on a fresh session.
    fn batch_size(&self) -> usize {
        1
    }

    /// Analyses of several frames `(t, position)` of one sample, one per frame, in order
    /// (default: [`Labeller::query`] for each).
    fn query_batch(
        &self,
        session: &mut Self::Session,
        _s: &SampleView,
        frames: &[(usize, &Self::Position)],
        cfg: &EngineConfig,
    ) -> Result<Vec<Analysis>> {
        frames.iter().map(|(_, p)| self.query(session, p, cfg)).collect()
    }

    /// One row per head (order of [`Labeller::heads`]) from the teacher `policy` over
    /// full actions (flat id, probability; sums to 1 over the kept moves).
    fn head_rows(&self, s: &SampleView, t: usize, pos: &Self::Position, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_parse_and_defaults() {
        let l = Limits::parse(&["nodes=1_000".into(), " depth = 12 ".into()]).unwrap();
        assert_eq!(l.u64_or("nodes", 5).unwrap(), 1000);
        assert_eq!(l.u64_or("depth", 5).unwrap(), 12);
        assert_eq!(l.u64_or("movetime", 5).unwrap(), 5);
        assert!(Limits::parse(&["nodes".into()]).is_err());
        let r = l.with_defaults(&[("nodes", "9"), ("threads", "1")], &["nodes", "depth", "threads"]).unwrap();
        assert_eq!((r.get("nodes"), r.get("threads")), (Some("1_000"), Some("1")));
        assert!(l.with_defaults(&[], &["nodes"]).is_err());
        assert_eq!(r.to_json()["depth"], "12");
    }
}
