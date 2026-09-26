//! Candidate-action games (gmrules): trajectories → gm-shards. The generic half of
//! every gmrules converter — a game module (e.g. `kuhn_poker.rs`) parses its source
//! format into [`Record`]s (a gmrules trajectory + provenance) and calls [`convert`].
//!
//! Every record is REPLAYED by the game's rules (gmrules::replay: every chance
//! outcome, seat and action verified; optional check of the recorded payoffs) and its
//! frames are encoded by `gmrules::encode_frame`, the encoder the simulator (gmsim)
//! uses — IL frames and RL frames are the same code.
//!
//! One sample per stream (`gmrules::Replay::streams`): per-seat layout — one per seat
//! (optionally only the seats in `Record::seats`), interleaved — one per record.
//!
//! ```text
//! obs.entities    [T, E, Fe] f32 (bits when spec.bits_entities)
//! obs.entity_mask [T, E] bool
//! obs.entity_type [T, E] i32          (only when spec.entity_types > 0)
//! obs.globals     [T, G] f32 (bits when spec.bits_globals)
//! obs.actions     [T, A, Fa] f32 (bits when spec.bits_actions)
//! obs.action_mask [T, A] bool         the legal candidates (ascending canonical ids)
//! tgt.action_idx  [T] i64             index of the taken action among the candidates
//! tgt.value       [T] f32             outcome of the frame's seat (Game::outcome, [-1, 1])
//! tgt.value_mask  [T] bool            only with allow_truncated (false on unfinished records)
//! tgt.action_probs_idx [T, K] i64     only with soft_k = K > 0: teacher distributions
//! tgt.action_probs_val [T, K] f32     (Record::soft_targets) over candidate indices
//! ```
//!
//! Shard meta: the frame spec (`FrameSpec::schema_meta`: the model's META_ARCH_KEYS),
//! `schema: "candidate.v1"`, `rules_version`, the game `config`; converter_version =
//! `Game::VERSION`. Sample meta: `record`, `seat` (−1 = interleaved), the per-seat
//! `outcome` / `returns`, the record's own meta, and `events` (the trajectory as
//! `[seat, id]` pairs, seat −1 = chance) with `store_events`.

use anyhow::Result;
use gmrules::registry::{self, DynReplay};
use gmrules::{replay, Decision, Event, FrameSpec, Game, Layout, Seat};
use gmshard::{Sample, ShardWriter};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::Path;

pub const SCHEMA: &str = "candidate.v1";

/// One source record as a gmrules trajectory.
#[derive(Clone, Debug, Default)]
pub struct Record {
    /// Provenance id (file:line, hand number, ...).
    pub id: String,
    pub events: Vec<Event>,
    /// Free-form sample meta (player names, ratings, ...).
    pub meta: Value,
    /// Per-seat layout: emit only these seats' streams (high-quality filters, e.g. the
    /// strong player's decisions only); None = every seat. Ignored when interleaved.
    pub seats: Option<Vec<Seat>>,
    /// Payoffs the source reports: the record is rejected when the replayed
    /// `Game::returns` differ (a desynchronised or corrupt record).
    pub expected_returns: Option<Vec<f64>>,
    /// Teacher distributions (engine / solver labels): `(event index of the Act,
    /// [(canonical action id, probability)])`, written as the soft targets
    /// `tgt.action_probs_idx` / `action_probs_val` (candidate indices) when
    /// `ConvertOptions::soft_k > 0`; decisions without an entry get no teacher.
    pub soft_targets: Vec<(usize, Vec<(u32, f32)>)>,
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub shard_size: usize,
    /// Write each sample's trajectory into its meta (golden tests, debugging; costs
    /// index.json size).
    pub store_events: bool,
    /// Keep records that stop before the end (value_mask false on their frames).
    pub allow_truncated: bool,
    /// K > 0: write the records' teacher distributions as `tgt.action_probs_idx` [T, K]
    /// i64 (candidate indices, −1 = pad) / `tgt.action_probs_val` [T, K] f32 — the
    /// most probable K legal actions of each labelled decision.
    pub soft_k: usize,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions { shard_size: 1024, store_events: false, allow_truncated: false, soft_k: 0 }
    }
}

/// Conversion counters (printed as JSON by the CLI).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub records: usize,
    pub converted: usize,
    /// the parser could not read the record
    pub unparseable: usize,
    pub illegal: usize,
    pub truncated: usize,
    pub returns_mismatch: usize,
    pub empty: usize,
    pub samples: usize,
    pub frames: usize,
    pub forced_frames: usize,
    /// labelled decisions (soft targets written)
    pub soft_frames: usize,
    /// teacher probability mass on actions that are not legal candidates (dropped)
    pub soft_illegal_mass: f64,
    pub shards: usize,
    /// The first rejection reasons (for the log).
    pub errors: Vec<String>,
}

#[derive(Debug)]
enum Reject {
    Illegal(String),
    Truncated,
    Returns(String),
    Empty,
}

/// `index.json` meta of a gmrules game's shards: the frame spec, the schema, the
/// rules version and the (normalised) game config.
pub fn meta_of(spec: &FrameSpec, rules_version: u32, config: Value) -> Value {
    let mut m = spec.schema_meta();
    m["schema"] = json!(SCHEMA);
    m["rules_version"] = json!(rules_version);
    m["config"] = config;
    m
}

pub fn shard_meta<G: Game>(cfg: &G::Config) -> Value {
    meta_of(&G::spec(cfg), G::VERSION, serde_json::to_value(cfg).unwrap_or(Value::Null))
}

/// Candidate-index soft target of one decision: the most probable `k` legal actions
/// of `dist` (canonical ids → candidate indices; −1 / 0 padding) and the probability
/// mass that fell on illegal actions.
pub fn soft_row(d: &Decision, dist: &[(u32, f32)], k: usize) -> (Vec<i64>, Vec<f32>, f64) {
    let mut legal: Vec<(usize, f32)> = Vec::with_capacity(dist.len());
    let mut illegal = 0.0f64;
    for &(a, p) in dist {
        match d.frame.index_of(a) {
            Some(i) if p > 0.0 => legal.push((i, p)),
            Some(_) => {}
            None => illegal += p.max(0.0) as f64,
        }
    }
    legal.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    legal.truncate(k);
    let mut idx = vec![-1i64; k];
    let mut val = vec![0.0f32; k];
    for (j, &(i, p)) in legal.iter().enumerate() {
        idx[j] = i as i64;
        val[j] = p;
    }
    (idx, val, illegal)
}

struct Soft<'a> {
    k: usize,
    by_event: std::collections::HashMap<usize, &'a [(u32, f32)]>,
}

fn stream_sample(spec: &FrameSpec, decisions: &[&Decision], outcome: Option<&[f32]>, meta: Value, allow_truncated: bool,
                 soft: Option<&Soft>, st: &mut (usize, f64)) -> Sample {
    let t = decisions.len();
    let mut entities = Vec::with_capacity(t * spec.entities_len());
    let mut entity_mask = Vec::with_capacity(t * spec.max_entities);
    let mut entity_type = Vec::with_capacity(t * spec.max_entities);
    let mut globals = Vec::with_capacity(t * spec.global_features);
    let mut actions = Vec::with_capacity(t * spec.actions_len());
    let mut action_mask = Vec::with_capacity(t * spec.max_actions);
    let mut action_idx = Vec::with_capacity(t);
    let mut value = Vec::with_capacity(t);
    for d in decisions {
        let f = &d.frame;
        entities.extend_from_slice(&f.entities);
        entity_mask.extend(f.entity_mask.iter().map(|&m| m != 0));
        entity_type.extend_from_slice(&f.entity_type);
        globals.extend_from_slice(&f.globals);
        actions.extend_from_slice(&f.actions);
        action_mask.extend(f.action_mask.iter().map(|&m| m != 0));
        action_idx.push(d.action_idx as i64);
        value.push(outcome.map_or(0.0, |o| o[d.seat]));
    }
    let (e, fe, g, a, fa) = (spec.max_entities, spec.entity_features, spec.global_features, spec.max_actions, spec.action_features);
    let mut s = Sample::new(meta);
    s = if spec.bits_entities { s.bits("obs.entities", &[e, fe], &entities) } else { s.f32("obs.entities", &[e, fe], entities) };
    s = s.bool("obs.entity_mask", &[e], entity_mask);
    if spec.entity_types > 0 {
        s = s.i32("obs.entity_type", &[e], entity_type);
    }
    s = if spec.bits_globals { s.bits("obs.globals", &[g], &globals) } else { s.f32("obs.globals", &[g], globals) };
    s = if spec.bits_actions { s.bits("obs.actions", &[a, fa], &actions) } else { s.f32("obs.actions", &[a, fa], actions) };
    s = s.bool("obs.action_mask", &[a], action_mask).i64("tgt.action_idx", &[], action_idx).f32("tgt.value", &[], value);
    if allow_truncated {
        s = s.bool("tgt.value_mask", &[], vec![outcome.is_some(); t]);
    }
    if let Some(soft) = soft {
        let (mut idx, mut val) = (Vec::with_capacity(t * soft.k), Vec::with_capacity(t * soft.k));
        for d in decisions {
            match soft.by_event.get(&d.event) {
                Some(dist) => {
                    let (i, v, bad) = soft_row(d, dist, soft.k);
                    idx.extend(i);
                    val.extend(v);
                    st.0 += 1;
                    st.1 += bad;
                }
                None => {
                    idx.extend(std::iter::repeat_n(-1i64, soft.k));
                    val.extend(std::iter::repeat_n(0.0f32, soft.k));
                }
            }
        }
        s = s.i64("tgt.action_probs_idx", &[soft.k], idx).f32("tgt.action_probs_val", &[soft.k], val);
    }
    s
}

/// The replay of a record with the typed rules (same result as the registry's).
fn replay_typed<G: Game>(cfg: &G::Config, events: &[Event]) -> std::result::Result<DynReplay, String> {
    let spec = G::spec(cfg);
    let r = replay::<G>(cfg, events).map_err(|e| e.to_string())?;
    let terminal = r.terminal();
    Ok(DynReplay {
        streams: r.streams(&spec),
        outcome: terminal.then(|| r.state.outcome()),
        returns: terminal.then(|| r.state.returns()),
        terminal,
        turns: r.turns,
        decisions: r.decisions,
        spec,
    })
}

/// The samples of one record (replayed and verified), with its frame counts.
type Converted = (Vec<Sample>, usize, usize, (usize, f64));

fn record_samples(r: std::result::Result<DynReplay, String>, rec: &Record, opts: &ConvertOptions)
    -> std::result::Result<Converted, Reject> {
    let r = r.map_err(Reject::Illegal)?;
    let spec = &r.spec;
    if !r.terminal && !opts.allow_truncated {
        return Err(Reject::Truncated);
    }
    if let (Some(want), Some(got)) = (&rec.expected_returns, &r.returns) {
        if want.len() != got.len() || want.iter().zip(got).any(|(a, b)| (a - b).abs() > 1e-6) {
            return Err(Reject::Returns(format!("recorded returns {want:?}, replayed {got:?}")));
        }
    }
    let events_meta = opts.store_events.then(|| Value::from(rec.events.iter().map(|e| {
        let (s, i) = e.to_pair();
        json!([s, i])
    }).collect::<Vec<_>>()));
    let soft = (opts.soft_k > 0).then(|| Soft {
        k: opts.soft_k,
        by_event: rec.soft_targets.iter().map(|(e, d)| (*e, d.as_slice())).collect(),
    });
    let mut soft_st = (0usize, 0.0f64);
    let mut out = Vec::new();
    let (mut frames, mut forced) = (0, 0);
    for st in &r.streams {
        if let (Layout::PerSeat, Some(keep), Some(seat)) = (spec.layout(), &rec.seats, st.seat) {
            if !keep.contains(&seat) {
                continue;
            }
        }
        let decisions: Vec<&Decision> = st.decisions.iter().map(|&i| &r.decisions[i]).collect();
        frames += decisions.len();
        forced += decisions.iter().filter(|d| d.frame.legal.len() == 1).count();
        let mut meta = json!({
            "record": rec.id,
            "seat": st.seat.map_or(-1, |s| s as i64),
            "outcome": r.outcome,
            "returns": r.returns,
        });
        if let Value::Object(m) = &rec.meta {
            for (k, v) in m {
                meta[k] = v.clone();
            }
        }
        if let Some(ev) = &events_meta {
            meta["events"] = ev.clone();
        }
        out.push(stream_sample(spec, &decisions, r.outcome.as_deref(), meta, opts.allow_truncated, soft.as_ref(),
                               &mut soft_st));
    }
    if out.is_empty() {
        return Err(Reject::Empty);
    }
    Ok((out, frames, forced, soft_st))
}

/// Replay, verify and write `records` as gm-shards of game `G` under `out` (converted
/// in parallel chunks, written in input order — the output is deterministic). The
/// records are consumed lazily (stream them: corpora of 1e8+ records never need to
/// fit in memory); an `Err` item is a record the parser could not read — counted as
/// `unparseable` and skipped.
pub fn convert<G: Game>(cfg: &G::Config, records: impl Iterator<Item = Result<Record>>, out: &Path,
                        opts: &ConvertOptions) -> Result<Stats> {
    G::check_config(cfg).map_err(|e| anyhow::anyhow!("{} config: {e}", G::NAME))?;
    let w = ShardWriter::new(out, G::NAME, G::VERSION, shard_meta::<G>(cfg), opts.shard_size)?;
    convert_with(w, records, opts, |ev| replay_typed::<G>(cfg, ev))
}

/// [`convert`] for a game picked by NAME from the gmrules registry (`config` = JSON,
/// "" = default) — every registered game, no per-game code (`gmdata rules convert`).
pub fn convert_dyn(game: &str, config: &str, records: impl Iterator<Item = Result<Record>>, out: &Path,
                   opts: &ConvertOptions) -> Result<Stats> {
    let g = registry::lookup(game).map_err(anyhow::Error::msg)?;
    let spec = g.spec(config).map_err(anyhow::Error::msg)?;
    let cfg: Value = serde_json::from_str(&g.normalize_config(config).map_err(anyhow::Error::msg)?)?;
    let w = ShardWriter::new(out, g.name(), g.version(), meta_of(&spec, g.version(), cfg), opts.shard_size)?;
    convert_with(w, records, opts, |ev| g.replay(config, ev))
}

fn convert_with<F>(mut w: ShardWriter, records: impl Iterator<Item = Result<Record>>, opts: &ConvertOptions,
                   replay_fn: F) -> Result<Stats>
where
    F: Fn(&[Event]) -> std::result::Result<DynReplay, String> + Sync,
{
    let mut st = Stats::default();
    let mut records = records.peekable();
    const CHUNK: usize = 4096;
    let note = |st: &mut Stats, msg: String| {
        if st.errors.len() < 10 {
            st.errors.push(msg);
        }
    };
    while records.peek().is_some() {
        let mut chunk: Vec<Record> = Vec::with_capacity(CHUNK);
        for r in records.by_ref().take(CHUNK) {
            match r {
                Ok(rec) => chunk.push(rec),
                Err(e) => {
                    st.records += 1;
                    st.unparseable += 1;
                    note(&mut st, format!("unparseable: {e:#}"));
                }
            }
        }
        let results: Vec<_> = chunk.par_iter().map(|rec| record_samples(replay_fn(&rec.events), rec, opts)).collect();
        for (rec, res) in chunk.iter().zip(results) {
            st.records += 1;
            let reason = match res {
                Ok((samples, frames, forced, (soft_n, soft_bad))) => {
                    st.converted += 1;
                    st.frames += frames;
                    st.forced_frames += forced;
                    st.soft_frames += soft_n;
                    st.soft_illegal_mass += soft_bad;
                    st.samples += samples.len();
                    for s in samples {
                        w.push(s)?;
                    }
                    continue;
                }
                Err(Reject::Illegal(e)) => {
                    st.illegal += 1;
                    format!("illegal: {e}")
                }
                Err(Reject::Truncated) => {
                    st.truncated += 1;
                    "the record stops before the end of the game".into()
                }
                Err(Reject::Returns(e)) => {
                    st.returns_mismatch += 1;
                    e
                }
                Err(Reject::Empty) => {
                    st.empty += 1;
                    "no decision frame to keep".into()
                }
            };
            note(&mut st, format!("{}: {reason}", rec.id));
        }
    }
    w.flush()?;
    st.shards = w.written_shards;
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrules::games::kuhn_poker::{KuhnConfig, KuhnPoker};
    use gmshard::ShardReader;

    #[test]
    fn writes_candidate_shards() {
        let dir = std::env::temp_dir().join(format!("gmdata-cand-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = KuhnConfig { players: 2 };
        let ev = |deal: [u32; 2], acts: &[u32]| {
            let mut e: Vec<Event> = deal.iter().map(|&c| Event::Chance(c)).collect();
            e.extend(acts.iter().enumerate().map(|(i, &a)| Event::Act(i % 2, a)));
            e
        };
        let recs = vec![
            Record { id: "a".into(), events: ev([1, 2], &[0, 1, 1]), meta: json!({"x": 1}), seats: None, expected_returns: Some(vec![-2.0, 2.0]), soft_targets: vec![(3, vec![(0, 0.25), (1, 0.75)]), (4, vec![(1, 1.0), (7, 0.5)])] },
            Record { id: "b".into(), events: ev([1, 2], &[0, 1]), meta: Value::Null, seats: None, expected_returns: None, ..Default::default() },
            Record { id: "c".into(), events: ev([1, 1], &[]), meta: Value::Null, seats: None, expected_returns: None, ..Default::default() },
            Record { id: "d".into(), events: ev([2, 0], &[1, 0]), meta: Value::Null, seats: Some(vec![1]), expected_returns: Some(vec![9.0, -1.0]), ..Default::default() },
            Record { id: "e".into(), events: ev([2, 0], &[1, 0]), meta: Value::Null, seats: Some(vec![1]), expected_returns: None, ..Default::default() },
        ];
        let opts = ConvertOptions { store_events: true, soft_k: 2, ..Default::default() };
        let st = convert::<KuhnPoker>(&cfg, recs.into_iter().map(Ok), &dir, &opts).unwrap();
        assert_eq!((st.records, st.converted, st.truncated, st.illegal, st.returns_mismatch), (5, 2, 1, 1, 1));
        assert_eq!(st.samples, 3);
        let sh = ShardReader::open(&dir.join("kuhn_poker_00000")).unwrap();
        assert_eq!(sh.meta()["layout"], "per_seat");
        assert_eq!(sh.meta()["max_actions"], 2);
        assert_eq!(sh.converter_version, 1);
        assert_eq!(sh.n_samples(), 3);
        // record a: seat 0 passes then calls (2 frames), seat 1 bets (1 frame)
        assert_eq!(sh.read_i64("tgt.action_idx").unwrap()[..3], [0, 1, 1]);
        assert_eq!(sh.read_f32("tgt.value").unwrap()[..3], [-1.0, -1.0, 1.0]);
        assert_eq!(sh.samples[2].meta["seat"], 1);
        assert_eq!(sh.samples[0].meta["events"][2], json!([0, 0]));
        assert_eq!(sh.samples[0].meta["x"], 1);
        assert_eq!(sh.index["tensors"]["obs.actions"]["storage"], "packbits");
        // soft targets: record a's seat-1 bet (event 3) and seat-0 call (event 4); the
        // illegal id 7 is dropped and counted
        assert_eq!((st.soft_frames, st.soft_illegal_mass), (2, 0.5));
        let idx = sh.read_i64("tgt.action_probs_idx").unwrap();
        let val = sh.read_f32("tgt.action_probs_val").unwrap();
        // frames: [seat0 pass (event 2), seat0 call (event 4)], [seat1 bet (event 3)]
        assert_eq!(idx[..6], [-1, -1, 1, -1, 1, 0]);
        assert_eq!(val[..6], [0.0, 0.0, 1.0, 0.0, 0.75, 0.25]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
