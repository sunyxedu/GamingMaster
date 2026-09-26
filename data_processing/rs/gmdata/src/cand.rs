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
#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub shard_size: usize,
    /// Write each sample's trajectory into its meta (golden tests, debugging; costs
    /// index.json size).
    pub store_events: bool,
    /// Keep records that stop before the end (value_mask false on their frames).
    pub allow_truncated: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions { shard_size: 1024, store_events: false, allow_truncated: false }
    }
}

/// Conversion counters (printed as JSON by the CLI).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub records: usize,
    pub converted: usize,
    pub illegal: usize,
    pub truncated: usize,
    pub returns_mismatch: usize,
    pub empty: usize,
    pub samples: usize,
    pub frames: usize,
    pub forced_frames: usize,
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

fn stream_sample(spec: &FrameSpec, decisions: &[&Decision], outcome: Option<&[f32]>, meta: Value, allow_truncated: bool)
    -> Sample {
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
fn record_samples(r: std::result::Result<DynReplay, String>, rec: &Record, opts: &ConvertOptions)
    -> std::result::Result<(Vec<Sample>, usize, usize), Reject> {
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
        out.push(stream_sample(spec, &decisions, r.outcome.as_deref(), meta, opts.allow_truncated));
    }
    if out.is_empty() {
        return Err(Reject::Empty);
    }
    Ok((out, frames, forced))
}

/// Replay, verify and write `records` as gm-shards of game `G` under `out` (converted
/// in parallel chunks, written in input order — the output is deterministic).
pub fn convert<G: Game>(cfg: &G::Config, records: impl Iterator<Item = Record>, out: &Path, opts: &ConvertOptions)
    -> Result<Stats> {
    G::check_config(cfg).map_err(|e| anyhow::anyhow!("{} config: {e}", G::NAME))?;
    let w = ShardWriter::new(out, G::NAME, G::VERSION, shard_meta::<G>(cfg), opts.shard_size)?;
    convert_with(w, records, opts, |ev| replay_typed::<G>(cfg, ev))
}

/// [`convert`] for a game picked by NAME from the gmrules registry (`config` = JSON,
/// "" = default) — every registered game, no per-game code (`gmdata rules convert`).
pub fn convert_dyn(game: &str, config: &str, records: impl Iterator<Item = Record>, out: &Path, opts: &ConvertOptions)
    -> Result<Stats> {
    let g = registry::lookup(game).map_err(anyhow::Error::msg)?;
    let spec = g.spec(config).map_err(anyhow::Error::msg)?;
    let cfg: Value = serde_json::from_str(&g.normalize_config(config).map_err(anyhow::Error::msg)?)?;
    let w = ShardWriter::new(out, g.name(), g.version(), meta_of(&spec, g.version(), cfg), opts.shard_size)?;
    convert_with(w, records, opts, |ev| g.replay(config, ev))
}

fn convert_with<F>(mut w: ShardWriter, records: impl Iterator<Item = Record>, opts: &ConvertOptions, replay_fn: F) -> Result<Stats>
where
    F: Fn(&[Event]) -> std::result::Result<DynReplay, String> + Sync,
{
    let mut st = Stats::default();
    let mut records = records.peekable();
    const CHUNK: usize = 4096;
    while records.peek().is_some() {
        let chunk: Vec<Record> = records.by_ref().take(CHUNK).collect();
        let results: Vec<_> = chunk.par_iter().map(|rec| record_samples(replay_fn(&rec.events), rec, opts)).collect();
        for (rec, res) in chunk.iter().zip(results) {
            st.records += 1;
            let reason = match res {
                Ok((samples, frames, forced)) => {
                    st.converted += 1;
                    st.frames += frames;
                    st.forced_frames += forced;
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
            if st.errors.len() < 10 {
                st.errors.push(format!("{}: {reason}", rec.id));
            }
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
            Record { id: "a".into(), events: ev([1, 2], &[0, 1, 1]), meta: json!({"x": 1}), seats: None, expected_returns: Some(vec![-2.0, 2.0]) },
            Record { id: "b".into(), events: ev([1, 2], &[0, 1]), meta: Value::Null, seats: None, expected_returns: None },
            Record { id: "c".into(), events: ev([1, 1], &[]), meta: Value::Null, seats: None, expected_returns: None },
            Record { id: "d".into(), events: ev([2, 0], &[1, 0]), meta: Value::Null, seats: Some(vec![1]), expected_returns: Some(vec![9.0, -1.0]) },
            Record { id: "e".into(), events: ev([2, 0], &[1, 0]), meta: Value::Null, seats: Some(vec![1]), expected_returns: None },
        ];
        let opts = ConvertOptions { store_events: true, ..Default::default() };
        let st = convert::<KuhnPoker>(&cfg, recs.into_iter(), &dir, &opts).unwrap();
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
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
