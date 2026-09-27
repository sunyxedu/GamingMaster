//! Honor of Kings 1v1: the Hokoff offline datasets (Tencent, "Hokoff: Real Game Dataset
//! from Honor of Kings and its Offline Reinforcement Learning Benchmarks", NeurIPS 2023
//! D&B) → shards for the "hok" adapter (model/games/hok.py).
//!
//! Input: a Hokoff `all_data.hdf5` — dataset `datas` [N, 911] f32, one row per decision
//! frame of one player, each trajectory's rows contiguous and closed by done = 1. Row
//! layout (hokoff hok1v1/offline_train/datasets.py `data_split`):
//!
//!   [0, 725)    observation: the hok1v1 feature vector (own hero 235 | enemy hero 235 |
//!               main hero 14 | own soldiers 4 × 18 | enemy soldiers 4 × 18 | own towers
//!               2 × 18 | enemy towers 2 × 18 | global 25)
//!   [725, 731)  action: button, move_x, move_z, skill_x, skill_z, target
//!   731         reward (the environment's shaped reward), 732 done
//!   [733, 905)  legal action: button 12 | move_x 16 | move_z 16 | skill_x 16 |
//!               skill_z 16 | target 12 × 8 (a row per button)
//!   [905, 911)  sub action: the heads the taken button uses (button, move_x, move_z,
//!               skill_x, skill_z, target)
//!
//! Output (one sample per trajectory):
//!   obs.features                 [T,725] f32
//!   tgt.button_idx … target_idx  [T] i64   (−1 where the taken button does not use the head)
//!   tgt.legal_button [T,12], tgt.legal_move_x / move_z / skill_x / skill_z [T,16],
//!   tgt.legal_target [T,12,8], tgt.legal_sub_of [T,6]   bool
//! No tgt.value: the rows carry no game result (the shaped reward is not an outcome).
//! Trajectories go to `<out>/train` or, every `--val-every`-th, `<out>/val`.

// without the `hok` feature only the frame logic (and its tests) is compiled
#![cfg_attr(not(feature = "hok"), allow(dead_code, unused_imports))]

use anyhow::{ensure, Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FEATURES: usize = 725;
pub const ROW: usize = 911;
const ACT: usize = 725;
const REWARD: usize = 731;
const DONE: usize = 732;
const LEGAL: usize = 733;
const SUB: usize = 905;
const BUTTONS: usize = 12;
const DIRS: usize = 16;
const TARGETS: usize = 8;
/// argument heads (sub-action columns 1..=5) and their sizes
const HEADS: [(&str, usize); 5] =
    [("move_x", DIRS), ("move_z", DIRS), ("skill_x", DIRS), ("skill_z", DIRS), ("target", TARGETS)];
/// the legal-action slices of the four direction heads: (start, len) within the 172
const DIR_LEGAL: [(usize, usize); 4] = [(12, 16), (28, 16), (44, 16), (60, 16)];
const TARGET_LEGAL: usize = 76;

/// One trajectory's rows → a sample; counts illegal labels into `stats`.
pub fn trajectory_to_sample(rows: &[f32], source: &str, episode: usize,
                            stats: &mut BTreeMap<String, usize>) -> Result<Sample> {
    ensure!(!rows.is_empty() && rows.len().is_multiple_of(ROW), "trajectory rows must be whole {ROW}-column rows");
    let t = rows.len() / ROW;
    let mut features = Vec::with_capacity(t * FEATURES);
    let mut idx: Vec<Vec<i64>> = (0..6).map(|_| Vec::with_capacity(t)).collect();
    let mut legal_button = Vec::with_capacity(t * BUTTONS);
    let mut legal_dir: Vec<Vec<bool>> = (0..4).map(|_| Vec::with_capacity(t * DIRS)).collect();
    let mut legal_target = Vec::with_capacity(t * BUTTONS * TARGETS);
    let mut sub_of = Vec::with_capacity(t * 6);
    let mut ret = 0.0f64;
    for r in rows.as_chunks::<ROW>().0 {
        features.extend_from_slice(&r[..FEATURES]);
        ret += r[REWARD] as f64;
        let legal = &r[LEGAL..SUB];
        let sub = &r[SUB..ROW];
        let button = r[ACT].round() as i64;
        ensure!((0..BUTTONS as i64).contains(&button), "button {button} out of range");
        if legal[button as usize] < 0.5 {
            *stats.entry("illegal_button".into()).or_default() += 1;
        }
        idx[0].push(button);
        for (h, &(name, n)) in HEADS.iter().enumerate() {
            let a = r[ACT + 1 + h].round() as i64;
            let used = sub[1 + h] > 0.5;
            if !used {
                idx[1 + h].push(-1);
                continue;
            }
            ensure!((0..n as i64).contains(&a), "{name} {a} out of range");
            let ok = if h < 4 {
                legal[DIR_LEGAL[h].0 + a as usize] > 0.5
            } else {
                legal[TARGET_LEGAL + button as usize * TARGETS + a as usize] > 0.5
            };
            if !ok {
                *stats.entry(format!("illegal_{name}")).or_default() += 1;
            }
            idx[1 + h].push(a);
        }
        legal_button.extend(legal[..BUTTONS].iter().map(|&v| v > 0.5));
        for (h, &(s, n)) in DIR_LEGAL.iter().enumerate() {
            legal_dir[h].extend(legal[s..s + n].iter().map(|&v| v > 0.5));
        }
        legal_target.extend(legal[TARGET_LEGAL..TARGET_LEGAL + BUTTONS * TARGETS].iter().map(|&v| v > 0.5));
        sub_of.extend(sub.iter().map(|&v| v > 0.5));
    }
    let mut s = Sample::new(json!({"source": source, "episode": episode, "frames": t,
                                   "shaped_return": ret}))
        .f32("obs.features", &[FEATURES], features)
        .i64("tgt.button_idx", &[], std::mem::take(&mut idx[0]));
    for (h, &(name, _)) in HEADS.iter().enumerate() {
        s = s.i64(&format!("tgt.{name}_idx"), &[], std::mem::take(&mut idx[1 + h]));
    }
    s = s.bool("tgt.legal_button", &[BUTTONS], legal_button);
    for (h, &(name, n)) in HEADS[..4].iter().enumerate() {
        s = s.bool(&format!("tgt.legal_{name}"), &[n], std::mem::take(&mut legal_dir[h]));
    }
    Ok(s.bool("tgt.legal_target", &[BUTTONS, TARGETS], legal_target)
        .bool("tgt.legal_sub_of", &[6], sub_of))
}

/// A Hokoff HDF5 file → train / val shards; returns the conversion stats.
#[cfg(feature = "hok")]
pub fn convert(input: &Path, out: &Path, name: &str, shard_size: usize, val_every: usize,
               max_episodes: Option<usize>) -> Result<serde_json::Value> {
    ensure!(val_every >= 2, "--val-every must be >= 2");
    let file = hdf5::File::open(input).with_context(|| format!("open {}", input.display()))?;
    let ds = file.dataset("datas").context("no dataset `datas` (not a Hokoff all_data.hdf5)")?;
    let shape = ds.shape();
    ensure!(shape.len() == 2 && shape[1] == ROW, "datas has shape {shape:?}, expected [N, {ROW}]");
    let n = shape[0];
    let meta = json!({"features": FEATURES, "source": "hokoff", "dataset": name});
    let mut train = ShardWriter::new(&out.join("train"), "hok", 1, meta.clone(), shard_size)?;
    let mut val = ShardWriter::new(&out.join("val"), "hok", 1, meta, shard_size)?;
    let mut stats: BTreeMap<String, usize> = BTreeMap::new();
    let (mut episodes, mut cur) = (0usize, Vec::<f32>::new());
    // the whole dataset at once (≈ 5.7 GB for 1.56M rows): gzip-chunked, read through HDF5
    let flat: Vec<f32> = ds.read_raw().context("read datas")?;
    ensure!(flat.len() == n * ROW, "read {} values, expected {}", flat.len(), n * ROW);
    for r in flat.as_chunks::<ROW>().0 {
        cur.extend_from_slice(r);
        if r[DONE] > 0.5 {
            let s = trajectory_to_sample(&cur, name, episodes, &mut stats)?;
            if episodes % val_every == 0 { val.push(s)?; } else { train.push(s)?; }
            episodes += 1;
            cur.clear();
            if max_episodes.map(|m| episodes >= m).unwrap_or(false) {
                break;
            }
        }
    }
    if !cur.is_empty() {
        *stats.entry("unterminated_rows".into()).or_default() += cur.len() / ROW;
    }
    train.flush()?;
    val.flush()?;
    let mut v = json!({"rows": n, "episodes": episodes, "train_frames": train.written_frames,
                       "val_frames": val.written_frames, "train_shards": train.written_shards,
                       "val_shards": val.written_shards});
    for (k, c) in stats {
        v[k] = json!(c);
    }
    Ok(v)
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Convert a Hokoff 1v1 all_data.hdf5 to gm-shards (<out>/train, <out>/val); needs
    /// gmdata built with `--features hok` (HDF5, built from source)
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// the dataset's name, recorded in every sample (e.g. hard_expert)
        #[arg(long)]
        name: String,
        #[arg(long, default_value_t = 64)]
        shard_size: usize,
        /// every N-th trajectory goes to val
        #[arg(long, default_value_t = 50)]
        val_every: usize,
        #[arg(long)]
        max_episodes: Option<usize>,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        #[cfg(feature = "hok")]
        Cmd::Convert { input, out, name, shard_size, val_every, max_episodes } => {
            println!("{}", convert(&input, &out, &name, shard_size, val_every, max_episodes)?);
            Ok(())
        }
        #[cfg(not(feature = "hok"))]
        Cmd::Convert { .. } => anyhow::bail!("gmdata was built without the `hok` feature (HDF5): \
                                              cargo build --release -p gmdata --features hok"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(button: usize, args: [usize; 5], sub: [bool; 6], done: bool) -> Vec<f32> {
        let mut r = vec![0.0f32; ROW];
        for (i, v) in r[..FEATURES].iter_mut().enumerate() {
            *v = (i % 7) as f32 / 7.0;
        }
        r[ACT] = button as f32;
        for h in 0..5 {
            r[ACT + 1 + h] = args[h] as f32;
        }
        r[REWARD] = 0.25;
        r[DONE] = if done { 1.0 } else { 0.0 };
        for v in r[LEGAL..SUB].iter_mut() {
            *v = 1.0;
        }
        r[LEGAL + 0] = 0.0; // button 0 illegal
        for (i, &s) in sub.iter().enumerate() {
            r[SUB + i] = if s { 1.0 } else { 0.0 };
        }
        r
    }

    #[test]
    fn unused_heads_become_minus_one_and_masks_split() {
        let mut rows = row(2, [3, 4, 5, 6, 1], [true, true, true, false, false, false], false);
        rows.extend(row(4, [1, 2, 7, 8, 2], [true, false, false, true, true, true], true));
        let mut stats = BTreeMap::new();
        let s = trajectory_to_sample(&rows, "unit", 0, &mut stats).unwrap();
        assert_eq!(s.frames().unwrap(), 2);
        let get = |name: &str| s.tensors.iter().find(|x| x.name == name).unwrap();
        assert_eq!(get("tgt.button_idx").col.as_i64().unwrap(), &[2, 4]);
        assert_eq!(get("tgt.move_x_idx").col.as_i64().unwrap(), &[3, -1]);
        assert_eq!(get("tgt.skill_x_idx").col.as_i64().unwrap(), &[-1, 7]);
        assert_eq!(get("tgt.target_idx").col.as_i64().unwrap(), &[-1, 2]);
        assert_eq!(get("tgt.legal_target").frame_shape, vec![BUTTONS, TARGETS]);
        assert_eq!(get("obs.features").frame_shape, vec![FEATURES]);
        assert!(stats.is_empty(), "{stats:?}");
        // a label outside its mask is counted
        let mut bad = row(0, [3, 4, 5, 6, 1], [true, true, true, false, false, false], true);
        bad[LEGAL + 12 + 3] = 0.0;
        let _ = trajectory_to_sample(&bad, "unit", 1, &mut stats).unwrap();
        assert_eq!(stats.get("illegal_button"), Some(&1));
        assert_eq!(stats.get("illegal_move_x"), Some(&1));
    }
}
