//! gio training records → `gm-shard.v1` directories for the multi-game trainers
//! (`GameShardDataset(dir, game="generals")`, training/train_multi_game.py).
//!
//! The shard holds exactly the tensors `SeqReplayDataset` decodes from each
//! record (pinned by tests/test_rust_generals_parity.py), with SeqReplayDataset's
//! defaults for absent fields:
//!
//!   raw.grid_onehot        bits [T, 11, H, W]  grid_seq planes 0..10 (one-hot tile state)
//!   raw.army               f32  [T, H, W]      grid_seq plane 11 (log1p(army)/10)
//!   raw.legal_source_mask  bits [T, 1, H, W]   legal_source_mask_seq
//!   obs.globals_seq        f32  [T, G]
//!   tgt.count              i64  [T]            1 move / 0 pass / -1 dropped move
//!   tgt.src_y, src_x, dir_id, ratio_id  i64 [T, K]
//!   tgt.value              f32  [T]
//!
//! `data_processing/games/generals.py::expand` rebuilds `obs.grid_seq` and the
//! bool `obs.legal_source_mask_seq` at load time (3.6 KB per 25×25 frame on disk
//! instead of ~40 KB of JSON). The gio placeholders `enemy_heatmap` /
//! `next_visibility` (all zero) and the tick features (= frame index) are dropped.
//! Shard meta: grid_h, grid_w, grid_channels, global_features, max_actions_per_tick.
//!
//! Inputs are searched recursively for `*.json` and taken in path order (the
//! order of `GeneralsGioDataset`). Unreadable, empty or malformed records and
//! records whose meta differs from the first one are skipped with a warning.
//! Shard numbering continues after existing `generals_NNNNN` directories in
//! `--out`, so convert into a fresh directory.

use anyhow::{bail, ensure, Context, Result};
use gmshard::{Sample, ShardWriter};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const CONVERTER_VERSION: u32 = 1;
const GRID_CHANNELS: usize = 12;
const ONEHOT: usize = 11;
const DEFAULT_GLOBALS: usize = 4;
const DEFAULT_K: usize = 4;
/// records parsed in parallel per batch (bounds memory: a record is ~40 KB of JSON per frame)
const CHUNK: usize = 64;

/// The fields `SeqReplayDataset` reads; everything else is skipped by serde.
#[derive(Deserialize)]
struct Record {
    grid_seq: Vec<Vec<Vec<Vec<f32>>>>,
    globals_seq: Option<Vec<Vec<f32>>>,
    legal_source_mask_seq: Option<Vec<Vec<Vec<Vec<bool>>>>>,
    max_actions_per_tick: Option<usize>,
    count: Option<Vec<i64>>,
    src_y: Option<Vec<Vec<i64>>>,
    src_x: Option<Vec<Vec<i64>>>,
    dir_id: Option<Vec<Vec<i64>>>,
    ratio_id: Option<Vec<Vec<i64>>>,
    value: Option<Vec<f32>>,
}

/// `*.json` files under `root` (recursive, not following directory symlinks), in
/// path-component order; `root` itself when it is a file.
pub fn find_records(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    fn walk(d: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(d).with_context(|| format!("read {}", d.display()))? {
            let e = e?;
            let p = e.path();
            let ft = e.file_type()?;
            if ft.is_dir() {
                walk(&p, out)?;
            } else if !p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(".json")) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, &mut out)?;
    out.sort();
    Ok(out)
}

/// First `t` rows of an optional per-frame field (SeqReplayDataset's `[:T]`).
fn frames<'a, X>(v: &'a Option<Vec<X>>, t: usize, name: &str) -> Result<Option<&'a [X]>> {
    match v {
        None => Ok(None),
        Some(v) => {
            ensure!(v.len() >= t, "{name} has {} frames, grid_seq {t}", v.len());
            Ok(Some(&v[..t]))
        }
    }
}

fn slots(v: &Option<Vec<Vec<i64>>>, t: usize, k: usize, name: &str) -> Result<Vec<i64>> {
    match frames(v, t, name)? {
        None => Ok(vec![0; t * k]),
        Some(rows) => {
            let mut out = Vec::with_capacity(t * k);
            for r in rows {
                ensure!(r.len() == k, "{name} row has {} slots, max_actions_per_tick {k}", r.len());
                out.extend_from_slice(r);
            }
            Ok(out)
        }
    }
}

/// One record → (sample, shard meta it implies, frames).
fn record_to_sample(r: &Record, file: &str) -> Result<(Sample, Value, usize)> {
    let t = r.grid_seq.len();
    ensure!(t > 0, "empty record");
    let c = r.grid_seq[0].len();
    ensure!(c == GRID_CHANNELS, "grid_seq has {c} planes, expected {GRID_CHANNELS} (11 one-hot + army)");
    let h = r.grid_seq[0][0].len();
    ensure!(h > 0, "empty grid");
    let w = r.grid_seq[0][0][0].len();
    let mut onehot = Vec::with_capacity(t * ONEHOT * h * w);
    let mut army = Vec::with_capacity(t * h * w);
    for frame in &r.grid_seq {
        ensure!(frame.len() == c, "ragged grid_seq planes");
        for (ci, plane) in frame.iter().enumerate() {
            ensure!(plane.len() == h, "ragged grid_seq rows");
            for row in plane {
                ensure!(row.len() == w, "ragged grid_seq columns");
                if ci < ONEHOT {
                    if let Some(x) = row.iter().find(|&&x| x != 0.0 && x != 1.0) {
                        bail!("grid_seq one-hot plane {ci} holds {x}");
                    }
                    onehot.extend_from_slice(row);
                } else {
                    army.extend_from_slice(row);
                }
            }
        }
    }
    let (globals, g) = match frames(&r.globals_seq, t, "globals_seq")? {
        None => (vec![0.0f32; t * DEFAULT_GLOBALS], DEFAULT_GLOBALS),
        Some(rows) => {
            let g = rows[0].len();
            let mut out = Vec::with_capacity(t * g);
            for row in rows {
                ensure!(row.len() == g, "ragged globals_seq");
                out.extend_from_slice(row);
            }
            (out, g)
        }
    };
    let legal: Vec<f32> = match frames(&r.legal_source_mask_seq, t, "legal_source_mask_seq")? {
        None => vec![1.0; t * h * w],
        Some(rows) => {
            let mut out = Vec::with_capacity(t * h * w);
            for m in rows {
                ensure!(m.len() == 1 && m[0].len() == h && m[0].iter().all(|row| row.len() == w),
                        "legal_source_mask_seq frame is not [1, {h}, {w}]");
                for row in &m[0] {
                    out.extend(row.iter().map(|&b| b as u8 as f32));
                }
            }
            out
        }
    };
    let k = r.max_actions_per_tick.unwrap_or(DEFAULT_K);
    let count = frames(&r.count, t, "count")?.map(|v| v.to_vec()).unwrap_or_else(|| vec![0; t]);
    let value = frames(&r.value, t, "value")?.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; t]);
    let sample = Sample::new(json!({ "file": file }))
        .bits("raw.grid_onehot", &[ONEHOT, h, w], &onehot)
        .f32("raw.army", &[h, w], army)
        .bits("raw.legal_source_mask", &[1, h, w], &legal)
        .f32("obs.globals_seq", &[g], globals)
        .i64("tgt.count", &[], count)
        .i64("tgt.src_y", &[k], slots(&r.src_y, t, k, "src_y")?)
        .i64("tgt.src_x", &[k], slots(&r.src_x, t, k, "src_x")?)
        .i64("tgt.dir_id", &[k], slots(&r.dir_id, t, k, "dir_id")?)
        .i64("tgt.ratio_id", &[k], slots(&r.ratio_id, t, k, "ratio_id")?)
        .f32("tgt.value", &[], value);
    let meta = json!({"grid_h": h, "grid_w": w, "grid_channels": c, "global_features": g,
                      "max_actions_per_tick": k});
    Ok((sample, meta, t))
}

fn convert_file(path: &Path) -> Result<(Sample, Value, usize)> {
    let bytes = std::fs::read(path)?;
    let rec: Record = serde_json::from_slice(&bytes).context("not a gio record")?;
    let file = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    record_to_sample(&rec, &file)
}

/// Convert every record under `inputs` (in order) into shards under `out`.
pub fn shard(inputs: &[PathBuf], out: &Path, shard_size: usize) -> Result<Value> {
    let mut files = Vec::new();
    for inp in inputs {
        let f = find_records(inp)?;
        ensure!(!f.is_empty(), "no gio records (*.json) under {}", inp.display());
        files.extend(f);
    }
    let mut writer: Option<ShardWriter> = None;
    let mut shard_meta: Option<Value> = None;
    let (mut converted, mut failed, mut mismatched, mut n_frames) = (0usize, 0usize, 0usize, 0usize);
    for chunk in files.chunks(CHUNK) {
        let results: Vec<_> = chunk.par_iter().map(|p| convert_file(p)).collect();
        for (p, res) in chunk.iter().zip(results) {
            match res {
                Ok((s, meta, t)) => {
                    if shard_meta.is_none() {
                        writer = Some(ShardWriter::new(out, "generals", CONVERTER_VERSION, meta.clone(), shard_size)?);
                        shard_meta = Some(meta.clone());
                    }
                    if Some(&meta) != shard_meta.as_ref() {
                        mismatched += 1;
                        eprintln!("[warn] {}: skipped — meta {meta} differs from the shard's {}",
                                  p.display(), shard_meta.as_ref().unwrap());
                        continue;
                    }
                    writer.as_mut().unwrap().push(s)?;
                    converted += 1;
                    n_frames += t;
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("[warn] {}: skipped — {e:#}", p.display());
                }
            }
        }
    }
    let shards = match writer.as_mut() {
        Some(w) => {
            w.flush()?;
            w.written_shards
        }
        None => 0,
    };
    ensure!(converted > 0, "none of the {} records could be converted", files.len());
    Ok(json!({"records": files.len(), "converted": converted, "failed": failed,
              "meta_mismatch": mismatched, "frames": n_frames, "shards": shards,
              "meta": shard_meta}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(t: usize, h: usize, w: usize, g: usize) -> Value {
        let mut grid = vec![vec![vec![vec![0.0f32; w]; h]; GRID_CHANNELS]; t];
        for (ti, f) in grid.iter_mut().enumerate() {
            f[ti % ONEHOT][0][0] = 1.0;
            f[ONEHOT][0][0] = 0.25;
        }
        json!({
            "grid_seq": grid,
            "globals_seq": vec![vec![0.5f32; g]; t],
            "legal_source_mask_seq": vec![vec![vec![vec![true; w]; h]]; t],
            "max_actions_per_tick": 2,
            "count": (0..t as i64).map(|i| i % 2).collect::<Vec<_>>(),
            "src_y": vec![vec![1i64, 0]; t], "src_x": vec![vec![0i64, 0]; t],
            "dir_id": vec![vec![3i64, 0]; t], "ratio_id": vec![vec![1i64, 0]; t],
            "value": vec![-0.5f32; t],
            "enemy_heatmap": vec![vec![vec![vec![0.0f32; w]; h]]; t],
        })
    }

    #[test]
    fn converts_valid_records_and_skips_the_rest() {
        let dir = std::env::temp_dir().join(format!("gmdata-generals-shard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let inp = dir.join("in");
        std::fs::create_dir_all(inp.join("sub")).unwrap();
        std::fs::write(inp.join("a.json"), serde_json::to_vec(&record(3, 2, 3, 4)).unwrap()).unwrap();
        std::fs::write(inp.join("sub").join("b.json"), serde_json::to_vec(&record(2, 2, 3, 4)).unwrap()).unwrap();
        std::fs::write(inp.join("c.json"), serde_json::to_vec(&record(2, 3, 3, 4)).unwrap()).unwrap(); // other grid
        std::fs::write(inp.join("d.json"), b"{not json").unwrap();
        std::fs::write(inp.join("notes.txt"), b"ignored").unwrap();
        let s = shard(&[inp.clone()], &dir.join("out"), 8).unwrap();
        assert_eq!(s["records"], 4);
        assert_eq!(s["converted"], 2);
        assert_eq!(s["meta_mismatch"], 1);
        assert_eq!(s["failed"], 1);
        assert_eq!(s["frames"], 5);
        let idx: Value = serde_json::from_slice(&std::fs::read(dir.join("out/generals_00000/index.json")).unwrap()).unwrap();
        assert_eq!(idx["game"], "generals");
        assert_eq!(idx["meta"]["grid_h"], 2);
        assert_eq!(idx["meta"]["max_actions_per_tick"], 2);
        assert_eq!(idx["tensors"]["raw.grid_onehot"]["storage"], "packbits");
        assert_eq!(idx["tensors"]["tgt.src_y"]["frame_shape"], json!([2]));
        // path order: a.json, c.json (skipped), d.json (skipped), sub/b.json
        assert_eq!(idx["samples"][0]["meta"]["file"], "a.json");
        assert_eq!(idx["samples"][1]["meta"]["file"], "b.json");
        assert_eq!(idx["samples"][1]["start"], 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_non_binary_one_hot_planes() {
        let mut r = record(1, 2, 2, 4);
        r["grid_seq"][0][3][1][1] = json!(0.5);
        let rec: Record = serde_json::from_value(r).unwrap();
        assert!(record_to_sample(&rec, "x.json").is_err());
    }
}
