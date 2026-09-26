//! Overlay shards: a new shard directory that reuses every tensor file of an existing
//! (base) shard and adds columns — the engine teacher labels written by `gmdata label`.
//!
//! ```text
//! base/chess_00012/            out/chess_00012/
//!   index.json                   index.json        base index + new tensor entries,
//!                                                  meta.teacher, overlay provenance
//!   obs.board_seq.npy   ──link──  obs.board_seq.npy
//!   tgt.from_idx.npy    ──link──  tgt.from_idx.npy
//!   ...                           ...
//!                                 tgt.from_probs_idx.npy   new (full length, n_frames rows)
//!                                 tgt.engine_value.npy     new
//! ```
//!
//! * Base files are never modified or copied when avoidable: each is hard-linked
//!   (after resolving symlinks), falling back to an absolute symlink (e.g. across
//!   filesystems, or when the kernel refuses hard links to files of another owner), then
//!   to a plain copy ([`LinkPolicy`]). The weakest mode used is reported ([`LinkMode`])
//!   and recorded in `index.json` "overlay". A symlinked overlay breaks if the base is
//!   moved; hard links and copies do not.
//! * Every new column must have exactly `n_frames` rows; any [`crate::Column`] dtype
//!   works (i64 `[T, K]`, f32, bool, packed bits, …). Names are `obs.*` / `tgt.*` /
//!   `raw.*` and must not collide with base tensors (labelling a labelled shard again
//!   is an error — label the base).
//! * [`OverlayWriter::set_teacher`] sets `index.json` `meta.teacher`: it becomes part of
//!   the shard meta, which `GameShardDataset` requires to be identical across a data
//!   directory — so it must only hold run-level settings (engine, limits, transform),
//!   never per-shard numbers. Per-shard facts go in top-level fields
//!   ([`OverlayWriter::set_field`], e.g. `teacher_stats`), which the loaders ignore.
//! * [`OverlayWriter::commit`] builds the directory under a temporary sibling name
//!   (`.<name>.tmp-<pid>-<nanos>`), syncs it, and renames it into place, so `out`
//!   either does not exist or is complete — callers resume by skipping existing outputs.

use crate::{Column, IndexTensor, ShardReader, Tensor};
use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// How base tensor files are brought into the overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkPolicy {
    /// hard link → symlink → copy (first that works, per file)
    #[default]
    Auto,
    /// absolute symlinks (→ copy where symlinks are unsupported)
    Symlink,
    /// always copy
    Copy,
}

/// How a file ended up in the overlay; ordered from strongest to weakest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LinkMode {
    Hardlink,
    Symlink,
    Copy,
}

impl LinkMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkMode::Hardlink => "hardlink",
            LinkMode::Symlink => "symlink",
            LinkMode::Copy => "copy",
        }
    }
}

/// Builds one overlay shard from a base shard (see the module docs).
#[derive(Debug, Clone)]
pub struct OverlayWriter {
    base_dir: PathBuf,
    base_index: Value,
    n_frames: usize,
    base_files: BTreeMap<String, String>,
    cols: BTreeMap<String, Tensor>,
    teacher: Option<Value>,
    fields: BTreeMap<String, Value>,
    policy: LinkPolicy,
}

/// Top-level `index.json` keys an overlay may not override via [`OverlayWriter::set_field`].
const RESERVED: [&str; 8] = ["format", "game", "converter_version", "meta", "n_frames", "tensors", "samples", "overlay"];

impl OverlayWriter {
    pub fn new(base: &ShardReader) -> Self {
        Self {
            base_dir: base.dir.clone(),
            base_index: base.index.clone(),
            n_frames: base.n_frames,
            base_files: base.tensors.iter().map(|(n, t)| (n.clone(), t.file.clone())).collect(),
            cols: BTreeMap::new(),
            teacher: None,
            fields: BTreeMap::new(),
            policy: LinkPolicy::Auto,
        }
    }

    pub fn link_policy(mut self, policy: LinkPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Frames every new column must have.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Add a full-length column (`n_frames` rows).
    pub fn add(&mut self, t: Tensor) -> Result<&mut Self> {
        let Some((group, key)) = t.name.split_once('.') else { bail!("tensor name {:?} is not <group>.<key>", t.name) };
        if !matches!(group, "obs" | "tgt" | "raw") || key.is_empty() || t.name.contains(['/', '\\']) {
            bail!("tensor name {:?}: group must be obs / tgt / raw", t.name);
        }
        if self.base_files.contains_key(&t.name) || self.cols.contains_key(&t.name) {
            bail!("tensor {} already exists in the shard (overlays add columns; label the base shard)", t.name);
        }
        let file = format!("{}.npy", t.name);
        if self.base_files.values().any(|f| *f == file) {
            bail!("tensor file {file} collides with a base file");
        }
        if let Column::Bits(_, per) = &t.col {
            if *per != t.frame_len() {
                bail!("{}: bit column frame size {per} != frame shape {:?}", t.name, t.frame_shape);
            }
        }
        if t.col.len() % t.frame_len() != 0 || t.frames() != self.n_frames {
            bail!("{}: {} elements = {} frames of {:?}, the shard has {} frames", t.name, t.col.len(),
                  t.col.len() as f64 / t.frame_len() as f64, t.frame_shape, self.n_frames);
        }
        self.cols.insert(t.name.clone(), t);
        Ok(self)
    }

    pub fn add_f32(&mut self, name: &str, frame_shape: &[usize], data: Vec<f32>) -> Result<&mut Self> {
        self.add(Tensor::new(name, frame_shape, Column::F32(data)))
    }

    pub fn add_i64(&mut self, name: &str, frame_shape: &[usize], data: Vec<i64>) -> Result<&mut Self> {
        self.add(Tensor::new(name, frame_shape, Column::I64(data)))
    }

    pub fn add_bool(&mut self, name: &str, frame_shape: &[usize], data: &[bool]) -> Result<&mut Self> {
        self.add(Tensor::new(name, frame_shape, Column::Bool(data.iter().map(|&b| b as u8).collect())))
    }

    /// {0,1} float32 column stored bit-packed.
    pub fn add_bits(&mut self, name: &str, frame_shape: &[usize], data: &[f32]) -> Result<&mut Self> {
        let per = frame_shape.iter().product::<usize>().max(1);
        let packed = crate::pack_bits(per, data).with_context(|| name.to_string())?;
        self.add(Tensor::new(name, frame_shape, Column::Bits(packed, per)))
    }

    /// `index.json` `meta.teacher` (run-level settings only; see the module docs).
    pub fn set_teacher(&mut self, teacher: Value) -> &mut Self {
        self.teacher = Some(teacher);
        self
    }

    /// An extra top-level `index.json` field (per-shard facts such as `teacher_stats`).
    pub fn set_field(&mut self, key: &str, v: Value) -> Result<&mut Self> {
        if RESERVED.contains(&key) {
            bail!("index field {key:?} is reserved");
        }
        self.fields.insert(key.to_string(), v);
        Ok(self)
    }

    /// Write the overlay to `out` (which must not exist) atomically; returns the weakest
    /// link mode used for the base files.
    pub fn commit(&self, out: &Path) -> Result<LinkMode> {
        if out.exists() {
            bail!("{} already exists", out.display());
        }
        let name = out.file_name().context("overlay output has no directory name")?.to_string_lossy().to_string();
        let parent = match out.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        std::fs::create_dir_all(&parent).with_context(|| format!("create {}", parent.display()))?;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let tmp = parent.join(format!(".{name}.tmp-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        let guard = TmpGuard(Some(tmp.clone()));
        let mode = self.fill(&tmp)?;
        sync_dir(&tmp);
        std::fs::rename(&tmp, out).with_context(|| format!("rename {} -> {}", tmp.display(), out.display()))?;
        std::mem::forget(guard);
        sync_dir(&parent);
        Ok(mode)
    }

    fn fill(&self, tmp: &Path) -> Result<LinkMode> {
        let base_abs = std::fs::canonicalize(&self.base_dir).with_context(|| format!("resolve {}", self.base_dir.display()))?;
        let mut weakest = LinkMode::Hardlink;
        let mut linked = BTreeSet::new();
        for file in self.base_files.values() {
            if !linked.insert(file.clone()) {
                continue;
            }
            let src = std::fs::canonicalize(base_abs.join(file)).with_context(|| format!("resolve {}/{file}", base_abs.display()))?;
            weakest = weakest.max(place(&src, &tmp.join(file), self.policy)?);
        }
        let mut index = self.base_index.clone();
        let obj = index.as_object_mut().context("base index.json is not an object")?;
        let tensors = obj.entry("tensors").or_insert_with(|| json!({})).as_object_mut().context("tensor table")?;
        for (tname, t) in &self.cols {
            let file = format!("{tname}.npy");
            let mut shape = vec![self.n_frames];
            shape.extend_from_slice(&t.frame_shape);
            let path = tmp.join(&file);
            t.col.write_npy(&path, &shape)?;
            sync_file(&path)?;
            tensors.insert(tname.clone(), serde_json::to_value(IndexTensor::of(t, file))?);
        }
        if let Some(teacher) = &self.teacher {
            let meta = obj.entry("meta").or_insert(Value::Null);
            if !meta.is_object() {
                *meta = Value::Object(Map::new());
            }
            meta.as_object_mut().unwrap().insert("teacher".into(), teacher.clone());
        }
        for (k, v) in &self.fields {
            obj.insert(k.clone(), v.clone());
        }
        obj.insert("overlay".into(), json!({"base": base_abs.to_string_lossy(), "link": weakest.as_str()}));
        let path = tmp.join("index.json");
        std::fs::write(&path, serde_json::to_vec(&index)?).with_context(|| format!("write {}", path.display()))?;
        sync_file(&path)?;
        Ok(weakest)
    }
}

/// Bring `src` (absolute, symlink-free) to `dst` under `policy`.
fn place(src: &Path, dst: &Path, policy: LinkPolicy) -> Result<LinkMode> {
    if policy == LinkPolicy::Auto && std::fs::hard_link(src, dst).is_ok() {
        return Ok(LinkMode::Hardlink);
    }
    #[cfg(unix)]
    if policy != LinkPolicy::Copy && std::os::unix::fs::symlink(src, dst).is_ok() {
        return Ok(LinkMode::Symlink);
    }
    std::fs::copy(src, dst).with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
    Ok(LinkMode::Copy)
}

fn sync_file(path: &Path) -> Result<()> {
    std::fs::File::open(path)?.sync_all().with_context(|| format!("sync {}", path.display()))
}

/// Best effort: directory fsync is not supported everywhere.
fn sync_dir(path: &Path) {
    if let Ok(f) = std::fs::File::open(path) {
        let _ = f.sync_all();
    }
}

/// Removes a half-built temporary directory on early return.
struct TmpGuard(Option<PathBuf>);

impl Drop for TmpGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::tests::sample;
    use crate::testutil::tmpdir;
    use crate::{DType, ShardWriter};

    fn base_shard(root: &Path) -> ShardReader {
        let mut w = ShardWriter::new(&root.join("base"), "toy", 3, json!({"board": 3}), 8).unwrap();
        for i in 0..3 {
            w.push(sample(i)).unwrap();
        }
        w.flush().unwrap();
        ShardReader::open(&root.join("base").join("toy_00000")).unwrap()
    }

    fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let p = e.unwrap().path();
                (p.file_name().unwrap().to_string_lossy().to_string(), std::fs::read(&p).unwrap())
            })
            .collect()
    }

    fn overlay(base: &ShardReader) -> OverlayWriter {
        let n = base.n_frames;
        let mut ov = OverlayWriter::new(base);
        ov.add_i64("tgt.teacher_idx", &[4], (0..n as i64 * 4).map(|k| if k % 4 == 3 { -1 } else { k }).collect()).unwrap();
        ov.add_f32("tgt.teacher_q", &[4], (0..n * 4).map(|k| if k % 4 == 3 { f32::NAN } else { k as f32 / 100.0 }).collect())
            .unwrap();
        ov.add_f32("tgt.engine_value", &[], (0..n).map(|k| k as f32 / n as f32).collect()).unwrap();
        ov.add_bool("tgt.engine_value_mask", &[], &(0..n).map(|k| k % 3 != 0).collect::<Vec<_>>()).unwrap();
        ov.add_bits("tgt.legal_extra", &[5], &(0..n * 5).map(|k| (k % 2) as f32).collect::<Vec<_>>()).unwrap();
        ov.set_teacher(json!({"engine": "fake 1.0", "tau": 0.05}));
        ov.set_field("teacher_stats", json!({"labelled_frames": 7})).unwrap();
        ov
    }

    #[test]
    fn overlay_links_base_and_adds_columns() {
        let root = tmpdir("overlay");
        let base = base_shard(&root);
        let before = snapshot(&base.dir);
        let out = root.join("out").join("toy_00000");
        let mode = overlay(&base).commit(&out).unwrap();
        assert_eq!(mode, LinkMode::Hardlink);
        assert_eq!(snapshot(&base.dir), before, "base shard modified");
        let r = ShardReader::open(&out).unwrap();
        assert_eq!((r.game.as_str(), r.converter_version, r.n_frames, r.samples.clone()),
                   (base.game.as_str(), base.converter_version, base.n_frames, base.samples.clone()));
        assert_eq!(r.meta()["board"], 3);
        assert_eq!(r.meta()["teacher"]["engine"], "fake 1.0");
        assert_eq!(r.index["teacher_stats"]["labelled_frames"], 7);
        assert_eq!(r.index["overlay"]["link"], "hardlink");
        // old columns identical (and the very same inode), new ones typed as added
        for (name, info) in &base.tensors {
            assert_eq!(r.tensors[name], *info);
            assert_eq!(std::fs::read(out.join(&info.file)).unwrap(), before[&info.file]);
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let (a, b) = (std::fs::metadata(out.join(&info.file)).unwrap(), std::fs::metadata(base.dir.join(&info.file)).unwrap());
                assert_eq!((a.dev(), a.ino()), (b.dev(), b.ino()));
            }
        }
        assert_eq!(r.tensors.len(), base.tensors.len() + 5);
        assert_eq!(r.tensor_info("tgt.teacher_idx").unwrap().dtype, DType::I64);
        assert_eq!(r.tensor_info("tgt.engine_value_mask").unwrap().dtype, DType::Bool);
        assert_eq!(r.tensor_info("tgt.legal_extra").unwrap().dtype, DType::Bits);
        assert_eq!(r.index["tensors"]["tgt.legal_extra"]["storage"], "packbits");
        let idx = r.read_i64("tgt.teacher_idx").unwrap();
        assert_eq!((idx[3], idx[4]), (-1, 4));
        let q = r.read_f32("tgt.teacher_q").unwrap();
        assert!(q[3].is_nan() && q[1] == 0.01);
        assert_eq!(r.read_bool("tgt.engine_value_mask").unwrap()[..3], [false, true, true]);
        // no temp dirs left behind; a second commit refuses to overwrite
        let names: Vec<_> = std::fs::read_dir(root.join("out")).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names.len(), 1);
        assert!(overlay(&base).commit(&out).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn symlink_and_copy_fallbacks() {
        let root = tmpdir("overlay-modes");
        let base = base_shard(&root);
        for (policy, want) in [(LinkPolicy::Copy, LinkMode::Copy), (LinkPolicy::Symlink, LinkMode::Symlink)] {
            if cfg!(not(unix)) && want == LinkMode::Symlink {
                continue;
            }
            let out = root.join(format!("{policy:?}")).join("toy_00000");
            assert_eq!(overlay(&base).link_policy(policy).commit(&out).unwrap(), want);
            let meta = std::fs::symlink_metadata(out.join("obs.x.npy")).unwrap();
            assert_eq!(meta.file_type().is_symlink(), want == LinkMode::Symlink);
            let r = ShardReader::open(&out).unwrap();
            assert_eq!(r.index["overlay"]["link"], want.as_str());
            assert_eq!(r.read_f32("obs.x").unwrap(), base.read_f32("obs.x").unwrap());
        }
        // an overlay of a symlinked shard directory links the real files
        #[cfg(unix)]
        {
            let alias = root.join("alias");
            std::os::unix::fs::symlink(&base.dir, &alias).unwrap();
            let via = ShardReader::open(&alias).unwrap();
            let out = root.join("via").join("toy_00000");
            assert_eq!(overlay(&via).commit(&out).unwrap(), LinkMode::Hardlink);
            assert!(!std::fs::symlink_metadata(out.join("obs.x.npy")).unwrap().file_type().is_symlink());
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn rejects_bad_columns() {
        let root = tmpdir("overlay-bad");
        let base = base_shard(&root);
        let n = base.n_frames;
        let mut ov = OverlayWriter::new(&base);
        assert!(ov.add_f32("tgt.short", &[], vec![0.0; n - 1]).is_err());
        assert!(ov.add_f32("tgt.ragged", &[2], vec![0.0; 2 * n + 1]).is_err());
        assert!(ov.add_f32("tgt.y", &[], vec![0.0; n]).is_err(), "collides with a base tensor");
        assert!(ov.add_f32("teacher", &[], vec![0.0; n]).is_err());
        assert!(ov.add_f32("foo.bar", &[], vec![0.0; n]).is_err());
        assert!(ov.add_f32("tgt.a/b", &[], vec![0.0; n]).is_err());
        assert!(ov.add_bits("tgt.half", &[], &vec![0.5; n]).is_err());
        ov.add_f32("tgt.ok", &[], vec![0.0; n]).unwrap();
        assert!(ov.add_f32("tgt.ok", &[], vec![0.0; n]).is_err());
        assert!(ov.set_field("samples", json!([])).is_err());
        // a failing commit leaves nothing behind
        let out = root.join("out").join("toy_00000");
        std::fs::remove_file(base.dir.join("obs.x.npy")).unwrap();
        assert!(ov.commit(&out).is_err());
        assert!(!out.exists());
        assert_eq!(std::fs::read_dir(root.join("out")).unwrap().count(), 0);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
