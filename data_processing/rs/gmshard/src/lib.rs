//! `gm-shard.v1` — the on-disk format every Rust converter writes.
//!
//! A shard is a directory:
//!
//! ```text
//! <out>/<game>_00012/
//!   index.json          format, game, converter_version, shard meta,
//!                       tensor table, per-sample (start, length, meta)
//!   obs.board_seq.npy   float32 [N, 14, 8, 8]   (N = frames of all samples)
//!   tgt.from_idx.npy    int64   [N]
//!   ...
//! ```
//!
//! Every tensor's leading axis is the frame axis; sample `i` owns rows
//! `start..start+length` of every tensor. Tensor names are `obs.<key>` /
//! `tgt.<key>` — exactly the keys the game's `ObservationEncoder` and
//! `ActionDecoder.il_loss` expect. `data_processing/games/rs_shards.py`
//! turns a shard directory into the `.pt` shards `GameShardDataset` reads,
//! or serves it directly. No dependency on NumPy or Torch on this side;
//! `.npy` is written by hand.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub mod npy;

/// Element type of a tensor column, plus the torch dtype the Python side should cast to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I64,
    I32,
    I8,
    U8,
    U16,
    Bool, // stored as u8
    Bits, // logical float32 {0,1}, stored packed
}

impl DType {
    fn torch(&self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::I8 => "int8",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::Bool => "bool",
            DType::Bits => "float32",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Column {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    Bool(Vec<u8>),
    /// {0,1}-valued float32 tensor stored bit-packed per frame (numpy `packbits`, big bit
    /// order); `usize` = logical elements per frame. Loaded back as float32.
    Bits(Vec<u8>, usize),
}

impl Column {
    pub fn new(dt: DType) -> Self {
        match dt {
            DType::F32 => Column::F32(Vec::new()),
            DType::I64 => Column::I64(Vec::new()),
            DType::I32 => Column::I32(Vec::new()),
            DType::I8 => Column::I8(Vec::new()),
            DType::U8 => Column::U8(Vec::new()),
            DType::U16 => Column::U16(Vec::new()),
            DType::Bool => Column::Bool(Vec::new()),
            DType::Bits => Column::Bits(Vec::new(), 0),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Column::F32(_) => DType::F32,
            Column::I64(_) => DType::I64,
            Column::I32(_) => DType::I32,
            Column::I8(_) => DType::I8,
            Column::U8(_) => DType::U8,
            Column::U16(_) => DType::U16,
            Column::Bool(_) => DType::Bool,
            Column::Bits(..) => DType::Bits,
        }
    }

    /// Logical element count (Bits: frames × elements per frame).
    pub fn len(&self) -> usize {
        match self {
            Column::F32(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::I32(v) => v.len(),
            Column::I8(v) => v.len(),
            Column::U8(v) => v.len(),
            Column::U16(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Bits(v, per) => if *per == 0 { 0 } else { v.len() / per.div_ceil(8) * per },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn append(&mut self, other: &Column) -> Result<()> {
        match (self, other) {
            (Column::F32(a), Column::F32(b)) => a.extend_from_slice(b),
            (Column::I64(a), Column::I64(b)) => a.extend_from_slice(b),
            (Column::I32(a), Column::I32(b)) => a.extend_from_slice(b),
            (Column::I8(a), Column::I8(b)) => a.extend_from_slice(b),
            (Column::U8(a), Column::U8(b)) => a.extend_from_slice(b),
            (Column::U16(a), Column::U16(b)) => a.extend_from_slice(b),
            (Column::Bool(a), Column::Bool(b)) => a.extend_from_slice(b),
            (Column::Bits(a, pa), Column::Bits(b, pb)) => {
                if *pa == 0 {
                    *pa = *pb;
                }
                if pa != pb {
                    bail!("bit column frame size mismatch");
                }
                a.extend_from_slice(b)
            }
            _ => bail!("column dtype mismatch"),
        }
        Ok(())
    }

    fn write_npy(&self, path: &Path, shape: &[usize]) -> Result<()> {
        match self {
            Column::F32(v) => npy::write(path, shape, v),
            Column::I64(v) => npy::write(path, shape, v),
            Column::I32(v) => npy::write(path, shape, v),
            Column::I8(v) => npy::write(path, shape, v),
            Column::U8(v) | Column::Bool(v) => npy::write(path, shape, v),
            Column::U16(v) => npy::write(path, shape, v),
            Column::Bits(v, per) => npy::write(path, &[shape[0], per.div_ceil(8)], v),
        }
    }
}

/// One tensor: name (`obs.x` / `tgt.y`), per-frame shape, data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub frame_shape: Vec<usize>,
    pub col: Column,
}

impl Tensor {
    pub fn frame_len(&self) -> usize {
        self.frame_shape.iter().product::<usize>().max(1)
    }

    pub fn frames(&self) -> usize {
        self.col.len() / self.frame_len()
    }
}

/// A sample = one sequence (T frames) of every tensor + free-form meta.
#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub tensors: Vec<Tensor>,
    pub meta: Value,
}

impl Sample {
    pub fn new(meta: Value) -> Self {
        Self { tensors: Vec::new(), meta }
    }

    pub fn f32(mut self, name: &str, frame_shape: &[usize], data: Vec<f32>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::F32(data) });
        self
    }

    pub fn i64(mut self, name: &str, frame_shape: &[usize], data: Vec<i64>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::I64(data) });
        self
    }

    pub fn bool(mut self, name: &str, frame_shape: &[usize], data: Vec<bool>) -> Self {
        self.tensors.push(Tensor {
            name: name.into(),
            frame_shape: frame_shape.to_vec(),
            col: Column::Bool(data.into_iter().map(|b| b as u8).collect()),
        });
        self
    }

    /// {0,1} float32 tensor stored bit-packed (32× smaller than f32 on disk).
    /// Panics if a value is not exactly 0.0 or 1.0.
    pub fn bits(mut self, name: &str, frame_shape: &[usize], data: &[f32]) -> Self {
        let per = frame_shape.iter().product::<usize>().max(1);
        assert!(data.len() % per == 0, "bits tensor {name}: length not a multiple of the frame size");
        let nb = per.div_ceil(8);
        let mut packed = vec![0u8; data.len() / per * nb];
        for (f, frame) in data.chunks(per).enumerate() {
            for (i, &x) in frame.iter().enumerate() {
                assert!(x == 0.0 || x == 1.0, "bits tensor {name}: value {x} is not 0/1");
                if x == 1.0 {
                    packed[f * nb + i / 8] |= 0x80 >> (i % 8);
                }
            }
        }
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::Bits(packed, per) });
        self
    }

    pub fn u8(mut self, name: &str, frame_shape: &[usize], data: Vec<u8>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::U8(data) });
        self
    }

    pub fn i32(mut self, name: &str, frame_shape: &[usize], data: Vec<i32>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::I32(data) });
        self
    }

    /// Number of frames (all tensors must agree).
    pub fn frames(&self) -> Result<usize> {
        let mut t: Option<usize> = None;
        for x in &self.tensors {
            if x.col.len() % x.frame_len() != 0 {
                bail!("tensor {} length {} not a multiple of frame size {}", x.name, x.col.len(), x.frame_len());
            }
            let n = x.frames();
            match t {
                None => t = Some(n),
                Some(m) if m != n => bail!("tensor {} has {} frames, expected {}", x.name, n, m),
                _ => {}
            }
        }
        t.context("sample has no tensors")
    }
}

#[derive(Serialize)]
struct TensorInfo {
    file: String,
    dtype: &'static str,
    frame_shape: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage: Option<&'static str>,
}

#[derive(Serialize)]
struct SampleInfo {
    start: usize,
    length: usize,
    meta: Value,
}

#[derive(Serialize)]
struct Index<'a> {
    format: &'static str,
    game: &'a str,
    converter_version: u32,
    meta: &'a Value,
    n_frames: usize,
    tensors: BTreeMap<String, TensorInfo>,
    samples: &'a [SampleInfo],
}

/// Accumulates samples and writes `<out>/<game>_<NNNNN>/` directories.
pub struct ShardWriter {
    out: PathBuf,
    game: String,
    converter_version: u32,
    meta: Value,
    samples_per_shard: usize,
    next_idx: usize,
    cols: BTreeMap<String, Tensor>,
    infos: Vec<SampleInfo>,
    frames: usize,
    pub written_shards: usize,
    pub written_samples: usize,
    pub written_frames: usize,
}

impl ShardWriter {
    pub fn new(out: &Path, game: &str, converter_version: u32, meta: Value, samples_per_shard: usize) -> Result<Self> {
        std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
        // resume-safe numbering: continue after existing shards of this game
        let mut next_idx = 0;
        for e in std::fs::read_dir(out)? {
            let name = e?.file_name().to_string_lossy().to_string();
            if let Some(rest) = name.strip_prefix(&format!("{game}_")) {
                if let Ok(i) = rest.parse::<usize>() {
                    next_idx = next_idx.max(i + 1);
                }
            }
        }
        Ok(Self {
            out: out.to_path_buf(),
            game: game.to_string(),
            converter_version,
            meta,
            samples_per_shard: samples_per_shard.max(1),
            next_idx,
            cols: BTreeMap::new(),
            infos: Vec::new(),
            frames: 0,
            written_shards: 0,
            written_samples: 0,
            written_frames: 0,
        })
    }

    pub fn push(&mut self, s: Sample) -> Result<()> {
        let t = s.frames()?;
        // the first sample fixes the schema for the writer's whole lifetime, so
        // every shard of one conversion has the same tensor set
        if self.cols.is_empty() {
            for x in &s.tensors {
                self.cols.insert(
                    x.name.clone(),
                    Tensor { name: x.name.clone(), frame_shape: x.frame_shape.clone(), col: Column::new(x.col.dtype()) },
                );
            }
        }
        if s.tensors.len() != self.cols.len() || s.tensors.iter().any(|x| !self.cols.contains_key(&x.name)) {
            bail!("sample tensor set differs from the shard's ({:?})", self.cols.keys().collect::<Vec<_>>());
        }
        for x in &s.tensors {
            let c = self.cols.get_mut(&x.name).unwrap();
            if c.frame_shape != x.frame_shape {
                bail!("tensor {} frame shape {:?} != shard's {:?}", x.name, x.frame_shape, c.frame_shape);
            }
            c.col.append(&x.col)?;
        }
        self.infos.push(SampleInfo { start: self.frames, length: t, meta: s.meta });
        self.frames += t;
        if self.infos.len() >= self.samples_per_shard {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.infos.is_empty() {
            return Ok(());
        }
        let name = format!("{}_{:05}", self.game, self.next_idx);
        let dir = self.out.join(&name);
        let tmp = self.out.join(format!(".{name}.tmp"));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp)?;
        let mut tensors = BTreeMap::new();
        for (tname, t) in &self.cols {
            let file = format!("{tname}.npy");
            let mut shape = vec![self.frames];
            shape.extend_from_slice(&t.frame_shape);
            t.col.write_npy(&tmp.join(&file), &shape)?;
            let storage = matches!(t.col, Column::Bits(..)).then_some("packbits");
            tensors.insert(tname.clone(), TensorInfo { file, dtype: t.col.dtype().torch(), frame_shape: t.frame_shape.clone(), storage });
        }
        let index = Index {
            format: "gm-shard.v1",
            game: &self.game,
            converter_version: self.converter_version,
            meta: &self.meta,
            n_frames: self.frames,
            tensors,
            samples: &self.infos,
        };
        let mut f = std::fs::File::create(tmp.join("index.json"))?;
        f.write_all(&serde_json::to_vec(&index)?)?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::rename(&tmp, &dir)?;
        self.written_shards += 1;
        self.written_samples += self.infos.len();
        self.written_frames += self.frames;
        self.next_idx += 1;
        self.infos.clear();
        self.frames = 0;
        for t in self.cols.values_mut() {
            t.col = Column::new(t.col.dtype());
        }
        Ok(())
    }
}

impl Drop for ShardWriter {
    fn drop(&mut self) {
        if !self.infos.is_empty() {
            eprintln!("warning: ShardWriter dropped with {} unflushed samples (call flush())", self.infos.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_index_and_arrays() {
        let dir = std::env::temp_dir().join(format!("gmshard-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::new(&dir, "toy", 3, serde_json::json!({"k": 1}), 2).unwrap();
        for i in 0..3 {
            let t = 2 + i;
            let s = Sample::new(serde_json::json!({"i": i}))
                .f32("obs.x", &[2, 2], vec![i as f32; t * 4])
                .i64("tgt.y", &[], (0..t as i64).collect())
                .bool("tgt.b", &[], vec![true; t])
                .bits("obs.p", &[3, 3], &vec![1.0; t * 9]);
            w.push(s).unwrap();
        }
        w.flush().unwrap();
        assert_eq!(w.written_shards, 2);
        assert_eq!(w.written_samples, 3);
        let idx: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("toy_00000").join("index.json")).unwrap()).unwrap();
        assert_eq!(idx["format"], "gm-shard.v1");
        assert_eq!(idx["n_frames"], 5);
        assert_eq!(idx["samples"][1]["start"], 2);
        assert_eq!(idx["tensors"]["tgt.b"]["dtype"], "bool");
        assert!(dir.join("toy_00001").join("obs.x.npy").exists());
        assert_eq!(idx["tensors"]["obs.p"]["storage"], "packbits");
        assert_eq!(idx["tensors"]["obs.p"]["dtype"], "float32");
        // mismatched tensor set is rejected
        let bad = Sample::new(Value::Null).f32("obs.x", &[2, 2], vec![0.0; 4]);
        assert!(w.push(bad).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
