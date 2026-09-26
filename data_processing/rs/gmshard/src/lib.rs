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
//! `ActionDecoder.il_loss` expect — or `raw.<key>` for compact per-frame
//! fields that a registered loader expander turns into `obs` at load time. `data_processing/games/dataset.py::GameShardDataset`
//! reads shard directories directly (memory-mapped). No dependency on NumPy or Torch on this side;
//! `.npy` is written (and read back) by hand.
//!
//! Besides the [`ShardWriter`] the crate has a [`ShardReader`] (index + every column
//! dtype the writer emits, packed bit columns included, per-sample frame ranges, typed
//! row accessors on [`Tensor`]) and an [`OverlayWriter`], which derives a new shard from
//! an existing one by linking its tensor files and adding columns (engine teacher labels,
//! see `gmdata label`).

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub mod npy;
pub mod overlay;
pub mod read;

pub use overlay::{LinkMode, LinkPolicy, OverlayWriter};
pub use read::{SampleInfo, ShardReader, TensorInfo};

/// `index.json` "format" of every shard this crate writes.
pub const FORMAT: &str = "gm-shard.v1";

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
    /// The `index.json` "dtype" (the torch dtype the Python loader produces).
    pub fn torch(&self) -> &'static str {
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

    /// The `index.json` "storage" field (`Some("packbits")` for [`DType::Bits`]).
    pub fn storage(&self) -> Option<&'static str> {
        matches!(self, DType::Bits).then_some("packbits")
    }

    /// Inverse of ([`DType::torch`], [`DType::storage`]): the column type of an index entry.
    pub fn from_index(dtype: &str, storage: Option<&str>) -> Result<Self> {
        Ok(match (dtype, storage) {
            ("float32", Some("packbits")) => DType::Bits,
            (_, Some(s)) if s != "packbits" => bail!("unknown tensor storage {s:?}"),
            (d, Some(_)) => bail!("packbits storage needs dtype float32, not {d:?}"),
            ("float32", None) => DType::F32,
            ("int64", None) => DType::I64,
            ("int32", None) => DType::I32,
            ("int8", None) => DType::I8,
            ("uint8", None) => DType::U8,
            ("uint16", None) => DType::U16,
            ("bool", None) => DType::Bool,
            (d, None) => bail!("unsupported tensor dtype {d:?}"),
        })
    }
}

/// Pack a {0,1}-valued float32 tensor frame by frame (numpy `packbits` along the
/// frame axis, big bit order: element `i` of a frame is bit `0x80 >> (i % 8)` of byte
/// `i / 8`, each frame padded to whole bytes with zero bits). Errors on any other value.
pub fn pack_bits(frame_len: usize, data: &[f32]) -> Result<Vec<u8>> {
    let per = frame_len.max(1);
    if !data.len().is_multiple_of(per) {
        bail!("bit tensor length {} is not a multiple of the frame size {per}", data.len());
    }
    let nb = per.div_ceil(8);
    let mut packed = vec![0u8; data.len() / per * nb];
    for (f, frame) in data.chunks(per).enumerate() {
        for (i, &x) in frame.iter().enumerate() {
            if x == 1.0 {
                packed[f * nb + i / 8] |= 0x80 >> (i % 8);
            } else if x != 0.0 {
                bail!("bit tensor value {x} (frame {f}, element {i}) is not 0/1");
            }
        }
    }
    Ok(packed)
}

/// Inverse of [`pack_bits`]: `packed` holds whole frames of `frame_len.div_ceil(8)` bytes.
pub fn unpack_bits(frame_len: usize, packed: &[u8]) -> Vec<f32> {
    let per = frame_len.max(1);
    let nb = per.div_ceil(8);
    let mut out = Vec::with_capacity(packed.len() / nb * per);
    for frame in packed.chunks(nb) {
        for i in 0..per {
            out.push(if frame[i / 8] & (0x80 >> (i % 8)) != 0 { 1.0 } else { 0.0 });
        }
    }
    out
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

    /// Write as `.npy`; `shape` = [frames, frame_shape...] (Bits: stored [frames, bytes]).
    pub(crate) fn write_npy(&self, path: &Path, shape: &[usize]) -> Result<()> {
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

    /// A column of type `dt` from a parsed `.npy` (`frame_len` = logical elements per
    /// frame, used by Bits).
    pub fn from_npy(dt: DType, a: &npy::NpyArray, frame_len: usize) -> Result<Self> {
        Ok(match dt {
            DType::F32 => Column::F32(a.to_vec()?),
            DType::I64 => Column::I64(a.to_vec()?),
            DType::I32 => Column::I32(a.to_vec()?),
            DType::I8 => Column::I8(a.to_vec()?),
            DType::U8 => Column::U8(a.to_vec()?),
            DType::U16 => Column::U16(a.to_vec()?),
            DType::Bool => Column::Bool(a.to_vec()?),
            DType::Bits => Column::Bits(a.to_vec()?, frame_len.max(1)),
        })
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        if let Column::F32(v) = self { Some(v) } else { None }
    }

    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column::I64(v) = self { Some(v) } else { None }
    }

    /// Bool / U8 bytes (Bool: 0 or 1).
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Column::U8(v) | Column::Bool(v) => Some(v),
            _ => None,
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
    pub fn new(name: &str, frame_shape: &[usize], col: Column) -> Self {
        Self { name: name.into(), frame_shape: frame_shape.to_vec(), col }
    }

    pub fn frame_len(&self) -> usize {
        self.frame_shape.iter().product::<usize>().max(1)
    }

    pub fn frames(&self) -> usize {
        self.col.len() / self.frame_len()
    }

    fn check_rows(&self, start: usize, len: usize) -> Result<()> {
        if start + len > self.frames() {
            bail!("{}: frames {start}..{} out of range ({} frames)", self.name, start + len, self.frames());
        }
        Ok(())
    }

    /// Frames `start..start+len` as float32: F32 borrowed, Bits unpacked to {0,1}.
    pub fn rows_f32(&self, start: usize, len: usize) -> Result<Cow<'_, [f32]>> {
        self.check_rows(start, len)?;
        let per = self.frame_len();
        match &self.col {
            Column::F32(v) => Ok(Cow::Borrowed(&v[start * per..(start + len) * per])),
            Column::Bits(v, _) => {
                let nb = per.div_ceil(8);
                Ok(Cow::Owned(unpack_bits(per, &v[start * nb..(start + len) * nb])))
            }
            c => bail!("{}: {:?} tensor read as float32", self.name, c.dtype()),
        }
    }

    /// Frames `start..start+len` as int64: I64 borrowed, other integer / bool columns widened.
    pub fn rows_i64(&self, start: usize, len: usize) -> Result<Cow<'_, [i64]>> {
        self.check_rows(start, len)?;
        let per = self.frame_len();
        let r = start * per..(start + len) * per;
        Ok(match &self.col {
            Column::I64(v) => Cow::Borrowed(&v[r]),
            Column::I32(v) => Cow::Owned(v[r].iter().map(|&x| x as i64).collect()),
            Column::I8(v) => Cow::Owned(v[r].iter().map(|&x| x as i64).collect()),
            Column::U8(v) | Column::Bool(v) => Cow::Owned(v[r].iter().map(|&x| x as i64).collect()),
            Column::U16(v) => Cow::Owned(v[r].iter().map(|&x| x as i64).collect()),
            c => bail!("{}: {:?} tensor read as int64", self.name, c.dtype()),
        })
    }

    /// Frames `start..start+len` as bools: Bool / U8 (≠ 0) and Bits columns.
    pub fn rows_bool(&self, start: usize, len: usize) -> Result<Vec<bool>> {
        self.check_rows(start, len)?;
        let per = self.frame_len();
        match &self.col {
            Column::Bool(v) | Column::U8(v) => Ok(v[start * per..(start + len) * per].iter().map(|&x| x != 0).collect()),
            Column::Bits(..) => Ok(self.rows_f32(start, len)?.iter().map(|&x| x == 1.0).collect()),
            c => bail!("{}: {:?} tensor read as bool", self.name, c.dtype()),
        }
    }

    /// The packed bytes of frames `start..start+len` of a Bits column.
    pub fn rows_packed(&self, start: usize, len: usize) -> Result<&[u8]> {
        self.check_rows(start, len)?;
        match &self.col {
            Column::Bits(v, per) => {
                let nb = per.div_ceil(8);
                Ok(&v[start * nb..(start + len) * nb])
            }
            c => bail!("{}: {:?} tensor is not bit-packed", self.name, c.dtype()),
        }
    }

    /// Bit-exact check of re-encoded float32 frames against the stored frames starting at
    /// `start` (`expected.len()` must be whole frames). F32 columns compare `f32::to_bits`,
    /// Bits columns the unpacked {0,1} values. The error names the first mismatch.
    pub fn check_rows_f32(&self, start: usize, expected: &[f32]) -> Result<()> {
        let per = self.frame_len();
        if !expected.len().is_multiple_of(per) {
            bail!("{}: {} re-encoded values are not whole frames of {per}", self.name, expected.len());
        }
        let stored = self.rows_f32(start, expected.len() / per)?;
        if let Some(i) = stored.iter().zip(expected).position(|(a, b)| a.to_bits() != b.to_bits()) {
            bail!("{}: frame {} element {}: stored {} but re-encoded {}", self.name, start + i / per, i % per,
                  stored[i], expected[i]);
        }
        Ok(())
    }

    /// Frames `start..start+len` as a new tensor (same name / dtype / frame shape).
    pub fn slice(&self, start: usize, len: usize) -> Result<Tensor> {
        self.check_rows(start, len)?;
        let per = self.frame_len();
        let r = start * per..(start + len) * per;
        let col = match &self.col {
            Column::F32(v) => Column::F32(v[r].to_vec()),
            Column::I64(v) => Column::I64(v[r].to_vec()),
            Column::I32(v) => Column::I32(v[r].to_vec()),
            Column::I8(v) => Column::I8(v[r].to_vec()),
            Column::U8(v) => Column::U8(v[r].to_vec()),
            Column::U16(v) => Column::U16(v[r].to_vec()),
            Column::Bool(v) => Column::Bool(v[r].to_vec()),
            Column::Bits(..) => Column::Bits(self.rows_packed(start, len)?.to_vec(), per),
        };
        Ok(Tensor { name: self.name.clone(), frame_shape: self.frame_shape.clone(), col })
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
        let packed = pack_bits(per, data).unwrap_or_else(|e| panic!("bits tensor {name}: {e}"));
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::Bits(packed, per) });
        self
    }

    pub fn u8(mut self, name: &str, frame_shape: &[usize], data: Vec<u8>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::U8(data) });
        self
    }

    pub fn u16(mut self, name: &str, frame_shape: &[usize], data: Vec<u16>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::U16(data) });
        self
    }

    pub fn i8(mut self, name: &str, frame_shape: &[usize], data: Vec<i8>) -> Self {
        self.tensors.push(Tensor { name: name.into(), frame_shape: frame_shape.to_vec(), col: Column::I8(data) });
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

/// One entry of the `index.json` tensor table (as written).
#[derive(Serialize)]
pub(crate) struct IndexTensor {
    pub(crate) file: String,
    pub(crate) dtype: &'static str,
    pub(crate) frame_shape: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) storage: Option<&'static str>,
}

impl IndexTensor {
    pub(crate) fn of(t: &Tensor, file: String) -> Self {
        let dt = t.col.dtype();
        IndexTensor { file, dtype: dt.torch(), frame_shape: t.frame_shape.clone(), storage: dt.storage() }
    }
}

#[derive(Serialize)]
struct IndexSample {
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
    tensors: BTreeMap<String, IndexTensor>,
    samples: &'a [IndexSample],
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
    infos: Vec<IndexSample>,
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
        self.infos.push(IndexSample { start: self.frames, length: t, meta: s.meta });
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
            tensors.insert(tname.clone(), IndexTensor::of(t, file));
        }
        let index = Index {
            format: FORMAT,
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
pub(crate) mod testutil {
    use std::path::PathBuf;

    /// A fresh, empty directory under the system temp dir.
    pub(crate) fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "gmshard-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_packing_roundtrip() {
        let data: Vec<f32> = (0..3 * 13).map(|i| ((i * 5) % 7 < 3) as u8 as f32).collect();
        let packed = pack_bits(13, &data).unwrap();
        assert_eq!(packed.len(), 3 * 2);
        assert_eq!(unpack_bits(13, &packed), data);
        // first element of a frame is the high bit of its first byte
        assert_eq!(pack_bits(3, &[1.0, 0.0, 1.0]).unwrap(), vec![0b1010_0000]);
        assert!(pack_bits(3, &[0.5, 0.0, 1.0]).is_err());
        assert!(pack_bits(3, &[0.0; 4]).is_err());
        for dt in [DType::F32, DType::I64, DType::I32, DType::I8, DType::U8, DType::U16, DType::Bool, DType::Bits] {
            assert_eq!(DType::from_index(dt.torch(), dt.storage()).unwrap(), dt);
        }
    }

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
