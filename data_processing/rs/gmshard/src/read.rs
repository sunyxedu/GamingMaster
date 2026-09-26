//! Reading `gm-shard.v1` directories back: [`ShardReader`].
//!
//! ```no_run
//! # use gmshard::ShardReader;
//! # fn main() -> anyhow::Result<()> {
//! let shard = ShardReader::open("data/chess/chess_00000".as_ref())?;
//! let from = shard.read("tgt.from_idx")?;          // whole column, as stored
//! let planes = shard.read("obs.board_seq")?;       // packbits column (Bits)
//! for i in 0..shard.n_samples() {
//!     let r = shard.sample_range(i);               // this sample's frames
//!     let labels = from.rows_i64(r.start, r.len())?;
//!     let board = planes.rows_f32(r.start, r.len())?; // unpacked {0,1} float32
//! #   let _ = (labels, board);
//! }
//! # Ok(()) }
//! ```
//!
//! Every column is validated against its index entry on read (dtype ↔ `.npy`
//! descriptor, shape = [n_frames, frame_shape…], packed [n_frames, ⌈elements/8⌉]).
//! The whole `index.json` is kept in [`ShardReader::index`] so derived shards
//! ([`crate::OverlayWriter`]) preserve fields this crate does not know about.

use crate::{npy, Column, DType, Tensor, FORMAT};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// One tensor-table entry of `index.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub file: String,
    /// Column type; [`DType::Bits`] for `"storage": "packbits"`.
    pub dtype: DType,
    pub frame_shape: Vec<usize>,
}

impl TensorInfo {
    /// Logical elements per frame.
    pub fn frame_len(&self) -> usize {
        self.frame_shape.iter().product::<usize>().max(1)
    }
}

/// One sample of `index.json`: rows `start..start+length` of every tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleInfo {
    pub start: usize,
    pub length: usize,
    pub meta: Value,
}

impl SampleInfo {
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.length
    }
}

/// An opened shard directory: parsed index, tensor table and sample ranges.
/// Columns are read on demand ([`ShardReader::read`]).
#[derive(Debug, Clone)]
pub struct ShardReader {
    pub dir: PathBuf,
    /// The complete `index.json`.
    pub index: Value,
    pub game: String,
    pub converter_version: u32,
    pub n_frames: usize,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub samples: Vec<SampleInfo>,
}

fn usize_field(v: &Value, key: &str, ctx: &str) -> Result<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize).with_context(|| format!("{ctx}: missing/invalid {key:?}"))
}

impl ShardReader {
    /// Whether `dir` holds a `gm-shard.v1` index (cheap: parses index.json only).
    pub fn is_shard(dir: &Path) -> bool {
        std::fs::read(dir.join("index.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .map(|v| v.get("format").and_then(Value::as_str) == Some(FORMAT))
            .unwrap_or(false)
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("index.json");
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let index: Value = serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        let ctx = path.display().to_string();
        if index.get("format").and_then(Value::as_str) != Some(FORMAT) {
            bail!("{ctx}: not a {FORMAT} shard");
        }
        let game = index.get("game").and_then(Value::as_str).with_context(|| format!("{ctx}: no game"))?.to_string();
        let converter_version = index.get("converter_version").and_then(Value::as_u64).unwrap_or(1) as u32;
        let n_frames = usize_field(&index, "n_frames", &ctx)?;
        let mut tensors = BTreeMap::new();
        let table = index.get("tensors").and_then(Value::as_object).with_context(|| format!("{ctx}: no tensor table"))?;
        for (name, e) in table {
            let file = e.get("file").and_then(Value::as_str).with_context(|| format!("{ctx}: {name}: no file"))?;
            if file.contains('/') || file.contains('\\') || file.starts_with('.') {
                bail!("{ctx}: {name}: tensor file {file:?} is not a plain file name");
            }
            let dtype = e.get("dtype").and_then(Value::as_str).with_context(|| format!("{ctx}: {name}: no dtype"))?;
            let storage = e.get("storage").and_then(Value::as_str);
            let dtype = DType::from_index(dtype, storage).with_context(|| format!("{ctx}: {name}"))?;
            let frame_shape = e
                .get("frame_shape")
                .and_then(Value::as_array)
                .with_context(|| format!("{ctx}: {name}: no frame_shape"))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize).with_context(|| format!("{ctx}: {name}: bad frame_shape")))
                .collect::<Result<Vec<_>>>()?;
            tensors.insert(name.clone(), TensorInfo { name: name.clone(), file: file.to_string(), dtype, frame_shape });
        }
        let mut samples = Vec::new();
        for (i, s) in index.get("samples").and_then(Value::as_array).with_context(|| format!("{ctx}: no samples"))?.iter().enumerate() {
            let sctx = format!("{ctx}: sample {i}");
            let (start, length) = (usize_field(s, "start", &sctx)?, usize_field(s, "length", &sctx)?);
            if start + length > n_frames {
                bail!("{sctx}: frames {start}..{} beyond n_frames {n_frames}", start + length);
            }
            samples.push(SampleInfo { start, length, meta: s.get("meta").cloned().unwrap_or(Value::Null) });
        }
        Ok(Self { dir: dir.to_path_buf(), index, game, converter_version, n_frames, tensors, samples })
    }

    /// Shard-level converter meta (`index.json` "meta"; `Null` when absent).
    pub fn meta(&self) -> &Value {
        self.index.get("meta").unwrap_or(&Value::Null)
    }

    /// Directory name (e.g. `chess_00012`).
    pub fn name(&self) -> String {
        self.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn sample_range(&self, i: usize) -> Range<usize> {
        self.samples[i].range()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("{}: no tensor {name:?}", self.dir.display()))
    }

    /// One whole column, as stored (Bits stay packed; see the `Tensor::rows_*` accessors).
    pub fn read(&self, name: &str) -> Result<Tensor> {
        let info = self.tensor_info(name)?;
        let path = self.dir.join(&info.file);
        let a = npy::read(&path)?;
        let per = info.frame_len();
        let want: Vec<usize> = match info.dtype {
            DType::Bits => vec![self.n_frames, per.div_ceil(8)],
            _ => std::iter::once(self.n_frames).chain(info.frame_shape.iter().copied()).collect(),
        };
        if a.shape != want {
            bail!("{}: shape {:?}, index says {:?}", path.display(), a.shape, want);
        }
        let col = Column::from_npy(info.dtype, &a, per).with_context(|| format!("{}", path.display()))?;
        Ok(Tensor { name: name.to_string(), frame_shape: info.frame_shape.clone(), col })
    }

    /// Every column of the shard.
    pub fn read_all(&self) -> Result<BTreeMap<String, Tensor>> {
        self.tensors.keys().map(|n| Ok((n.clone(), self.read(n)?))).collect()
    }

    /// A whole column as float32 (F32, or Bits unpacked).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.read(name)?.rows_f32(0, self.n_frames)?.into_owned())
    }

    /// A whole column as int64 (any integer / bool column, widened).
    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.read(name)?.rows_i64(0, self.n_frames)?.into_owned())
    }

    /// A whole column as bools (Bool / U8 ≠ 0 / Bits).
    pub fn read_bool(&self, name: &str) -> Result<Vec<bool>> {
        self.read(name)?.rows_bool(0, self.n_frames)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{Sample, ShardWriter};
    use crate::testutil::tmpdir;
    use serde_json::json;

    /// Sample `i` with T = 2 + i frames of every dtype the writer emits.
    pub(crate) fn sample(i: usize) -> Sample {
        let t = 2 + i;
        let bits: Vec<f32> = (0..t * 11).map(|k| ((k * 7 + i) % 3 == 0) as u8 as f32).collect();
        Sample::new(json!({"i": i}))
            .f32("obs.x", &[2, 3], (0..t * 6).map(|k| k as f32 * 0.5 - i as f32).collect())
            .bits("obs.p", &[11], &bits)
            .i64("tgt.y", &[], (0..t as i64).map(|k| k - 3 + i as i64 * 1000).collect())
            .i64("tgt.k", &[4], (0..t as i64 * 4).map(|k| if k % 3 == 0 { -1 } else { k }).collect())
            .bool("tgt.b", &[], (0..t).map(|k| k % 2 == 1).collect())
            .i32("tgt.i32", &[], (0..t as i32).map(|k| -k).collect())
            .i8("tgt.i8", &[], (0..t as i8).map(|k| -k).collect())
            .u8("raw.u8", &[2], (0..t as u8 * 2).collect())
            .u16("raw.u16", &[], (0..t as u16).map(|k| k * 300).collect())
    }

    #[test]
    fn roundtrip_against_writer() {
        let dir = tmpdir("read");
        let mut w = ShardWriter::new(&dir, "toy", 7, json!({"board": 3}), 3).unwrap();
        for i in 0..4 {
            w.push(sample(i)).unwrap();
        }
        w.flush().unwrap();
        assert_eq!(w.written_shards, 2);
        let r = ShardReader::open(&dir.join("toy_00000")).unwrap();
        assert!(ShardReader::is_shard(&r.dir) && !ShardReader::is_shard(&dir));
        assert_eq!((r.game.as_str(), r.converter_version, r.n_frames, r.n_samples()), ("toy", 7, 2 + 3 + 4, 3));
        assert_eq!(r.meta(), &json!({"board": 3}));
        assert_eq!(r.name(), "toy_00000");
        assert_eq!(r.tensor_info("obs.p").unwrap().dtype, DType::Bits);
        assert_eq!(r.tensor_info("tgt.b").unwrap().dtype, DType::Bool);
        let all = r.read_all().unwrap();
        assert_eq!(all.len(), 9);
        for i in 0..3 {
            let rg = r.sample_range(i);
            assert_eq!(r.samples[i].meta, json!({"i": i}));
            let want = sample(i);
            for t in &want.tensors {
                let got = &all[&t.name];
                assert_eq!(got.frame_shape, t.frame_shape, "{}", t.name);
                assert_eq!(got.col.dtype(), t.col.dtype(), "{}", t.name);
                let sl = got.slice(rg.start, rg.len()).unwrap();
                match (&sl.col, &t.col) {
                    (Column::F32(a), Column::F32(b)) => assert_eq!(a, b),
                    (Column::I64(a), Column::I64(b)) => assert_eq!(a, b),
                    (Column::I32(a), Column::I32(b)) => assert_eq!(a, b),
                    (Column::I8(a), Column::I8(b)) => assert_eq!(a, b),
                    (Column::U8(a), Column::U8(b)) | (Column::Bool(a), Column::Bool(b)) => assert_eq!(a, b),
                    (Column::U16(a), Column::U16(b)) => assert_eq!(a, b),
                    (Column::Bits(a, pa), Column::Bits(b, pb)) => assert_eq!((a, pa), (b, pb)),
                    _ => panic!("{} dtype changed", t.name),
                }
            }
            // typed row accessors
            let bits = all["obs.p"].rows_f32(rg.start, rg.len()).unwrap();
            let Column::Bits(packed, _) = &want.tensors[1].col else { panic!() };
            assert_eq!(crate::pack_bits(11, &bits).unwrap(), *packed);
            all["obs.p"].check_rows_f32(rg.start, &bits).unwrap();
            let y = all["tgt.y"].rows_i64(rg.start, rg.len()).unwrap();
            assert_eq!(y[0], -3 + i as i64 * 1000);
            let b = all["tgt.b"].rows_bool(rg.start, rg.len()).unwrap();
            assert_eq!(b, (0..rg.len()).map(|k| k % 2 == 1).collect::<Vec<_>>());
            let wide = all["tgt.i8"].rows_i64(rg.start, rg.len()).unwrap();
            assert_eq!(wide.last().copied(), Some(-(rg.len() as i64 - 1)));
        }
        // whole-column conveniences
        assert_eq!(r.read_i64("tgt.y").unwrap().len(), r.n_frames);
        assert_eq!(r.read_f32("obs.p").unwrap().len(), r.n_frames * 11);
        assert_eq!(r.read_bool("tgt.b").unwrap().len(), r.n_frames);
        assert!(r.read_f32("tgt.y").is_err() && r.read("tgt.nope").is_err());
        // bit-exact check reports the first mismatch
        let mut x = all["obs.x"].rows_f32(0, 2).unwrap().into_owned();
        all["obs.x"].check_rows_f32(0, &x).unwrap();
        x[7] = -x[7] + 0.25;
        let e = all["obs.x"].check_rows_f32(0, &x).unwrap_err().to_string();
        assert!(e.contains("frame 1 element 1"), "{e}");
        let neg_zero: Vec<f32> = x.iter().map(|_| -0.0).collect();
        assert!(all["obs.x"].check_rows_f32(0, &neg_zero).is_err());
        let r2 = ShardReader::open(&dir.join("toy_00001")).unwrap();
        assert_eq!((r2.n_samples(), r2.sample_range(0)), (1, 0..5));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_inconsistent_shards() {
        let dir = tmpdir("read-bad");
        let mut w = ShardWriter::new(&dir, "toy", 1, Value::Null, 10).unwrap();
        w.push(sample(0)).unwrap();
        w.flush().unwrap();
        let d = dir.join("toy_00000");
        let idx_path = d.join("index.json");
        let good: Value = serde_json::from_slice(&std::fs::read(&idx_path).unwrap()).unwrap();
        let write = |v: &Value| std::fs::write(&idx_path, serde_json::to_vec(v).unwrap()).unwrap();
        // frame shape disagreeing with the .npy
        let mut v = good.clone();
        v["tensors"]["obs.x"]["frame_shape"] = json!([3, 3]);
        write(&v);
        assert!(ShardReader::open(&d).unwrap().read("obs.x").is_err());
        // dtype disagreeing with the .npy descriptor
        let mut v = good.clone();
        v["tensors"]["tgt.y"]["dtype"] = json!("int32");
        write(&v);
        assert!(ShardReader::open(&d).unwrap().read("tgt.y").is_err());
        // sample beyond n_frames, foreign format, path traversal
        let mut v = good.clone();
        v["samples"][0]["length"] = json!(99);
        write(&v);
        assert!(ShardReader::open(&d).is_err());
        let mut v = good.clone();
        v["format"] = json!("other");
        write(&v);
        assert!(ShardReader::open(&d).is_err() && !ShardReader::is_shard(&d));
        let mut v = good.clone();
        v["tensors"]["obs.x"]["file"] = json!("../obs.x.npy");
        write(&v);
        assert!(ShardReader::open(&d).is_err());
        let mut v = good;
        v["tensors"]["obs.x"]["storage"] = json!("zstd");
        write(&v);
        assert!(ShardReader::open(&d).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
