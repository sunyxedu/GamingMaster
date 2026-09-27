//! Tensor schemas of decision frames. Every decision frame of a gmrules game is a fixed
//! set of named, typed, fixed-shape tensors in three groups (declared by the game's
//! [`crate::FrameSpec`]):
//!
//! - `obs`: what the acting seat observes — the gm-shard `obs.<name>` tensors and the
//!   simulator's observation (the network input);
//! - `legal`: the legality masks the RL decoder samples under (the env's `legal`
//!   dict); never written to shards (IL legality, if any, is a target);
//! - `target`: the IL labels of the TAKEN action (gm-shard `tgt.<name>`); the spec's
//!   `action_keys` (i64 scalars among them) ARE the policy's action: what the
//!   simulator's `step` takes, the decoder's replay keys.
//!
//! [`TensorSpec`] describes a tensor, [`Buf`] owns its data (one frame, or a batch of
//! rows), [`BufMut`] / [`BufRef`] view it, and [`TensorsMut`] is what encoders write
//! into (one frame's tensors of one group, in schema order).

use serde::Serialize;
use std::ops::Range;

/// Element type of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    /// one byte per element, 0 or 1
    Bool,
    I32,
    I64,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Bool => "bool",
            DType::I32 => "i32",
            DType::I64 => "i64",
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::Bool => 1,
            DType::I64 => 8,
        }
    }
}

/// One tensor of a frame: `shape` is PER FRAME (`[]` = one scalar per frame).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// An f32 tensor holding only 0.0 / 1.0: stored bit-packed in gm-shards (32x
    /// smaller, loaded back as float32); the conformance battery checks the values.
    pub bits: bool,
}

impl TensorSpec {
    pub fn new(name: &str, shape: &[usize], dtype: DType) -> Self {
        TensorSpec { name: name.to_string(), shape: shape.to_vec(), dtype, bits: false }
    }

    pub fn f32(name: &str, shape: &[usize]) -> Self {
        Self::new(name, shape, DType::F32)
    }

    /// An f32 tensor of 0.0 / 1.0 values, stored bit-packed.
    pub fn bits(name: &str, shape: &[usize]) -> Self {
        TensorSpec { bits: true, ..Self::new(name, shape, DType::F32) }
    }

    pub fn bool(name: &str, shape: &[usize]) -> Self {
        Self::new(name, shape, DType::Bool)
    }

    pub fn i32(name: &str, shape: &[usize]) -> Self {
        Self::new(name, shape, DType::I32)
    }

    pub fn i64(name: &str, shape: &[usize]) -> Self {
        Self::new(name, shape, DType::I64)
    }

    /// Elements per frame.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Index of tensor `name` in `specs`.
pub fn index_of(specs: &[TensorSpec], name: &str) -> Option<usize> {
    specs.iter().position(|t| t.name == name)
}

/// The data of one tensor (one frame, or `rows` frames back to back).
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    F32(Vec<f32>),
    Bool(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

/// A mutable view of (part of) a [`Buf`].
#[derive(Debug)]
pub enum BufMut<'a> {
    F32(&'a mut [f32]),
    Bool(&'a mut [u8]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
}

/// A shared view of (part of) a [`Buf`].
#[derive(Clone, Copy, Debug)]
pub enum BufRef<'a> {
    F32(&'a [f32]),
    Bool(&'a [u8]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

macro_rules! each {
    ($v:expr, $x:ident => $e:expr) => {
        match $v {
            Self::F32($x) => $e,
            Self::Bool($x) => $e,
            Self::I32($x) => $e,
            Self::I64($x) => $e,
        }
    };
}

fn raw_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only instantiated with f32 / u8 / i32 / i64 (no padding, every bit
    // pattern valid); u8 has alignment 1
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// First index where two slices differ bit-exactly (f32 by `to_bits`: -0.0 != 0.0).
fn first_diff<T: Copy, K: PartialEq>(a: &[T], b: &[T], key: impl Fn(T) -> K) -> Option<usize> {
    a.iter().zip(b).position(|(&x, &y)| key(x) != key(y))
}

impl Buf {
    pub fn zeros(dtype: DType, len: usize) -> Buf {
        match dtype {
            DType::F32 => Buf::F32(vec![0.0; len]),
            DType::Bool => Buf::Bool(vec![0; len]),
            DType::I32 => Buf::I32(vec![0; len]),
            DType::I64 => Buf::I64(vec![0; len]),
        }
    }

    /// Zeroed buffers of one frame of every tensor in `specs`.
    pub fn frame(specs: &[TensorSpec]) -> Vec<Buf> {
        specs.iter().map(|t| Buf::zeros(t.dtype, t.numel())).collect()
    }

    /// Zeroed buffers of `rows` frames of every tensor in `specs`.
    pub fn rows(specs: &[TensorSpec], rows: usize) -> Vec<Buf> {
        specs.iter().map(|t| Buf::zeros(t.dtype, rows * t.numel())).collect()
    }

    /// Empty buffers of every tensor in `specs` with room for `frames` frames.
    pub fn with_capacity(specs: &[TensorSpec], frames: usize) -> Vec<Buf> {
        specs
            .iter()
            .map(|t| {
                let n = frames * t.numel();
                match t.dtype {
                    DType::F32 => Buf::F32(Vec::with_capacity(n)),
                    DType::Bool => Buf::Bool(Vec::with_capacity(n)),
                    DType::I32 => Buf::I32(Vec::with_capacity(n)),
                    DType::I64 => Buf::I64(Vec::with_capacity(n)),
                }
            })
            .collect()
    }

    pub fn dtype(&self) -> DType {
        match self {
            Buf::F32(_) => DType::F32,
            Buf::Bool(_) => DType::Bool,
            Buf::I32(_) => DType::I32,
            Buf::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn view(&self) -> BufRef<'_> {
        match self {
            Buf::F32(v) => BufRef::F32(v),
            Buf::Bool(v) => BufRef::Bool(v),
            Buf::I32(v) => BufRef::I32(v),
            Buf::I64(v) => BufRef::I64(v),
        }
    }

    pub fn view_mut(&mut self) -> BufMut<'_> {
        match self {
            Buf::F32(v) => BufMut::F32(v),
            Buf::Bool(v) => BufMut::Bool(v),
            Buf::I32(v) => BufMut::I32(v),
            Buf::I64(v) => BufMut::I64(v),
        }
    }

    pub fn range(&self, r: Range<usize>) -> BufRef<'_> {
        match self {
            Buf::F32(v) => BufRef::F32(&v[r]),
            Buf::Bool(v) => BufRef::Bool(&v[r]),
            Buf::I32(v) => BufRef::I32(&v[r]),
            Buf::I64(v) => BufRef::I64(&v[r]),
        }
    }

    pub fn range_mut(&mut self, r: Range<usize>) -> BufMut<'_> {
        match self {
            Buf::F32(v) => BufMut::F32(&mut v[r]),
            Buf::Bool(v) => BufMut::Bool(&mut v[r]),
            Buf::I32(v) => BufMut::I32(&mut v[r]),
            Buf::I64(v) => BufMut::I64(&mut v[r]),
        }
    }

    /// Disjoint mutable chunks of `size` elements (the last may be shorter).
    pub fn chunks_mut(&mut self, size: usize) -> Vec<BufMut<'_>> {
        match self {
            Buf::F32(v) => v.chunks_mut(size).map(BufMut::F32).collect(),
            Buf::Bool(v) => v.chunks_mut(size).map(BufMut::Bool).collect(),
            Buf::I32(v) => v.chunks_mut(size).map(BufMut::I32).collect(),
            Buf::I64(v) => v.chunks_mut(size).map(BufMut::I64).collect(),
        }
    }

    /// Append `src` (same dtype).
    pub fn extend(&mut self, src: BufRef<'_>) {
        match (self, src) {
            (Buf::F32(a), BufRef::F32(b)) => a.extend_from_slice(b),
            (Buf::Bool(a), BufRef::Bool(b)) => a.extend_from_slice(b),
            (Buf::I32(a), BufRef::I32(b)) => a.extend_from_slice(b),
            (Buf::I64(a), BufRef::I64(b)) => a.extend_from_slice(b),
            (a, b) => panic!("Buf::extend: {} buffer, {} data", a.dtype().name(), b.dtype().name()),
        }
    }

    pub fn fill_zero(&mut self) {
        self.view_mut().fill_zero()
    }

    /// The native-endian bytes of the data (gmsim hands them to numpy).
    pub fn as_bytes(&self) -> &[u8] {
        self.view().as_bytes()
    }

    /// Bit-exact comparison; the error names the first differing element.
    pub fn check_equal(&self, other: &Buf) -> Result<(), String> {
        self.view().check_equal(other.view())
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        if let Buf::F32(v) = self { Some(v) } else { None }
    }

    pub fn as_bool(&self) -> Option<&[u8]> {
        if let Buf::Bool(v) = self { Some(v) } else { None }
    }

    pub fn as_i32(&self) -> Option<&[i32]> {
        if let Buf::I32(v) = self { Some(v) } else { None }
    }

    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Buf::I64(v) = self { Some(v) } else { None }
    }
}

impl<'a> BufRef<'a> {
    pub fn dtype(&self) -> DType {
        match self {
            BufRef::F32(_) => DType::F32,
            BufRef::Bool(_) => DType::Bool,
            BufRef::I32(_) => DType::I32,
            BufRef::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        match *self {
            BufRef::F32(v) => raw_bytes(v),
            BufRef::Bool(v) => v,
            BufRef::I32(v) => raw_bytes(v),
            BufRef::I64(v) => raw_bytes(v),
        }
    }

    pub fn to_buf(&self) -> Buf {
        match *self {
            BufRef::F32(v) => Buf::F32(v.to_vec()),
            BufRef::Bool(v) => Buf::Bool(v.to_vec()),
            BufRef::I32(v) => Buf::I32(v.to_vec()),
            BufRef::I64(v) => Buf::I64(v.to_vec()),
        }
    }

    /// Bit-exact comparison; the error names the first differing element.
    pub fn check_equal(&self, other: BufRef<'_>) -> Result<(), String> {
        if self.dtype() != other.dtype() || self.len() != other.len() {
            return Err(format!("{}[{}] vs {}[{}]", self.dtype().name(), self.len(), other.dtype().name(), other.len()));
        }
        let at = match (*self, other) {
            (BufRef::F32(a), BufRef::F32(b)) => first_diff(a, b, f32::to_bits).map(|i| (i, format!("{} vs {}", a[i], b[i]))),
            (BufRef::Bool(a), BufRef::Bool(b)) => first_diff(a, b, |x| x).map(|i| (i, format!("{} vs {}", a[i], b[i]))),
            (BufRef::I32(a), BufRef::I32(b)) => first_diff(a, b, |x| x).map(|i| (i, format!("{} vs {}", a[i], b[i]))),
            (BufRef::I64(a), BufRef::I64(b)) => first_diff(a, b, |x| x).map(|i| (i, format!("{} vs {}", a[i], b[i]))),
            _ => unreachable!(),
        };
        match at {
            Some((i, what)) => Err(format!("[{i}]: {what}")),
            None => Ok(()),
        }
    }
}

impl BufMut<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            BufMut::F32(_) => DType::F32,
            BufMut::Bool(_) => DType::Bool,
            BufMut::I32(_) => DType::I32,
            BufMut::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn fill_zero(&mut self) {
        match self {
            BufMut::F32(v) => v.fill(0.0),
            BufMut::Bool(v) => v.fill(0),
            BufMut::I32(v) => v.fill(0),
            BufMut::I64(v) => v.fill(0),
        }
    }

    /// A reborrowed view of elements `r`.
    pub fn range_mut(&mut self, r: Range<usize>) -> BufMut<'_> {
        match self {
            BufMut::F32(v) => BufMut::F32(&mut v[r]),
            BufMut::Bool(v) => BufMut::Bool(&mut v[r]),
            BufMut::I32(v) => BufMut::I32(&mut v[r]),
            BufMut::I64(v) => BufMut::I64(&mut v[r]),
        }
    }

    pub fn as_ref(&self) -> BufRef<'_> {
        match self {
            BufMut::F32(v) => BufRef::F32(v),
            BufMut::Bool(v) => BufRef::Bool(v),
            BufMut::I32(v) => BufRef::I32(v),
            BufMut::I64(v) => BufRef::I64(v),
        }
    }

    /// Copy `src` (same dtype and length) into this view.
    pub fn copy_from(&mut self, src: BufRef<'_>) {
        match (self, src) {
            (BufMut::F32(a), BufRef::F32(b)) => a.copy_from_slice(b),
            (BufMut::Bool(a), BufRef::Bool(b)) => a.copy_from_slice(b),
            (BufMut::I32(a), BufRef::I32(b)) => a.copy_from_slice(b),
            (BufMut::I64(a), BufRef::I64(b)) => a.copy_from_slice(b),
            (a, b) => panic!("BufMut::copy_from: {} view, {} data", a.dtype().name(), b.dtype().name()),
        }
    }
}

fn mismatch(spec: &TensorSpec, want: DType) -> ! {
    panic!("tensor {} is {}, not {}", spec.name, spec.dtype.name(), want.name())
}

/// One frame's tensors of one group (obs, legal or target), in schema order — what
/// the encoders write into. Accessors take the tensor's index in its group (games
/// define constants for them, or look them up once with [`TensorsMut::index`]) and
/// panic on a dtype mismatch (a bug in the game's schema / encoder).
pub struct TensorsMut<'a> {
    specs: &'a [TensorSpec],
    bufs: Vec<BufMut<'a>>,
}

impl<'a> TensorsMut<'a> {
    /// Views of one frame: `bufs[i]` must hold `specs[i].numel()` elements of its dtype.
    pub fn new(specs: &'a [TensorSpec], bufs: Vec<BufMut<'a>>) -> Self {
        assert_eq!(specs.len(), bufs.len(), "TensorsMut: {} specs, {} buffers", specs.len(), bufs.len());
        for (t, b) in specs.iter().zip(&bufs) {
            assert!(t.dtype == b.dtype() && t.numel() == b.len(), "TensorsMut: tensor {} is {}{:?}, buffer {}[{}]",
                    t.name, t.dtype.name(), t.shape, b.dtype().name(), b.len());
        }
        TensorsMut { specs, bufs }
    }

    /// Views of whole owned frame buffers ([`Buf::frame`]).
    pub fn of(specs: &'a [TensorSpec], bufs: &'a mut [Buf]) -> Self {
        Self::new(specs, bufs.iter_mut().map(Buf::view_mut).collect())
    }

    pub fn len(&self) -> usize {
        self.bufs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bufs.is_empty()
    }

    pub fn specs(&self) -> &'a [TensorSpec] {
        self.specs
    }

    pub fn spec(&self, i: usize) -> &TensorSpec {
        &self.specs[i]
    }

    /// Index of tensor `name` (panics when the schema has none).
    pub fn index(&self, name: &str) -> usize {
        index_of(self.specs, name).unwrap_or_else(|| {
            panic!("no tensor {name:?} in {:?}", self.specs.iter().map(|t| &t.name).collect::<Vec<_>>())
        })
    }

    pub fn f32(&mut self, i: usize) -> &mut [f32] {
        let spec = &self.specs[i];
        match &mut self.bufs[i] {
            BufMut::F32(v) => v,
            _ => mismatch(spec, DType::F32),
        }
    }

    pub fn bool(&mut self, i: usize) -> &mut [u8] {
        let spec = &self.specs[i];
        match &mut self.bufs[i] {
            BufMut::Bool(v) => v,
            _ => mismatch(spec, DType::Bool),
        }
    }

    pub fn i32(&mut self, i: usize) -> &mut [i32] {
        let spec = &self.specs[i];
        match &mut self.bufs[i] {
            BufMut::I32(v) => v,
            _ => mismatch(spec, DType::I32),
        }
    }

    pub fn i64(&mut self, i: usize) -> &mut [i64] {
        let spec = &self.specs[i];
        match &mut self.bufs[i] {
            BufMut::I64(v) => v,
            _ => mismatch(spec, DType::I64),
        }
    }

    /// Every view at once (disjoint borrows of several tensors).
    pub fn bufs_mut(&mut self) -> &mut [BufMut<'a>] {
        &mut self.bufs
    }

    pub fn get(&self, i: usize) -> BufRef<'_> {
        self.bufs[i].as_ref()
    }

    /// Zero every tensor.
    pub fn clear(&mut self) {
        for b in &mut self.bufs {
            b.fill_zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_views_and_comparison() {
        let specs = vec![TensorSpec::bits("planes", &[2, 3]), TensorSpec::bool("mask", &[4]), TensorSpec::i64("idx", &[])];
        assert_eq!(specs[0].numel(), 6);
        assert_eq!(specs[2].numel(), 1);
        let mut bufs = Buf::frame(&specs);
        {
            let mut t = TensorsMut::of(&specs, &mut bufs);
            let i = t.index("mask");
            t.bool(i)[2] = 1;
            t.f32(0)[5] = 1.0;
            t.i64(2)[0] = 7;
        }
        assert_eq!(bufs[1], Buf::Bool(vec![0, 0, 1, 0]));
        assert_eq!(bufs[2].as_i64(), Some(&[7i64][..]));
        assert_eq!(bufs[0].as_bytes().len(), 24);
        let mut other = bufs.clone();
        assert!(bufs[0].check_equal(&other[0]).is_ok());
        other[0] = Buf::F32(vec![0.0, 0.0, 0.0, 0.0, -0.0, 1.0]);
        assert!(bufs[0].check_equal(&other[0]).unwrap_err().contains("[4]"));
        let mut rows = Buf::rows(&specs, 3);
        assert_eq!(rows[1].len(), 12);
        let chunks = rows[1].chunks_mut(4);
        assert_eq!(chunks.len(), 3);
        let mut cat = Buf::with_capacity(&specs, 2).remove(1);
        cat.extend(bufs[1].view());
        cat.extend(bufs[1].view());
        assert_eq!(cat.len(), 8);
    }

    #[test]
    #[should_panic(expected = "not f32")]
    fn dtype_mismatch_panics() {
        let specs = vec![TensorSpec::bool("mask", &[4])];
        let mut bufs = Buf::frame(&specs);
        TensorsMut::of(&specs, &mut bufs).f32(0);
    }
}
