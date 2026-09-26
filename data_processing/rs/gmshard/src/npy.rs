//! Minimal NumPy `.npy` writer (format 1.0) and reader (formats 1.0 / 2.0 / 3.0,
//! little-endian C-order arrays) — no dependency on either side: `numpy.load` /
//! `torch.from_numpy` read what [`write`] produces, and [`read`] reads what the
//! writer (or `numpy.save`) produces.

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::Path;

pub trait NpyElement: Copy {
    const DESCR: &'static str;
    fn write_le(&self, out: &mut Vec<u8>);
    /// Decode one element from exactly `size_of::<Self>()` little-endian bytes.
    fn read_le(bytes: &[u8]) -> Self;
}

macro_rules! npy_elem {
    ($t:ty, $descr:expr) => {
        impl NpyElement for $t {
            const DESCR: &'static str = $descr;
            fn write_le(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes.try_into().expect("element size"))
            }
        }
    };
}

npy_elem!(u8, "|u1");
npy_elem!(i64, "<i8");
npy_elem!(i8, "|i1");
npy_elem!(u16, "<u2");
npy_elem!(i16, "<i2");
npy_elem!(i32, "<i4");
npy_elem!(u32, "<u4");
npy_elem!(f32, "<f4");
npy_elem!(f64, "<f8");

pub fn encode<T: NpyElement>(shape: &[usize], data: &[T]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    assert_eq!(n, data.len(), "npy shape/data mismatch");
    let shape_txt = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", T::DESCR, shape_txt);
    // total header length (magic 6 + version 2 + len 2 + header) must be a multiple of 64
    let pad = 64 - ((10 + header.len() + 1) % 64);
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + n * std::mem::size_of::<T>());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for x in data {
        x.write_le(&mut out);
    }
    out
}

pub fn write<T: NpyElement>(path: &Path, shape: &[usize], data: &[T]) -> Result<()> {
    let bytes = encode(shape, data);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

/// A parsed `.npy` file: dtype descriptor (`<f4`, `|u1`, …), shape and the raw
/// little-endian element bytes (C order).
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl NpyArray {
    /// Number of elements (product of the shape; 1 for a 0-d array).
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as `T`. The descriptor must be `T::DESCR` (numpy's `|b1` bools
    /// also read as `u8`); anything else is an error, never a reinterpretation.
    pub fn to_vec<T: NpyElement>(&self) -> Result<Vec<T>> {
        let ok = self.descr == T::DESCR || (T::DESCR == "|u1" && self.descr == "|b1");
        if !ok {
            bail!("npy dtype {} cannot be read as {}", self.descr, T::DESCR);
        }
        let size = std::mem::size_of::<T>();
        if self.data.len() != self.len() * size {
            bail!("npy data is {} bytes, shape {:?} of {} needs {}", self.data.len(), self.shape, self.descr,
                  self.len() * size);
        }
        Ok(self.data.chunks_exact(size).map(T::read_le).collect())
    }
}

/// Parse `.npy` bytes (little-endian or byte-order-free descriptors, C order only).
pub fn decode(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file (bad magic)");
    }
    let (hlen, hstart) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated .npy header");
            }
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => bail!("unsupported .npy format version {v}"),
    };
    let header = bytes.get(hstart..hstart + hlen).context("truncated .npy header")?;
    let header = std::str::from_utf8(header).context(".npy header is not UTF-8")?;
    let descr = header_str(header, "descr")?;
    if descr.starts_with('>') {
        bail!("big-endian .npy ({descr}) is not supported");
    }
    if header_value(header, "fortran_order")?.starts_with("True") {
        bail!("Fortran-order .npy is not supported");
    }
    let shape_txt = header_value(header, "shape")?;
    let open = shape_txt.find('(').context("malformed .npy shape")?;
    let close = shape_txt.find(')').context("malformed .npy shape")?;
    let shape = shape_txt[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('L').parse::<usize>().with_context(|| format!("bad .npy dimension {s:?}")))
        .collect::<Result<Vec<_>>>()?;
    Ok(NpyArray { descr, shape, data: bytes[hstart + hlen..].to_vec() })
}

/// Read and parse an `.npy` file.
pub fn read(path: &Path) -> Result<NpyArray> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// The raw text after `'key':` in a header dict (up to the end of the header).
fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let pat = format!("'{key}':");
    let at = header.find(&pat).with_context(|| format!(".npy header has no {key:?}"))?;
    Ok(header[at + pat.len()..].trim_start())
}

/// A quoted string value of a header dict.
fn header_str(header: &str, key: &str) -> Result<String> {
    let v = header_value(header, key)?;
    let q = v.chars().next().context("empty .npy header value")?;
    if q != '\'' && q != '"' {
        bail!(".npy header {key:?} is not a string");
    }
    let end = v[1..].find(q).context("unterminated .npy header string")?;
    Ok(v[1..1 + end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_aligned_and_parseable() {
        let bytes = encode(&[2, 3], &[1i32, 2, 3, 4, 5, 6]);
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0);
        let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
        assert!(header.contains("'descr': '<i4'") && header.contains("'shape': (2, 3)"));
        assert_eq!(bytes.len(), 10 + hlen + 24);
        let one = encode(&[3], &[7u8, 8, 9]);
        let h = u16::from_le_bytes([one[8], one[9]]) as usize;
        assert!(std::str::from_utf8(&one[10..10 + h]).unwrap().contains("'shape': (3,)"));
    }

    #[test]
    fn roundtrip_every_writer_dtype() {
        let f = decode(&encode(&[2, 2], &[1.5f32, -2.0, 0.0, 7.25])).unwrap();
        assert_eq!((f.descr.as_str(), f.shape.clone()), ("<f4", vec![2, 2]));
        assert_eq!(f.to_vec::<f32>().unwrap(), vec![1.5, -2.0, 0.0, 7.25]);
        let i = decode(&encode(&[3], &[-1i64, 0, i64::MAX])).unwrap();
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![-1, 0, i64::MAX]);
        assert_eq!(decode(&encode(&[2], &[-3i32, 4])).unwrap().to_vec::<i32>().unwrap(), vec![-3, 4]);
        assert_eq!(decode(&encode(&[2], &[-3i8, 4])).unwrap().to_vec::<i8>().unwrap(), vec![-3, 4]);
        assert_eq!(decode(&encode(&[2], &[3u8, 255])).unwrap().to_vec::<u8>().unwrap(), vec![3, 255]);
        assert_eq!(decode(&encode(&[2], &[3u16, 65535])).unwrap().to_vec::<u16>().unwrap(), vec![3, 65535]);
        let empty = decode(&encode::<f32>(&[0, 5], &[])).unwrap();
        assert_eq!((empty.shape.clone(), empty.len()), (vec![0, 5], 0));
        // a dtype mismatch is an error, not a reinterpretation
        assert!(decode(&encode(&[1], &[1.0f32])).unwrap().to_vec::<i32>().is_err());
    }

    #[test]
    fn reads_numpy_style_headers() {
        // numpy writes bools as |b1 and uses format 2.0 for long headers
        let mut b = b"\x93NUMPY\x02\x00".to_vec();
        let h = "{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }\n";
        b.extend_from_slice(&(h.len() as u32).to_le_bytes());
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[1, 0, 1]);
        assert_eq!(decode(&b).unwrap().to_vec::<u8>().unwrap(), vec![1, 0, 1]);
        let mut be = encode(&[1], &[1.0f32]);
        let at = be.windows(3).position(|w| w == b"<f4").unwrap();
        be[at] = b'>';
        assert!(decode(&be).is_err());
        let mut fortran = encode(&[1], &[1.0f32]);
        let at = fortran.windows(5).position(|w| w == b"False").unwrap();
        fortran[at..at + 5].copy_from_slice(b"True ");
        assert!(decode(&fortran).is_err());
        assert!(decode(b"not numpy at all").is_err());
    }
}
