//! Minimal NumPy `.npy` (format 1.0) writer — no dependency on either side:
//! `numpy.load` / `torch.from_numpy` read these directly.

use anyhow::Result;
use std::io::Write;
use std::path::Path;

pub trait NpyElement: Copy {
    const DESCR: &'static str;
    fn write_le(&self, out: &mut Vec<u8>);
}

macro_rules! npy_elem {
    ($t:ty, $descr:expr) => {
        impl NpyElement for $t {
            const DESCR: &'static str = $descr;
            fn write_le(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
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
}
