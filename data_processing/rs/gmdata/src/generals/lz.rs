//! LZ-string decompression (the `lz-string` JavaScript library's format),
//! enough to read generals.io `.gior` replays: `compressToUint8Array` output
//! is the UTF-16 code units of the compressed string as big-endian byte pairs.

/// Decompress the raw UTF-16 code-unit stream (`decompress` in lz-string).
pub fn decompress_units(units: &[u16]) -> Option<String> {
    if units.is_empty() {
        return Some(String::new());
    }
    let get = |i: usize| -> u32 { units.get(i).copied().unwrap_or(0) as u32 };
    let reset: u32 = 32768;
    let mut dict: Vec<Vec<u16>> = Vec::with_capacity(4096);
    let mut enlarge_in: u64 = 4;
    let mut dict_size: usize = 4;
    let mut num_bits: u32 = 3;
    let mut result: Vec<u16> = Vec::new();

    let mut val = get(0);
    let mut position = reset;
    let mut index = 1usize;

    let read_bits = |n: u32, val: &mut u32, position: &mut u32, index: &mut usize| -> u32 {
        let mut bits = 0u32;
        let maxpower = 1u32 << n;
        let mut power = 1u32;
        while power != maxpower {
            let resb = *val & *position;
            *position >>= 1;
            if *position == 0 {
                *position = reset;
                *val = get(*index);
                *index += 1;
            }
            if resb > 0 {
                bits |= power;
            }
            power <<= 1;
        }
        bits
    };

    for i in 0..3u16 {
        dict.push(vec![i]);
    }
    let first = read_bits(2, &mut val, &mut position, &mut index);
    let c: u16 = match first {
        0 => read_bits(8, &mut val, &mut position, &mut index) as u16,
        1 => read_bits(16, &mut val, &mut position, &mut index) as u16,
        _ => return Some(String::new()),
    };
    dict.push(vec![c]);
    let mut w: Vec<u16> = vec![c];
    result.push(c);
    loop {
        if index > units.len() {
            return Some(String::new());
        }
        let mut code = read_bits(num_bits, &mut val, &mut position, &mut index) as usize;
        match code {
            0 => {
                let ch = read_bits(8, &mut val, &mut position, &mut index) as u16;
                dict.push(vec![ch]);
                code = dict_size;
                dict_size += 1;
                enlarge_in -= 1;
            }
            1 => {
                let ch = read_bits(16, &mut val, &mut position, &mut index) as u16;
                dict.push(vec![ch]);
                code = dict_size;
                dict_size += 1;
                enlarge_in -= 1;
            }
            2 => return Some(String::from_utf16_lossy(&result)),
            _ => {}
        }
        if enlarge_in == 0 {
            enlarge_in = 1u64 << num_bits;
            num_bits += 1;
        }
        let entry: Vec<u16> = if code < dict.len() {
            dict[code].clone()
        } else if code == dict_size {
            let mut e = w.clone();
            e.push(w[0]);
            e
        } else {
            return None;
        };
        result.extend_from_slice(&entry);
        let mut new_entry = w.clone();
        new_entry.push(entry[0]);
        dict.push(new_entry);
        dict_size += 1;
        enlarge_in -= 1;
        w = entry;
        if enlarge_in == 0 {
            enlarge_in = 1u64 << num_bits;
            num_bits += 1;
        }
    }
}

/// `decompressFromUint8Array`: big-endian byte pairs → code units → decompress.
pub fn decompress_uint8(bytes: &[u8]) -> Option<String> {
    let mut units = Vec::with_capacity(bytes.len() / 2 + 1);
    let mut i = 0;
    while i + 1 < bytes.len() {
        units.push(((bytes[i] as u16) << 8) | bytes[i + 1] as u16);
        i += 2;
    }
    if i < bytes.len() {
        units.push((bytes[i] as u16) << 8);
    }
    decompress_units(&units)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny LZ-string *compressor* (straight port) so the round trip is testable offline.
    fn compress_units(input: &str) -> Vec<u16> {
        use std::collections::HashMap;
        let chars: Vec<u16> = input.encode_utf16().collect();
        let mut dict: HashMap<Vec<u16>, usize> = HashMap::new();
        let mut dict_to_create: HashMap<Vec<u16>, bool> = HashMap::new();
        let mut w: Vec<u16> = Vec::new();
        let mut enlarge_in: u64 = 2;
        let mut dict_size = 3usize;
        let mut num_bits = 2u32;
        let mut data: Vec<u16> = Vec::new();
        let mut data_val: u32 = 0;
        let mut data_position: u32 = 0;
        let bits_per_char = 16;
        fn push_bit(bit: u32, data: &mut Vec<u16>, val: &mut u32, pos: &mut u32, bpc: u32) {
            *val = (*val << 1) | bit;
            if *pos == bpc - 1 {
                *pos = 0;
                data.push(*val as u16);
                *val = 0;
            } else {
                *pos += 1;
            }
        }
        let mut emit = |value: u32, n: u32, data: &mut Vec<u16>, val: &mut u32, pos: &mut u32| {
            let mut v = value;
            for _ in 0..n {
                push_bit(v & 1, data, val, pos, bits_per_char);
                v >>= 1;
            }
        };
        for &c in &chars {
            let cv = vec![c];
            if !dict.contains_key(&cv) {
                dict.insert(cv.clone(), dict_size);
                dict_size += 1;
                dict_to_create.insert(cv.clone(), true);
            }
            let mut wc = w.clone();
            wc.push(c);
            if dict.contains_key(&wc) {
                w = wc;
            } else {
                if dict_to_create.contains_key(&w) {
                    if w[0] < 256 {
                        emit(0, num_bits, &mut data, &mut data_val, &mut data_position);
                        emit(w[0] as u32, 8, &mut data, &mut data_val, &mut data_position);
                    } else {
                        emit(1, num_bits, &mut data, &mut data_val, &mut data_position);
                        emit(w[0] as u32, 16, &mut data, &mut data_val, &mut data_position);
                    }
                    enlarge_in -= 1;
                    if enlarge_in == 0 {
                        enlarge_in = 1 << num_bits;
                        num_bits += 1;
                    }
                    dict_to_create.remove(&w);
                } else {
                    emit(dict[&w] as u32, num_bits, &mut data, &mut data_val, &mut data_position);
                }
                enlarge_in -= 1;
                if enlarge_in == 0 {
                    enlarge_in = 1 << num_bits;
                    num_bits += 1;
                }
                dict.insert(wc, dict_size);
                dict_size += 1;
                w = vec![c];
            }
        }
        if !w.is_empty() {
            if dict_to_create.contains_key(&w) {
                if w[0] < 256 {
                    emit(0, num_bits, &mut data, &mut data_val, &mut data_position);
                    emit(w[0] as u32, 8, &mut data, &mut data_val, &mut data_position);
                } else {
                    emit(1, num_bits, &mut data, &mut data_val, &mut data_position);
                    emit(w[0] as u32, 16, &mut data, &mut data_val, &mut data_position);
                }
                enlarge_in -= 1;
                if enlarge_in == 0 {
                    enlarge_in = 1 << num_bits;
                    num_bits += 1;
                }
                dict_to_create.remove(&w);
            } else {
                emit(dict[&w] as u32, num_bits, &mut data, &mut data_val, &mut data_position);
            }
            enlarge_in -= 1;
            if enlarge_in == 0 {
                num_bits += 1;
            }
        }
        emit(2, num_bits, &mut data, &mut data_val, &mut data_position);
        loop {
            data_val <<= 1;
            if data_position == bits_per_char - 1 {
                data.push(data_val as u16);
                break;
            }
            data_position += 1;
        }
        data
    }

    #[test]
    fn round_trip_json_like_text() {
        let text = r#"[13,"abc",20,20,["Human.exe","LEARN"],[38,0],[5,17],[40,42],[0,399],[1,2,3]]"#;
        let units = compress_units(text);
        assert_eq!(decompress_units(&units).as_deref(), Some(text));
        let bytes: Vec<u8> = units.iter().flat_map(|u| u.to_be_bytes()).collect();
        assert_eq!(decompress_uint8(&bytes).as_deref(), Some(text));
    }

}
