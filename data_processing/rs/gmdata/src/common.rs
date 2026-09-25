//! Shared helpers: input discovery, archive reading, HTTP, shard writing.

use anyhow::{Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Files under `path` (recursive, sorted) whose name ends with one of `exts`
/// (case-insensitive); `path` itself when it is a file.
pub fn find_files(path: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    fn walk(d: &Path, exts: &[&str], out: &mut Vec<PathBuf>) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(d).with_context(|| format!("read {}", d.display()))?.collect::<std::result::Result<_, _>>()?;
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            if p.is_dir() {
                walk(&p, exts, out)?;
            } else {
                let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
                if exts.iter().any(|x| name.ends_with(&x.to_lowercase())) {
                    out.push(p);
                }
            }
        }
        Ok(())
    }
    walk(path, exts, &mut out)?;
    out.sort();
    Ok(out)
}

/// Read a file, transparently gunzipping `.gz` content (by magic bytes).
pub fn read_maybe_gz(bytes: Vec<u8>) -> Result<Vec<u8>> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
        return Ok(out);
    }
    Ok(bytes)
}

pub fn http() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("GamingMaster-research/0.1 (imitation-learning data pipeline)")
        .timeout(Duration::from_secs(600))
        .build()?)
}

/// Download `url` to `dest` (atomic rename; skipped when `dest` exists). Returns false on 404.
pub fn download(client: &reqwest::blocking::Client, url: &str, dest: &Path) -> Result<bool> {
    if dest.exists() {
        eprintln!("have {}", dest.display());
        return Ok(true);
    }
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p)?;
    }
    let resp = client.get(url).send()?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }
    let mut resp = resp.error_for_status()?;
    let tmp = dest.with_extension("part");
    let mut f = std::fs::File::create(&tmp)?;
    let n = std::io::copy(&mut resp, &mut f)?;
    drop(f);
    std::fs::rename(&tmp, dest)?;
    eprintln!("fetched {url} ({:.1} MB)", n as f64 / 1e6);
    Ok(true)
}
