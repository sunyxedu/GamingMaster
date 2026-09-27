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


/// SHA-256 of a file, hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Percent-encode a Hub file path for a `resolve` URL (keeps `/`).
fn hf_path_url(path: &str) -> String {
    let mut out = String::new();
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Download the files of a Hugging Face dataset repo whose path `want` accepts into `out`
/// (same relative paths), each checked against the SHA-256 of the Hub's LFS pointer (files
/// without one: against their size).  Resumable: a file already present with the right
/// checksum is skipped; a failed download is retried 10 times, backing off from 30 s to 15 min
/// (the Hub rate-limits bursts with 429); `HF_TOKEN` (optional) authenticates.  Writes `out/hf_files.json`
/// (repo, commit, every file's path / size / sha256).
pub fn fetch_hf_files(repo: &str, want: &dyn Fn(&str) -> bool, out: &Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    use std::io::Write;
    let client = http()?;
    // an access token (HF_TOKEN) raises the Hub's rate limits; anonymous access works too
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let get = |url: &str| {
        let r = client.get(url);
        match &token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    };
    let get_json = |url: &str| -> Result<serde_json::Value> {
        let mut wait = 30;
        for attempt in 0..10 {
            match get(url).send().and_then(|r| r.error_for_status()).and_then(|r| r.json::<serde_json::Value>()) {
                Ok(v) => return Ok(v),
                Err(e) if attempt < 9 => {
                    eprintln!("[fetch-hf] {url}: {e}; retrying in {wait}s");
                    std::thread::sleep(Duration::from_secs(wait));
                    wait = (wait * 2).min(900);
                }
                Err(e) => return Err(e.into()),
            }
        }
        unreachable!()
    };
    let info = get_json(&format!("https://huggingface.co/api/datasets/{repo}"))?;
    let commit = info["sha"].as_str().unwrap_or("main").to_string();
    let tree: Vec<serde_json::Value> = serde_json::from_value(
        get_json(&format!("https://huggingface.co/api/datasets/{repo}/tree/{commit}?recursive=true"))?)?;
    let files: Vec<(String, u64, Option<String>)> = tree.iter()
        .filter(|x| x["type"] == "file")
        .filter_map(|x| {
            let path = x["path"].as_str()?.to_string();
            want(&path).then(|| (path, x["size"].as_u64().unwrap_or(0), x["lfs"]["oid"].as_str().map(str::to_string)))
        })
        .collect();
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("hf_files.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "repo": repo, "commit": commit,
        "files": files.iter().map(|(p, n, h)| serde_json::json!({"path": p, "bytes": n, "sha256": h})).collect::<Vec<_>>(),
    }))?)?;
    let total: u64 = files.iter().map(|f| f.1).sum();
    eprintln!("fetch-hf {repo}@{commit}: {} files, {:.2} GB", files.len(), total as f64 / 1e9);
    let (mut got, mut cached) = (0usize, 0usize);
    for (i, (path, size, sha)) in files.iter().enumerate() {
        let dest = out.join(path);
        let ok = |p: &Path| -> Result<bool> {
            Ok(p.exists() && match sha {
                Some(h) => sha256_file(p)? == *h,
                None => std::fs::metadata(p)?.len() == *size,
            })
        };
        if ok(&dest)? {
            cached += 1;
            continue;
        }
        if let Some(d) = dest.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = PathBuf::from(format!("{}.part", dest.display()));
        let url = format!("https://huggingface.co/datasets/{repo}/resolve/{commit}/{}", hf_path_url(path));
        let mut last_err = None;
        // the Hub answers 429 (too many requests) to bursts: back off for minutes, not seconds
        let mut wait = 30u64;
        for attempt in 0..10 {
            let r = (|| -> Result<String> {
                let mut resp = get(&url).send()?.error_for_status()?;
                let mut file = std::fs::File::create(&tmp)?;
                let mut h = Sha256::new();
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = resp.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    h.update(&buf[..n]);
                    file.write_all(&buf[..n])?;
                }
                file.flush()?;
                Ok(hex::encode(h.finalize()))
            })();
            match r {
                Ok(digest) if sha.as_ref().map(|h| *h == digest).unwrap_or(true) => {
                    last_err = None;
                    break;
                }
                Ok(digest) => last_err = Some(anyhow::anyhow!("{path}: sha256 {digest} != {}", sha.as_deref().unwrap_or(""))),
                Err(e) => last_err = Some(e),
            }
            eprintln!("[fetch-hf] {path}: attempt {} failed: {}; retrying in {wait}s", attempt + 1, last_err.as_ref().unwrap());
            std::thread::sleep(Duration::from_secs(wait));
            wait = (wait * 2).min(900);
        }
        if let Some(e) = last_err {
            std::fs::remove_file(&tmp).ok();
            return Err(e);
        }
        std::fs::rename(&tmp, &dest)?;
        got += 1;
        eprintln!("[{}/{}] {path} {:.1} MB", i + 1, files.len(), *size as f64 / 1e6);
        std::thread::sleep(Duration::from_secs(2)); // polite: no bursts
    }
    Ok(serde_json::json!({"repo": repo, "commit": commit, "files": files.len(), "downloaded": got, "cached": cached,
                          "bytes": total}))
}
