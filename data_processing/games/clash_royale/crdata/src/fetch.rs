//! Download the replay Parquet parts from Hugging Face with SHA-256
//! verification against the dataset's own `manifest.json`. Resumable: a part
//! whose checksum already matches is skipped.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

pub const DEFAULT_REPO: &str = "VanguardX101/IL_Replay";

#[derive(Deserialize)]
struct Manifest {
    files: Vec<ManifestFile>,
}

#[derive(Deserialize, Clone)]
pub struct ManifestFile {
    pub path: String,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: String,
}

fn url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/datasets/{repo}/resolve/main/{path}")
}

fn sha256_file(path: &Path) -> Result<String> {
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

pub fn manifest(client: &reqwest::blocking::Client, repo: &str) -> Result<Vec<ManifestFile>> {
    let text = client.get(url(repo, "manifest.json")).send()?.error_for_status()?.text()?;
    let m: Manifest = serde_json::from_str(&text).context("manifest.json")?;
    Ok(m.files)
}

/// Download `replays/part-*.parquet` (and optionally the `actions/` table) into `out`.
pub fn fetch(repo: &str, out: &Path, include_actions: bool, limit: Option<usize>) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("gamingmaster-crdata/0.1")
        .timeout(std::time::Duration::from_secs(600))
        .build()?;
    let files = manifest(&client, repo)?;
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "repo": repo, "files": files.iter().map(|f| serde_json::json!({"path": f.path, "rows": f.rows, "bytes": f.bytes, "sha256": f.sha256})).collect::<Vec<_>>()
    }))?)?;
    let wanted: Vec<&ManifestFile> = files
        .iter()
        .filter(|f| f.path.starts_with("replays/") || (include_actions && f.path.starts_with("actions/")))
        .take(limit.unwrap_or(usize::MAX))
        .collect();
    let total_bytes: u64 = wanted.iter().map(|f| f.bytes).sum();
    eprintln!("fetch: {} files, {:.2} GB", wanted.len(), total_bytes as f64 / 1e9);
    for (i, f) in wanted.iter().enumerate() {
        let dest = out.join(&f.path);
        if dest.exists() && sha256_file(&dest)? == f.sha256 {
            eprintln!("[{}/{}] {} ok (cached)", i + 1, wanted.len(), f.path);
            continue;
        }
        std::fs::create_dir_all(dest.parent().unwrap())?;
        let tmp = dest.with_extension("parquet.part");
        let mut resp = client.get(url(repo, &f.path)).send()?.error_for_status()?;
        let mut file = std::fs::File::create(&tmp)?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
            file.write_all(&buf[..n])?;
            got += n as u64;
        }
        file.flush()?;
        let digest = hex::encode(h.finalize());
        if digest != f.sha256 {
            std::fs::remove_file(&tmp).ok();
            bail!("{}: sha256 mismatch (got {digest}, want {})", f.path, f.sha256);
        }
        std::fs::rename(&tmp, &dest)?;
        eprintln!("[{}/{}] {} {:.1} MB ok", i + 1, wanted.len(), f.path, got as f64 / 1e6);
    }
    Ok(())
}
