//! generals.io metadata API + S3 replay download.

use super::gior::{parse_gior, Replay};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

pub const API: &str = "https://generals.io/api/replaysForUsername";

pub fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("gamingmaster-gmdata/0.1")
        .timeout(Duration::from_secs(30))
        .danger_accept_invalid_certs(true) // the reference client does not verify either
        .build()?)
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// One page of replay metadata (newest first). Empty vec = exhausted.
pub fn fetch_page(client: &reqwest::blocking::Client, user: &str, offset: usize, count: usize) -> Result<Vec<Value>> {
    let url = format!("{API}?u={}&offset={offset}&count={count}", urlencode(user));
    let text = client.get(&url).send()?.error_for_status()?.text()?;
    let v: Value = serde_json::from_str(&text).context("replay list JSON")?;
    Ok(v.as_array().cloned().unwrap_or_default())
}

/// All metadata for a user, up to `total` entries.
pub fn fetch_list(client: &reqwest::blocking::Client, user: &str, total: usize, per_request: usize) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut offset = 0;
    while out.len() < total {
        let count = per_request.min(total - out.len());
        let page = fetch_page(client, user, offset, count)?;
        if page.is_empty() {
            break;
        }
        let n = page.len();
        out.extend(page);
        offset += n;
        if n < count {
            break;
        }
    }
    Ok(out)
}

pub fn replay_url(server: &str, id: &str) -> String {
    format!("https://generalsio-replays-{server}.s3.amazonaws.com/{id}.gior")
}

/// Download + decode one replay with retries.
pub fn fetch_replay(client: &reqwest::blocking::Client, server: &str, id: &str, retries: u32) -> Result<Replay> {
    let url = replay_url(server, id);
    let mut last = None;
    for attempt in 0..=retries {
        match client.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.bytes()) {
            Ok(b) => return parse_gior(&b).with_context(|| format!("decode {id}")),
            Err(e) => {
                last = Some(e);
                if attempt < retries {
                    std::thread::sleep(Duration::from_millis(2000 * (attempt as u64 + 1)));
                }
            }
        }
    }
    bail!("failed to fetch {id}: {}", last.map(|e| e.to_string()).unwrap_or_default())
}

/// Download many replays to `<out>/<id>.json` (skipping existing) with `concurrency` threads.
/// Returns the ids that failed.
pub fn download_replays(ids: &[String], out: &Path, server: &str, concurrency: usize) -> Result<Vec<String>> {
    std::fs::create_dir_all(out)?;
    let client = Arc::new(client()?);
    let ids: Vec<String> = ids.to_vec();
    let failed = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..concurrency.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= ids.len() {
                    break;
                }
                let id = &ids[i];
                let path = out.join(format!("{id}.json"));
                if path.exists() {
                    continue;
                }
                match fetch_replay(&client, server, id, 3) {
                    Ok(r) => {
                        let tmp = out.join(format!(".{id}.json.tmp"));
                        let ok = std::fs::File::create(&tmp)
                            .map(|f| std::io::BufWriter::new(f))
                            .map_err(anyhow::Error::from)
                            .and_then(|mut w| {
                                serde_json::to_writer(&mut w, &r)?;
                                std::io::Write::flush(&mut w)?;
                                Ok(())
                            })
                            .is_ok();
                        if ok {
                            let _ = std::fs::rename(&tmp, &path);
                        }
                    }
                    Err(e) => {
                        eprintln!("  download failed: {id}: {e}");
                        failed.lock().unwrap().push(id.clone());
                    }
                }
            });
        }
    });
    Ok(failed.into_inner().unwrap())
}
