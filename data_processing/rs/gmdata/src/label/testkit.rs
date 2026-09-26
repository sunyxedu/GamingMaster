//! Test helpers for labeller unit tests (compiled for `cargo test` only).
//!
//! Typical game test (no engine binary needed):
//!
//! ```ignore
//! let root = tmpdir("chess-label");
//! crate::chess::convert(&fixture_pgn, &root.join("base"), 256, 0, None, 1, false, Default::default())?;
//! let cmd = EngineCommand::in_process("fake-sf", || UciFake::new(|pos: &str| Some(score_legal_moves(pos))));
//! let summary = driver::run(ChessLabeller::default(), &quiet_opts(&root, cmd))?;
//! let out = ShardReader::open(&root.join("out").join("chess_00000"))?;
//! let idx = overlay_tensor(&out, "tgt.teacher_idx", DType::I64, &[K]);
//! ```
//!
//! Real-engine integration tests should look the binary up with [`engine_path`] and
//! return early (skip) when it is absent.

use super::driver::RunOpts;
use super::engine::EngineCommand;
use gmshard::{DType, ShardReader, Tensor};
use std::path::{Path, PathBuf};

/// A fresh, empty temporary directory.
pub fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "gmdata-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Driver options for `root/base` → `root/out`, 2 workers, no progress output.
pub fn quiet_opts(root: &Path, cmd: EngineCommand) -> RunOpts {
    let mut o = RunOpts::new(&root.join("base"), &root.join("out"), cmd);
    o.quiet = true;
    o.workers = 2;
    o
}

/// Read an overlay column, asserting its dtype and frame shape.
pub fn overlay_tensor(r: &ShardReader, name: &str, dtype: DType, frame_shape: &[usize]) -> Tensor {
    let info = r.tensor_info(name).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(info.dtype, dtype, "{name} dtype");
    assert_eq!(info.frame_shape, frame_shape, "{name} frame shape");
    r.read(name).unwrap()
}

/// An installed engine for integration tests: `$<env_var>` if set, else
/// `${GM_ENGINES:-$HOME/gm-engines}/bin/<name>`, else `<name>` on `PATH`; `None` → skip.
pub fn engine_path(env_var: &str, name: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env_var).map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let root = std::env::var_os("GM_ENGINES")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("gm-engines")));
    if let Some(p) = root.map(|r| r.join("bin").join(name)) {
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(name)).find(|p| p.is_file()))
}
