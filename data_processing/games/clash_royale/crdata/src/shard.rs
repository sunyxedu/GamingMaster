//! Shard writer: many samples' frame columns concatenated into one directory of
//! `.npy` arrays plus `index.json` (per-sample offsets and metadata).
//!
//!     shard-000012/
//!       index.json       {"format": "cr-frames.v1", "step_ticks", "recent", "samples": [...]}
//!       tick.npy         int32   [N]
//!       hand_ids.npy     uint16  [N, 4]
//!       recent_card.npy  uint16  [N, K]
//!       ...

use crate::frames::{FrameColumns, Sample, SampleMeta, HAND};
use crate::npy;
use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct Index<'a> {
    format: &'static str,
    catalog_version: &'a str,
    step_ticks: u32,
    recent: usize,
    hand: usize,
    grid: [u8; 2],
    n_frames: usize,
    samples: &'a [SampleMeta],
}

pub struct ShardWriter {
    out_root: PathBuf,
    samples_per_shard: usize,
    step_ticks: u32,
    recent: usize,
    catalog_version: String,
    next_shard: usize,
    cols: FrameColumns,
    metas: Vec<SampleMeta>,
    pub written_shards: usize,
    pub written_samples: usize,
    pub written_frames: usize,
}

impl ShardWriter {
    pub fn new(out_root: &Path, samples_per_shard: usize, step_ticks: u32, recent: usize, catalog_version: &str, first_shard: usize) -> Result<Self> {
        std::fs::create_dir_all(out_root)?;
        Ok(Self {
            out_root: out_root.to_path_buf(),
            samples_per_shard,
            step_ticks,
            recent,
            catalog_version: catalog_version.to_string(),
            next_shard: first_shard,
            cols: FrameColumns::default(),
            metas: Vec::new(),
            written_shards: 0,
            written_samples: 0,
            written_frames: 0,
        })
    }

    pub fn push(&mut self, mut s: Sample) -> Result<()> {
        s.meta.start = self.cols.len();
        s.meta.length = s.cols.len();
        self.cols.append(&s.cols);
        self.metas.push(s.meta);
        if self.metas.len() >= self.samples_per_shard {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.metas.is_empty() {
            return Ok(());
        }
        let dir = self.out_root.join(format!("shard-{:06}", self.next_shard));
        let tmp = self.out_root.join(format!(".shard-{:06}.tmp", self.next_shard));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp)?;
        let n = self.cols.len();
        let k = self.recent;
        let c = &self.cols;
        npy::write(&tmp.join("tick.npy"), &[n], &c.tick)?;
        npy::write(&tmp.join("time_s.npy"), &[n], &c.time_s)?;
        npy::write(&tmp.join("phase.npy"), &[n], &c.phase)?;
        npy::write(&tmp.join("overtime.npy"), &[n], &c.overtime)?;
        npy::write(&tmp.join("elixir.npy"), &[n], &c.elixir)?;
        npy::write(&tmp.join("opp_elixir.npy"), &[n], &c.opp_elixir)?;
        npy::write(&tmp.join("hand_ids.npy"), &[n, HAND], &c.hand_ids)?;
        npy::write(&tmp.join("hand_form.npy"), &[n, HAND], &c.hand_form)?;
        npy::write(&tmp.join("hand_kind.npy"), &[n, HAND], &c.hand_kind)?;
        npy::write(&tmp.join("hand_cost.npy"), &[n, HAND], &c.hand_cost)?;
        npy::write(&tmp.join("hand_prob.npy"), &[n, HAND], &c.hand_prob)?;
        npy::write(&tmp.join("next_card.npy"), &[n], &c.next_card)?;
        npy::write(&tmp.join("recent_side.npy"), &[n, k], &c.recent_side)?;
        npy::write(&tmp.join("recent_card.npy"), &[n, k], &c.recent_card)?;
        npy::write(&tmp.join("recent_x.npy"), &[n, k], &c.recent_x)?;
        npy::write(&tmp.join("recent_y.npy"), &[n, k], &c.recent_y)?;
        npy::write(&tmp.join("recent_age.npy"), &[n, k], &c.recent_age)?;
        npy::write(&tmp.join("waited.npy"), &[n], &c.waited)?;
        npy::write(&tmp.join("card_slot.npy"), &[n], &c.card_slot)?;
        npy::write(&tmp.join("place_x.npy"), &[n], &c.place_x)?;
        npy::write(&tmp.join("place_y.npy"), &[n], &c.place_y)?;
        npy::write(&tmp.join("ability.npy"), &[n], &c.ability)?;
        npy::write(&tmp.join("ability_card.npy"), &[n], &c.ability_card)?;
        let index = Index {
            format: "cr-frames.v1",
            catalog_version: &self.catalog_version,
            step_ticks: self.step_ticks,
            recent: k,
            hand: HAND,
            grid: [18, 32],
            n_frames: n,
            samples: &self.metas,
        };
        std::fs::write(tmp.join("index.json"), serde_json::to_vec(&index)?)?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::rename(&tmp, &dir)?;
        self.written_shards += 1;
        self.written_samples += self.metas.len();
        self.written_frames += n;
        self.next_shard += 1;
        self.cols = FrameColumns::default();
        self.metas.clear();
        Ok(())
    }
}
