//! Card vocabulary + elixir timeline, loaded from `catalog/cards.json` and
//! `catalog/timeline.json` (checked into the repo next to this crate).
//!
//! Card ids are dense: 0 = empty slot, 1..=N = catalog cards in game-id order,
//! N+1 = unknown (a key the catalog does not know — event cards for modes we
//! do not model, or a future card). Dataset card keys carry a *form* suffix
//! (`-ev1` evolution, `-hero` hero); the base key selects the id and the form
//! is kept as a separate small integer.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CardKind {
    Troop = 0,
    Building = 1,
    Spell = 2,
    Other = 3,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CardEntry {
    pub id: u16,
    pub key: String,
    pub name: String,
    pub kind: String,
    pub cost: Option<f32>,
    #[serde(default)]
    pub attacks_air: Option<bool>,
    #[serde(default)]
    pub attacks_ground: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CardsFile {
    version: String,
    id_unknown: u16,
    cards: Vec<CardEntry>,
}

#[derive(Debug, Clone)]
pub struct Card {
    pub id: u16,
    pub key: String,
    pub kind: CardKind,
    pub cost: f32,
}

#[derive(Debug, Clone)]
pub struct Catalog {
    pub version: String,
    pub cards: Vec<Card>, // index = id - 1
    by_key: HashMap<String, u16>,
    pub unknown_id: u16,
}

/// Form encoded alongside a card id: 0 base, 1.. evolution level, 200 hero.
pub const FORM_BASE: u8 = 0;
pub const FORM_HERO: u8 = 200;

/// Split a dataset card key into (base key, form).
pub fn split_form(key: &str) -> (&str, u8) {
    if let Some(base) = key.strip_suffix("-hero") {
        return (base, FORM_HERO);
    }
    if let Some(pos) = key.rfind("-ev") {
        let (base, suffix) = key.split_at(pos);
        if let Ok(level) = suffix[3..].parse::<u8>() {
            return (base, level);
        }
    }
    (key, FORM_BASE)
}

impl Catalog {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let file: CardsFile = serde_json::from_str(&text).context("parse cards.json")?;
        let mut cards = Vec::with_capacity(file.cards.len());
        let mut by_key = HashMap::new();
        for (i, c) in file.cards.iter().enumerate() {
            anyhow::ensure!(c.id as usize == i + 1, "cards.json ids must be dense 1..N in order");
            let kind = match c.kind.as_str() {
                "troop" => CardKind::Troop,
                "building" => CardKind::Building,
                "spell" => CardKind::Spell,
                _ => CardKind::Other,
            };
            by_key.insert(c.key.clone(), c.id);
            cards.push(Card { id: c.id, key: c.key.clone(), kind, cost: c.cost.unwrap_or(0.0) });
        }
        Ok(Self { version: file.version, cards, by_key, unknown_id: file.id_unknown })
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// (id, form) for a dataset card key; unknown keys map to `unknown_id`.
    pub fn resolve(&self, key: &str) -> (u16, u8) {
        let (base, form) = split_form(key);
        (self.by_key.get(base).copied().unwrap_or(self.unknown_id), form)
    }

    pub fn cost(&self, id: u16) -> f32 {
        self.cards.get(id.wrapping_sub(1) as usize).map(|c| c.cost).unwrap_or(0.0)
    }

    pub fn kind(&self, id: u16) -> CardKind {
        self.cards.get(id.wrapping_sub(1) as usize).map(|c| c.kind).unwrap_or(CardKind::Other)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RateEntry {
    length_seconds: f64,
    seconds_per_elixir: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct SectionEntry {
    length_seconds: f64,
    section_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TimelineFile {
    tick_ms: u32,
    starting_elixir: f64,
    max_elixir: f64,
    sections: Vec<SectionEntry>,
    elixir_rates: Vec<RateEntry>,
}

#[derive(Debug, Clone)]
pub struct Timeline {
    pub tick_ms: u32,
    pub starting_elixir: f64,
    pub max_elixir: f64,
    /// consecutive (length_s, seconds_per_elixir) segments from t = 0
    pub rates: Vec<(f64, f64)>,
    pub normal_s: f64,
    pub overtime_s: f64,
}

impl Timeline {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let f: TimelineFile = serde_json::from_str(&text).context("parse timeline.json")?;
        let normal_s = f.sections.iter().filter(|s| s.section_type != "Overtime").map(|s| s.length_seconds).sum();
        let overtime_s = f.sections.iter().filter(|s| s.section_type == "Overtime").map(|s| s.length_seconds).sum();
        Ok(Self {
            tick_ms: f.tick_ms,
            starting_elixir: f.starting_elixir,
            max_elixir: f.max_elixir,
            rates: f.elixir_rates.iter().map(|r| (r.length_seconds, r.seconds_per_elixir)).collect(),
            normal_s,
            overtime_s,
        })
    }

    pub fn ticks_per_second(&self) -> f64 {
        1000.0 / self.tick_ms as f64
    }

    /// Elixir regeneration rate (elixir per second) at time `t_s`.
    pub fn rate_at(&self, t_s: f64) -> f64 {
        let mut start = 0.0;
        let mut last = self.rates.last().map(|r| r.1).unwrap_or(2.8);
        for &(len, spe) in &self.rates {
            if t_s < start + len {
                return 1.0 / spe;
            }
            start += len;
            last = spe;
        }
        1.0 / last
    }

    /// 0 = single, 1 = double, 2 = triple elixir (index of the rate segment, capped at 2).
    pub fn phase_at(&self, t_s: f64) -> u8 {
        let mut start = 0.0;
        for (i, &(len, _)) in self.rates.iter().enumerate() {
            if t_s < start + len {
                return i.min(2) as u8;
            }
            start += len;
        }
        (self.rates.len().saturating_sub(1)).min(2) as u8
    }

    pub fn is_overtime(&self, t_s: f64) -> bool {
        t_s >= self.normal_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_split() {
        assert_eq!(split_form("hog-rider"), ("hog-rider", 0));
        assert_eq!(split_form("knight-ev1"), ("knight", 1));
        assert_eq!(split_form("musketeer-hero"), ("musketeer", FORM_HERO));
        assert_eq!(split_form("x-evolution"), ("x-evolution", 0));
    }

    #[test]
    fn timeline_phases() {
        let tl = Timeline {
            tick_ms: 50,
            starting_elixir: 6.0,
            max_elixir: 10.0,
            rates: vec![(120.0, 2.8), (120.0, 1.4), (60.0, 0.93)],
            normal_s: 180.0,
            overtime_s: 120.0,
        };
        assert_eq!(tl.phase_at(0.0), 0);
        assert_eq!(tl.phase_at(119.9), 0);
        assert_eq!(tl.phase_at(120.0), 1);
        assert_eq!(tl.phase_at(240.0), 2);
        assert!((tl.rate_at(10.0) - 1.0 / 2.8).abs() < 1e-9);
        assert!(!tl.is_overtime(179.9) && tl.is_overtime(180.0));
    }
}
