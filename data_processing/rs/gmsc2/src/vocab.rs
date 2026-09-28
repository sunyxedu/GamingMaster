//! The frames' vocabulary: the client build's own ids (`RequestData`: unit types, abilities,
//! buffs, upgrades) as compact indices, stored as JSON (made by `gmdata sc2 data`, kept at
//! `data_processing/games/sc2/sc2_vocab_<build>.json`) so the converter and the simulator
//! read one table.
//!
//! Indices: units / actions / buffs 0 = none, 1 = unknown (an id the table lacks), 2.. the
//! table's entries in id order; upgrades 0.. (the frame's upgrade bits).  An action is a
//! *general* ability — the game's `remaps_to_ability_id` (every "attack" order is Attack,
//! every "stop" Stop); the table keeps the specific → general map of all abilities.

use anyhow::{bail, Context, Result};
use sc2api::proto;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// The right-click command.
pub const SMART: u32 = 1;

/// AbilityData.Target: what a general ability's command takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    None = 1,
    Point = 2,
    Unit = 3,
    PointOrUnit = 4,
    PointOrNone = 5,
}

impl Target {
    fn of(v: i32) -> Target {
        match v {
            2 => Target::Point,
            3 => Target::Unit,
            4 => Target::PointOrUnit,
            5 => Target::PointOrNone,
            _ => Target::None,
        }
    }
    /// target kinds (0 none / 1 point / 2 unit) the command accepts
    pub fn kinds(self) -> [bool; 3] {
        match self {
            Target::None => [true, false, false],
            Target::Point => [false, true, false],
            Target::Unit => [false, false, true],
            Target::PointOrUnit => [false, true, true],
            Target::PointOrNone => [true, true, false],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Action {
    pub id: u32,
    pub name: String,
    pub target: Target,
}

#[derive(Clone, Debug, Default)]
pub struct Vocab {
    pub build: u32,
    pub units: Vec<(u32, String)>,
    pub actions: Vec<Action>,
    pub buffs: Vec<(u32, String)>,
    pub upgrades: Vec<(u32, String)>,
    /// every ability id → its general ability id
    pub general: HashMap<u32, u32>,
    unit_idx: HashMap<u32, i64>,
    action_idx: HashMap<u32, i64>,
    buff_idx: HashMap<u32, i64>,
    upgrade_idx: HashMap<u32, usize>,
}

impl Vocab {
    fn index(&mut self) {
        self.unit_idx = self.units.iter().enumerate().map(|(i, (id, _))| (*id, i as i64 + 2)).collect();
        self.action_idx = self.actions.iter().enumerate().map(|(i, a)| (a.id, i as i64 + 2)).collect();
        self.buff_idx = self.buffs.iter().enumerate().map(|(i, (id, _))| (*id, i as i64 + 2)).collect();
        self.upgrade_idx = self.upgrades.iter().enumerate().map(|(i, (id, _))| (*id, i)).collect();
    }

    /// The table of a running game's data (every named entry, in id order; the general
    /// abilities are those the available abilities remap to, or themselves).
    pub fn from_data(d: &proto::ResponseData, build: u32) -> Vocab {
        let named = |s: &Option<String>| s.as_deref().is_some_and(|x| !x.is_empty());
        let mut units: Vec<(u32, String)> = d.units.iter().filter(|u| named(&u.name))
            .map(|u| (u.unit_id.unwrap_or(0), u.name.clone().unwrap_or_default())).collect();
        units.sort();
        let by_id: HashMap<u32, &proto::AbilityData> = d.abilities.iter().map(|a| (a.ability_id.unwrap_or(0), a)).collect();
        let mut general = HashMap::new();
        for a in &d.abilities {
            let id = a.ability_id.unwrap_or(0);
            if id == 0 || !named(&a.link_name) {
                continue;
            }
            let g = a.remaps_to_ability_id.filter(|&g| g != 0).unwrap_or(id);
            general.insert(id, g);
        }
        let mut gen_ids: Vec<u32> = d.abilities.iter()
            .filter(|a| a.available.unwrap_or(false) && named(&a.link_name) && a.ability_id.unwrap_or(0) != 0)
            .map(|a| {
                let id = a.ability_id.unwrap_or(0);
                general.get(&id).copied().unwrap_or(id)
            }).collect();
        // Smart (1, the right-click: move / attack / gather / rally by target) is marked
        // unavailable in the data, yet it is the commonest human command
        if general.contains_key(&SMART) || by_id.contains_key(&SMART) {
            gen_ids.push(SMART);
            general.insert(SMART, SMART);
        }
        gen_ids.sort();
        gen_ids.dedup();
        let actions = gen_ids.into_iter().map(|g| {
            let a = by_id.get(&g);
            let name = a.map(|a| {
                let f = a.friendly_name.clone().unwrap_or_default();
                if f.is_empty() { a.link_name.clone().unwrap_or_default() } else { f }
            }).unwrap_or_default();
            let (name, target) = if g == SMART && name.is_empty() { ("Smart".to_string(), 4) } else {
                (name, a.and_then(|a| a.target).unwrap_or(if g == SMART { 4 } else { 1 }))
            };
            Action { id: g, name, target: Target::of(target) }
        }).collect();
        let mut buffs: Vec<(u32, String)> = d.buffs.iter().filter(|b| named(&b.name))
            .map(|b| (b.buff_id.unwrap_or(0), b.name.clone().unwrap_or_default())).collect();
        buffs.sort();
        let mut upgrades: Vec<(u32, String)> = d.upgrades.iter().filter(|u| named(&u.name))
            .map(|u| (u.upgrade_id.unwrap_or(0), u.name.clone().unwrap_or_default())).collect();
        upgrades.sort();
        let mut v = Vocab { build, units, actions, buffs, upgrades, general, ..Default::default() };
        v.index();
        v
    }

    pub fn to_json(&self) -> Value {
        let pairs = |xs: &[(u32, String)]| xs.iter().map(|(i, n)| json!([i, n])).collect::<Vec<_>>();
        let mut general: Vec<(u32, u32)> = self.general.iter().filter(|(a, g)| a != g).map(|(a, g)| (*a, *g)).collect();
        general.sort();
        json!({
            "build": self.build,
            "units": pairs(&self.units),
            "actions": self.actions.iter().map(|a| json!([a.id, a.name, a.target as i32])).collect::<Vec<_>>(),
            "buffs": pairs(&self.buffs),
            "upgrades": pairs(&self.upgrades),
            "general": general.iter().map(|(a, g)| json!([a, g])).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(v: &Value) -> Result<Vocab> {
        let pairs = |k: &str| -> Result<Vec<(u32, String)>> {
            v[k].as_array().with_context(|| format!("vocab: no {k}"))?.iter().map(|p| {
                Ok((p[0].as_u64().context("vocab id")? as u32, p[1].as_str().unwrap_or("").to_string()))
            }).collect()
        };
        let actions = v["actions"].as_array().context("vocab: no actions")?.iter().map(|p| {
            Ok(Action { id: p[0].as_u64().context("action id")? as u32, name: p[1].as_str().unwrap_or("").to_string(),
                        target: Target::of(p[2].as_i64().unwrap_or(1) as i32) })
        }).collect::<Result<Vec<_>>>()?;
        let mut general: HashMap<u32, u32> = actions.iter().map(|a| (a.id, a.id)).collect();
        for p in v["general"].as_array().context("vocab: no general")? {
            general.insert(p[0].as_u64().context("ability")? as u32, p[1].as_u64().context("general")? as u32);
        }
        let mut voc = Vocab {
            build: v["build"].as_u64().unwrap_or(0) as u32,
            units: pairs("units")?,
            actions,
            buffs: pairs("buffs")?,
            upgrades: pairs("upgrades")?,
            general,
            ..Default::default()
        };
        if voc.units.is_empty() || voc.actions.is_empty() {
            bail!("vocab: empty unit or action table");
        }
        voc.index();
        Ok(voc)
    }

    pub fn load(path: &Path) -> Result<Vocab> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Vocab::from_json(&serde_json::from_str(&text)?)
    }

    pub fn unit_id(&self, unit_type: u32) -> i64 {
        if unit_type == 0 {
            return 0;
        }
        self.unit_idx.get(&unit_type).copied().unwrap_or(1)
    }

    /// The action index of any ability id (through its general ability); 0 for id 0.
    pub fn action_id(&self, ability: u32) -> i64 {
        if ability == 0 {
            return 0;
        }
        let g = self.general.get(&ability).copied().unwrap_or(ability);
        self.action_idx.get(&g).copied().unwrap_or(1)
    }

    pub fn buff_id(&self, buff: u32) -> i64 {
        if buff == 0 {
            return 0;
        }
        self.buff_idx.get(&buff).copied().unwrap_or(1)
    }

    pub fn upgrade_index(&self, upgrade: u32) -> Option<usize> {
        self.upgrade_idx.get(&upgrade).copied()
    }

    /// The general ability of an action index (None for none / unknown).
    pub fn ability_of(&self, action: i64) -> Option<u32> {
        if action < 2 {
            return None;
        }
        self.actions.get(action as usize - 2).map(|a| a.id)
    }

    pub fn target_of(&self, action: i64) -> Option<Target> {
        if action < 2 {
            return None;
        }
        self.actions.get(action as usize - 2).map(|a| a.target)
    }

    /// Sizes the network's tables need (index 0 / 1 included).
    pub fn n_units(&self) -> usize {
        self.units.len() + 2
    }
    pub fn n_actions(&self) -> usize {
        self.actions.len() + 2
    }
    pub fn n_buffs(&self) -> usize {
        self.buffs.len() + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> proto::ResponseData {
        let ab = |id: u32, name: &str, remap: u32, target: i32, available: bool| proto::AbilityData {
            ability_id: Some(id), link_name: Some(name.into()), remaps_to_ability_id: Some(remap),
            target: Some(target), available: Some(available), ..Default::default()
        };
        proto::ResponseData {
            abilities: vec![ab(23, "attack", 3674, 4, true), ab(3674, "Attack", 0, 4, true), ab(4, "stop", 0, 1, true),
                            ab(99, "gone", 0, 2, false)],
            units: vec![proto::UnitTypeData { unit_id: Some(48), name: Some("Marine".into()), ..Default::default() },
                        proto::UnitTypeData { unit_id: Some(45), name: Some("SCV".into()), ..Default::default() },
                        proto::UnitTypeData { unit_id: Some(7), name: Some(String::new()), ..Default::default() }],
            upgrades: vec![proto::UpgradeData { upgrade_id: Some(7), name: Some("Stim".into()), ..Default::default() }],
            buffs: vec![proto::BuffData { buff_id: Some(27), name: Some("Stimpack".into()) }],
            effects: vec![],
        }
    }

    #[test]
    fn indices_and_round_trip() {
        let v = Vocab::from_data(&data(), 55958);
        assert_eq!(v.units.iter().map(|u| u.0).collect::<Vec<_>>(), vec![45, 48]);
        assert_eq!((v.unit_id(0), v.unit_id(45), v.unit_id(48), v.unit_id(1234)), (0, 2, 3, 1));
        // attack remaps to the general Attack; the unavailable ability is no action
        assert_eq!(v.actions.iter().map(|a| a.id).collect::<Vec<_>>(), vec![4, 3674]);
        assert_eq!(v.action_id(23), v.action_id(3674));
        assert_eq!((v.action_id(0), v.action_id(4), v.action_id(99)), (0, 2, 1));
        assert_eq!(v.ability_of(v.action_id(23)), Some(3674));
        assert_eq!(v.target_of(v.action_id(4)), Some(Target::None));
        assert_eq!(v.upgrade_index(7), Some(0));
        assert_eq!(v.buff_id(27), 2);
        let back = Vocab::from_json(&v.to_json()).unwrap();
        assert_eq!(back.to_json(), v.to_json());
        assert_eq!(back.action_id(23), v.action_id(23));
    }
}
