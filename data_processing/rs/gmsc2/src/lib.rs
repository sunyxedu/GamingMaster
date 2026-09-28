//! StarCraft II frames for the tier-3 network, shared by the replay converter (gmdata `sc2
//! convert`) and the simulator (gmsim, env/vec_sc2.py), so an RL frame is the IL frame of the
//! same game state: one player's raw interface (units, map, player), seen every [`STEP_MUL`]
//! game loops, and its raw unit commands as actions.
//!
//! Decisions.  The game is observed once per window of [`STEP_MUL`] loops; within a window
//! the player issues any number of commands (a strong human up to ~9 in 8 loops), each one a
//! decision on the window's frame, then "done" ends the window (action_type 0).  A decision
//! sees what it has already done in the window: `unit_commanded` (the units given a command so
//! far), the count (globals) and the last command's type.  The 3.16.1 client does not report
//! an action's game loop, so a window's commands keep their order but not their exact loops:
//! the env sends a decision's command at once and steps the game when the window is done —
//! the same frames, contexts and labels in both directions.  At most [`K_MAX`] commands a
//! window (the env ends the window after them).
//!
//! Unit tokens keep their slot while the unit stays observed (the network's per-slot
//! recurrence follows one unit): own units in slots [0, OWN), enemy [OWN, OWN + ENEMY),
//! neutral (resources, rocks, towers) in the rest; a new unit takes its range's lowest free
//! slot — neutral ones nearest to the player's own units first — and a unit with no free slot
//! stays out (counted; the resource plane still shows every mineral field and geyser).
//!
//! Frame (per window; `Encoder::encode`):
//!
//!   units         [U, F] u8   alliance / display / cloak (categorical), flags (bits), and
//!                             quantised features ([`feature`]; 0..255 = 0..1)
//!   unit_xy       [U, 2] u16  position / map size × 65535
//!   unit_type     [U] u16     vocabulary id (0 empty slot, 1 unknown type)
//!   unit_order    [U] u16     the first order's (general) ability, 0 none
//!   unit_buffs    [U, 2] u16  buff ids, 0 none
//!   unit_mask     [U] bool    a unit in the slot
//!   unit_own      [U] bool    the player's own unit (the possible actors)
//!   map_bits      [9, 64, 64] visible, explored, creep, pathable, placeable, resource, height
//!                             above 1/4, 1/2, 3/4 of the map's range — the map sampled at
//!                             64 × 64 (world (x, y) → cell (⌊64x/w⌋, ⌊64y/h⌋))
//!   globals       [G] f32     minerals, gas, supply, army / workers, idle workers, larva,
//!                             warp gates, time, own / enemy race, map size, window commands
//!   upgrades      [NU] bits   researched upgrades (the vocabulary's)
//!
//! Decision context: unit_commanded [U] bits, last_action (action id, 0 none), globals
//! [WINDOW_CMDS] = commands so far / K_MAX.
//!
//! Action:
//!
//!   action_type   0 = done (end the window), else the command's general ability (the game's
//!                 `remaps_to_ability_id`: every attack is Attack; the vocabulary's id)
//!   actors [U]    the command's units among the own tokens
//!   target_kind   0 none / 1 point / 2 unit
//!   target_unit   the target's token, −1 none
//!   target_point  the target on a LOC × LOC grid over the map (y · LOC + x), −1 none
//!   queued        shift-queued

pub mod vocab;

use sc2api::proto;
use std::collections::HashMap;
pub use vocab::Vocab;

/// Game loops per window (22.4 loops a second at "faster": 2.8 windows a second).
pub const STEP_MUL: u32 = 8;
/// Commands a window takes at most.
pub const K_MAX: usize = 16;
pub const U: usize = 320;
pub const OWN: usize = 192;
pub const ENEMY: usize = 80;
pub const NEUTRAL: usize = U - OWN - ENEMY;
pub const F: usize = 27;
pub const BUFFS: usize = 2;
pub const MAP: usize = 64;
pub const MAP_BITS: usize = 9;
pub const G: usize = 21;
/// Target-point grid (LOC × LOC over the map).
pub const LOC: usize = 256;

/// Unit feature columns (u8).  Three categorical columns (the adapter embeds them), one byte
/// of flags (bit k = FLAG_*), the rest continuous (0..255 stands for 0..1).
pub mod feature {
    pub const ALLIANCE: usize = 0; // 0 self, 1 ally, 2 neutral, 3 enemy
    pub const DISPLAY: usize = 1; // 0 visible, 1 snapshot, 2 hidden, 3 placeholder
    pub const CLOAK: usize = 2; // 0 cloaked, 1 cloaked detected, 2 not cloaked, 3 cloaked allied
    pub const FLAGS: usize = 3;
    pub const FLAG_FLYING: u8 = 1;
    pub const FLAG_BURROWED: u8 = 2;
    pub const FLAG_HALLUCINATION: u8 = 4;
    pub const FLAG_POWERED: u8 = 8;
    pub const FLAG_ACTIVE: u8 = 16;
    pub const FLAG_ADD_ON: u8 = 32;
    pub const N_FLAGS: usize = 6;
    /// the first continuous column
    pub const CONTINUOUS: usize = 4;
    pub const HEALTH: usize = 4; // ratio
    pub const HEALTH_MAX: usize = 5; // / 2000
    pub const SHIELD: usize = 6;
    pub const SHIELD_MAX: usize = 7; // / 2000
    pub const ENERGY: usize = 8;
    pub const ENERGY_MAX: usize = 9; // / 200
    pub const FACING: usize = 10; // 2: (sin + 1) / 2, (cos + 1) / 2
    pub const RADIUS: usize = 12; // / 8
    pub const BUILD: usize = 13;
    pub const MINERALS: usize = 14; // / 1800
    pub const VESPENE: usize = 15; // / 2250
    pub const HARVEST: usize = 16; // assigned / ideal, / 2
    pub const IDEAL: usize = 17; // / 16
    pub const COOLDOWN: usize = 18; // / 32
    pub const ORDERS: usize = 19; // / 4
    pub const PROGRESS: usize = 20; // the first order's
    pub const CARGO: usize = 21; // / 8
    pub const CARGO_MAX: usize = 22; // / 8
    pub const UPGRADES: usize = 23; // 3: attack, armour, shield level / 3
    pub const BUFF_TIME: usize = 26;
}

/// Global feature columns.
pub mod global {
    pub const MINERALS: usize = 0; // / 1000
    pub const VESPENE: usize = 1; // / 1000
    pub const FOOD_USED: usize = 2; // / 200
    pub const FOOD_CAP: usize = 3;
    pub const FOOD_ARMY: usize = 4;
    pub const FOOD_WORKERS: usize = 5; // / 100
    pub const IDLE_WORKERS: usize = 6; // / 50
    pub const ARMY_COUNT: usize = 7; // / 200
    pub const WARP_GATES: usize = 8; // / 20
    pub const LARVA: usize = 9; // / 20
    pub const TIME: usize = 10; // / 30 minutes
    pub const RACE: usize = 11; // 3: terran, zerg, protoss
    pub const ENEMY_RACE: usize = 14; // 4: terran, zerg, protoss, unknown (random)
    pub const MAP_W: usize = 18; // / 256
    pub const MAP_H: usize = 19;
    pub const WINDOW_CMDS: usize = 20; // / K_MAX
}

/// The map-bit planes.
pub mod plane {
    pub const VISIBLE: usize = 0;
    pub const EXPLORED: usize = 1;
    pub const CREEP: usize = 2;
    pub const PATHABLE: usize = 3;
    pub const PLACEABLE: usize = 4;
    pub const RESOURCE: usize = 5;
    /// 3: terrain height above 1/4, 1/2, 3/4 of the map's height range (the levels of low
    /// ground, main, high ground)
    pub const HEIGHT: usize = 6;
}

/// A game's static facts (ResponseGameInfo).
#[derive(Clone, Debug)]
pub struct MapInfo {
    pub w: usize,
    pub h: usize,
    /// at 64 × 64: pathable / placeable (0 / 1), height (0..255)
    pub pathing: Vec<u8>,
    pub placement: Vec<u8>,
    pub height: Vec<u8>,
    /// the height thermometer planes (3 × 64 × 64, 0 / 1)
    pub height_bits: Vec<u8>,
    /// race of each player id (index id − 1): 1 terran, 2 zerg, 3 protoss, 4 random
    pub races: Vec<i32>,
}

fn bit(data: &[u8], i: usize) -> u8 {
    (data.get(i / 8).copied().unwrap_or(0) >> (7 - (i % 8))) & 1
}

/// A game image (1 bit or 1 byte per cell) sampled at the 64 × 64 cell centres; 1-bit images
/// give 0 / 1.
fn sample(img: Option<&proto::ImageData>, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; MAP * MAP];
    let Some(img) = img else { return out };
    let bpp = img.bits_per_pixel.unwrap_or(8);
    let (iw, ih) = img.size.as_ref().map(|s| (s.x.unwrap_or(w as i32).max(1) as usize, s.y.unwrap_or(h as i32).max(1) as usize))
        .unwrap_or((w, h));
    let data = img.data.as_deref().unwrap_or(&[]);
    for r in 0..MAP {
        let y = (((r as f64 + 0.5) * ih as f64 / MAP as f64) as usize).min(ih - 1);
        for c in 0..MAP {
            let x = (((c as f64 + 0.5) * iw as f64 / MAP as f64) as usize).min(iw - 1);
            let i = y * iw + x;
            out[r * MAP + c] = if bpp == 1 { bit(data, i) } else { data.get(i).copied().unwrap_or(0) };
        }
    }
    out
}

impl MapInfo {
    pub fn from_game_info(gi: &proto::ResponseGameInfo) -> MapInfo {
        let sr = gi.start_raw.as_ref();
        let (w, h) = sr.and_then(|s| s.map_size.as_ref())
            .map(|m| (m.x.unwrap_or(1).max(1) as usize, m.y.unwrap_or(1).max(1) as usize)).unwrap_or((1, 1));
        let mut races = vec![0; gi.player_info.len()];
        for p in &gi.player_info {
            let i = p.player_id.unwrap_or(0) as usize;
            if (1..=races.len()).contains(&i) {
                races[i - 1] = p.race_actual.filter(|&r| r != 0).or(p.race_requested).unwrap_or(0);
            }
        }
        let pathing: Vec<u8> = sample(sr.and_then(|s| s.pathing_grid.as_ref()), w, h).iter().map(|&v| (v != 0) as u8).collect();
        let placement: Vec<u8> = sample(sr.and_then(|s| s.placement_grid.as_ref()), w, h).iter().map(|&v| (v != 0) as u8).collect();
        let height = sample(sr.and_then(|s| s.terrain_height.as_ref()), w, h);
        MapInfo::new(w, h, pathing, placement, height, races)
    }

    pub fn new(w: usize, h: usize, pathing: Vec<u8>, placement: Vec<u8>, height: Vec<u8>, races: Vec<i32>) -> MapInfo {
        // the range over the pathable or placeable cells (the playable ground)
        let ground: Vec<u8> = (0..MAP * MAP).filter(|&i| pathing[i] != 0 || placement[i] != 0).map(|i| height[i]).collect();
        let (lo, hi) = ground.iter().fold((u8::MAX, 0u8), |(a, b), &v| (a.min(v), b.max(v)));
        let mut height_bits = vec![0u8; 3 * MAP * MAP];
        if hi > lo {
            for k in 0..3 {
                let t = lo as f32 + (hi - lo) as f32 * (k + 1) as f32 / 4.0;
                for i in 0..MAP * MAP {
                    height_bits[k * MAP * MAP + i] = (height[i] as f32 > t) as u8;
                }
            }
        }
        MapInfo { w, h, pathing, placement, height, height_bits, races }
    }

    /// The 64 × 64 cell of a world point.
    pub fn cell(&self, x: f32, y: f32) -> usize {
        let c = ((x as f64 * MAP as f64 / self.w as f64).floor() as i64).clamp(0, MAP as i64 - 1) as usize;
        let r = ((y as f64 * MAP as f64 / self.h as f64).floor() as i64).clamp(0, MAP as i64 - 1) as usize;
        r * MAP + c
    }

    /// The LOC × LOC target cell of a world point.
    pub fn loc(&self, x: f32, y: f32) -> usize {
        let c = ((x as f64 * LOC as f64 / self.w as f64).floor() as i64).clamp(0, LOC as i64 - 1) as usize;
        let r = ((y as f64 * LOC as f64 / self.h as f64).floor() as i64).clamp(0, LOC as i64 - 1) as usize;
        r * LOC + c
    }

    /// The world point at a target cell's centre.
    pub fn loc_point(&self, loc: usize) -> (f32, f32) {
        let (r, c) = (loc / LOC, loc % LOC);
        (((c as f64 + 0.5) * self.w as f64 / LOC as f64) as f32, ((r as f64 + 0.5) * self.h as f64 / LOC as f64) as f32)
    }
}

/// One window's frame (module docs) and the unit tag of every slot.
#[derive(Clone, Debug)]
pub struct Frame {
    pub units: Vec<u8>,
    pub unit_xy: Vec<u16>,
    pub unit_type: Vec<u16>,
    pub unit_order: Vec<u16>,
    pub unit_buffs: Vec<u16>,
    pub unit_mask: Vec<bool>,
    pub unit_own: Vec<bool>,
    pub map_bits: Vec<u8>,
    pub globals: Vec<f32>,
    pub upgrades: Vec<u8>,
    pub tags: Vec<u64>,
    pub game_loop: u32,
}

/// A decision's context within its window (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub commanded: Vec<bool>,
    pub n: usize,
    pub last_action: i64,
}

impl Default for Window {
    fn default() -> Self {
        Window { commanded: vec![false; U], n: 0, last_action: 0 }
    }
}

impl Window {
    /// The context after `l` (a command) was issued.
    pub fn after(&self, l: &Label) -> Window {
        let mut w = self.clone();
        for (c, &a) in w.commanded.iter_mut().zip(&l.actors) {
            *c |= a;
        }
        w.n += 1;
        w.last_action = l.action_type;
        w
    }

    pub fn window_cmds(&self) -> f32 {
        self.n.min(K_MAX) as f32 / K_MAX as f32
    }
}

fn alliance_idx(a: i32) -> usize {
    match proto::Alliance::try_from(a) {
        Ok(proto::Alliance::Self_) => 0,
        Ok(proto::Alliance::Ally) => 1,
        Ok(proto::Alliance::Neutral) => 2,
        _ => 3,
    }
}

/// Token group of an alliance: 0 own (and allied), 1 enemy, 2 neutral.
fn group_of(u: &proto::Unit) -> usize {
    match alliance_idx(u.alliance.unwrap_or(0)) {
        0 | 1 => 0,
        3 => 1,
        _ => 2,
    }
}

const RANGES: [(usize, usize); 3] = [(0, OWN), (OWN, OWN + ENEMY), (OWN + ENEMY, U)];

fn q(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn pos(u: &proto::Unit) -> (f32, f32) {
    u.pos.as_ref().map(|p| (p.x.unwrap_or(0.0), p.y.unwrap_or(0.0))).unwrap_or((0.0, 0.0))
}

/// Frames of one player's game, in order: the slot of every observed unit (module docs).
#[derive(Clone, Debug)]
pub struct Encoder {
    pub map: MapInfo,
    pub player_id: u32,
    tags: Vec<u64>,
    slot_of: HashMap<u64, usize>,
    /// units observed without a free slot, summed over frames
    pub over_cap: usize,
}

impl Encoder {
    pub fn new(map: MapInfo, player_id: u32) -> Encoder {
        Encoder { map, player_id, tags: vec![0; U], slot_of: HashMap::new(), over_cap: 0 }
    }

    pub fn slot(&self, tag: u64) -> Option<usize> {
        self.slot_of.get(&tag).copied()
    }

    /// Slots: release the units gone, keep the rest, place the new ones.
    fn allocate(&mut self, units: &[&proto::Unit]) {
        let present: std::collections::HashSet<u64> = units.iter().map(|u| u.tag.unwrap_or(0)).collect();
        for s in 0..U {
            if self.tags[s] != 0 && !present.contains(&self.tags[s]) {
                self.slot_of.remove(&self.tags[s]);
                self.tags[s] = 0;
            }
        }
        let own: Vec<(f32, f32)> = units.iter().filter(|u| group_of(u) == 0).map(|u| pos(u)).collect();
        let near = |u: &proto::Unit| -> f32 {
            let (x, y) = pos(u);
            own.iter().map(|&(a, b)| (a - x) * (a - x) + (b - y) * (b - y)).fold(f32::INFINITY, f32::min)
        };
        let mut new: Vec<(usize, f32, u64)> = units.iter().filter(|u| !self.slot_of.contains_key(&u.tag.unwrap_or(0)))
            .map(|u| {
                let g = group_of(u);
                (g, if g == 2 { near(u) } else { 0.0 }, u.tag.unwrap_or(0))
            }).collect();
        new.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (g, _, tag) in new {
            let (lo, hi) = RANGES[g];
            match (lo..hi).find(|&s| self.tags[s] == 0) {
                Some(s) => {
                    self.tags[s] = tag;
                    self.slot_of.insert(tag, s);
                }
                None => self.over_cap += 1,
            }
        }
    }

    /// The window's frame (module docs; globals[WINDOW_CMDS] 0: set per decision).
    pub fn encode(&mut self, obs: &proto::Observation, voc: &Vocab) -> Frame {
        let map = self.map.clone();
        let raw = obs.raw_data.as_ref();
        let units: Vec<&proto::Unit> = raw.map(|r| r.units.iter().collect()).unwrap_or_default();
        self.allocate(&units);
        let mut f = Frame {
            units: vec![0; U * F],
            unit_xy: vec![0; U * 2],
            unit_type: vec![0; U],
            unit_order: vec![0; U],
            unit_buffs: vec![0; U * BUFFS],
            unit_mask: vec![false; U],
            unit_own: vec![false; U],
            map_bits: vec![0; MAP_BITS * MAP * MAP],
            globals: vec![0.0; G],
            upgrades: vec![0; voc.upgrades.len()],
            tags: self.tags.clone(),
            game_loop: obs.game_loop.unwrap_or(0),
        };
        let ratio = |a: Option<f32>, b: Option<f32>| match (a, b) {
            (Some(a), Some(b)) if b > 0.0 => a / b,
            _ => 0.0,
        };
        let pl = |p: usize| p * MAP * MAP..(p + 1) * MAP * MAP;
        for u in &units {
            let (x, y) = pos(u);
            if group_of(u) == 2 && (u.mineral_contents.unwrap_or(0) > 0 || u.vespene_contents.unwrap_or(0) > 0) {
                f.map_bits[plane::RESOURCE * MAP * MAP + map.cell(x, y)] = 1;
            }
            let Some(s) = self.slot(u.tag.unwrap_or(0)) else { continue };
            use feature::*;
            let row = &mut f.units[s * F..(s + 1) * F];
            let a = alliance_idx(u.alliance.unwrap_or(0));
            row[ALLIANCE] = a as u8;
            row[DISPLAY] = (u.display_type.unwrap_or(1) - 1).clamp(0, 3) as u8;
            row[CLOAK] = (u.cloak.unwrap_or(3) - 1).clamp(0, 3) as u8;
            let flag = |b: Option<bool>, v: u8| if b.unwrap_or(false) { v } else { 0 };
            row[FLAGS] = flag(u.is_flying, FLAG_FLYING) | flag(u.is_burrowed, FLAG_BURROWED)
                | flag(u.is_hallucination, FLAG_HALLUCINATION) | flag(u.is_powered, FLAG_POWERED)
                | flag(u.is_active, FLAG_ACTIVE) | flag(Some(u.add_on_tag.unwrap_or(0) != 0), FLAG_ADD_ON);
            row[HEALTH] = q(ratio(u.health, u.health_max));
            row[HEALTH_MAX] = q(u.health_max.unwrap_or(0.0) / 2000.0);
            row[SHIELD] = q(ratio(u.shield, u.shield_max));
            row[SHIELD_MAX] = q(u.shield_max.unwrap_or(0.0) / 2000.0);
            row[ENERGY] = q(ratio(u.energy, u.energy_max));
            row[ENERGY_MAX] = q(u.energy_max.unwrap_or(0.0) / 200.0);
            let fa = u.facing.unwrap_or(0.0);
            row[FACING] = q((fa.sin() + 1.0) / 2.0);
            row[FACING + 1] = q((fa.cos() + 1.0) / 2.0);
            row[RADIUS] = q(u.radius.unwrap_or(0.0) / 8.0);
            row[BUILD] = q(u.build_progress.unwrap_or(0.0));
            row[MINERALS] = q(u.mineral_contents.unwrap_or(0) as f32 / 1800.0);
            row[VESPENE] = q(u.vespene_contents.unwrap_or(0) as f32 / 2250.0);
            let (asg, ideal) = (u.assigned_harvesters.unwrap_or(0) as f32, u.ideal_harvesters.unwrap_or(0) as f32);
            row[HARVEST] = q(if ideal > 0.0 { asg / ideal / 2.0 } else { 0.0 });
            row[IDEAL] = q(ideal / 16.0);
            row[COOLDOWN] = q(u.weapon_cooldown.unwrap_or(0.0) / 32.0);
            row[ORDERS] = q(u.orders.len() as f32 / 4.0);
            row[PROGRESS] = q(u.orders.first().and_then(|o| o.progress).unwrap_or(0.0));
            row[CARGO] = q(u.cargo_space_taken.unwrap_or(0) as f32 / 8.0);
            row[CARGO_MAX] = q(u.cargo_space_max.unwrap_or(0) as f32 / 8.0);
            row[UPGRADES] = q(u.attack_upgrade_level.unwrap_or(0) as f32 / 3.0);
            row[UPGRADES + 1] = q(u.armor_upgrade_level.unwrap_or(0) as f32 / 3.0);
            row[UPGRADES + 2] = q(u.shield_upgrade_level.unwrap_or(0) as f32 / 3.0);
            row[BUFF_TIME] = q(ratio(u.buff_duration_remain.map(|v| v as f32), u.buff_duration_max.map(|v| v as f32)));
            f.unit_xy[s * 2] = ((x / map.w as f32).clamp(0.0, 1.0) * 65535.0).round() as u16;
            f.unit_xy[s * 2 + 1] = ((y / map.h as f32).clamp(0.0, 1.0) * 65535.0).round() as u16;
            f.unit_type[s] = voc.unit_id(u.unit_type.unwrap_or(0)).min(u16::MAX as i64) as u16;
            f.unit_order[s] = u.orders.first().map(|o| voc.action_id(o.ability_id.unwrap_or(0))).unwrap_or(0) as u16;
            for (k, b) in u.buff_ids.iter().take(BUFFS).enumerate() {
                f.unit_buffs[s * BUFFS + k] = voc.buff_id(*b).min(u16::MAX as i64) as u16;
            }
            f.unit_mask[s] = true;
            f.unit_own[s] = a == 0;
        }
        if let Some(ms) = raw.and_then(|r| r.map_state.as_ref()) {
            let vis = sample(ms.visibility.as_ref(), map.w, map.h);
            let creep = sample(ms.creep.as_ref(), map.w, map.h);
            for i in 0..MAP * MAP {
                // visibility: 0 hidden, 1 fogged (explored), 2 visible
                f.map_bits[pl(plane::VISIBLE).start + i] = (vis[i] == 2) as u8;
                f.map_bits[pl(plane::EXPLORED).start + i] = (vis[i] >= 1) as u8;
                f.map_bits[pl(plane::CREEP).start + i] = (creep[i] != 0) as u8;
            }
        }
        f.map_bits[pl(plane::PATHABLE)].copy_from_slice(&map.pathing);
        f.map_bits[pl(plane::PLACEABLE)].copy_from_slice(&map.placement);
        f.map_bits[plane::HEIGHT * MAP * MAP..(plane::HEIGHT + 3) * MAP * MAP].copy_from_slice(&map.height_bits);
        {
            use global::*;
            if let Some(pc) = obs.player_common.as_ref() {
                f.globals[MINERALS] = pc.minerals.unwrap_or(0) as f32 / 1000.0;
                f.globals[VESPENE] = pc.vespene.unwrap_or(0) as f32 / 1000.0;
                f.globals[FOOD_USED] = pc.food_used.unwrap_or(0) as f32 / 200.0;
                f.globals[FOOD_CAP] = pc.food_cap.unwrap_or(0) as f32 / 200.0;
                f.globals[FOOD_ARMY] = pc.food_army.unwrap_or(0) as f32 / 200.0;
                f.globals[FOOD_WORKERS] = pc.food_workers.unwrap_or(0) as f32 / 100.0;
                f.globals[IDLE_WORKERS] = (pc.idle_worker_count.unwrap_or(0) as f32 / 50.0).min(1.0);
                f.globals[ARMY_COUNT] = (pc.army_count.unwrap_or(0) as f32 / 200.0).min(1.0);
                f.globals[WARP_GATES] = (pc.warp_gate_count.unwrap_or(0) as f32 / 20.0).min(1.0);
                f.globals[LARVA] = (pc.larva_count.unwrap_or(0) as f32 / 20.0).min(1.0);
            }
            f.globals[TIME] = (f.game_loop as f32 / (22.4 * 60.0 * 30.0)).min(2.0);
            let me = self.player_id.saturating_sub(1) as usize;
            let race_of = |i: usize| map.races.get(i).copied().unwrap_or(0);
            let r = race_of(me);
            if (1..=3).contains(&r) {
                f.globals[RACE + (r - 1) as usize] = 1.0;
            }
            let e = race_of(if me == 0 { 1 } else { 0 });
            f.globals[ENEMY_RACE + if (1..=3).contains(&e) { (e - 1) as usize } else { 3 }] = 1.0;
            f.globals[MAP_W] = map.w as f32 / 256.0;
            f.globals[MAP_H] = map.h as f32 / 256.0;
        }
        if let Some(p) = raw.and_then(|r| r.player.as_ref()) {
            for up in &p.upgrade_ids {
                if let Some(i) = voc.upgrade_index(*up) {
                    f.upgrades[i] = 1;
                }
            }
        }
        f
    }
}

/// A decision's action (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub action_type: i64,
    pub actors: Vec<bool>,
    pub target_kind: i64,
    pub target_unit: i64,
    pub target_point: i64,
    pub queued: bool,
}

impl Label {
    /// End the window.
    pub fn done() -> Label {
        Label { action_type: 0, actors: vec![false; U], target_kind: 0, target_unit: -1, target_point: -1, queued: false }
    }
}

/// The player's raw unit commands of an observation's actions, in order (camera moves and
/// autocast toggles are no decisions).
pub fn unit_commands(actions: &[proto::Action]) -> Vec<&proto::ActionRawUnitCommand> {
    actions.iter().filter_map(|a| match a.action_raw.as_ref().and_then(|r| r.action.as_ref()) {
        Some(proto::action_raw::Action::UnitCommand(uc)) => Some(uc),
        _ => None,
    }).collect()
}

/// Why a command is no label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// its ability is not in the vocabulary
    UnknownAbility,
    /// none of its units has an own token
    NoActor,
}

/// The label of a command on the window's frame (a unit target without a token: no target).
pub fn label(uc: &proto::ActionRawUnitCommand, frame: &Frame, enc: &Encoder, voc: &Vocab) -> Result<Label, Skip> {
    let action_type = voc.action_id(uc.ability_id.unwrap_or(0).max(0) as u32);
    if action_type <= 1 {
        return Err(Skip::UnknownAbility);
    }
    let mut actors = vec![false; U];
    let mut any = false;
    for t in &uc.unit_tags {
        if let Some(s) = enc.slot(*t) {
            if frame.unit_own[s] {
                actors[s] = true;
                any = true;
            }
        }
    }
    if !any {
        return Err(Skip::NoActor);
    }
    let (target_kind, target_unit, target_point) = match &uc.target {
        Some(proto::action_raw_unit_command::Target::TargetWorldSpacePos(p)) => {
            (1, -1, enc.map.loc(p.x.unwrap_or(0.0), p.y.unwrap_or(0.0)) as i64)
        }
        Some(proto::action_raw_unit_command::Target::TargetUnitTag(t)) => match enc.slot(*t).filter(|&s| frame.unit_mask[s]) {
            Some(s) => (2, s as i64, -1),
            None => (0, -1, -1),
        },
        None => (0, -1, -1),
    };
    Ok(Label { action_type, actors, target_kind, target_unit, target_point, queued: uc.queue_command.unwrap_or(false) })
}

/// The raw command of a decision on the window's frame (None: done, or nothing to command).
pub fn decode(l: &Label, frame: &Frame, enc: &Encoder, voc: &Vocab) -> Option<proto::Action> {
    if l.action_type <= 1 {
        return None;
    }
    let ability = voc.ability_of(l.action_type)?;
    let unit_tags: Vec<u64> = (0..U).filter(|&s| l.actors.get(s).copied().unwrap_or(false) && frame.unit_own[s])
        .map(|s| frame.tags[s]).collect();
    if unit_tags.is_empty() {
        return None;
    }
    let target = match l.target_kind {
        1 if l.target_point >= 0 && (l.target_point as usize) < LOC * LOC => {
            let (x, y) = enc.map.loc_point(l.target_point as usize);
            Some(proto::action_raw_unit_command::Target::TargetWorldSpacePos(proto::Point2D { x: Some(x), y: Some(y) }))
        }
        2 if l.target_unit >= 0 && (l.target_unit as usize) < U && frame.unit_mask[l.target_unit as usize] => {
            Some(proto::action_raw_unit_command::Target::TargetUnitTag(frame.tags[l.target_unit as usize]))
        }
        _ => None,
    };
    let uc = proto::ActionRawUnitCommand { ability_id: Some(ability as i32), target, unit_tags, queue_command: Some(l.queued) };
    Some(proto::Action { action_raw: Some(proto::ActionRaw { action: Some(proto::action_raw::Action::UnitCommand(uc)) }),
                         ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(tag: u64, alliance: proto::Alliance, x: f32, y: f32, unit_type: u32) -> proto::Unit {
        proto::Unit { tag: Some(tag), alliance: Some(alliance as i32), pos: Some(proto::Point { x: Some(x), y: Some(y), z: Some(0.0) }),
                      unit_type: Some(unit_type), display_type: Some(1), health: Some(40.0), health_max: Some(45.0),
                      ..Default::default() }
    }

    fn map() -> MapInfo {
        let height = (0..MAP * MAP).map(|i| if i % MAP < 32 { 10 } else { 200 }).collect();
        MapInfo::new(128, 96, vec![1; MAP * MAP], vec![0; MAP * MAP], height, vec![1, 2])
    }

    fn obs(units: Vec<proto::Unit>) -> proto::Observation {
        proto::Observation { game_loop: Some(8), raw_data: Some(proto::ObservationRaw { units, ..Default::default() }),
                             ..Default::default() }
    }

    fn voc() -> Vocab {
        let ab = |id: u32, name: &str, remap: u32, target: i32| proto::AbilityData {
            ability_id: Some(id), link_name: Some(name.into()), remaps_to_ability_id: Some(remap), target: Some(target),
            available: Some(true), ..Default::default()
        };
        let d = proto::ResponseData {
            abilities: vec![ab(16, "Move", 0, 4), ab(23, "attack", 3674, 4), ab(3674, "Attack", 0, 4), ab(4, "Stop", 0, 1)],
            units: vec![proto::UnitTypeData { unit_id: Some(48), name: Some("Marine".into()), ..Default::default() },
                        proto::UnitTypeData { unit_id: Some(341), name: Some("MineralField".into()), ..Default::default() }],
            ..Default::default()
        };
        Vocab::from_data(&d, 55958)
    }

    #[test]
    fn slots_are_stable_while_units_live() {
        let v = voc();
        let mut e = Encoder::new(map(), 1);
        use proto::Alliance::*;
        let f0 = e.encode(&obs(vec![unit(5, Self_, 10.0, 10.0, 48), unit(9, Enemy, 90.0, 60.0, 48), unit(7, Self_, 11.0, 10.0, 48)]), &v);
        let (s5, s7, s9) = (e.slot(5).unwrap(), e.slot(7).unwrap(), e.slot(9).unwrap());
        assert!(s5 < OWN && s7 < OWN && (OWN..OWN + ENEMY).contains(&s9));
        assert!(f0.unit_own[s5] && !f0.unit_own[s9] && f0.unit_mask[s9]);
        // 5 dies, 8 appears: 7 and 9 keep their slots, 8 takes 5's (the lowest free)
        let f1 = e.encode(&obs(vec![unit(9, Enemy, 91.0, 60.0, 48), unit(7, Self_, 12.0, 10.0, 48), unit(8, Self_, 13.0, 10.0, 48)]), &v);
        assert_eq!((e.slot(7), e.slot(9), e.slot(8), e.slot(5)), (Some(s7), Some(s9), Some(s5), None));
        assert_eq!(f1.tags[s5], 8);
        assert!((f1.unit_xy[s9 * 2] as f32 / 65535.0 - 91.0 / 128.0).abs() < 1e-4);
        assert_eq!(f1.units[s7 * F + feature::HEALTH], q(40.0 / 45.0));
        assert_eq!((f1.units[s9 * F + feature::ALLIANCE], f1.units[s7 * F + feature::ALLIANCE]), (3, 0));
        assert_eq!(f1.globals[global::RACE], 1.0);
        assert_eq!(f1.globals[global::ENEMY_RACE + 1], 1.0);
        // the height levels: the right half is high ground
        let hb = |k: usize, c: usize| f1.map_bits[(plane::HEIGHT + k) * MAP * MAP + c];
        assert_eq!((hb(0, 10), hb(0, 40), hb(2, 40)), (0, 1, 1));
    }

    #[test]
    fn neutral_units_nearest_first_and_resources_on_the_plane() {
        let v = voc();
        let mut e = Encoder::new(map(), 1);
        use proto::Alliance::*;
        let mut us = vec![unit(1, Self_, 10.0, 10.0, 48)];
        for i in 0..(NEUTRAL as u64 + 5) {
            let mut m = unit(100 + i, Neutral, 120.0 - i as f32, 90.0, 341);
            m.mineral_contents = Some(1500);
            us.push(m);
        }
        let f = e.encode(&obs(us), &v);
        // the 5 farthest (tags 100..104, x 120..116) have no slot, the rest do
        assert!((100..105).all(|t| e.slot(t).is_none()));
        assert!((105..105 + NEUTRAL as u64).all(|t| e.slot(t).is_some()));
        assert_eq!(e.over_cap, 5);
        assert_eq!(f.map_bits[plane::RESOURCE * MAP * MAP + e.map.cell(120.0, 90.0)], 1, "every mineral field on the plane");
    }

    #[test]
    fn labels_round_trip_through_decode() {
        let v = voc();
        let mut e = Encoder::new(map(), 1);
        use proto::Alliance::*;
        let f = e.encode(&obs(vec![unit(5, Self_, 10.0, 10.0, 48), unit(7, Self_, 11.0, 10.0, 48), unit(9, Enemy, 90.0, 60.0, 48)]), &v);
        let uc = proto::ActionRawUnitCommand { ability_id: Some(23), unit_tags: vec![5, 7, 999],
            target: Some(proto::action_raw_unit_command::Target::TargetUnitTag(9)), queue_command: Some(true) };
        let l = label(&uc, &f, &e, &v).unwrap();
        assert_eq!(l.action_type, v.action_id(3674), "attack → the general Attack");
        assert_eq!((l.target_kind, l.target_unit, l.queued), (2, e.slot(9).unwrap() as i64, true));
        assert_eq!(l.actors.iter().filter(|&&a| a).count(), 2, "the unknown tag 999 is no actor");
        let a = decode(&l, &f, &e, &v).unwrap();
        let Some(proto::action_raw::Action::UnitCommand(back)) = a.action_raw.unwrap().action else { panic!() };
        assert_eq!(back.ability_id, Some(3674));
        let mut tags = back.unit_tags.clone();
        tags.sort();
        assert_eq!(tags, vec![5, 7]);
        assert_eq!(back.target, Some(proto::action_raw_unit_command::Target::TargetUnitTag(9)));
        // a point target: LOC cells over the map, decoded at the cell centre
        let uc = proto::ActionRawUnitCommand { ability_id: Some(16), unit_tags: vec![5],
            target: Some(proto::action_raw_unit_command::Target::TargetWorldSpacePos(proto::Point2D { x: Some(64.2), y: Some(30.1) })),
            queue_command: None };
        let l = label(&uc, &f, &e, &v).unwrap();
        let (x, y) = e.map.loc_point(l.target_point as usize);
        assert!((x - 64.2).abs() <= 0.5 * 128.0 / LOC as f32 && (y - 30.1).abs() <= 0.5 * 96.0 / LOC as f32);
        // an enemy as actor, an unknown ability: no labels
        let uc = proto::ActionRawUnitCommand { ability_id: Some(16), unit_tags: vec![9], ..Default::default() };
        assert_eq!(label(&uc, &f, &e, &v), Err(Skip::NoActor));
        let uc = proto::ActionRawUnitCommand { ability_id: Some(77777), unit_tags: vec![5], ..Default::default() };
        assert_eq!(label(&uc, &f, &e, &v), Err(Skip::UnknownAbility));
        // the window context
        let w = Window::default().after(&l);
        assert_eq!((w.n, w.last_action, w.commanded[e.slot(5).unwrap()]), (1, l.action_type, true));
    }
}
