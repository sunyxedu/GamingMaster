//! StarCraft II frames for the tier-3 network, shared by the replay converter (gmdata `sc2
//! convert`) and the simulator (gmsim / env/vec_sc2.py), so an RL frame is the IL frame of the
//! same game state: the raw interface (units, map, player) of one player, encoded every
//! [`STEP_MUL`] game loops, and its raw unit commands as actions.
//!
//! Frame (per player):
//!
//!   units       [U, F] f32   the unit tokens: own units first (≤ [`OWN`]), then visible /
//!                            snapshot enemies (≤ [`ENEMY`]), then neutral ones (resources,
//!                            rocks; ≤ [`NEUTRAL`]); unused quota goes to the next group; in
//!                            each group by (unit type, tag).  Features: [`feature`] table
//!   unit_type   [U] i64      vocabulary id (0 empty slot, 1 unknown type)
//!   unit_order  [U] i64      the first order's (general) ability id, 0 none
//!   unit_buffs  [U, 2] i64   buff ids, 0 none
//!   unit_mask   [U] bool
//!   map         [C, 64, 64] u8  visible, explored, creep, pathable, placeable, height,
//!                            own units, enemy units — the map grid sampled at 64 × 64 (world
//!                            (x, y) → cell (⌊64 x / w⌋, ⌊64 y / h⌋); the game's image rows
//!                            run from y = 0), 0..255
//!   globals     [G] f32      minerals, gas, supply, army / workers, idle workers, larva, warp
//!                            gates, time, own race, enemy race, map size ([`global`] table)
//!   upgrades    [NU] f32     researched upgrades (bits over the vocabulary's upgrades)
//!
//! Action (one per frame: the player's first unit command in the frame's [`STEP_MUL`] loops):
//!
//!   action_type   general ability id of the command (the game's `remaps_to_ability_id`,
//!                 e.g. every "attack" is Attack), 0 = no command
//!   actors [U]    the command's own units among the unit tokens
//!   target_kind   0 none / 1 point / 2 unit
//!   target_unit   the target unit's token, −1 none
//!   target_point  the target cell (y · 64 + x), −1 none
//!   queued        the command is queued (shift)

pub mod vocab;

use sc2api::proto;
use std::collections::HashMap;
pub use vocab::Vocab;

/// Game loops per frame (22.4 loops a second at "faster": ~2.8 decisions a second).
pub const STEP_MUL: u32 = 8;
/// Unit token slots and the per-group quotas.
pub const U: usize = 256;
pub const OWN: usize = 176;
pub const ENEMY: usize = 56;
pub const NEUTRAL: usize = 24;
pub const F: usize = 43;
pub const BUFFS: usize = 2;
pub const MAP: usize = 64;
pub const C: usize = 8;
pub const G: usize = 20;

/// Unit feature columns.
pub mod feature {
    pub const ALLIANCE: usize = 0; // 4: self, ally, neutral, enemy
    pub const DISPLAY: usize = 4; // 4: visible, snapshot, hidden, placeholder
    pub const HEALTH: usize = 8;
    pub const HEALTH_MAX: usize = 9;
    pub const SHIELD: usize = 10;
    pub const SHIELD_MAX: usize = 11;
    pub const ENERGY: usize = 12;
    pub const ENERGY_MAX: usize = 13;
    pub const X: usize = 14;
    pub const Y: usize = 15;
    pub const FACING: usize = 16; // 2: sin, cos
    pub const RADIUS: usize = 18;
    pub const BUILD: usize = 19;
    pub const FLYING: usize = 20;
    pub const BURROWED: usize = 21;
    pub const HALLUCINATION: usize = 22;
    pub const CLOAK: usize = 23; // 4: cloaked, cloaked detected, not cloaked, cloaked allied
    pub const MINERALS: usize = 27;
    pub const VESPENE: usize = 28;
    pub const HARVEST: usize = 29; // assigned / ideal harvesters
    pub const IDEAL: usize = 30;
    pub const COOLDOWN: usize = 31;
    pub const ORDERS: usize = 32;
    pub const PROGRESS: usize = 33;
    pub const CARGO: usize = 34;
    pub const CARGO_MAX: usize = 35;
    pub const POWERED: usize = 36;
    pub const ACTIVE: usize = 37;
    pub const UPGRADES: usize = 38; // 3: attack, armour, shield level
    pub const ADD_ON: usize = 41;
    pub const BUFF_TIME: usize = 42;
}

/// Global feature columns.
pub mod global {
    pub const MINERALS: usize = 0;
    pub const VESPENE: usize = 1;
    pub const FOOD_USED: usize = 2;
    pub const FOOD_CAP: usize = 3;
    pub const FOOD_ARMY: usize = 4;
    pub const FOOD_WORKERS: usize = 5;
    pub const IDLE_WORKERS: usize = 6;
    pub const ARMY_COUNT: usize = 7;
    pub const WARP_GATES: usize = 8;
    pub const LARVA: usize = 9;
    pub const TIME: usize = 10;
    pub const RACE: usize = 11; // 3: terran, zerg, protoss
    pub const ENEMY_RACE: usize = 14; // 4: terran, zerg, protoss, unknown / random
    pub const MAP_W: usize = 18;
    pub const MAP_H: usize = 19;
}

/// Static facts of a game (from ResponseGameInfo) the frames need.
#[derive(Clone, Debug)]
pub struct MapInfo {
    pub w: usize,
    pub h: usize,
    /// the static planes at 64 × 64: pathable, placeable, height (0..255)
    pub pathing: Vec<u8>,
    pub placement: Vec<u8>,
    pub height: Vec<u8>,
    /// race of each player id (1-based index − 1): 1 terran, 2 zerg, 3 protoss, 4 random
    pub races: Vec<i32>,
}

fn bit(data: &[u8], i: usize) -> u8 {
    (data.get(i / 8).copied().unwrap_or(0) >> (7 - (i % 8))) & 1
}

/// A game image (1 bit or 1 byte per cell) sampled at the 64 × 64 cell centres.
fn sample(img: Option<&proto::ImageData>, w: usize, h: usize, scale_bits: u8) -> Vec<u8> {
    let mut out = vec![0u8; MAP * MAP];
    let Some(img) = img else { return out };
    let bpp = img.bits_per_pixel.unwrap_or(8);
    let (iw, ih) = img.size.as_ref().map(|s| (s.x.unwrap_or(w as i32) as usize, s.y.unwrap_or(h as i32) as usize)).unwrap_or((w, h));
    let data = img.data.as_deref().unwrap_or(&[]);
    for r in 0..MAP {
        let y = ((r as f64 + 0.5) * ih as f64 / MAP as f64) as usize;
        for c in 0..MAP {
            let x = ((c as f64 + 0.5) * iw as f64 / MAP as f64) as usize;
            let i = y.min(ih.saturating_sub(1)) * iw + x.min(iw.saturating_sub(1));
            out[r * MAP + c] = if bpp == 1 { bit(data, i) * scale_bits } else { data.get(i).copied().unwrap_or(0) };
        }
    }
    out
}

impl MapInfo {
    pub fn from_game_info(gi: &proto::ResponseGameInfo) -> MapInfo {
        let sr = gi.start_raw.as_ref();
        let (w, h) = sr.and_then(|s| s.map_size.as_ref()).map(|m| (m.x.unwrap_or(1).max(1) as usize, m.y.unwrap_or(1).max(1) as usize)).unwrap_or((1, 1));
        let races = gi.player_info.iter().map(|p| p.race_actual.unwrap_or(p.race_requested.unwrap_or(0))).collect();
        MapInfo {
            w,
            h,
            pathing: sample(sr.and_then(|s| s.pathing_grid.as_ref()), w, h, 255),
            placement: sample(sr.and_then(|s| s.placement_grid.as_ref()), w, h, 255),
            height: sample(sr.and_then(|s| s.terrain_height.as_ref()), w, h, 255),
            races,
        }
    }

    /// The 64 × 64 cell of a world point.
    pub fn cell(&self, x: f32, y: f32) -> usize {
        let c = ((x as f64 * MAP as f64 / self.w as f64).floor() as i64).clamp(0, MAP as i64 - 1) as usize;
        let r = ((y as f64 * MAP as f64 / self.h as f64).floor() as i64).clamp(0, MAP as i64 - 1) as usize;
        r * MAP + c
    }

    /// The world point at a cell's centre.
    pub fn point(&self, cell: usize) -> (f32, f32) {
        let (r, c) = (cell / MAP, cell % MAP);
        (((c as f64 + 0.5) * self.w as f64 / MAP as f64) as f32, ((r as f64 + 0.5) * self.h as f64 / MAP as f64) as f32)
    }
}

/// One frame (module docs) and the unit tag of every slot.
#[derive(Clone, Debug)]
pub struct Frame {
    pub units: Vec<f32>,
    pub unit_type: Vec<i64>,
    pub unit_order: Vec<i64>,
    pub unit_buffs: Vec<i64>,
    pub unit_mask: Vec<bool>,
    pub map: Vec<u8>,
    pub globals: Vec<f32>,
    pub upgrades: Vec<f32>,
    pub tags: Vec<u64>,
    /// slot of each own unit's tag / every unit's tag
    pub slot_of: HashMap<u64, usize>,
    pub own_slots: usize,
}

fn alliance_idx(a: i32) -> usize {
    match proto::Alliance::try_from(a) {
        Ok(proto::Alliance::Self_) => 0,
        Ok(proto::Alliance::Ally) => 1,
        Ok(proto::Alliance::Neutral) => 2,
        _ => 3,
    }
}

/// Encode `player`'s observation (its raw data) — the frame of module docs.
pub fn encode(obs: &proto::Observation, map: &MapInfo, player_id: u32, voc: &Vocab) -> Frame {
    let raw = obs.raw_data.as_ref();
    let units: Vec<&proto::Unit> = raw.map(|r| r.units.iter().collect()).unwrap_or_default();
    let group = |u: &&proto::Unit| match alliance_idx(u.alliance.unwrap_or(0)) {
        0 | 1 => 0,
        3 => 1,
        _ => 2,
    };
    let mut groups: [Vec<&proto::Unit>; 3] = Default::default();
    for u in &units {
        groups[group(u)].push(u);
    }
    for g in groups.iter_mut() {
        g.sort_by_key(|u| (u.unit_type.unwrap_or(0), u.tag.unwrap_or(0)));
    }
    // quotas, the unused remainder handed on in order
    let mut take = [0usize; 3];
    let quota = [OWN, ENEMY, NEUTRAL];
    let mut left = U;
    for i in 0..3 {
        take[i] = groups[i].len().min(quota[i]).min(left);
        left -= take[i];
    }
    for i in 0..3 {
        let more = groups[i].len().saturating_sub(take[i]).min(left);
        take[i] += more;
        left -= more;
    }
    let chosen: Vec<&proto::Unit> = (0..3).flat_map(|i| groups[i].iter().take(take[i]).copied()).collect();
    let mut f = Frame {
        units: vec![0.0; U * F],
        unit_type: vec![0; U],
        unit_order: vec![0; U],
        unit_buffs: vec![0; U * BUFFS],
        unit_mask: vec![false; U],
        map: vec![0; C * MAP * MAP],
        globals: vec![0.0; G],
        upgrades: vec![0.0; voc.upgrades.len()],
        tags: vec![0; U],
        slot_of: HashMap::new(),
        own_slots: take[0],
    };
    let ratio = |a: Option<f32>, b: Option<f32>| match (a, b) {
        (Some(a), Some(b)) if b > 0.0 => (a / b).clamp(0.0, 1.0),
        _ => 0.0,
    };
    for (s, u) in chosen.iter().enumerate() {
        use feature::*;
        let row = &mut f.units[s * F..(s + 1) * F];
        row[ALLIANCE + alliance_idx(u.alliance.unwrap_or(0))] = 1.0;
        let disp = (u.display_type.unwrap_or(1) - 1).clamp(0, 3) as usize;
        row[DISPLAY + disp] = 1.0;
        row[HEALTH] = ratio(u.health, u.health_max);
        row[HEALTH_MAX] = (u.health_max.unwrap_or(0.0) / 500.0).min(4.0);
        row[SHIELD] = ratio(u.shield, u.shield_max);
        row[SHIELD_MAX] = (u.shield_max.unwrap_or(0.0) / 500.0).min(4.0);
        row[ENERGY] = ratio(u.energy, u.energy_max);
        row[ENERGY_MAX] = (u.energy_max.unwrap_or(0.0) / 200.0).min(2.0);
        let (x, y) = u.pos.as_ref().map(|p| (p.x.unwrap_or(0.0), p.y.unwrap_or(0.0))).unwrap_or((0.0, 0.0));
        row[X] = x / map.w as f32;
        row[Y] = y / map.h as f32;
        let fa = u.facing.unwrap_or(0.0);
        row[FACING] = fa.sin();
        row[FACING + 1] = fa.cos();
        row[RADIUS] = u.radius.unwrap_or(0.0) / 5.0;
        row[BUILD] = u.build_progress.unwrap_or(0.0);
        row[FLYING] = u.is_flying.unwrap_or(false) as u8 as f32;
        row[BURROWED] = u.is_burrowed.unwrap_or(false) as u8 as f32;
        row[HALLUCINATION] = u.is_hallucination.unwrap_or(false) as u8 as f32;
        let cloak = (u.cloak.unwrap_or(3) - 1).clamp(0, 3) as usize;
        row[CLOAK + cloak] = 1.0;
        row[MINERALS] = u.mineral_contents.unwrap_or(0) as f32 / 1800.0;
        row[VESPENE] = u.vespene_contents.unwrap_or(0) as f32 / 2250.0;
        let (a, i) = (u.assigned_harvesters.unwrap_or(0) as f32, u.ideal_harvesters.unwrap_or(0) as f32);
        row[HARVEST] = if i > 0.0 { (a / i).min(2.0) } else { 0.0 };
        row[IDEAL] = i / 16.0;
        row[COOLDOWN] = (u.weapon_cooldown.unwrap_or(0.0) / 32.0).min(1.0);
        row[ORDERS] = (u.orders.len() as f32 / 4.0).min(1.0);
        row[PROGRESS] = u.orders.first().and_then(|o| o.progress).unwrap_or(0.0);
        row[CARGO] = u.cargo_space_taken.unwrap_or(0) as f32 / 8.0;
        row[CARGO_MAX] = u.cargo_space_max.unwrap_or(0) as f32 / 8.0;
        row[POWERED] = u.is_powered.unwrap_or(false) as u8 as f32;
        row[ACTIVE] = u.is_active.unwrap_or(false) as u8 as f32;
        row[UPGRADES] = u.attack_upgrade_level.unwrap_or(0) as f32 / 3.0;
        row[UPGRADES + 1] = u.armor_upgrade_level.unwrap_or(0) as f32 / 3.0;
        row[UPGRADES + 2] = u.shield_upgrade_level.unwrap_or(0) as f32 / 3.0;
        row[ADD_ON] = (u.add_on_tag.unwrap_or(0) != 0) as u8 as f32;
        row[BUFF_TIME] = ratio(u.buff_duration_remain.map(|x| x as f32), u.buff_duration_max.map(|x| x as f32));
        f.unit_type[s] = voc.unit_id(u.unit_type.unwrap_or(0));
        f.unit_order[s] = u.orders.first().map(|o| voc.action_id(o.ability_id.unwrap_or(0))).unwrap_or(0);
        for (k, b) in u.buff_ids.iter().take(BUFFS).enumerate() {
            f.unit_buffs[s * BUFFS + k] = voc.buff_id(*b);
        }
        f.unit_mask[s] = true;
        let tag = u.tag.unwrap_or(0);
        f.tags[s] = tag;
        f.slot_of.insert(tag, s);
    }
    // map planes
    let plane = |p: usize| -> std::ops::Range<usize> { p * MAP * MAP..(p + 1) * MAP * MAP };
    if let Some(ms) = raw.and_then(|r| r.map_state.as_ref()) {
        let vis = sample(ms.visibility.as_ref(), map.w, map.h, 1);
        let r0 = plane(0);
        let r1 = plane(1);
        for (i, &v) in vis.iter().enumerate() {
            f.map[r0.start + i] = if v == 2 { 255 } else { 0 };
            f.map[r1.start + i] = if v >= 1 { 255 } else { 0 };
        }
        let creep = sample(ms.creep.as_ref(), map.w, map.h, 255);
        let r2 = plane(2);
        f.map[r2].copy_from_slice(&creep);
    }
    f.map[plane(3)].copy_from_slice(&map.pathing);
    f.map[plane(4)].copy_from_slice(&map.placement);
    f.map[plane(5)].copy_from_slice(&map.height);
    for u in &units {
        let a = alliance_idx(u.alliance.unwrap_or(0));
        let p = match a {
            0 | 1 => 6,
            3 => 7,
            _ => continue,
        };
        let (x, y) = u.pos.as_ref().map(|p| (p.x.unwrap_or(0.0), p.y.unwrap_or(0.0))).unwrap_or((0.0, 0.0));
        let i = p * MAP * MAP + map.cell(x, y);
        f.map[i] = f.map[i].saturating_add(32);
    }
    // globals
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
    f.globals[TIME] = (obs.game_loop.unwrap_or(0) as f32 / (22.4 * 60.0 * 30.0)).min(2.0);
    let me = player_id.saturating_sub(1) as usize;
    let race_of = |i: usize| map.races.get(i).copied().unwrap_or(0);
    let r = race_of(me);
    if (1..=3).contains(&r) {
        f.globals[RACE + (r - 1) as usize] = 1.0;
    }
    let e = race_of(1 - me.min(1));
    f.globals[ENEMY_RACE + if (1..=3).contains(&e) { (e - 1) as usize } else { 3 }] = 1.0;
    f.globals[MAP_W] = map.w as f32 / 256.0;
    f.globals[MAP_H] = map.h as f32 / 256.0;
    if let Some(p) = raw.and_then(|r| r.player.as_ref()) {
        for up in &p.upgrade_ids {
            if let Some(i) = voc.upgrade_index(*up) {
                f.upgrades[i] = 1.0;
            }
        }
    }
    f
}

/// The action of a frame (module docs).
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
    pub fn noop() -> Label {
        Label { action_type: 0, actors: vec![false; U], target_kind: 0, target_unit: -1, target_point: -1, queued: false }
    }
}

/// The label of `frame` from the actions reported after it (the player's first raw unit
/// command whose actors are among the frame's own unit tokens; camera moves and autocast
/// toggles are no decisions here).
pub fn label(actions: &[proto::Action], frame: &Frame, map: &MapInfo, voc: &Vocab) -> Label {
    for a in actions {
        let Some(proto::action_raw::Action::UnitCommand(uc)) = a.action_raw.as_ref().and_then(|r| r.action.as_ref()) else { continue };
        let mut actors = vec![false; U];
        let mut any = false;
        for t in &uc.unit_tags {
            if let Some(&s) = frame.slot_of.get(t) {
                if s < frame.own_slots {
                    actors[s] = true;
                    any = true;
                }
            }
        }
        if !any {
            continue;
        }
        let action_type = voc.action_id(uc.ability_id.unwrap_or(0).max(0) as u32);
        let (target_kind, target_unit, target_point) = match &uc.target {
            Some(proto::action_raw_unit_command::Target::TargetWorldSpacePos(p)) => {
                (1, -1, map.cell(p.x.unwrap_or(0.0), p.y.unwrap_or(0.0)) as i64)
            }
            Some(proto::action_raw_unit_command::Target::TargetUnitTag(t)) => match frame.slot_of.get(t) {
                Some(&s) => (2, s as i64, -1),
                None => (0, -1, -1), // a target outside the tokens (unseen / over the cap)
            },
            None => (0, -1, -1),
        };
        return Label { action_type, actors, target_kind, target_unit, target_point, queued: uc.queue_command.unwrap_or(false) };
    }
    Label::noop()
}

/// The raw command of a network action on `frame` (None = no command).
pub fn decode(l: &Label, frame: &Frame, map: &MapInfo, voc: &Vocab) -> Option<proto::Action> {
    if l.action_type <= 1 {
        return None;
    }
    let ability = voc.ability_of(l.action_type)?;
    let unit_tags: Vec<u64> = (0..frame.own_slots.min(U)).filter(|&s| l.actors.get(s).copied().unwrap_or(false)).map(|s| frame.tags[s]).collect();
    if unit_tags.is_empty() {
        return None;
    }
    let target = match l.target_kind {
        1 if l.target_point >= 0 => {
            let (x, y) = map.point(l.target_point as usize);
            Some(proto::action_raw_unit_command::Target::TargetWorldSpacePos(proto::Point2D { x: Some(x), y: Some(y) }))
        }
        2 if l.target_unit >= 0 && (l.target_unit as usize) < U && frame.unit_mask[l.target_unit as usize] => {
            Some(proto::action_raw_unit_command::Target::TargetUnitTag(frame.tags[l.target_unit as usize]))
        }
        _ => None,
    };
    let uc = proto::ActionRawUnitCommand { ability_id: Some(ability as i32), target, unit_tags, queue_command: Some(l.queued) };
    Some(proto::Action { action_raw: Some(proto::ActionRaw { action: Some(proto::action_raw::Action::UnitCommand(uc)) }), ..Default::default() })
}
