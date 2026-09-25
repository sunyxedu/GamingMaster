//! generals.io replay simulator — one player's perspective with fog of war.
//! Semantics follow the reference Python simulator (move validation, chase /
//! defensive / larger-army ordering, city+general growth every 2 turns, land
//! growth every 50, surrender resolution through `afks`).

use super::gior::{Move, Replay};
use anyhow::{bail, Result};
use serde::Serialize;

pub const MAX_GRID: usize = 25;

#[derive(Clone, Copy)]
struct Tile {
    terrain: u8, // 0 blank, 1 city, 2 general, 3 mountain
    army: i64,
    occupied: i64, // -1 or player index
}

struct PlayerState {
    name: String,
    land: Vec<usize>,
    army: i64,
    vision: Vec<u8>, // 0 unseen, 1 owned, 2 visible
}

#[derive(Debug, Clone, Serialize)]
pub struct SimOutput {
    pub id: String,
    pub player: String,
    pub winner: Option<String>,
    pub first: Vec<i64>,
    pub map: Vec<Vec<Vec<i64>>>,
    pub armies: Vec<Vec<Vec<i64>>>,
    #[serde(rename = "info_myArmy")]
    pub info_my_army: Vec<i64>,
    #[serde(rename = "info_myLand")]
    pub info_my_land: Vec<i64>,
    #[serde(rename = "info_opArmy")]
    pub info_op_army: Vec<i64>,
    #[serde(rename = "info_opLand")]
    pub info_op_land: Vec<i64>,
    #[serde(rename = "info_myCities")]
    pub info_my_cities: Vec<i64>,
    #[serde(rename = "info_opCities")]
    pub info_op_cities: Vec<i64>,
    pub move_start: Vec<[i64; 2]>,
    pub move_end: Vec<[i64; 2]>,
    pub move_is_50: Vec<i64>,
    pub move_dropped: Vec<bool>,
}

pub struct Simulator<'a> {
    r: &'a Replay,
    w: usize,
    h: usize,
    player_id: usize,
    map: Vec<Tile>,
    players: Vec<PlayerState>,
}

impl<'a> Simulator<'a> {
    /// `strict`: reject replays the simulator cannot reproduce faithfully
    /// (oversized maps, modern features, modifiers) — the training pipeline
    /// must keep it on.
    pub fn new(r: &'a Replay, player_name: &str, strict: bool) -> Result<Self> {
        let w = r.map_width as usize;
        let h = r.map_height as usize;
        if strict {
            if w > MAX_GRID || h > MAX_GRID {
                bail!("map {w}x{h} exceeds the {MAX_GRID}x{MAX_GRID} model grid");
            }
            for (name, present) in [
                ("swamps", !r.swamps.is_empty()),
                ("deserts", !r.deserts.is_empty()),
                ("lookouts", !r.lookouts.is_empty()),
                ("observatories", !r.observatories.is_empty()),
                ("lights", !r.lights.is_empty()),
                ("neutrals", !r.neutrals.is_empty()),
            ] {
                if present {
                    bail!("unsupported map feature: {name}");
                }
            }
            if !r.modifiers.is_empty() {
                bail!("unsupported modifiers: {:?}", r.modifiers);
            }
        }
        let player_id = r
            .usernames
            .iter()
            .position(|u| u == player_name)
            .ok_or_else(|| anyhow::anyhow!("Player {player_name} not found in replay"))?;
        Ok(Self { r, w, h, player_id, map: Vec::new(), players: Vec::new() })
    }

    fn init_map(&mut self) {
        self.map = vec![Tile { terrain: 0, army: 0, occupied: -1 }; self.w * self.h];
        self.players = self
            .r
            .usernames
            .iter()
            .map(|n| PlayerState { name: n.clone(), land: Vec::new(), army: 0, vision: vec![0; self.w * self.h] })
            .collect();
        for (i, &g) in self.r.generals.iter().enumerate() {
            let g = g as usize;
            self.map[g] = Tile { terrain: 2, army: 1, occupied: i as i64 };
            self.players[i].land.push(g);
        }
        for (i, &c) in self.r.cities.iter().enumerate() {
            self.map[c as usize] = Tile { terrain: 1, army: self.r.city_armies[i], occupied: -1 };
        }
        for &m in &self.r.mountains {
            self.map[m as usize] = Tile { terrain: 3, army: 0, occupied: -1 };
        }
    }

    fn execute(&mut self, mv: &Move) -> bool {
        let s = mv.start as usize;
        let e = mv.end as usize;
        let p = mv.index;
        if s >= self.map.len() || e >= self.map.len() {
            return false;
        }
        let st = self.map[s];
        let et = self.map[e];
        if st.occupied != p || st.army <= 1 {
            return false;
        }
        if et.terrain == 3 {
            return false;
        }
        let (to_move, remaining) = if mv.is_half() {
            let m = st.army / 2;
            (m, st.army - m)
        } else {
            (st.army - 1, 1)
        };
        if et.occupied == p {
            self.map[e].army += to_move;
            self.map[s].army = remaining;
        } else {
            if to_move > et.army {
                if et.occupied != -1 {
                    let prev = &mut self.players[et.occupied as usize].land;
                    if let Some(pos) = prev.iter().position(|&x| x == e) {
                        prev.remove(pos);
                    }
                }
                self.players[p as usize].land.push(e);
                self.map[e].occupied = p;
                self.map[e].army = to_move - et.army;
            } else {
                self.map[e].army -= to_move;
            }
            self.map[s].army = remaining;
        }
        true
    }

    fn priority(&self, mv: &Move, other: Option<&Move>, original: usize) -> (i64, i64, i64, i64, usize) {
        let chase = match other {
            Some(o) if mv.end == o.start && mv.start != o.end => 0,
            _ => 1,
        };
        let defensive = if (mv.end as usize) < self.map.len() && self.map[mv.end as usize].occupied == mv.index { 0 } else { 1 };
        let attacking_general = self.r.generals.contains(&mv.end);
        let army = self.map.get(mv.start as usize).map(|t| t.army).unwrap_or(0);
        (chase, defensive, attacking_general as i64, -army, original)
    }

    fn process(&mut self, m1: Option<&Move>, m2: Option<&Move>) -> Vec<(Move, bool)> {
        let moves: Vec<(&Move, usize)> = [m1, m2].iter().enumerate().filter_map(|(i, m)| m.map(|m| (m, i))).collect();
        match moves.len() {
            0 => Vec::new(),
            1 => {
                let ok = self.execute(moves[0].0);
                vec![(moves[0].0.clone(), ok)]
            }
            _ => {
                let (a, ia) = moves[0];
                let (b, ib) = moves[1];
                let pa = self.priority(a, Some(b), ia);
                let pb = self.priority(b, Some(a), ib);
                if pa <= pb {
                    let oa = self.execute(a);
                    let ob = self.execute(b);
                    vec![(a.clone(), oa), (b.clone(), ob)]
                } else {
                    let ob = self.execute(b);
                    let oa = self.execute(a);
                    vec![(a.clone(), oa), (b.clone(), ob)]
                }
            }
        }
    }

    fn update_vision(&mut self) {
        let (w, h) = (self.w as i64, self.h as i64);
        for p in &mut self.players {
            p.vision.iter_mut().for_each(|v| *v = 0);
            p.army = 0;
            for &idx in &p.land {
                let (r, c) = ((idx / self.w) as i64, (idx % self.w) as i64);
                p.vision[idx] = 1;
                p.army += self.map[idx].army;
                for (dr, dc) in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr >= 0 && nr < h && nc >= 0 && nc < w {
                        let n = (nr * w + nc) as usize;
                        if p.vision[n] != 1 {
                            p.vision[n] = 2;
                        }
                    }
                }
            }
        }
    }

    fn encode_state(&self) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let p = &self.players[self.player_id];
        let pid = self.player_id as i64;
        let mut state = vec![vec![-1i64; self.w]; self.h];
        let mut army = vec![vec![-1i64; self.w]; self.h];
        for i in 0..self.h {
            for j in 0..self.w {
                let idx = i * self.w + j;
                let t = self.map[idx];
                match p.vision[idx] {
                    0 => state[i][j] = if t.terrain == 0 || t.terrain == 2 { 0 } else { 1 },
                    2 => {
                        state[i][j] = match (t.terrain, t.occupied) {
                            (0, -1) => 2,
                            (1, -1) => 3,
                            (3, _) => 4,
                            (0, o) if o != pid => 8,
                            (1, o) if o != pid => 9,
                            (2, o) if o != pid => 10,
                            _ => -1,
                        };
                        if t.terrain <= 2 {
                            army[i][j] = t.army;
                        }
                    }
                    _ => {
                        state[i][j] = match t.terrain {
                            0 => 5,
                            1 => 6,
                            2 => 7,
                            _ => -1,
                        };
                        army[i][j] = t.army;
                    }
                }
            }
        }
        (state, army)
    }

    fn increment(&mut self, turn: i64) {
        if turn == 0 {
            return;
        }
        if turn % 2 == 0 {
            for &c in &self.r.cities {
                if self.map[c as usize].occupied != -1 {
                    self.map[c as usize].army += 1;
                }
            }
            for &g in &self.r.generals {
                self.map[g as usize].army += 1;
            }
        }
        if turn % 50 == 0 {
            let lands: Vec<Vec<usize>> = self.players.iter().map(|p| p.land.clone()).collect();
            for land in lands {
                for idx in land {
                    self.map[idx].army += 1;
                }
            }
        }
    }

    fn winner(&self) -> Option<usize> {
        for (i, &g) in self.r.generals.iter().enumerate() {
            let t = self.map[g as usize];
            if t.occupied != -1 && t.occupied != i as i64 {
                return Some(t.occupied as usize);
            }
        }
        None
    }

    pub fn simulate(mut self) -> SimOutput {
        self.init_map();
        let mut out = SimOutput {
            id: self.r.id.clone(),
            player: self.players.get(self.player_id).map(|p| p.name.clone()).unwrap_or_default(),
            winner: None,
            first: vec![],
            map: vec![],
            armies: vec![],
            info_my_army: vec![],
            info_my_land: vec![],
            info_op_army: vec![],
            info_op_land: vec![],
            info_my_cities: vec![],
            info_op_cities: vec![],
            move_start: vec![],
            move_end: vec![],
            move_is_50: vec![],
            move_dropped: vec![],
        };
        let moves = &self.r.moves;
        let max_turn = moves.last().map(|m| m.turn).unwrap_or(0);
        let mut mi = 0usize;
        let n_players = self.players.len();
        let opponent = if n_players == 2 { Some(1 - self.player_id) } else { None };
        let w = self.w as i64;

        for turn in 0..(max_turn + 2) {
            let mut turn_moves: [Option<Move>; 2] = [None, None];
            let mut cnt = 0;
            while mi < moves.len() && moves[mi].turn + 1 == turn {
                if cnt < 2 {
                    turn_moves[cnt] = Some(moves[mi].clone());
                }
                cnt += 1;
                mi += 1;
            }
            let executed = self.process(turn_moves[0].as_ref(), turn_moves[1].as_ref());
            let (mut ps, mut pe, mut p50) = (-1i64, -1i64, -1i64);
            let mut dropped = false;
            for (mv, ok) in &executed {
                if mv.index as usize == self.player_id {
                    if *ok {
                        ps = mv.start;
                        pe = mv.end;
                        p50 = mv.is_half() as i64;
                    } else {
                        dropped = true;
                    }
                }
            }
            dropped = dropped && ps == -1;
            self.increment(turn);
            self.update_vision();
            let (state, army) = self.encode_state();
            let mut pstate = vec![vec![-1i64; MAX_GRID]; MAX_GRID];
            let mut parmy = vec![vec![-1i64; MAX_GRID]; MAX_GRID];
            for i in 0..self.h.min(MAX_GRID) {
                for j in 0..self.w.min(MAX_GRID) {
                    pstate[i][j] = state[i][j];
                    parmy[i][j] = army[i][j];
                }
            }
            let me = &self.players[self.player_id];
            let (op_army, op_land) = match opponent {
                Some(o) => (self.players[o].army, self.players[o].land.len() as i64),
                None => (0, 0),
            };
            let mut my_cities = 0i64;
            let mut op_cities = 0i64;
            for &c in &self.r.cities {
                let occ = self.map[c as usize].occupied;
                if occ == self.player_id as i64 {
                    my_cities += 1;
                } else if let Some(o) = opponent {
                    if occ == o as i64 {
                        op_cities += 1;
                    }
                }
            }
            my_cities += 1;
            if let Some(o) = opponent {
                if self.map[self.r.generals[o] as usize].occupied == o as i64 {
                    op_cities += 1;
                }
            }
            let is_first = ((self.player_id == 0 && turn % 2 == 0) || (self.player_id == 1 && turn % 2 == 1)) as i64;
            out.first.push(is_first);
            out.map.push(pstate);
            out.armies.push(parmy);
            out.info_my_army.push(me.army);
            out.info_my_land.push(me.land.len() as i64);
            out.info_op_army.push(op_army);
            out.info_op_land.push(op_land);
            out.info_my_cities.push(my_cities);
            out.info_op_cities.push(op_cities);
            if ps != -1 {
                out.move_start.push([ps / w, ps % w]);
                out.move_end.push([pe / w, pe % w]);
            } else {
                out.move_start.push([-1, -1]);
                out.move_end.push([-1, -1]);
            }
            out.move_is_50.push(p50);
            out.move_dropped.push(dropped);
            if let Some(win) = self.winner() {
                out.winner = Some(self.players[win].name.clone());
                break;
            }
        }
        if out.winner.is_none() && n_players == 2 {
            if let Some(afk) = self.r.afks.first() {
                if afk.index == 0 || afk.index == 1 {
                    out.winner = Some(self.players[1 - afk.index as usize].name.clone());
                }
            }
        }
        out
    }
}
