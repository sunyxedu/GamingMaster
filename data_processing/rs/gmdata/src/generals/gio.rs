//! Simulator output → the `gio` training record (what `SeqReplayDataset`
//! reads). Frame `t` pairs observation `t` with the move decided at `t+1`.
//! Numerics reproduce the Python converter bit-for-bit where it used float32
//! (the army channel) and float64 elsewhere.

use super::sim::{SimOutput, MAX_GRID};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Gio {
    pub grid_seq: Vec<Vec<Vec<Vec<f64>>>>, // [T, 12, 25, 25]
    pub globals_seq: Vec<[f64; 4]>,
    pub legal_source_mask_seq: Vec<Vec<Vec<Vec<bool>>>>, // [T, 1, 25, 25]
    pub max_actions_per_tick: usize,
    pub count: Vec<i64>,
    pub src_y: Vec<Vec<i64>>,
    pub src_x: Vec<Vec<i64>>,
    pub dir_id: Vec<Vec<i64>>,
    pub ratio_id: Vec<Vec<i64>>,
    pub value: Vec<f64>,
    pub tick_seq: Vec<i64>,
    pub tick_bucket_seq: Vec<i64>,
    pub enemy_heatmap: Vec<Vec<Vec<Vec<f64>>>>,
    pub next_visibility: Vec<Vec<Vec<Vec<f64>>>>,
}

pub fn direction(start: [i64; 2], end: [i64; 2]) -> i64 {
    if start[0] == -1 || end[0] == -1 {
        return 0;
    }
    let dr = end[0] - start[0];
    let dc = end[1] - start[1];
    if dr > 0 {
        0
    } else if dc > 0 {
        1
    } else if dr < 0 {
        2
    } else {
        3
    }
}

pub fn convert(sim: &SimOutput, k: usize, estimate_value: bool) -> Gio {
    let t_raw = sim.first.len();
    let t_total = t_raw.saturating_sub(1);
    let (h, w) = (MAX_GRID, MAX_GRID);
    let mut g = Gio {
        grid_seq: Vec::with_capacity(t_total),
        globals_seq: Vec::with_capacity(t_total),
        legal_source_mask_seq: Vec::with_capacity(t_total),
        max_actions_per_tick: k,
        count: vec![],
        src_y: vec![],
        src_x: vec![],
        dir_id: vec![],
        ratio_id: vec![],
        value: vec![],
        tick_seq: vec![],
        tick_bucket_seq: vec![],
        enemy_heatmap: vec![],
        next_visibility: vec![],
    };
    let zero_plane = || vec![vec![vec![0.0f64; w]; h]; 1];
    for t in 0..t_total {
        let state = &sim.map[t];
        let armies = &sim.armies[t];
        let mut grid = vec![vec![vec![0.0f64; w]; h]; 12];
        for i in 0..h {
            for j in 0..w {
                let v = state[i][j];
                if (0..=10).contains(&v) {
                    grid[v as usize][i][j] = 1.0;
                }
                let a = armies[i][j];
                if a > 0 {
                    // np.log1p(int)/10.0 in float64, stored into a float32 array, widened on output
                    let f32v = (((a as f64).ln_1p()) / 10.0) as f32;
                    grid[11][i][j] = f32v as f64;
                }
            }
        }
        g.grid_seq.push(grid);
        let my_army = sim.info_my_army[t];
        let my_land = sim.info_my_land[t];
        let op_army = sim.info_op_army[t];
        let op_land = sim.info_op_land[t];
        g.globals_seq.push([
            (my_army as f64).ln_1p() / 10.0,
            (my_land as f64).ln_1p() / 10.0,
            (op_army as f64).ln_1p() / 10.0,
            (op_land as f64).ln_1p() / 10.0,
        ]);
        let mut mask = vec![vec![vec![false; w]; h]; 1];
        for i in 0..h {
            for j in 0..w {
                if (5..=7).contains(&state[i][j]) && armies[i][j] > 1 {
                    mask[0][i][j] = true;
                }
            }
        }
        g.legal_source_mask_seq.push(mask);

        let ms = sim.move_start[t + 1];
        let me = sim.move_end[t + 1];
        let is50 = sim.move_is_50[t + 1];
        let dropped = sim.move_dropped.get(t + 1).copied().unwrap_or(false);
        if ms[0] >= 0 {
            g.count.push(1);
            let mut sy = vec![0i64; k];
            let mut sx = vec![0i64; k];
            let mut d = vec![0i64; k];
            let mut r = vec![0i64; k];
            sy[0] = ms[0];
            sx[0] = ms[1];
            d[0] = direction(ms, me);
            r[0] = if is50 != 0 { 1 } else { 0 };
            g.src_y.push(sy);
            g.src_x.push(sx);
            g.dir_id.push(d);
            g.ratio_id.push(r);
        } else {
            g.count.push(if dropped { -1 } else { 0 });
            g.src_y.push(vec![0; k]);
            g.src_x.push(vec![0; k]);
            g.dir_id.push(vec![0; k]);
            g.ratio_id.push(vec![0; k]);
        }

        if estimate_value {
            let heuristic = if op_land == 0 {
                1.0
            } else if my_land == 0 {
                -1.0
            } else {
                let my_c = sim.info_my_cities.get(t).copied().unwrap_or(0);
                let op_c = sim.info_op_cities.get(t).copied().unwrap_or(0);
                let diff = my_c - op_c;
                let adj_my = my_army + diff.max(0) * 50;
                let adj_op = op_army + (-diff).max(0) * 50;
                let ratio = adj_my as f64 / (adj_op.max(1)) as f64;
                (ratio - 1.0).tanh()
            };
            let val = match &sim.winner {
                Some(win) => {
                    let outcome = if *win == sim.player { 1.0 } else { -1.0 };
                    let denom = (t_total as f64 - 1.0).max(1.0);
                    let wgt = (t as f64 / denom).powi(2);
                    wgt * outcome + (1.0 - wgt) * heuristic
                }
                None => heuristic,
            };
            g.value.push(val);
        } else {
            g.value.push(0.0);
        }
        g.tick_seq.push(t as i64);
        g.tick_bucket_seq.push((((t as i64) % 50) + 1) / 2);
        g.enemy_heatmap.push(zero_plane());
        g.next_visibility.push(zero_plane());
    }
    g
}
