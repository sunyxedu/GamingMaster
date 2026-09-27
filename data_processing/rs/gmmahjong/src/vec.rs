//! N hanchan games stepped together ([`MjGame`]): the batched simulator behind gmsim's
//! `MahjongSim` / env/vec_mahjong.py, with gmsim's per-seat row conventions — row
//! `env * 4 + seat` is that seat's own stream, active when the seat owes a decision (the
//! seats of a claim window act on the same step), `forced` when the decision has a single
//! legal action (applied whatever is sent).
//!
//! Buffers per row (zeros on inactive rows): `tiles_seq [34, 37]` f32 (the converter's tile
//! quarters / 4, as data_processing/games/mahjong.py expands them), `players_seq [4, 12]`,
//! `table_seq [12]`; legality `action_type [8]` and `tile_cond [8, 34]` (u8 0/1; types
//! without a tile argument: every tile, which the decoder never scores).

use crate::rules::NK;
use crate::sim::{MjGame, PLAYERS, TABLE, TILES_Q};
use crate::state::{ObsBuf, N_TYPES, USES_TILE};

pub const SEATS: usize = 4;

#[derive(Default)]
pub struct Buffers {
    pub tiles: Vec<f32>,
    pub players: Vec<f32>,
    pub table: Vec<f32>,
    pub legal_type: Vec<u8>,
    pub tile_cond: Vec<u8>,
    pub active: Vec<u8>,
    pub forced: Vec<u8>,
    pub to_play: Vec<i64>,
    pub step_idx: Vec<i64>,
    pub new_episode: Vec<u8>,
}

#[derive(Default)]
pub struct StepInfo {
    /// [N, 4] the placement value on the step a game ends
    pub rewards: Vec<f32>,
    /// [N, 4] final scores (points) of the ended games
    pub returns: Vec<f64>,
    pub done: Vec<u8>,
    pub terminated: Vec<u8>,
    pub truncated: Vec<u8>,
    /// the first-placed seat of an ended game, else −1
    pub winner: Vec<i64>,
    /// decision steps of an ended game, else 0
    pub episode_len: Vec<i64>,
    /// [R] the action applied per row (−1 where the row did not act)
    pub applied_type: Vec<i64>,
    pub applied_tile: Vec<i64>,
    pub acted: Vec<u8>,
}

struct Env {
    game: MjGame,
    seed: u64,
    steps: i64,
    frames: [i64; SEATS],
    fresh: bool,
    done: bool,
    error: Option<String>,
}

pub struct MjVec {
    envs: Vec<Env>,
    pub autoreset: bool,
    /// games that stopped on an engine / legality disagreement (a bug: 0 expected)
    pub failures: usize,
    pub last_error: Option<String>,
    threads: usize,
}

fn first_choice(p: &crate::sim::Pending) -> (i64, i64) {
    for t in 0..N_TYPES {
        if p.legal.types[t] {
            if !USES_TILE[t] {
                return (t as i64, 0);
            }
            if let Some(k) = (0..NK).find(|&k| p.legal.tiles[t][k]) {
                return (t as i64, k as i64);
            }
        }
    }
    (0, 0)
}

fn n_choices(p: &crate::sim::Pending) -> usize {
    (0..N_TYPES).filter(|&t| p.legal.types[t])
        .map(|t| if USES_TILE[t] { (0..NK).filter(|&k| p.legal.tiles[t][k]).count() } else { 1 }).sum()
}

impl MjVec {
    pub fn new(n: usize, seeds: &[u64], autoreset: bool, threads: usize) -> Result<Self, String> {
        let mut v = MjVec { envs: Vec::new(), autoreset, failures: 0, last_error: None, threads: threads.max(1) };
        for i in 0..n {
            v.envs.push(Self::fresh_env(seeds.get(i).copied().unwrap_or(i as u64))?);
        }
        Ok(v)
    }

    fn fresh_env(seed: u64) -> Result<Env, String> {
        Ok(Env { game: MjGame::new(seed)?, seed, steps: 0, frames: [0; SEATS], fresh: true, done: false, error: None })
    }

    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn n_rows(&self) -> usize {
        self.envs.len() * SEATS
    }

    pub fn resize(&mut self, n: usize) -> Result<(), String> {
        while self.envs.len() > n {
            self.envs.pop();
        }
        while self.envs.len() < n {
            let s = self.envs.len() as u64;
            self.envs.push(Self::fresh_env(s)?);
        }
        Ok(())
    }

    pub fn reset(&mut self, ids: &[usize], seeds: &[u64]) -> Result<(), String> {
        for (j, &i) in ids.iter().enumerate() {
            if i >= self.envs.len() {
                return Err(format!("env {i} out of range"));
            }
            self.envs[i] = Self::fresh_env(seeds.get(j).copied().unwrap_or(i as u64))?;
        }
        Ok(())
    }

    pub fn done(&self, env: usize) -> bool {
        self.envs[env].done
    }

    pub fn scores(&self, env: usize) -> [i64; 4] {
        self.envs[env].game.scores()
    }

    pub fn observe(&self) -> Buffers {
        let r = self.n_rows();
        let mut b = Buffers {
            tiles: vec![0.0; r * TILES_Q],
            players: vec![0.0; r * PLAYERS],
            table: vec![0.0; r * TABLE],
            legal_type: vec![0; r * N_TYPES],
            tile_cond: vec![0; r * N_TYPES * NK],
            active: vec![0; r],
            forced: vec![0; r],
            to_play: (0..r).map(|row| (row % SEATS) as i64).collect(),
            step_idx: vec![0; r],
            new_episode: vec![0; r],
        };
        for (e, env) in self.envs.iter().enumerate() {
            for s in 0..SEATS {
                let row = e * SEATS + s;
                b.step_idx[row] = env.frames[s];
                b.new_episode[row] = env.fresh as u8;
            }
            if env.done {
                continue;
            }
            for p in env.game.pending() {
                let row = e * SEATS + p.seat;
                let mut o = ObsBuf::default();
                env.game.encode(p, &mut o);
                for (d, q) in b.tiles[row * TILES_Q..(row + 1) * TILES_Q].iter_mut().zip(&o.tiles_q) {
                    *d = *q as f32 / 4.0;
                }
                b.players[row * PLAYERS..(row + 1) * PLAYERS].copy_from_slice(&o.players);
                b.table[row * TABLE..(row + 1) * TABLE].copy_from_slice(&o.table);
                for t in 0..N_TYPES {
                    b.legal_type[row * N_TYPES + t] = p.legal.types[t] as u8;
                    for k in 0..NK {
                        let v = if USES_TILE[t] { p.legal.types[t] && p.legal.tiles[t][k] } else { true };
                        b.tile_cond[(row * N_TYPES + t) * NK + k] = v as u8;
                    }
                }
                b.active[row] = 1;
                b.forced[row] = (n_choices(p) == 1) as u8;
            }
        }
        b
    }

    /// Apply every acting row's (type, tile) (arrays over all rows; a forced row gets its
    /// only legal action).
    pub fn step(&mut self, atype: &[i64], tile: &[i64]) -> Result<StepInfo, String> {
        let r = self.n_rows();
        if atype.len() != r || tile.len() != r {
            return Err(format!("{} / {} actions for {r} rows", atype.len(), tile.len()));
        }
        let n = self.envs.len();
        let mut info = StepInfo {
            rewards: vec![0.0; n * SEATS],
            returns: vec![0.0; n * SEATS],
            done: vec![0; n],
            terminated: vec![0; n],
            truncated: vec![0; n],
            winner: vec![-1; n],
            episode_len: vec![0; n],
            applied_type: vec![-1; r],
            applied_tile: vec![-1; r],
            acted: vec![0; r],
        };
        let step_env = |e: usize, env: &mut Env| -> Vec<(usize, i64, i64)> {
            if env.done {
                return vec![];
            }
            let mut applied = Vec::new();
            let choices: Vec<(i64, i64, Option<crate::sim::Hint>)> = env.game.pending().iter().map(|p| {
                let row = e * SEATS + p.seat;
                let (t, k) = if n_choices(p) == 1 { first_choice(p) } else { (atype[row], tile[row]) };
                applied.push((p.seat, t, k));
                (t, k, None)
            }).collect();
            for (s, _, _) in &applied {
                env.frames[*s] += 1;
            }
            env.fresh = false;
            env.steps += 1;
            if let Err(err) = env.game.act(&choices) {
                env.error = Some(err);
                env.done = true;
            } else if env.game.done() {
                env.done = true;
            }
            applied
        };
        let results: Vec<Vec<(usize, i64, i64)>> = if self.threads > 1 && n > 1 {
            let chunk = n.div_ceil(self.threads);
            std::thread::scope(|sc| {
                let handles: Vec<_> = self.envs.chunks_mut(chunk).enumerate().map(|(c, envs)| {
                    let step_env = &step_env;
                    sc.spawn(move || envs.iter_mut().enumerate().map(|(j, env)| step_env(c * chunk + j, env)).collect::<Vec<_>>())
                }).collect();
                handles.into_iter().flat_map(|h| h.join().expect("a simulator thread panicked")).collect()
            })
        } else {
            self.envs.iter_mut().enumerate().map(|(e, env)| step_env(e, env)).collect()
        };
        for (e, applied) in results.into_iter().enumerate() {
            for (s, t, k) in applied {
                let row = e * SEATS + s;
                info.applied_type[row] = t;
                info.applied_tile[row] = k;
                info.acted[row] = 1;
            }
            let env = &mut self.envs[e];
            if !env.done || applied_is_old(env) {
                continue;
            }
            info.done[e] = 1;
            info.episode_len[e] = env.steps;
            if let Some(err) = env.error.take() {
                self.failures += 1;
                self.last_error = Some(format!("env {e} (seed {}): {err}", env.seed));
                info.truncated[e] = 1;
            } else {
                info.terminated[e] = 1;
                let v = env.game.outcome();
                let sc = env.game.scores();
                for s in 0..SEATS {
                    info.rewards[e * SEATS + s] = v[s];
                    info.returns[e * SEATS + s] = sc[s] as f64;
                }
                info.winner[e] = (0..SEATS).find(|&s| v[s] == 1.0).map(|s| s as i64).unwrap_or(-1);
            }
            env.steps = -1; // reported
            if self.autoreset {
                let seed = env.seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
                *env = Self::fresh_env(seed)?;
            }
        }
        Ok(info)
    }
}

/// An ended game already reported by an earlier step (steps = −1).
fn applied_is_old(env: &Env) -> bool {
    env.steps < 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_games_step_to_the_end_and_autoreset() {
        for threads in [1, 3] {
            let mut v = MjVec::new(4, &[1, 2, 3, 4], false, threads).unwrap();
            let mut ended = 0;
            let mut steps = 0;
            while ended < 4 {
                let b = v.observe();
                assert!(b.active.iter().any(|&a| a == 1) || (0..4).all(|e| v.done(e)));
                // the first legal action of every acting row (hora first: games end)
                let r = v.n_rows();
                let (mut at, mut tl) = (vec![0i64; r], vec![0i64; r]);
                for row in 0..r {
                    if b.active[row] == 0 {
                        continue;
                    }
                    let legal = &b.legal_type[row * N_TYPES..(row + 1) * N_TYPES];
                    let t = if legal[5] == 1 { 5 } else { (0..N_TYPES).find(|&t| legal[t] == 1).unwrap() };
                    let k = (0..NK).find(|&k| b.tile_cond[(row * N_TYPES + t) * NK + k] == 1).unwrap();
                    at[row] = t as i64;
                    tl[row] = k as i64;
                }
                let info = v.step(&at, &tl).unwrap();
                for e in 0..4 {
                    if info.done[e] == 1 {
                        ended += 1;
                        assert_eq!(info.terminated[e], 1, "{:?}", v.last_error);
                        let mut rw: Vec<f32> = info.rewards[e * 4..e * 4 + 4].to_vec();
                        rw.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        assert_eq!(rw, crate::sim::PLACEMENT);
                        assert!(info.episode_len[e] > 20);
                    }
                }
                steps += 1;
                assert!(steps < 100_000);
            }
            assert_eq!(v.failures, 0);
            // an ended game reports once; its rows stay inactive
            let b = v.observe();
            assert!(b.active.iter().all(|&a| a == 0));
            let r = v.n_rows();
            let info = v.step(&vec![0; r], &vec![0; r]).unwrap();
            assert!(info.done.iter().all(|&d| d == 0));
        }
        // autoreset: a new game right after the end
        let mut v = MjVec::new(1, &[7], true, 1).unwrap();
        let mut ends = 0;
        for _ in 0..20_000 {
            let b = v.observe();
            let r = v.n_rows();
            let (mut at, mut tl) = (vec![0i64; r], vec![0i64; r]);
            for row in 0..r {
                if b.active[row] == 1 {
                    let legal = &b.legal_type[row * N_TYPES..(row + 1) * N_TYPES];
                    let t = if legal[5] == 1 { 5 } else { (0..N_TYPES).find(|&t| legal[t] == 1).unwrap() };
                    at[row] = t as i64;
                    tl[row] = (0..NK).find(|&k| b.tile_cond[(row * N_TYPES + t) * NK + k] == 1).unwrap() as i64;
                }
            }
            let info = v.step(&at, &tl).unwrap();
            ends += info.done[0] as usize;
            if ends == 2 {
                break;
            }
        }
        assert_eq!(ends, 2, "two games in a row");
        let b = v.observe();
        assert!(b.new_episode.iter().all(|&x| x == 1) && b.step_idx.iter().all(|&x| x == 0), "the autoreset game's first frame");
    }
}
