//! [`VecSim`]: N episodes of one game stepped together — the RL environment behind
//! gmsim / `env/vec_candidate.py`. Pure logic (no I/O, no Python) so it is tested here.
//!
//! Rows. Per-seat layout (hidden-information / simultaneous games): `n_seats` rows per
//! env, row `env * n_seats + seat` = that seat's own stream; a row is ACTIVE (has a
//! frame, needs an action) only when its seat is to act, and its `step_idx` counts its
//! own frames. Interleaved layout (perfect-information alternating games): one row per
//! env holding every decision frame from the mover's perspective. Frames come from
//! [`encode_frame`] — the converters' encoder — so RL observations are IL
//! observations by construction.
//!
//! Chance events are drawn from the env's own [`Rng`] (seeded per episode), so an
//! episode is a pure function of its seed and the actions. Each env logs its
//! trajectory (the converters' record format).

use crate::frame::{encode_frame, FrameMut};
use crate::game::{FrameSpec, Game, Layout, Seat, Turn};
use crate::rng::Rng;
use crate::trajectory::Event;

#[derive(Clone, Debug, Default)]
pub struct SimOptions {
    /// Restart an env as soon as its episode ends (its rows come back with
    /// `new_episode`); the lockstep RL trainer keeps this off.
    pub autoreset: bool,
    /// Truncate episodes after this many decision turns (a draw: zero reward);
    /// None = the game's `max_decisions`.
    pub max_turns: Option<usize>,
}

/// Per-row batch buffers (row-major, `R` = rows).
#[derive(Clone, Debug, Default)]
pub struct RowBuffers {
    pub entities: Vec<f32>,
    pub entity_mask: Vec<u8>,
    pub entity_type: Vec<i32>,
    pub globals: Vec<f32>,
    pub actions: Vec<f32>,
    pub action_mask: Vec<u8>,
    pub n_legal: Vec<i64>,
    /// Seat of the row's current frame (per-seat rows: always the row's seat).
    pub to_play: Vec<i64>,
    /// Frame index of the current frame in the row's stream (frames seen so far).
    pub step_idx: Vec<i64>,
    /// The row has a frame and needs an action.
    pub active: Vec<u8>,
    /// The row's env started a new episode in the last reset / step (every row of
    /// that env, acting or not: seats that act later start their stream at step_idx 0).
    pub new_episode: Vec<u8>,
    /// Active with a single legal candidate (applied whatever is sent).
    pub forced: Vec<u8>,
}

/// Result of one [`VecSim::step`] (per env unless noted; `S` = seats).
#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    /// [N * S] reward earned on this step (the outcome at the end, else 0).
    pub rewards: Vec<f32>,
    /// [N * S] raw returns of episodes that ended on this step (0 otherwise).
    pub returns: Vec<f64>,
    pub done: Vec<u8>,
    pub terminated: Vec<u8>,
    pub truncated: Vec<u8>,
    /// Seat with the strictly highest outcome of an episode that ended, else −1.
    pub winner: Vec<i64>,
    /// Decision turns of an episode that ended on this step (0 otherwise).
    pub episode_len: Vec<i64>,
    /// [R] candidate index applied on each row (−1 = the row did not act).
    pub applied: Vec<i64>,
}

struct Env<G: Game> {
    state: G,
    rng: Rng,
    done: bool,
    turns: usize,
    /// frames of each seat in this episode (per-seat step_idx)
    frames: Vec<i64>,
    events: Vec<Event>,
    fresh: bool,
}

pub struct VecSim<G: Game> {
    cfg: G::Config,
    spec: FrameSpec,
    layout: Layout,
    opts: SimOptions,
    envs: Vec<Env<G>>,
    buf: RowBuffers,
    legal: Vec<Vec<u32>>,
    outcomes: Vec<(u32, f64)>,
}

impl<G: Game> VecSim<G> {
    /// `n_envs` envs, all finished until [`reset`](Self::reset).
    pub fn new(cfg: G::Config, n_envs: usize, opts: SimOptions) -> Result<Self, String> {
        let spec = G::spec(&cfg);
        let layout = spec.layout();
        if layout == Layout::Interleaved && spec.simultaneous {
            return Err("interleaved layout cannot hold simultaneous turns".into());
        }
        let mut sim = VecSim { cfg, spec, layout, opts, envs: Vec::new(), buf: RowBuffers::default(), legal: Vec::new(), outcomes: Vec::new() };
        sim.resize(n_envs);
        Ok(sim)
    }

    pub fn spec(&self) -> &FrameSpec {
        &self.spec
    }

    pub fn config(&self) -> &G::Config {
        &self.cfg
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn rows_per_env(&self) -> usize {
        match self.layout {
            Layout::PerSeat => self.spec.n_seats,
            Layout::Interleaved => 1,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.envs.len() * self.rows_per_env()
    }

    pub fn buffers(&self) -> &RowBuffers {
        &self.buf
    }

    /// Candidate ids of a row's current frame.
    pub fn legal(&self, row: usize) -> &[u32] {
        &self.legal[row]
    }

    pub fn done(&self, env: usize) -> bool {
        self.envs[env].done
    }

    pub fn trajectory(&self, env: usize) -> &[Event] {
        &self.envs[env].events
    }

    pub fn state(&self, env: usize) -> &G {
        &self.envs[env].state
    }

    /// Resize to `n` envs (all finished until reset).
    pub fn resize(&mut self, n: usize) {
        let s = &self.spec;
        self.envs = (0..n)
            .map(|_| Env { state: G::new_game(&self.cfg), rng: Rng::new(0), done: true, turns: 0, frames: vec![0; s.n_seats], events: Vec::new(), fresh: false })
            .collect();
        let r = n * self.rows_per_env();
        let s = &self.spec;
        self.buf = RowBuffers {
            entities: vec![0.0; r * s.entities_len()],
            entity_mask: vec![0; r * s.max_entities],
            entity_type: vec![0; r * s.max_entities],
            globals: vec![0.0; r * s.global_features],
            actions: vec![0.0; r * s.actions_len()],
            action_mask: vec![0; r * s.max_actions],
            n_legal: vec![0; r],
            to_play: vec![0; r],
            step_idx: vec![0; r],
            active: vec![0; r],
            new_episode: vec![0; r],
            forced: vec![0; r],
        };
        self.legal = vec![Vec::new(); r];
    }

    fn row(&self, env: usize, seat: Seat) -> usize {
        match self.layout {
            Layout::PerSeat => env * self.spec.n_seats + seat,
            Layout::Interleaved => env,
        }
    }

    fn max_turns(&self) -> usize {
        self.opts.max_turns.unwrap_or(usize::MAX).min(self.spec.max_decisions)
    }

    /// Start new episodes in `env_ids` with the given per-episode seeds.
    pub fn reset(&mut self, env_ids: &[usize], seeds: &[u64]) -> Result<(), String> {
        if env_ids.len() != seeds.len() {
            return Err(format!("reset: {} env ids but {} seeds", env_ids.len(), seeds.len()));
        }
        for e in &mut self.envs {
            e.fresh = false;
        }
        for (&e, &seed) in env_ids.iter().zip(seeds) {
            if e >= self.envs.len() {
                return Err(format!("reset: env {e} out of range ({} envs)", self.envs.len()));
            }
            self.start(e, seed)?;
        }
        self.refresh_new_episode();
        for &e in env_ids {
            self.refresh(e)?;
        }
        Ok(())
    }

    fn start(&mut self, e: usize, seed: u64) -> Result<(), String> {
        let n_seats = self.spec.n_seats;
        let env = &mut self.envs[e];
        env.state = G::new_game(&self.cfg);
        env.rng = Rng::new(seed);
        env.done = false;
        env.turns = 0;
        env.frames = vec![0; n_seats];
        env.events.clear();
        env.fresh = true;
        self.resolve_chance(e)
    }

    fn resolve_chance(&mut self, e: usize) -> Result<(), String> {
        let env = &mut self.envs[e];
        while env.state.turn() == Turn::Chance {
            self.outcomes.clear();
            env.state.chance_outcomes(&mut self.outcomes);
            if self.outcomes.is_empty() {
                return Err(format!("{}: chance node without outcomes", G::NAME));
            }
            let w: Vec<f64> = self.outcomes.iter().map(|&(_, p)| p).collect();
            let o = self.outcomes[env.rng.weighted(&w)].0;
            env.state.apply_chance(o).map_err(|err| err.0)?;
            env.events.push(Event::Chance(o));
        }
        if env.state.turn() == Turn::Terminal {
            env.done = true; // decided by chance alone (e.g. a misdeal rule): no frame
        }
        Ok(())
    }

    fn refresh_new_episode(&mut self) {
        let rpe = self.rows_per_env();
        for (e, env) in self.envs.iter().enumerate() {
            self.buf.new_episode[e * rpe..(e + 1) * rpe].fill(env.fresh as u8);
        }
    }

    /// Re-encode env `e`'s rows from its state.
    fn refresh(&mut self, e: usize) -> Result<(), String> {
        let rpe = self.rows_per_env();
        let s = &self.spec;
        let (el, ne, gl, al, na) = (s.entities_len(), s.max_entities, s.global_features, s.actions_len(), s.max_actions);
        for r in e * rpe..(e + 1) * rpe {
            row_mut(&mut self.buf, &mut self.legal, r, (el, ne, gl, al, na)).clear();
            self.buf.active[r] = 0;
            self.buf.forced[r] = 0;
            self.buf.n_legal[r] = 0;
            let seat = if self.layout == Layout::PerSeat { r - e * rpe } else { 0 };
            self.buf.to_play[r] = seat as i64;
            self.buf.step_idx[r] = match self.layout {
                Layout::PerSeat => self.envs[e].frames[seat],
                Layout::Interleaved => self.envs[e].turns as i64,
            };
        }
        if self.envs[e].done {
            return Ok(());
        }
        let Turn::Act(seats) = self.envs[e].state.turn() else {
            return Err(format!("{}: env {e} neither done nor at a decision", G::NAME));
        };
        if self.layout == Layout::Interleaved && seats.len() != 1 {
            return Err(format!("{}: {} seats act at once in an interleaved game", G::NAME, seats.len()));
        }
        for seat in seats.iter() {
            let r = self.row(e, seat);
            let mut fm = row_mut(&mut self.buf, &mut self.legal, r, (el, ne, gl, al, na));
            encode_frame(&self.envs[e].state, seat, &self.spec, &mut fm).map_err(|err| err.0)?;
            let n = self.legal[r].len();
            self.buf.active[r] = 1;
            self.buf.n_legal[r] = n as i64;
            self.buf.forced[r] = (n == 1) as u8;
            self.buf.to_play[r] = seat as i64;
        }
        Ok(())
    }

    /// Apply one candidate index per row (`[R]`; rows that do not act are ignored,
    /// forced rows take their single candidate whatever is sent). Errors on an index
    /// that is not a legal candidate of an acting row, before anything is applied.
    pub fn step(&mut self, action_idx: &[i64]) -> Result<StepInfo, String> {
        let r_total = self.n_rows();
        if action_idx.len() != r_total {
            return Err(format!("step: {} actions for {r_total} rows", action_idx.len()));
        }
        let n = self.envs.len();
        let ns = self.spec.n_seats;
        for (r, &i) in action_idx.iter().enumerate() {
            let checked = self.buf.active[r] == 1 && self.buf.forced[r] == 0;
            if checked && (i < 0 || i >= self.buf.n_legal[r]) {
                let e = r / self.rows_per_env();
                return Err(format!(
                    "{} env {e} row {r} (seat {}): action_idx {i} is not a legal candidate (n_legal {}) - sample from legal['action']",
                    G::NAME, self.buf.to_play[r], self.buf.n_legal[r]
                ));
            }
        }
        let mut info = StepInfo {
            rewards: vec![0.0; n * ns],
            returns: vec![0.0; n * ns],
            done: vec![0; n],
            terminated: vec![0; n],
            truncated: vec![0; n],
            winner: vec![-1; n],
            episode_len: vec![0; n],
            applied: vec![-1; r_total],
        };
        let live: Vec<usize> = (0..n).filter(|&e| !self.envs[e].done).collect();
        for e in &mut self.envs {
            e.fresh = false;
        }
        let max_turns = self.max_turns();
        for &e in &live {
            let Turn::Act(seats) = self.envs[e].state.turn() else {
                return Err(format!("{}: live env {e} not at a decision", G::NAME));
            };
            for seat in seats.iter() {
                let r = self.row(e, seat);
                let idx = if self.buf.forced[r] == 1 { 0 } else { action_idx[r] as usize };
                let a = self.legal[r][idx];
                let env = &mut self.envs[e];
                env.state.apply_action(seat, a).map_err(|err| err.0)?;
                env.events.push(Event::Act(seat, a));
                env.frames[seat] += 1;
                info.applied[r] = idx as i64;
            }
            self.envs[e].turns += 1;
            self.resolve_chance(e)?;
            let env = &mut self.envs[e];
            if env.state.turn() == Turn::Terminal {
                env.done = true;
                let out = env.state.outcome();
                let ret = env.state.returns();
                info.rewards[e * ns..(e + 1) * ns].copy_from_slice(&out);
                info.returns[e * ns..(e + 1) * ns].copy_from_slice(&ret);
                info.done[e] = 1;
                info.terminated[e] = 1;
                info.winner[e] = winner(&out);
                info.episode_len[e] = env.turns as i64;
            } else if env.turns >= max_turns {
                env.done = true;
                info.done[e] = 1;
                info.truncated[e] = 1;
                info.episode_len[e] = env.turns as i64;
            }
        }
        if self.opts.autoreset {
            for &e in &live {
                if self.envs[e].done {
                    let seed = self.envs[e].rng.next_u64();
                    self.start(e, seed)?;
                }
            }
        }
        self.refresh_new_episode();
        for &e in &live {
            self.refresh(e)?;
        }
        Ok(info)
    }
}

/// Row `r` of the batch buffers as a frame (dims: entities, max_entities, globals,
/// actions, max_actions lengths).
fn row_mut<'a>(b: &'a mut RowBuffers, legal: &'a mut [Vec<u32>], r: usize,
               (el, ne, gl, al, na): (usize, usize, usize, usize, usize)) -> FrameMut<'a> {
    FrameMut {
        entities: &mut b.entities[r * el..(r + 1) * el],
        entity_mask: &mut b.entity_mask[r * ne..(r + 1) * ne],
        entity_type: &mut b.entity_type[r * ne..(r + 1) * ne],
        globals: &mut b.globals[r * gl..(r + 1) * gl],
        actions: &mut b.actions[r * al..(r + 1) * al],
        action_mask: &mut b.action_mask[r * na..(r + 1) * na],
        legal: &mut legal[r],
    }
}

/// Seat with the strictly highest outcome, −1 on a tie for the top.
pub fn winner(outcome: &[f32]) -> i64 {
    let mut best = 0usize;
    for s in 1..outcome.len() {
        if outcome[s] > outcome[best] {
            best = s;
        }
    }
    if outcome.iter().enumerate().any(|(s, &v)| s != best && v == outcome[best]) {
        -1
    } else {
        best as i64
    }
}
