//! [`VecSim`]: N episodes of one game stepped together — the RL environment behind
//! gmsim / `env/vec_gmsim.py`. Pure logic (no I/O, no Python) so it is tested here.
//!
//! Rows. Per-seat layout (hidden-information / simultaneous games): `n_seats` rows per
//! env, row `env * n_seats + seat` = that seat's own stream; a row is ACTIVE (has a
//! frame, needs an action) only when its seat is to act, and its `step_idx` counts its
//! own frames. Interleaved layout (perfect-information alternating games): one row per
//! env holding every decision frame from the mover's perspective. Frames come from
//! [`encode_frame`] — the converters' encoder — so RL observations are IL
//! observations by construction.
//!
//! Actions: per row the game's action keys (`FrameSpec::action_keys`, e.g. the
//! candidate index), decoded by [`decode_action`] on the state the frame was encoded
//! on. Chance events are drawn from the env's own [`Rng`] (seeded per episode), so an
//! episode is a pure function of its seed and the actions — also when the envs are
//! stepped in parallel (`SimOptions::threads`). Each env logs its trajectory (the
//! converters' record format).

use crate::frame::{action_keys_of, decode_action, encode_frame, encode_target, new_target, FrameMut};
use crate::game::{FrameSpec, Game, Layout, Seat, Turn};
use crate::rng::Rng;
use crate::schema::{Buf, BufMut, TensorsMut};
use crate::trajectory::Event;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SimOptions {
    /// Restart an env as soon as its episode ends (its rows come back with
    /// `new_episode`); the lockstep RL trainer keeps this off.
    pub autoreset: bool,
    /// Truncate episodes after this many decision turns (a draw: zero reward);
    /// None = the game's `max_decisions`.
    pub max_turns: Option<usize>,
    /// Threads stepping the envs: 1 = the calling thread, 0 = rayon's global pool,
    /// n > 1 = a dedicated pool of n threads. Results never depend on it.
    pub threads: usize,
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions { autoreset: false, max_turns: None, threads: 1 }
    }
}

/// Per-row batch buffers (row-major; `R` = rows).
#[derive(Clone, Debug, Default)]
pub struct RowBuffers {
    /// `FrameSpec::obs` order, `R × numel` each.
    pub obs: Vec<Buf>,
    /// `FrameSpec::legal` order, `R × numel` each.
    pub legal: Vec<Buf>,
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
    /// Active with a single legal action (applied whatever is sent).
    pub forced: Vec<u8>,
}

/// Result of one [`VecSim::step`] (per env unless noted; `S` = seats, `K` = action keys).
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
    /// [R * K] action keys of the action applied on each row (−1 = the row did not act).
    pub applied: Vec<i64>,
    /// `[R]` the row acted on this step.
    pub acted: Vec<u8>,
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
    /// scratch target buffers (action keys of applied actions)
    target: Vec<Buf>,
}

/// The rows of one env: views into the batch buffers.
struct Rows<'a> {
    obs: Vec<BufMut<'a>>,
    legal: Vec<BufMut<'a>>,
    legal_ids: &'a mut [Vec<u32>],
    n_legal: &'a mut [i64],
    to_play: &'a mut [i64],
    step_idx: &'a mut [i64],
    active: &'a mut [u8],
    new_episode: &'a mut [u8],
    forced: &'a mut [u8],
}

impl Rows<'_> {
    /// Row `k` (within the env) as a frame.
    fn frame<'s>(&'s mut self, spec: &'s FrameSpec, k: usize) -> FrameMut<'s> {
        let view = |bufs: &'s mut Vec<BufMut<'_>>, specs: &'s [crate::schema::TensorSpec]| {
            let v: Vec<BufMut<'s>> = bufs.iter_mut().zip(specs).map(|(b, t)| b.range_mut(k * t.numel()..(k + 1) * t.numel())).collect();
            TensorsMut::new(specs, v)
        };
        FrameMut { obs: view(&mut self.obs, &spec.obs), legal: view(&mut self.legal, &spec.legal), legal_ids: &mut self.legal_ids[k] }
    }
}

/// The step outputs of one env: views into the StepInfo buffers.
struct Out<'a> {
    rewards: &'a mut [f32],
    returns: &'a mut [f64],
    done: &'a mut u8,
    terminated: &'a mut u8,
    truncated: &'a mut u8,
    winner: &'a mut i64,
    episode_len: &'a mut i64,
    applied: &'a mut [i64],
    acted: &'a mut [u8],
}

/// The fixed per-sim facts the per-env functions need.
struct Ctx<'a, G: Game> {
    cfg: &'a G::Config,
    spec: &'a FrameSpec,
    layout: Layout,
    rpe: usize,
    n_keys: usize,
    max_turns: usize,
    autoreset: bool,
}

impl<G: Game> Ctx<'_, G> {
    /// Row (within the env) of `seat`'s frames.
    fn row(&self, seat: Seat) -> usize {
        match self.layout {
            Layout::PerSeat => seat,
            Layout::Interleaved => 0,
        }
    }

    fn start(&self, env: &mut Env<G>, seed: u64) -> Result<(), String> {
        env.state = G::new_game(self.cfg);
        env.rng = Rng::new(seed);
        env.done = false;
        env.turns = 0;
        env.frames.iter_mut().for_each(|f| *f = 0);
        env.events.clear();
        env.fresh = true;
        resolve_chance(env)
    }

    /// Re-encode the env's rows from its state.
    fn refresh(&self, env: &Env<G>, rows: &mut Rows<'_>) -> Result<(), String> {
        rows.new_episode.fill(env.fresh as u8);
        for k in 0..self.rpe {
            rows.frame(self.spec, k).clear();
            rows.active[k] = 0;
            rows.forced[k] = 0;
            rows.n_legal[k] = 0;
            let seat = if self.layout == Layout::PerSeat { k } else { 0 };
            rows.to_play[k] = seat as i64;
            rows.step_idx[k] = match self.layout {
                Layout::PerSeat => env.frames[seat],
                Layout::Interleaved => env.turns as i64,
            };
        }
        if env.done {
            return Ok(());
        }
        let Turn::Act(seats) = env.state.turn() else {
            return Err(format!("{}: env neither done nor at a decision", G::NAME));
        };
        if self.layout == Layout::Interleaved && seats.len() != 1 {
            return Err(format!("{}: {} seats act at once in an interleaved game", G::NAME, seats.len()));
        }
        for seat in seats.iter() {
            let k = self.row(seat);
            encode_frame(&env.state, seat, self.spec, &mut rows.frame(self.spec, k)).map_err(|e| e.0)?;
            let n = rows.legal_ids[k].len();
            rows.active[k] = 1;
            rows.n_legal[k] = n as i64;
            rows.forced[k] = (n == 1) as u8;
            rows.to_play[k] = seat as i64;
        }
        Ok(())
    }

    /// Decode the actions of the env's acting rows (`keys` = its rows' action keys);
    /// `out[k]` = the canonical action of row k (u32::MAX = the row does not act).
    fn decode(&self, env: &Env<G>, rows: &Rows<'_>, keys: &[i64], out: &mut [u32]) -> Result<(), String> {
        out.fill(u32::MAX);
        if env.done {
            return Ok(());
        }
        let Turn::Act(seats) = env.state.turn() else {
            return Err(format!("{}: live env not at a decision", G::NAME));
        };
        for seat in seats.iter() {
            let k = self.row(seat);
            let legal = &rows.legal_ids[k];
            out[k] = if rows.forced[k] == 1 {
                legal[0]
            } else {
                decode_action(&env.state, seat, self.spec, legal, &keys[k * self.n_keys..(k + 1) * self.n_keys])
                    .map_err(|e| format!("row {k} (seat {seat}): {e} - sample from the legal masks"))?
            };
        }
        Ok(())
    }

    /// Apply the decoded actions of a live env, advance it to its next decision (or
    /// the end), autoreset, re-encode its rows.
    fn apply(&self, env: &mut Env<G>, rows: &mut Rows<'_>, out: &mut Out<'_>, actions: &[u32]) -> Result<(), String> {
        env.fresh = false;
        if env.done {
            rows.new_episode.fill(0);
            return Ok(());
        }
        let Turn::Act(seats) = env.state.turn() else {
            return Err(format!("{}: live env not at a decision", G::NAME));
        };
        // action keys of the applied actions, on the pre-turn state
        for seat in seats.iter() {
            let k = self.row(seat);
            let a = actions[k];
            let mut t = std::mem::take(&mut env.target);
            encode_target(&env.state, seat, self.spec, &rows.legal_ids[k], a, &mut TensorsMut::of(&self.spec.target, &mut t))
                .map_err(|e| e.0)?;
            out.applied[k * self.n_keys..(k + 1) * self.n_keys].copy_from_slice(&action_keys_of(self.spec, &t));
            env.target = t;
            out.acted[k] = 1;
        }
        for seat in seats.iter() {
            let a = actions[self.row(seat)];
            env.state.apply_action(seat, a).map_err(|e| e.0)?;
            env.events.push(Event::Act(seat, a));
            env.frames[seat] += 1;
        }
        env.turns += 1;
        resolve_chance(env)?;
        if env.state.turn() == Turn::Terminal {
            env.done = true;
            let o = env.state.outcome();
            let r = env.state.returns();
            if o.len() != out.rewards.len() || r.len() != out.returns.len() {
                return Err(format!("{}: outcome / returns must have n_seats entries", G::NAME));
            }
            out.rewards.copy_from_slice(&o);
            out.returns.copy_from_slice(&r);
            *out.done = 1;
            *out.terminated = 1;
            *out.winner = winner(&o);
            *out.episode_len = env.turns as i64;
        } else if env.turns >= self.max_turns {
            env.done = true;
            *out.done = 1;
            *out.truncated = 1;
            *out.episode_len = env.turns as i64;
        }
        if env.done && self.autoreset {
            let seed = env.rng.next_u64();
            self.start(env, seed)?;
        }
        self.refresh(env, rows)
    }
}

fn resolve_chance<G: Game>(env: &mut Env<G>) -> Result<(), String> {
    let mut outcomes = Vec::new();
    while env.state.turn() == Turn::Chance {
        outcomes.clear();
        env.state.chance_outcomes(&mut outcomes);
        if outcomes.is_empty() {
            return Err(format!("{}: chance node without outcomes", G::NAME));
        }
        let w: Vec<f64> = outcomes.iter().map(|&(_, p)| p).collect();
        let o = outcomes[env.rng.weighted(&w)].0;
        env.state.apply_chance(o).map_err(|err| err.0)?;
        env.events.push(Event::Chance(o));
    }
    if env.state.turn() == Turn::Terminal {
        env.done = true; // decided by chance alone (e.g. a misdeal rule): no frame
    }
    Ok(())
}

/// Per-env views of the row buffers.
fn split_rows<'a>(buf: &'a mut RowBuffers, legal_ids: &'a mut [Vec<u32>], spec: &FrameSpec, n: usize, rpe: usize) -> Vec<Rows<'a>> {
    let mut obs: Vec<std::vec::IntoIter<BufMut<'a>>> =
        buf.obs.iter_mut().zip(&spec.obs).map(|(b, t)| b.chunks_mut(rpe * t.numel()).into_iter()).collect();
    let mut legal: Vec<std::vec::IntoIter<BufMut<'a>>> =
        buf.legal.iter_mut().zip(&spec.legal).map(|(b, t)| b.chunks_mut(rpe * t.numel()).into_iter()).collect();
    let mut ids = legal_ids.chunks_mut(rpe);
    let mut n_legal = buf.n_legal.chunks_mut(rpe);
    let mut to_play = buf.to_play.chunks_mut(rpe);
    let mut step_idx = buf.step_idx.chunks_mut(rpe);
    let mut active = buf.active.chunks_mut(rpe);
    let mut new_episode = buf.new_episode.chunks_mut(rpe);
    let mut forced = buf.forced.chunks_mut(rpe);
    (0..n)
        .map(|_| Rows {
            obs: obs.iter_mut().map(|it| it.next().unwrap()).collect(),
            legal: legal.iter_mut().map(|it| it.next().unwrap()).collect(),
            legal_ids: ids.next().unwrap(),
            n_legal: n_legal.next().unwrap(),
            to_play: to_play.next().unwrap(),
            step_idx: step_idx.next().unwrap(),
            active: active.next().unwrap(),
            new_episode: new_episode.next().unwrap(),
            forced: forced.next().unwrap(),
        })
        .collect()
}

/// Per-env views of the step outputs.
fn split_out(info: &mut StepInfo, n_seats: usize, rpe: usize, n_keys: usize) -> Vec<Out<'_>> {
    let n = info.done.len();
    let mut rewards = info.rewards.chunks_mut(n_seats);
    let mut returns = info.returns.chunks_mut(n_seats);
    let mut done = info.done.iter_mut();
    let mut terminated = info.terminated.iter_mut();
    let mut truncated = info.truncated.iter_mut();
    let mut winner = info.winner.iter_mut();
    let mut episode_len = info.episode_len.iter_mut();
    let mut applied = info.applied.chunks_mut(rpe * n_keys);
    let mut acted = info.acted.chunks_mut(rpe);
    (0..n)
        .map(|_| Out {
            rewards: rewards.next().unwrap(),
            returns: returns.next().unwrap(),
            done: done.next().unwrap(),
            terminated: terminated.next().unwrap(),
            truncated: truncated.next().unwrap(),
            winner: winner.next().unwrap(),
            episode_len: episode_len.next().unwrap(),
            applied: applied.next().unwrap(),
            acted: acted.next().unwrap(),
        })
        .collect()
}

/// The first error of per-env results (by env index).
fn first_err(res: Vec<Result<(), String>>, what: &str) -> Result<(), String> {
    match res.into_iter().enumerate().find_map(|(e, r)| r.err().map(|m| (e, m))) {
        Some((e, m)) => Err(format!("{what} env {e}: {m}")),
        None => Ok(()),
    }
}

pub struct VecSim<G: Game> {
    cfg: G::Config,
    spec: FrameSpec,
    layout: Layout,
    opts: SimOptions,
    envs: Vec<Env<G>>,
    buf: RowBuffers,
    legal: Vec<Vec<u32>>,
    pool: Option<rayon::ThreadPool>,
}

impl<G: Game> VecSim<G> {
    /// `n_envs` envs, all finished until [`reset`](Self::reset).
    pub fn new(cfg: G::Config, n_envs: usize, opts: SimOptions) -> Result<Self, String> {
        G::check_config(&cfg).map_err(|e| format!("{} config: {e}", G::NAME))?;
        let spec = G::spec(&cfg);
        spec.validate()?;
        let layout = spec.layout();
        if layout == Layout::Interleaved && spec.simultaneous {
            return Err("interleaved layout cannot hold simultaneous turns".into());
        }
        let pool = if opts.threads > 1 {
            Some(rayon::ThreadPoolBuilder::new().num_threads(opts.threads).build().map_err(|e| e.to_string())?)
        } else {
            None
        };
        let mut sim = VecSim { cfg, spec, layout, opts, envs: Vec::new(), buf: RowBuffers::default(), legal: Vec::new(), pool };
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

    /// Legal action ids of a row's current frame.
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
            .map(|_| Env {
                state: G::new_game(&self.cfg),
                rng: Rng::new(0),
                done: true,
                turns: 0,
                frames: vec![0; s.n_seats],
                events: Vec::new(),
                fresh: false,
                target: new_target(s),
            })
            .collect();
        let r = n * self.rows_per_env();
        self.buf = RowBuffers {
            obs: Buf::rows(&s.obs, r),
            legal: Buf::rows(&s.legal, r),
            n_legal: vec![0; r],
            to_play: vec![0; r],
            step_idx: vec![0; r],
            active: vec![0; r],
            new_episode: vec![0; r],
            forced: vec![0; r],
        };
        self.legal = vec![Vec::new(); r];
    }

    fn max_turns(&self) -> usize {
        self.opts.max_turns.unwrap_or(usize::MAX).min(self.spec.max_decisions)
    }

    /// Run `f` on the sim's pool (or the calling thread / the global pool).
    fn run<R: Send>(pool: &Option<rayon::ThreadPool>, threads: usize, f: impl FnOnce(bool) -> R + Send) -> R {
        match (pool, threads) {
            (Some(p), _) => p.install(|| f(true)),
            (None, 0) => f(true),
            _ => f(false),
        }
    }

    /// Start new episodes in `env_ids` with the given per-episode seeds (every other
    /// env keeps its state; only the reset envs' rows are flagged `new_episode`).
    pub fn reset(&mut self, env_ids: &[usize], seeds: &[u64]) -> Result<(), String> {
        if env_ids.len() != seeds.len() {
            return Err(format!("reset: {} env ids but {} seeds", env_ids.len(), seeds.len()));
        }
        let n = self.envs.len();
        let mut seed_of: Vec<Option<u64>> = vec![None; n];
        for (&e, &seed) in env_ids.iter().zip(seeds) {
            if e >= n {
                return Err(format!("reset: env {e} out of range ({n} envs)"));
            }
            seed_of[e] = Some(seed);
        }
        let ctx = Ctx::<G> {
            cfg: &self.cfg,
            spec: &self.spec,
            layout: self.layout,
            rpe: self.rows_per_env(),
            n_keys: self.spec.action_keys.len(),
            max_turns: self.max_turns(),
            autoreset: self.opts.autoreset,
        };
        let rows = split_rows(&mut self.buf, &mut self.legal, &self.spec, n, ctx.rpe);
        let envs = &mut self.envs;
        let res: Vec<Result<(), String>> = Self::run(&self.pool, self.opts.threads, |par| {
            let one = |((env, mut rows), seed): ((&mut Env<G>, Rows<'_>), &Option<u64>)| -> Result<(), String> {
                match seed {
                    Some(s) => {
                        ctx.start(env, *s)?;
                        ctx.refresh(env, &mut rows)
                    }
                    None => {
                        env.fresh = false;
                        rows.new_episode.fill(0);
                        Ok(())
                    }
                }
            };
            if par {
                envs.par_iter_mut().zip(rows).zip(&seed_of).map(one).collect()
            } else {
                envs.iter_mut().zip(rows).zip(&seed_of).map(one).collect()
            }
        });
        first_err(res, "reset")
    }

    /// Apply the action keys of every row (`[R * K]`, row-major; rows that do not act
    /// are ignored, forced rows take their single legal action whatever is sent).
    /// Errors on keys that are no legal action of an acting row — before anything is
    /// applied.
    pub fn step(&mut self, keys: &[i64]) -> Result<StepInfo, String> {
        let n = self.envs.len();
        let rpe = self.rows_per_env();
        let k = self.spec.action_keys.len();
        let r_total = n * rpe;
        if keys.len() != r_total * k {
            return Err(format!("step: {} action values for {r_total} rows x {k} action keys {:?}", keys.len(), self.spec.action_keys));
        }
        let ns = self.spec.n_seats;
        let ctx = Ctx::<G> {
            cfg: &self.cfg,
            spec: &self.spec,
            layout: self.layout,
            rpe,
            n_keys: k,
            max_turns: self.max_turns(),
            autoreset: self.opts.autoreset,
        };
        let mut info = StepInfo {
            rewards: vec![0.0; n * ns],
            returns: vec![0.0; n * ns],
            done: vec![0; n],
            terminated: vec![0; n],
            truncated: vec![0; n],
            winner: vec![-1; n],
            episode_len: vec![0; n],
            applied: vec![-1; r_total * k],
            acted: vec![0; r_total],
        };
        let mut decoded = vec![u32::MAX; r_total];
        let mut rows = split_rows(&mut self.buf, &mut self.legal, &self.spec, n, rpe);
        // pass 1 (read-only): decode + validate every acting row
        {
            let envs = &self.envs;
            let rows_ref = &rows;
            let res: Vec<Result<(), String>> = Self::run(&self.pool, self.opts.threads, |par| {
                let one = |(e, out): (usize, &mut [u32])| ctx.decode(&envs[e], &rows_ref[e], &keys[e * rpe * k..(e + 1) * rpe * k], out);
                if par {
                    decoded.par_chunks_mut(rpe).enumerate().map(one).collect()
                } else {
                    decoded.chunks_mut(rpe).enumerate().map(one).collect()
                }
            });
            first_err(res, G::NAME)?;
        }
        // pass 2: apply, advance, re-encode
        let outs = split_out(&mut info, ns, rpe, k);
        let envs = &mut self.envs;
        let decoded = &decoded;
        let res: Vec<Result<(), String>> = Self::run(&self.pool, self.opts.threads, |par| {
            let one = |(e, ((env, rows), mut out)): (usize, ((&mut Env<G>, &mut Rows<'_>), Out<'_>))| {
                ctx.apply(env, rows, &mut out, &decoded[e * rpe..(e + 1) * rpe])
            };
            if par {
                envs.par_iter_mut().zip(rows.par_iter_mut()).zip(outs).enumerate().map(one).collect()
            } else {
                envs.iter_mut().zip(rows.iter_mut()).zip(outs).enumerate().map(one).collect()
            }
        });
        first_err(res, G::NAME)?;
        Ok(info)
    }

    /// Action keys (`[R * K]`) of a uniformly random legal action on every acting row
    /// (zeros elsewhere) — random play for any schema; a pure function of `seed` and
    /// the current state.
    pub fn random_actions(&self, seed: u64) -> Result<Vec<i64>, String> {
        let rpe = self.rows_per_env();
        let k = self.spec.action_keys.len();
        let mut out = vec![0i64; self.n_rows() * k];
        let mut t = new_target(&self.spec);
        for (e, env) in self.envs.iter().enumerate() {
            if env.done {
                continue;
            }
            let Turn::Act(seats) = env.state.turn() else { continue };
            for seat in seats.iter() {
                let r = e * rpe + if self.layout == Layout::PerSeat { seat } else { 0 };
                let legal = &self.legal[r];
                let mut rng = Rng::new(seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let a = legal[rng.below(legal.len() as u64) as usize];
                encode_target(&env.state, seat, &self.spec, legal, a, &mut TensorsMut::of(&self.spec.target, &mut t))
                    .map_err(|e| e.0)?;
                out[r * k..(r + 1) * k].copy_from_slice(&action_keys_of(&self.spec, &t));
            }
        }
        Ok(out)
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
