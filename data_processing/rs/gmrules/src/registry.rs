//! Type-erased access to every registered game by name (what gmsim exposes to
//! Python). Register a game by adding ONE line to [`GAMES`].
//!
//! Configs travel as JSON strings (`""` = the game's default config).

use crate::frame::{decode_action, encode_frame, encode_target, new_target, Frame};
use crate::game::{FrameSpec, Game, Layout, Seat, Turn};
use crate::schema::{Buf, TensorsMut};
use crate::games;
use crate::invariance;
use crate::rng::Rng;
use crate::sim::{RowBuffers, SimOptions, StepInfo, VecSim};
use crate::trajectory::{replay, Decision, Event, Stream};
use std::marker::PhantomData;

/// Every game gmsim can simulate: `&Entry::<Type>(PhantomData)`, one per line.
pub static GAMES: &[&dyn DynGame] = &[
    &Entry::<games::kuhn_poker::KuhnPoker>(PhantomData),
    &Entry::<games::goofspiel::Goofspiel>(PhantomData),
    &Entry::<games::go::Go>(PhantomData),
    &Entry::<games::pig::Pig>(PhantomData),
    &Entry::<games::breakthrough::Breakthrough>(PhantomData),
    &Entry::<games::chess::StandardChess>(PhantomData),
    &Entry::<games::chess_variants::ChessVariants>(PhantomData),
    &Entry::<games::xiangqi::Xiangqi>(PhantomData),
    &Entry::<games::backgammon::Backgammon>(PhantomData),
    &Entry::<games::doudizhu::DouDizhu>(PhantomData),
    &Entry::<games::skat::Skat>(PhantomData),
    &Entry::<games::shogi::Shogi>(PhantomData),
    &Entry::<games::bridge::Bridge>(PhantomData),
    &Entry::<games::hunl::Hunl>(PhantomData),
    &Entry::<games::tetris::Tetris>(PhantomData),
];

pub fn game_names() -> Vec<&'static str> {
    GAMES.iter().map(|g| g.name()).collect()
}

pub fn lookup(name: &str) -> Result<&'static dyn DynGame, String> {
    GAMES
        .iter()
        .copied()
        .find(|g| g.name() == name)
        .ok_or_else(|| format!("unknown gmrules game {name:?}; registered: {:?}", game_names()))
}

/// Parse a JSON config (`""` / whitespace = default).
pub fn parse_config<G: Game>(json: &str) -> Result<G::Config, String> {
    let j = if json.trim().is_empty() { "{}" } else { json };
    let cfg: G::Config = serde_json::from_str(j).map_err(|e| format!("{} config {json:?}: {e}", G::NAME))?;
    G::check_config(&cfg).map_err(|e| format!("{} config {json:?}: {e}", G::NAME))?;
    let spec = G::spec(&cfg);
    spec.validate().map_err(|e| format!("{}: {e}", G::NAME))?;
    if let Some(t) = G::teams(&cfg) {
        crate::game::validate_teams(&t, spec.n_seats).map_err(|e| format!("{}: {e}", G::NAME))?;
    }
    Ok(cfg)
}

/// A replay of any game (see [`crate::trajectory::replay`]).
pub struct DynReplay {
    pub spec: FrameSpec,
    pub decisions: Vec<Decision>,
    pub streams: Vec<Stream>,
    pub terminal: bool,
    pub outcome: Option<Vec<f32>>,
    pub returns: Option<Vec<f64>>,
    pub turns: usize,
}

pub trait DynGame: Send + Sync {
    fn name(&self) -> &'static str;
    fn version(&self) -> u32;
    fn spec(&self, cfg: &str) -> Result<FrameSpec, String>;
    /// [`Game::teams`] of the config (None = every seat for itself).
    fn teams(&self, cfg: &str) -> Result<Option<Vec<usize>>, String>;
    /// The config with every default filled in (JSON).
    fn normalize_config(&self, cfg: &str) -> Result<String, String>;
    fn new_state(&self, cfg: &str) -> Result<Box<dyn DynState>, String>;
    fn new_sim(&self, cfg: &str, n_envs: usize, opts: SimOptions) -> Result<Box<dyn DynVecSim>, String>;
    fn replay(&self, cfg: &str, events: &[Event]) -> Result<DynReplay, String>;
    fn perturb_hidden(&self, cfg: &str, events: &[Event], seat: Seat, seed: u64) -> Result<Option<Vec<Event>>, String>;
    fn random_trajectory(&self, cfg: &str, seed: u64, max_events: usize) -> Result<Vec<Event>, String>;
    /// [`invariance::check_game`]; returns a one-line report.
    fn check(&self, cfg: &str, n_games: usize, seed: u64) -> Result<String, String>;
}

pub struct Entry<G: Game>(pub PhantomData<fn() -> G>);

impl<G: Game> DynGame for Entry<G> {
    fn name(&self) -> &'static str {
        G::NAME
    }

    fn version(&self) -> u32 {
        G::VERSION
    }

    fn spec(&self, cfg: &str) -> Result<FrameSpec, String> {
        Ok(G::spec(&parse_config::<G>(cfg)?))
    }

    fn teams(&self, cfg: &str) -> Result<Option<Vec<usize>>, String> {
        Ok(G::teams(&parse_config::<G>(cfg)?))
    }

    fn normalize_config(&self, cfg: &str) -> Result<String, String> {
        serde_json::to_string(&parse_config::<G>(cfg)?).map_err(|e| e.to_string())
    }

    fn new_state(&self, cfg: &str) -> Result<Box<dyn DynState>, String> {
        let c = parse_config::<G>(cfg)?;
        let spec = G::spec(&c);
        Ok(Box::new(State { g: G::new_game(&c), spec, history: Vec::new() }))
    }

    fn new_sim(&self, cfg: &str, n_envs: usize, opts: SimOptions) -> Result<Box<dyn DynVecSim>, String> {
        Ok(Box::new(VecSim::<G>::new(parse_config::<G>(cfg)?, n_envs, opts)?))
    }

    fn replay(&self, cfg: &str, events: &[Event]) -> Result<DynReplay, String> {
        let c = parse_config::<G>(cfg)?;
        let spec = G::spec(&c);
        let r = replay::<G>(&c, events).map_err(|e| e.to_string())?;
        let terminal = r.terminal();
        Ok(DynReplay {
            streams: r.streams(&spec),
            spec,
            outcome: terminal.then(|| r.state.outcome()),
            returns: terminal.then(|| r.state.returns()),
            terminal,
            turns: r.turns,
            decisions: r.decisions,
        })
    }

    fn perturb_hidden(&self, cfg: &str, events: &[Event], seat: Seat, seed: u64) -> Result<Option<Vec<Event>>, String> {
        let c = parse_config::<G>(cfg)?;
        Ok(G::perturb_hidden(&c, events, seat, &mut Rng::new(seed)))
    }

    fn random_trajectory(&self, cfg: &str, seed: u64, max_events: usize) -> Result<Vec<Event>, String> {
        let c = parse_config::<G>(cfg)?;
        Ok(invariance::random_trajectory::<G>(&c, &mut Rng::new(seed), max_events))
    }

    fn check(&self, cfg: &str, n_games: usize, seed: u64) -> Result<String, String> {
        let c = parse_config::<G>(cfg)?;
        let r = invariance::check_game::<G>(&c, n_games, seed)?;
        Ok(format!("{r:?}"))
    }
}

/// One game state behind a trait object (step-by-step play: differential tests,
/// debugging). Keeps the trajectory of applied events.
pub trait DynState: Send + Sync {
    fn spec(&self) -> &FrameSpec;
    fn turn(&self) -> Turn;
    fn chance_outcomes(&self) -> Vec<(u32, f64)>;
    fn legal_actions(&self, seat: Seat) -> Vec<u32>;
    fn apply_chance(&mut self, outcome: u32) -> Result<(), String>;
    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), String>;
    fn outcome(&self) -> Vec<f32>;
    fn returns(&self) -> Vec<f64>;
    fn frame(&self, seat: Seat) -> Result<Frame, String>;
    /// Target tensors of `seat` taking the legal `action` now.
    fn target(&self, seat: Seat, action: u32) -> Result<Vec<Buf>, String>;
    /// The legal action of `seat` with these action keys.
    fn decode(&self, seat: Seat, keys: &[i64]) -> Result<u32, String>;
    fn history(&self) -> &[Event];
    fn state_string(&self) -> String;
    fn action_string(&self, seat: Seat, action: u32) -> String;
    fn clone_box(&self) -> Box<dyn DynState>;
}

struct State<G: Game> {
    g: G,
    spec: FrameSpec,
    history: Vec<Event>,
}

impl<G: Game> DynState for State<G> {
    fn spec(&self) -> &FrameSpec {
        &self.spec
    }

    fn turn(&self) -> Turn {
        self.g.turn()
    }

    fn chance_outcomes(&self) -> Vec<(u32, f64)> {
        let mut v = Vec::new();
        if self.g.turn() == Turn::Chance {
            self.g.chance_outcomes(&mut v);
        }
        v
    }

    fn legal_actions(&self, seat: Seat) -> Vec<u32> {
        let mut v = Vec::new();
        if matches!(self.g.turn(), Turn::Act(s) if s.contains(seat)) {
            self.g.legal_actions(seat, &mut v);
        }
        v
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), String> {
        if !self.chance_outcomes().iter().any(|&(o, p)| o == outcome && p > 0.0) {
            return Err(format!("{}: chance outcome {outcome} impossible here", G::NAME));
        }
        self.g.apply_chance(outcome).map_err(|e| e.0)?;
        self.history.push(Event::Chance(outcome));
        Ok(())
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), String> {
        if !self.legal_actions(seat).contains(&action) {
            return Err(format!("{}: action {action} is illegal for seat {seat}", G::NAME));
        }
        self.g.apply_action(seat, action).map_err(|e| e.0)?;
        self.history.push(Event::Act(seat, action));
        Ok(())
    }

    fn outcome(&self) -> Vec<f32> {
        self.g.outcome()
    }

    fn returns(&self) -> Vec<f64> {
        self.g.returns()
    }

    fn frame(&self, seat: Seat) -> Result<Frame, String> {
        if !matches!(self.g.turn(), Turn::Act(s) if s.contains(seat)) {
            return Err(format!("{}: seat {seat} is not to act", G::NAME));
        }
        let mut f = Frame::new(&self.spec);
        encode_frame(&self.g, seat, &self.spec, &mut f.as_mut(&self.spec)).map_err(|e| e.0)?;
        Ok(f)
    }

    fn target(&self, seat: Seat, action: u32) -> Result<Vec<Buf>, String> {
        let legal = self.legal_actions(seat);
        let mut t = new_target(&self.spec);
        encode_target(&self.g, seat, &self.spec, &legal, action, &mut TensorsMut::of(&self.spec.target, &mut t)).map_err(|e| e.0)?;
        Ok(t)
    }

    fn decode(&self, seat: Seat, keys: &[i64]) -> Result<u32, String> {
        let legal = self.legal_actions(seat);
        if legal.is_empty() {
            return Err(format!("{}: seat {seat} is not to act", G::NAME));
        }
        decode_action(&self.g, seat, &self.spec, &legal, keys)
    }

    fn history(&self) -> &[Event] {
        &self.history
    }

    fn state_string(&self) -> String {
        self.g.state_string()
    }

    fn action_string(&self, seat: Seat, action: u32) -> String {
        self.g.action_string(seat, action)
    }

    fn clone_box(&self) -> Box<dyn DynState> {
        Box::new(State { g: self.g.clone(), spec: self.spec.clone(), history: self.history.clone() })
    }
}

/// A [`VecSim`] behind a trait object.
pub trait DynVecSim: Send + Sync {
    fn spec(&self) -> &FrameSpec;
    fn layout(&self) -> Layout;
    fn n_envs(&self) -> usize;
    fn n_rows(&self) -> usize;
    fn rows_per_env(&self) -> usize;
    fn resize(&mut self, n: usize);
    fn reset(&mut self, env_ids: &[usize], seeds: &[u64]) -> Result<(), String>;
    /// New episodes that first play recorded trajectory prefixes ([`VecSim::reset_from`]).
    fn reset_from(&mut self, env_ids: &[usize], seeds: &[u64], prefixes: &[Vec<Event>]) -> Result<(), String>;
    /// Action keys of every row, `[R * K]` row-major.
    fn step(&mut self, keys: &[i64]) -> Result<StepInfo, String>;
    /// Action keys `[R * K]` of uniformly random legal actions of the acting rows.
    fn random_actions(&self, seed: u64) -> Result<Vec<i64>, String>;
    fn buffers(&self) -> &RowBuffers;
    fn legal(&self, row: usize) -> &[u32];
    fn done(&self, env: usize) -> bool;
    fn trajectory(&self, env: usize) -> &[Event];
    fn state_string(&self, env: usize) -> String;
}

impl<G: Game> DynVecSim for VecSim<G> {
    fn spec(&self) -> &FrameSpec {
        VecSim::spec(self)
    }
    fn layout(&self) -> Layout {
        VecSim::layout(self)
    }
    fn n_envs(&self) -> usize {
        VecSim::n_envs(self)
    }
    fn n_rows(&self) -> usize {
        VecSim::n_rows(self)
    }
    fn rows_per_env(&self) -> usize {
        VecSim::rows_per_env(self)
    }
    fn resize(&mut self, n: usize) {
        VecSim::resize(self, n)
    }
    fn reset(&mut self, env_ids: &[usize], seeds: &[u64]) -> Result<(), String> {
        VecSim::reset(self, env_ids, seeds)
    }
    fn reset_from(&mut self, env_ids: &[usize], seeds: &[u64], prefixes: &[Vec<Event>]) -> Result<(), String> {
        VecSim::reset_from(self, env_ids, seeds, prefixes)
    }
    fn step(&mut self, keys: &[i64]) -> Result<StepInfo, String> {
        VecSim::step(self, keys)
    }
    fn random_actions(&self, seed: u64) -> Result<Vec<i64>, String> {
        VecSim::random_actions(self, seed)
    }
    fn buffers(&self) -> &RowBuffers {
        VecSim::buffers(self)
    }
    fn legal(&self, row: usize) -> &[u32] {
        VecSim::legal(self, row)
    }
    fn done(&self, env: usize) -> bool {
        VecSim::done(self, env)
    }
    fn trajectory(&self, env: usize) -> &[Event] {
        VecSim::trajectory(self, env)
    }
    fn state_string(&self, env: usize) -> String {
        self.state(env).state_string()
    }
}
