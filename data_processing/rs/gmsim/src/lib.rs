//! `gmsim._native` — batched simulators and replay of every gmrules game for Python
//! (PyO3, abi3-py310). A thin wrapper: all logic lives in `gmrules` (tested there);
//! arrays cross the boundary as native-endian `bytearray`s that the Python package
//! (`python/gmsim/__init__.py`) views as numpy arrays. Events are int64 pairs
//! `(seat, id)` with seat −1 for chance. Configs are JSON strings ("" = default).
//! Rust errors and panics (e.g. an invalid config) raise ValueError.

use gmrules::registry::{self, DynReplay, DynState, DynVecSim};
use gmrules::{Event, Frame, FrameSpec, SimOptions, Turn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn panic_msg(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic in gmrules".into()
    }
}

/// Run `f`, turning its error or panic into ValueError.
fn guard<T>(f: impl FnOnce() -> Result<T, String>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r.map_err(PyValueError::new_err),
        Err(p) => Err(PyValueError::new_err(panic_msg(p))),
    }
}

/// Plain-old-data element types that cross the boundary as raw bytes.
trait Pod: Copy {}
impl Pod for f32 {}
impl Pod for f64 {}
impl Pod for i32 {}
impl Pod for i64 {}
impl Pod for u8 {}

fn as_bytes<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: Pod types have no padding and no invalid bit patterns; u8 has alignment 1
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytearray<'py, T: Pod>(py: Python<'py>, v: &[T]) -> Bound<'py, PyByteArray> {
    PyByteArray::new(py, as_bytes(v))
}

fn parse_i64(b: &[u8], what: &str) -> PyResult<Vec<i64>> {
    if !b.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(format!("{what}: {} bytes is not a whole number of int64", b.len())));
    }
    Ok(b.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect())
}

fn parse_events(b: &[u8]) -> PyResult<Vec<Event>> {
    let v = parse_i64(b, "events")?;
    if !v.len().is_multiple_of(2) {
        return Err(PyValueError::new_err("events: int64 pairs (seat, id) expected"));
    }
    v.chunks_exact(2).map(|p| Event::from_pair(p[0], p[1]).map_err(PyValueError::new_err)).collect()
}

fn events_bytes<'py>(py: Python<'py>, events: &[Event]) -> Bound<'py, PyBytes> {
    let flat: Vec<i64> = events.iter().flat_map(|e| {
        let (s, i) = e.to_pair();
        [s, i]
    }).collect();
    PyBytes::new(py, as_bytes(&flat))
}

fn spec_dict<'py>(py: Python<'py>, s: &FrameSpec) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_seats", s.n_seats)?;
    d.set_item("max_entities", s.max_entities)?;
    d.set_item("entity_features", s.entity_features)?;
    d.set_item("entity_types", s.entity_types)?;
    d.set_item("global_features", s.global_features)?;
    d.set_item("max_actions", s.max_actions)?;
    d.set_item("action_features", s.action_features)?;
    d.set_item("hidden_info", s.hidden_info)?;
    d.set_item("simultaneous", s.simultaneous)?;
    d.set_item("bits_entities", s.bits_entities)?;
    d.set_item("bits_globals", s.bits_globals)?;
    d.set_item("bits_actions", s.bits_actions)?;
    d.set_item("max_decisions", s.max_decisions)?;
    d.set_item("layout", s.layout().name())?;
    Ok(d)
}

fn frames_into<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, frames: &[&Frame]) -> PyResult<()> {
    let cat_f32 = |get: &dyn Fn(&Frame) -> &[f32]| frames.iter().flat_map(|f| get(f).iter().copied()).collect::<Vec<f32>>();
    let cat_u8 = |get: &dyn Fn(&Frame) -> &[u8]| frames.iter().flat_map(|f| get(f).iter().copied()).collect::<Vec<u8>>();
    d.set_item("entities", bytearray(py, &cat_f32(&|f| &f.entities)))?;
    d.set_item("entity_mask", bytearray(py, &cat_u8(&|f| &f.entity_mask)))?;
    let et: Vec<i32> = frames.iter().flat_map(|f| f.entity_type.iter().copied()).collect();
    d.set_item("entity_type", bytearray(py, &et))?;
    d.set_item("globals", bytearray(py, &cat_f32(&|f| &f.globals)))?;
    d.set_item("actions", bytearray(py, &cat_f32(&|f| &f.actions)))?;
    d.set_item("action_mask", bytearray(py, &cat_u8(&|f| &f.action_mask)))?;
    let legal: Vec<Vec<u32>> = frames.iter().map(|f| f.legal.clone()).collect();
    d.set_item("legal", legal)?;
    Ok(())
}

/// Names of the registered games.
#[pyfunction]
fn games() -> Vec<&'static str> {
    registry::game_names()
}

/// `{"name", "version", "config" (normalised JSON), "spec": {...}}`.
#[pyfunction]
#[pyo3(signature = (game, config=""))]
fn game_info<'py>(py: Python<'py>, game: &str, config: &str) -> PyResult<Bound<'py, PyDict>> {
    let (g, spec, cfg) = guard(|| {
        let g = registry::lookup(game)?;
        Ok((g, g.spec(config)?, g.normalize_config(config)?))
    })?;
    let d = PyDict::new(py);
    d.set_item("name", g.name())?;
    d.set_item("version", g.version())?;
    d.set_item("config", cfg)?;
    d.set_item("spec", spec_dict(py, &spec)?)?;
    Ok(d)
}

fn replay_dict<'py>(py: Python<'py>, r: &DynReplay) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let dec = &r.decisions;
    d.set_item("n", dec.len())?;
    d.set_item("seat", bytearray(py, &dec.iter().map(|x| x.seat as i64).collect::<Vec<_>>()))?;
    d.set_item("action", bytearray(py, &dec.iter().map(|x| x.action as i64).collect::<Vec<_>>()))?;
    d.set_item("action_idx", bytearray(py, &dec.iter().map(|x| x.action_idx as i64).collect::<Vec<_>>()))?;
    d.set_item("turn", bytearray(py, &dec.iter().map(|x| x.turn as i64).collect::<Vec<_>>()))?;
    d.set_item("event", bytearray(py, &dec.iter().map(|x| x.event as i64).collect::<Vec<_>>()))?;
    frames_into(py, &d, &dec.iter().map(|x| &x.frame).collect::<Vec<_>>())?;
    d.set_item("terminal", r.terminal)?;
    d.set_item("outcome", r.outcome.clone())?;
    d.set_item("returns", r.returns.clone())?;
    d.set_item("turns", r.turns)?;
    let streams: Vec<(i64, Vec<usize>)> = r.streams.iter().map(|s| (s.seat.map_or(-1, |x| x as i64), s.decisions.clone())).collect();
    d.set_item("streams", streams)?;
    d.set_item("spec", spec_dict(py, &r.spec)?)?;
    Ok(d)
}

/// Verified replay of `events` (int64 pairs) → decision frames (see gmrules::trajectory).
#[pyfunction]
#[pyo3(signature = (game, config, events))]
fn replay<'py>(py: Python<'py>, game: &str, config: &str, events: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let ev = parse_events(events)?;
    let r = guard(|| registry::lookup(game)?.replay(config, &ev))?;
    replay_dict(py, &r)
}

/// A trajectory differing only in information hidden from `seat` (None = nothing to perturb).
#[pyfunction]
fn perturb_hidden<'py>(py: Python<'py>, game: &str, config: &str, events: &[u8], seat: usize, seed: u64) -> PyResult<Option<Bound<'py, PyBytes>>> {
    let ev = parse_events(events)?;
    let p = guard(|| registry::lookup(game)?.perturb_hidden(config, &ev, seat, seed))?;
    Ok(p.map(|p| events_bytes(py, &p)))
}

/// A uniformly random playout (int64 pairs).
#[pyfunction]
#[pyo3(signature = (game, config, seed, max_events=100_000))]
fn random_trajectory<'py>(py: Python<'py>, game: &str, config: &str, seed: u64, max_events: usize) -> PyResult<Bound<'py, PyBytes>> {
    let ev = guard(|| registry::lookup(game)?.random_trajectory(config, seed, max_events))?;
    Ok(events_bytes(py, &ev))
}

/// The gmrules conformance battery (causality, privacy, frame checks, simulator
/// replays); raises ValueError on a violation, returns the report.
#[pyfunction]
#[pyo3(signature = (game, config="", n_games=200, seed=0))]
fn check_game(game: &str, config: &str, n_games: usize, seed: u64) -> PyResult<String> {
    guard(|| registry::lookup(game)?.check(config, n_games, seed))
}

/// N envs of one game stepped together (gmrules::sim::VecSim).
#[pyclass(module = "gmsim._native")]
struct Sim {
    inner: Box<dyn DynVecSim>,
    game: String,
}

#[pymethods]
impl Sim {
    #[new]
    #[pyo3(signature = (game, config="", n_envs=1, autoreset=false, max_turns=None))]
    fn new(game: &str, config: &str, n_envs: usize, autoreset: bool, max_turns: Option<usize>) -> PyResult<Self> {
        let inner = guard(|| registry::lookup(game)?.new_sim(config, n_envs, SimOptions { autoreset, max_turns }))?;
        Ok(Sim { inner, game: game.to_string() })
    }

    #[getter]
    fn game(&self) -> &str {
        &self.game
    }

    #[getter]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        spec_dict(py, self.inner.spec())
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.n_envs()
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    #[getter]
    fn rows_per_env(&self) -> usize {
        self.inner.rows_per_env()
    }

    #[getter]
    fn layout(&self) -> &'static str {
        self.inner.layout().name()
    }

    /// Resize to `n` envs (all finished until reset).
    fn resize(&mut self, n: usize) {
        self.inner.resize(n)
    }

    /// Start new episodes in `env_ids` with per-episode `seeds`.
    fn reset(&mut self, env_ids: Vec<usize>, seeds: Vec<u64>) -> PyResult<()> {
        guard(|| self.inner.reset(&env_ids, &seeds))
    }

    /// Apply one candidate index per row (int64 bytes, [R]); returns the step info.
    fn step<'py>(&mut self, py: Python<'py>, action_idx: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let a = parse_i64(action_idx, "action_idx")?;
        let info = guard(|| self.inner.step(&a))?;
        let d = PyDict::new(py);
        d.set_item("rewards", bytearray(py, &info.rewards))?;
        d.set_item("returns", bytearray(py, &info.returns))?;
        d.set_item("done", bytearray(py, &info.done))?;
        d.set_item("terminated", bytearray(py, &info.terminated))?;
        d.set_item("truncated", bytearray(py, &info.truncated))?;
        d.set_item("winner", bytearray(py, &info.winner))?;
        d.set_item("episode_len", bytearray(py, &info.episode_len))?;
        d.set_item("applied", bytearray(py, &info.applied))?;
        Ok(d)
    }

    /// Every row's current frame + row metadata (fresh buffers each call).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = self.inner.buffers();
        let d = PyDict::new(py);
        d.set_item("entities", bytearray(py, &b.entities))?;
        d.set_item("entity_mask", bytearray(py, &b.entity_mask))?;
        d.set_item("entity_type", bytearray(py, &b.entity_type))?;
        d.set_item("globals", bytearray(py, &b.globals))?;
        d.set_item("actions", bytearray(py, &b.actions))?;
        d.set_item("action_mask", bytearray(py, &b.action_mask))?;
        d.set_item("n_legal", bytearray(py, &b.n_legal))?;
        d.set_item("to_play", bytearray(py, &b.to_play))?;
        d.set_item("step_idx", bytearray(py, &b.step_idx))?;
        d.set_item("active", bytearray(py, &b.active))?;
        d.set_item("new_episode", bytearray(py, &b.new_episode))?;
        d.set_item("forced", bytearray(py, &b.forced))?;
        Ok(d)
    }

    /// Canonical action ids of a row's candidates.
    fn legal_ids(&self, row: usize) -> PyResult<Vec<u32>> {
        if row >= self.inner.n_rows() {
            return Err(PyValueError::new_err(format!("row {row} out of range")));
        }
        Ok(self.inner.legal(row).to_vec())
    }

    /// The env's trajectory so far (int64 pairs).
    fn trajectory<'py>(&self, py: Python<'py>, env: usize) -> PyResult<Bound<'py, PyBytes>> {
        if env >= self.inner.n_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(events_bytes(py, self.inner.trajectory(env)))
    }

    fn done(&self, env: usize) -> bool {
        self.inner.done(env)
    }

    fn state_string(&self, env: usize) -> String {
        self.inner.state_string(env)
    }
}

/// One game state, stepped event by event (differential tests, debugging).
#[pyclass(module = "gmsim._native")]
struct State {
    inner: Box<dyn DynState>,
}

#[pymethods]
impl State {
    #[new]
    #[pyo3(signature = (game, config=""))]
    fn new(game: &str, config: &str) -> PyResult<Self> {
        Ok(State { inner: guard(|| registry::lookup(game)?.new_state(config))? })
    }

    /// ("chance", []) | ("act", [seats]) | ("terminal", []).
    fn turn(&self) -> (&'static str, Vec<usize>) {
        match self.inner.turn() {
            Turn::Chance => ("chance", vec![]),
            Turn::Act(s) => ("act", s.iter().collect()),
            Turn::Terminal => ("terminal", vec![]),
        }
    }

    fn chance_outcomes(&self) -> Vec<(u32, f64)> {
        self.inner.chance_outcomes()
    }

    fn legal_actions(&self, seat: usize) -> Vec<u32> {
        self.inner.legal_actions(seat)
    }

    fn apply_chance(&mut self, outcome: u32) -> PyResult<()> {
        guard(|| self.inner.apply_chance(outcome))
    }

    fn apply_action(&mut self, seat: usize, action: u32) -> PyResult<()> {
        guard(|| self.inner.apply_action(seat, action))
    }

    fn outcome(&self) -> Vec<f32> {
        self.inner.outcome()
    }

    fn returns(&self) -> Vec<f64> {
        self.inner.returns()
    }

    /// `seat`'s decision frame now (bytearrays + "legal").
    fn frame<'py>(&self, py: Python<'py>, seat: usize) -> PyResult<Bound<'py, PyDict>> {
        let f = guard(|| self.inner.frame(seat))?;
        let d = PyDict::new(py);
        frames_into(py, &d, &[&f])?;
        Ok(d)
    }

    fn history(&self) -> Vec<(i64, i64)> {
        self.inner.history().iter().map(|e| e.to_pair()).collect()
    }

    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        spec_dict(py, self.inner.spec())
    }

    fn action_string(&self, seat: usize, action: u32) -> String {
        self.inner.action_string(seat, action)
    }

    fn clone(&self) -> State {
        State { inner: self.inner.clone_box() }
    }

    fn __str__(&self) -> String {
        self.inner.state_string()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(games, m)?)?;
    m.add_function(wrap_pyfunction!(game_info, m)?)?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_hidden, m)?)?;
    m.add_function(wrap_pyfunction!(random_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(check_game, m)?)?;
    m.add_class::<Sim>()?;
    m.add_class::<State>()?;
    Ok(())
}
