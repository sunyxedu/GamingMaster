//! `gmsim._native` — batched simulators and replay of every gmrules game for Python
//! (PyO3, abi3-py310). A thin wrapper: all logic lives in `gmrules` (tested there);
//! tensors cross the boundary as native-endian `bytearray`s keyed by tensor name,
//! which the Python package (`python/gmsim/__init__.py`) views as numpy arrays using
//! the game's spec (shapes, dtypes). Events are int64 pairs `(seat, id)` with seat −1
//! for chance. Configs are JSON strings ("" = default). Rust errors and panics (e.g.
//! an invalid config) raise ValueError. `Sim.reset` / `Sim.step` release the GIL.

use gmrules::registry::{self, DynReplay, DynState, DynVecSim};
use gmrules::{Buf, Event, Frame, FrameSpec, SimOptions, TensorSpec, Turn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
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

/// Run `f`, turning its error or panic into a message.
fn catch<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => Err(panic_msg(p)),
    }
}

/// Run `f`, turning its error or panic into ValueError.
fn guard<T>(f: impl FnOnce() -> Result<T, String>) -> PyResult<T> {
    catch(f).map_err(PyValueError::new_err)
}

/// Plain-old-data element types that cross the boundary as raw bytes.
trait Pod: Copy {}
impl Pod for f32 {}
impl Pod for f64 {}
impl Pod for i32 {}
impl Pod for i64 {}
impl Pod for u8 {}
impl Pod for u16 {}

fn as_bytes<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: Pod types have no padding and no invalid bit patterns; u8 has alignment 1
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytearray<'py, T: Pod>(py: Python<'py>, v: &[T]) -> Bound<'py, PyByteArray> {
    PyByteArray::new(py, as_bytes(v))
}

fn parse_i64(b: &[u8], what: &str) -> PyResult<Vec<i64>> {
    if b.len() % 8 != 0 {
        return Err(PyValueError::new_err(format!("{what}: {} bytes is not a whole number of int64", b.len())));
    }
    Ok(b.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect())
}

fn parse_events(b: &[u8]) -> PyResult<Vec<Event>> {
    let v = parse_i64(b, "events")?;
    if v.len() % 2 != 0 {
        return Err(PyValueError::new_err("events: int64 pairs (seat, id) expected"));
    }
    v.chunks_exact(2).map(|p| Event::from_pair(p[0], p[1]).map_err(PyValueError::new_err)).collect()
}

fn events_bytes<'py>(py: Python<'py>, events: &[Event]) -> Bound<'py, PyBytes> {
    let flat: Vec<i64> = events
        .iter()
        .flat_map(|e| {
            let (s, i) = e.to_pair();
            [s, i]
        })
        .collect();
    PyBytes::new(py, as_bytes(&flat))
}

fn tensor_list<'py>(py: Python<'py>, ts: &[TensorSpec]) -> PyResult<Bound<'py, PyList>> {
    let l = PyList::empty(py);
    for t in ts {
        let d = PyDict::new(py);
        d.set_item("name", &t.name)?;
        d.set_item("shape", t.shape.clone())?;
        d.set_item("dtype", t.dtype.name())?;
        d.set_item("bits", t.bits)?;
        l.append(d)?;
    }
    Ok(l)
}

fn json_to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    // the meta is small: round-trip through Python's json module
    let json = py.import("json")?;
    json.call_method1("loads", (v.to_string(),))
}

fn spec_dict<'py>(py: Python<'py>, s: &FrameSpec) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_seats", s.n_seats)?;
    d.set_item("hidden_info", s.hidden_info)?;
    d.set_item("simultaneous", s.simultaneous)?;
    d.set_item("max_decisions", s.max_decisions)?;
    d.set_item("layout", s.layout().name())?;
    d.set_item("schema", &s.schema)?;
    d.set_item("obs", tensor_list(py, &s.obs)?)?;
    d.set_item("legal", tensor_list(py, &s.legal)?)?;
    d.set_item("target", tensor_list(py, &s.target)?)?;
    d.set_item("action_keys", s.action_keys.clone())?;
    d.set_item("meta", json_to_py(py, &s.meta)?)?;
    match &s.candidate {
        Some(c) => d.set_item("candidate", json_to_py(py, &serde_json::to_value(c).map_err(|e| PyValueError::new_err(e.to_string()))?)?)?,
        None => d.set_item("candidate", py.None())?,
    }
    Ok(d)
}

/// `{name: bytearray}` of `rows` frames of each tensor, concatenated in order.
fn tensors_dict<'py>(py: Python<'py>, specs: &[TensorSpec], frames: &[&[Buf]]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (i, t) in specs.iter().enumerate() {
        let len: usize = frames.iter().map(|f| f[i].as_bytes().len()).sum();
        let arr = PyByteArray::new_with(py, len, |out| {
            let mut at = 0;
            for f in frames {
                let b = f[i].as_bytes();
                out[at..at + b.len()].copy_from_slice(b);
                at += b.len();
            }
            Ok(())
        })?;
        d.set_item(&t.name, arr)?;
    }
    Ok(d)
}

/// `{name: bytearray}` of whole batch buffers.
fn batch_dict<'py>(py: Python<'py>, specs: &[TensorSpec], bufs: &[Buf]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (t, b) in specs.iter().zip(bufs) {
        d.set_item(&t.name, PyByteArray::new(py, b.as_bytes()))?;
    }
    Ok(d)
}

fn frames_into<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, spec: &FrameSpec, frames: &[&Frame]) -> PyResult<()> {
    let obs: Vec<&[Buf]> = frames.iter().map(|f| f.obs.as_slice()).collect();
    let legal: Vec<&[Buf]> = frames.iter().map(|f| f.legal.as_slice()).collect();
    d.set_item("obs", tensors_dict(py, &spec.obs, &obs)?)?;
    d.set_item("legal", tensors_dict(py, &spec.legal, &legal)?)?;
    let ids: Vec<Vec<u32>> = frames.iter().map(|f| f.legal_ids.clone()).collect();
    d.set_item("legal_ids", ids)?;
    Ok(())
}

/// Names of the registered games.
#[pyfunction]
fn games() -> Vec<&'static str> {
    registry::game_names()
}

/// `{"name", "version", "config" (normalised JSON), "spec": {...}, "teams"}` (teams:
/// team id per seat, or None).
#[pyfunction]
#[pyo3(signature = (game, config=""))]
fn game_info<'py>(py: Python<'py>, game: &str, config: &str) -> PyResult<Bound<'py, PyDict>> {
    let (g, spec, cfg, teams) = guard(|| {
        let g = registry::lookup(game)?;
        Ok((g, g.spec(config)?, g.normalize_config(config)?, g.teams(config)?))
    })?;
    let d = PyDict::new(py);
    d.set_item("name", g.name())?;
    d.set_item("version", g.version())?;
    d.set_item("config", cfg)?;
    d.set_item("spec", spec_dict(py, &spec)?)?;
    d.set_item("teams", teams)?;
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
    frames_into(py, &d, &r.spec, &dec.iter().map(|x| &x.frame).collect::<Vec<_>>())?;
    let targets: Vec<&[Buf]> = dec.iter().map(|x| x.target.as_slice()).collect();
    d.set_item("target", tensors_dict(py, &r.spec.target, &targets)?)?;
    d.set_item("terminal", r.terminal)?;
    d.set_item("outcome", r.outcome.clone())?;
    d.set_item("returns", r.returns.clone())?;
    d.set_item("turns", r.turns)?;
    let streams: Vec<(i64, Vec<usize>)> = r.streams.iter().map(|s| (s.seat.map_or(-1, |x| x as i64), s.decisions.clone())).collect();
    d.set_item("streams", streams)?;
    d.set_item("spec", spec_dict(py, &r.spec)?)?;
    Ok(d)
}

/// Verified replay of `events` (int64 pairs) → decision frames + targets (see
/// gmrules::trajectory).
#[pyfunction]
#[pyo3(signature = (game, config, events))]
fn replay<'py>(py: Python<'py>, game: &str, config: &str, events: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let ev = parse_events(events)?;
    let r = py.detach(|| guard(|| registry::lookup(game)?.replay(config, &ev)))?;
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

/// The gmrules conformance battery (causality, privacy, frame checks, action-key
/// round trips, simulator replays); raises ValueError on a violation, returns the report.
#[pyfunction]
#[pyo3(signature = (game, config="", n_games=200, seed=0))]
fn check_game(py: Python<'_>, game: &str, config: &str, n_games: usize, seed: u64) -> PyResult<String> {
    py.detach(|| guard(|| registry::lookup(game)?.check(config, n_games, seed)))
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
    #[pyo3(signature = (game, config="", n_envs=1, autoreset=false, max_turns=None, threads=1))]
    fn new(game: &str, config: &str, n_envs: usize, autoreset: bool, max_turns: Option<usize>, threads: usize) -> PyResult<Self> {
        let inner = guard(|| registry::lookup(game)?.new_sim(config, n_envs, SimOptions { autoreset, max_turns, threads }))?;
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
    fn reset(&mut self, py: Python<'_>, env_ids: Vec<usize>, seeds: Vec<u64>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| guard(|| inner.reset(&env_ids, &seeds)))
    }

    /// Start new episodes in `env_ids` that first play recorded trajectory prefixes
    /// (one per env, each flat int64 `[seat, id, seat, id, ...]`), then go on with
    /// chance from `seeds`.
    fn reset_from(&mut self, py: Python<'_>, env_ids: Vec<usize>, seeds: Vec<u64>, prefixes: Vec<Vec<i64>>) -> PyResult<()> {
        let pre = prefixes
            .iter()
            .map(|p| {
                if p.len() % 2 != 0 {
                    return Err(PyValueError::new_err("prefix: flat (seat, id) pairs expected"));
                }
                p.chunks_exact(2).map(|c| Event::from_pair(c[0], c[1]).map_err(PyValueError::new_err)).collect()
            })
            .collect::<PyResult<Vec<Vec<Event>>>>()?;
        let inner = &mut self.inner;
        py.detach(|| guard(|| inner.reset_from(&env_ids, &seeds, &pre)))
    }

    /// Apply the action keys of every row (int64 bytes, [R, K] row-major); returns the
    /// step info.
    fn step<'py>(&mut self, py: Python<'py>, keys: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let a = parse_i64(keys, "action keys")?;
        let inner = &mut self.inner;
        let info = py.detach(|| guard(|| inner.step(&a)))?;
        let d = PyDict::new(py);
        d.set_item("rewards", bytearray(py, &info.rewards))?;
        d.set_item("returns", bytearray(py, &info.returns))?;
        d.set_item("done", bytearray(py, &info.done))?;
        d.set_item("terminated", bytearray(py, &info.terminated))?;
        d.set_item("truncated", bytearray(py, &info.truncated))?;
        d.set_item("winner", bytearray(py, &info.winner))?;
        d.set_item("episode_len", bytearray(py, &info.episode_len))?;
        d.set_item("applied", bytearray(py, &info.applied))?;
        d.set_item("acted", bytearray(py, &info.acted))?;
        Ok(d)
    }

    /// Action keys (int64 bytes, [R, K]) of uniformly random legal actions of the
    /// acting rows (zeros elsewhere).
    fn random_actions<'py>(&self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyByteArray>> {
        let a = guard(|| self.inner.random_actions(seed))?;
        Ok(bytearray(py, &a))
    }

    /// Every row's current frame + row metadata (fresh buffers each call).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = self.inner.buffers();
        let spec = self.inner.spec();
        let d = PyDict::new(py);
        d.set_item("obs", batch_dict(py, &spec.obs, &b.obs)?)?;
        d.set_item("legal", batch_dict(py, &spec.legal, &b.legal)?)?;
        d.set_item("n_legal", bytearray(py, &b.n_legal))?;
        d.set_item("to_play", bytearray(py, &b.to_play))?;
        d.set_item("step_idx", bytearray(py, &b.step_idx))?;
        d.set_item("active", bytearray(py, &b.active))?;
        d.set_item("new_episode", bytearray(py, &b.new_episode))?;
        d.set_item("forced", bytearray(py, &b.forced))?;
        Ok(d)
    }

    /// Canonical legal action ids of a row's current frame.
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

    fn done(&self, env: usize) -> PyResult<bool> {
        if env >= self.inner.n_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(self.inner.done(env))
    }

    /// Done flag of every env (bytes, uint8 [N]).
    fn done_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        let d: Vec<u8> = (0..self.inner.n_envs()).map(|e| self.inner.done(e) as u8).collect();
        bytearray(py, &d)
    }

    fn state_string(&self, env: usize) -> PyResult<String> {
        if env >= self.inner.n_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(self.inner.state_string(env))
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

    /// `seat`'s decision frame now: {"obs": {name: bytearray}, "legal": {...}, "legal_ids"}.
    fn frame<'py>(&self, py: Python<'py>, seat: usize) -> PyResult<Bound<'py, PyDict>> {
        let f = guard(|| self.inner.frame(seat))?;
        let d = PyDict::new(py);
        frames_into(py, &d, self.inner.spec(), &[&f])?;
        Ok(d)
    }

    /// Target tensors of `seat` taking the legal `action` now: {name: bytearray}.
    fn target<'py>(&self, py: Python<'py>, seat: usize, action: u32) -> PyResult<Bound<'py, PyDict>> {
        let t = guard(|| self.inner.target(seat, action))?;
        tensors_dict(py, &self.inner.spec().target, &[t.as_slice()])
    }

    /// The legal action of `seat` whose action keys are `keys`.
    fn decode(&self, seat: usize, keys: Vec<i64>) -> PyResult<u32> {
        guard(|| self.inner.decode(seat, &keys))
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

/// N riichi mahjong hanchan games (gmmahjong::vec::MjVec: riichienv-core's engine observed
/// through the converter's state — a frame is the gmdata converter's frame).  Rows are
/// `env * 4 + seat`; buffers as in gmmahjong's `vec` module docs.
#[pyclass(module = "gmsim._native")]
struct MahjongSim {
    inner: gmmahjong::vec::MjVec,
}

#[pymethods]
impl MahjongSim {
    #[new]
    #[pyo3(signature = (n_envs=1, seeds=vec![], autoreset=false, threads=1))]
    fn new(n_envs: usize, seeds: Vec<u64>, autoreset: bool, threads: usize) -> PyResult<Self> {
        Ok(MahjongSim { inner: guard(|| gmmahjong::vec::MjVec::new(n_envs, &seeds, autoreset, threads))? })
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
    fn failures(&self) -> usize {
        self.inner.failures
    }

    #[getter]
    fn last_error(&self) -> Option<String> {
        self.inner.last_error.clone()
    }

    fn resize(&mut self, n: usize) -> PyResult<()> {
        guard(|| self.inner.resize(n))
    }

    fn reset(&mut self, py: Python<'_>, env_ids: Vec<usize>, seeds: Vec<u64>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| guard(|| inner.reset(&env_ids, &seeds)))
    }

    fn done(&self, env: usize) -> PyResult<bool> {
        if env >= self.inner.n_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(self.inner.done(env))
    }

    fn scores(&self, env: usize) -> PyResult<Vec<i64>> {
        if env >= self.inner.n_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(self.inner.scores(env).to_vec())
    }

    /// Every row's current frame (fresh buffers each call).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = py.detach(|| self.inner.observe());
        let d = PyDict::new(py);
        d.set_item("tiles_seq", bytearray(py, &b.tiles))?;
        d.set_item("players_seq", bytearray(py, &b.players))?;
        d.set_item("table_seq", bytearray(py, &b.table))?;
        d.set_item("action_type", bytearray(py, &b.legal_type))?;
        d.set_item("tile_cond", bytearray(py, &b.tile_cond))?;
        d.set_item("active", bytearray(py, &b.active))?;
        d.set_item("forced", bytearray(py, &b.forced))?;
        d.set_item("to_play", bytearray(py, &b.to_play))?;
        d.set_item("step_idx", bytearray(py, &b.step_idx))?;
        d.set_item("new_episode", bytearray(py, &b.new_episode))?;
        Ok(d)
    }

    /// Apply every acting row's (action_type, tile_idx) (int64 bytes, one per row).
    fn step<'py>(&mut self, py: Python<'py>, action_type: &[u8], tile_idx: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let at = parse_i64(action_type, "action_type")?;
        let tl = parse_i64(tile_idx, "tile_idx")?;
        let inner = &mut self.inner;
        let info = py.detach(|| guard(|| inner.step(&at, &tl)))?;
        let d = PyDict::new(py);
        d.set_item("rewards", bytearray(py, &info.rewards))?;
        d.set_item("returns", bytearray(py, &info.returns))?;
        d.set_item("done", bytearray(py, &info.done))?;
        d.set_item("terminated", bytearray(py, &info.terminated))?;
        d.set_item("truncated", bytearray(py, &info.truncated))?;
        d.set_item("winner", bytearray(py, &info.winner))?;
        d.set_item("episode_len", bytearray(py, &info.episode_len))?;
        d.set_item("applied_type", bytearray(py, &info.applied_type))?;
        d.set_item("applied_tile", bytearray(py, &info.applied_tile))?;
        d.set_item("acted", bytearray(py, &info.acted))?;
        Ok(d)
    }
}

/// N StarCraft II games against the built-in AI (gmsc2::env::Sc2Vec: an SC2 process per env
/// on its own thread; the decisions and frames of gmsc2's module docs — a frame is the gmdata
/// converter's).  One row per env; buffers keyed like the shards' obs tensors.
#[pyclass(module = "gmsim._native")]
struct Sc2Sim {
    inner: gmsc2::env::Sc2Vec,
}

fn u8s_of(b: &[u8], n: usize, what: &str) -> PyResult<Vec<u8>> {
    if b.len() != n {
        return Err(PyValueError::new_err(format!("{what}: {} bytes, expected {n}", b.len())));
    }
    Ok(b.to_vec())
}

#[pymethods]
impl Sc2Sim {
    #[new]
    #[pyo3(signature = (n_envs, install, vocab, maps, races=vec![4], ai_race=4, difficulty=3, max_loops=26880,
                        port=25000, seeds=vec![], autoreset=true, tmp=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, n_envs: usize, install: String, vocab: String, maps: Vec<String>, races: Vec<i32>, ai_race: i32,
           difficulty: i32, max_loops: u32, port: u16, seeds: Vec<u64>, autoreset: bool, tmp: Option<String>) -> PyResult<Self> {
        let voc = gmsc2::Vocab::load(std::path::Path::new(&vocab)).map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let tmp = tmp.map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("gm-sc2-sim-{}", std::process::id())));
        let cfg = gmsc2::env::Config { install: install.into(), maps, races, ai_race, difficulty, max_loops, port, tmp };
        let inner = py.detach(|| gmsc2::env::Sc2Vec::new(n_envs, cfg, voc, &seeds, autoreset))
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        Ok(Sc2Sim { inner })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.n_envs()
    }

    #[getter]
    fn failures(&self) -> usize {
        self.inner.failures
    }

    #[getter]
    fn last_error(&self) -> Option<String> {
        self.inner.last_error.clone()
    }

    #[getter]
    fn n_upgrades(&self) -> usize {
        self.inner.voc.upgrades.len()
    }

    fn reset(&mut self, py: Python<'_>, env_ids: Vec<usize>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.reset(&env_ids)).map_err(|e| PyValueError::new_err(format!("{e:#}")))
    }

    /// Every env's current decision (fresh buffers each call).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = py.detach(|| self.inner.observe());
        let d = PyDict::new(py);
        d.set_item("units", bytearray(py, &b.units))?;
        d.set_item("unit_xy", bytearray(py, &b.unit_xy))?;
        d.set_item("unit_type", bytearray(py, &b.unit_type))?;
        d.set_item("unit_order", bytearray(py, &b.unit_order))?;
        d.set_item("unit_buffs", bytearray(py, &b.unit_buffs))?;
        d.set_item("unit_mask", bytearray(py, &b.unit_mask))?;
        d.set_item("unit_own", bytearray(py, &b.unit_own))?;
        d.set_item("unit_commanded", bytearray(py, &b.unit_commanded))?;
        d.set_item("map_bits", bytearray(py, &b.map_bits))?;
        d.set_item("globals", bytearray(py, &b.globals))?;
        d.set_item("upgrades", bytearray(py, &b.upgrades))?;
        d.set_item("last_action", bytearray(py, &b.last_action))?;
        d.set_item("window_start", bytearray(py, &b.window_start))?;
        Ok(d)
    }

    /// One decision per env: action_type / target_kind / target_unit / target_point int64,
    /// actors uint8 [n × U], queued uint8 (bytes).
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(&mut self, py: Python<'py>, action_type: &[u8], actors: &[u8], target_kind: &[u8], target_unit: &[u8],
                 target_point: &[u8], queued: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.n_envs();
        let (at, tk) = (parse_i64(action_type, "action_type")?, parse_i64(target_kind, "target_kind")?);
        let (tu, tp) = (parse_i64(target_unit, "target_unit")?, parse_i64(target_point, "target_point")?);
        let ac = u8s_of(actors, n * gmsc2::U, "actors")?;
        let qu = u8s_of(queued, n, "queued")?;
        if [at.len(), tk.len(), tu.len(), tp.len()].iter().any(|&l| l != n) {
            return Err(PyValueError::new_err(format!("sc2: every action needs {n} rows")));
        }
        let labels: Vec<gmsc2::Label> = (0..n).map(|i| gmsc2::Label {
            action_type: at[i], actors: ac[i * gmsc2::U..(i + 1) * gmsc2::U].iter().map(|&a| a != 0).collect(),
            target_kind: tk[i], target_unit: tu[i], target_point: tp[i], queued: qu[i] != 0,
        }).collect();
        let inner = &mut self.inner;
        let info = py.detach(|| inner.step(&labels)).map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let d = PyDict::new(py);
        d.set_item("rewards", bytearray(py, &info.rewards))?;
        d.set_item("done", bytearray(py, &info.done))?;
        d.set_item("truncated", bytearray(py, &info.truncated))?;
        d.set_item("sent", bytearray(py, &info.sent))?;
        d.set_item("refused", bytearray(py, &info.refused))?;
        d.set_item("episode_len", bytearray(py, &info.episode_len))?;
        d.set_item("game_loop", bytearray(py, &info.game_loop))?;
        Ok(d)
    }
}

/// The Clash Royale catalog and timeline of the process (crdata's; loaded once — a live
/// battle's frame builder borrows them for its lifetime).
static CR_CATALOG: std::sync::OnceLock<(String, crdata::catalog::Catalog, crdata::catalog::Timeline)> = std::sync::OnceLock::new();

fn cr_catalog(dir: &str) -> PyResult<&'static (String, crdata::catalog::Catalog, crdata::catalog::Timeline)> {
    let c = CR_CATALOG.get_or_init(|| {
        let d = std::path::Path::new(dir);
        let cat = crdata::catalog::Catalog::load(&d.join("cards.json")).expect("clash royale catalog cards.json");
        let tl = crdata::catalog::Timeline::load(&d.join("timeline.json")).expect("clash royale catalog timeline.json");
        (dir.to_string(), cat, tl)
    });
    if c.0 != dir {
        return Err(PyValueError::new_err(format!("clash royale: catalog {} already loaded (asked for {dir})", c.0)));
    }
    Ok(c)
}

/// One player's Clash Royale frames during a live battle — crdata's `SideFrames`, the
/// converter's builder: plays / abilities are observed in tick order (grid cells in the
/// TEAM frame, team half y < 16), a frame read at a tick holds the events before it, as the
/// converter's `raw.*` columns of one row.
#[pyclass(module = "gmsim._native", unsendable)]
struct CrFrames {
    inner: crdata::frames::SideFrames<'static>,
    cols: crdata::frames::FrameColumns,
}

fn cr_side(side: u8) -> PyResult<crdata::replay::SideId> {
    match side {
        0 => Ok(crdata::replay::SideId::Team),
        1 => Ok(crdata::replay::SideId::Opp),
        _ => Err(PyValueError::new_err(format!("side {side} is not 0 (team) / 1 (opponent)"))),
    }
}

#[pymethods]
impl CrFrames {
    #[new]
    #[pyo3(signature = (catalog, deck_ids, deck_forms, side, step_ticks=10, recent=8))]
    fn new(catalog: &str, deck_ids: Vec<u16>, deck_forms: Vec<u8>, side: u8, step_ticks: u32, recent: usize) -> PyResult<Self> {
        let c = cr_catalog(catalog)?;
        if deck_forms.len() != deck_ids.len() {
            return Err(PyValueError::new_err("deck_ids / deck_forms lengths differ"));
        }
        let deck: Vec<(u16, u8)> = deck_ids.into_iter().zip(deck_forms).collect();
        let cfg = crdata::frames::FrameConfig { step_ticks, recent };
        let inner = crdata::frames::SideFrames::new(&deck, cr_side(side)?, &c.1, &c.2, cfg)
            .ok_or_else(|| PyValueError::new_err("a deck of 8 cards"))?;
        Ok(CrFrames { inner, cols: Default::default() })
    }

    /// The deck tokens' card ids / forms (the converter's token order).
    #[getter]
    fn tok_ids(&self) -> Vec<u16> {
        self.inner.tok_ids.clone()
    }

    #[getter]
    fn tok_forms(&self) -> Vec<u8> {
        self.inner.tok_form.clone()
    }

    /// A play of `side` (0 team, 1 opponent) at `tick`, cell (gx, gy) in the team frame;
    /// False when no hand cycle explains the own plays any more.
    fn observe_play(&mut self, side: u8, card: u16, form: u8, tick: u32, gx: i8, gy: i8) -> PyResult<bool> {
        let e = crdata::replay::Event { tick, side: cr_side(side)?,
                                        kind: crdata::replay::EventKind::Play { card, form, gx, gy } };
        Ok(self.inner.observe(&e))
    }

    fn observe_ability(&mut self, side: u8, card: u16, tick: u32) -> PyResult<bool> {
        let e = crdata::replay::Event { tick, side: cr_side(side)?, kind: crdata::replay::EventKind::Ability { card } };
        Ok(self.inner.observe(&e))
    }

    /// The raw columns of the frame at `tick` (one row each; the label columns are not
    /// the builder's).
    fn frame<'py>(&mut self, py: Python<'py>, tick: u32) -> PyResult<Bound<'py, PyDict>> {
        let mut c = crdata::frames::FrameColumns::default();
        self.inner.push_frame(tick, &mut c);
        let d = PyDict::new(py);
        d.set_item("tick", c.tick)?;
        d.set_item("time_s", c.time_s)?;
        d.set_item("phase", c.phase)?;
        d.set_item("overtime", c.overtime)?;
        d.set_item("elixir", c.elixir)?;
        d.set_item("opp_elixir", c.opp_elixir)?;
        d.set_item("deck_ids", c.deck_ids)?;
        d.set_item("deck_form", c.deck_form)?;
        d.set_item("deck_kind", c.deck_kind)?;
        d.set_item("deck_cost", c.deck_cost)?;
        d.set_item("deck_in_hand", c.deck_in_hand)?;
        d.set_item("deck_next", c.deck_next)?;
        d.set_item("recent_side", c.recent_side)?;
        d.set_item("recent_card", c.recent_card)?;
        d.set_item("recent_x", c.recent_x)?;
        d.set_item("recent_y", c.recent_y)?;
        d.set_item("recent_age", c.recent_age)?;
        let _ = &mut self.cols;
        Ok(d)
    }
}

/// Clash Royale catalog facts for the env: (id, key, cost, kind) of every card, as the
/// converter reads them.
#[pyfunction]
fn cr_catalog_cards(catalog: &str) -> PyResult<Vec<(u16, String, f32, u8)>> {
    let c = cr_catalog(catalog)?;
    Ok(c.1.cards.iter().map(|e| (e.id, e.key.clone(), c.1.cost(e.id), c.1.kind(e.id) as u8)).collect())
}

/// The frame of a live Pokémon Showdown request (gmpokemon::live_frame: the converter's frame
/// of that request): `log` = the battle's sim protocol so far, `side` 0 / 1, `phase` "turn" /
/// "forced" / "preview", `format` its format id (Showdown's toID: "gen9ou").  Returns the obs
/// tensors as bytearrays (species_ids / move_ids / slot_moves / slot_item / slot_ability /
/// slot_tera_type i64, poke_feats / field_seq f32, slot_mask / slot_no_item u8, format_id i64)
/// and `slots` — the side's (nickname, species) per slot, `active` its active slot (−1: none).
#[pyfunction]
fn pokemon_live_frame<'py>(py: Python<'py>, log: &str, side: usize, phase: &str, format: &str) -> PyResult<Bound<'py, PyDict>> {
    use gmpokemon::{frame_ids, live_frame, parse_log, slot_idents, Phase, UnkNames};
    if side > 1 {
        return Err(PyValueError::new_err(format!("side {side} is not 0 / 1")));
    }
    let ph = match phase {
        "turn" => Phase::Turn,
        "forced" => Phase::Forced,
        "preview" => Phase::Preview,
        _ => return Err(PyValueError::new_err(format!("phase {phase:?}: turn / forced / preview"))),
    };
    let (rec, slots) = py.detach(|| {
        let p = parse_log(log);
        let slots = slot_idents(&p.battle, side);
        let _ = live_frame; // the same as snapshot(parse_log(log).battle, ..), parsed once here
        (gmpokemon::snapshot(&p.battle, side, ph), slots)
    });
    let f = &rec.frame;
    let ids = frame_ids(f, &mut UnkNames::default());
    let d = PyDict::new(py);
    d.set_item("species_ids", bytearray(py, &ids.species))?;
    d.set_item("move_ids", bytearray(py, &ids.moves))?;
    d.set_item("poke_feats", bytearray(py, &f.feats))?;
    d.set_item("slot_mask", bytearray(py, &f.mask.iter().map(|&b| b as u8).collect::<Vec<u8>>()))?;
    d.set_item("field_seq", bytearray(py, &f.field))?;
    d.set_item("slot_moves", bytearray(py, &ids.slot_moves))?;
    d.set_item("slot_item", bytearray(py, &ids.slot_item))?;
    d.set_item("slot_no_item", bytearray(py, &f.slot_no_item.iter().map(|&b| b as u8).collect::<Vec<u8>>()))?;
    d.set_item("slot_ability", bytearray(py, &ids.slot_ability))?;
    d.set_item("slot_tera_type", bytearray(py, &ids.slot_tera))?;
    d.set_item("format_id", gmpokemon::format_id(format))?;
    d.set_item("active", f.active)?;
    d.set_item("slots", slots)?;
    Ok(d)
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
    m.add_class::<MahjongSim>()?;
    m.add_class::<Sc2Sim>()?;
    m.add_class::<CrFrames>()?;
    m.add_function(wrap_pyfunction!(cr_catalog_cards, m)?)?;
    m.add_function(wrap_pyfunction!(pokemon_live_frame, m)?)?;
    Ok(())
}
