//! N games of one player against the built-in AI, each an SC2 process on its own thread —
//! the simulator of the crate's decisions (module docs): a command is sent to the game at
//! once and the frame stays until the window is done ("done", or [`K_MAX`] commands), then
//! the game steps [`STEP_MUL`] loops and the next window's frame is encoded.  Reward 0 until
//! the end: +1 a victory, −1 a defeat, 0 a tie or the `max_loops` timeout (truncated).

use crate::{decode, Encoder, Frame, Label, MapInfo, Vocab, Window, K_MAX, STEP_MUL};
use anyhow::{bail, Context, Result};
use sc2api::{proto, Install, Sc2};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

/// What every game of a vector shares.
#[derive(Clone, Debug)]
pub struct Config {
    pub install: PathBuf,
    /// map files (absolute, or relative to the install's Maps/)
    pub maps: Vec<String>,
    /// the player's race per env (cycled): 1 terran, 2 zerg, 3 protoss, 4 random
    pub races: Vec<i32>,
    /// the AI's race (4 random) and difficulty (1 very easy … 10 cheat insane)
    pub ai_race: i32,
    pub difficulty: i32,
    /// game loops before a game is cut (truncated, reward 0)
    pub max_loops: u32,
    /// first websocket port (env i: port + i)
    pub port: u16,
    pub tmp: PathBuf,
}

/// The result of one decision.
#[derive(Clone, Debug, Default)]
pub struct Out {
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
    /// the command reached the game as a raw action (false: done, or nothing to command)
    pub sent: bool,
    /// the game refused it (an ActionResult other than Success)
    pub refused: bool,
    pub game_loop: u32,
}

struct Game {
    sc2: Sc2,
    enc: Encoder,
    frame: Frame,
    window: Window,
    done: bool,
}

fn race_of(v: i32) -> proto::Race {
    proto::Race::try_from(v).unwrap_or(proto::Race::Random)
}

impl Game {
    fn start(sc2: Option<Sc2>, cfg: &Config, env: usize, seed: u64, voc: &Vocab) -> Result<Game> {
        let mut sc2 = match sc2 {
            Some(s) => s,
            None => {
                let inst = Install::find(&cfg.install, Some(voc.build))?;
                Sc2::launch(&inst, cfg.port + env as u16, &cfg.tmp.join(env.to_string()), None)?
            }
        };
        if cfg.maps.is_empty() {
            bail!("no maps");
        }
        let map = &cfg.maps[(seed as usize) % cfg.maps.len()];
        let race = cfg.races.get(env % cfg.races.len().max(1)).copied().unwrap_or(4);
        let difficulty = proto::Difficulty::try_from(cfg.difficulty).unwrap_or(proto::Difficulty::Medium);
        sc2.create_vs_ai(map, race_of(cfg.ai_race), difficulty, seed as u32)?;
        let player = sc2.join_game(race_of(race))?;
        let map_info = MapInfo::from_game_info(&sc2.game_info()?);
        let mut enc = Encoder::new(map_info, player);
        let o = sc2.observe()?;
        let frame = enc.encode(o.observation.as_ref().context("no observation")?, voc);
        Ok(Game { sc2, enc, frame, window: Window::default(), done: false })
    }

    fn step(&mut self, l: &Label, cfg: &Config, voc: &Vocab) -> Result<Out> {
        let mut out = Out { game_loop: self.frame.game_loop, ..Default::default() };
        if self.done {
            out.done = true;
            return Ok(out);
        }
        if l.action_type >= 2 && self.window.n < K_MAX {
            if let Some(a) = decode(l, &self.frame, &self.enc, voc) {
                let res = self.sc2.act(vec![a])?;
                out.sent = true;
                out.refused = res.iter().any(|&r| r != proto::ActionResult::Success as i32);
            }
            self.window = self.window.after(l);
            if self.window.n < K_MAX {
                return Ok(out);
            }
        }
        // the window is done: the game moves on
        self.sc2.step(STEP_MUL)?;
        let o = self.sc2.observe()?;
        if !o.player_result.is_empty() {
            let me = self.enc.player_id;
            let r = o.player_result.iter().find(|r| r.player_id == Some(me)).map(|r| r.result()).unwrap_or(proto::Result::Tie);
            out.reward = match r {
                proto::Result::Victory => 1.0,
                proto::Result::Defeat => -1.0,
                _ => 0.0,
            };
            out.done = true;
            self.done = true;
            return Ok(out);
        }
        let obs = o.observation.as_ref().context("no observation")?;
        self.frame = self.enc.encode(obs, voc);
        self.window = Window::default();
        out.game_loop = self.frame.game_loop;
        if self.frame.game_loop >= cfg.max_loops {
            out.done = true;
            out.truncated = true;
            self.done = true;
        }
        Ok(out)
    }

    /// Finish a game still running (truncated, or reset early) so the process can take the
    /// next one.
    fn leave(&mut self) {
        if !self.done {
            let _ = self.sc2.end_game();
        }
    }
}

enum Msg {
    Reset(u64),
    Step(Label),
}

type Reply = Result<(Frame, Window, Out), String>;

struct Worker {
    tx: mpsc::Sender<Msg>,
    // behind a mutex: the vector is shared with Python threads (Sync)
    rx: std::sync::Mutex<mpsc::Receiver<Reply>>,
    _thread: std::thread::JoinHandle<()>,
}

impl Worker {
    fn recv(&self, env: usize) -> Result<Reply> {
        self.rx.lock().map_err(|_| anyhow::anyhow!("env {env}: poisoned"))?.recv()
            .map_err(|_| anyhow::anyhow!("env {env}: worker gone"))
    }
}

fn spawn(env: usize, cfg: Arc<Config>, voc: Arc<Vocab>) -> Worker {
    let (tx, wrx) = mpsc::channel::<Msg>();
    let (wtx, rx) = mpsc::channel::<Reply>();
    let thread = std::thread::spawn(move || {
        let mut game: Option<Game> = None;
        for msg in wrx {
            let reply = (|| -> Result<(Frame, Window, Out)> {
                match msg {
                    Msg::Reset(seed) => {
                        // a new game on the same process when it lives, else a fresh one
                        let sc2 = game.take().map(|mut g| {
                            g.leave();
                            g.sc2
                        });
                        let g = match Game::start(sc2, &cfg, env, seed, &voc) {
                            Ok(g) => g,
                            Err(_) => Game::start(None, &cfg, env, seed, &voc)?,
                        };
                        let r = (g.frame.clone(), g.window.clone(), Out { game_loop: g.frame.game_loop, ..Default::default() });
                        game = Some(g);
                        Ok(r)
                    }
                    Msg::Step(l) => {
                        let g = game.as_mut().context("step before reset")?;
                        let out = g.step(&l, &cfg, &voc)?;
                        Ok((g.frame.clone(), g.window.clone(), out))
                    }
                }
            })();
            if reply.is_err() {
                game = None; // the process may be gone: the next reset starts afresh
            }
            if wtx.send(reply.map_err(|e| format!("{e:#}"))).is_err() {
                return;
            }
        }
    });
    Worker { tx, rx: std::sync::Mutex::new(rx), _thread: thread }
}

/// The per-decision result of every env (after an autoreset, the new game's first frame).
#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub rewards: Vec<f32>,
    pub done: Vec<u8>,
    pub truncated: Vec<u8>,
    pub sent: Vec<u8>,
    pub refused: Vec<u8>,
    pub episode_len: Vec<i64>,
    pub game_loop: Vec<i64>,
}

pub struct Sc2Vec {
    workers: Vec<Worker>,
    pub frames: Vec<(Frame, Window)>,
    pub autoreset: bool,
    seeds: Vec<u64>,
    episodes: Vec<u64>,
    lens: Vec<i64>,
    pub failures: usize,
    pub last_error: Option<String>,
    pub voc: Arc<Vocab>,
}

impl Sc2Vec {
    pub fn new(n: usize, cfg: Config, voc: Vocab, seeds: &[u64], autoreset: bool) -> Result<Sc2Vec> {
        let (cfg, voc) = (Arc::new(cfg), Arc::new(voc));
        let workers: Vec<Worker> = (0..n).map(|i| spawn(i, cfg.clone(), voc.clone())).collect();
        let seeds: Vec<u64> = (0..n).map(|i| seeds.get(i).copied().unwrap_or(i as u64)).collect();
        let mut v = Sc2Vec { workers, frames: vec![], autoreset, seeds, episodes: vec![0; n], lens: vec![0; n], failures: 0,
                             last_error: None, voc };
        let ids: Vec<usize> = (0..n).collect();
        v.frames = vec![(Frame::empty(&v.voc), Window::default()); n];
        v.reset(&ids)?;
        Ok(v)
    }

    pub fn n_envs(&self) -> usize {
        self.workers.len()
    }

    fn seed_of(&self, i: usize) -> u64 {
        self.seeds[i].wrapping_mul(1_000_003).wrapping_add(self.episodes[i])
    }

    /// Start new games in `ids` (all at once: the processes load in parallel).
    pub fn reset(&mut self, ids: &[usize]) -> Result<()> {
        for &i in ids {
            let s = self.seed_of(i);
            self.workers[i].tx.send(Msg::Reset(s)).map_err(|_| anyhow::anyhow!("env {i}: worker gone"))?;
        }
        for &i in ids {
            let r = self.workers[i].recv(i)?;
            match r {
                Ok((f, w, _)) => self.frames[i] = (f, w),
                Err(e) => bail!("env {i}: reset: {e}"),
            }
            self.episodes[i] += 1;
            self.lens[i] = 0;
        }
        Ok(())
    }

    /// One decision per env (in parallel).  A game that fails is restarted (counted, the
    /// env reports done + truncated); with `autoreset`, a finished game is replaced.
    pub fn step(&mut self, labels: &[Label]) -> Result<StepInfo> {
        let n = self.n_envs();
        if labels.len() != n {
            bail!("{} labels for {n} envs", labels.len());
        }
        for (i, l) in labels.iter().enumerate() {
            self.workers[i].tx.send(Msg::Step(l.clone())).map_err(|_| anyhow::anyhow!("env {i}: worker gone"))?;
        }
        let mut info = StepInfo { rewards: vec![0.0; n], done: vec![0; n], truncated: vec![0; n], sent: vec![0; n],
                                  refused: vec![0; n], episode_len: vec![0; n], game_loop: vec![0; n] };
        let mut again = vec![];
        for i in 0..n {
            let r = self.workers[i].recv(i)?;
            self.lens[i] += 1;
            match r {
                Ok((f, w, out)) => {
                    info.rewards[i] = out.reward;
                    info.done[i] = out.done as u8;
                    info.truncated[i] = out.truncated as u8;
                    info.sent[i] = out.sent as u8;
                    info.refused[i] = out.refused as u8;
                    info.game_loop[i] = out.game_loop as i64;
                    self.frames[i] = (f, w);
                }
                Err(e) => {
                    self.failures += 1;
                    self.last_error = Some(format!("env {i}: {e}"));
                    info.done[i] = 1;
                    info.truncated[i] = 1;
                }
            }
            info.episode_len[i] = self.lens[i];
            if info.done[i] == 1 && self.autoreset {
                again.push(i);
            }
        }
        if !again.is_empty() {
            self.reset(&again)?;
        }
        Ok(info)
    }
}

impl Frame {
    /// A frame of no units (before the first reset).
    pub fn empty(voc: &Vocab) -> Frame {
        use crate::*;
        Frame { units: vec![0; U * F], unit_xy: vec![0; U * 2], unit_type: vec![0; U], unit_order: vec![0; U],
                unit_buffs: vec![0; U * BUFFS], unit_mask: vec![false; U], unit_own: vec![false; U],
                map_bits: vec![0; MAP_BITS * MAP * MAP], globals: vec![0.0; G], upgrades: vec![0; voc.upgrades.len()],
                tags: vec![0; U], game_loop: 0 }
    }
}

/// The batch buffers of every env's current decision (the shard tensors' layout).
#[derive(Clone, Debug, Default)]
pub struct Buffers {
    pub units: Vec<u8>,
    pub unit_xy: Vec<u16>,
    pub unit_type: Vec<u16>,
    pub unit_order: Vec<u16>,
    pub unit_buffs: Vec<u16>,
    pub unit_mask: Vec<u8>,
    pub unit_own: Vec<u8>,
    pub unit_commanded: Vec<u8>,
    pub map_bits: Vec<u8>,
    pub globals: Vec<f32>,
    pub upgrades: Vec<u8>,
    pub last_action: Vec<i64>,
    pub window_start: Vec<u8>,
}

impl Sc2Vec {
    pub fn observe(&self) -> Buffers {
        let mut b = Buffers::default();
        for (f, w) in &self.frames {
            b.units.extend_from_slice(&f.units);
            b.unit_xy.extend_from_slice(&f.unit_xy);
            b.unit_type.extend_from_slice(&f.unit_type);
            b.unit_order.extend_from_slice(&f.unit_order);
            b.unit_buffs.extend_from_slice(&f.unit_buffs);
            b.unit_mask.extend(f.unit_mask.iter().map(|&m| m as u8));
            b.unit_own.extend(f.unit_own.iter().map(|&m| m as u8));
            b.unit_commanded.extend(w.commanded.iter().map(|&m| m as u8));
            b.map_bits.extend_from_slice(&f.map_bits);
            let mut g = f.globals.clone();
            g[crate::global::WINDOW_CMDS] = w.window_cmds();
            b.globals.extend(g);
            b.upgrades.extend_from_slice(&f.upgrades);
            b.last_action.push(w.last_action);
            b.window_start.push((w.n == 0) as u8);
        }
        b
    }
}
