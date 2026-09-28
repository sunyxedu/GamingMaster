//! StarCraft II API client: one headless SC2 process (the Linux client), spoken to in the
//! game's own protobuf protocol (Blizzard s2client-proto, vendored in `proto/`, MIT) over
//! its websocket `ws://127.0.0.1:<port>/sc2api`.
//!
//! Launch mirrors pysc2 on Linux: `Versions/Base<build>/SC2_x64 -listen 127.0.0.1 -port P
//! -dataDir <root>/ -tempDir <tmp>/` with `<root>/Libs` on `LD_LIBRARY_PATH`; raw-interface
//! use needs no rendering (no -eglpath).  A replay replays only on the client build that
//! recorded it (`Install::find` picks the build).

pub mod proto {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sc2api_protocol.rs"));
}

use anyhow::{anyhow, bail, Context, Result};
use prost::Message;
use proto::{request, response, Request, Response};
use std::fs::File;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tungstenite::{client, protocol::WebSocket, Message as WsMessage};

/// An SC2 install: `<root>/Versions/Base<build>/SC2_x64` (+ `<root>/Libs`, `Maps`).
#[derive(Debug, Clone)]
pub struct Install {
    pub root: PathBuf,
    pub build: u32,
}

impl Install {
    /// The install at `root`; `build` = the given one, or the only / highest
    /// `Versions/Base<build>` present.
    pub fn find(root: &Path, build: Option<u32>) -> Result<Self> {
        let mut builds = vec![];
        for e in std::fs::read_dir(root.join("Versions")).with_context(|| format!("{}/Versions", root.display()))? {
            let name = e?.file_name().to_string_lossy().to_string();
            if let Some(b) = name.strip_prefix("Base").and_then(|s| s.parse::<u32>().ok()) {
                builds.push(b);
            }
        }
        builds.sort();
        let build = match build {
            Some(b) if builds.contains(&b) => b,
            Some(b) => bail!("SC2 build {b} not installed under {} (have {builds:?})", root.display()),
            None => *builds.last().ok_or_else(|| anyhow!("no Versions/Base* under {}", root.display()))?,
        };
        Ok(Self { root: root.to_path_buf(), build })
    }

    pub fn exe(&self) -> PathBuf {
        self.root.join("Versions").join(format!("Base{}", self.build)).join("SC2_x64")
    }
}

/// A running SC2 process and its websocket.
pub struct Sc2 {
    child: Child,
    ws: WebSocket<TcpStream>,
    pub install: Install,
}

impl Sc2 {
    /// Start SC2 listening on `port` (its temp files in `tmp`) and connect (≤ 120 s).
    pub fn launch(install: &Install, port: u16, tmp: &Path, log: Option<&Path>) -> Result<Self> {
        std::fs::create_dir_all(tmp)?;
        let libs = install.root.join("Libs");
        let ld = match std::env::var("LD_LIBRARY_PATH") {
            Ok(v) if !v.is_empty() => format!("{v}:{}/", libs.display()),
            _ => format!("{}/", libs.display()),
        };
        let (out, err) = match log {
            Some(p) => {
                let f = File::options().create(true).append(true).open(p)?;
                (Stdio::from(f.try_clone()?), Stdio::from(f))
            }
            None => (Stdio::null(), Stdio::null()),
        };
        let mut child = Command::new(install.exe())
            .args(["-listen", "127.0.0.1", "-port", &port.to_string(),
                   "-dataDir", &format!("{}/", install.root.display()),
                   "-tempDir", &format!("{}/", tmp.display())])
            .env("LD_LIBRARY_PATH", ld)
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .spawn()
            .with_context(|| format!("launch {}", install.exe().display()))?;
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(status) = child.try_wait()? {
                bail!("SC2 exited before accepting a connection ({status})");
            }
            let err = match TcpStream::connect(("127.0.0.1", port)) {
                Ok(stream) => {
                    stream.set_nodelay(true)?;
                    match client(format!("ws://127.0.0.1:{port}/sc2api"), stream) {
                        Ok((ws, _)) => return Ok(Self { child, ws, install: install.clone() }),
                        Err(e) => e.to_string(),
                    }
                }
                Err(e) => e.to_string(),
            };
            if Instant::now() > deadline {
                let _ = child.kill();
                bail!("SC2 on port {port} did not accept a websocket in 120 s: {err}");
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// One request → its response (errors in the response become Err).
    pub fn request(&mut self, req: request::Request) -> Result<response::Response> {
        let msg = Request { request: Some(req), ..Default::default() };
        self.ws.send(WsMessage::Binary(msg.encode_to_vec().into()))?;
        loop {
            match self.ws.read()? {
                WsMessage::Binary(b) => {
                    let r = Response::decode(b.as_ref()).context("decode an SC2 response")?;
                    if !r.error.is_empty() {
                        bail!("SC2 error: {}", r.error.join("; "));
                    }
                    return r.response.ok_or_else(|| anyhow!("SC2: response without a body"));
                }
                WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_) => continue,
                other => bail!("SC2: unexpected websocket message {other:?}"),
            }
        }
    }

    pub fn replay_info(&mut self, replay: &Path) -> Result<proto::ResponseReplayInfo> {
        let req = proto::RequestReplayInfo {
            replay: Some(proto::request_replay_info::Replay::ReplayPath(replay.display().to_string())),
            download_data: Some(false),
        };
        match self.request(request::Request::ReplayInfo(req))? {
            response::Response::ReplayInfo(r) => {
                if r.error.is_some() {
                    bail!("replay info: {:?} {}", r.error(), r.error_details());
                }
                Ok(r)
            }
            other => bail!("replay_info: unexpected response {other:?}"),
        }
    }

    /// Start a replay from `observed_player`'s view with the raw interface (+ score).
    pub fn start_replay(&mut self, replay: &Path, observed_player: i32, map_data: Option<Vec<u8>>) -> Result<()> {
        let options = proto::InterfaceOptions {
            raw: Some(true),
            score: Some(true),
            show_cloaked: Some(false),
            ..Default::default()
        };
        let req = proto::RequestStartReplay {
            replay: Some(proto::request_start_replay::Replay::ReplayPath(replay.display().to_string())),
            map_data,
            observed_player_id: Some(observed_player),
            options: Some(options),
            disable_fog: Some(false),
            realtime: Some(false),
            ..Default::default()
        };
        match self.request(request::Request::StartReplay(req))? {
            response::Response::StartReplay(r) => {
                if r.error.is_some() {
                    bail!("start replay: {:?} {}", r.error(), r.error_details());
                }
                Ok(())
            }
            other => bail!("start_replay: unexpected response {other:?}"),
        }
    }

    pub fn game_info(&mut self) -> Result<proto::ResponseGameInfo> {
        match self.request(request::Request::GameInfo(proto::RequestGameInfo {}))? {
            response::Response::GameInfo(r) => Ok(r),
            other => bail!("game_info: unexpected response {other:?}"),
        }
    }

    pub fn step(&mut self, loops: u32) -> Result<()> {
        match self.request(request::Request::Step(proto::RequestStep { count: Some(loops) }))? {
            response::Response::Step(_) => Ok(()),
            other => bail!("step: unexpected response {other:?}"),
        }
    }

    pub fn observe(&mut self) -> Result<proto::ResponseObservation> {
        let req = proto::RequestObservation { disable_fog: Some(false), game_loop: None };
        match self.request(request::Request::Observation(req))? {
            response::Response::Observation(r) => Ok(r),
            other => bail!("observe: unexpected response {other:?}"),
        }
    }

    /// The game's static data (abilities, unit types, upgrades, buffs): available once a game
    /// or replay runs.
    pub fn data(&mut self) -> Result<proto::ResponseData> {
        let req = proto::RequestData { ability_id: Some(true), unit_type_id: Some(true), upgrade_id: Some(true),
                                       buff_id: Some(true), effect_id: Some(false) };
        match self.request(request::Request::Data(req))? {
            response::Response::Data(r) => Ok(r),
            other => bail!("data: unexpected response {other:?}"),
        }
    }

    /// A single-player game on the local `map_path` (relative to Maps/ or absolute) against
    /// the built-in AI of `ai_race` / `difficulty`, seeded; then `join_game` as `race`.
    pub fn create_vs_ai(&mut self, map_path: &str, ai_race: proto::Race, difficulty: proto::Difficulty, seed: u32) -> Result<()> {
        let req = proto::RequestCreateGame {
            map: Some(proto::request_create_game::Map::LocalMap(proto::LocalMap { map_path: Some(map_path.to_string()), map_data: None })),
            player_setup: vec![
                proto::PlayerSetup { r#type: Some(proto::PlayerType::Participant as i32), ..Default::default() },
                proto::PlayerSetup { r#type: Some(proto::PlayerType::Computer as i32), race: Some(ai_race as i32),
                                     difficulty: Some(difficulty as i32), ..Default::default() },
            ],
            disable_fog: Some(false),
            random_seed: Some(seed),
            realtime: Some(false),
        };
        match self.request(request::Request::CreateGame(req))? {
            response::Response::CreateGame(r) => {
                if r.error.is_some() {
                    bail!("create game: {:?} {}", r.error(), r.error_details());
                }
                Ok(())
            }
            other => bail!("create_game: unexpected response {other:?}"),
        }
    }

    /// Join the created game as a participant of `race` with the raw interface (+ score);
    /// returns our player id.
    pub fn join_game(&mut self, race: proto::Race) -> Result<u32> {
        let options = proto::InterfaceOptions { raw: Some(true), score: Some(true), show_cloaked: Some(false), ..Default::default() };
        let req = proto::RequestJoinGame {
            participation: Some(proto::request_join_game::Participation::Race(race as i32)),
            options: Some(options),
            ..Default::default()
        };
        match self.request(request::Request::JoinGame(req))? {
            response::Response::JoinGame(r) => {
                if r.error.is_some() {
                    bail!("join game: {:?} {}", r.error(), r.error_details());
                }
                Ok(r.player_id.unwrap_or(1))
            }
            other => bail!("join_game: unexpected response {other:?}"),
        }
    }

    /// Send raw actions (the per-action results: Success or an error code).
    pub fn act(&mut self, actions: Vec<proto::Action>) -> Result<Vec<i32>> {
        match self.request(request::Request::Action(proto::RequestAction { actions }))? {
            response::Response::Action(r) => Ok(r.result),
            other => bail!("action: unexpected response {other:?}"),
        }
    }

    /// End the running single-player game (a debug surrender), so the process takes a new
    /// `create_game`; then step once for the game to register it.
    pub fn end_game(&mut self) -> Result<()> {
        let cmd = proto::DebugCommand { command: Some(proto::debug_command::Command::EndGame(proto::DebugEndGame {
            end_result: Some(proto::debug_end_game::EndResult::Surrender as i32) })) };
        match self.request(request::Request::Debug(proto::RequestDebug { debug: vec![cmd] }))? {
            response::Response::Debug(_) => {}
            other => bail!("debug: unexpected response {other:?}"),
        }
        let _ = self.step(1);
        Ok(())
    }

    /// Leave (the process exits); also on drop.
    pub fn quit(&mut self) {
        let _ = self.request(request::Request::Quit(proto::RequestQuit {}));
    }
}

impl Drop for Sc2 {
    fn drop(&mut self) {
        self.quit();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip() {
        let req = Request {
            request: Some(request::Request::Step(proto::RequestStep { count: Some(8) })),
            ..Default::default()
        };
        let back = Request::decode(req.encode_to_vec().as_slice()).unwrap();
        match back.request {
            Some(request::Request::Step(s)) => assert_eq!(s.count, Some(8)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn install_picks_the_highest_build() {
        let d = std::env::temp_dir().join(format!("sc2api-test-{}", std::process::id()));
        for b in ["Base55958", "Base75689"] {
            std::fs::create_dir_all(d.join("Versions").join(b)).unwrap();
        }
        let i = Install::find(&d, None).unwrap();
        assert_eq!(i.build, 75689);
        assert!(i.exe().ends_with("Versions/Base75689/SC2_x64"));
        assert_eq!(Install::find(&d, Some(55958)).unwrap().build, 55958);
        assert!(Install::find(&d, Some(1)).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
