//! Lux AI Season 3: Kaggle episode replays (JSON) → shards for the "lux" adapter.
//!
//! One sample = one (episode, agent, match): matches reset at `match_steps == 0`;
//! observation t pairs with the action stored at steps[t+1]; both agents emitted.
//!
//!   obs.map_seq     [T,12,24,24] f32  tile empty/nebula/asteroid, energy/20 (visible),
//!                                     sensor mask, relic, own/opp unit count/4,
//!                                     own/opp energy/400, match progress, ones
//!   obs.units_seq   [T,16,16] f32     [x/24, y/24, energy/400, 1, 0…]
//!   obs.unit_mask   [T,16] bool
//!   obs.globals_seq [T,11] f32        points/500 ×2, wins/5 ×2, match_steps/101, steps/506,
//!                                     unit_move_cost/5, unit_sap_cost/50, unit_sap_range/8,
//!                                     unit_sensor_range/8, env_cfg_known
//!   tgt.unit_actions [T,16] i64       action type 0..5
//!   tgt.sap_offset   [T,16] i64       sap units: (dy+7)*15 + (dx+7) for |dx|,|dy| ≤ 7, else −1
//!   tgt.unit_mask    [T,16] bool
//!   tgt.value        [T] f32          match result for the agent
//!
//! converter_version 2 (shard meta {"global_features": 11, "sap_range": 7}) added the
//! env-parameter globals and the sap target. The env parameters come from
//! steps[0][agent].info.env_cfg, then configuration.env_cfg; a key found in neither takes
//! the Lux S3 default (move 2, sap cost 10, sap range 4, sensor 2), and env_cfg_known = 1
//! only when all four keys came from one source (sample meta `env_cfg`). The sap offset
//! window is SAP_R = 7, the largest unit_sap_range Lux S3 draws (3..=7); a sap outside it
//! gets −1 (stat `sap_out_of_range`).
//!
//! JSON map arrays are indexed [x][y]; planes are [y][x].

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const H: usize = 24;
const W: usize = 24;
const C: usize = 12;
const U: usize = 16;
const F: usize = 16;
const G: usize = 11;
const CONVERTER_VERSION: u32 = 2;
/// Sap target window: offsets (dx, dy) with |dx|, |dy| ≤ SAP_R (Lux S3 unit_sap_range ∈ 3..=7).
const SAP_R: i64 = 7;
const SAP_ACTION: i64 = 5;
/// (env_cfg key, Lux S3 default, normaliser)
const ENV_KEYS: [(&str, f64, f64); 4] =
    [("unit_move_cost", 2.0, 5.0), ("unit_sap_cost", 10.0, 50.0), ("unit_sap_range", 4.0, 8.0), ("unit_sensor_range", 2.0, 8.0)];

/// Env parameters of an episode (as seen by one agent).
#[derive(Debug, Clone, PartialEq)]
struct EnvCfg {
    values: [f64; 4],
    known: bool,
}

impl EnvCfg {
    fn parse(ep: &Value, agent: usize) -> Self {
        let sources = [&ep["steps"][0][agent]["info"]["env_cfg"], &ep["configuration"]["env_cfg"]];
        let mut values = [0.0; 4];
        let mut from: [Option<usize>; 4] = [None; 4];
        for (k, (key, default, _)) in ENV_KEYS.iter().enumerate() {
            values[k] = *default;
            for (si, src) in sources.iter().enumerate() {
                if let Some(v) = src.get(key).and_then(|v| v.as_f64()) {
                    values[k] = v;
                    from[k] = Some(si);
                    break;
                }
            }
        }
        let known = from[0].is_some() && from.iter().all(|f| *f == from[0]);
        EnvCfg { values, known }
    }

    fn globals(&self) -> [f32; 5] {
        let mut g = [0f32; 5];
        for (k, (_, _, norm)) in ENV_KEYS.iter().enumerate() {
            g[k] = (self.values[k] / norm) as f32;
        }
        g[4] = self.known as u8 as f32;
        g
    }

    fn meta(&self) -> Value {
        let mut m = serde_json::Map::new();
        for (k, (key, _, _)) in ENV_KEYS.iter().enumerate() {
            m.insert(key.to_string(), json!(self.values[k]));
        }
        m.insert("known".into(), json!(self.known));
        Value::Object(m)
    }
}

/// Sap target class of an env sap action [5, dx, dy]: (dy+R)(2R+1) + (dx+R), None outside the window.
fn sap_offset(dx: i64, dy: i64) -> Option<i64> {
    (dx.abs() <= SAP_R && dy.abs() <= SAP_R).then(|| (dy + SAP_R) * (2 * SAP_R + 1) + (dx + SAP_R))
}

fn parse_obs(entry: &Value) -> Option<Value> {
    let raw = entry.get("observation")?.get("obs")?;
    match raw {
        Value::String(s) => serde_json::from_str(s).ok(),
        Value::Object(_) => Some(raw.clone()),
        _ => None,
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
fn i(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| v.as_f64().map(|x| x as i64).unwrap_or(0))
}
fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(false),
        Value::Null => false,
        _ => true,
    }
}

/// [x][y] JSON grid → value at (y, x)
fn grid(o: &Value, x: usize, y: usize) -> &Value {
    &o[x][y]
}

fn build_frame(o: &Value, team: usize, env: &EnvCfg, maps: &mut Vec<f32>, units: &mut Vec<f32>, mask: &mut Vec<bool>,
               globs: &mut Vec<f32>) {
    let opp = 1 - team;
    let mut p = vec![0f32; C * H * W];
    let idx = |c: usize, y: usize, x: usize| c * H * W + y * W + x;
    let tt = &o["map_features"]["tile_type"];
    let en = &o["map_features"]["energy"];
    let sm = &o["sensor_mask"];
    for y in 0..H {
        for x in 0..W {
            let t = i(grid(tt, x, y));
            p[idx(0, y, x)] = (t == 0) as u8 as f32;
            p[idx(1, y, x)] = (t == 1) as u8 as f32;
            p[idx(2, y, x)] = (t == 2) as u8 as f32;
            let vis = truthy(grid(sm, x, y));
            // np.where(sm, energy(float32)/20.0, 0.0): float32 energy / python float → float64, stored float32
            let e32 = f(grid(en, x, y)) as f32;
            p[idx(3, y, x)] = if vis { (e32 as f64 / 20.0) as f32 } else { 0.0 };
            p[idx(4, y, x)] = vis as u8 as f32;
        }
    }
    if let (Some(nodes), Some(nmask)) = (o["relic_nodes"].as_array(), o["relic_nodes_mask"].as_array()) {
        for (n, m) in nodes.iter().zip(nmask) {
            let (x, y) = (i(&n[0]), i(&n[1]));
            if truthy(m) && (0..W as i64).contains(&x) && (0..H as i64).contains(&y) {
                p[idx(5, y as usize, x as usize)] = 1.0;
            }
        }
    }
    let pos = &o["units"]["position"];
    let ens = &o["units"]["energy"];
    let um = &o["units_mask"];
    for (side, cc, ec) in [(team, 6, 8), (opp, 7, 9)] {
        let n = um[side].as_array().map(|a| a.len()).unwrap_or(0);
        for k in 0..n {
            if !truthy(&um[side][k]) {
                continue;
            }
            let (x, y) = (i(&pos[side][k][0]), i(&pos[side][k][1]));
            if (0..W as i64).contains(&x) && (0..H as i64).contains(&y) {
                p[idx(cc, y as usize, x as usize)] += (1.0f64 / 4.0) as f32;
                p[idx(ec, y as usize, x as usize)] += (f(&ens[side][k]).max(0.0) / 400.0) as f32;
            }
        }
    }
    let ms = f(&o["match_steps"]);
    for y in 0..H {
        for x in 0..W {
            p[idx(10, y, x)] = (ms / 100.0) as f32;
            p[idx(11, y, x)] = 1.0;
        }
    }
    maps.extend_from_slice(&p);
    let mut uf = vec![0f32; U * F];
    let mut umask = vec![false; U];
    let n = um[team].as_array().map(|a| a.len()).unwrap_or(0).min(U);
    for k in 0..n {
        if !truthy(&um[team][k]) {
            continue;
        }
        umask[k] = true;
        uf[k * F] = (f(&pos[team][k][0]) / W as f64) as f32;
        uf[k * F + 1] = (f(&pos[team][k][1]) / H as f64) as f32;
        uf[k * F + 2] = (f(&ens[team][k]) / 400.0) as f32;
        uf[k * F + 3] = 1.0;
    }
    units.extend_from_slice(&uf);
    mask.extend_from_slice(&umask);
    let pts = &o["team_points"];
    let wins = &o["team_wins"];
    globs.extend_from_slice(&[
        (f(&pts[team]) / 500.0) as f32,
        (f(&pts[opp]) / 500.0) as f32,
        (f(&wins[team]) / 5.0) as f32,
        (f(&wins[opp]) / 5.0) as f32,
        (ms / 101.0) as f32,
        (f(&o["steps"]) / 506.0) as f32,
    ]);
    globs.extend_from_slice(&env.globals());
}

fn actions_at<'a>(steps: &'a [Value], t: usize, agent: usize) -> Option<&'a Vec<Value>> {
    let a = steps.get(t)?.get(agent)?.get("action")?;
    a.as_array().filter(|v| !v.is_empty())
}

fn match_value(parsed: &[Option<Value>], a: usize, b: usize, team: usize) -> f32 {
    if let (Some(s), Some(Some(e))) = (&parsed[a], parsed.get(b)) {
        let dw_own = f(&e["team_wins"][team]) - f(&s["team_wins"][team]);
        let dw_opp = f(&e["team_wins"][1 - team]) - f(&s["team_wins"][1 - team]);
        if dw_own != dw_opp {
            return if dw_own > dw_opp { 1.0 } else { -1.0 };
        }
    }
    for t in (a..b).rev() {
        if let Some(o) = &parsed[t] {
            let d = f(&o["team_points"][team]) - f(&o["team_points"][1 - team]);
            return d.signum() as f32 * (d != 0.0) as u8 as f32;
        }
    }
    0.0
}

/// (samples, sap actions outside the SAP_R window)
fn episode_samples(ep: &Value, source: &str) -> (Vec<Sample>, usize) {
    let mut sap_out = 0usize;
    let Some(steps) = ep["steps"].as_array() else { return (Vec::new(), sap_out) };
    let episode_id = ep.get("id").cloned().unwrap_or_else(|| json!(source));
    let n_agents = steps.first().and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0).min(2);
    let mut out = Vec::new();
    for agent in 0..n_agents {
        let parsed: Vec<Option<Value>> = steps.iter().map(|s| parse_obs(&s[agent])).collect();
        if parsed.iter().all(|o| o.is_none()) {
            continue;
        }
        let env = EnvCfg::parse(ep, agent);
        let player = steps[0][agent]["observation"]["player"].as_str().unwrap_or("");
        let team = player.strip_prefix("player_").and_then(|s| s.parse::<usize>().ok()).unwrap_or(agent);
        let bounds: Vec<usize> = parsed.iter().enumerate().filter(|(_, o)| o.as_ref().map(|o| f(&o["match_steps"]) == 0.0).unwrap_or(false)).map(|(t, _)| t).collect();
        let mut ends = bounds[1..].to_vec();
        ends.push(steps.len());
        for (mi, (&a, &b)) in bounds.iter().zip(ends.iter()).enumerate() {
            let (mut maps, mut units, mut masks, mut globs, mut acts, mut saps) = (vec![], vec![], vec![], vec![], vec![], vec![]);
            let mut t_len = 0;
            for t in a..b {
                let (Some(o), Some(action)) = (&parsed[t], actions_at(steps, t + 1, agent)) else { break };
                build_frame(o, team, &env, &mut maps, &mut units, &mut masks, &mut globs);
                let mut ua = vec![0i64; U];
                let mut so = vec![-1i64; U];
                for (slot, a) in action.iter().take(U).enumerate() {
                    ua[slot] = i(&a[0]).clamp(0, 5);
                    if ua[slot] == SAP_ACTION {
                        if let Some(raw) = a.as_array().filter(|r| r.len() >= 3) {
                            match sap_offset(i(&raw[1]), i(&raw[2])) {
                                Some(o) => so[slot] = o,
                                None => sap_out += 1,
                            }
                        }
                    }
                }
                acts.extend_from_slice(&ua);
                saps.extend_from_slice(&so);
                t_len += 1;
            }
            if t_len == 0 {
                continue;
            }
            let value = match_value(&parsed, a, b, team);
            out.push(Sample::new(json!({"source": source, "episode_id": episode_id, "agent": agent, "team": team,
                                         "match_index": mi, "step_range": [a, a + t_len], "match_value": value,
                                         "env_cfg": env.meta()}))
                .f32("obs.map_seq", &[C, H, W], maps)
                .f32("obs.units_seq", &[U, F], units)
                .bool("obs.unit_mask", &[U], masks.clone())
                .f32("obs.globals_seq", &[G], globs)
                .i64("tgt.unit_actions", &[U], acts)
                .i64("tgt.sap_offset", &[U], saps)
                .bool("tgt.unit_mask", &[U], masks)
                .f32("tgt.value", &[], vec![value; t_len]));
        }
    }
    (out, sap_out)
}

const KAGGLE: &str = "https://api.kaggle.com/v1/competitions.CompetitionApiService";

fn kaggle_token() -> Result<String> {
    if let Ok(t) = std::env::var("KAGGLE_API_TOKEN") {
        return Ok(t.trim().to_string());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let p = Path::new(&home).join(".kaggle").join("access_token");
    Ok(std::fs::read_to_string(&p).map_err(|_| anyhow::anyhow!("set KAGGLE_API_TOKEN or create ~/.kaggle/access_token"))?.trim().to_string())
}

fn kaggle_call(client: &reqwest::blocking::Client, token: &str, method: &str, body: Value) -> Result<reqwest::blocking::Response> {
    Ok(client.post(format!("{KAGGLE}/{method}")).bearer_auth(token).json(&body).send()?.error_for_status()?)
}

/// High-quality episodes: both agents belong to the top `top_teams` of the final
/// leaderboard (and to their public submissions). Resumable; replays are ~15 MB each.
pub fn fetch(out: &Path, competition: &str, top_teams: usize, max_episodes: usize, sleep_ms: u64) -> Result<()> {
    let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build()?;
    let token = kaggle_token()?;
    std::fs::create_dir_all(out)?;
    let lb: Value = kaggle_call(&client, &token, "GetLeaderboard", json!({"competitionName": competition}))?.json()?;
    let teams: Vec<(i64, String, String)> = lb["submissions"].as_array().cloned().unwrap_or_default().iter().take(top_teams)
        .map(|t| (t["teamId"].as_i64().unwrap_or(0), t["teamName"].as_str().unwrap_or("").to_string(), t["score"].as_str().unwrap_or("").to_string()))
        .collect();
    let top: std::collections::HashSet<i64> = teams.iter().map(|t| t.0).collect();
    eprintln!("top {} teams: {}", teams.len(), teams.iter().map(|t| format!("{} ({})", t.1, t.2)).collect::<Vec<_>>().join(", "));
    let mut seen = std::collections::HashSet::new();
    let mut manifest = Vec::new();
    let mut downloaded = 0usize;
    'teams: for (team, _, _) in &teams {
        let subs: Value = kaggle_call(&client, &token, "ListTeamPublicSubmissions", json!({"teamId": team}))?.json()?;
        for sub in subs["submissions"].as_array().cloned().unwrap_or_default() {
            let eps: Value = kaggle_call(&client, &token, "ListSubmissionEpisodes", json!({"submissionId": sub["id"]}))?.json()?;
            for ep in eps["episodes"].as_array().cloned().unwrap_or_default() {
                let id = ep["id"].as_i64().unwrap_or(0);
                let agents = ep["agents"].as_array().cloned().unwrap_or_default();
                let both_top = agents.len() == 2 && agents.iter().all(|a| top.contains(&a["teamId"].as_i64().unwrap_or(-1)));
                if ep["state"] != "COMPLETED" || !both_top || !seen.insert(id) {
                    continue;
                }
                manifest.push(json!({"id": id, "agents": agents}));
                let dest = out.join(format!("episode-{id}-replay.json"));
                if !dest.exists() {
                    let mut resp = kaggle_call(&client, &token, "GetEpisodeReplay", json!({"episodeId": id}))?;
                    let tmp = dest.with_extension("part");
                    std::io::copy(&mut resp, &mut std::fs::File::create(&tmp)?)?;
                    std::fs::rename(&tmp, &dest)?;
                    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
                }
                downloaded += 1;
                eprintln!("[{downloaded}/{max_episodes}] episode {id}: {}", agents.iter().map(|a| a["teamName"].as_str().unwrap_or("?")).collect::<Vec<_>>().join(" vs "));
                if downloaded >= max_episodes {
                    break 'teams;
                }
            }
        }
    }
    std::fs::write(out.join("episodes.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download high-quality episodes: both agents from the top-K leaderboard teams
    Fetch {
        #[arg(long, default_value = "data/lux/raw")]
        out: PathBuf,
        #[arg(long, default_value = "lux-ai-season-3")]
        competition: String,
        #[arg(long, default_value_t = 10)]
        top_teams: usize,
        #[arg(long, default_value_t = 200)]
        max_episodes: usize,
        #[arg(long, default_value_t = 1000)]
        sleep_ms: u64,
    },
    /// Convert episode replay .json files (recursive) to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 64)]
        shard_size: usize,
        #[arg(long)]
        max_episodes: Option<usize>,
    },
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_episodes: Option<usize>) -> Result<Value> {
    let files = common::find_files(input, &[".json"])?;
    let meta = json!({"global_features": G, "sap_range": SAP_R});
    let mut w = ShardWriter::new(out, "lux", CONVERTER_VERSION, meta, shard_size)?;
    let (mut episodes, mut bad, mut samples, mut sap_out) = (0usize, 0usize, 0usize, 0usize);
    for path in files {
        if max_episodes.map(|m| episodes >= m).unwrap_or(false) {
            break;
        }
        let ep: Value = match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(v) => v,
            None => {
                bad += 1;
                continue;
            }
        };
        if ep["steps"].as_array().map(|s| s.is_empty()).unwrap_or(true) {
            bad += 1;
            continue;
        }
        episodes += 1;
        let (ss, out_of_range) = episode_samples(&ep, &path.file_name().unwrap().to_string_lossy());
        sap_out += out_of_range;
        for s in ss {
            samples += 1;
            w.push(s)?;
        }
    }
    w.flush()?;
    Ok(json!({"episodes": episodes, "bad_files": bad, "samples": samples, "shards": w.written_shards,
              "sap_out_of_range": sap_out}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, competition, top_teams, max_episodes, sleep_ms } => fetch(&out, &competition, top_teams, max_episodes, sleep_ms),
        Cmd::Convert { input, out, shard_size, max_episodes } => {
            println!("{}", convert(&input, &out, shard_size, max_episodes)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/lux/ep0.json");
        serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
    }

    #[test]
    fn sap_offset_encoding() {
        assert_eq!(sap_offset(0, 0), Some(7 * 15 + 7));
        assert_eq!(sap_offset(-7, -7), Some(0));
        assert_eq!(sap_offset(7, 7), Some(15 * 15 - 1));
        assert_eq!(sap_offset(2, -1), Some(99)); // the fixture's [9, 2, -1]
        assert_eq!(sap_offset(8, 0), None);
        assert_eq!(sap_offset(0, -8), None);
        // inverse: dx = o % 15 - 7, dy = o / 15 - 7
        for o in 0..15 * 15 {
            assert_eq!(sap_offset(o % 15 - SAP_R, o / 15 - SAP_R), Some(o));
        }
    }

    #[test]
    fn env_cfg_lookup() {
        let mut ep = fixture();
        // the fixture's configuration.env_cfg has none of the unit keys → defaults, unknown
        let e = EnvCfg::parse(&ep, 0);
        assert_eq!(e, EnvCfg { values: [2.0, 10.0, 4.0, 2.0], known: false });
        assert_eq!(e.globals(), [0.4, 0.2, 0.5, 0.25, 0.0]);
        // all four in steps[0][agent].info.env_cfg → known
        ep["steps"][0][1]["info"] = json!({"env_cfg": {"unit_move_cost": 3, "unit_sap_cost": 40, "unit_sap_range": 6,
                                                        "unit_sensor_range": 4, "map_width": 24}});
        let e = EnvCfg::parse(&ep, 1);
        assert_eq!(e, EnvCfg { values: [3.0, 40.0, 6.0, 4.0], known: true });
        assert_eq!(e.globals()[4], 1.0);
        assert!(!EnvCfg::parse(&ep, 0).known); // agent 0 has no info
        // keys split over the two sources: values used, but not "known"
        ep["steps"][0][1]["info"]["env_cfg"].as_object_mut().unwrap().remove("unit_sensor_range");
        ep["configuration"]["env_cfg"]["unit_sensor_range"] = json!(3);
        let e = EnvCfg::parse(&ep, 1);
        assert_eq!(e, EnvCfg { values: [3.0, 40.0, 6.0, 3.0], known: false });
        // all four in configuration.env_cfg
        for (k, v) in [("unit_move_cost", 1), ("unit_sap_cost", 30), ("unit_sap_range", 5)] {
            ep["configuration"]["env_cfg"][k] = json!(v);
        }
        assert_eq!(EnvCfg::parse(&ep, 0), EnvCfg { values: [1.0, 30.0, 5.0, 3.0], known: true });
    }

    #[test]
    fn convert_with_env_cfg_in_info() {
        let mut ep = fixture();
        for agent in 0..2 {
            ep["steps"][0][agent]["info"] = json!({"env_cfg": {"unit_move_cost": 4, "unit_sap_cost": 25,
                                                                "unit_sap_range": 7, "unit_sensor_range": 3}});
        }
        let dir = std::env::temp_dir().join(format!("gmdata-lux-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("in")).unwrap();
        std::fs::write(dir.join("in").join("ep.json"), serde_json::to_vec(&ep).unwrap()).unwrap();
        let stats = convert(&dir.join("in"), &dir.join("out"), 64, None).unwrap();
        assert_eq!(stats["sap_out_of_range"], 0);
        let idx: Value =
            serde_json::from_slice(&std::fs::read(dir.join("out").join("lux_00000").join("index.json")).unwrap()).unwrap();
        assert_eq!(idx["converter_version"], 2);
        assert_eq!(idx["meta"], json!({"global_features": 11, "sap_range": 7}));
        assert_eq!(idx["tensors"]["obs.globals_seq"]["frame_shape"], json!([11]));
        assert_eq!(idx["tensors"]["tgt.sap_offset"]["frame_shape"], json!([16]));
        let meta = &idx["samples"][0]["meta"]["env_cfg"];
        assert_eq!(meta["known"], true);
        assert_eq!(meta["unit_sap_cost"], 25.0);
        // in memory: env globals on every frame, the sap target of the [9, 2, -1] action
        let (ss, out) = episode_samples(&ep, "ep.json");
        assert_eq!(out, 0);
        for smp in &ss {
            let g = smp.tensors.iter().find(|t| t.name == "obs.globals_seq").unwrap();
            let gmshard::Column::F32(v) = &g.col else { panic!("globals dtype") };
            for fr in v.chunks(G) {
                assert_eq!(&fr[6..], &[0.8, 0.5, 0.875, 0.375, 1.0]);
            }
        }
        let col = |smp: &Sample, name: &str| -> Vec<i64> {
            let t = smp.tensors.iter().find(|t| t.name == name).unwrap();
            let gmshard::Column::I64(v) = &t.col else { panic!("{name} dtype") };
            v.clone()
        };
        for smp in &ss {
            let (so, ua) = (col(smp, "tgt.sap_offset"), col(smp, "tgt.unit_actions"));
            for (o, a) in so.chunks(U).zip(ua.chunks(U)) {
                assert_eq!(o[2], 99); // slot 2 always plays [9, 2, -1]
                assert!(o.iter().zip(a).all(|(&o, &a)| (a == SAP_ACTION) == (o >= 0)));
            }
        }
        // a sap outside the window is counted, target −1
        let mut far = fixture();
        far["steps"][1][0]["action"][2] = json!([5, 8, 0]);
        let (ss, out) = episode_samples(&far, "far.json");
        assert_eq!(out, 1);
        assert_eq!(col(&ss[0], "tgt.sap_offset")[2], -1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
