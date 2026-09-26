//! Pokémon Showdown replays (observer-view sim protocol) → shards for the pokemon adapter.
//!
//! converter_version 2. One sample per (replay, side) holding one frame per decision the
//! side's player answers, in log order — the requests a live Showdown bot receives:
//!
//!   team preview   phase 2, turn 0 — the lead, when the format has team preview (`|poke|` /
//!                  `|teampreview|` lines): a switch to the lead's team-preview slot.  Both
//!                  leads are chosen simultaneously, so both sides' frames are the state before
//!                  the first lead `|switch|` line (neither sees the other side's lead).
//!   turn           phase 0, turn t ≥ 1 — one frame per side at EVERY `|turn|t` line: the state
//!                  before turn t's simultaneous choices, whatever happens during the turn (so
//!                  which frames exist never depends on the turn's outcome).  Labelled with the
//!                  side's first visible choice of the turn: its `|move|` (also `|cant|..|MOVE`,
//!                  a chosen move that could not be used; `|cant|HOLDER|..|MOVE|[of] USER` is an
//!                  Armor Tail / Dazzling / Queenly Majesty / Damp holder blocking USER's move,
//!                  no choice of HOLDER) or a voluntary `|switch|`.  Turns whose choice is not
//!                  visible — KO'd / flinched / asleep / paralysed before moving, a locked move
//!                  (`[from]lockedmove`) or recharge turn (no choice at all), Struggle, forced
//!                  out first, the battle ended — keep their frame with the label masked:
//!                  tgt.action_known 0, action_type −1.
//!   forced switch  phase 1 — every other in-battle `|switch|` of the side: KO replacements,
//!                  pivots (U-turn, Volt Switch, Flip Turn, Parting Shot, Chilly Reception,
//!                  Teleport, Baton Pass, Shed Tail), Eject Button / Eject Pack, Emergency Exit /
//!                  Wimp Out.  The frame is the state when Showdown issues the switch request:
//!                  right before the switch line for mid-turn requests (one side's), and for the
//!                  end-of-turn request (after `|upkeep|`, sent at once to every side with a
//!                  fainted active or a pending eject) the state before the FIRST switch line of
//!                  that block, so a side never sees the other side's simultaneous replacement.
//!                  A request left unanswered (forfeit at the prompt) yields no frame — nor does
//!                  a simultaneous end-of-turn request the OTHER player forfeited at (neither
//!                  switch is logged).  Revival Blessing's choice of a fainted Pokémon logs no
//!                  `|switch|` and has no frame.
//!
//!   obs.species_ids [T,12] i64 (crc32 hash buckets)   obs.move_ids [T,4] i64
//!   obs.poke_feats [T,12,17] f32   obs.slot_mask [T,12] bool   obs.field_seq [T,22] f32
//!   tgt.action_type [T] i64    0 move / 1 switch / −1 no observed choice
//!   tgt.action_known [T] bool  the frame carries an action label
//!   tgt.move_id [T] i64        hash bucket (move_buckets) of the used move's NAME, −1 off move
//!                              frames — the identity target (the policy scores the whole move
//!                              vocabulary; a live bot masks it to the moves of its request)
//!   tgt.move_idx [T] i64       the used move's slot in obs.move_ids when it was revealed BEFORE
//!                              the frame (legacy slot head), else −1 (first use, 5th+ reveal)
//!   tgt.switch_idx / tera / active_idx [T] i64, tgt.value [T] f32
//!   tgt.legal_action_type [T,2] bool   [move, switch]: forced / preview frames [0, 1]; turn
//!                              frames [1, a live own bench Pokémon exists]
//!   tgt.legal_switch [T,6] bool  own slots that may come in (present, not fainted, not active;
//!                              unknown slots of a side without team preview count as legal)
//!
//! obs.field_seq = the v1 layout (weather, terrain, turn/50, both sides' conditions) + [20]
//! forced-switch request, [21] team-preview request.  obs.move_ids stays strictly causal: the
//! first four moves the active Pokémon revealed before the frame.  The legal masks are
//! functions of the frame's state only; a label outside them is counted (stat `mask_mismatch`)
//! and left to the decoder, which gives it no weight under a legality mask.  `|split|` blocks
//! of omniscient logs: only the public line is read.
//!
//! State tracking beyond v1: `-swapsideconditions` (Court Change) swaps the sides' hazards and
//! screens; `-transform` gives the transformed Pokémon the target's revealed moves and boosts
//! for as long as it stays in (moves used meanwhile are not added to its own moveset);
//! Revival Blessing's `-heal` clears the fainted flag.  Replays without a `|win|` yield no samples.
//!
//! Re-conversion (v1 shards are obsolete: different frames, targets and field width; shard
//! meta `{"field_features": 22, "move_buckets": 1024, "species_buckets": 512}`) — on the
//! cluster from the checkout root (`cargo build --release -p gmdata` in data_processing/; the
//! fetch is resumable and skips replays already on disk — without internet access on the
//! nodes, fetch elsewhere and rsync the raw directory), then every 5th shard → val:
//!
//!   G=$PWD/data_processing/target/release/gmdata; D=$HOME/gm-data/v4
//!   $G pokemon fetch --format gen9ou --min-rating 1500 --max-replays 2000 --sleep-ms 500 \
//!       --out $HOME/gm-data/raw/pokemon
//!   $G pokemon convert --hq --input $HOME/gm-data/raw/pokemon --out $D/shards/pokemon
//!   mkdir -p $D/split/pokemon/train $D/split/pokemon/val; i=0
//!   for s in $D/shards/pokemon/pokemon_*; do
//!     [ $((i % 5)) -eq 4 ] && v=val || v=train; ln -s $s $D/split/pokemon/$v/; i=$((i + 1)); done
//!
//! Train on `--data pokemon=$D/split/pokemon/train --val pokemon=$D/split/pokemon/val`; fresh
//! runs take pokemon_field_features / pokemon_move_buckets from the shard meta.  A checkpoint
//! with a v1 pokemon adapter (20 field features, the slot head) is refused on v2 shards by
//! check_meta_arch: its pokemon adapter must be rebuilt (a fresh adapter on the trained core).
//!
//! Spectator logs are third-person: the converter can only show what the log reveals, so a
//! side's own moves, items, abilities, stats and tera type — all known to the player — appear
//! only once revealed, and an Illusion switch is filed under the disguise.  First-person
//! trajectories remove that gap: Metamon (github.com/UT-Austin-RPL/metamon, MIT) rebuilds each
//! player's own view from Showdown replays (the player's team filled in from what the battle
//! reveals, the rest estimated), with the action taken at each step.  A future `gmdata pokemon
//! convert-metamon` could emit this schema from those trajectories: map their steps onto the
//! frames here (turn / forced switch, team preview where present), fill the own-team slots and
//! obs.move_ids with the full sets, and derive legal_* from the known own team.  Only the
//! player's own side may be filled in that way — the opponent stays revealed-only, and
//! estimated fields are a train/deploy gap (a live bot knows its exact set).

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const CONVERTER_VERSION: u32 = 2;
const TEAM: usize = 6;
const FEAT: usize = 17;
const FIELD: usize = 22;
const MOVES: usize = 4;
const F_HP: usize = 0;
const F_ACTIVE: usize = 1;
const F_FAINTED: usize = 2;
const F_STATUS: usize = 3;
const F_BOOSTS: usize = 10;
const F_REVEALED: usize = 15;
const F_TERA: usize = 16;
/// obs.field_seq flag: the frame answers a forced-switch request
const FLD_FORCED: usize = 20;
/// obs.field_seq flag: the frame answers the team-preview (lead) request
const FLD_PREVIEW: usize = 21;
const BOOSTS: [&str; 5] = ["atk", "def", "spa", "spd", "spe"];
const SCREENS: [&str; 3] = ["reflect", "lightscreen", "auroraveil"];
/// moves whose user switches out afterwards (normalised names): its side gets a switch request
const SELF_SWITCH: [&str; 8] = ["uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "teleport", "batonpass",
    "shedtail"];
/// items / abilities that switch their holder out (`-enditem` / `-activate`, normalised)
const EJECT: [&str; 4] = ["ejectbutton", "ejectpack", "emergencyexit", "wimpout"];

fn norm(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

fn bucket(name: &str, buckets: u32) -> i64 {
    if name.is_empty() {
        return 0;
    }
    (crc32fast::hash(norm(name).as_bytes()) % (buckets - 1) + 1) as i64
}

fn status_idx(s: &str) -> Option<usize> {
    ["", "brn", "par", "psn", "tox", "slp", "frz"].iter().position(|&x| x == s)
}

fn weather_idx(w: &str) -> usize {
    match w {
        "sunnyday" | "desolateland" => 0,
        "raindance" | "primordialsea" => 1,
        "sandstorm" => 2,
        "snow" | "snowscape" | "hail" => 3,
        _ => 4,
    }
}

fn terrain_idx(t: &str) -> Option<usize> {
    ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"].iter().position(|&x| x == t)
}

fn base_species(details: &str) -> String {
    details.split(',').next().unwrap_or("").trim().to_string()
}

/// Python `float(s)` for the forms seen in HP strings.
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok().filter(|x| x.is_finite() || t.to_lowercase().contains("inf"))
}

fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let t2 = t.strip_prefix('+').unwrap_or(t);
    if t2.is_empty() || t2.starts_with('+') {
        return None;
    }
    t2.parse::<i64>().ok()
}

/// "78/100" / "0 fnt" / "45/100 par" → (frac, status, fainted)
fn parse_hp(hp: &str) -> (f64, Option<usize>, bool) {
    let parts: Vec<&str> = hp.split_whitespace().collect();
    if parts.is_empty() {
        return (1.0, None, false);
    }
    let (fs, rest) = (parts[0], &parts[1..]);
    if rest.first() == Some(&"fnt") {
        return (0.0, None, true);
    }
    let frac = if let Some((a, b)) = fs.split_once('/') {
        match (py_float(a), py_float(b)) {
            (Some(n), Some(d)) => (n / d.max(1.0)).clamp(0.0, 1.0),
            _ => 1.0,
        }
    } else {
        py_float(fs).map(|x| (x / 100.0).clamp(0.0, 1.0)).unwrap_or(1.0)
    };
    let st = rest.first().and_then(|r| status_idx(r));
    if frac == 0.0 {
        return (0.0, st, true);
    }
    (frac, st, false)
}

#[derive(Clone, Default)]
struct Mon {
    species: String,
    hp: f64,
    status: usize,
    boosts: [i64; 5],
    fainted: bool,
    revealed: bool,
    tera: bool,
    moves: Vec<String>,
    /// moves while transformed: the target's revealed moves, then the ones used; None when
    /// not transformed (cleared when the Pokémon leaves the field)
    transformed: Option<Vec<String>>,
}

impl Mon {
    fn new(species: String) -> Self {
        Mon { species, hp: 1.0, ..Default::default() }
    }
    /// the moveset in effect (the transformed one while transformed)
    fn moves_now(&self) -> &[String] {
        self.transformed.as_deref().unwrap_or(&self.moves)
    }
    fn reveal_move(&mut self, mv: &str) {
        let list = match self.transformed.as_mut() {
            Some(t) => t,
            None => &mut self.moves,
        };
        let n = norm(mv);
        if !list.iter().any(|m| norm(m) == n) {
            list.push(mv.to_string());
        }
    }
    /// leaves the field (switched / dragged out, fainted): volatile state resets
    fn leave_field(&mut self) {
        self.boosts = [0; 5];
        self.transformed = None;
    }
}

#[derive(Clone, Default)]
struct SideSt {
    name: String,
    team: Vec<Mon>,
    active: Option<usize>,
    nick2slot: HashMap<String, usize>,
    nick_order: Vec<String>,
    hazards: HashMap<String, i64>,
    /// the team came from team preview (`|poke|` lines): slots past it are empty, not unknown
    previewed: bool,
}

impl SideSt {
    fn claimed(&self, i: usize) -> bool {
        self.nick2slot.values().any(|&v| v == i)
    }
    fn assign(&mut self, nick: &str, i: usize) -> usize {
        if !self.nick2slot.contains_key(nick) {
            self.nick_order.push(nick.to_string());
        }
        self.nick2slot.insert(nick.to_string(), i);
        i
    }
    fn slot_for(&mut self, nick: &str, details: &str) -> usize {
        if let Some(&s) = self.nick2slot.get(nick) {
            return s;
        }
        let sp = base_species(details);
        let n = norm(&sp);
        for i in 0..self.team.len() {
            if norm(&self.team[i].species) == n && !self.claimed(i) {
                return self.assign(nick, i);
            }
        }
        for i in 0..self.team.len() {
            let pn = norm(&self.team[i].species);
            if !pn.is_empty() && (n.starts_with(&pn) || pn.starts_with(&n)) && !self.claimed(i) {
                return self.assign(nick, i);
            }
        }
        if self.team.len() < TEAM {
            self.team.push(Mon::new(sp));
            let i = self.team.len() - 1;
            return self.assign(nick, i);
        }
        let i = self.active.unwrap_or(0);
        self.assign(nick, i)
    }
    fn slot_of(&self, nick: &str) -> Option<usize> {
        self.nick2slot.get(nick).copied().or(self.active).filter(|&s| s < self.team.len())
    }
}

#[derive(Clone)]
struct Frame {
    species: Vec<String>,
    moves: Vec<String>,
    feats: Vec<f32>,
    mask: Vec<bool>,
    field: Vec<f32>,
    active: i64,
}

/// The request a frame answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Turn,
    Forced,
    Preview,
}

/// One decision frame: the snapshot, its legality masks, and the label once seen.
#[derive(Clone)]
struct Rec {
    frame: Frame,
    /// −1 until a choice is observed
    action_type: i64,
    move_idx: i64,
    move_name: String,
    switch_idx: i64,
    tera: i64,
    legal_type: [bool; 2],
    legal_switch: [bool; TEAM],
}

impl Rec {
    fn label_move(&mut self, mv: &str, tera: bool) {
        let n = norm(mv);
        self.action_type = 0;
        self.move_idx = self.frame.moves.iter().take(MOVES).position(|m| !m.is_empty() && norm(m) == n).map_or(-1, |i| i as i64);
        self.move_name = mv.to_string();
        self.tera = tera as i64;
    }
    /// Labels a switch; false when the label is outside the frame's masks.  The masks stay
    /// what the snapshot allowed (never label-dependent): the decoder gives such a target no
    /// weight under a legality mask.
    fn label_switch(&mut self, slot: usize) -> bool {
        self.action_type = 1;
        self.switch_idx = slot as i64;
        slot < TEAM && self.legal_switch[slot] && self.legal_type[1]
    }
}

struct Battle {
    sides: [SideSt; 2],
    weather: Option<usize>,
    terrain: Option<usize>,
    turn: i64,
    winner: Option<usize>,
}

fn side_idx(pos: &str) -> Option<usize> {
    match pos {
        "p1" => Some(0),
        "p2" => Some(1),
        _ => None,
    }
}

fn ident(tok: &str) -> (String, String) {
    let (pos, nick) = tok.split_once(':').unwrap_or((tok, ""));
    (pos.trim().chars().take(2).collect(), nick.trim().to_string())
}

fn encode(b: &Battle, side: usize) -> Frame {
    let other = 1 - side;
    let mut feats = vec![0f32; 2 * TEAM * FEAT];
    let mut mask = vec![false; 2 * TEAM];
    for (base, s) in [(0, &b.sides[side]), (TEAM, &b.sides[other])] {
        for (i, p) in s.team.iter().take(TEAM).enumerate() {
            let j = base + i;
            mask[j] = !p.species.is_empty();
            let f = &mut feats[j * FEAT..(j + 1) * FEAT];
            f[F_HP] = p.hp as f32;
            f[F_ACTIVE] = (s.active == Some(i)) as u8 as f32;
            f[F_FAINTED] = p.fainted as u8 as f32;
            f[F_STATUS + p.status] = 1.0;
            for (k, &bv) in p.boosts.iter().enumerate() {
                f[F_BOOSTS + k] = ((bv as f64 / 6.0).clamp(-1.0, 1.0)) as f32;
            }
            f[F_REVEALED] = p.revealed as u8 as f32;
            f[F_TERA] = p.tera as u8 as f32;
        }
    }
    let mut fld = vec![0f32; FIELD];
    if let Some(w) = b.weather {
        fld[w] = 1.0;
    }
    if let Some(t) = b.terrain {
        fld[5 + t] = 1.0;
    }
    fld[9] = (b.turn as f64 / 50.0).min(1.0) as f32;
    for (off, s) in [(10, &b.sides[side]), (15, &b.sides[other])] {
        let h = |k: &str| *s.hazards.get(k).unwrap_or(&0);
        fld[off] = (h("spikes").min(3) as f64 / 3.0) as f32;
        fld[off + 1] = (h("toxicspikes").min(2) as f64 / 2.0) as f32;
        fld[off + 2] = (h("stealthrock") > 0) as u8 as f32;
        fld[off + 3] = (h("stickyweb") > 0) as u8 as f32;
        fld[off + 4] = (SCREENS.iter().filter(|&&sc| h(sc) > 0).count().min(3) as f64 / 3.0) as f32;
    }
    let own = &b.sides[side];
    let mut moves = vec![String::new(); MOVES];
    if let Some(a) = own.active.filter(|&a| a < own.team.len()) {
        for (k, m) in own.team[a].moves_now().iter().take(MOVES).enumerate() {
            moves[k] = m.clone();
        }
    }
    let mut species: Vec<String> = own.team.iter().map(|p| p.species.clone()).collect();
    species.resize(TEAM, String::new());
    let mut os: Vec<String> = b.sides[other].team.iter().map(|p| p.species.clone()).collect();
    os.resize(TEAM, String::new());
    species.extend(os);
    Frame { species, moves, feats, mask, field: fld, active: own.active.unwrap_or(0) as i64 }
}

/// The frame of `side` answering a `phase` request in the current state (label unset).
fn snapshot(b: &Battle, side: usize, phase: Phase) -> Rec {
    let mut frame = encode(b, side);
    match phase {
        Phase::Forced => frame.field[FLD_FORCED] = 1.0,
        Phase::Preview => frame.field[FLD_PREVIEW] = 1.0,
        Phase::Turn => {}
    }
    let own = &b.sides[side];
    let mut legal_switch = [false; TEAM];
    for (k, ok) in legal_switch.iter_mut().enumerate() {
        *ok = match own.team.get(k) {
            Some(p) => !p.species.is_empty() && !p.fainted && (phase == Phase::Preview || own.active != Some(k)),
            None => !own.previewed,
        };
    }
    let legal_type = match phase {
        Phase::Turn => [true, legal_switch.iter().any(|&x| x)],
        Phase::Forced | Phase::Preview => [false, true],
    };
    Rec { frame, action_type: -1, move_idx: -1, move_name: String::new(), switch_idx: -1, tera: 0, legal_type, legal_switch }
}

const IGNORED: [&str; 32] = ["", "j", "l", "c", "c:", "chat", "raw", "html", "uhtml", "inactive", "inactiveoff", "t:", "gametype", "gen",
    "tier", "rated", "rule", "clearpoke", "start", "teamsize", "player", "-message", "message", "debug",
    "error", "seed", "choice", "unlink", "-hint", "-anim", "n", "join"];

struct Parse {
    records: [Vec<Rec>; 2],
    winner: Option<usize>,
    names: [String; 2],
    /// frame / label statistics (frames per phase, unlabelled reasons, forced-switch causes)
    stats: BTreeMap<String, usize>,
    unknown: BTreeMap<String, usize>,
}

fn parse_log(log: &str) -> Parse {
    let mut b = Battle { sides: Default::default(), weather: None, terrain: None, turn: 0, winner: None };
    let mut stats: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut records: [Vec<Rec>; 2] = [Vec::new(), Vec::new()];
    // the side's frame of the current turn (index into records[side])
    let mut turn_rec: [Option<usize>; 2] = [None, None];
    // how the side's turn choice resolved: None = not yet visible, else "move" / "switch"
    // (labelled) or the reason its frame stays unlabelled
    let mut acted: [Option<&'static str>; 2] = [None, None];
    // a snapshotted request awaiting the side's switch line (team preview, end-of-turn block)
    let mut open: [Option<usize>; 2] = [None, None];
    let mut need_replace = [false; 2]; // the active fainted, no switch-in since
    let mut pending_pivot = [false; 2]; // a self-switch move this turn (resolves mid-turn)
    let mut pending_eject = [false; 2]; // Eject Button / Pack, Emergency Exit / Wimp Out
    let mut upkeep = false; // the turn's residual phase ended: switches now answer end-of-turn requests
    let mut tera_pending = [false; 2];
    let mut preview = false;
    let mut lead_block = false;
    let mut ended = false;
    let mut secret = false; // the next line is the private half of a |split| block
    macro_rules! stat {
        ($k:expr) => {
            *stats.entry($k.to_string()).or_default() += 1
        };
    }
    // a |turn| / |win| / |tie| line closes the current decision window
    macro_rules! close_turn {
        () => {
            for side in 0..2 {
                if let Some(r) = turn_rec[side].take() {
                    if records[side][r].action_type < 0 {
                        stat!(format!("unlabelled_{}", acted[side].unwrap_or("no_visible_action")));
                    }
                }
                if let Some(r) = open[side].take() {
                    if records[side][r].action_type < 0 {
                        stat!("unlabelled_unanswered_request");
                    }
                }
                acted[side] = None;
                tera_pending[side] = false;
                need_replace[side] = false;
                pending_pivot[side] = false;
                pending_eject[side] = false;
            }
            upkeep = false;
        };
    }
    for raw in log.split('\n') {
        if !raw.starts_with('|') {
            continue;
        }
        if std::mem::take(&mut secret) {
            continue;
        }
        let parts: Vec<&str> = raw.split('|').collect();
        let cmd = parts.get(1).copied().unwrap_or("");
        let args: &[&str] = if parts.len() > 2 { &parts[2..] } else { &[] };
        let a = |i: usize| args.get(i).copied().unwrap_or("");
        match cmd {
            "split" => {
                // omniscient logs (local server, self-play): |split|SIDE, the SIDE-only line (exact
                // HP), then the public line — only the public line is game state for both players
                secret = true;
                continue;
            }
            "player" if args.len() >= 2 && side_idx(a(0)).is_some() => {
                if !a(1).is_empty() {
                    b.sides[side_idx(a(0)).unwrap()].name = a(1).to_string();
                }
                continue;
            }
            "poke" if args.len() >= 2 => {
                if let Some(si) = side_idx(a(0)) {
                    let s = &mut b.sides[si];
                    s.previewed = true;
                    if s.team.len() < TEAM {
                        s.team.push(Mon::new(base_species(a(1))));
                    }
                }
                preview = true;
                continue;
            }
            "teampreview" => {
                preview = true;
                continue;
            }
            "switch" | "drag" | "replace" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                // the request this switch answers; its frame(s) are snapshots from BEFORE the line
                let mut target: Option<usize> = None;
                if cmd == "switch" && !ended {
                    if b.turn == 0 && b.sides[pi].active.is_none() {
                        // the lead: chosen at team preview, simultaneously by both players
                        if preview && !lead_block {
                            lead_block = true;
                            for s in 0..2 {
                                records[s].push(snapshot(&b, s, Phase::Preview));
                                open[s] = Some(records[s].len() - 1);
                                stat!("frames_preview");
                            }
                        }
                        target = open[pi].take();
                    } else if open[pi].is_some() {
                        target = open[pi].take();
                    } else if need_replace[pi] || pending_eject[pi] || pending_pivot[pi] {
                        stat!(if need_replace[pi] {
                            "forced_replacement"
                        } else if pending_eject[pi] {
                            "forced_eject"
                        } else {
                            "forced_pivot"
                        });
                        if upkeep {
                            // the end-of-turn request goes to every side with a pending
                            // replacement at once: all their frames are this pre-switch state
                            for s in 0..2 {
                                if open[s].is_none() && (s == pi || need_replace[s] || pending_eject[s]) {
                                    records[s].push(snapshot(&b, s, Phase::Forced));
                                    open[s] = Some(records[s].len() - 1);
                                    stat!("frames_forced");
                                }
                            }
                            target = open[pi].take();
                        } else {
                            records[pi].push(snapshot(&b, pi, Phase::Forced));
                            target = Some(records[pi].len() - 1);
                            stat!("frames_forced");
                        }
                    } else if acted[pi].is_none() && turn_rec[pi].is_some() {
                        acted[pi] = Some("switch"); // voluntary: the turn's choice
                        target = turn_rec[pi];
                    } else {
                        // after the side acted, with no recognised cause (an unlisted pivot)
                        stat!("forced_other");
                        records[pi].push(snapshot(&b, pi, Phase::Forced));
                        target = Some(records[pi].len() - 1);
                        stat!("frames_forced");
                    }
                    need_replace[pi] = false;
                    pending_pivot[pi] = false;
                    pending_eject[pi] = false;
                }
                let slot = b.sides[pi].slot_for(&nick, a(1));
                if let Some(r) = target {
                    if !records[pi][r].label_switch(slot) {
                        stat!("mask_mismatch");
                    }
                }
                let s = &mut b.sides[pi];
                if let Some(prev) = s.active.filter(|&x| x < s.team.len()) {
                    s.team[prev].leave_field();
                }
                s.active = Some(slot);
                let p = &mut s.team[slot];
                p.revealed = true;
                p.leave_field();
                if args.len() >= 3 && !a(2).is_empty() {
                    let (hp, st, fnt) = parse_hp(a(2));
                    p.hp = hp;
                    p.fainted = fnt;
                    if let Some(st) = st {
                        p.status = st;
                    }
                }
                continue;
            }
            "move" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let mv = a(1);
                if SELF_SWITCH.contains(&norm(mv).as_str()) {
                    pending_pivot[pi] = true;
                }
                if let Some(from) = args[2..].iter().find(|x| x.contains("[from]")) {
                    // not the side's choice: a locked-in move (no choice this turn), or a move
                    // called / copied / reflected by another effect
                    if from.contains("lockedmove") && acted[pi].is_none() {
                        acted[pi] = Some("locked");
                    }
                    continue;
                }
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                if norm(mv) == "struggle" {
                    if acted[pi].is_none() {
                        acted[pi] = Some("struggle");
                    }
                    continue;
                }
                if acted[pi].is_none() {
                    acted[pi] = Some("move");
                    if let Some(r) = turn_rec[pi] {
                        records[pi][r].label_move(mv, tera_pending[pi]);
                    }
                }
                b.sides[pi].team[slot].reveal_move(mv);
                continue;
            }
            "cant" if !args.is_empty() => {
                // |cant|POKEMON|REASON[|MOVE]: MOVE = the chosen move it could not use.  With an
                // `[of] SOURCE` kwarg (Armor Tail, Dazzling, Queenly Majesty, Damp) POKEMON is the
                // ability holder that blocked SOURCE's move: MOVE is SOURCE's choice (already
                // labelled and revealed by SOURCE's `|move|..|[still]` line) and POKEMON has not
                // acted — neither its label nor its moveset may take the foe's move.
                if args[1..].iter().any(|x| x.trim_start().starts_with("[of]")) {
                    continue;
                }
                let (pos, nick) = ident(a(0));
                if let Some(pi) = side_idx(&pos) {
                    let mv = a(2);
                    if acted[pi].is_none() {
                        if mv.is_empty() {
                            acted[pi] = Some("cant");
                        } else {
                            acted[pi] = Some("move");
                            if let Some(r) = turn_rec[pi] {
                                records[pi][r].label_move(mv, tera_pending[pi]);
                            }
                        }
                    }
                    if !mv.is_empty() {
                        if let Some(slot) = b.sides[pi].slot_of(&nick) {
                            b.sides[pi].team[slot].reveal_move(mv);
                        }
                    }
                }
                continue;
            }
            "-activate" | "-enditem" if args.len() >= 2 => {
                let (pos, _) = ident(a(0));
                if let Some(pi) = side_idx(&pos) {
                    // `[from]` = removed by another effect (Knock Off, Trick, …), not activated
                    let what = norm(a(1).rsplit(':').next().unwrap_or(""));
                    if EJECT.contains(&what.as_str()) && !args[2..].iter().any(|x| x.contains("[from]")) {
                        pending_eject[pi] = true;
                    }
                }
                continue;
            }
            "-damage" | "-heal" | "-sethp" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                let p = &mut b.sides[pi].team[slot];
                let (hp, st, fnt) = parse_hp(a(1));
                p.hp = hp;
                if fnt {
                    p.fainted = true;
                } else if cmd == "-heal" && hp > 0.0 {
                    p.fainted = false; // Revival Blessing
                }
                if let Some(st) = st {
                    p.status = st;
                }
                continue;
            }
            "-status" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].status = status_idx(a(1)).unwrap_or(0);
                }
                continue;
            }
            "-curestatus" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].status = 0;
                }
                continue;
            }
            "-boost" | "-unboost" | "-setboost" if args.len() >= 3 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let Some(k) = BOOSTS.iter().position(|&x| x == a(1)) else { continue };
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                let Some(amt) = py_int(a(2)) else { continue };
                let bv = &mut b.sides[pi].team[slot].boosts[k];
                *bv = match cmd {
                    "-boost" => (*bv + amt).clamp(-6, 6),
                    "-unboost" => (*bv - amt).clamp(-6, 6),
                    _ => amt.clamp(-6, 6),
                };
                continue;
            }
            "-clearboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].boosts = [0; 5];
                }
                continue;
            }
            "-clearnegativeboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    for x in b.sides[pi].team[slot].boosts.iter_mut() {
                        *x = (*x).max(0);
                    }
                }
                continue;
            }
            "-clearallboost" => {
                for s in b.sides.iter_mut() {
                    if let Some(act) = s.active.filter(|&x| x < s.team.len()) {
                        s.team[act].boosts = [0; 5];
                    }
                }
                continue;
            }
            "-copyboost" | "-invertboost" | "-swapboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    let bs = &mut b.sides[pi].team[slot].boosts;
                    if cmd == "-invertboost" {
                        for x in bs.iter_mut() {
                            *x = -*x;
                        }
                    } else {
                        *bs = [0; 5];
                    }
                }
                continue;
            }
            "-transform" if args.len() >= 2 => {
                // |-transform|POKEMON|TARGET: POKEMON copies TARGET's moves and stat stages
                let (pos, nick) = ident(a(0));
                let (tpos, tnick) = ident(a(1));
                let (Some(pi), Some(ti)) = (side_idx(&pos), side_idx(&tpos)) else { continue };
                let (Some(slot), Some(tslot)) = (b.sides[pi].slot_of(&nick), b.sides[ti].slot_of(&tnick)) else { continue };
                let (moves, boosts) = {
                    let t = &b.sides[ti].team[tslot];
                    (t.moves_now().to_vec(), t.boosts)
                };
                let p = &mut b.sides[pi].team[slot];
                p.transformed = Some(moves);
                p.boosts = boosts;
                continue;
            }
            "faint" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    let p = &mut b.sides[pi].team[slot];
                    p.fainted = true;
                    p.hp = 0.0;
                    p.leave_field();
                    if b.sides[pi].active == Some(slot) {
                        need_replace[pi] = true;
                    }
                }
                continue;
            }
            "upkeep" => {
                upkeep = true;
                pending_pivot = [false; 2]; // a self-switch that did not happen by now never will
                continue;
            }
            "-weather" if !args.is_empty() => {
                let w = norm(a(0));
                b.weather = if w == "none" || w.is_empty() { None } else { Some(weather_idx(&w)) };
                continue;
            }
            "-fieldstart" if !args.is_empty() => {
                if let Some(t) = terrain_idx(&norm(a(0).rsplit(':').next().unwrap_or(""))) {
                    b.terrain = Some(t);
                }
                continue;
            }
            "-fieldend" if !args.is_empty() => {
                if let Some(t) = terrain_idx(&norm(a(0).rsplit(':').next().unwrap_or(""))) {
                    if b.terrain == Some(t) {
                        b.terrain = None;
                    }
                }
                continue;
            }
            "-sidestart" | "-sideend" if args.len() >= 2 => {
                let (pos, _) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let name = norm(a(1).rsplit(':').next().unwrap_or(""));
                let known = matches!(name.as_str(), "spikes" | "toxicspikes" | "stealthrock" | "stickyweb") || SCREENS.contains(&name.as_str());
                if known {
                    let h = b.sides[pi].hazards.entry(name.clone()).or_insert(0);
                    if cmd == "-sidestart" {
                        let cap = match name.as_str() {
                            "spikes" => 3,
                            "toxicspikes" => 2,
                            _ => 1,
                        };
                        *h = (*h + 1).min(cap);
                    } else {
                        *h = 0;
                    }
                }
                continue;
            }
            "-swapsideconditions" => {
                // Court Change: every side condition changes sides
                let [s0, s1] = &mut b.sides;
                std::mem::swap(&mut s0.hazards, &mut s1.hazards);
                continue;
            }
            "-terastallize" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].tera = true;
                }
                tera_pending[pi] = true;
                continue;
            }
            "turn" if !args.is_empty() => {
                let Some(n) = py_int(a(0)) else { continue };
                close_turn!();
                if ended {
                    continue;
                }
                b.turn = n;
                for (side, recs) in records.iter_mut().enumerate() {
                    recs.push(snapshot(&b, side, Phase::Turn));
                    turn_rec[side] = Some(recs.len() - 1);
                    stat!("frames_turn");
                }
                continue;
            }
            "win" if !args.is_empty() => {
                close_turn!();
                ended = true;
                let name = a(0).trim();
                for (i, s) in b.sides.iter().enumerate() {
                    if s.name == name {
                        b.winner = Some(i);
                    }
                }
                continue;
            }
            "tie" => {
                close_turn!();
                ended = true;
                continue;
            }
            _ => {}
        }
        if IGNORED.contains(&cmd) || cmd == "leave" || cmd == "showteam" || cmd.starts_with("c:") {
            continue;
        }
        *unknown.entry(cmd.to_string()).or_default() += 1;
    }
    close_turn!();
    let _ = upkeep; // (reset by close_turn!, read by the next line only inside the loop)
    Parse { records, winner: b.winner, names: [b.sides[0].name.clone(), b.sides[1].name.clone()], stats, unknown }
}

fn replay_samples(r: &Value, sb: u32, mb: u32, p: &Parse) -> Vec<Sample> {
    let mut out = Vec::new();
    let Some(w) = p.winner else { return out };
    for side in 0..2 {
        let recs = &p.records[side];
        if recs.is_empty() {
            continue;
        }
        let t = recs.len();
        let (mut sp, mut mv, mut feats, mut mask, mut field) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut at, mut known, mut mi, mut mid, mut si, mut te, mut ai) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut lt, mut ls) = (vec![], vec![]);
        for rec in recs {
            let f = &rec.frame;
            sp.extend(f.species.iter().map(|s| bucket(s, sb)));
            mv.extend(f.moves.iter().map(|m| bucket(m, mb)));
            feats.extend_from_slice(&f.feats);
            mask.extend_from_slice(&f.mask);
            field.extend_from_slice(&f.field);
            ai.push(f.active);
            at.push(rec.action_type);
            known.push(rec.action_type >= 0);
            mi.push(rec.move_idx);
            mid.push(if rec.action_type == 0 { bucket(&rec.move_name, mb) } else { -1 });
            si.push(rec.switch_idx);
            te.push(rec.tera);
            lt.extend_from_slice(&rec.legal_type);
            ls.extend_from_slice(&rec.legal_switch);
        }
        let v = if w == side { 1.0 } else { -1.0 };
        let fmt = r.get("formatid").or_else(|| r.get("format")).cloned().unwrap_or_else(|| json!("?"));
        out.push(Sample::new(json!({"replay_id": r.get("id").cloned().unwrap_or(json!("?")), "side": (if side == 0 { "p1" } else { "p2" }),
                                     "player": p.names[side], "rating": r.get("rating").cloned().unwrap_or(Value::Null),
                                     "format": fmt, "won": w == side}))
            .i64("obs.species_ids", &[2 * TEAM], sp)
            .i64("obs.move_ids", &[MOVES], mv)
            .f32("obs.poke_feats", &[2 * TEAM, FEAT], feats)
            .bool("obs.slot_mask", &[2 * TEAM], mask)
            .f32("obs.field_seq", &[FIELD], field)
            .i64("tgt.action_type", &[], at)
            .bool("tgt.action_known", &[], known)
            .i64("tgt.move_id", &[], mid)
            .i64("tgt.move_idx", &[], mi)
            .i64("tgt.switch_idx", &[], si)
            .i64("tgt.tera", &[], te)
            .i64("tgt.active_idx", &[], ai)
            .bool("tgt.legal_action_type", &[2], lt)
            .bool("tgt.legal_switch", &[TEAM], ls)
            .f32("tgt.value", &[], vec![v; t]));
    }
    out
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download public replays of a format via the replay search API (resumable, polite)
    Fetch {
        #[arg(long, default_value = "gen9ou")]
        format: String,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        #[arg(long, default_value_t = 1000)]
        max_replays: usize,
        #[arg(long, default_value_t = 1000)]
        sleep_ms: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert replay .json files to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        #[arg(long)]
        max_replays: Option<usize>,
        #[arg(long, default_value_t = 512)]
        species_buckets: u32,
        /// hash buckets of the move vocabulary (obs.move_ids, the tgt.move_id identity target)
        #[arg(long, default_value_t = 1024)]
        move_buckets: u32,
        /// drop replays shorter than this many turns
        #[arg(long, default_value_t = 0)]
        min_turns: i64,
        /// drop replays that end in a forfeit (usually a concession: only the last turn is affected)
        #[arg(long)]
        exclude_forfeit: bool,
        /// drop replays decided by the inactivity timer
        #[arg(long)]
        exclude_inactivity: bool,
        /// imitate only the winning side
        #[arg(long)]
        winner_only: bool,
        /// high-quality preset: rating >= 1500, >= 5 turns, no inactivity losses
        #[arg(long)]
        hq: bool,
    },
}

#[derive(Clone, Copy, Default)]
pub struct Quality {
    pub min_turns: i64,
    pub exclude_forfeit: bool,
    pub exclude_inactivity: bool,
    pub winner_only: bool,
}

fn n_turns(log: &str) -> i64 {
    log.lines().filter_map(|l| l.strip_prefix("|turn|")).filter_map(|t| t.trim().parse::<i64>().ok()).max().unwrap_or(0)
}

const BASE: &str = "https://replay.pokemonshowdown.com";

fn fetch(format: &str, min_rating: i64, max: usize, sleep_ms: u64, out: &Path) -> Result<Value> {
    let client = common::http()?;
    std::fs::create_dir_all(out)?;
    let (mut downloaded, mut listed, mut low, mut existing, mut errors) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut before: Option<i64> = None;
    let pause = || std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    'pages: while downloaded < max {
        let mut url = format!("{BASE}/search.json?format={format}");
        if let Some(b) = before {
            url.push_str(&format!("&before={b}"));
        }
        let page: Vec<Value> = match client.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.json()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[fetch] search page failed ({e}); retrying in 10s");
                std::thread::sleep(std::time::Duration::from_secs(10));
                continue;
            }
        };
        if page.is_empty() {
            break;
        }
        let has_more = page.len() >= 51;
        before = page.last().and_then(|x| x["uploadtime"].as_i64());
        for item in page.iter().take(if has_more { 50 } else { page.len() }) {
            if downloaded >= max {
                break 'pages;
            }
            listed += 1;
            let Some(id) = item["id"].as_str() else { continue };
            if min_rating > 0 && item["rating"].as_i64().unwrap_or(0) < min_rating {
                low += 1;
                continue;
            }
            let dest = out.join(format!("{id}.json"));
            if dest.exists() {
                existing += 1;
                continue;
            }
            pause();
            match client.get(format!("{BASE}/{id}.json")).send().and_then(|r| r.error_for_status()).and_then(|r| r.bytes()) {
                Ok(bts) => {
                    let tmp = dest.with_extension("tmp");
                    std::fs::write(&tmp, &bts)?;
                    std::fs::rename(&tmp, &dest)?;
                    downloaded += 1;
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("[fetch] {id} failed: {e}");
                }
            }
        }
        if !has_more || before.is_none() {
            break;
        }
        pause();
    }
    Ok(json!({"listed": listed, "downloaded": downloaded, "skipped_rating": low, "skipped_existing": existing, "errors": errors}))
}

#[allow(clippy::too_many_arguments)]
pub fn convert(input: &Path, out: &Path, shard_size: usize, min_rating: i64, max_replays: Option<usize>, sb: u32, mb: u32,
               q: Quality) -> Result<Value> {
    let meta = json!({"field_features": FIELD, "move_buckets": mb, "species_buckets": sb});
    let mut w = ShardWriter::new(out, "pokemon", CONVERTER_VERSION, meta, shard_size)?;
    let mut st: BTreeMap<&str, usize> = ["seen", "kept", "samples", "frames", "no_winner", "low_rating", "not_replay", "low_quality"].into_iter().map(|k| (k, 0)).collect();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().map(|e| e == "json").unwrap_or(false)).collect();
    files.sort();
    for path in files {
        if max_replays.map(|m| st["kept"] >= m).unwrap_or(false) {
            break;
        }
        let r: Value = match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(v @ Value::Object(_)) if v.get("log").is_some() => v,
            _ => {
                *st.get_mut("not_replay").unwrap() += 1;
                continue;
            }
        };
        *st.get_mut("seen").unwrap() += 1;
        if min_rating > 0 && r["rating"].as_f64().unwrap_or(0.0) < min_rating as f64 {
            *st.get_mut("low_rating").unwrap() += 1;
            continue;
        }
        let log = r["log"].as_str().unwrap_or("");
        if (q.min_turns > 0 && n_turns(log) < q.min_turns)
            || (q.exclude_forfeit && log.contains(" forfeited."))
            || (q.exclude_inactivity && log.contains("due to inactivity")) {
            *st.get_mut("low_quality").unwrap() += 1;
            continue;
        }
        let p = parse_log(log);
        for (k, v) in &p.unknown {
            *unknown.entry(k.clone()).or_default() += v;
        }
        if p.winner.is_none() {
            *st.get_mut("no_winner").unwrap() += 1;
            continue;
        }
        let mut ss = replay_samples(&r, sb, mb, &p);
        if q.winner_only {
            ss.retain(|smp| smp.meta["won"] == json!(true));
        }
        if ss.is_empty() {
            continue;
        }
        *st.get_mut("kept").unwrap() += 1;
        for (k, v) in &p.stats {
            *labels.entry(k.clone()).or_default() += v;
        }
        for s in ss {
            *st.get_mut("samples").unwrap() += 1;
            *st.get_mut("frames").unwrap() += s.frames()?;
            w.push(s)?;
        }
    }
    w.flush()?;
    let mut v = json!(st);
    v["shards"] = json!(w.written_shards);
    v["labels"] = json!(labels);
    v["unknown"] = json!(unknown);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { format, min_rating, max_replays, sleep_ms, out } => {
            println!("{}", fetch(&format, min_rating, max_replays, sleep_ms, &out)?);
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, min_rating, max_replays, species_buckets, move_buckets, min_turns, exclude_forfeit,
                       exclude_inactivity, winner_only, hq } => {
            let mut q = Quality { min_turns, exclude_forfeit, exclude_inactivity, winner_only };
            let mut min_rating = min_rating;
            if hq {
                min_rating = min_rating.max(1500);
                q.min_turns = q.min_turns.max(5);
                q.exclude_inactivity = true;
            }
            println!("{}", convert(&input, &out, shard_size, min_rating, max_replays, species_buckets, move_buckets, q)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: [&str; 12] = ["|player|p1|Alice|", "|player|p2|Bob|", "|poke|p1|Garchomp, M|", "|poke|p1|Ditto|",
        "|poke|p1|Corviknight, F|", "|poke|p2|Gholdengo|", "|poke|p2|Great Tusk|", "|poke|p2|Kingambit, M|", "|teampreview",
        "|start", "|switch|p1a: Garchomp|Garchomp, M|100/100", "|switch|p2a: Gholdengo|Gholdengo|100/100"];

    fn parse(body: &[&str]) -> Parse {
        let mut lines: Vec<&str> = HEAD.to_vec();
        lines.extend_from_slice(body);
        lines.push("|win|Alice");
        parse_log(&lines.join("\n"))
    }

    fn turn(r: &Rec) -> i64 {
        (r.frame.field[9] * 50.0).round() as i64
    }

    fn forced(r: &Rec) -> bool {
        r.frame.field[FLD_FORCED] == 1.0
    }

    fn preview(r: &Rec) -> bool {
        r.frame.field[FLD_PREVIEW] == 1.0
    }

    /// is the Pokémon in slot j (own 0..6 / opponent 6..12) of the frame active?
    fn active(r: &Rec, j: usize) -> bool {
        r.frame.feats[j * FEAT + F_ACTIVE] == 1.0
    }

    #[test]
    fn lead_frames_are_the_pre_lead_state() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Nasty Plot|p2a: Gholdengo"]);
        for side in 0..2 {
            let r = &p.records[side][0];
            assert!(preview(r) && !forced(r) && turn(r) == 0);
            assert_eq!((r.action_type, r.switch_idx), (1, 0));
            assert!((0..12).all(|j| !active(r, j)), "the lead frame sees no lead");
            assert_eq!(r.legal_type, [false, true]);
            assert_eq!(r.legal_switch, [true, true, true, false, false, false]);
        }
        // without team preview the lead is not a choice: no frame
        let log = ["|player|p1|Alice|", "|player|p2|Bob|", "|start", "|switch|p1a: Garchomp|Garchomp, M|100/100",
            "|switch|p2a: Gholdengo|Gholdengo|100/100", "|turn|1", "|win|Alice"].join("\n");
        let p = parse_log(&log);
        assert!(p.records.iter().all(|r| r.len() == 1 && turn(&r[0]) == 1));
        assert_eq!(p.records[0][0].legal_switch, [false, true, true, true, true, true], "unknown slots may hold a Pokémon");
    }

    #[test]
    fn move_target_is_the_identity_and_the_slot_only_once_revealed() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp",
            "|upkeep", "|turn|2", "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp",
            "|upkeep", "|turn|3", "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo"]);
        let r = &p.records[0];
        assert_eq!(r.len(), 4);
        assert_eq!((r[1].action_type, r[1].move_idx, r[1].move_name.as_str()), (0, -1, "Earthquake"), "first use: no slot");
        assert!(r[1].frame.moves.iter().all(|m| m.is_empty()), "obs.move_ids is causal");
        assert_eq!((r[2].move_idx, r[2].move_name.as_str()), (-1, "Swords Dance"));
        assert_eq!(r[2].frame.moves[0], "Earthquake");
        assert_eq!((r[3].move_idx, r[3].move_name.as_str()), (0, "Earthquake"));
        assert_eq!(p.records[1][2].move_idx, 0);
    }

    #[test]
    fn unobserved_and_no_choice_turns_keep_an_unlabelled_frame() {
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|cant|p1a: Garchomp|flinch", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Outrage|p2a: Gholdengo", "|move|p2a: Gholdengo|Shadow Ball|p1a: Garchomp", "|upkeep", "|turn|3",
            "|move|p1a: Garchomp|Outrage|p2a: Gholdengo|[from]lockedmove", "|upkeep", "|turn|4",
            "|cant|p1a: Garchomp|move: Taunt|Swords Dance"]);
        let r = &p.records[0];
        assert_eq!(r.iter().map(|x| x.action_type).collect::<Vec<_>>(), [1, -1, 0, -1, 0]);
        assert_eq!(r[4].move_name, "Swords Dance", "|cant|..|MOVE names the chosen move");
        assert_eq!(p.records[1][3].action_type, -1, "no visible choice");
        assert_eq!(p.stats["unlabelled_cant"], 1);
        assert_eq!(p.stats["unlabelled_locked"], 1);
    }

    #[test]
    fn a_foe_blocked_by_an_ability_is_not_the_holders_choice() {
        // Armor Tail / Dazzling / Queenly Majesty / Damp: `|cant|HOLDER|ability: X|MOVE|[of] USER`
        // names the ability holder; MOVE is the user's (blocked) choice, logged by its own
        // `|move|..|[still]` line.  The holder acts later in the turn (or already did).
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Sucker Punch|p1a: Garchomp|[still]",
            "|cant|p1a: Garchomp|ability: Armor Tail|Sucker Punch|[of] p2a: Gholdengo",
            "|move|p1a: Garchomp|Earthquake|p2a: Gholdengo", "|upkeep", "|turn|2",
            "|move|p1a: Garchomp|Swords Dance|p1a: Garchomp", "|move|p2a: Gholdengo|Explosion|p1a: Garchomp|[still]",
            "|cant|p1a: Garchomp|ability: Damp|Explosion|[of] p2a: Gholdengo", "|upkeep", "|turn|3"]);
        let (r, o) = (&p.records[0], &p.records[1]);
        assert_eq!((r[1].action_type, r[1].move_name.as_str()), (0, "Earthquake"), "the holder's own move");
        assert_eq!((o[1].action_type, o[1].move_name.as_str()), (0, "Sucker Punch"));
        assert_eq!((r[2].move_name.as_str(), o[2].move_name.as_str()), ("Swords Dance", "Explosion"));
        let moves = |x: &Rec| x.frame.moves.iter().filter(|m| !m.is_empty()).cloned().collect::<Vec<_>>();
        assert_eq!(moves(&r[3]), ["Earthquake", "Swords Dance"], "no foe move in the holder's moveset");
        assert_eq!(moves(&o[3]), ["Sucker Punch", "Explosion"]);
        assert!(!p.stats.contains_key("unlabelled_cant"));
    }

    #[test]
    fn forced_switches_get_request_time_frames() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|U-turn|p2a: Gholdengo", "|-damage|p2a: Gholdengo|60/100",
            "|switch|p1a: Corviknight|Corviknight, F|100/100|[from] U-turn", "|move|p2a: Gholdengo|Make It Rain|p1a: Corviknight",
            "|-damage|p1a: Corviknight|0 fnt", "|faint|p1a: Corviknight", "|upkeep", "|switch|p1a: Ditto|Ditto|100/100",
            "|-transform|p1a: Ditto|p2a: Gholdengo|[from] ability: Imposter", "|turn|2"]);
        let r = &p.records[0];
        let kinds: Vec<(i64, bool, i64, i64)> = r.iter().map(|x| (turn(x), forced(x), x.action_type, x.switch_idx)).collect();
        assert_eq!(kinds, [(0, false, 1, 0), (1, false, 0, -1), (1, true, 1, 2), (1, true, 1, 1), (2, false, -1, -1)]);
        // the pivot frame is after the U-turn hit, before Corviknight comes in
        assert!(active(&r[2], 0) && r[2].frame.feats[TEAM * FEAT + F_HP] == 0.6);
        assert_eq!((r[2].legal_type, r[2].legal_switch), ([false, true], [false, true, true, false, false, false]));
        // the replacement frame sees the fainted active; Garchomp (pivoted out) or Ditto may come in
        assert!(active(&r[3], 2) && r[3].frame.feats[2 * FEAT + F_FAINTED] == 1.0);
        assert_eq!(r[3].legal_switch, [true, true, false, false, false, false]);
        assert_eq!(p.records[1].len(), 3, "the opponent answers no request");
        assert_eq!((p.stats["forced_pivot"], p.stats["forced_replacement"]), (1, 1));
    }

    #[test]
    fn simultaneous_replacements_do_not_see_each_other() {
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|Outrage|p2a: Gholdengo", "|-damage|p2a: Gholdengo|0 fnt", "|faint|p2a: Gholdengo",
            "|-damage|p1a: Garchomp|0 fnt|[from] item: Rocky Helmet", "|faint|p1a: Garchomp", "|upkeep",
            "|switch|p2a: Kingambit|Kingambit, M|100/100", "|switch|p1a: Corviknight|Corviknight, F|100/100", "|turn|2"]);
        let (a, b) = (&p.records[0][2], &p.records[1][2]);
        assert!(forced(a) && forced(b));
        assert_eq!((a.switch_idx, b.switch_idx), (2, 2));
        // p1 answered the same request as p2: its frame does not show Kingambit (opponent slot 2) in
        assert!(!active(a, TEAM + 2) && active(a, TEAM), "p1's frame: Gholdengo is still the (fainted) opposing active");
        assert!(active(b, 0) && !active(b, 2));
    }

    #[test]
    fn pending_causes_are_scoped() {
        // an Eject Button knocked off is not activated; a U-turn blocked by Protect switches nobody
        let p = parse(&["|turn|1", "|move|p1a: Garchomp|U-turn|p2a: Gholdengo", "|-activate|p2a: Gholdengo|move: Protect",
            "|move|p2a: Gholdengo|Knock Off|p1a: Garchomp", "|-enditem|p1a: Garchomp|Eject Button|[from] move: Knock Off|[of] p2a: Gholdengo",
            "|upkeep", "|turn|2", "|switch|p1a: Ditto|Ditto|100/100", "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto"]);
        let r = &p.records[0];
        assert_eq!(r.len(), 3);
        assert_eq!((forced(&r[2]), r[2].action_type, r[2].switch_idx), (false, 1, 1), "a voluntary switch next turn");
        // an activated Eject Button makes the holder's next switch a forced one
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|-damage|p1a: Garchomp|50/100",
            "|-enditem|p1a: Garchomp|Eject Button", "|switch|p1a: Ditto|Ditto|100/100"]);
        let r = &p.records[0];
        assert_eq!(r.iter().map(|x| (forced(x), x.action_type)).collect::<Vec<_>>(), [(false, 1), (false, -1), (true, 1)]);
        assert_eq!(p.stats["forced_eject"], 1);
    }

    #[test]
    fn court_change_and_transform() {
        let p = parse(&["|turn|1", "|move|p2a: Gholdengo|Stealth Rock|p1a: Garchomp", "|-sidestart|p1: Alice|move: Stealth Rock",
            "|move|p1a: Garchomp|Court Change|p1a: Garchomp", "|-swapsideconditions", "|upkeep", "|turn|2",
            "|move|p2a: Gholdengo|Make It Rain|p1a: Garchomp", "|upkeep", "|turn|3", "|switch|p1a: Ditto|Ditto|100/100",
            "|-transform|p1a: Ditto|p2a: Gholdengo", "|upkeep", "|turn|4", "|move|p1a: Ditto|Shadow Ball|p2a: Gholdengo", "|upkeep",
            "|turn|5", "|switch|p1a: Garchomp|Garchomp, M|100/100", "|upkeep", "|turn|6", "|switch|p1a: Ditto|Ditto|100/100", "|upkeep",
            "|turn|7"]);
        let own = |t: i64| p.records[0].iter().find(|r| turn(r) == t && !forced(r)).unwrap();
        assert_eq!((own(2).frame.field[12], own(2).frame.field[17]), (0.0, 1.0), "the rock moved to p2's side");
        let opp = p.records[1].iter().find(|r| turn(r) == 2).unwrap();
        assert_eq!((opp.frame.field[12], opp.frame.field[17]), (1.0, 0.0));
        // transformed: Gholdengo's revealed moves (+ Shadow Ball used meanwhile), then back to Ditto's own moves
        let revealed = ["Stealth Rock".to_string(), "Make It Rain".to_string(), String::new(), String::new()];
        assert_eq!(own(4).frame.moves, revealed);
        assert_eq!((own(4).move_idx, own(4).move_name.as_str()), (-1, "Shadow Ball"));
        assert_eq!(own(5).frame.moves[2], "Shadow Ball");
        assert!(own(7).frame.moves.iter().all(|m| m.is_empty()), "Ditto's own moveset stays unrevealed");
    }

    #[test]
    fn split_secret_lines_are_not_read() {
        let secret = ["|turn|1", "|split|p1", "|switch|p1a: Ditto|Ditto|250/250", "|switch|p1a: Ditto|Ditto|100/100",
            "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto", "|split|p1", "|-damage|p1a: Ditto|123/250", "|-damage|p1a: Ditto|49/100",
            "|upkeep", "|turn|2"];
        let public = ["|turn|1", "|switch|p1a: Ditto|Ditto|100/100", "|move|p2a: Gholdengo|Shadow Ball|p1a: Ditto",
            "|-damage|p1a: Ditto|49/100", "|upkeep", "|turn|2"];
        let (p, q) = (parse(&secret), parse(&public));
        for side in 0..2 {
            assert_eq!(p.records[side].len(), q.records[side].len());
            for (x, y) in p.records[side].iter().zip(&q.records[side]) {
                assert_eq!((&x.frame.feats, x.action_type, x.switch_idx), (&y.frame.feats, y.action_type, y.switch_idx));
            }
        }
        assert_eq!(p.records[0].iter().filter(|r| r.action_type == 1).count(), 2, "the lead + one switch, no phantom frame");
    }

    #[test]
    fn fixture_replays_convert_without_mismatches() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/pokemon");
        let mut frames = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let r: Value = serde_json::from_slice(&std::fs::read(e.unwrap().path()).unwrap()).unwrap();
            let p = parse_log(r["log"].as_str().unwrap());
            for k in ["mask_mismatch", "forced_other", "unlabelled_unanswered_request"] {
                assert!(!p.stats.contains_key(k), "{k} in {}", r["id"]);
            }
            for recs in &p.records {
                frames += recs.len();
                for x in recs {
                    assert!(x.action_type < 0 || x.legal_type[x.action_type as usize]);
                    assert!(x.action_type != 1 || x.legal_switch[x.switch_idx as usize]);
                    assert!(x.move_idx < 0 || !x.frame.moves[x.move_idx as usize].is_empty());
                }
            }
        }
        assert!(frames > 1500, "{frames}");
    }
}
