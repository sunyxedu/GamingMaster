//! Pokémon Showdown battles: the public (spectator) sim protocol → per-side decision frames —
//! the parser and encoder shared by the converter (gmdata `pokemon convert`, whose module docs
//! describe the frames and converter versions) and the simulator (gmsim / env/vec_pokemon.py:
//! [`live_frame`] is the frame a live request gets, the converter's frame of that request).

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
pub const CONVERTER_VERSION: u32 = 5;
pub const TEAM: usize = 6;
pub const FEAT: usize = 18;
pub const FIELD: usize = 22;
pub const MOVES: usize = 4;
pub const F_HP: usize = 0;
pub const F_ACTIVE: usize = 1;
pub const F_FAINTED: usize = 2;
pub const F_STATUS: usize = 3;
pub const F_BOOSTS: usize = 10;
pub const F_REVEALED: usize = 15;
pub const F_TERA: usize = 16;
/// level / 100 (converter v5): 1 in the level-100 tiers, 0.05 in Little Cup, per Pokémon in
/// the random battles
pub const F_LEVEL: usize = 17;

/// The formats a sample may come from (converter v5, obs.format_id; append-only: an id never
/// changes meaning).  0 = a format not in the table.  All are Gen 9 6v6 singles with the
/// standard Gen 9 mechanics (team preview except the random battles); tiers differ only in
/// which Pokémon / moves / items are allowed.  Formats with other mechanics (National Dex:
/// Mega Evolution and Z-moves; Almost Any Ability, Balanced Hackmons, STABmons; 1v1, Battle
/// Stadium Singles' pick-3; doubles) are not in the table.
pub const FORMATS: [&str; 14] = ["", "gen9ou", "gen9uu", "gen9ru", "gen9nu", "gen9pu", "gen9zu", "gen9lc", "gen9ubers",
    "gen9ubersuu", "gen9monotype", "gen9anythinggoes", "gen9randombattle", "gen9randombattleblitz"];

/// The format id of a replay record: its `formatid`, else its display `format` normalised the
/// way Showdown's toID does ("[Gen 9] OU" → "gen9ou").
pub fn record_format(r: &Value) -> String {
    let raw = r.get("formatid").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
        .or_else(|| r.get("format").and_then(|v| v.as_str()))
        .unwrap_or("");
    norm(raw)
}

pub fn format_id(format: &str) -> i64 {
    FORMATS.iter().position(|&f| !f.is_empty() && f == format).unwrap_or(0) as i64
}

/// The level in a details string ("Garchomp, L80, M" → 80); None = not given (level 100).
pub fn details_level(details: &str) -> Option<u8> {
    details.split(',').skip(1).map(str::trim).find_map(|t| t.strip_prefix('L')?.parse::<u8>().ok())
}
/// obs.field_seq flag: the frame answers a forced-switch request
pub const FLD_FORCED: usize = 20;
/// obs.field_seq flag: the frame answers the team-preview (lead) request
pub const FLD_PREVIEW: usize = 21;
pub const BOOSTS: [&str; 5] = ["atk", "def", "spa", "spd", "spe"];
pub const SCREENS: [&str; 3] = ["reflect", "lightscreen", "auroraveil"];
/// moves whose user switches out afterwards (normalised names): its side gets a switch request
pub const SELF_SWITCH: [&str; 8] = ["uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "teleport", "batonpass",
    "shedtail"];
/// items / abilities that switch their holder out (`-enditem` / `-activate`, normalised)
pub const EJECT: [&str; 4] = ["ejectbutton", "ejectpack", "emergencyexit", "wimpout"];

pub fn norm(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

/// The exact move / species vocabularies (module docs, "Vocabulary"), embedded at build time.
pub const VOCAB_JSON: &str = include_str!("../../../games/pokemon/showdown_vocab.json");
pub const VOCAB_FORMAT: &str = "gm-pokemon-vocab.v1";
/// id of the empty name (no move / no Pokémon)
pub const ID_NONE: i64 = 0;
/// id of a name the vocabulary does not know
pub const ID_UNK: i64 = 1;
/// the team-preview placeholder suffix of a species name (`Urshifu-*`)
pub const HIDDEN_FORME: &str = "-*";

/// Showdown's `toID` (sim/dex-data.ts): lowercase, then keep only a-z0-9.
pub fn to_id(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit()).collect()
}

/// The vocabulary key of a species name as the log spells it (Showdown's species.get rule
/// for Nidoran♀ / ♂, `*` for a team-preview placeholder).
pub fn species_key(name: &str) -> String {
    let t = name.trim();
    let mut key = to_id(t);
    if key == "nidoran" && t.ends_with('♀') {
        key = "nidoranf".into();
    } else if key == "nidoran" && t.ends_with('♂') {
        key = "nidoranm".into();
    }
    if t.ends_with(HIDDEN_FORME) {
        key.push('*');
    }
    key
}

pub struct Vocab {
    pub version: String,
    /// key (entries and aliases) → id
    pub moves: HashMap<String, i64>,
    pub species: HashMap<String, i64>,
    pub items: HashMap<String, i64>,
    pub abilities: HashMap<String, i64>,
    pub types: HashMap<String, i64>,
    /// id counts, reserved ids included (the shard meta move_vocab / species_vocab /
    /// item_vocab / ability_vocab / type_vocab)
    pub n_moves: usize,
    pub n_species: usize,
    pub n_items: usize,
    pub n_abilities: usize,
    pub n_types: usize,
}

impl Vocab {
    pub fn parse(text: &str) -> Result<Vocab> {
        let doc: Value = serde_json::from_str(text).context("parse showdown_vocab.json")?;
        anyhow::ensure!(doc["format"] == VOCAB_FORMAT, "showdown_vocab.json: format {} is not {VOCAB_FORMAT}", doc["format"]);
        let ids = &doc["ids"];
        anyhow::ensure!(ids["none"] == ID_NONE && ids["unk"] == ID_UNK && ids["first"] == 2, "unexpected reserved ids {ids}");
        let table = |entries: &str, aliases: &str| -> Result<(HashMap<String, i64>, usize)> {
            let rows = doc[entries].as_array().with_context(|| format!("no {entries} table"))?;
            let mut m = HashMap::new();
            for (i, row) in rows.iter().enumerate() {
                let key = row[0].as_str().with_context(|| format!("{entries}[{i}]: no key"))?;
                anyhow::ensure!(m.insert(key.to_string(), 2 + i as i64).is_none(), "{entries}: duplicate key {key}");
            }
            let none = Value::Array(Vec::new());
            let alias_rows = if aliases.is_empty() { &none } else { &doc[aliases] };
            for a in alias_rows.as_array().with_context(|| format!("no {aliases} table"))? {
                let (Some(alias), Some(target)) = (a[0].as_str(), a[1].as_str()) else { anyhow::bail!("{aliases}: bad row {a}") };
                let id = *m.get(target).with_context(|| format!("{aliases}: {alias} → unknown {target}"))?;
                anyhow::ensure!(m.insert(alias.to_string(), id).is_none(), "{aliases}: {alias} shadows an entry");
            }
            Ok((m, rows.len() + 2))
        };
        let (moves, n_moves) = table("moves", "move_aliases")?;
        let (species, n_species) = table("species", "species_aliases")?;
        let (items, n_items) = table("items", "")?;
        let (abilities, n_abilities) = table("abilities", "")?;
        let (types, n_types) = table("types", "")?;
        let version = doc["version"].as_str().context("no version")?.to_string();
        Ok(Vocab { version, moves, species, items, abilities, types, n_moves, n_species, n_items, n_abilities, n_types })
    }

    /// The exact id of an item / ability / type name: 0 empty, 1 (UNK) unknown.
    pub fn named_id(table: &HashMap<String, i64>, name: &str) -> i64 {
        if name.trim().is_empty() {
            return ID_NONE;
        }
        *table.get(&to_id(name)).unwrap_or(&ID_UNK)
    }

    /// The exact id of a move name: 0 empty, 1 (UNK) unknown.
    pub fn move_id(&self, name: &str) -> i64 {
        if name.trim().is_empty() {
            return ID_NONE;
        }
        *self.moves.get(&to_id(name)).unwrap_or(&ID_UNK)
    }

    /// The exact id of a species name: 0 empty, 1 (UNK) unknown.
    pub fn species_id(&self, name: &str) -> i64 {
        if name.trim().is_empty() {
            return ID_NONE;
        }
        *self.species.get(&species_key(name)).unwrap_or(&ID_UNK)
    }
}

/// The vocabulary this binary was built with.
pub fn vocab() -> &'static Vocab {
    static V: OnceLock<Vocab> = OnceLock::new();
    V.get_or_init(|| Vocab::parse(VOCAB_JSON).expect("the embedded showdown_vocab.json is valid"))
}

/// Names that mapped to UNK in the written samples: name → ids emitted (obs + targets).
#[derive(Default)]
pub struct UnkNames {
    pub moves: BTreeMap<String, usize>,
    pub species: BTreeMap<String, usize>,
}

impl UnkNames {
    pub fn move_id(&mut self, v: &Vocab, name: &str) -> i64 {
        let id = v.move_id(name);
        if id == ID_UNK {
            *self.moves.entry(name.trim().to_string()).or_default() += 1;
        }
        id
    }
    pub fn species_id(&mut self, v: &Vocab, name: &str) -> i64 {
        let id = v.species_id(name);
        if id == ID_UNK {
            *self.species.entry(name.trim().to_string()).or_default() += 1;
        }
        id
    }
}

pub fn status_idx(s: &str) -> Option<usize> {
    ["", "brn", "par", "psn", "tox", "slp", "frz"].iter().position(|&x| x == s)
}

pub fn weather_idx(w: &str) -> usize {
    match w {
        "sunnyday" | "desolateland" => 0,
        "raindance" | "primordialsea" => 1,
        "sandstorm" => 2,
        "snow" | "snowscape" | "hail" => 3,
        _ => 4,
    }
}

pub fn terrain_idx(t: &str) -> Option<usize> {
    ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"].iter().position(|&x| x == t)
}

pub fn base_species(details: &str) -> String {
    details.split(',').next().unwrap_or("").trim().to_string()
}

/// Python `float(s)` for the forms seen in HP strings.
pub fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok().filter(|x| x.is_finite() || t.to_lowercase().contains("inf"))
}

pub fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let t2 = t.strip_prefix('+').unwrap_or(t);
    if t2.is_empty() || t2.starts_with('+') {
        return None;
    }
    t2.parse::<i64>().ok()
}

/// "78/100" / "0 fnt" / "45/100 par" → (frac, status, fainted)
pub fn parse_hp(hp: &str) -> (f64, Option<usize>, bool) {
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
pub struct Mon {
    pub species: String,
    pub hp: f64,
    pub status: usize,
    pub boosts: [i64; 5],
    pub fainted: bool,
    pub revealed: bool,
    pub tera: bool,
    pub moves: Vec<String>,
    /// moves while transformed: the target's revealed moves, then the ones used; None when
    /// not transformed (cleared when the Pokémon leaves the field)
    pub transformed: Option<Vec<String>>,
    /// the held item the log has revealed ("" = not revealed), `no_item` once it is known
    /// to hold none (consumed, knocked off, stolen, flung)
    pub item: String,
    pub no_item: bool,
    /// the ability the log has revealed ("" = not revealed)
    pub ability: String,
    /// the tera type the log has revealed ("" = not terastallized yet)
    pub tera_type: String,
    /// its level (details `L<n>`; 100 when not given)
    pub level: u8,
}

impl Mon {
    pub fn new(species: String) -> Self {
        Mon { species, hp: 1.0, level: 100, ..Default::default() }
    }
    pub fn with_details(details: &str) -> Self {
        let mut m = Mon::new(base_species(details));
        m.level = details_level(details).unwrap_or(100);
        m
    }
    /// the moveset in effect (the transformed one while transformed)
    pub fn moves_now(&self) -> &[String] {
        self.transformed.as_deref().unwrap_or(&self.moves)
    }
    pub fn reveal_move(&mut self, mv: &str) {
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
    pub fn leave_field(&mut self) {
        self.boosts = [0; 5];
        self.transformed = None;
    }
}

#[derive(Clone, Default)]
pub struct SideSt {
    pub name: String,
    pub team: Vec<Mon>,
    pub active: Option<usize>,
    pub nick2slot: HashMap<String, usize>,
    pub nick_order: Vec<String>,
    pub hazards: HashMap<String, i64>,
    /// the team came from team preview (`|poke|` lines): slots past it are empty, not unknown
    pub previewed: bool,
}

impl SideSt {
    pub fn claimed(&self, i: usize) -> bool {
        self.nick2slot.values().any(|&v| v == i)
    }
    pub fn assign(&mut self, nick: &str, i: usize) -> usize {
        if !self.nick2slot.contains_key(nick) {
            self.nick_order.push(nick.to_string());
        }
        self.nick2slot.insert(nick.to_string(), i);
        i
    }
    pub fn slot_for(&mut self, nick: &str, details: &str) -> usize {
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
            self.team.push(Mon::with_details(details));
            let i = self.team.len() - 1;
            return self.assign(nick, i);
        }
        let i = self.active.unwrap_or(0);
        self.assign(nick, i)
    }
    pub fn slot_of(&self, nick: &str) -> Option<usize> {
        self.nick2slot.get(nick).copied().or(self.active).filter(|&s| s < self.team.len())
    }
}

#[derive(Clone)]
pub struct Frame {
    pub species: Vec<String>,
    pub moves: Vec<String>,
    pub feats: Vec<f32>,
    pub mask: Vec<bool>,
    pub field: Vec<f32>,
    pub active: i64,
    /// every slot's revealed info (own slots 0..6, the opponent's 6..12): the first MOVES
    /// moves (the transformed set while transformed), held item, no item, ability, tera type
    pub slot_moves: Vec<String>,
    pub slot_item: Vec<String>,
    pub slot_no_item: Vec<bool>,
    pub slot_ability: Vec<String>,
    pub slot_tera: Vec<String>,
}

/// The request a frame answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Turn,
    Forced,
    Preview,
}

/// One decision frame: the snapshot, its legality masks, and the label once seen.
#[derive(Clone)]
pub struct Rec {
    pub frame: Frame,
    /// −1 until a choice is observed
    pub action_type: i64,
    pub move_idx: i64,
    pub move_name: String,
    pub switch_idx: i64,
    pub tera: i64,
    pub legal_type: [bool; 2],
    pub legal_switch: [bool; TEAM],
}

impl Rec {
    pub fn label_move(&mut self, mv: &str, tera: bool) {
        let n = norm(mv);
        self.action_type = 0;
        self.move_idx = self.frame.moves.iter().take(MOVES).position(|m| !m.is_empty() && norm(m) == n).map_or(-1, |i| i as i64);
        self.move_name = mv.to_string();
        self.tera = tera as i64;
    }
    /// Labels a switch; false when the label is outside the frame's masks.  The masks stay
    /// what the snapshot allowed (never label-dependent): the decoder gives such a target no
    /// weight under a legality mask.
    pub fn label_switch(&mut self, slot: usize) -> bool {
        self.action_type = 1;
        self.switch_idx = slot as i64;
        slot < TEAM && self.legal_switch[slot] && self.legal_type[1]
    }
}

#[derive(Clone)]
pub struct Battle {
    pub sides: [SideSt; 2],
    pub weather: Option<usize>,
    pub terrain: Option<usize>,
    pub turn: i64,
    pub winner: Option<usize>,
}

/// The state a forced-switch request was answered in: the battle at the last decision commit
/// (`|t:|`, logged by Showdown's turnLoop when the chosen actions start) if it came after the
/// request's cause (`since`), else the current state (logs without commit markers).  The
/// switch action's own switch-out effects (Natural Cure, the end of Neutralizing Gas or a
/// primal weather, ...) are logged after the commit but before the `|switch|` line.
pub fn request_state<'a>(b: &'a Battle, commit: &'a Option<(usize, Battle)>, since: usize) -> &'a Battle {
    match commit {
        Some((at, st)) if *at > since => st,
        _ => b,
    }
}

pub fn side_idx(pos: &str) -> Option<usize> {
    match pos {
        "p1" => Some(0),
        "p2" => Some(1),
        _ => None,
    }
}

pub fn ident(tok: &str) -> (String, String) {
    let (pos, nick) = tok.split_once(':').unwrap_or((tok, ""));
    (pos.trim().chars().take(2).collect(), nick.trim().to_string())
}

pub fn encode(b: &Battle, side: usize) -> Frame {
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
            f[F_LEVEL] = p.level as f32 / 100.0;
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
    let (mut slot_moves, mut slot_item, mut slot_no_item, mut slot_ability, mut slot_tera) =
        (vec![String::new(); 2 * TEAM * MOVES], vec![String::new(); 2 * TEAM], vec![false; 2 * TEAM],
         vec![String::new(); 2 * TEAM], vec![String::new(); 2 * TEAM]);
    for (base, s) in [(0, &b.sides[side]), (TEAM, &b.sides[other])] {
        for (i, p) in s.team.iter().take(TEAM).enumerate() {
            let j = base + i;
            for (k, m) in p.moves_now().iter().take(MOVES).enumerate() {
                slot_moves[j * MOVES + k] = m.clone();
            }
            slot_item[j] = p.item.clone();
            slot_no_item[j] = p.no_item;
            slot_ability[j] = p.ability.clone();
            slot_tera[j] = p.tera_type.clone();
        }
    }
    Frame { species, moves, feats, mask, field: fld, active: own.active.unwrap_or(0) as i64, slot_moves, slot_item,
            slot_no_item, slot_ability, slot_tera }
}

/// The `item: NAME` / `ability: NAME` an argument names (the effect of a `[from]` tag or of
/// `-activate` / `-block` lines), as (is_item, name).
pub fn effect_of(arg: &str) -> Option<(bool, String)> {
    let t = arg.trim().trim_start_matches("[from]").trim();
    if let Some(n) = t.strip_prefix("item:") {
        return Some((true, n.trim().to_string()));
    }
    t.strip_prefix("ability:").map(|n| (false, n.trim().to_string()))
}

/// The Pokémon an `[of] POKEMON` tag of the line names.
pub fn of_arg<'a>(args: &'a [&'a str]) -> Option<&'a str> {
    args.iter().find_map(|x| x.trim().strip_prefix("[of]").map(|v| v.trim()))
}

/// Reveals the items / abilities a line names (converter v4): `-item` / `-enditem` /
/// `-ability` lines, and every `[from] item: X` / `[from] ability: X` (or bare `item: X` /
/// `ability: X` effect argument) of any line — the holder is the `[of]` Pokémon when the tag
/// comes with one (Rocky Helmet, Rough Skin, Drizzle: the effect's source), else the line's
/// first Pokémon.  Only what the public log shows; nothing is inferred.
pub fn reveal_effects(b: &mut Battle, cmd: &str, args: &[&str]) {
    let mon = |b: &mut Battle, tok: &str| -> Option<(usize, usize)> {
        let (pos, nick) = ident(tok);
        let pi = side_idx(&pos)?;
        Some((pi, b.sides[pi].slot_of(&nick)?))
    };
    let first = args.first().copied().unwrap_or("");
    match cmd {
        "-item" if args.len() >= 2 => {
            if let Some((pi, sl)) = mon(b, first) {
                let p = &mut b.sides[pi].team[sl];
                p.item = args[1].trim().to_string();
                p.no_item = false;
            }
            return;
        }
        "-enditem" if args.len() >= 2 => {
            if let Some((pi, sl)) = mon(b, first) {
                let p = &mut b.sides[pi].team[sl];
                p.item = args[1].trim().to_string();
                p.no_item = true;
            }
            return;
        }
        "-ability" if args.len() >= 2 => {
            // |-ability|POKEMON|ABILITY|[from] ability: Trace|[of] SOURCE: POKEMON traced
            // SOURCE's ABILITY — its own is Trace, the source's ABILITY
            let traced = args[2..].iter().any(|x| effect_of(x) == Some((false, "Trace".into())));
            if let Some((pi, sl)) = mon(b, first) {
                b.sides[pi].team[sl].ability = if traced { "Trace".into() } else { args[1].trim().to_string() };
            }
            if traced {
                if let Some((qi, sl)) = of_arg(&args[2..]).and_then(|o| mon(b, o)) {
                    b.sides[qi].team[sl].ability = args[1].trim().to_string();
                }
            }
            return;
        }
        _ => {}
    }
    let of = of_arg(args);
    for (k, arg) in args.iter().enumerate() {
        let Some((is_item, name)) = effect_of(arg) else { continue };
        if name.is_empty() {
            continue;
        }
        let tagged = arg.trim_start().starts_with("[from]");
        // a [from] tag with an [of]: the effect's source holds it; else the line's Pokémon
        let holder = if tagged { of.unwrap_or(first) } else if k > 0 { first } else { continue };
        if let Some((pi, sl)) = mon(b, holder) {
            let p = &mut b.sides[pi].team[sl];
            if is_item {
                if p.item.is_empty() || p.no_item {
                    p.item = name;
                    p.no_item = false;
                }
            } else if p.ability.is_empty() {
                p.ability = name;
            }
        }
    }
}

/// The frame of `side` answering a `phase` request in the current state (label unset).
pub fn snapshot(b: &Battle, side: usize, phase: Phase) -> Rec {
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

pub const IGNORED: [&str; 32] = ["", "j", "l", "c", "c:", "chat", "raw", "html", "uhtml", "inactive", "inactiveoff", "t:", "gametype", "gen",
    "tier", "rated", "rule", "clearpoke", "start", "teamsize", "player", "-message", "message", "debug",
    "error", "seed", "choice", "unlink", "-hint", "-anim", "n", "join"];

pub struct Parse {
    pub records: [Vec<Rec>; 2],
    pub winner: Option<usize>,
    pub names: [String; 2],
    /// frame / label statistics (frames per phase, unlabelled reasons, forced-switch causes)
    pub stats: BTreeMap<String, usize>,
    pub unknown: BTreeMap<String, usize>,
    /// the battle state at the end of the log (a live battle: the state its requests see)
    pub battle: Battle,
}

pub fn parse_log(log: &str) -> Parse {
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
    // the last decision commit (`|t:|`): (line number, battle state there) — forced-switch
    // frames are that state when it follows the request's cause (request_state)
    let mut commit: Option<(usize, Battle)> = None;
    let mut line_no = 0usize;
    let mut turn_at = 0usize; // line number of the last |turn|
    // line number of the side's latest forced-switch cause (its active fainted, an eject
    // activation) or move (a pivot, or the last action before an unrecognised forced switch)
    let mut cause_at = [0usize; 2];
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
        line_no += 1;
        if !matches!(cmd, "t:" | "split" | "player" | "poke" | "switch" | "drag" | "replace") {
            reveal_effects(&mut b, cmd, args);
        }
        match cmd {
            "t:" => {
                // Showdown's turnLoop logs `|` + `|t:|TIME` when the answered request's actions
                // start: nothing is logged between a request and its commit
                commit = Some((line_no, b.clone()));
                continue;
            }
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
                        s.team.push(Mon::with_details(a(1)));
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
                            // replacement at once: all their frames are the state it was answered
                            // in (the commit, else this pre-switch state)
                            for s in 0..2 {
                                if open[s].is_none() && (s == pi || need_replace[s] || pending_eject[s]) {
                                    let st = request_state(&b, &commit, cause_at[s].max(turn_at));
                                    records[s].push(snapshot(st, s, Phase::Forced));
                                    open[s] = Some(records[s].len() - 1);
                                    stat!("frames_forced");
                                }
                            }
                            target = open[pi].take();
                        } else {
                            let st = request_state(&b, &commit, cause_at[pi].max(turn_at));
                            records[pi].push(snapshot(st, pi, Phase::Forced));
                            target = Some(records[pi].len() - 1);
                            stat!("frames_forced");
                        }
                    } else if acted[pi].is_none() && turn_rec[pi].is_some() {
                        acted[pi] = Some("switch"); // voluntary: the turn's choice
                        target = turn_rec[pi];
                    } else {
                        // after the side acted, with no recognised cause (an unlisted pivot)
                        stat!("forced_other");
                        let st = request_state(&b, &commit, cause_at[pi].max(turn_at));
                        records[pi].push(snapshot(st, pi, Phase::Forced));
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
                // team preview hid the forme (`Urshifu-*`): the switch-in names it
                let sp = base_species(a(1));
                if p.species.ends_with(HIDDEN_FORME) && !sp.is_empty() {
                    p.species = sp;
                }
                if let Some(l) = details_level(a(1)) {
                    p.level = l;
                }
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
                cause_at[pi] = line_no;
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
            "-start" if args.len() >= 2 && norm(a(1).rsplit(':').next().unwrap_or("")) == "encore" => {
                // Encore replaces the target's chosen move with the encored one (Showdown's
                // onOverrideAction; the |move| line carries no [from]): landing before the target
                // acted this turn, the move it then logs is not necessarily its choice.  (Switches
                // precede every move, so the choice was a move — which one stays unknown.)
                let (pos, _) = ident(a(0));
                if let Some(pi) = side_idx(&pos) {
                    if acted[pi].is_none() && turn_rec[pi].is_some() {
                        acted[pi] = Some("encore_override");
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
                        cause_at[pi] = line_no;
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
                } else if cmd == "-heal" && hp > 0.0 && p.fainted {
                    // Revival Blessing: Showdown clears the revived Pokémon's status, and fainting
                    // already ended its terastallization
                    p.fainted = false;
                    p.status = 0;
                    p.tera = false;
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
            "-invertboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    for x in b.sides[pi].team[slot].boosts.iter_mut() {
                        *x = -*x;
                    }
                }
                continue;
            }
            "-copyboost" | "-swapboost" if args.len() >= 2 => {
                // |-copyboost|RECEIVER|SOURCE[|STATS]: RECEIVER takes SOURCE's stages (Psych Up);
                // |-swapboost|A|B[|STATS]: A and B exchange them (Guard / Power / Heart Swap);
                // every stat when no list is given
                let ((pos, nick), (pos2, nick2)) = (ident(a(0)), ident(a(1)));
                let (Some(pi), Some(qi)) = (side_idx(&pos), side_idx(&pos2)) else { continue };
                let (Some(x), Some(y)) = (b.sides[pi].slot_of(&nick), b.sides[qi].slot_of(&nick2)) else { continue };
                let stats: Vec<usize> = match args.get(2).filter(|s| !s.trim().is_empty() && !s.trim_start().starts_with('[')) {
                    Some(list) => list.split(',').filter_map(|s| BOOSTS.iter().position(|&k| k == s.trim())).collect(),
                    None => (0..BOOSTS.len()).collect(),
                };
                let (bx, by) = (b.sides[pi].team[x].boosts, b.sides[qi].team[y].boosts);
                for k in stats {
                    b.sides[pi].team[x].boosts[k] = by[k];
                    if cmd == "-swapboost" {
                        b.sides[qi].team[y].boosts[k] = bx[k];
                    }
                }
                continue;
            }
            "detailschange" if args.len() >= 2 => {
                // |detailschange|POKEMON|DETAILS: a permanent forme change (Mega Evolution, Primal
                // Reversion, Ogerpon / Terapagos tera formes, Palafin-Hero, ...) or its regression
                // when the Pokémon faints (`[silent]`): the slot shows the new forme from here on
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let sp = base_species(a(1));
                if let (Some(slot), false) = (b.sides[pi].slot_of(&nick), sp.is_empty()) {
                    b.sides[pi].team[slot].species = sp;
                    if let Some(l) = details_level(a(1)) {
                        b.sides[pi].team[slot].level = l;
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
                        cause_at[pi] = line_no;
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
                    b.sides[pi].team[slot].tera_type = a(1).trim().to_string();
                }
                tera_pending[pi] = true;
                continue;
            }
            "turn" if !args.is_empty() => {
                let Some(n) = py_int(a(0)) else { continue };
                close_turn!();
                turn_at = line_no;
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
    Parse { records, winner: b.winner, names: [b.sides[0].name.clone(), b.sides[1].name.clone()], stats, unknown, battle: b }
}

/// The frame of a live battle's request: `side`'s state at the end of `log` (the public sim
/// protocol so far, the request just issued) with `phase`'s flags — the frame the converter
/// writes for that request once the battle's log is complete: a turn's frame is the state at
/// its `|turn|` line, a forced switch's the state when Showdown issues the request (nothing
/// is logged between the request and its answer's commit), team preview's the state before
/// the leads.  Its `legal_*` are the log's; a live bot takes the request's (exact: trapping,
/// PP, disabled moves).
pub fn live_frame(log: &str, side: usize, phase: Phase) -> Rec {
    snapshot(&parse_log(log).battle, side, phase)
}

/// The identity of `side`'s team slots in a battle state: (nickname, species) per slot the
/// log has shown (a live bot maps its request's Pokémon to the frame's slots by these).
pub fn slot_idents(b: &Battle, side: usize) -> Vec<(String, String)> {
    let s = &b.sides[side];
    (0..s.team.len()).map(|i| {
        let nick = s.nick2slot.iter().find(|(_, &v)| v == i).map(|(k, _)| k.clone()).unwrap_or_default();
        (nick, s.team[i].species.clone())
    }).collect()
}

/// A frame's names as exact vocabulary ids — the obs tensors of one frame (names the
/// vocabulary does not know are UNK and counted in `unk`).
pub struct FrameIds {
    pub species: Vec<i64>,
    pub moves: Vec<i64>,
    pub slot_moves: Vec<i64>,
    pub slot_item: Vec<i64>,
    pub slot_ability: Vec<i64>,
    pub slot_tera: Vec<i64>,
}

pub fn frame_ids(f: &Frame, unk: &mut UnkNames) -> FrameIds {
    let voc = vocab();
    FrameIds {
        species: f.species.iter().map(|s| unk.species_id(voc, s)).collect(),
        moves: f.moves.iter().map(|m| unk.move_id(voc, m)).collect(),
        slot_moves: f.slot_moves.iter().map(|m| unk.move_id(voc, m)).collect(),
        slot_item: f.slot_item.iter().map(|x| Vocab::named_id(&voc.items, x)).collect(),
        slot_ability: f.slot_ability.iter().map(|x| Vocab::named_id(&voc.abilities, x)).collect(),
        slot_tera: f.slot_tera.iter().map(|x| Vocab::named_id(&voc.types, x)).collect(),
    }
}

