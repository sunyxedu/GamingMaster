//! Parse the RoyaleAPI replay Parquet rows (`royaleapi-battle-replay.v1`
//! payloads) into a small typed structure. Only what the frame builder needs
//! is kept; the 5-way redundant coordinate block collapses to one 18×32 cell.

use crate::catalog::Catalog;
use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideId {
    Team,
    Opp,
}

#[derive(Debug, Clone)]
pub struct DeckCard {
    pub key: String,
    pub id: u16,
    pub form: u8,
    pub level: u8,
}

#[derive(Debug, Clone)]
pub struct Side {
    pub crowns: u8,
    pub deck: Vec<DeckCard>,
    pub tower_key: String,
    pub final_tower_hp: Option<(f32, f32, f32)>, // king, left, right
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventKind {
    /// card id, form at play (0 unknown/base), grid x, grid y (team-frame: team half is y < 16)
    Play { card: u16, form: u8, gx: i8, gy: i8 },
    /// champion / hero ability: the card whose ability it is, resolved from the public
    /// record up to the event ([`resolve_ability_sources`]); `unknown_id` when ambiguous
    Ability { card: u16 },
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub tick: u32,
    pub side: SideId,
    pub kind: EventKind,
}

#[derive(Debug, Clone)]
pub struct Replay {
    pub tag: String,
    pub battle_type: String,
    pub game_mode: String,
    /// team perspective: +1 victory, -1 defeat, 0 draw
    pub result: i8,
    pub team: Side,
    pub opp: Side,
    pub duration_s: f64,
    pub events: Vec<Event>,
}

// ---- payload JSON (only the fields we read) ----------------------------------

#[derive(Deserialize)]
struct Payload {
    battle: Battle,
    events: Vec<RawEvent>,
    replay: ReplayMeta,
}

#[derive(Deserialize)]
struct Battle {
    #[serde(default)]
    battle_type: String,
    #[serde(default)]
    game_mode: String,
    #[serde(default)]
    result: String,
    team: BattleSide,
    opponent: BattleSide,
}

#[derive(Deserialize)]
struct BattleSide {
    #[serde(default)]
    crowns: u8,
    players: Vec<Player>,
}

#[derive(Deserialize)]
struct Player {
    deck: Vec<RawCard>,
    #[serde(default)]
    tower_card: Option<RawCard>,
    #[serde(default)]
    final_tower_hitpoints: Option<TowerHp>,
}

#[derive(Deserialize)]
struct TowerHp {
    #[serde(default)]
    king: f32,
    #[serde(default)]
    princess_left: f32,
    #[serde(default)]
    princess_right: f32,
}

#[derive(Deserialize)]
struct RawCard {
    card_key: String,
    #[serde(default)]
    level: u8,
}

#[derive(Deserialize)]
struct RawEvent {
    kind: String,
    side: String,
    #[serde(default)]
    card_key: Option<String>,
    replay_tick_20hz: u32,
    #[serde(default)]
    form_at_play: Option<String>,
    #[serde(default)]
    coordinates: Option<Coordinates>,
    #[serde(default)]
    deck_card_key_candidates: Vec<String>,
    #[serde(default)]
    ability_source_candidates: Vec<String>,
}

#[derive(Deserialize)]
struct Coordinates {
    grid_cell_floor: Cell,
    #[serde(default)]
    inside_18x32_arena: Option<bool>,
}

#[derive(Deserialize)]
struct Cell {
    x: i32,
    y: i32,
}

#[derive(Deserialize)]
struct ReplayMeta {
    duration: Duration,
}

#[derive(Deserialize)]
struct Duration {
    #[serde(default)]
    timeline_seconds: f64,
    #[serde(default)]
    display_seconds: f64,
}

fn side_of(p: &BattleSide, cat: &Catalog) -> Result<Side> {
    let pl = p.players.first().context("side without players")?;
    let deck = pl
        .deck
        .iter()
        .map(|c| {
            let (id, form) = cat.resolve(&c.card_key);
            DeckCard { key: c.card_key.clone(), id, form, level: c.level }
        })
        .collect();
    Ok(Side {
        crowns: p.crowns,
        deck,
        tower_key: pl.tower_card.as_ref().map(|t| t.card_key.clone()).unwrap_or_default(),
        final_tower_hp: pl.final_tower_hitpoints.as_ref().map(|t| (t.king, t.princess_left, t.princess_right)),
    })
}

pub fn parse_payload(tag: &str, payload_json: &str, cat: &Catalog) -> Result<Replay> {
    let p: Payload = serde_json::from_str(payload_json).context("payload_json")?;
    let team = side_of(&p.battle.team, cat)?;
    let opp = side_of(&p.battle.opponent, cat)?;
    let result = match p.battle.result.as_str() {
        "victory" => 1,
        "defeat" => -1,
        _ => 0,
    };
    // (event, ability source candidates as card ids — empty for plays)
    let mut parsed: Vec<(Event, Vec<u16>)> = Vec::with_capacity(p.events.len());
    for e in &p.events {
        let side = match e.side.as_str() {
            "team" => SideId::Team,
            "opponent" => SideId::Opp,
            _ => continue,
        };
        // the deck key carries the form the player owns; the event key is the base card
        let key_with_form = e.deck_card_key_candidates.first().or(e.card_key.as_ref());
        let (kind, candidates) = match e.kind.as_str() {
            "play_card" => {
                let Some(key) = key_with_form else { continue };
                let (card, mut form) = cat.resolve(key);
                if let Some(f) = &e.form_at_play {
                    if f == "evolved" || f.starts_with("ev") {
                        form = form.max(1);
                    }
                }
                let Some(c) = &e.coordinates else { continue };
                if c.inside_18x32_arena == Some(false) {
                    continue;
                }
                let gx = c.grid_cell_floor.x.clamp(0, 17) as i8;
                let gy = c.grid_cell_floor.y.clamp(0, 31) as i8;
                (EventKind::Play { card, form, gx, gy }, Vec::new())
            }
            "activate_ability" => (
                EventKind::Ability { card: cat.unknown_id },
                e.ability_source_candidates.iter().map(|k| cat.resolve(k).0).collect(),
            ),
            _ => continue,
        };
        parsed.push((Event { tick: e.replay_tick_20hz, side, kind }, candidates));
    }
    parsed.sort_by_key(|(e, _)| e.tick);
    let events = resolve_ability_sources(parsed, cat.unknown_id);
    let duration_s = if p.replay.duration.timeline_seconds > 0.0 {
        p.replay.duration.timeline_seconds
    } else {
        p.replay.duration.display_seconds
    };
    Ok(Replay {
        tag: tag.to_string(),
        battle_type: p.battle.battle_type,
        game_mode: p.battle.game_mode,
        result,
        team,
        opp,
        duration_s,
        events,
    })
}

/// Attribute every ability event causally. RoyaleAPI's `ability_source_candidates`
/// is the acting side's ability-capable DECK cards (sorted by key, one list per side
/// and match), so its first entry can name a card that side has not played yet —
/// private deck information. The source is instead the candidate the side has
/// already played (a play strictly before the event's tick): the unique one, or
/// `unknown` when none or several qualify. Public record up to the event only;
/// nothing later decides it. `parsed` is sorted by tick.
pub fn resolve_ability_sources(parsed: Vec<(Event, Vec<u16>)>, unknown: u16) -> Vec<Event> {
    let side_ix = |s: SideId| if s == SideId::Team { 0 } else { 1 };
    let mut played: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
    let mut applied = 0usize; // parsed[..applied] (all ticks < the current one) are in `played`
    let mut out = Vec::with_capacity(parsed.len());
    for i in 0..parsed.len() {
        let tick = parsed[i].0.tick;
        while applied < parsed.len() && parsed[applied].0.tick < tick {
            if let EventKind::Play { card, .. } = parsed[applied].0.kind {
                played[side_ix(parsed[applied].0.side)].push(card);
            }
            applied += 1;
        }
        let (mut e, candidates) = (parsed[i].0, &parsed[i].1);
        if let EventKind::Ability { .. } = e.kind {
            let mine = &played[side_ix(e.side)];
            let mut hits: Vec<u16> = candidates.iter().copied().filter(|c| mine.contains(c)).collect();
            hits.sort_unstable();
            hits.dedup();
            e.kind = EventKind::Ability { card: if hits.len() == 1 { hits[0] } else { unknown } };
        }
        out.push(e);
    }
    out
}

/// Accessor over a Utf8 or LargeUtf8 column: index -> Option<&str>.
fn string_column<'a>(col: &'a dyn Array) -> Option<Box<dyn Fn(usize) -> Option<&'a str> + 'a>> {
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Some(Box::new(move |i| if a.is_null(i) { None } else { Some(a.value(i)) }));
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Some(Box::new(move |i| if a.is_null(i) { None } else { Some(a.value(i)) }));
    }
    None
}

/// Raw (tag, payload_json) rows of one Parquet part file.
pub fn read_rows(path: &Path) -> Result<Vec<(String, String)>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let tag_idx = schema.index_of("replay_tag").context("column replay_tag")?;
    let payload_idx = schema.index_of("payload_json").context("column payload_json")?;
    let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), [tag_idx, payload_idx]);
    let reader = builder.with_projection(mask).with_batch_size(1024).build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let tags = batch.column_by_name("replay_tag").context("replay_tag")?;
        let payloads = batch.column_by_name("payload_json").context("payload_json")?;
        let tags = string_column(tags.as_ref()).context("replay_tag must be utf8 / large_utf8")?;
        let payloads = string_column(payloads.as_ref()).context("payload_json must be utf8 / large_utf8")?;
        for i in 0..batch.num_rows() {
            if let (Some(t), Some(p)) = (tags(i), payloads(i)) {
                out.push((t.to_string(), p.to_string()));
            }
        }
    }
    Ok(out)
}

/// Raw (tag, payload_json) rows of a JSON-lines file: one object per line with the tag
/// under `tag` / `replay_tag` and the payload under `payload` (an object or a JSON
/// string) / `payload_json` — the format of `tests/fixtures/clash_royale/replays.jsonl`.
pub fn read_jsonl(path: &Path) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), i + 1))?;
        let tag = v.get("tag").or_else(|| v.get("replay_tag")).and_then(|t| t.as_str()).unwrap_or_default();
        let payload = match v.get("payload").or_else(|| v.get("payload_json")) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(p) => p.to_string(),
            None => anyhow::bail!("{}:{}: no payload", path.display(), i + 1),
        };
        out.push((tag.to_string(), payload));
    }
    Ok(out)
}

/// Rows of one input part: Parquet (the dataset) or JSON lines ([`read_jsonl`]).
pub fn read_input_rows(path: &Path) -> Result<Vec<(String, String)>> {
    if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
        read_jsonl(path)
    } else {
        read_rows(path)
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    /// A tiny synthetic payload in the dataset's schema.
    pub fn payload(team_deck: &[&str], opp_deck: &[&str], plays: &[(&str, &str, u32, i32, i32)], duration: f64) -> String {
        let deck = |d: &[&str]| -> String {
            d.iter()
                .map(|k| format!(r#"{{"card_key":"{k}","level":15,"name":"{k}"}}"#))
                .collect::<Vec<_>>()
                .join(",")
        };
        let events = plays
            .iter()
            .enumerate()
            .map(|(i, (side, key, tick, x, y))| {
                format!(
                    r#"{{"kind":"play_card","side":"{side}","card_key":"{key}","card_play_number":1,"replay_tick_20hz":{tick},
                       "time_seconds":{},"form_at_play":"unknown","deck_card_key_candidates":["{key}"],"ability_source_candidates":[],
                       "coordinates":{{"grid_cell_floor":{{"x":{x},"y":{y}}},"inside_18x32_arena":true}},"source_index":{i}}}"#,
                    *tick as f64 / 20.0
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"battle":{{"battle_type":"pathOfLegend","game_mode":"Ranked","result":"victory",
                 "team":{{"crowns":1,"players":[{{"deck":[{}],"tower_card":{{"card_key":"tower-princess","level":15,"name":"Tower Princess"}},
                          "final_tower_hitpoints":{{"king":4000,"princess_left":0,"princess_right":2000}}}}]}},
                 "opponent":{{"crowns":0,"players":[{{"deck":[{}],"tower_card":{{"card_key":"tower-princess","level":15,"name":"Tower Princess"}}}}]}}}},
               "events":[{}],"replay":{{"duration":{{"timeline_seconds":{duration},"display_seconds":{duration}}}}},
               "schema_version":"royaleapi-battle-replay.v1","source":{{"site":"RoyaleAPI"}}}}"#,
            deck(team_deck),
            deck(opp_deck),
            events
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use std::path::PathBuf;

    fn cat() -> Catalog {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../catalog/cards.json");
        Catalog::load(&p).unwrap()
    }

    #[test]
    fn parses_synthetic_payload() {
        let cat = cat();
        let team = ["hog-rider", "ice-spirit", "ice-golem", "skeletons-ev1", "cannon-ev1", "musketeer-hero", "fireball", "the-log"];
        let opp = ["giant", "witch", "knight", "archers", "zap", "minions", "fireball", "tornado"];
        let js = fixtures::payload(&team, &opp, &[("team", "hog-rider", 40, 9, 14), ("opponent", "giant", 60, 8, 20)], 100.0);
        let r = parse_payload("t1", &js, &cat).unwrap();
        assert_eq!(r.result, 1);
        assert_eq!(r.team.deck.len(), 8);
        assert_eq!(r.team.deck[3].form, 1); // skeletons-ev1
        assert_eq!(r.team.deck[5].form, crate::catalog::FORM_HERO);
        assert_eq!(r.events.len(), 2);
        match r.events[0].kind {
            EventKind::Play { card, gx, gy, .. } => {
                assert_eq!(card, cat.resolve("hog-rider").0);
                assert_eq!((gx, gy), (9, 14));
            }
            _ => panic!(),
        }
        assert_eq!(r.team.final_tower_hp, Some((4000.0, 0.0, 2000.0)));
    }

    #[test]
    fn ability_source_is_the_unique_candidate_played_before() {
        const UNK: u16 = 999;
        let play = |tick, side, card| (Event { tick, side, kind: EventKind::Play { card, form: 0, gx: 0, gy: 0 } }, vec![]);
        let abil = |tick, side, cands: &[u16]| (Event { tick, side, kind: EventKind::Ability { card: UNK } }, cands.to_vec());
        let (t, o) = (SideId::Team, SideId::Opp);
        let parsed = vec![
            abil(5, t, &[7, 9]),  // nothing played yet
            play(10, t, 7),
            abil(10, t, &[7, 9]), // same tick as the play: not before it
            play(12, o, 9),       // the other side's 9 does not count for team
            abil(20, t, &[7, 9]), // 7 is the only played candidate
            play(30, t, 9),
            abil(40, t, &[7, 9]), // both played: ambiguous
            abil(41, o, &[9, 3]), // opponent: its own 9
        ];
        let got: Vec<u16> = resolve_ability_sources(parsed, UNK)
            .iter()
            .filter_map(|e| match e.kind {
                EventKind::Ability { card } => Some(card),
                _ => None,
            })
            .collect();
        assert_eq!(got, vec![UNK, UNK, 7, UNK, 9]);
    }
}
