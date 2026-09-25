//! `.gior` → replay JSON with exactly the keys the Python pipeline wrote
//! (`replays/<id>.json` is consumed by the RL map pool and the viewers).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Move {
    pub index: i64,
    pub start: i64,
    pub end: i64,
    pub is50: Value, // 0/1 or bool in the wild; truthiness is what matters
    pub turn: i64,
}

impl Move {
    pub fn is_half(&self) -> bool {
        match &self.is50 {
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(false),
            Value::Null => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Afk {
    pub index: i64,
    pub turn: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chat {
    pub text: Value,
    pub prefix: Value,
    #[serde(rename = "playerIndex")]
    pub player_index: Value,
    pub turn: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Replay {
    pub version: i64,
    pub id: String,
    #[serde(rename = "mapWidth")]
    pub map_width: i64,
    #[serde(rename = "mapHeight")]
    pub map_height: i64,
    pub usernames: Vec<String>,
    pub stars: Value,
    pub cities: Vec<i64>,
    #[serde(rename = "cityArmies")]
    pub city_armies: Vec<i64>,
    pub generals: Vec<i64>,
    pub mountains: Vec<i64>,
    pub moves: Vec<Move>,
    pub afks: Vec<Afk>,
    pub teams: Value,
    pub map: Value,
    pub neutrals: Vec<Value>,
    #[serde(rename = "neutralArmies")]
    pub neutral_armies: Vec<Value>,
    pub swamps: Vec<Value>,
    pub chat: Vec<Chat>,
    #[serde(rename = "playerColors")]
    pub player_colors: Value,
    pub lights: Vec<Value>,
    pub speed: Value,
    pub city_density: Value,
    pub mountain_density: Value,
    pub swamp_density: Value,
    pub modifiers: Vec<Value>,
    pub observatories: Vec<Value>,
    pub lookouts: Vec<Value>,
    pub deserts: Vec<Value>,
    pub city_fairness: Value,
    pub spawn_fairness: Value,
    pub desert_density: Value,
    pub lookout_density: Value,
    pub observatory_density: Value,
}

fn arr_i64(v: &Value, what: &str) -> Result<Vec<i64>> {
    match v {
        Value::Null => Ok(Vec::new()),
        Value::Array(a) => a.iter().map(|x| x.as_i64().with_context(|| format!("{what}: non-integer {x}"))).collect(),
        _ => bail!("{what}: expected array"),
    }
}

fn arr_val(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.clone(),
        _ => Vec::new(),
    }
}

fn or_null(v: Option<&Value>) -> Value {
    v.cloned().unwrap_or(Value::Null)
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Deserialize the decompressed JSON array of a `.gior` file.
pub fn deserialize(obj: &Value) -> Result<Replay> {
    let a = obj.as_array().context("gior payload is not an array")?;
    let g = |i: usize| -> Value { or_null(a.get(i)) };
    let version = g(0).as_i64().context("version")?;
    let moves = arr_val(&g(10))
        .iter()
        .map(|m| {
            let m = m.as_array().context("move")?;
            Ok(Move {
                index: m[0].as_i64().context("move.index")?,
                start: m[1].as_i64().context("move.start")?,
                end: m[2].as_i64().context("move.end")?,
                is50: or_null(m.get(3)),
                turn: m[4].as_i64().context("move.turn")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let afks = arr_val(&g(11))
        .iter()
        .map(|x| {
            let x = x.as_array().context("afk")?;
            Ok(Afk { index: x[0].as_i64().context("afk.index")?, turn: x[1].as_i64().context("afk.turn")? })
        })
        .collect::<Result<Vec<_>>>()?;
    let usernames: Vec<String> = arr_val(&g(4)).iter().map(|u| u.as_str().unwrap_or("").to_string()).collect();
    let chat = arr_val(&g(17))
        .iter()
        .map(|c| {
            let c = c.as_array().cloned().unwrap_or_default();
            Chat { text: or_null(c.first()), prefix: or_null(c.get(1)), player_index: or_null(c.get(2)), turn: or_null(c.get(3)) }
        })
        .collect();
    let player_colors = if truthy(&g(18)) {
        g(18)
    } else {
        Value::Array((0..usernames.len() as i64).map(Value::from).collect())
    };
    let options: Vec<Value> = if truthy(&g(20)) {
        arr_val(&g(20))
    } else {
        vec![
            Value::from(1),
            Value::from(1.0),
            Value::from(1.0),
            Value::from(0.0),
            Value::from(0),
            Value::from(0),
            Value::from(0.0),
            Value::from(0.0),
            Value::from(0.0),
        ]
    };
    let opt = |i: usize| or_null(options.get(i));
    Ok(Replay {
        version,
        id: g(1).as_str().unwrap_or("").to_string(),
        map_width: g(2).as_i64().context("mapWidth")?,
        map_height: g(3).as_i64().context("mapHeight")?,
        usernames,
        stars: g(5),
        cities: arr_i64(&g(6), "cities")?,
        city_armies: arr_i64(&g(7), "cityArmies")?,
        generals: arr_i64(&g(8), "generals")?,
        mountains: arr_i64(&g(9), "mountains")?,
        moves,
        afks,
        teams: g(12),
        map: g(13),
        neutrals: arr_val(&g(14)),
        neutral_armies: arr_val(&g(15)),
        swamps: arr_val(&g(16)),
        chat,
        player_colors,
        lights: arr_val(&g(19)),
        speed: opt(0),
        city_density: opt(1),
        mountain_density: opt(2),
        swamp_density: opt(3),
        modifiers: arr_val(&g(21)),
        observatories: arr_val(&g(22)),
        lookouts: arr_val(&g(23)),
        deserts: arr_val(&g(24)),
        city_fairness: if version > 12 { opt(4) } else { Value::from(-1) },
        spawn_fairness: if version > 12 { opt(5) } else { Value::from(-1) },
        desert_density: if version > 12 { opt(6) } else { Value::from(0.0) },
        lookout_density: if version > 12 { opt(7) } else { Value::from(0.0) },
        observatory_density: if version > 12 { opt(8) } else { Value::from(0.0) },
    })
}

/// Raw `.gior` bytes → Replay.
pub fn parse_gior(bytes: &[u8]) -> Result<Replay> {
    let text = super::lz::decompress_uint8(bytes).context("LZ-string decompression failed")?;
    if text.is_empty() {
        bail!("LZ-string decompression produced no output");
    }
    let obj: Value = serde_json::from_str(&text).context("gior JSON")?;
    deserialize(&obj)
}
