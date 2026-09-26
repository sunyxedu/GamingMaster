//! Xiangqi: PGN game records → from-to gm-shards (rules and frames:
//! `gmrules::games::xiangqi`, replayed and verified by `traj::convert`).
//!
//! Source: the Chinese Chess Practical Dataset (CCPD, Yu-Han Tseng and Bo-Nian Chen,
//! 2026, <https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset>, CC BY 4.0 —
//! attribution in `ATTRIBUTION.txt` next to the archive): 58,468 PGN files (Big5,
//! Chinese notation) — master games 1950–2013 filed three times (by player, by
//! tournament, by opening), computer games, and midgame / endgame / opening
//! fragments. 55,419 files start from the initial position; 29,090 distinct games.
//! `fetch` downloads the GitHub archive of a pinned commit (60 MB).
//!
//! ```sh
//! scripts/fetch_xiangqi.sh data/xiangqi ~/gm-data/xiangqi    # fetch + --hq train / val, ~1 min
//! # or step by step
//! gmdata xiangqi fetch --out ~/gm-data/xiangqi               # ccpd-368a47a9.zip + ATTRIBUTION.txt
//! gmdata xiangqi convert --input ~/gm-data/xiangqi/ccpd-368a47a9.zip --out data/xiangqi/train --hq --split train
//! gmdata xiangqi convert --input ~/gm-data/xiangqi/ccpd-368a47a9.zip --out data/xiangqi/val --hq --split val
//! ```
//!
//! Corpus (commit 368a47a9): `--hq` train 26,980 games / 2,230,893 frames (621 MB), val
//! 567 / 46,371 (13 MB); without `--hq` 28,126 games / 2,315,453 frames (644 MB). Every
//! converted record replays legally (`illegal` 0); rejected: 1 unparseable record (a
//! transcription error: the move leaves the king in check), 2 whose checkmate
//! contradicts the `[Result]`, 960 without any known result (mostly opening lines).
//!
//! Input: a `.zip`, a directory (every `.pgn` below it, sorted) or one `.pgn` file; a
//! file may hold several games. Text encodings: UTF-8, Big5 (CCPD) and GBK — detected
//! per file. Move notations, per token: Chinese (WXF Chinese: 炮二平五, 馬８進７,
//! 前炮平４, traditional or simplified characters, file numbers from each player's
//! right), WXF English (C2.5, H8+7, C+.5), ICCS (H2-E2, ranks 0-9 from Red's side) and
//! Fairy-Stockfish / UCI coordinates (h3e3, ranks 1-10; `[Format "ICCS"]` makes
//! dashless coordinates ICCS). Every move is resolved against the legal moves of the
//! position (an unknown, illegal or ambiguous token makes the record unparseable), then
//! `traj::convert` replays the whole game again with the rules and encodes the frames.
//!
//! Filters: games from the initial position only (`[FEN]` absent or the start FEN;
//! the rest counted as `other_start`); duplicates — the same moves, whatever the
//! headers say; the first copy in path order that is converted claims the game, so a
//! copy without a result never hides a copy with one (`duplicate_result_conflicts`
//! counts later copies whose result differs from the kept one's) — dropped;
//! `--min-plies`; `--hq`: master games only (CCPD's 大師對局 tree), ≥ 20 plies, known
//! result. `--split train|val` keeps the games whose move hash (FNV-1a of the moves:
//! every copy of a game on the same side) is (not) ≡ 0 mod `--val-mod` (default 50:
//! ~2 % validation), disjoint by construction.
//!
//! Results. Resignations, agreed draws and time forfeits end a record before the
//! rules do: the recorded result becomes the value target (`Record::final_outcome`,
//! samples marked `adjudicated`) — the `[Result]` header (1-0 / 0-1 / 1/2-1/2) or,
//! when it is `*` / missing, the result the `[Event]` text states in colour-anchored
//! words (先勝 / 先負 / 先和, a draw 和: [`event_result_red`]; `result_from_event`);
//! when the header and the text disagree the result is unknown (`result_conflict`).
//! Records without a known result are dropped as `truncated` unless
//! `--allow-truncated`. When the replay reaches a terminal state: by checkmate /
//! stalemate at the end of the record the recorded result must agree (else
//! `returns_mismatch`; an unknown result takes the rules'); by an adjudication rule
//! (repetition with perpetual check / chase, move limit, material) the rules' result
//! labels the game (`rules_end`; `rules_disagree` counts the records whose recorded
//! result differs) — and when the players went on after the rules ended the game (a
//! repetition nobody claimed — the arbiter's call in human play, automatic in the
//! simulator — or moves after a mate), the record is cut there (`cut`, `cut_plies`
//! dropped: 2,349 records / 97,652 plies, 4 % of the corpus; `--past-end reject`
//! drops such records instead). The frames therefore never show a position the
//! simulator would not reach.
//!
//! Teacher labels: the from-to schema labels its heads through its own targets, so
//! engine soft targets (e.g. Pikafish MultiPV) are added as `gmdata label` overlays —
//! see the README section "gmrules games" (xiangqi).

use crate::common;
use crate::traj::{self, ConvertOptions, Record};
use anyhow::{anyhow, bail, Context, Result};
use clap::{Subcommand, ValueEnum};
use gmrules::games::xiangqi::{
    self as xq, parse_uci, EndReason, Xiangqi, XiangqiConfig, ADVISOR, CANNON, ELEPHANT, FILES, HORSE, KING, RED,
    ROOK, SOLDIER, START_FEN,
};
use gmrules::{Event, Game};
use serde::Serialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The CCPD commit `fetch` downloads (pinned: the corpus the docs describe).
pub const CCPD_COMMIT: &str = "368a47a947773dd8692c026e286dd19b6277b993";
pub const CCPD_REPO: &str = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset";
pub const CCPD_LICENCE: &str = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)";
pub const CCPD_ATTRIBUTION: &str = "Yu-Han Tseng and Bo-Nian Chen (2026). Chinese Chess Practical Dataset (CCPD).";

// ---------------------------------------------------------------------------
// text
// ---------------------------------------------------------------------------

/// Characters of the Chinese move notation (traditional and simplified).
const MOVE_CHARS: &str = "進进退平車车俥馬马傌炮砲包兵卒相象仕士帥帅將将前後后中";

/// Decode a record file: UTF-8 when valid, else Big5 or GBK — whichever yields more
/// notation characters.
pub fn decode_text(bytes: &[u8]) -> String {
    let b = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    if let Ok(s) = std::str::from_utf8(b) {
        return s.to_string();
    }
    let score = |s: &str| s.chars().filter(|c| MOVE_CHARS.contains(*c)).count();
    let (big5, _, big5_bad) = encoding_rs::BIG5.decode(b);
    let (gbk, _, gbk_bad) = encoding_rs::GBK.decode(b);
    let (sb, sg) = (score(&big5), score(&gbk));
    if sg > sb || (sg == sb && big5_bad && !gbk_bad) { gbk.into_owned() } else { big5.into_owned() }
}

// ---------------------------------------------------------------------------
// PGN
// ---------------------------------------------------------------------------

/// One game of a PGN file.
#[derive(Clone, Debug, Default)]
pub struct PgnGame {
    pub headers: Vec<(String, String)>,
    pub tokens: Vec<String>,
}

impl PgnGame {
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v.as_str())
    }
}

/// Split a PGN text into games (a header after move text starts a new game).
pub fn split_games(text: &str) -> Vec<PgnGame> {
    let mut games = Vec::new();
    let mut cur = PgnGame::default();
    let mut movetext = String::new();
    let mut in_moves = false;
    let flush = |cur: &mut PgnGame, movetext: &mut String, games: &mut Vec<PgnGame>| {
        cur.tokens = movetext_tokens(movetext);
        if !cur.headers.is_empty() || !cur.tokens.is_empty() {
            games.push(std::mem::take(cur));
        }
        movetext.clear();
    };
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') && t.len() >= 2 {
            if in_moves {
                flush(&mut cur, &mut movetext, &mut games);
                in_moves = false;
            }
            let inner = &t[1..t.len() - 1];
            if let Some((k, v)) = inner.split_once(char::is_whitespace) {
                let v = v.trim();
                let v = v.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(v);
                cur.headers.push((k.to_string(), v.to_string()));
            }
        } else if !t.is_empty() {
            in_moves = true;
            movetext.push_str(t);
            movetext.push('\n');
        }
    }
    flush(&mut cur, &mut movetext, &mut games);
    games
}

/// The move tokens of a movetext: comments `{}` / `;`, variations `()`, move numbers,
/// NAGs and results removed.
pub fn movetext_tokens(text: &str) -> Vec<String> {
    let mut clean = String::with_capacity(text.len());
    let (mut brace, mut paren) = (0usize, 0usize);
    let mut line_comment = false;
    for ch in text.chars() {
        if line_comment {
            if ch == '\n' {
                line_comment = false;
                clean.push(' ');
            }
            continue;
        }
        match ch {
            '{' => brace += 1,
            '}' if brace > 0 => brace -= 1,
            '(' if brace == 0 => paren += 1,
            ')' if brace == 0 && paren > 0 => paren -= 1,
            ';' if brace == 0 && paren == 0 => line_comment = true,
            _ if brace == 0 && paren == 0 => clean.push(ch),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for raw in clean.split(|c: char| c.is_whitespace() || c == '\u{3000}') {
        if raw.is_empty() || raw.starts_with('$') || matches!(raw, "1-0" | "0-1" | "1/2-1/2" | "*" | "½-½") {
            continue;
        }
        if raw.chars().all(|c| c.is_ascii_digit() || c == '.') {
            continue;
        }
        // a move number glued to the move: "1.炮二平五", "12...h8e8"
        let tok = match raw.find('.') {
            Some(i) if i > 0 && raw[..i].bytes().all(|b| b.is_ascii_digit()) => raw[i..].trim_start_matches('.'),
            _ => raw,
        };
        if !tok.is_empty() {
            out.push(tok.to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// notation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Forward,
    Back,
    Side,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pos {
    Front,
    Rear,
    Middle,
    /// n-th from the front (1-based; stacked soldiers)
    Nth(usize),
}

fn numeral(c: char) -> Option<usize> {
    let n = match c {
        '一' | '１' | '1' => 1,
        '二' | '２' | '2' => 2,
        '三' | '３' | '3' => 3,
        '四' | '４' | '4' => 4,
        '五' | '５' | '5' => 5,
        '六' | '６' | '6' => 6,
        '七' | '７' | '7' => 7,
        '八' | '８' | '8' => 8,
        '九' | '９' | '9' => 9,
        _ => return None,
    };
    Some(n)
}

fn chinese_piece(c: char) -> Option<usize> {
    Some(match c {
        '帥' | '帅' | '將' | '将' => KING,
        '仕' | '士' => ADVISOR,
        '相' | '象' => ELEPHANT,
        '馬' | '马' | '傌' => HORSE,
        '車' | '车' | '俥' => ROOK,
        '炮' | '砲' | '包' => CANNON,
        '兵' | '卒' => SOLDIER,
        _ => return None,
    })
}

fn wxf_piece(c: char) -> Option<usize> {
    Some(match c.to_ascii_uppercase() {
        'K' | 'G' => KING,
        'A' => ADVISOR,
        'E' | 'B' => ELEPHANT,
        'H' | 'N' => HORSE,
        'R' => ROOK,
        'C' => CANNON,
        'P' => SOLDIER,
        _ => return None,
    })
}

fn chinese_act(c: char) -> Option<Act> {
    Some(match c {
        '進' | '进' => Act::Forward,
        '退' => Act::Back,
        '平' => Act::Side,
        _ => return None,
    })
}

fn chinese_pos(c: char) -> Option<Pos> {
    Some(match c {
        '前' => Pos::Front,
        '後' | '后' => Pos::Rear,
        '中' => Pos::Middle,
        _ => return None,
    })
}

/// A move in file-relative notation (Chinese / WXF), before resolution.
#[derive(Clone, Copy, Debug)]
struct Relative {
    piece: usize,
    /// origin file number (1-9, from the mover's right)
    file: Option<usize>,
    pos: Option<Pos>,
    act: Act,
    target: usize,
}

fn parse_chinese(tok: &str) -> Option<Relative> {
    let ch: Vec<char> = tok.trim_end_matches(['!', '?', '+', '#', '＋', '！', '？']).chars().collect();
    match ch.as_slice() {
        // 炮二平五
        &[p, f, a, t] if chinese_piece(p).is_some() && numeral(f).is_some() => Some(Relative {
            piece: chinese_piece(p)?,
            file: numeral(f),
            pos: None,
            act: chinese_act(a)?,
            target: numeral(t)?,
        }),
        // 前炮平四 / 中兵進一 / 二兵平五 (stacked soldiers counted from the front)
        &[q, p, a, t] if chinese_piece(p).is_some() => {
            let pos = chinese_pos(q).or_else(|| numeral(q).map(Pos::Nth))?;
            Some(Relative { piece: chinese_piece(p)?, file: None, pos: Some(pos), act: chinese_act(a)?, target: numeral(t)? })
        }
        // 前兵七平六 (stacked pieces on two files)
        &[q, p, f, a, t] if chinese_piece(p).is_some() => {
            let pos = chinese_pos(q).or_else(|| numeral(q).map(Pos::Nth))?;
            Some(Relative { piece: chinese_piece(p)?, file: numeral(f), pos: Some(pos), act: chinese_act(a)?, target: numeral(t)? })
        }
        _ => None,
    }
}

fn wxf_act(c: char) -> Option<Act> {
    Some(match c {
        '+' => Act::Forward,
        '-' => Act::Back,
        '.' | '=' => Act::Side,
        _ => return None,
    })
}

fn parse_wxf(tok: &str) -> Option<Relative> {
    let ch: Vec<char> = tok.trim_end_matches(['!', '?', '#']).chars().collect();
    match ch.as_slice() {
        // C2.5 / H8+7 / C+.5 (tandem: + front, - rear)
        &[p, f, a, t] if wxf_piece(p).is_some() => {
            let (file, pos) = match f {
                '+' => (None, Some(Pos::Front)),
                '-' => (None, Some(Pos::Rear)),
                _ => (Some(numeral(f)?), None),
            };
            Some(Relative { piece: wxf_piece(p)?, file, pos, act: wxf_act(a)?, target: numeral(t)? })
        }
        // +C.5 / -P+1
        &[q, p, a, t] if (q == '+' || q == '-') && wxf_piece(p).is_some() => Some(Relative {
            piece: wxf_piece(p)?,
            file: None,
            pos: Some(if q == '+' { Pos::Front } else { Pos::Rear }),
            act: wxf_act(a)?,
            target: numeral(t)?,
        }),
        _ => None,
    }
}

/// Board file index of file number `n` for `side` (numbered from the mover's right).
fn file_index(side: usize, n: usize) -> usize {
    if side == RED { FILES - n } else { n - 1 }
}

/// The legal move a relative (Chinese / WXF) move names.
fn resolve_relative(g: &Xiangqi, m: &Relative) -> Result<u32, String> {
    let b = g.board();
    let side = g.side_to_move();
    let mut cands: Vec<usize> = xq::Squares(b.pieces(side, m.piece)).collect();
    if let Some(n) = m.file {
        let f = file_index(side, n);
        cands.retain(|&s| s % FILES == f);
    }
    if let Some(pos) = m.pos {
        // group by file, order front → rear (front = further advanced for the mover)
        let mut picked = Vec::new();
        for f in 0..FILES {
            let mut col: Vec<usize> = cands.iter().copied().filter(|&s| s % FILES == f).collect();
            if col.len() < 2 {
                continue;
            }
            col.sort_by_key(|&s| s / FILES);
            if side == RED {
                col.reverse();
            }
            let k = match pos {
                Pos::Front => Some(0),
                Pos::Rear => Some(col.len() - 1),
                Pos::Middle => (col.len() == 3).then_some(1),
                Pos::Nth(n) => (n >= 1 && n <= col.len()).then(|| n - 1),
            };
            if let Some(k) = k {
                picked.push(col[k]);
            }
        }
        cands = picked;
    }
    let fwd: i64 = if side == RED { 1 } else { -1 };
    let legal = g.legal_moves();
    let mut found: Vec<u32> = Vec::new();
    for &o in &cands {
        let (f, r) = ((o % FILES) as i64, (o / FILES) as i64);
        let dir = match m.act {
            Act::Forward => fwd,
            Act::Back => -fwd,
            Act::Side => 0,
        };
        let dest = match (m.act, m.piece) {
            (Act::Side, _) => Some((file_index(side, m.target) as i64, r)),
            (_, KING | ROOK | CANNON | SOLDIER) => Some((f, r + dir * m.target as i64)),
            (_, pt) => {
                let tf = file_index(side, m.target) as i64;
                let df = (tf - f).abs();
                let dr = match (pt, df) {
                    (ADVISOR, 1) => Some(1),
                    (ELEPHANT, 2) => Some(2),
                    (HORSE, 1) => Some(2),
                    (HORSE, 2) => Some(1),
                    _ => None,
                };
                dr.map(|dr| (tf, r + dir * dr))
            }
        };
        if let Some((tf, tr)) = dest {
            if (0..FILES as i64).contains(&tf) && (0..10).contains(&tr) {
                let a = xq::action_of(o, (tr * FILES as i64 + tf) as usize);
                if legal.binary_search(&a).is_ok() && !found.contains(&a) {
                    found.push(a);
                }
            }
        }
    }
    match found.as_slice() {
        [a] => Ok(*a),
        [] => Err("no legal move matches".into()),
        _ => Err(format!("ambiguous ({} legal moves match)", found.len())),
    }
}

/// Coordinate conventions of ASCII square moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coords {
    /// ranks 1-10 (Fairy-Stockfish / UCI)
    Uci,
    /// ranks 0-9 (ICCS)
    Iccs,
}

fn parse_coords(tok: &str, coords: Coords) -> Option<u32> {
    let t = tok.trim_end_matches(['!', '?', '+', '#']).to_ascii_lowercase().replace('-', "");
    match coords {
        Coords::Uci => parse_uci(&t),
        Coords::Iccs => {
            let c: Vec<char> = t.chars().collect();
            if c.len() != 4 {
                return None;
            }
            let sq = |f: char, r: char| -> Option<usize> {
                let (f, r) = ((f as i64) - ('a' as i64), r.to_digit(10)? as i64);
                ((0..9).contains(&f) && (0..10).contains(&r)).then_some((r * 9 + f) as usize)
            };
            Some(xq::action_of(sq(c[0], c[1])?, sq(c[2], c[3])?))
        }
    }
}

/// The ASCII coordinate convention of a game: ICCS when `[Format "ICCS"]` or when a
/// coordinate token has a dash or a rank 0, else Fairy-Stockfish / UCI.
fn coords_of(game: &PgnGame) -> Coords {
    if game.header("Format").is_some_and(|f| f.eq_ignore_ascii_case("ICCS")) {
        return Coords::Iccs;
    }
    let iccs = game.tokens.iter().any(|t| {
        let b = t.as_bytes();
        t.is_ascii()
            && b.len() >= 4
            && b[0].is_ascii_alphabetic()
            && (t.contains('-') && b.len() == 5 || (b[1] == b'0') || (b.len() == 4 && b[3] == b'0' && b[2].is_ascii_alphabetic()))
    });
    if iccs { Coords::Iccs } else { Coords::Uci }
}

/// The legal move of `g` a notation token names (Chinese, WXF, ICCS or UCI).
pub fn resolve_token(g: &Xiangqi, tok: &str, coords: Coords) -> Result<u32, String> {
    let rel = if tok.is_ascii() { parse_wxf(tok) } else { parse_chinese(tok) };
    if let Some(m) = rel {
        return resolve_relative(g, &m).map_err(|e| format!("{tok:?}: {e}"));
    }
    if tok.is_ascii() {
        if let Some(a) = parse_coords(tok, coords) {
            return if g.legal_moves().binary_search(&a).is_ok() {
                Ok(a)
            } else {
                Err(format!("{tok:?}: {} is not legal", xq::uci(a)))
            };
        }
    }
    Err(format!("{tok:?}: unknown notation"))
}

// ---------------------------------------------------------------------------
// records
// ---------------------------------------------------------------------------

/// What to do with a record whose moves go on after the rules ended the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PastEnd {
    /// keep the moves up to the rules' end, labelled by the rules' result
    Cut,
    /// drop the record
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Split {
    All,
    Train,
    Val,
}

/// Parser-side counters (merged into the conversion stats; the result / end counters
/// count the records handed to the converter, i.e. after the filters and duplicates).
#[derive(Clone, Debug, Default, Serialize)]
pub struct ParseStats {
    pub files: usize,
    pub games: usize,
    /// not from the initial position
    pub other_start: usize,
    pub duplicates: usize,
    /// duplicates whose result differs from the kept copy's (inconsistent headers)
    pub duplicate_result_conflicts: usize,
    /// filtered out: `--hq` category / result, `--min-plies`, `--split`
    pub filtered: usize,
    /// `[Result]` unknown (`*`, missing): the result the `[Event]` text states was used
    pub result_from_event: usize,
    /// `[Result]` and the `[Event]` text state different results: the result is
    /// unknown (the converter keeps the record only when the rules decide the game)
    pub result_conflict: usize,
    /// the rules ended the game at the record's last move by an adjudication rule
    pub rules_end: usize,
    /// the record went on after the rules' end: cut there
    pub cut: usize,
    /// plies of the cut records after the cut (not converted)
    pub cut_plies: usize,
    /// rules_end / cut records whose recorded result differs from the rules' result
    pub rules_disagree: usize,
    /// end reasons of the records the rules ended
    pub reasons: std::collections::BTreeMap<String, usize>,
}

/// Record filters and the parser's state (duplicate keys, counters).
#[derive(Default)]
pub struct Filter {
    /// master games only (CCPD 大師對局), >= 20 plies, known result
    pub hq: bool,
    pub min_plies: usize,
    pub past_end: PastEnd,
    pub split: Split,
    /// a game is in the validation split when its key ≡ 0 mod `val_mod` (0 = 50)
    pub val_mod: u64,
    /// unfinished records are converted too (`--allow-truncated`): they claim their
    /// duplicate key
    pub allow_truncated: bool,
    /// moves key → the result of the kept copy
    pub seen: RefCell<HashMap<u64, Option<f32>>>,
    pub stats: RefCell<ParseStats>,
}

impl Default for PastEnd {
    fn default() -> Self {
        PastEnd::Cut
    }
}

impl Default for Split {
    fn default() -> Self {
        Split::All
    }
}

/// The recorded result for Red (None = unknown / unfinished).
pub fn result_red(r: &str) -> Option<f32> {
    match r.trim() {
        "1-0" => Some(1.0),
        "0-1" => Some(-1.0),
        "1/2-1/2" | "½-½" => Some(0.0),
        _ => None,
    }
}

/// The result for Red that a CCPD `[Event]` text states in words anchored to the
/// colours: 先勝 / 先負 / 先和 ("the first mover — Red — wins / loses / draws"; also
/// 紅先勝, 先 勝, simplified 先胜 / 先负), (紅勝) / (黑勝) in brackets, or a draw — 和 as a
/// word of its own (a draw needs no colours). None when the text states none, more
/// than one, or Black moves first (黑先). Unanchored words are ignored: in "A 勝 B" A
/// need not be Red, and 紅勝 / 紅負 are mostly the name 徐天紅 followed by 勝 / 負.
pub fn event_result_red(event: &str) -> Option<f32> {
    let e: String = event.split(|c: char| c.is_whitespace() || c == '\u{3000}').filter(|w| !w.is_empty())
        .collect::<Vec<_>>().join(" ");
    if e.contains("黑先") {
        return None;
    }
    let mut found: Vec<f32> = Vec::new();
    let mut note = |v: f32| {
        if !found.contains(&v) {
            found.push(v)
        }
    };
    let any = |pats: &[&str]| pats.iter().any(|p| e.contains(p));
    if any(&["先勝", "先 勝", "先胜", "先 胜", "(紅勝)", "（紅勝）", "(红胜)", "（红胜）"]) {
        note(1.0);
    }
    if any(&["先負", "先 負", "先负", "先 负", "(黑勝)", "（黑勝）", "(黑胜)", "（黑胜）"]) {
        note(-1.0);
    }
    let words: Vec<&str> = e.split(' ').collect();
    if any(&["先和", "先 和", "(和)", "（和）"]) || words.contains(&"和") {
        note(0.0);
    }
    match found.as_slice() {
        [v] => Some(*v),
        _ => None,
    }
}

/// The result of a record for Red and where it comes from: the `[Result]` header;
/// when that is unknown, the `[Event]` text ([`event_result_red`]); when the two
/// disagree, unknown.
fn declared_result(game: &PgnGame) -> (Option<f32>, &'static str) {
    let header = game.header("Result").and_then(result_red);
    let event = game.header("Event").and_then(event_result_red);
    match (header, event) {
        (Some(h), Some(e)) if h != e => (None, "conflict"),
        (Some(h), _) => (Some(h), "header"),
        (None, Some(e)) => (Some(e), "event"),
        (None, None) => (None, "unknown"),
    }
}

fn is_start(fen: Option<&str>) -> bool {
    match fen {
        None => true,
        Some(f) => {
            let f = f.trim();
            f.is_empty() || f.split_whitespace().take(2).eq(START_FEN.split_whitespace().take(2))
        }
    }
}

/// The identity of a game: FNV-1a 64 of its moves (stable across platforms and Rust
/// versions — the train / val split depends on it).
pub fn moves_key(events: &[Event]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for e in events {
        let (_, id) = e.to_pair();
        for b in (id as u32).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// One PGN game → a record (None = filtered out or a duplicate; Err = unparseable).
///
/// Duplicates: the same moves (the result is not part of the key — the three CCPD
/// views file one game up to three times, not always with the same headers); the
/// first copy in path order that the converter will keep claims the key, so an
/// unfinished copy (unknown result) never hides a later copy with a result.
pub fn game_record(game: &PgnGame, id: &str, category: &str, filter: &Filter) -> Result<Option<Record>> {
    let mut st = filter.stats.borrow_mut();
    st.games += 1;
    if !is_start(game.header("FEN")) {
        st.other_start += 1;
        return Ok(None);
    }
    let (declared, source) = declared_result(game);
    let master = category.contains("大師對局") || category.contains("大师对局");
    if filter.hq && (!master || declared.is_none()) {
        st.filtered += 1;
        return Ok(None);
    }
    let coords = coords_of(game);
    let mut g = Xiangqi::new_game(&XiangqiConfig::default());
    let mut events = Vec::with_capacity(game.tokens.len());
    let mut cut = false;
    for (i, tok) in game.tokens.iter().enumerate() {
        if g.end().is_some() {
            if filter.past_end == PastEnd::Reject {
                bail!("the rules end the game after {i} plies, the record goes on");
            }
            cut = true;
            break;
        }
        let a = resolve_token(&g, tok, coords).map_err(|e| anyhow!("ply {}: {e}", i + 1))?;
        events.push(Event::Act(g.side_to_move(), a));
        g.play(a).map_err(|e| anyhow!("ply {}: {e}", i + 1))?;
    }
    let plies = events.len();
    if plies < filter.min_plies.max(if filter.hq { 20 } else { 0 }) {
        st.filtered += 1;
        return Ok(None);
    }
    // the value target: the recorded result, unless the rules end the game — at a
    // checkmate / stalemate the recorded result must agree (traj::convert checks it),
    // at an adjudication (repetition, move limit, material) or a cut the rules decide
    let end = g.end();
    let adjudicated_by_rules = end.is_some_and(|(_, why)| cut || !matches!(why, EndReason::Checkmate | EndReason::Stalemate));
    let final_outcome = if adjudicated_by_rules { None } else { declared.map(|d| vec![d, -d]) };
    let converts = end.is_some() || final_outcome.is_some() || filter.allow_truncated;
    let key = moves_key(&events);
    let first = filter.seen.borrow().get(&key).copied();
    if let Some(kept) = first {
        st.duplicates += 1;
        if declared.is_some() && kept.is_some() && declared != kept {
            st.duplicate_result_conflicts += 1;
        }
        return Ok(None);
    }
    if converts {
        filter.seen.borrow_mut().insert(key, declared);
    }
    let val = key % (if filter.val_mod == 0 { 50 } else { filter.val_mod }) == 0;
    if (filter.split == Split::Train && val) || (filter.split == Split::Val && !val) {
        st.filtered += 1;
        return Ok(None);
    }
    let mut meta = json!({"plies": plies, "category": category, "result_source": source});
    for (k, name) in [("Event", "event"), ("Date", "date"), ("Red", "red"), ("Black", "black"), ("Result", "result"),
                      ("ECCO", "ecco")] {
        if let Some(v) = game.header(k).filter(|v| !v.is_empty()) {
            meta[name] = json!(v);
        }
    }
    match source {
        "event" => st.result_from_event += 1,
        "conflict" => st.result_conflict += 1,
        _ => {}
    }
    if let Some((red, why)) = end {
        *st.reasons.entry(why.name().to_string()).or_default() += 1;
        if adjudicated_by_rules {
            if cut {
                st.cut += 1;
                st.cut_plies += game.tokens.len() - plies;
                meta["cut_at"] = json!(plies);
                meta["record_plies"] = json!(game.tokens.len());
            } else {
                st.rules_end += 1;
            }
            if declared.is_some_and(|d| d != red as f32) {
                st.rules_disagree += 1;
            }
            meta["rules_end"] = json!(why.name());
        }
    }
    Ok(Some(Record { id: id.to_string(), events, meta, final_outcome, ..Default::default() }))
}

/// The record files of an input: `(id prefix, category, bytes)`, sorted by path.
enum Source {
    Zip { archive: zip::ZipArchive<std::fs::File>, names: Vec<String>, next: usize },
    Files { root: PathBuf, files: Vec<PathBuf>, next: usize },
}

impl Iterator for Source {
    type Item = Result<(String, String, Vec<u8>)>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Source::Zip { archive, names, next } => {
                let name = names.get(*next)?.clone();
                *next += 1;
                let mut buf = Vec::new();
                let res = archive.by_name(&name).map_err(anyhow::Error::from).and_then(|mut f| {
                    f.read_to_end(&mut buf)?;
                    Ok(())
                });
                // the path below the archive's top directory: "Dataset/對局/..."
                let rel = name.split_once('/').map_or(name.as_str(), |(_, r)| r).to_string();
                let category = Path::new(&rel).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                Some(res.map(|_| (rel, category, buf)))
            }
            Source::Files { root, files, next } => {
                let path = files.get(*next)?.clone();
                *next += 1;
                let rel = path.strip_prefix(&*root).unwrap_or(&path).to_string_lossy().to_string();
                let category = Path::new(&rel).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                Some(std::fs::read(&path).with_context(|| format!("read {}", path.display())).map(|b| (rel, category, b)))
            }
        }
    }
}

fn open_source(input: &Path) -> Result<Source> {
    if input.is_file() && input.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        let archive = zip::ZipArchive::new(std::fs::File::open(input).with_context(|| format!("open {}", input.display()))?)?;
        let mut names: Vec<String> =
            archive.file_names().filter(|n| n.to_lowercase().ends_with(".pgn")).map(String::from).collect();
        names.sort();
        anyhow::ensure!(!names.is_empty(), "no .pgn files in {}", input.display());
        return Ok(Source::Zip { archive, names, next: 0 });
    }
    let files = common::find_files(input, &[".pgn"])?;
    anyhow::ensure!(!files.is_empty(), "no .pgn files under {}", input.display());
    let root = if input.is_file() { input.parent().unwrap_or(Path::new("")).to_path_buf() } else { input.to_path_buf() };
    Ok(Source::Files { root, files, next: 0 })
}

/// Every record of `input` (streamed file by file; one Err per unreadable file or
/// unparseable game).
pub fn records<'a>(input: &Path, filter: &'a Filter) -> Result<impl Iterator<Item = Result<Record>> + 'a> {
    let src = open_source(input)?;
    Ok(src.flat_map(move |item| -> Box<dyn Iterator<Item = Result<Record>> + 'a> {
        match item {
            Err(e) => Box::new(std::iter::once(Err(e))),
            Ok((rel, category, bytes)) => {
                filter.stats.borrow_mut().files += 1;
                let text = decode_text(&common::read_maybe_gz(bytes).unwrap_or_default());
                let games = split_games(&text);
                let many = games.len() > 1;
                Box::new(games.into_iter().enumerate().filter_map(move |(k, game)| {
                    let id = if many { format!("{rel}#{k}") } else { rel.clone() };
                    game_record(&game, &id, &category, filter).with_context(|| id.clone()).transpose()
                }))
            }
        }
    }))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
pub enum Cmd {
    /// Download the CCPD archive (GitHub, pinned commit) into DIR
    Fetch {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = CCPD_COMMIT)]
        commit: String,
    },
    /// Convert PGN records (.zip / directory / .pgn) to from-to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// high-quality preset: master games (CCPD 大師對局), >= 20 plies, known result
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 0)]
        min_plies: usize,
        /// records that go on after the rules ended the game (an unclaimed repetition)
        #[arg(long, value_enum, default_value_t = PastEnd::Cut)]
        past_end: PastEnd,
        /// train / val: the games whose hash is (not) = 0 mod --val-mod
        #[arg(long, value_enum, default_value_t = Split::All)]
        split: Split,
        #[arg(long, default_value_t = 50)]
        val_mod: u64,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        #[arg(long)]
        max_records: Option<usize>,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep unfinished records (unknown result: tgt.value_mask false; the causality
        /// invariance test converts truncated records)
        #[arg(long)]
        allow_truncated: bool,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, commit } => {
            std::fs::create_dir_all(&out)?;
            let short: String = commit.chars().take(8).collect();
            let dest = out.join(format!("ccpd-{short}.zip"));
            let url = format!("https://codeload.github.com/Yvonne761/Chinese-Chess-Practical-Dataset/zip/{commit}");
            if !common::download(&common::http()?, &url, &dest)? {
                bail!("{url}: not found");
            }
            std::fs::write(out.join("ATTRIBUTION.txt"), format!(
                "{CCPD_ATTRIBUTION}\nRetrieved from {CCPD_REPO} (commit {commit}).\nLicence: {CCPD_LICENCE}.\n\
                 Converted to gm-shards by GamingMaster `gmdata xiangqi convert` (moves replayed and re-encoded; \
                 duplicates and non-standard starts dropped).\n"))?;
            let bytes = std::fs::metadata(&dest)?.len();
            println!("{}", json!({"zip": dest, "bytes": bytes, "commit": commit, "licence": CCPD_LICENCE}));
            Ok(())
        }
        Cmd::Convert { input, out, hq, min_plies, past_end, split, val_mod, shard_size, max_records, store_events,
                       allow_truncated } => {
            let filter = Filter { hq, min_plies, past_end, split, val_mod, allow_truncated, ..Default::default() };
            let recs = records(&input, &filter)?.take(max_records.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, ..Default::default() };
            let st = traj::convert::<Xiangqi>(&XiangqiConfig::default(), recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            if let Value::Object(p) = serde_json::to_value(&*filter.stats.borrow())? {
                for (k, x) in p {
                    v[k] = x;
                }
            }
            println!("{v}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(tokens: &[&str]) -> Result<Xiangqi, String> {
        let mut g = Xiangqi::new_game(&XiangqiConfig::default());
        for t in tokens {
            let a = resolve_token(&g, t, Coords::Uci)?;
            g.play(a)?;
        }
        Ok(g)
    }

    #[test]
    fn chinese_notation_resolves_like_coordinates() {
        // 1. 炮二平五 炮８平５ 2. 馬二進三 馬８進７ 3. 車一平二 卒７進１ (traditional, full-width digits)
        let g = play(&["炮二平五", "炮８平５", "馬二進三", "馬８進７", "車一平二", "卒７進１"]).unwrap();
        let c = play(&["h3e3", "h8e8", "h1g3", "h10g8", "i1h1", "g7g6"]).unwrap();
        assert_eq!(g.fen(), c.fen());
        // simplified characters, WXF and ICCS name the same moves
        let s = play(&["炮二平五", "炮8平5", "马二进三", "马8进7", "车一平二", "卒7进1"]).unwrap();
        assert_eq!(s.fen(), c.fen());
        let w = play(&["C2.5", "C8.5", "H2+3", "H8+7", "R1.2", "P7+1"]).unwrap();
        assert_eq!(w.fen(), c.fen());
        let mut i = Xiangqi::new_game(&XiangqiConfig::default());
        for t in ["H2-E2", "H7-E7", "H0-G2", "H9-G7", "I0-H0", "G6-G5"] {
            let a = resolve_token(&i, t, Coords::Iccs).unwrap();
            i.play(a).unwrap();
        }
        assert_eq!(i.fen(), c.fen());
        // front / rear: after both red cannons stand on the e-file
        let g = play(&["炮二平五", "馬８進７", "炮八平五"]);
        assert!(g.is_err(), "炮八平五 would put a second cannon on file 5 — legal: {g:?}");
        let g = play(&["炮二平五", "馬８進７", "炮八進四", "車９平８", "炮八平五", "卒３進１"]).unwrap();
        let front = resolve_token(&g, "前炮進一", Coords::Uci).unwrap();
        let rear = resolve_token(&g, "後炮退一", Coords::Uci).unwrap();
        assert_eq!(xq::uci(front), "e7e8");
        assert_eq!(xq::uci(rear), "e3e2");
        // the file alone is ambiguous here, unless only one of the two can make the move
        assert_eq!(resolve_token(&g, "炮五進一", Coords::Uci).map(xq::uci), Ok("e7e8".to_string()));
        assert!(resolve_token(&g, "炮五平四", Coords::Uci).unwrap_err().contains("ambiguous"));
        assert!(resolve_token(&g, "炮九平五", Coords::Uci).is_err());
        assert!(resolve_token(&g, "炮二平五x", Coords::Uci).unwrap_err().contains("unknown notation"));
    }

    #[test]
    fn pgn_splitting_and_tokens() {
        let text = "[Game \"Chinese Chess\"]\n[Result \"1-0\"]\n\n1. 炮二平五 炮８平５ {comment 炮}\n2. 馬二進三 (2. 車一平二) 馬８進７\n1-0\n\
                    [Result \"*\"]\n1.h3e3 h8e8 $1 2. h1g3\n";
        let games = split_games(text);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].tokens, vec!["炮二平五", "炮８平５", "馬二進三", "馬８進７"]);
        assert_eq!(games[0].header("result"), Some("1-0"));
        assert_eq!(games[1].tokens, vec!["h3e3", "h8e8", "h1g3"]);
        assert_eq!(result_red("1/2-1/2"), Some(0.0));
        assert!(is_start(None) && is_start(Some(START_FEN)) && !is_start(Some("4k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1")));
        // Big5 and GBK bytes of "炮二平五" decode to the notation
        let (big5, _, _) = encoding_rs::BIG5.encode("[Result \"1-0\"]\n1. 炮二平五 馬８進７\n");
        assert!(decode_text(&big5).contains("炮二平五"));
        let (gbk, _, _) = encoding_rs::GBK.encode("[Result \"1-0\"]\n1. 炮二平五 马8进7\n");
        assert!(decode_text(&gbk).contains("马8进7"));
    }

    fn game(result: &str, moves: &str) -> PgnGame {
        split_games(&format!("[Result \"{result}\"]\n{moves}\n")).remove(0)
    }

    #[test]
    fn records_end_by_result_rules_and_cut() {
        let f = Filter::default();
        // resignation: not terminal, the recorded result is the outcome
        let r = game_record(&game("0-1", "1. 炮二平五 炮８平５"), "a", "x", &f).unwrap().unwrap();
        assert_eq!(r.final_outcome, Some(vec![-1.0, 1.0]));
        assert_eq!(r.events.len(), 2);
        assert_eq!(r.meta["result_source"], "header");
        // unknown result: no outcome (the converter drops it as truncated)
        let r = game_record(&game("*", "1. 炮二平五 炮８平５ 2. 馬二進三"), "b", "x", &f).unwrap().unwrap();
        assert_eq!(r.final_outcome, None);
        // a duplicate (the same moves, whatever the headers say) is dropped
        assert!(game_record(&game("1-0", "1. 炮二平五 炮８平５"), "c", "x", &f).unwrap().is_none());
        // an unclaimed repetition: the record is cut at the third occurrence
        let rep = "1. 馬二進三 馬８進７ 2. 馬三退二 馬７退８ 3. 馬二進三 馬８進７ 4. 馬三退二 馬７退８ 5. 馬二進三 馬８進７";
        let r = game_record(&game("1-0", rep), "d", "x", &f).unwrap().unwrap();
        assert_eq!(r.events.len(), 8);
        assert_eq!(r.final_outcome, None);
        assert_eq!(r.meta["rules_end"], "repetition");
        assert_eq!((r.meta["cut_at"].as_u64(), r.meta["record_plies"].as_u64()), (Some(8), Some(10)));
        let st = f.stats.borrow().clone();
        assert_eq!((st.games, st.duplicates, st.cut, st.cut_plies, st.rules_disagree), (4, 1, 1, 2, 1));
        let strict = Filter { past_end: PastEnd::Reject, ..Default::default() };
        assert!(game_record(&game("1-0", rep), "d", "x", &strict).is_err());
        // --hq: master games with a result and >= 20 plies only
        let hq = Filter { hq: true, ..Default::default() };
        assert!(game_record(&game("1-0", "1. 炮二平五 炮８平５"), "e", "Dataset/對局/大師對局/x", &hq).unwrap().is_none());
        // a non-standard start is skipped
        let other = split_games("[FEN \"4k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1\"]\n[Result \"1-0\"]\n1. 帥四進一\n").remove(0);
        assert!(game_record(&other, "f", "x", &f).unwrap().is_none());
    }

    #[test]
    fn duplicates_and_splits_follow_the_moves() {
        // an unfinished copy does not claim the game: a later copy with a result is kept,
        // and a copy after that is a duplicate
        let f = Filter::default();
        let moves = "1. 炮二平五 馬８進７ 2. 馬二進三";
        assert!(game_record(&game("*", moves), "a", "x", &f).unwrap().unwrap().final_outcome.is_none());
        assert_eq!(game_record(&game("1-0", moves), "b", "x", &f).unwrap().unwrap().final_outcome, Some(vec![1.0, -1.0]));
        assert!(game_record(&game("*", moves), "c", "x", &f).unwrap().is_none());
        assert!(game_record(&game("0-1", moves), "d", "x", &f).unwrap().is_none());
        assert_eq!((f.stats.borrow().duplicates, f.stats.borrow().duplicate_result_conflicts), (2, 1));
        // with --allow-truncated the first copy claims it
        let t = Filter { allow_truncated: true, ..Default::default() };
        assert!(game_record(&game("*", moves), "a", "x", &t).unwrap().is_some());
        assert!(game_record(&game("1-0", moves), "b", "x", &t).unwrap().is_none());
        // train / val: disjoint, covering, and every copy of a game on the same side
        let reds = ["h3e3", "b3e3", "h1g3", "b1c3", "c4c5", "g4g5", "c1e3", "g1e3", "i1i2", "a1a2"];
        let blacks = ["h8e8", "b8e8", "h10g8", "b10c8", "c7c6", "g7g6"];
        let (mut tr, mut va) = (0, 0);
        for (k, r) in reds.iter().enumerate() {
            for b in blacks {
                let m = format!("{r} {b}");
                let train = Filter { split: Split::Train, val_mod: 4, ..Default::default() };
                let val = Filter { split: Split::Val, val_mod: 4, ..Default::default() };
                let a = game_record(&game("1-0", &m), "g", "x", &train).unwrap().is_some();
                let v = game_record(&game("1-0", &m), "g", "x", &val).unwrap().is_some();
                assert!(a != v, "{m}");
                // the same game with another result header: the same split
                let a2 = game_record(&game(["0-1", "*"][k % 2], &m), "h", "x",
                                     &Filter { split: Split::Train, val_mod: 4, allow_truncated: true, ..Default::default() })
                    .unwrap().is_some();
                assert_eq!(a, a2, "{m}");
                tr += a as usize;
                va += v as usize;
            }
        }
        assert!(tr > 20 && va > 5, "{tr} / {va}");
        // the key is a fixed function of the moves (stable across builds)
        assert_eq!(moves_key(&[]), 0xcbf2_9ce4_8422_2325);
        let g = play(&["h3e3"]).unwrap();
        assert_eq!(g.plies(), 1);
        assert_eq!(moves_key(&[Event::Act(0, xq::parse_uci("h3e3").unwrap())]),
                   moves_key(&[Event::Act(1, xq::parse_uci("h3e3").unwrap())]));
    }

    #[test]
    fn results_from_the_event_text() {
        for (event, want) in [
            ("1976年中國象棋團體賽預賽 河北 李來群 先勝 湖北 張宏群", Some(1.0)),
            ("2003年全國象棋團體賽 哈爾濱 張婉瑩 先負 農協 李淑英", Some(-1.0)),
            ("大連益春堂金松紅先負廣東東莞日報許銀川（二臺）", Some(-1.0)),
            ("杭州宋城 張正源 (紅先和) 通訊 朱祖勤", Some(0.0)),
            ("象甲聯賽 廣東李鴻嘉 和 浙江陳寒峰", Some(0.0)),
            ("武漢 李義庭 先 負  上海 楊官璘", Some(-1.0)),
            ("Chimo vs BrainChess (紅勝)", Some(1.0)),
            ("象棋经略先胜谢谢", Some(1.0)),
            // unanchored or ambiguous: not used
            ("2009年全香港公開中國象棋個人賽 香港 鄭子強 勝 香港 鄭彥隆", None),
            ("19891023徐天紅勝劉殿中", None),
            ("19880519徐天紅和胡榮華", None),
            ("廣東 楊官璘  先和 湖北 李義庭 廣東楊官璘 先勝 上海 胡榮華", None),
            ("06 紅兩頭蛇黑先高右橫車和左車巡河", None),
            ("", None),
        ] {
            assert_eq!(event_result_red(event), want, "{event}");
        }
        let f = Filter::default();
        let with = |result: &str, event: &str, moves: &str| {
            split_games(&format!("[Event \"{event}\"]\n[Result \"{result}\"]\n{moves}\n")).remove(0)
        };
        // `*` + a result in the event: used
        let r = game_record(&with("*", "甲 先負 乙", "1. 炮二平五 炮８平５"), "a", "x", &f).unwrap().unwrap();
        assert_eq!((r.final_outcome, &r.meta["result_source"]), (Some(vec![-1.0, 1.0]), &json!("event")));
        // header and event disagree: unknown (dropped as truncated unless the rules decide)
        let r = game_record(&with("1-0", "甲 先負 乙", "1. 炮二平五 馬８進７"), "b", "x", &f).unwrap().unwrap();
        assert_eq!((r.final_outcome, &r.meta["result_source"]), (None, &json!("conflict")));
        let hq = Filter { hq: true, ..Default::default() };
        assert!(game_record(&with("1-0", "甲 先負 乙", "1. 炮二平五 馬８進７"), "c", "Dataset/對局/大師對局/y", &hq)
            .unwrap().is_none());
        let st = f.stats.borrow().clone();
        assert_eq!((st.result_from_event, st.result_conflict), (1, 1));
    }
}
