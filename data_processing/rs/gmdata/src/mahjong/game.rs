//! One game's decision points (`walk_game`) and its per-seat frame streams (`convert_game`).

use super::rules::*;
use super::state::*;
use gmshard::Sample;
use serde_json::{json, Value};

const PLACEMENT: [f32; 4] = [1.0, (1.0f64 / 3.0) as f32, (-1.0f64 / 3.0) as f32, -1.0];

pub(super) fn ty(ev: &Value) -> &str {
    ev.get("type").and_then(|v| v.as_str()).unwrap_or("")
}
fn actor(ev: &Value) -> Option<i64> {
    ev.get("actor").and_then(|v| v.as_i64())
}
fn target(ev: &Value) -> Option<i64> {
    ev.get("target").and_then(|v| v.as_i64())
}

/// A decision of one seat: the state it decides in, what prompted it, what it may do, and
/// what the log says it did.
pub(super) struct Decision<'a> {
    pub st: &'a Kyoku,
    pub seat: usize,
    pub ctx: &'a Ctx,
    pub legal: &'a Legal,
    /// the recorded response; None when the log does not show it (an own turn followed
    /// by an unsupported event)
    pub label: Option<(i64, i64)>,
    /// imitation-loss weight: 0 when the recorded pass does not show the seat's choice
    /// (a call it could have made was pre-empted by a higher-priority call)
    pub weight: f32,
    pub focus: Option<usize>,
    pub own: bool,
    pub focus_actor: Option<usize>,
    /// index of the prompting event: the tsumo, chi / pon, dahai or kakan (keys the
    /// differential test's comparison)
    #[cfg_attr(not(test), allow(dead_code))]
    pub event: usize,
}

/// The seat's own-turn response to the tsumo at `i` (dora events in between skipped).
fn own_turn_label(evs: &[Value], i: usize, a: usize) -> R<Option<(i64, i64)>> {
    let mut j = i + 1;
    while j < evs.len() && ty(&evs[j]) == "dora" {
        j += 1;
    }
    let Some(nx) = evs.get(j) else { return Ok(None) };
    let is_me = actor(nx) == Some(a as i64);
    Ok(match ty(nx) {
        "dahai" if is_me => Some((DAHAI, tile_kind(s(nx, "pai")?)? as i64)),
        "reach" if is_me => match evs.get(j + 1) {
            Some(d) if ty(d) == "dahai" && actor(d) == Some(a as i64) => Some((RIICHI, tile_kind(s(d, "pai")?)? as i64)),
            _ => None,
        },
        "ankan" if is_me => Some((KAN, tile_kind(Kyoku::consumed(nx)?.first().ok_or(Skip)?)? as i64)),
        "kakan" if is_me => Some((KAN, tile_kind(s(nx, "pai")?)? as i64)),
        "hora" if is_me && target(nx) == Some(a as i64) => Some((HORA, 0)),
        "ryukyoku" => Some((RYUKYOKU, 0)),
        _ => None,
    })
}

/// What the table did with the discard (or kakan) at `i` by `d`.
struct Response {
    /// seats that won on it (a double ron lists two)
    ronners: Vec<usize>,
    /// the chi / pon / daiminkan made of it: (seat, action type, tile argument)
    call: Option<(usize, i64, i64)>,
    /// the kyoku ended right there with a draw
    draw: bool,
}

fn response(evs: &[Value], i: usize, d: usize) -> R<Response> {
    let mut j = i + 1;
    while j < evs.len() && matches!(ty(&evs[j]), "reach_accepted" | "dora") {
        j += 1;
    }
    let mut ronners = Vec::new();
    while j < evs.len() && ty(&evs[j]) == "hora" && target(&evs[j]) == Some(d as i64) && actor(&evs[j]) != Some(d as i64) {
        ronners.push(seat(&evs[j], "actor")?);
        j += 1;
    }
    let (mut call, mut draw) = (None, false);
    if ronners.is_empty() {
        if let Some(nx) = evs.get(j) {
            match ty(nx) {
                t @ ("chi" | "pon" | "daiminkan") if target(nx) == Some(d as i64) => {
                    let k = tile_kind(s(nx, "pai")?)?;
                    let v = match t {
                        "chi" => {
                            let mut base = k;
                            for p in Kyoku::consumed(nx)? {
                                base = base.min(tile_kind(&p)?);
                            }
                            (CHI, base as i64)
                        }
                        "pon" => (PON, k as i64),
                        _ => (KAN, k as i64),
                    };
                    call = Some((seat(nx, "actor")?, v.0, v.1));
                }
                "ryukyoku" => draw = true,
                _ => {}
            }
        }
    }
    Ok(Response { ronners, call, draw })
}

/// Label and loss weight of seat `s`'s claim decision on a discard (or a kakan), from the
/// table's `resp`.  A recorded pass is the seat's choice unless another seat's call
/// pre-empted an option it had: a ron pre-empts chi / pon / kan (a second ron is not
/// pre-empted: Tenhou allows the double ron), a pon / daiminkan pre-empts chi.  Such passes
/// get weight 0, and so do all claims followed by an `unexplained_draw` (the triple ron).
fn claim_label(s: usize, legal: &Legal, resp: &Response, unexplained_draw: bool) -> ((i64, i64), f32) {
    if resp.ronners.contains(&s) {
        return ((HORA, 0), 1.0);
    }
    if let Some((c, t, tile)) = resp.call {
        if c == s {
            return ((t, tile), 1.0);
        }
    }
    let calls = legal.types[CHI as usize] || legal.types[PON as usize] || legal.types[KAN as usize];
    let pre_empted = if unexplained_draw {
        true
    } else if !resp.ronners.is_empty() {
        calls
    } else if let Some((_, t, _)) = resp.call {
        (t == PON || t == KAN) && legal.types[CHI as usize]
    } else {
        false
    };
    ((PASS, 0), if pre_empted { 0.0 } else { 1.0 })
}

/// Visit every decision point of the game, in log order: each own draw (Ctx::Draw),
/// each chi / pon (the forced discard, Ctx::AfterCall), and for every discard / kakan the
/// three other seats (Ctx::Claim, whether or not they have anything but pass).  Returns
/// (kyoku count, final scores).
pub(super) fn walk_game(evs: &[Value], visit: &mut dyn FnMut(&Decision)) -> R<(usize, [i64; 4])> {
    let mut st: Option<Kyoku> = None;
    let mut n_kyoku = 0;
    let mut consumed = vec![false; evs.len()];
    for (i, ev) in evs.iter().enumerate() {
        if consumed[i] {
            continue;
        }
        let t = ty(ev);
        if t == "start_kyoku" {
            st = Some(Kyoku::new(ev)?);
            n_kyoku += 1;
            continue;
        }
        let Some(k) = st.as_mut() else { continue };
        match t {
            "tsumo" => {
                // the frame is the state right after the draw: an open kan's dora the log
                // writes between the rinshan draw and the drawer's discard is revealed with
                // that discard, after this decision
                let a = seat(ev, "actor")?;
                let drawn = k.draw(a, s(ev, "pai")?)?;
                let label = own_turn_label(evs, i, a)?;
                let ctx = Ctx::Draw { drawn };
                let legal = k.legal(a, &ctx);
                visit(&Decision { st: k, seat: a, ctx: &ctx, legal: &legal, label, weight: 1.0, focus: Some(drawn), own: true,
                                  focus_actor: None, event: i });
            }
            "dahai" => {
                let a = seat(ev, "actor")?;
                let tsumogiri = ev.get("tsumogiri").and_then(|v| v.as_bool()).unwrap_or(false);
                let kk = k.discard(a, s(ev, "pai")?, tsumogiri)?;
                // Tenhou flips an open kan's new dora with this discard: one log spelling
                // writes it just before the dahai (already applied), the other just after it
                let mut j = i + 1;
                while j < evs.len() && matches!(ty(&evs[j]), "dora" | "reach_accepted") {
                    if ty(&evs[j]) == "dora" {
                        k.add_dora(&evs[j])?;
                        consumed[j] = true;
                    }
                    j += 1;
                }
                // the claim decisions precede a riichi declaration's reach_accepted (the stick
                // is deposited only if nobody rons), which the main loop applies afterwards
                let resp = response(evs, i, a)?;
                let ctx = Ctx::Claim { kind: kk, discarder: a, chankan: false };
                let claims: Vec<(usize, Legal)> = (1..4).map(|rel| (a + rel) % 4).map(|s| (s, k.legal(s, &ctx))).collect();
                // a draw straight after a discard with live-wall tiles left and no abortive draw
                // pending has no public cause but the triple ron (三家和), which the logs write as
                // a bare ryukyoku: the claims' outcome is unknown.  After the last discard it is
                // the exhaustive draw — unless all three seats could ron that tile (three legal
                // rons all declined is no real alternative to the triple ron)
                let triple_ron_legal = claims.iter().all(|(_, l)| l.types[HORA as usize]);
                let unexplained_draw = resp.draw && !k.abortive_pending() && (k.wall > 0 || triple_ron_legal);
                for (sidx, legal) in &claims {
                    let (label, weight) = claim_label(*sidx, legal, &resp, unexplained_draw);
                    visit(&Decision { st: k, seat: *sidx, ctx: &ctx, legal, label: Some(label), weight, focus: Some(kk),
                                      own: false, focus_actor: Some(a), event: i });
                }
                for rel in 1..4 {
                    let sidx = (a + rel) % 4;
                    if !resp.ronners.contains(&sidx) {
                        k.note_pass(sidx, kk);
                    }
                }
            }
            "chi" | "pon" => {
                k.call(ev)?;
                let a = seat(ev, "actor")?;
                if let Some(nx) = evs.get(i + 1) {
                    if ty(nx) == "dahai" && actor(nx) == Some(a as i64) {
                        let label = (DAHAI, tile_kind(s(nx, "pai")?)? as i64);
                        let ctx = Ctx::AfterCall {
                            called: tile_kind(s(ev, "pai")?)?,
                            consumed: Kyoku::consumed(ev)?.iter().map(|p| tile_kind(p)).collect::<R<Vec<_>>>()?,
                            is_chi: t == "chi",
                        };
                        let legal = k.legal(a, &ctx);
                        visit(&Decision { st: k, seat: a, ctx: &ctx, legal: &legal, label: Some(label), weight: 1.0, focus: None,
                                          own: true, focus_actor: None, event: i });
                    }
                }
            }
            "daiminkan" => k.call(ev)?,
            "ankan" => {
                // Tenhou: a kokushi hand cannot rob another seat's ankan (他家が暗槓した牌では
                // 国士無双はあがれない), so an ankan prompts no decision
                k.ankan(ev)?;
            }
            "kakan" => {
                // chankan: the other seats may ron the added tile (before the kan's rinshan
                // draw and before a pending open-kan dora is flipped)
                let a = seat(ev, "actor")?;
                let kk = k.kakan(ev)?;
                let resp = response(evs, i, a)?;
                // nothing but a triple chankan (三家和) ends the kyoku right at a kakan (an
                // abortive four-kan draw waits for the rinshan discard): outcome unknown
                let unexplained_draw = resp.draw;
                for rel in 1..4 {
                    let sidx = (a + rel) % 4;
                    let ctx = Ctx::Claim { kind: kk, discarder: a, chankan: true };
                    let legal = k.legal(sidx, &ctx);
                    let (label, weight) = claim_label(sidx, &legal, &resp, unexplained_draw);
                    visit(&Decision { st: k, seat: sidx, ctx: &ctx, legal: &legal, label: Some(label), weight, focus: Some(kk),
                                      own: false, focus_actor: Some(a), event: i });
                }
                for rel in 1..4 {
                    let sidx = (a + rel) % 4;
                    if !resp.ronners.contains(&sidx) {
                        k.note_pass(sidx, kk);
                    }
                }
            }
            "reach" => k.pending[seat(ev, "actor")?] = true,
            "reach_accepted" => k.accept_riichi(seat(ev, "actor")?),
            "dora" => k.add_dora(ev)?,
            "hora" => {
                k.add_deltas(ev)?;
                k.kyotaku = 0;
            }
            "ryukyoku" => k.add_deltas(ev)?,
            _ => {}
        }
    }
    let Some(st) = st else { return Err(Skip) };
    Ok((n_kyoku, st.scores))
}

/// Per-sample conversion statistics.
#[derive(Default, Clone, Debug)]
pub(super) struct FrameStats {
    /// frames whose ground truth fell outside the computed masks (kept with all-true masks)
    pub mismatch: usize,
    pub claim_frames: usize,
    pub chankan_frames: usize,
    /// claim frames with loss weight 0 (a recorded pass whose options were pre-empted)
    pub preempted: usize,
    /// frames where hora was legal, and those where the seat then did not win
    pub hora_legal: usize,
    pub hora_declined: usize,
}

impl FrameStats {
    pub fn add(&mut self, o: &FrameStats) {
        self.mismatch += o.mismatch;
        self.claim_frames += o.claim_frames;
        self.chankan_frames += o.chankan_frames;
        self.preempted += o.preempted;
        self.hora_legal += o.hora_legal;
        self.hora_declined += o.hora_declined;
    }
}

#[derive(Default)]
struct Frames {
    obs: ObsBuf,
    atype: Vec<i64>,
    tile: Vec<i64>,
    weight: Vec<f32>,
    legal_type: Vec<f32>,
    legal_tile: Vec<f32>,
    stats: FrameStats,
}

/// Whether decision `d` becomes a frame: every own-turn decision the log answers, and
/// every claim decision with something besides pass legal.  A claim answered by an action
/// the masks do not allow still gets a frame (all-true masks, counted as a mismatch)
/// rather than being dropped silently — real logs have none (tests pin it).
fn frame_label(d: &Decision) -> Option<(i64, i64)> {
    let label = d.label?;
    if d.own || d.legal.has_claim() || label.0 != PASS {
        Some(label)
    } else {
        None
    }
}

fn emit(f: &mut Frames, d: &Decision, label: (i64, i64)) {
    d.st.encode(d.seat, d.focus, d.own, d.focus_actor, &mut f.obs);
    f.atype.push(label.0);
    f.tile.push(label.1);
    f.weight.push(d.weight);
    let (types, tiles, ok) = d.legal.frame(label.0, label.1);
    if ok {
        f.legal_type.extend(types.iter().map(|&b| b as u8 as f32));
        f.legal_tile.extend(tiles.iter().map(|&b| b as u8 as f32));
    } else {
        // never drop data silently: keep the frame unconstrained and count it
        f.legal_type.extend([1.0f32; N_TYPES]);
        f.legal_tile.extend([1.0f32; NK]);
        f.stats.mismatch += 1;
    }
    let st = &mut f.stats;
    if !d.own {
        st.claim_frames += 1;
        st.chankan_frames += matches!(d.ctx, Ctx::Claim { chankan: true, .. }) as usize;
        st.preempted += (d.weight == 0.0) as usize;
    }
    if d.legal.types[HORA as usize] {
        st.hora_legal += 1;
        st.hora_declined += (label.0 != HORA) as usize;
    }
}

/// One sample per seat with at least `min_frames` frames, with its statistics, and the
/// game's mask mismatches over every seat (a log that breaks the rules somewhere).
pub(super) fn convert_game(evs: &[Value], name: &str, min_frames: usize) -> R<(Vec<(Sample, FrameStats)>, usize)> {
    let mut frames: Vec<Frames> = (0..4).map(|_| Frames::default()).collect();
    let (n_kyoku, scores) = walk_game(evs, &mut |d: &Decision| {
        if let Some(label) = frame_label(d) {
            emit(&mut frames[d.seat], d, label);
        }
    })?;
    if n_kyoku == 0 {
        return Err(Skip);
    }
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by_key(|&s| (-scores[s], s));
    let mut placement = [0usize; 4];
    for (r, &s) in order.iter().enumerate() {
        placement[s] = r;
    }
    let mismatch = frames.iter().map(|f| f.stats.mismatch).sum();
    let mut out = Vec::new();
    for (sidx, f) in frames.into_iter().enumerate() {
        let t = f.atype.len();
        if t < min_frames {
            continue;
        }
        let v = PLACEMENT[placement[sidx]];
        let smp = Sample::new(json!({"name": name, "seat": sidx, "kyokus": n_kyoku, "placement": placement[sidx] + 1,
                                     "final_score": scores[sidx]}))
            .u8("raw.tiles_q", &[NK, TILE_FEATURES], f.obs.tiles_q)
            .f32("obs.players_seq", &[4, 12], f.obs.players)
            .f32("obs.table_seq", &[12], f.obs.table)
            .i64("tgt.action_type", &[], f.atype)
            .i64("tgt.tile_idx", &[], f.tile)
            .f32("tgt.value", &[], vec![v; t])
            .bits("tgt.legal_action_type", &[N_TYPES], &f.legal_type)
            .bits("tgt.legal_tile_of", &[NK], &f.legal_tile)
            .bits("tgt.il_weight", &[], &f.weight);
        out.push((smp, f.stats));
    }
    Ok((out, mismatch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mahjong::tests::fixture_games;

    fn legal_with(types: &[i64]) -> Legal {
        let mut l = Legal::none();
        l.allow(PASS, [true; NK]);
        for &t in types {
            l.allow(t, [true; NK]);
        }
        l
    }

    fn resp(ronners: &[usize], call: Option<(usize, i64, i64)>, draw: bool) -> Response {
        Response { ronners: ronners.to_vec(), call, draw }
    }

    #[test]
    fn pre_empted_passes_get_weight_zero() {
        let (chi, chi_pon) = (legal_with(&[CHI]), legal_with(&[CHI, PON]));
        let (ron, chi_ron) = (legal_with(&[HORA]), legal_with(&[CHI, HORA]));
        let pass = |w: f32| ((PASS, 0), w);
        // the seat's own ron / call: its label, weight 1
        assert_eq!(claim_label(1, &ron, &resp(&[1, 2], None, false), false), ((HORA, 0), 1.0));
        assert_eq!(claim_label(1, &chi_pon, &resp(&[], Some((1, CHI, 3)), false), false), ((CHI, 3), 1.0));
        // nobody took the tile: every pass is the seat's choice
        for l in [&chi, &chi_pon, &ron, &chi_ron] {
            assert_eq!(claim_label(1, l, &resp(&[], None, false), false), pass(1.0));
        }
        // another seat's ron pre-empts chi / pon / kan, never a second ron (the double ron)
        assert_eq!(claim_label(1, &chi, &resp(&[2], None, false), false), pass(0.0));
        assert_eq!(claim_label(1, &chi_ron, &resp(&[2], None, false), false), pass(0.0));
        assert_eq!(claim_label(1, &ron, &resp(&[2], None, false), false), pass(1.0));
        // a pon / daiminkan pre-empts chi; nothing pre-empts a ron or a pon
        assert_eq!(claim_label(1, &chi, &resp(&[], Some((3, PON, 5)), false), false), pass(0.0));
        assert_eq!(claim_label(1, &chi, &resp(&[], Some((2, KAN, 5)), false), false), pass(0.0));
        assert_eq!(claim_label(1, &ron, &resp(&[], Some((3, PON, 5)), false), false), pass(1.0));
        assert_eq!(claim_label(2, &legal_with(&[PON]), &resp(&[], Some((1, CHI, 3)), false), false), pass(1.0));
        // a draw no public event explains (the triple ron): no claim outcome is known
        for l in [&chi, &ron] {
            assert_eq!(claim_label(1, l, &resp(&[], None, true), true), pass(0.0));
        }
    }

    /// A synthetic kyoku whose 70th (last) draw is seat 1's `last`, discarded at once: seats 0,
    /// 2 and 3 hold `hands[0..3]` (seat 1 a hand that `last` does not complete), every other draw
    /// is a chun discarded straight away (nobody's wait, nothing to call); then `end`.
    fn houtei_kyoku(hands: [&[&str]; 3], last: &str, end: Value) -> Vec<Value> {
        let seat1 = ["1m", "1m", "9m", "9m", "1p", "1p", "9p", "9p", "1s", "1s", "9s", "9s", "E"];
        let tehais = vec![hands[0].to_vec(), seat1.to_vec(), hands[1].to_vec(), hands[2].to_vec()];
        let mut evs = vec![json!({"type": "start_game", "kyoku_first": 0}),
                           json!({"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "kyoku": 1, "honba": 0,
                                  "kyotaku": 0, "oya": 0, "scores": [25000, 25000, 25000, 25000], "tehais": tehais})];
        for d in 0..WALL as usize {
            let (a, p) = (d % 4, if d + 1 == WALL as usize { last } else { "C" });
            evs.push(json!({"type": "tsumo", "actor": a, "pai": p}));
            evs.push(json!({"type": "dahai", "actor": a, "pai": p, "tsumogiri": true}));
        }
        evs.push(end);
        evs.push(json!({"type": "end_kyoku"}));
        evs
    }

    /// (seat, ron legal, label, weight) of the claim decisions on the last discard.
    fn last_claims(evs: &[Value]) -> Vec<(usize, bool, i64, f32)> {
        let last = evs.iter().rposition(|e| ty(e) == "dahai").unwrap();
        let mut out = Vec::new();
        walk_game(evs, &mut |d: &Decision| {
            if !d.own && d.event == last {
                assert_eq!(d.st.wall, 0, "houtei");
                out.push((d.seat, d.legal.types[HORA as usize], d.label.unwrap().0, d.weight));
            }
        })
        .ok()
        .unwrap();
        out
    }

    /// A bare draw after the last discard is the exhaustive draw — unless all three other seats
    /// could ron that houtei tile: then it is the triple ron (三家和, logged as a bare ryukyoku
    /// like any abortive draw; three seats all declining a legal ron is no real alternative), and
    /// their claims' outcome is unknown (weight 0), as after any earlier discard.
    #[test]
    fn a_triple_ron_on_the_houtei_tile_has_unknown_claims() {
        // three hands waiting on 5p (46p kanchan); houtei is their yaku
        let w0: &[&str] = &["1m", "2m", "3m", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "4p", "6p"];
        let w2: &[&str] = &["2m", "3m", "4m", "5s", "6s", "7s", "8s", "8s", "8s", "S", "S", "4p", "6p"];
        let w3: &[&str] = &["3m", "4m", "5m", "6s", "7s", "8s", "9s", "9s", "9s", "W", "W", "4p", "6p"];
        let draw = json!({"type": "ryukyoku", "deltas": [0, 0, 0, 0]});
        let claims = last_claims(&houtei_kyoku([w0, w2, w3], "5p", draw.clone()));
        assert_eq!(claims.len(), 3);
        for (seat, ron, label, weight) in claims {
            assert!(ron && label == PASS && weight == 0.0, "seat {seat}: {ron} {label} {weight}");
        }
        // seat 3 waiting on 1p instead: only two rons are possible, the draw is the exhaustive
        // one and the passes are real declines
        let other: &[&str] = &["3m", "4m", "5m", "6s", "7s", "8s", "9s", "9s", "9s", "W", "W", "W", "1p"];
        let claims = last_claims(&houtei_kyoku([w0, w2, other], "5p", draw));
        assert_eq!(claims.iter().map(|c| (c.0, c.1, c.3)).collect::<Vec<_>>(), [(2, true, 1.0), (3, false, 1.0), (0, true, 1.0)]);
    }

    /// The triple ron (三家和), which the logs write as a bare ryukyoku after a discard: all
    /// three other seats had a legal ron (Tenhou aborts only on three declared rons), and
    /// none of their claim frames has a known label (weight 0).
    #[test]
    fn triple_ron_claims_are_legal_rons_with_weight_zero() {
        let mut found = 0;
        for (name, evs) in fixture_games() {
            let mut seen: Vec<(usize, usize, bool, i64, f32)> = Vec::new();
            walk_game(&evs, &mut |d: &Decision| {
                if d.own || ty(&evs[d.event]) != "dahai" || d.st.wall == 0 || d.st.abortive_pending() {
                    return;
                }
                let mut j = d.event + 1;
                while j < evs.len() && matches!(ty(&evs[j]), "dora" | "reach_accepted") {
                    j += 1;
                }
                if evs.get(j).map(ty) == Some("ryukyoku") {
                    seen.push((d.event, d.seat, d.legal.types[HORA as usize], d.label.unwrap().0, d.weight));
                }
            })
            .ok()
            .unwrap();
            assert_eq!(seen.len() % 3, 0, "{name}");
            for (event, seat, ron_legal, label, weight) in seen {
                assert!(ron_legal && label == PASS && weight == 0.0, "{name}: event {event} seat {seat}");
                found += 1;
            }
        }
        assert!(found >= 3, "{found}");
    }
}
