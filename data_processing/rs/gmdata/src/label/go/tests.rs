//! Go labeller tests: replay / verification (fixtures, corruption, real shards when
//! present), end to end with the fake KataGo ([`super::fake::KataFake`]: captures, ko,
//! passes, handicap, retries, coordinate checks, cache), and with the real KataGo when
//! it is installed (skipped otherwise).
//!
//! Real data: `GM_LABEL_GO_SHARDS` (a directory of go shards, searched recursively),
//! else `${GM_DATA_SHARDS:-$HOME/gm-data/shards}/go`. Real engine: `GM_KATAGO` or
//! `${GM_ENGINES:-$HOME/gm-engines}/bin/katago` or `katago` on PATH, with the network
//! `GM_KATAGO_MODEL` or `${GM_ENGINES:-$HOME/gm-engines}/nets/katago/default.bin.gz`.

use super::fake::{KataFake, FAKE_MODEL, FAKE_VERSION};
use super::*;
use crate::label::driver::{self, find_shards, RunOpts};
use crate::label::engine::LineHandler;
use crate::label::game::ShardData;
use crate::label::testkit::{engine_path, overlay_tensor, quiet_opts, tmpdir};
use gmshard::{Column, DType};

const K: usize = DEFAULT_MAX_MOVES;

/// Capture to live, B to move: black's corner group {ba, ab, bb} is in atari (last
/// liberty aa) and survives only by capturing the two white stones {ac, bc} (in atari,
/// last liberty cc) at C7 = SGF "cc" = (2, 2) → point 20. The black stone at "cg" =
/// (6, 2) occupies the vertically mirrored point.
const CAPTURE: &str = "(;GM[1]SZ[9]KM[7.5]RU[Chinese]AB[ba][ab][bb][ad][bd][cg]AW[ca][cb][ac][bc];B[cc];W[gc];B[gg];W[ee];B[eg])";
const CAPTURE_POINT: i64 = 2 * 9 + 2;
/// B captures at "cb" = (1, 2), taking the white stone at "bb" = (1, 1) → white may not
/// retake at point 10 on frame 1 (simple ko).
const KO: &str = "(;GM[1]SZ[9]KM[6.5]RU[Japanese]AB[ba][ab][bc]AW[ca][db][cc][bb];B[cb];W[gg];B[bb];W[cg];B[gc])";
const KO_POINT: i64 = 9 + 1;
/// Frame 2 follows white's own pass (turn_order); moves 4 and 5 are passes, so frames
/// 6 and 7 come after a double pass.
const PASSES: &str = "(;GM[1]SZ[9]KM[7]RU[Chinese];B[ee];W[];W[cc];B[gg];W[];B[];W[dd];B[ff])";
const PASSES_MASK: [bool; 8] = [true, true, false, true, true, true, false, false];
/// Two handicap stones, white moves first, no komi (territory default 6.5).
const HANDICAP: &str = "(;GM[1]SZ[9]HA[2]RU[Japanese]AB[cc][gg];W[ee];B[cg];W[gc])";

fn lab9() -> GoLabeller {
    GoLabeller { n: 9, ..GoLabeller::default() }
}

fn fake_cmd() -> EngineCommand {
    EngineCommand::in_process("fake-katago", KataFake::new)
}

/// Convert SGF texts (named in order) with the real converter into `root/base`.
fn convert(root: &Path, n: i32, sgfs: &[(&str, &str)]) {
    let dir = root.join("sgf");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in sgfs {
        std::fs::write(dir.join(format!("{name}.sgf")), text).unwrap();
    }
    let stats = crate::go::convert(&dir, &root.join("base"), 256, n, None, 1, true, true, Default::default()).unwrap();
    assert_eq!(stats["converted"], sgfs.len(), "{stats}");
}

fn scenario_root(tag: &str) -> PathBuf {
    let root = tmpdir(tag);
    convert(&root, 9, &[("a_capture", CAPTURE), ("b_ko", KO), ("c_passes", PASSES), ("d_handicap", HANDICAP)]);
    root
}

fn load(dir: &Path) -> ShardData {
    ShardData::load(ShardReader::open(dir).unwrap()).unwrap()
}

fn f32s(t: &gmshard::Tensor) -> Vec<f32> {
    t.rows_f32(0, t.frames()).unwrap().into_owned()
}

fn i64s(t: &gmshard::Tensor) -> Vec<i64> {
    let Column::I64(v) = &t.col else { panic!("{} is not i64", t.name) };
    v.clone()
}

fn bools(t: &gmshard::Tensor) -> Vec<bool> {
    let Column::Bool(v) = &t.col else { panic!("{} is not bool", t.name) };
    v.iter().map(|&x| x != 0).collect()
}

#[test]
fn replay_rebuilds_the_scenarios() {
    let root = scenario_root("go-replay");
    let data = load(&root.join("base").join("go_00000"));
    let lab = lab9();
    let mut all = Vec::new();
    for i in 0..data.reader.n_samples() {
        let v = data.view(i);
        let pos = lab.replay(&v).unwrap();
        lab.verify(&v, &pos).unwrap();
        all.push(pos);
    }
    // capture: 10 setup stones, black to move; after the capture the two white stones are gone
    let cap = &all[0];
    assert_eq!(cap[0].game.q.initial.len(), 10);
    assert_eq!((cap[0].mover, cap[0].game.q.initial_player, cap[0].game.q.komi), (BLACK, BLACK, 7.5));
    assert_eq!(cap[0].game.q.rules, json!("chinese"));
    assert_eq!((cap[0].board.at((2, 0)), cap[0].board.at((2, 1)), cap[0].board.at((2, 2))), (WHITE, WHITE, EMPTY));
    assert_eq!((cap[1].board.at((2, 0)), cap[1].board.at((2, 1)), cap[1].board.at((2, 2))), (EMPTY, EMPTY, BLACK));
    // ko: frame 1 has the ko point, and white's retake there is illegal
    let ko = &all[1];
    assert_eq!(ko[1].board.ko, Some((1, 1)));
    assert!(!ko[1].board.clone().play(WHITE, Some((1, 1))));
    assert_eq!(ko[0].game.q.rules, json!("japanese"));
    // passes: skip reasons
    let skips: Vec<Option<&str>> = all[2].iter().map(|p| p.skip).collect();
    assert_eq!(skips, vec![None, None, Some("turn_order"), None, None, None, Some("after_double_pass"), Some("after_double_pass")]);
    assert_eq!(all[2][0].game.q.komi, 7.0);
    // handicap: white first, komi unknown → territory default
    let h = &all[3];
    assert_eq!((h[0].mover, h[0].game.q.initial_player, h[0].game.q.komi, h[0].game.initial_black), (WHITE, WHITE, 6.5, 2));
    assert_eq!(h[0].game.q.initial, vec![(BLACK, 2 * 9 + 2), (BLACK, 6 * 9 + 6)]);
    // cache keys: same stones, different history → different keys
    let k0 = lab.key(&all[0][0]);
    assert_eq!(k0, lab.key(&all[0][0]));
    assert_ne!(k0, lab.key(&all[0][1]));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn replay_verifies_every_fixture_game_and_catches_corruption() {
    let root = tmpdir("go-fixtures");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/go");
    crate::go::convert(&fixtures, &root.join("base"), 16, 19, None, 1, true, true, Default::default()).unwrap();
    let lab = GoLabeller::default();
    let (mut samples, mut frames, mut skipped) = (0, 0, 0);
    for (dir, _) in find_shards(&root.join("base"), &root.join("out")).unwrap() {
        lab.check_shard(&ShardReader::open(&dir).unwrap()).unwrap();
        let data = load(&dir);
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let pos = lab.replay(&v).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", dir.display()));
            lab.verify(&v, &pos).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", dir.display()));
            samples += 1;
            frames += pos.len();
            skipped += pos.iter().filter(|p| p.skip.is_some()).count();
        }
    }
    assert!(samples >= 50 && frames > 5000, "{samples} samples, {frames} frames");
    assert!(skipped < frames / 20, "{skipped} skipped frames");
    // flip one stone plane bit of frame 5 of sample 1 → verification names the frame
    let dir = root.join("base").join("go_00000");
    let mut data = load(&dir);
    let t = data.tensors.get_mut("obs.board_seq").unwrap();
    let Column::Bits(bytes, per) = &mut t.col else { panic!("board_seq is packed") };
    let start = data.reader.samples[1].start + 5;
    let frame_bytes = per.div_ceil(8);
    bytes[start * frame_bytes + 20] ^= 0x10;
    let v = data.view(1);
    let err = format!("{:#}", lab.verify(&v, &lab.replay(&v).unwrap()).unwrap_err());
    assert!(err.contains("obs.board_seq: frame 5 element"), "{err}");
    // a wrong komi in the globals is caught too
    let mut data = load(&dir);
    let Column::F32(g) = &mut data.tensors.get_mut("obs.globals_seq").unwrap().col else { panic!() };
    g[data.reader.samples[2].start * GLOBAL_FEATURES + 1] += 0.25;
    let v = data.view(2);
    let err = format!("{:#}", lab.verify(&v, &lab.replay(&v).unwrap()).unwrap_err());
    assert!(err.contains("obs.globals_seq: frame 0 element 1"), "{err}");
    // shards of another board size / converter are refused
    assert!(lab9().check_shard(&ShardReader::open(&dir).unwrap()).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// Label the scenarios with `cmd`; returns (summary, base shard, labelled shard).
fn label_scenarios(root: &Path, lab: GoLabeller, cmd: EngineCommand, workers: usize) -> (driver::Summary, ShardData, ShardData) {
    let mut o = quiet_opts(root, cmd);
    o.workers = workers;
    let s = driver::run(lab, &o).unwrap();
    (s, load(&root.join("base").join("go_00000")), load(&root.join("out").join("go_00000")))
}

#[test]
fn end_to_end_with_the_fake_engine() {
    let root = scenario_root("go-e2e");
    let (s, base, out) = label_scenarios(&root, lab9(), fake_cmd(), 2);
    let st = &s.stats;
    assert_eq!((st.samples, st.failed_samples(), st.engine_errors, st.bad_analysis), (4, 0, 0, 0), "{st:?}");
    assert_eq!((st.skipped.get("turn_order"), st.skipped.get("after_double_pass")), (Some(&1), Some(&2)));
    assert_eq!(st.labelled + 3, st.frames);
    assert_eq!(s.teacher["engine"], format!("KataGo {FAKE_VERSION} {FAKE_MODEL}"));
    assert_eq!((s.teacher["policy"].clone(), s.teacher["raw_unit"].clone(), s.teacher["prior"].clone()), (json!({"kind": "visits"}), json!("score_lead"), json!(true)));
    assert_eq!(s.teacher["heads"]["place"], json!({"kind": "categorical", "n": 81, "k": K, "storage": "sparse"}));
    assert_eq!(s.teacher["limits"], json!({"visits": "100"}));
    assert_eq!(s.teacher["game"]["rules"], json!({"area": "chinese", "territory": "japanese"}));

    let r = &out.reader;
    let idx = i64s(&overlay_tensor(r, "tgt.teacher_idx", DType::I64, &[K]));
    let vis = f32s(&overlay_tensor(r, "tgt.teacher_visits", DType::F32, &[K]));
    let q = f32s(&overlay_tensor(r, "tgt.teacher_q", DType::F32, &[K]));
    let raw = f32s(&overlay_tensor(r, "tgt.teacher_raw", DType::F32, &[K]));
    let prior = f32s(&overlay_tensor(r, "tgt.teacher_prior", DType::F32, &[K]));
    let ev = f32s(&overlay_tensor(r, "tgt.engine_value", DType::F32, &[]));
    let evm = bools(&overlay_tensor(r, "tgt.engine_value_mask", DType::Bool, &[]));
    let pi = i64s(&overlay_tensor(r, "tgt.place_probs_idx", DType::I64, &[K]));
    let pv = f32s(&overlay_tensor(r, "tgt.place_probs_val", DType::F32, &[K]));
    let pp = f32s(&overlay_tensor(r, "tgt.pass_probs", DType::F32, &[]));
    let pm = bools(&overlay_tensor(r, "tgt.pass_probs_mask", DType::Bool, &[]));
    let board = f32s(&base.tensors["obs.board_seq"]);
    let row = |v: &[i64], f: usize| v[f * K..(f + 1) * K].to_vec();

    // capture scenario, frame 0: the capture is the most visited move and the top place target
    let f0 = base.reader.samples[0].start;
    assert_eq!(idx[f0 * K], CAPTURE_POINT, "{:?}", row(&idx, f0));
    assert_eq!(pi[f0 * K], CAPTURE_POINT);
    // ko scenario, frame 1: the shard's ko plane marks point 10 and no teacher move retakes there
    let f1 = base.reader.samples[1].start + 1;
    assert_eq!(board[f1 * 4 * 81 + 3 * 81 + KO_POINT as usize], 1.0);
    assert!(evm[f1] && !row(&idx, f1).contains(&KO_POINT), "{:?}", row(&idx, f1));
    // passes scenario: which frames carry a teacher
    let sp = base.reader.sample_range(2);
    assert_eq!(evm[sp.clone()].to_vec(), PASSES_MASK.to_vec());
    for f in 0..r.n_frames {
        let ids = row(&idx, f);
        let n = ids.iter().take_while(|&&a| a >= 0).count();
        assert_eq!(pm[f], evm[f]);
        if !evm[f] {
            assert!(n == 0 && ev[f] == 0.0 && pp[f] == 0.0 && pv[f * K..(f + 1) * K].iter().all(|&x| x == 0.0));
            assert!(q[f * K..(f + 1) * K].iter().chain(&prior[f * K..(f + 1) * K]).all(|x| x.is_nan()));
            continue;
        }
        assert!(n > 0 && ids[n..].iter().all(|&a| a == -1));
        let (v, qs) = (&vis[f * K..f * K + n], &q[f * K..f * K + n]);
        assert!(v.windows(2).all(|w| w[0] >= w[1]), "sorted by visits");
        assert!(prior[f * K..f * K + n].iter().all(|p| (0.0..=1.0).contains(p)));
        assert!(raw[f * K..f * K + n].iter().zip(qs).all(|(r, q)| (r - 40.0 * (q - 0.5)).abs() < 1e-3), "fake scoreLead = 40(wr - 1/2)");
        // the fake's root winrate is its best move's winrate
        let best = qs.iter().copied().fold(0.0, f32::max);
        assert!((ev[f] - (2.0 * best - 1.0)).abs() < 1e-5);
        // re-derive both head targets from the raw columns (normalised visits)
        let total: f32 = v.iter().sum();
        let pass_mass: f32 = ids[..n].iter().zip(v).filter(|(a, _)| **a == 81).map(|(_, x)| x / total).sum();
        assert!((pp[f] - pass_mass).abs() < 1e-5);
        let place: Vec<(i64, f32)> = ids[..n].iter().zip(v).filter(|(a, _)| **a != 81).map(|(a, x)| (*a, x / (total * (1.0 - pass_mass)))).collect();
        for (a, p) in &place {
            if *p > 0.0 {
                let j = pi[f * K..(f + 1) * K].iter().position(|x| x == a).unwrap_or_else(|| panic!("frame {f}: {a} missing"));
                assert!((pv[f * K + j] - p).abs() < 1e-5, "frame {f} point {a}");
            }
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fake_engine_failures_are_retried_on_a_fresh_process() {
    // every fake process fails its 2nd analysis query; the retry (a fresh process's 1st) works
    for (tag, make) in [
        ("crash", (|| KataFake::new().crash_on(2)) as fn() -> KataFake),
        ("error", || KataFake::new().error_on(2)),
        ("hang", || KataFake::new().hang_on(2, Duration::from_millis(900))),
    ] {
        let root = scenario_root(&format!("go-flaky-{tag}"));
        let mut o = quiet_opts(&root, EngineCommand::in_process("flaky", make));
        o.timeout = Duration::from_millis(300);
        o.workers = 1;
        let s = driver::run(lab9(), &o).unwrap();
        assert!(s.stats.retries > 0, "{tag}: {:?}", s.stats);
        if tag == "hang" {
            assert!(s.stats.timeouts > 0, "{:?}", s.stats);
        }
        assert_eq!((s.stats.engine_errors, s.stats.labelled + 3), (0, s.stats.frames), "{tag}: {:?}", s.stats);
        std::fs::remove_dir_all(&root).ok();
    }
    // a dead engine: every query rejected → the run fails loudly
    let root = scenario_root("go-dead");
    let o = quiet_opts(&root, EngineCommand::in_process("dead", || KataFake::new().error_on(1)));
    let err = format!("{:#}", driver::run(lab9(), &o).unwrap_err());
    assert!(err.contains("engine queries failed") && err.contains("injected error"), "{err}");
    std::fs::remove_dir_all(&root).ok();
}

/// Wraps the fake and mirrors every reported move vertically (row r → N−1−r), as if
/// coordinates were numbered from the wrong edge.
struct Mirrored(KataFake);

impl LineHandler for Mirrored {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let out = self.0.handle(line)?;
        Some(
            out.into_iter()
                .map(|l| {
                    let mut v: Value = serde_json::from_str(&l).unwrap();
                    if let Some(infos) = v.get_mut("moveInfos").and_then(Value::as_array_mut) {
                        for m in infos {
                            if let Some(Some(i)) = m["move"].as_str().and_then(|s| katago::parse_gtp(s, 9)) {
                                m["move"] = json!(katago::gtp_of((8 - i / 9) * 9 + i % 9, 9));
                            }
                        }
                    }
                    v.to_string()
                })
                .collect(),
        )
    }
}

#[test]
fn mirrored_coordinates_are_detected() {
    let root = scenario_root("go-mirror");
    let cmd = EngineCommand::in_process("mirrored", || Mirrored(KataFake::new()));
    let mut o = quiet_opts(&root, cmd);
    o.max_engine_error_frac = 1.0;
    o.workers = 1;
    let s = driver::run(lab9(), &o).unwrap();
    // the capture game's mirrored best move lands on the black stone at "ih"
    assert!(s.stats.engine_errors >= 4, "{:?}", s.stats);
    assert!(s.stats.errors.iter().any(|e| e.contains("coordinate mismatch")), "{:?}", s.stats.errors);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn repeated_games_hit_the_cache_and_chunked_queries_agree() {
    let root = tmpdir("go-cache");
    convert(&root, 9, &[("a", CAPTURE), ("b", CAPTURE), ("c", KO)]);
    let mut o = quiet_opts(&root, fake_cmd());
    o.workers = 1;
    let s = driver::run(lab9(), &o).unwrap();
    assert_eq!(s.stats.cache_hits, 5, "the second copy of the capture game is served from the cache: {:?}", s.stats);
    assert_eq!(s.stats.queries + s.stats.cache_hits, s.stats.labelled);
    // one query per 2 turns gives the same labels (the fake is deterministic)
    let mut o2 = o.clone();
    o2.out = root.join("chunked");
    o2.cache_size = 0;
    let s2 = driver::run(GoLabeller { turns_per_query: 2, ..lab9() }, &o2).unwrap();
    assert_eq!((s2.stats.cache_hits, s2.stats.labelled), (0, s.stats.labelled));
    for f in std::fs::read_dir(root.join("out").join("go_00000")).unwrap() {
        let name = f.unwrap().file_name();
        if name != "index.json" {
            let a = std::fs::read(root.join("out").join("go_00000").join(&name)).unwrap();
            assert_eq!(a, std::fs::read(root.join("chunked").join("go_00000").join(&name)).unwrap(), "{name:?}");
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn cli_builds_the_katago_command() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        a: Args,
    }
    let root = tmpdir("go-cli");
    let model = root.join("net.bin.gz");
    std::fs::write(&model, b"x").unwrap();
    let a = Cli::try_parse_from(["go", "--input", "i", "--out", "o", "--engine", "/bin/sh", "--model", model.to_str().unwrap(),
                                 "--analysis-threads", "4", "--max-moves", "8", "--area-rules", r#"{"koRule":"POSITIONAL"}"#])
        .unwrap()
        .a;
    let lab = GoLabeller::new(&a).unwrap();
    let (path, text) = lab.config.clone().unwrap();
    assert!(text.contains("numAnalysisThreads = 4\n") && text.contains("nnMaxBatchSize = 4\n"));
    assert!(path.file_name().unwrap().to_string_lossy().starts_with("gmdata-katago-"));
    assert_eq!((lab.max_moves, lab.model_name.as_deref()), (8, Some("net.bin.gz")));
    assert_eq!(lab.area_rules, json!({"koRule": "POSITIONAL"}));
    assert_eq!(lab.heads()[0].kind, HeadKind::Categorical { n: 361, k: 8 });
    // explicit engine args: no generated config; the model name comes from -model
    let b = Cli::try_parse_from(["go", "--input", "i", "--out", "o", "--engine", "/bin/sh", "--engine-arg", "analysis",
                                 "--engine-arg", "-model", "--engine-arg", "/x/y/kata.bin.gz"])
        .unwrap()
        .a;
    let lb = GoLabeller::new(&b).unwrap();
    assert!(lb.config.is_none());
    assert_eq!(lb.model_name.as_deref(), Some("kata.bin.gz"));
    assert_eq!(lb.teacher_meta(&EngineConfig {
        command: fake_cmd(), options: vec![], limits: Limits::default(), timeout: Duration::from_secs(1), max_moves: 8,
    })["katago_config"], json!("from --engine-arg"));
    // bad values
    for bad in [&["--komi-area", "7.3"][..], &["--board-size", "30"], &["--territory-rules", "{oops"]] {
        let mut argv = vec!["go", "--input", "i", "--out", "o", "--engine", "/bin/sh"];
        argv.extend_from_slice(bad);
        assert!(GoLabeller::new(&Cli::try_parse_from(argv).unwrap().a).is_err(), "{bad:?}");
    }
    assert!(lab9().resolve_limits(&Limits::parse(&["visits=1".into()]).unwrap()).is_err());
    assert!(lab9().resolve_limits(&Limits::parse(&["nodes=10".into()]).unwrap()).is_err());
    // run() refuses a missing network before touching anything
    let c = Cli::try_parse_from(["go", "--input", "i", "--out", "o", "--engine", "/bin/sh", "--model", "/nonexistent.bin.gz"]).unwrap().a;
    assert!(format!("{:#}", run(c).unwrap_err()).contains("not found"));
    std::fs::remove_dir_all(&root).unwrap();
}

// ------------------------------------------------------------------ real data / engine

fn real_go_shards() -> Vec<PathBuf> {
    let root = std::env::var_os("GM_LABEL_GO_SHARDS").map(PathBuf::from).or_else(|| {
        std::env::var_os("GM_DATA_SHARDS")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("gm-data").join("shards")))
            .map(|r| r.join("go"))
    });
    let Some(root) = root.filter(|r| r.is_dir()) else { return Vec::new() };
    let nowhere = std::env::temp_dir().join("gmdata-no-such-output");
    find_shards(&root, &nowhere).unwrap_or_default().into_iter().map(|(d, _)| d).take(2).collect()
}

/// Every sample of (up to two) real converter shards replays and re-encodes bit-exactly.
#[test]
fn real_shards_replay_and_verify() {
    let shards = real_go_shards();
    if shards.is_empty() {
        eprintln!("skipping: no real go shards (set GM_LABEL_GO_SHARDS)");
        return;
    }
    let lab = GoLabeller::default();
    for dir in shards {
        let reader = ShardReader::open(&dir).unwrap();
        lab.check_shard(&reader).unwrap();
        let data = ShardData::load(reader).unwrap();
        let (mut frames, mut skipped, mut failed) = (0usize, 0usize, Vec::new());
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            match lab.replay(&v).and_then(|p| lab.verify(&v, &p).map(|_| p)) {
                Ok(p) => {
                    frames += p.len();
                    skipped += p.iter().filter(|x| x.skip.is_some()).count();
                }
                Err(e) => failed.push(format!("sample {i}: {e:#}")),
            }
        }
        eprintln!("{}: {} samples, {frames} frames verified, {skipped} skipped, {} failed", dir.display(), data.reader.n_samples(), failed.len());
        assert!(failed.is_empty(), "{failed:?}");
        assert_eq!(frames, data.reader.n_frames);
    }
}

/// The installed KataGo + network, or None (skip).
fn real_katago() -> Option<(PathBuf, PathBuf)> {
    let exe = engine_path("GM_KATAGO", "katago")?;
    let model = std::env::var_os("GM_KATAGO_MODEL").map(PathBuf::from).or_else(default_model)?;
    model.is_file().then_some((exe, model))
}

fn real_lab(n: usize, model: &Path) -> (GoLabeller, Vec<String>) {
    let settings = katago::config_settings(8, 1, 8, 16);
    let text = katago::config_text(&settings);
    let path = std::env::temp_dir().join(format!("gmdata-katago-{:016x}.cfg", fnv1a(&text)));
    let args = vec!["analysis".into(), "-config".into(), path.display().to_string(), "-model".into(), model.display().to_string()];
    let lab = GoLabeller { n, analysis_threads: 8, settings, config: Some((path, text)), ..GoLabeller::default() };
    (lab, args)
}

fn real_opts(root: &Path, exe: &Path, args: &[String], visits: u64) -> RunOpts {
    let mut o = quiet_opts(root, EngineCommand::process(exe, args));
    o.workers = 1;
    o.limits = Limits::parse(&[format!("visits={visits}")]).unwrap();
    o.timeout = Duration::from_secs(300);
    o
}

/// KataGo end to end on the scenarios: it takes the capture, never retakes the ko, and
/// accepts every query (initial stones, colours, passes, coordinates).
#[test]
fn real_katago_scenarios() {
    let Some((exe, model)) = real_katago() else {
        eprintln!("skipping: KataGo or its network not installed (scripts/engines/fetch_engines.sh katago)");
        return;
    };
    let root = scenario_root("go-real");
    let (lab, args) = real_lab(9, &model);
    let s = driver::run(lab, &real_opts(&root, &exe, &args, 200)).unwrap();
    eprintln!("real KataGo scenarios: {}", serde_json::to_string(&s.stats).unwrap());
    assert_eq!((s.stats.engine_errors, s.stats.bad_analysis, s.stats.failed_samples()), (0, 0, 0), "{:?}", s.stats);
    assert!(s.teacher["engine"].as_str().unwrap().starts_with("KataGo 1."), "{}", s.teacher["engine"]);
    let out = load(&root.join("out").join("go_00000"));
    let idx = i64s(&out.tensors["tgt.teacher_idx"]);
    let evm = bools(&out.tensors["tgt.engine_value_mask"]);
    let f0 = out.reader.samples[0].start;
    assert_eq!(idx[f0 * K], CAPTURE_POINT, "KataGo's most visited move: {:?}", &idx[f0 * K..(f0 + 1) * K]);
    let f1 = out.reader.samples[1].start + 1;
    assert!(evm[f1] && !idx[f1 * K..(f1 + 1) * K].contains(&KO_POINT));
    assert_eq!(evm[out.reader.sample_range(2)].to_vec(), PASSES_MASK.to_vec());
    std::fs::remove_dir_all(&root).unwrap();
}

/// KataGo on a few games of a real shard: no engine errors, and its favourite move
/// agrees with the strong human's move far above chance.
#[test]
fn real_katago_on_real_games() {
    let (Some((exe, model)), Some(shard)) = (real_katago(), real_go_shards().into_iter().next()) else {
        eprintln!("skipping: needs KataGo and a real go shard");
        return;
    };
    let root = tmpdir("go-real-games");
    let (lab, args) = real_lab(19, &model);
    let mut o = real_opts(&root, &exe, &args, 50);
    o.input = shard.clone();
    o.max_samples = Some(2);
    let s = driver::run(lab, &o).unwrap();
    eprintln!("real KataGo, 2 games of {}: {}", shard.display(), serde_json::to_string(&s).unwrap());
    assert_eq!((s.stats.engine_errors, s.stats.bad_analysis, s.stats.failed_samples()), (0, 0, 0), "{:?}", s.stats);
    let out = load(&root.join("out").join(shard.file_name().unwrap()));
    let idx = i64s(&out.tensors["tgt.teacher_idx"]);
    let evm = bools(&out.tensors["tgt.engine_value_mask"]);
    let place = i64s(&out.tensors["tgt.place_idx"]);
    let passed = bools(&out.tensors["tgt.passed"]);
    let (mut agree, mut n) = (0, 0);
    for f in 0..out.reader.n_frames {
        if evm[f] {
            n += 1;
            let human = if passed[f] { 361 } else { place[f] };
            agree += usize::from(idx[f * K] == human);
        }
    }
    eprintln!("top-1 agreement with the human move: {agree}/{n}");
    assert!(n > 100 && agree * 10 > n * 3, "{agree}/{n}");
    std::fs::remove_dir_all(&root).unwrap();
}
