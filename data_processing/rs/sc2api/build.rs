//! Generate the SC2 API messages (proto/s2clientprotocol/*.proto, Blizzard, MIT) with the
//! vendored protoc — no system protobuf compiler needed.
fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);
    let files: Vec<String> = ["common", "data", "debug", "error", "query", "raw", "sc2api", "score",
                              "spatial", "ui"]
        .iter()
        .map(|f| format!("proto/s2clientprotocol/{f}.proto"))
        .collect();
    for f in &files {
        println!("cargo:rerun-if-changed={f}");
    }
    prost_build::compile_protos(&files, &["proto"]).expect("compile the SC2 protos");
}
