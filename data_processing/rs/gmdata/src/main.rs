//! `gmdata` — GamingMaster data pipelines in one binary.
//!
//!     gmdata generals pipeline -p Human.exe -o data/generals --target 1000
//!     gmdata chess convert --input lichess.pgn.zst --out data/chess
//!     ...
//!
//!     gmdata label chess --input data/chess --out data/chess_sf --engine ~/gm-engines/bin/stockfish
//!
//! Games whose rules live in `gmrules` (candidate-action games, e.g. `kuhn-poker`)
//! parse their source into trajectories and convert through `cand.rs` (replay with
//! the simulator's rules and encoders).
//!
//! Converters write `gm-shard.v1` directories (see the `gmshard` crate); `gmdata label`
//! adds engine teacher columns to existing shards as overlay shards (see `label/mod.rs`);
//! `python -m data_processing.games.rs_shards` turns them into `.pt` shards
//! for `GameShardDataset`. generals keeps its historical per-replay JSON
//! training records (read by `SeqReplayDataset`); `gmdata generals shard`
//! turns them into gm-shards for the multi-game trainers.

mod cand;
mod chess;
mod common;
mod generals;
mod go;
mod lux;
mod mahjong;
mod gomoku;
mod kuhn_poker;
mod label;
mod othello;
mod pokemon;
mod rules;
mod shogi;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gmdata", version, about = "GamingMaster data pipelines")]
struct Cli {
    #[command(subcommand)]
    game: Game,
}

#[derive(Subcommand)]
enum Game {
    /// generals.io replays
    Generals {
        #[command(subcommand)]
        cmd: generals::Cmd,
    },
    /// Chess (Lichess PGN dumps)
    Chess {
        #[command(subcommand)]
        cmd: chess::Cmd,
    },
    /// Othello (WTHOR databases)
    Othello {
        #[command(subcommand)]
        cmd: othello::Cmd,
    },
    /// Go (SGF records)
    Go {
        #[command(subcommand)]
        cmd: go::Cmd,
    },
    /// Shogi (floodgate CSA records)
    Shogi {
        #[command(subcommand)]
        cmd: shogi::Cmd,
    },
    /// Lux AI Season 3 (Kaggle episode replays)
    Lux {
        #[command(subcommand)]
        cmd: lux::Cmd,
    },
    /// Riichi mahjong (mjai logs)
    Mahjong {
        #[command(subcommand)]
        cmd: mahjong::Cmd,
    },
    /// Pokémon Showdown replays
    Pokemon {
        #[command(subcommand)]
        cmd: pokemon::Cmd,
    },
    /// Gomoku (Gomocup .psq records)
    Gomoku {
        #[command(subcommand)]
        cmd: gomoku::Cmd,
    },
    /// Kuhn poker (the toy reference game of the gmrules / candidate-action framework)
    KuhnPoker {
        #[command(subcommand)]
        cmd: kuhn_poker::Cmd,
    },
    /// Generic tools for every gmrules game: list, check, random-play / JSONL → shards
    Rules {
        #[command(subcommand)]
        cmd: rules::Cmd,
    },
    /// Engine teacher labels for existing gm-shards (overlay shards; see label/mod.rs)
    Label {
        #[command(subcommand)]
        cmd: label::Cmd,
    },
}

fn main() -> Result<()> {
    match Cli::parse().game {
        Game::Generals { cmd } => generals::run(cmd),
        Game::Othello { cmd } => othello::run(cmd),
        Game::Chess { cmd } => chess::run(cmd),
        Game::Gomoku { cmd } => gomoku::run(cmd),
        Game::Go { cmd } => go::run(cmd),
        Game::Lux { cmd } => lux::run(cmd),
        Game::Mahjong { cmd } => mahjong::run(cmd),
        Game::Pokemon { cmd } => pokemon::run(cmd),
        Game::Shogi { cmd } => shogi::run(cmd),
        Game::KuhnPoker { cmd } => kuhn_poker::run(cmd),
        Game::Rules { cmd } => rules::run(cmd),
        Game::Label { cmd } => label::run(cmd),
    }
}
