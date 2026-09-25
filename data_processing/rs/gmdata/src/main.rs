//! `gmdata` — GamingMaster data pipelines in one binary.
//!
//!     gmdata generals pipeline -p Human.exe -o data/generals --target 1000
//!     gmdata chess convert --input lichess.pgn.zst --out data/chess
//!     ...
//!
//! Converters write `gm-shard.v1` directories (see the `gmshard` crate);
//! `python -m data_processing.games.rs_shards` turns them into `.pt` shards
//! for `GameShardDataset`. generals keeps its historical per-replay JSON
//! training records (read by `SeqReplayDataset`).

mod common;
mod generals;
mod gomoku;
mod othello;

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
    /// Othello (WTHOR databases)
    Othello {
        #[command(subcommand)]
        cmd: othello::Cmd,
    },
    /// Gomoku (Gomocup .psq records)
    Gomoku {
        #[command(subcommand)]
        cmd: gomoku::Cmd,
    },
}

fn main() -> Result<()> {
    match Cli::parse().game {
        Game::Generals { cmd } => generals::run(cmd),
        Game::Othello { cmd } => othello::run(cmd),
        Game::Gomoku { cmd } => gomoku::run(cmd),
    }
}
