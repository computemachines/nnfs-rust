use nnfs::{ch10::Ch10Args, ch18::Ch18Args};
use nnfs::ch11::Ch11Args;
use nnfs::ch14::Ch14Args;
use nnfs::ch15::Ch15Args;
use nnfs::ch16::Ch16Args;
use nnfs::ch17::Ch17Args;
use nnfs_rust as nnfs;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    Ch10(Ch10Args),
    Ch11(Ch11Args),
    Ch14(Ch14Args),
    Ch15(Ch15Args),
    Ch16(Ch16Args),
    Ch17(Ch17Args),
    Ch18(Ch18Args),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Ch10(args) => nnfs::ch10::run(args),
        Commands::Ch11(args) => nnfs::ch11::run(args),
        Commands::Ch14(args) => nnfs::ch14::run(args),
        Commands::Ch15(args) => nnfs::ch15::run(args),
        Commands::Ch16(args) => nnfs::ch16::run(args),
        Commands::Ch17(args) => nnfs::ch17::run(args),
        Commands::Ch18(args) => nnfs::ch18::run(args),
    }
}
