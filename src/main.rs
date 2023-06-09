use nnfs::{ch10::Ch10Args};
use nnfs::ch11::Ch11Args;
use nnfs::ch14::Ch14Args;
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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Ch10(args) => nnfs::ch10::run(args),
        Commands::Ch11(args) => nnfs::ch11::run(args),
        Commands::Ch14(args) => nnfs::ch14::run(args),
    }
}
