use clap::{Args, arg};

#[derive(Args, Debug, Clone)]
pub struct Ch10Args {
    #[arg(short, long)]
    dummy: String,
}