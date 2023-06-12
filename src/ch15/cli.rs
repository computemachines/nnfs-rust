use std::path::PathBuf;

use clap::{Args, arg, Subcommand, ValueEnum};

#[derive(Args, Debug, Clone)]
pub struct Ch15Args {
    #[arg(value_enum, default_value_t = ReportMode::LossOnly)]
    pub mode: ReportMode,
    #[arg(long, short, default_value = "plots/")]
    pub output_dir: PathBuf,

    /// The percentage of epochs to render to the gif
    #[arg(long, short, default_value_t = 1.0)]
    pub render_frame_fraction: f64,

    /// The gif frame rate (frames per second)
    #[arg(long, short, default_value_t = 10.0)]
    pub frame_rate: f64,

    /// Total number of training epochs
    #[arg(long, short, default_value_t = 10000)]
    pub num_epochs: usize,

    /// Final loss/accuracy value log file
    #[arg(long, short, default_value = "report.log")]
    pub log_file: PathBuf,

    /// Number of neurons per layer
    #[arg(long, default_value_t = 64)]
    pub layer_neurons: usize,

    #[command(subcommand)]
    pub command: OptimizerCommand,

    /// L2 regularization penalty
    #[arg(long, default_value_t = 5e-4)]
    pub l2reg: f64,

    /// Dropout probability
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
}

#[derive(ValueEnum, Debug, Clone, PartialEq)]
pub enum ReportMode {
    Animate,
    LossOnly,
}

#[derive(Subcommand, Debug, Clone)]
pub enum OptimizerCommand {
    #[command(about = "SDG optimizer")]
    SDG(SDGCommand),
    #[command(about = "Adam optimizer")]
    Adam(AdamCommand),
    #[command(name="adagrad", about = "AdaGrad optimizer")]
    AdaGrad(AdaGradCommand),
    #[command(name="rmsprop", about = "RMSProp optimizer")]
    RMSProp(RMSPropCommand),
}

#[derive(Args, Debug, Clone)]
pub struct SDGCommand {
    #[arg(short, long, default_value = "1.0")]
    pub learning_rate: f64,
    #[arg(short, long, default_value = "0.0")]
    pub decay_rate: f64,
    #[arg(short, long, default_value = "0.0")]
    pub momentum: f64,
}

#[derive(Args, Debug, Clone)]
pub struct AdamCommand {
    #[arg(short, long, default_value = "0.05")]
    pub learning_rate: f64,
    #[arg(short, long, default_value = "5e-5")]
    pub decay_rate: f64,
    #[arg(short, long, default_value = "1e-7")]
    pub epsilon: f64,
    #[arg(long, default_value = "0.9")]
    pub beta_1: f64,
    #[arg(long, default_value = "0.999")]
    pub beta_2: f64,
}

#[derive(Args, Debug, Clone)]
pub struct AdaGradCommand {
    #[arg(short, long, default_value = "1.0")]
    pub learning_rate: f64,
    #[arg(short, long, default_value = "0.0")]
    pub epsilon: f64,
    #[arg(short, long, default_value = "1e-7")]
    pub decay_rate: f64,
}

#[derive(Args, Debug, Clone)]
pub struct RMSPropCommand {
    #[clap(short, long, default_value = "1.0")]
    pub learning_rate: f64,
    #[clap(short, long, default_value = "0")]
    pub decay_rate: f64,
    #[clap(short, long, default_value = "1e-7")]
    pub epsilon: f64,
    #[clap(short, long, default_value = "1e-3")]
    pub rho: f64,
}
