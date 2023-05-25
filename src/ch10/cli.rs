use clap::{Args, arg, Subcommand};

#[derive(Args, Debug, Clone)]
pub struct Ch10Args {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    #[clap(name = "animate")]
    Animate(AnimateArgs),

    #[clap(name = "loss-only")]
    LossOnly(LossOnlyArgs),
}

#[derive(Args, Debug, Clone)]
pub struct AnimateArgs {
    #[clap(short, long, default_value_t = 0)]
    pub skip_frame: u32,

    #[command(subcommand)]
    pub optimizer: OptimizerCommand,
}

#[derive(Args, Debug, Clone)]
pub struct LossOnlyArgs {
    #[clap(short, long, required = true)]
    pub output_dir: String,

    #[command(subcommand)]
    pub optimizer: OptimizerCommand,
}


#[derive(Subcommand, Debug, Clone)]
pub enum OptimizerCommand {
    #[clap(name = "sdg", about = "SDG optimizer")]
    SDG(SDGCommand),
    #[clap(name = "adam", about = "Adam optimizer")]
    Adam(AdamCommand),
    #[clap(name = "adagrad", about = "AdaGrad optimizer")]
    AdaGrad(AdaGradCommand),
    #[clap(name = "rmsprop", about = "RMSProp optimizer")]
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
    #[clap(short, long, default_value = "0.001")]
    pub learning_rate: f64,
    #[clap(short, long, default_value = "0.0")]
    pub decay_rate: f64,
    #[arg(long, default_value = "0.9")]
    pub beta_1: f64,
    #[arg(long, default_value = "0.999")]
    pub beta_2: f64,
}

#[derive(Args, Debug, Clone)]
pub struct AdaGradCommand {
    #[clap(short, long, default_value = "0.01")]
    pub learning_rate: f64,
    #[clap(short, long, default_value = "0.0")]
    pub epsilon: f64,
}

#[derive(Args, Debug, Clone)]
pub struct RMSPropCommand {
    #[clap(short, long, default_value = "0.001")]
    pub learning_rate: f64,
    #[clap(short, long, default_value = "0.9")]
    pub decay_rate: f64,
    #[clap(short, long, default_value = "1e-8")]
    pub epsilon: f64,
}
