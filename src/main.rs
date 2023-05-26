use std::env;

use nnfs::{
    analysis_functions,
    data::{spiral_data, visualize_nn_scatter},
    loss_functions::SoftmaxLossCategoricalCrossEntropy,
    neurons::LayerDense,
    optimizer::OptimizerSDG,
};
use nnfs_rust as nnfs;

use ndarray::prelude::*;
use plotters::prelude::*;

use seahorse::App;

fn main() {
    let app = App::new(env!("CARGO_PKG_NAME"))
        .description(env!("CARGO_PKG_DESCRIPTION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .version(env!("CARGO_PKG_VERSION"))
        .usage("nnfs-rust [command] [options] [--help]")
        .command(nnfs::ch10::command());
    app.run(env::args().collect());
}
