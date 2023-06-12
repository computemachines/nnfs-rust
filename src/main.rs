use ndarray::array;
use nnfs::activation_functions::{ReLU, Softmax};
use nnfs::ch10::Ch10Args;
use nnfs::ch11::Ch11Args;
use nnfs::ch14::Ch14Args;
use nnfs::ch15::Ch15Args;
use nnfs::loss_functions::Loss;
use nnfs::loss_functions::{regularization_loss, LossCategoricalCrossentropy};
use nnfs::neurons::LayerDense;
use nnfs::optimizer::{Optimizer, OptimizerSDG};
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
}

fn main() {
    // let mut dense1 = LayerDense::new(1, 3);
    // dense1.weight_regularizer_l2 = 0.01;
    // let mut activation1 = Softmax::new();

    // let mut loss = LossCategoricalCrossentropy::new();

    // let data = array![[1.0], [2.0], [3.0]];
    // let y_true = array![0, 0, 1];

    // let mut optimizer = OptimizerSDG::from(nnfs::optimizer::OptimizerSDGConfig {
    //     learning_rate: 0.1,
    //     ..Default::default()
    // });

    // for epoch in 0..100000 {
    //     dense1.forward(&data);
    //     activation1.forward(dense1.output.as_ref().unwrap());
    //     let data_loss = loss.forward(activation1.output.as_ref().unwrap(), &y_true);
    //     let regularization_loss = regularization_loss(&dense1);
    //     println!(
    //         "epoch: {}, loss: {}, regularization_loss: {}",
    //         epoch, data_loss, regularization_loss
    //     );
    //     loss.backward(activation1.output.as_ref().unwrap(), &y_true);
    //     activation1.backward(loss.dinputs.as_ref().unwrap());
    //     dense1.backward(activation1.dinputs.as_ref().unwrap());

    //     optimizer.pre_update_params();
    //     optimizer.update_params(&mut dense1);
    //     optimizer.post_update_params();
    // }
    // return;

    let cli = Cli::parse();
    match cli.command {
        Commands::Ch10(args) => nnfs::ch10::run(args),
        Commands::Ch11(args) => nnfs::ch11::run(args),
        Commands::Ch14(args) => nnfs::ch14::run(args),
        Commands::Ch15(args) => nnfs::ch15::run(args),
    }
}
