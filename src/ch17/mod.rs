use std::path::PathBuf;

use crate::{
    activation_functions::ReLU,
    analysis_functions,
    data::{lin_map, new_root_area, sine_data, spiral_data, visualize_nn_scatter, plot_regression_data},
    loss_functions::{BinaryCrossentropy, Loss, SoftmaxLossCategoricalCrossentropy},
    neurons::LayerDense,
    optimizer::{
        self, Optimizer, OptimizerAdaGrad, OptimizerAdaGradConfig, OptimizerAdam,
        OptimizerAdamConfig, OptimizerRMSProp, OptimizerRMSPropConfig, OptimizerSDG,
        OptimizerSDGConfig,
    },
};

use approx::AbsDiffEq;
use ndarray::prelude::*;
use plotters::prelude::*;

mod cli;
pub use cli::Ch17Args;

mod network;
use network::{Network, NetworkOutput};
use serde::Serialize;

use self::cli::{AdamCommand, ReportMode, SDGCommand};

struct NetworkConfig {
    pub output_dir: PathBuf,
    pub basename: String,
    pub log_file: PathBuf,
    pub num_epochs: usize,
}

fn process_args(args: &Ch17Args) -> (Box<dyn Optimizer>, NetworkConfig) {
    let prefix = format!("ch17");
    // create optimizer object
    let (name, optimizer): (String, Box<dyn Optimizer>) = match args.command {
        cli::OptimizerCommand::SDG(SDGCommand {
            learning_rate,
            decay_rate,
            momentum,
        }) => (
            format!(
                "{prefix}-2x{}-sdg-l{}-d{}-m{}",
                args.layer_neurons, learning_rate, decay_rate, momentum
            ),
            Box::new(OptimizerSDG::from(OptimizerSDGConfig {
                learning_rate,
                decay_rate,
                momentum,
            })),
        ),
        cli::OptimizerCommand::Adam(AdamCommand {
            learning_rate,
            decay_rate,
            epsilon,
            beta_1,
            beta_2,
        }) => (
            format!(
                "{prefix}-adam-2x{}-l{}-d{}-e{}-b1_{}-b2_{}",
                args.layer_neurons, learning_rate, decay_rate, epsilon, beta_1, beta_2
            ),
            Box::new(OptimizerAdam::from(OptimizerAdamConfig {
                learning_rate,
                decay_rate,
                beta_1,
                beta_2,
                epsilon,
            })),
        ),
        cli::OptimizerCommand::AdaGrad(cli::AdaGradCommand {
            learning_rate,
            epsilon,
            decay_rate,
        }) => (
            format!(
                "{prefix}-2x{}-adagrad-l{}-d{}-e{}",
                args.layer_neurons, learning_rate, decay_rate, epsilon
            ),
            Box::new(OptimizerAdaGrad::from(OptimizerAdaGradConfig {
                learning_rate,
                epsilon,
                decay_rate,
            })),
        ),
        cli::OptimizerCommand::RMSProp(cli::RMSPropCommand {
            learning_rate,
            epsilon,
            decay_rate,
            rho,
        }) => (
            format!(
                "{prefix}-2x{}-rms-l{}-d{}-e{}-r{}",
                args.layer_neurons, learning_rate, decay_rate, epsilon, rho
            ),
            Box::new(OptimizerRMSProp::from(OptimizerRMSPropConfig {
                learning_rate,
                epsilon,
                decay_rate,
                rho,
            })),
        ),
    };

    let config = NetworkConfig {
        output_dir: args.output_dir.clone(),
        basename: name,
        log_file: args.log_file.clone(),
        num_epochs: args.num_epochs,
    };
    (optimizer, config)
}

#[derive(Debug, Serialize)]
struct EpochRecord {
    epoch: usize,
    loss: f64,
    accuracy: f64,
    learning_rate: f64,
    test_loss: f64,
    test_accuracy: f64,
}

pub fn run(args: Ch17Args) {
    let (inputs, outputs_true) = sine_data(1000);
    // println!("inputs: \n{:#?}", inputs);
    // println!("outputs_true: \n{:#?}", outputs_true);

    let mut network = Network::new(args.layer_neurons);
    // Arbitrary precision for measuring accuracy. Regression doesn't have an accuracy metric.
    // We use a fake measure of accuracy to make it easier to assess the model's performance.
    let accuracy_precision = outputs_true.std(0.0) / 250.0;

    let (mut optimizer, config) = process_args(&args);

    // train in loop
    for epoch in 0..config.num_epochs {
        // perform a forward pass
        let NetworkOutput(data_loss, regularization_loss, prediction) =
            network.forward(&inputs, &outputs_true);

        let accuracy = analysis_functions::get_accuracy_regression(
            &prediction,
            &outputs_true,
            accuracy_precision,
        );
        
        // perform backward pass
        network.backward(&outputs_true);
        
        println!("epoch: {:05}, acc: {:.3}, loss: {:.3}, learning rate: {:0.5}", epoch, accuracy, data_loss, optimizer.current_learning_rate());
        if epoch % 10 == 0 {
        plot_regression_data(&inputs.as_slice().unwrap(), outputs_true.as_slice().unwrap(), prediction.as_slice().unwrap(), "sine_data.png");
        }
        // println!("prediction: \n{:#?}", prediction);
        // update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(&mut network.dense1);
        optimizer.update_params(&mut network.dense2);
        optimizer.post_update_params();
    }
    let NetworkOutput(_, _, prediction) =
            network.forward(&inputs, &outputs_true);
    plot_regression_data(&inputs.as_slice().unwrap(), outputs_true.as_slice().unwrap(), prediction.as_slice().unwrap(), "sine_data.png");

}
