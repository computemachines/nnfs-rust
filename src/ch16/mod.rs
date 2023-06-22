use std::path::PathBuf;

use crate::{
    activation_functions::ReLU,
    accuracy,
    data::{lin_map, new_root_area, spiral_data, visualize_nn_scatter},
    loss_functions::{SoftmaxLossCategoricalCrossentropy, BinaryCrossentropy, Loss},
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
pub use cli::Ch16Args;

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

fn process_args(args: &Ch16Args) -> (Box<dyn Optimizer>, NetworkConfig) {
    let prefix = format!("ch16");
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

pub fn run(args: Ch16Args) {
    let (data, labels) = spiral_data(100, 2);
    let (test_data, test_labels) = spiral_data(100, 2);

    let labels = labels.into_shape((100*2, 1)).unwrap().mapv(|x| x as f64);
    let test_labels = test_labels.into_shape((100*2, 1)).unwrap().mapv(|x| x as f64);

    let mut network = Network::new(args.layer_neurons, args.l2reg, args.dropout);

    let (mut optimizer, config) = process_args(&args);
    // let mut losses1 = Array1::zeros(config.num_epochs);
    let epoch_data_path = config
        .output_dir
        .join(format!("{}-epochs.csv", config.basename));
    let mut writer = csv::Writer::from_path(&epoch_data_path).unwrap();

    let gif_path: String = config
        .output_dir
        .join(format!("{}-animation.gif", config.basename))
        .to_str()
        .unwrap()
        .to_string();

    // if output dir does not exist, create it
    if !config.output_dir.exists() {
        std::fs::create_dir_all(&config.output_dir).unwrap();
    }

    let mut gif = if args.mode == ReportMode::Animate {
        Some(new_root_area(&gif_path, true))
    } else {
        None
    };

    // train in loop
    for epoch in 0..config.num_epochs {
        // perform a forward pass
        let NetworkOutput(data_loss, regularization_loss, prediction) =
            network.forward(&data, &labels);

        let accuracy = accuracy::get_accuracy_binary(&prediction, &labels);

        // evaluate model performance using test data and log results
        if epoch % 100 == 0 {
            let NetworkOutput(test_data_loss, test_regularization_loss, test_prediction) =
                network.validate(&test_data, &test_labels);

            let test_accuracy = accuracy::get_accuracy_binary(&test_prediction, &test_labels);

            println!(
                "
Epoch: {},
Data Loss: {:.3}, Regularization Loss: {:.3}, Accuracy: {:.3},
Test Data Loss: {:.3}, Test Accuracy: {:.3}",
                epoch, data_loss, regularization_loss, accuracy, test_data_loss, test_accuracy
            );

            writer
                .serialize(EpochRecord {
                    epoch,
                    loss: data_loss + regularization_loss,
                    accuracy,
                    learning_rate: optimizer.current_learning_rate(),
                    test_loss: test_data_loss + test_regularization_loss,
                    test_accuracy,
                })
                .unwrap();

            // if args.mode == ReportMode::Animate {
            //     let mut network_clone = network.clone();
            //     visualize_nn_scatter(
            //         &test_data,
            //         &test_labels,
            //         num_labels,
            //         |(x, y)| {
            //             let NetworkOutput(data_loss, regularization_loss, prediction_vector) =
            //                 network_clone.forward(&array![[x, y]], &labels);
            //             let g = colorgrad::rainbow();
            //             let max_arg = prediction_vector
            //                 .indexed_iter()
            //                 .max_by(|(_, l1), (_, l2)| l1.partial_cmp(l2).unwrap())
            //                 .unwrap()
            //                 .0;
            //             let label_color = g.at(max_arg.1 as f64 / num_labels as f64);
            //             let hsla = label_color.to_hsla();
            //             let saturation = 1.0 / (1.0 + data_loss + regularization_loss);
            //             let confidence = prediction_vector[[0, max_arg.1]];
            //             HSLColor(
            //                 hsla.0 / 360.,
            //                 hsla.1,
            //                 lin_map(confidence, (1.0 / num_labels as f64)..1.0, 1.0..0.1),
            //             )
            //             .to_rgba()
            //         },
            //         gif.as_ref().unwrap(),
            //     );
            // }
        }

        // perform backward pass
        network.backward(&prediction, &labels);

        // update weights and biases
        optimizer.pre_update_params();
        // if epoch > 5000 { // freeze first layer half way through
            optimizer.update_params(&mut network.dense1);
        // }
        optimizer.update_params(&mut network.dense2);
        optimizer.post_update_params();
    }
    println!("dense1 weights: \n{:?}", network.dense1.weights);
}
