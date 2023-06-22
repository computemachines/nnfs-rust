use std::path::PathBuf;

use crate::{
    activation_functions::ReLU,
    accuracy,
    data::{lin_map, new_root_area, spiral_data, visualize_nn_scatter},
    loss_functions::SoftmaxLossCategoricalCrossentropy,
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
pub use cli::Ch10Args;

// I'm deviating from the book slightly. Using structs and methods is much nicer.
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

fn process_args(args: &Ch10Args) -> (Box<dyn Optimizer>, NetworkConfig) {
    // create optimizer object
    let (name, optimizer): (String, Box<dyn Optimizer>) = match args.command {
        cli::OptimizerCommand::SDG(SDGCommand {
            learning_rate,
            decay_rate,
            momentum,
        }) => (
            format!("ch10-sdg-l{}-d{}-m{}", learning_rate, decay_rate, momentum),
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
                "ch10-adam-l{}-d{}-e{}-b1_{}-b2_{}",
                learning_rate, decay_rate, epsilon, beta_1, beta_2
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
                "ch10-adagrad-l{}-d{}-e{}",
                learning_rate, decay_rate, epsilon
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
                "ch10-rms-l{}-d{}-e{}-r{}",
                learning_rate, decay_rate, epsilon, rho
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
}

pub fn run(args: Ch10Args) {
    // dbg!(&args);
    let num_labels = 3;

    #[allow(non_snake_case)]
    let (data, labels) = spiral_data(100, num_labels);

    let mut network = Network::new(num_labels);

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

    let mut gif = if args.mode == ReportMode::Animate {
        Some(new_root_area(&gif_path, true))
    } else {
        None
    };

    // train in loop
    for epoch in 0..config.num_epochs {
        // perform a forward pass
        let NetworkOutput(loss, prediction) = network.forward(&data, &labels);

        // let prediction_other = loss_activation_other.output.take().unwrap();
        let accuracy = accuracy::get_accuracy(&prediction, &labels);
        // let accuracy_other = analysis_functions::get_accuracy(&prediction_other, &y);

        writer.serialize(EpochRecord {
            epoch,
            loss,
            accuracy,
            learning_rate: optimizer.current_learning_rate(),
        });

        if epoch % 1 == 0 {
            println!(
                "epoch: {}, loss: {:.3}, acc: {:.3}, lr: {:.3}",
                epoch,
                loss,
                accuracy,
                optimizer.current_learning_rate()
            );
            if args.mode == ReportMode::Animate {
                let mut network_clone = network.clone();
                visualize_nn_scatter(
                    &data,
                    &labels,
                    num_labels,
                    |(x, y)| {
                        let NetworkOutput(loss, prediction_vector) =
                            network_clone.forward(&array![[x, y]], &labels);
                        let g = colorgrad::rainbow();
                        let max_arg = prediction_vector
                            .indexed_iter()
                            .max_by(|(_, l1), (_, l2)| l1.partial_cmp(l2).unwrap())
                            .unwrap()
                            .0;
                        let label_color = g.at(max_arg.1 as f64 / num_labels as f64);
                        let hsla = label_color.to_hsla();
                        let saturation = 1.0 / (1.0 + loss);
                        let confidence = prediction_vector[[0, max_arg.1]];
                        HSLColor(
                            hsla.0 / 360.,
                            hsla.1,
                            lin_map(confidence, (1.0 / num_labels as f64)..1.0, 1.0..0.1),
                        )
                        .to_rgba()
                    },
                    gif.as_ref().unwrap(),
                );
            }
        }

        // perform backward pass
        network.backward(&prediction, &labels);

        // update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(&mut network.dense1);
        optimizer.update_params(&mut network.dense2);
        optimizer.post_update_params();
    }
}
