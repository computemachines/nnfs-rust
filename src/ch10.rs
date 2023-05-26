use crate::{
    activation_functions::ReLU,
    analysis_functions,
    data::{lin_map, new_root_area, spiral_data, visualize_nn_scatter},
    loss_functions::SoftmaxLossCategoricalCrossEntropy,
    neurons::LayerDense,
    optimizer::{
        Optimizer, OptimizerAdaGrad, OptimizerAdaGradConfig, OptimizerAdam, OptimizerAdamConfig,
        OptimizerRMSProp, OptimizerRMSPropConfig, OptimizerSDG, OptimizerSDGConfig,
    },
};

use approx::AbsDiffEq;
use ndarray::prelude::*;
use plotters::prelude::*;
use seahorse::{Command, Context, Flag, FlagType};

// I'm deviating from the book slightly. Using structs and methods is much easier.
struct NetworkOutput(f64, Array2<f64>);

struct NetworkConfig {
    pub filename_base: String,
    pub num_epochs: usize,
}

#[derive(Clone)]
struct Network {
    dense1: LayerDense,
    dense2: LayerDense,
    activation1: ReLU,
    loss_activation: SoftmaxLossCategoricalCrossEntropy,
}
impl Network {
    fn new(num_labels: usize) -> Self {
        // create dense layer with 2 input features and 64 output values
        let mut dense1 = LayerDense::new(2, 64);

        // create ReLU activation (to be used with dense layer)
        let mut activation1 = ReLU::new();

        // create second dense layer with 64 input features (as we take from lower
        // layer) and 3 output values (output labels)
        let mut dense2 = LayerDense::new(64, num_labels);

        // create softmax classifier with combined loss and activation
        let mut loss_activation = SoftmaxLossCategoricalCrossEntropy::new();

        Self {
            dense1,
            activation1,
            dense2,
            loss_activation,
        }
    }
    fn forward(&mut self, input: &Array2<f64>, y_true: &Array1<usize>) -> NetworkOutput {
        self.dense1.forward(input);
        self.activation1
            .forward(self.dense1.output.as_ref().unwrap());
        self.dense2
            .forward(self.activation1.output.as_ref().unwrap());
        let loss = self
            .loss_activation
            .forward_labels(self.dense2.output.as_ref().unwrap(), &y_true);
        let prediction = self.loss_activation.output.take().unwrap();
        NetworkOutput(loss, prediction)
    }
    fn backward(&mut self, prediction: &Array2<f64>, y_true: &Array1<usize>) {
        self.loss_activation.backward_labels(prediction, y_true);
        self.dense2
            .backward(self.loss_activation.dinputs.as_ref().unwrap());
        self.activation1
            .backward(self.dense2.dinputs.as_ref().unwrap());
        self.dense1
            .backward(self.activation1.dinputs.as_ref().unwrap());
    }
}

pub enum OptimizerTypes {
    SDG,
    ADAGrad,
    RMSProp,
    AdaM,
}
pub fn command() -> Command {
    Command::new("ch10")
        .description("Chapter 10 Optimizers")
        .usage("nnfs-rust ch10 (sdg|adagrad|rmsprop|adam) <hyperparameters> [--help]")
        .command(
            Command::new("sdg")
                .usage("nnfs-rust ch10 sdg <hyperparameters> [--help]")
                .flag(Flag::new("learning_rate", FlagType::Float).alias("lr"))
                .flag(Flag::new("decay_rate", FlagType::Float).alias("dr"))
                .flag(Flag::new("momentum", FlagType::Float).alias("m"))
                .action(|c| run(OptimizerTypes::SDG, c)),
        )
        .command(
            Command::new("adagrad")
                .usage("nnfs-rust ch10 sdg <hyperparameters> [--help]")
                .flag(Flag::new("learning_rate", FlagType::Float).alias("lr"))
                .flag(Flag::new("decay_rate", FlagType::Float).alias("dr"))
                .flag(Flag::new("epsilon", FlagType::Float).alias("e"))
                .action(|c| run(OptimizerTypes::ADAGrad, c)),
        )
        .command(
            Command::new("rmsprop")
                .usage("nnfs-rust ch10 sdg <hyperparameters> [--help]")
                .flag(Flag::new("learning_rate", FlagType::Float).alias("lr"))
                .flag(Flag::new("decay_rate", FlagType::Float).alias("dr"))
                .flag(Flag::new("epsilon", FlagType::Float).alias("e"))
                .flag(Flag::new("rho", FlagType::Float).alias("r"))
        .action(|c| run(OptimizerTypes::RMSProp, c)),
        )
        .command(
            Command::new("adam")
                .usage("nnfs-rust ch10 sdg <hyperparameters> [--help]")
                .flag(Flag::new("learning_rate", FlagType::Float).alias("lr"))
                .flag(Flag::new("decay_rate", FlagType::Float).alias("dr"))
                .flag(Flag::new("epsilon", FlagType::Float).alias("e"))
                .flag(Flag::new("beta1", FlagType::Float).alias("b1"))
                .flag(Flag::new("beta2", FlagType::Float).alias("b2"))
                .action(|c| run(OptimizerTypes::AdaM, c)),
        )
}

fn process_args(optimizer_type: OptimizerTypes, c: &Context) -> (Box<impl Optimizer>, NetworkConfig) {
    // create optimizer object
    let config = OptimizerAdamConfig {
        // learning_rate: 1.0,
        // decay_rate: 1e-4,
        // momentum: 0.9,
        ..Default::default()
    };
    let mut optimizer = OptimizerAdam::from(config);

    let config = NetworkConfig {
        filename_base: format!(
            "plots/ch10-rms-lr{}-dr{}-e{}-rho{}.gif",
            config.learning_rate, config.decay_rate, config.epsilon, config.beta_1,
        ),
        num_epochs: 1000,
    };
    (Box::new(optimizer), config)
}

pub fn run(optimizer_type: OptimizerTypes, c: &Context) {
    let num_labels = 5;

    #[allow(non_snake_case)]
    let (data, labels) = spiral_data(100, num_labels);

    let mut network = Network::new(num_labels);

    let (mut optimizer, config) = process_args(optimizer_type, c);
    let mut losses1 = Array1::zeros(config.num_epochs);

    let gif_path = format!("plots/{}.gif", config.filename_base);

    let mut gif = new_root_area(&gif_path, true);

    // train in loop
    for epoch in 0..config.num_epochs {
        // perform a forward pass
        let NetworkOutput(loss, prediction) = network.forward(&data, &labels);
        // let prediction_other = loss_activation_other.output.take().unwrap();
        let accuracy = analysis_functions::get_accuracy(&prediction, &labels);
        // let accuracy_other = analysis_functions::get_accuracy(&prediction_other, &y);

        losses1[epoch] = loss;
        // losses2[epoch] = loss_other;

        if epoch % 1 == 0 {
            println!("Epoch: {}", epoch);
            println!("Loss: {}", loss);
            println!("Accuracy: {}", accuracy);
            println!("Learning rate: {}", optimizer.current_learning_rate());

            println!("Starting gif frame");

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
                &gif,
            );
            println!("Done gif frame.");
        }

        // perform backward pass
        network.backward(&prediction, &labels);

        // update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(&mut network.dense1);
        optimizer.update_params(&mut network.dense2);
        optimizer.post_update_params();
    }

    let plot_path = format!("plots/{}-loss.png", config.filename_base);
    let root = BitMapBackend::new(&plot_path, (1024 * 2, 768)).into_drawing_area();
    // root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_2d(0..config.num_epochs, 0.0..2.0)
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            losses1.iter().enumerate().map(|(x, y)| (x, *y)),
            BLACK,
        ))
        .unwrap();
}
