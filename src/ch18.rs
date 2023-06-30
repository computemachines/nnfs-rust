use crate::{
    accuracy::{Accuracy, AccuracyBinary, AccuracyRegression},
    activation_functions::{FinalActivation, Linear, ReLU, Sigmoid, Softmax, Step},
    data::{plot_regression_data, sine_data, spiral_data},
    loss_functions::{BinaryCrossentropy, Loss, LossCategoricalCrossentropy, MeanSquaredError},
    model::{Layer, Model, ModelTrainConfig, UninitializedModel},
    neurons::LayerDense,
    optimizer::{OptimizerAdam, OptimizerAdamConfig, OptimizerSDG, OptimizerSDGConfig},
};
use approx::assert_relative_eq;
use approx::{assert_abs_diff_eq, AbsDiffEq, RelativeEq};
use clap::{Args, Subcommand, ValueEnum};
use ndarray::prelude::*;

use serde;
use serde_pickle::{self, DeOptions};

#[derive(Args, Debug, Clone)]
pub struct Ch18Args {
    #[arg(value_enum, default_value_t=Entry::Main)]
    entry: Entry,
}
#[derive(Debug, Clone, ValueEnum)]
enum Entry {
    Main,
    Perceptron,
}

type PickleMatrix = Vec<Vec<f64>>;

pub fn run(args: Ch18Args) {
    match args.entry {
        Entry::Main => main(),
        Entry::Perceptron => perceptron_run(),
    }
}

pub fn main() {
    // Create the dataset
    #[allow(non_snake_case)]
    let (X, y) = sine_data(100);
    // Instantiate the (uninitialized) model
    let mut model = Model::new();

    // Add layers to the model
    model.add(LayerDense::new(1, 5));
    model.add(ReLU::new());
    model.add(LayerDense::new(5, 3));
    model.add(ReLU::new());
    model.add(LayerDense::new(3, 1));
    model.add_final_activation(Linear::new());

    // Set the loss, optimizer, and accuracy
    model.set(
        MeanSquaredError::new(),
        OptimizerAdam::from(OptimizerAdamConfig {
            learning_rate: 0.01,
            ..Default::default()
        }),
        AccuracyRegression::default(),
    );

    // Finalize the model. This changes the type of the model from UninitializedModel to Model.
    let mut model = model.finalize();

    model.forward(&X);
    model.backward(None, &y);
    model.train(
        &X,
        &y,
        ModelTrainConfig {
            epochs: 7000,
            print_every: Some(1000),
            ..Default::default()
        },
    );
}

// simple perceptron
pub fn perceptron_run() {
    let binary_inputs = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    // let and_outputs = array![[0.], [0.], [0.], [1.]];
    let or_outputs = array![[0.], [1.], [1.], [1.]];
    // let and_or_outputs = array![[0., 0.], [0., 1.], [0., 1.], [1., 1.]];
    // let xor_outputs = array![[0.], [1.], [1.], [0.]];

    let mut model = Model::new();

    model.add(LayerDense::new(2, 1));
    model.add_final_activation(Sigmoid::new());

    model.set(
        BinaryCrossentropy::new(),
        OptimizerAdam::from(OptimizerAdamConfig {
            learning_rate: 1.0,
            // decay_rate: 1e-3,
            ..Default::default()
        }),
        AccuracyBinary,
    );

    let mut model = model.finalize();

    model.load_weights_biases("model-weights.pkl", "model-biases.pkl");

    model.train(&binary_inputs, &or_outputs, ModelTrainConfig {
        epochs: 100,
        ..Default::default()
    });
    model.forward(&binary_inputs);
    println!("output: {}", model.output());

    // let test_inputs = array![[0.5, 0.5]];
    // let test_outputs = array![[0.]];
    // model.forward(&test_inputs);
    // let layer = model.layers[0].is_trainable().unwrap();
    // let output = model.output();
    // let loss = model.loss.calculate(output, &test_outputs);
    // println!("loss: {}", loss);

    // model.backward(None, &test_outputs);
}
