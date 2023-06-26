use crate::{
    accuracy::{Accuracy, AccuracyBinary, AccuracyRegression},
    activation_functions::{FinalActivation, Linear, ReLU, Sigmoid, Softmax, Step},
    data::{sine_data, spiral_data, plot_regression_data},
    loss_functions::{Loss, BinaryCrossentropy, LossCategoricalCrossentropy, MeanSquaredError},
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
    Debug,
}

type PickleMatrix = Vec<Vec<f64>>;

pub fn run(args: Ch18Args) {
    match args.entry {
        Entry::Main => main(),
        Entry::Perceptron => perceptron_run(),
        Entry::Debug => load_model(),
    }
}

pub fn load_model() {
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

    let mut model = model.finalize();

    model.load_weights_biases("model-weights.pkl", "model-biases.pkl");

    // model.train(&X, &y, ModelTrainConfig { epochs: 3000, print_every: 1000 });

    // model.forward(&array![[0.1], [0.2]]);
    // let inference = model.output();
    // println!("Inference: {:?}", inference);
    // let layer1_out = model.layers[0].forward(&array![[0.1], [0.2]]);
    // println!("layer1_out: \n{:?}", layer1_out);

    model.forward(&X);
    model.backward(None, &y);
    println!("weights: {:?}", model.layers[0].is_trainable().unwrap().weights);
    println!("dweights: {:?}", model.layers[0].is_trainable().unwrap().dweights);
    let inference = model.output();
    let loss = model.loss.calculate(inference, &y);
    // println!("Inference: {:?}", inference);
    plot_regression_data(X.as_slice().unwrap(), inference.as_slice().unwrap(), y.as_slice().unwrap(), "plot.png");
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

    
    model.load_weights_biases("model-weights.pkl", "model-biases.pkl");

    model.forward(&X);
    model.backward(None, &y);
    model.train(
        &X,
        &y,
        ModelTrainConfig {
            epochs: 7000,
            print_every: 1000,
        },
    );
    // Debugging the gradients
    let layer_index = 2;
    for (index, &weight_value) in model.layers[layer_index]
        .is_trainable()
        .unwrap()
        .weights
        .clone()
        .indexed_iter()
    {
        model.forward(&X);
        let inference = model.output();
        let loss0 = model.loss.calculate(inference, &y);
        model.backward(None, &y);

        // dbg!(model.layers[layer_index].is_trainable().unwrap().dweights.as_ref());
        // return;

        let weight_value = model.layers[layer_index].is_trainable().unwrap().weights[index];
        let dweight_value = 1e-6;
        model.layers[layer_index].is_trainable().unwrap().weights[index] =
            weight_value + dweight_value;
        model.forward(&X);
        let inference = model.output();
        let loss1 = model.loss.calculate(inference, &y);
        let gradient_finite_diff = (loss1 - loss0) / dweight_value;
        let gradient_backprop = model.layers[layer_index]
            .is_trainable()
            .unwrap()
            .dweights
            .as_ref()
            .unwrap()[index];

        // print the weights and their indices with fixed width {.8}
        println!("index: {:?} weight: {:<8.8} gradient_finite_diff: {:<8.10} gradient_backprop: {:<8.10}", index, weight_value, gradient_finite_diff, gradient_backprop);

        // assert_abs_diff_eq!(gradient_finite_diff, gradient_backprop, epsilon=1e-8);

        // reset the gradient value
        model.layers[layer_index]
            .is_trainable()
            .unwrap()
            .dweights
            .as_mut()
            .unwrap()[index] = weight_value;

        // test that subtracting the gradient from the weight value reduces the loss by the expected amount
        model.layers[layer_index].is_trainable().unwrap().weights[index] =
            weight_value - gradient_backprop * 0.0001;
        model.forward(&X);
        let inference = model.output();
        let loss2 = model.loss.calculate(inference, &y);
        println!(
            "loss0: {:<8.10} loss1: {:<8.10} loss2: {:<8.10}, delta: {:+e}",
            loss0,
            loss1,
            loss2,
            loss2 - loss0
        );
        println!(
            "expected delta: {:+e}",
            gradient_backprop * dweight_value * 0.0001
        );
        println!(
            "difference: {:+e}",
            (loss2 - loss0) - (gradient_backprop * dweight_value * 0.0001)
        );

        // reset the gradient value
        model.layers[layer_index]
            .is_trainable()
            .unwrap()
            .dweights
            .as_mut()
            .unwrap()[index] = weight_value;
    }
}

// simple perceptron
pub fn perceptron_run() {
    // Create the dataset
    // let (inputs, y_true) = sine_data(4);

    // let (data, labels) = spiral_data(100, 3);
    let binary_inputs = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let and_outputs = array![[0.], [0.], [0.], [1.]];
    let xor_outputs = array![[0.], [1.], [1.], [0.]];

    // Instantiate the (uninitialized) model
    let mut model = Model::new();

    // Add layers to the model
    model.add(LayerDense::new(2, 1));
    model.add_final_activation(Sigmoid::new());

    // Set the loss, optimizer, and accuracy
    model.set(
        BinaryCrossentropy::new(),
        OptimizerSDG::from(OptimizerSDGConfig {
            learning_rate: 1.0,
            // decay_rate: 1e-3,
            ..Default::default()
        }),
        AccuracyBinary,
    );

    // Finalize the model. This changes the type of the model from UninitializedModel to Model.
    let mut model = model.finalize();
    model.train(
        &binary_inputs,
        &and_outputs,
        ModelTrainConfig {
            epochs: 30,
            print_every: 5,
        },
    );

    // show output from trained model
    model.forward(&binary_inputs);
    let inference = model.output();
    println!("Inference: \n{}", inference);
}
