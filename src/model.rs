use std::fmt::Debug;

use ndarray::prelude::*;
use serde_pickle::DeOptions;

use crate::{
    accuracy::{self, Accuracy},
    activation_functions::{FinalActivation, Softmax},
    loss_functions::{self, Loss},
    neurons::LayerDense,
    optimizer::Optimizer,
};

pub trait Layer {
    fn output(&self) -> &Array2<f64>;
    fn dinputs(&self) -> &Array2<f64>;
    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64>;
    fn backward(&mut self, dinputs: &Array2<f64>) -> &Array2<f64>;
    fn is_trainable(&mut self) -> Option<&mut LayerDense> {
        None
    }
    fn regularization_loss(&self) -> f64 {
        0.0
    }
}

#[derive(Default)]
pub struct UninitializedModel<T, A, L> where A: FinalActivation<T>, L: Loss<T> {
    layers: Vec<Box<dyn Layer>>,
    final_activation: Option<A>,
    loss: Option<L>,
    optimizer: Option<Box<dyn crate::optimizer::Optimizer>>,
    accuracy: Option<Box<dyn Accuracy<T>>>,
}
impl<T, A, L> UninitializedModel<T, A, L> where A: FinalActivation<T>, L: Loss<T> {
    pub fn add<U: Layer + 'static>(&mut self, layer: U) {
        self.layers.push(Box::new(layer));
    }
    pub fn add_final_activation(&mut self, final_activation: A) {
        self.final_activation = Some(final_activation);
    }
    pub fn set(
        &mut self,
        loss: L,
        optimizer: impl Optimizer + 'static,
        accuracy: impl Accuracy<T> + 'static,
    ) {
        self.loss = Some(loss);
        self.optimizer = Some(Box::new(optimizer));
        self.accuracy = Some(Box::new(accuracy));
    }
    pub fn finalize(self) -> Model<T, A, L> {
        Model {
            layers: self.layers,
            final_activation: self
                .final_activation
                .expect("Must add final activation function before finalizing model"),
            loss: self
                .loss
                .expect("Must call set function before finalizing model"),
            optimizer: self
                .optimizer
                .expect("Must call set function before finalizing model"),
            accuracy: self
                .accuracy
                .expect("Must call set function before finalizing model"),
        }
    }
}

/// Feedforward Neural Network Model
/// TODO: add specialized impl backward code for combined SoftmaxCrossEntropy.
pub struct Model<T, A, L> where A: FinalActivation<T>, L: Loss<T> {
    pub layers: Vec<Box<dyn Layer>>,
    pub final_activation: A,
    pub loss: L,
    pub optimizer: Box<dyn crate::optimizer::Optimizer>,
    pub accuracy: Box<dyn Accuracy<T>>,
}

impl<T: Default + Debug, A, L> Model<T, A, L> where A: FinalActivation<T>, L: Loss<T> {
    pub fn new() -> UninitializedModel<T, A, L> {
        UninitializedModel {
            layers: Vec::new(),
            final_activation: None,
            loss: None,
            optimizer: None,
            accuracy: None,
        }
    }
    pub fn load_weights_biases(&mut self, weights_filename: &str, biases_filename: &str) {
        let weights: Vec<Vec<Vec<f64>>> = serde_pickle::from_reader(
            std::fs::File::open(weights_filename).unwrap(),
            DeOptions::default(),
        )
        .unwrap();
        let biases: Vec<Vec<f64>> = serde_pickle::from_reader(
            std::fs::File::open(biases_filename).unwrap(),
            DeOptions::default(),
        )
        .unwrap();

        // Create an iterator over layers.
        let mut layers = self.layers.iter_mut();

        for (layer_weights, layer_biases) in weights.into_iter().zip(biases.into_iter()) {
            // Find the next trainable layer, skipping non-trainable layers.
            let dense = loop {
                match layers.next() {
                    Some(layer) => match layer.is_trainable() {
                        Some(trainable_layer) => break trainable_layer,
                        None => continue,
                    },
                    None => panic!(
                        "Ran out of layers while there are still weights and biases to assign"
                    ),
                }
            };

            let weights_shape = dense.weights.dim();
            dense
                .weights
                .assign(&Array2::from_shape_fn(weights_shape, |(i, j)| {
                    layer_weights[j][i]
                }));

            dense
                .biases
                .assign(&Array1::from_shape_vec((layer_biases.len()), layer_biases).unwrap());
        }

        // Check if there are still layers left after all weights and biases have been used.
        if layers.any(|layer| layer.is_trainable().is_some()) {
            panic!("There are still trainable layers left after all weights and biases have been assigned");
        }
    }
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        let mut prev_output = inputs;
        for layer in &mut self.layers {
            prev_output = layer.forward(prev_output);
        }
        self.final_activation.forward(prev_output);
    }
    pub fn output(&self) -> &Array2<f64> {
        self.final_activation.output()
    }
    // Perform backward pass
    pub fn backward(&mut self, output: Option<&Array2<f64>>, outputs_true: &T) {
        // First call backward method on the loss this will set dinputs property
        // that the layers will try to access
        if output.is_some() {
            self.loss.backward(output.unwrap(), outputs_true);
        } else {
            self.loss
                .backward(self.final_activation.output(), outputs_true);
        }
        // Call backward method going through all the layers in reverse order
        // passing dinputs as a parameter
        let mut dinputs = self.loss.dinputs();
        for layer in self.layers.iter_mut().rev() {
            dinputs = layer.backward(dinputs);
        }
    }
    fn reg_loss(&self) -> f64 {
        self.layers
            .iter()
            .map(|layer| layer.regularization_loss())
            .sum()
    }
    // Train the model
    pub fn train(&mut self, inputs: &Array2<f64>, outputs_true: &T, config: ModelTrainConfig) {
        // Initialize accuracy object
        self.accuracy.init(outputs_true);

        // Main training loop
        for epoch in 1..config.epochs + 1 {
            // Perform forward pass
            self.forward(inputs);
            let outputs = self.output();

            // Calculate loss
            let data_loss = self.loss.calculate(&outputs, outputs_true);
            let regularization_loss = self.reg_loss();
            let loss = data_loss + regularization_loss;

            // Get predictions and calculate accuracy
            let prediction = self.final_activation.prediction();
            let accuracy = self.accuracy.calculate(&prediction, outputs_true);

            // Perform backward pass
            self.backward(None, outputs_true);

            // Optimize (update parameters)
            self.optimizer.pre_update_params();
            for trainable_layer in self.layers.iter_mut().map(|layer| layer.is_trainable()) {
                if let Some(layer) = trainable_layer {
                    self.optimizer.update_params(layer);
                }
            }
            self.optimizer.post_update_params();

            // Print a summary
            if epoch % config.print_every == 0 {
                println!(
                    "epoch: {}, acc: {:.3}, loss: {:.10} (data_loss: {:.3}, reg_loss: {:.3})",
                    epoch, accuracy, loss, data_loss, regularization_loss
                );
            }
        }
    }
}

// impl Model<Array2<f64>, Softmax, SoftmaxCrossEntropy> {
//     pub fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
//         self.forward(inputs);
//         self.final_activation.output().to_owned()
//     }
// }

pub struct ModelTrainConfig {
    pub epochs: usize,
    pub print_every: usize,
}
impl Default for ModelTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            print_every: 100,
        }
    }
}
