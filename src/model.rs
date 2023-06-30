use std::fmt::Debug;

use ndarray::prelude::*;
use serde_pickle::{DeOptions, SerOptions};

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
pub struct UninitializedModel<T, A, L>
where
    A: FinalActivation<T>,
    L: Loss<T>,
{
    layers: Vec<Box<dyn Layer>>,
    final_activation: Option<A>,
    loss: Option<L>,
    optimizer: Option<Box<dyn crate::optimizer::Optimizer>>,
    accuracy: Option<Box<dyn Accuracy<T>>>,
}
impl<T, A, L> UninitializedModel<T, A, L>
where
    A: FinalActivation<T>,
    L: Loss<T>,
{
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
///
/// T is the type of the final activation function's prediction method output
/// not the final activation layer output.
pub struct Model<T, A, L>
where
    A: FinalActivation<T>,
    L: Loss<T>,
{
    pub layers: Vec<Box<dyn Layer>>,
    pub final_activation: A,
    pub loss: L,
    pub optimizer: Box<dyn crate::optimizer::Optimizer>,
    pub accuracy: Box<dyn Accuracy<T>>,
}
impl<T, A, L> Model<T, A, L>
where
    A: FinalActivation<T>,
    L: Loss<T>,
{
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

    pub fn save_weights_biases(&mut self, weights_filename: &str, biases_filename: &str) {
        // use serde_pickle to save weights and biases
        let mut weights_t: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut biases: Vec<Vec<f64>> = Vec::new();

        for layer in &mut self.layers {
            if let Some(dense) = layer.is_trainable() {
                let dense_weights_t = dense.weights.t().outer_iter().map(|row| row.to_vec()).collect();
                let dense_biases = dense.biases.to_vec();
                weights_t.push(dense_weights_t);
                biases.push(dense_biases);
            }
        }

        serde_pickle::to_writer(
            &mut std::fs::File::create(weights_filename).unwrap(),
            &weights_t,
            SerOptions::default(),
        ).unwrap();
        serde_pickle::to_writer(
            &mut std::fs::File::create(biases_filename).unwrap(),
            &biases,
            SerOptions::default(),
        ).unwrap();
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        let mut prev_output = inputs;
        for layer in &mut self.layers {
            prev_output = layer.forward(prev_output);
        }
        // dbg!(prev_output);
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
        dinputs = self.final_activation.backward(dinputs);
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
}

// the slice function is different for 1d and 2d arrays
impl<A, L> Model<Array1<usize>, A, L>
where
    A: FinalActivation<Array1<usize>>,
    L: Loss<Array1<usize>>,
{
    // Train the model
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        outputs_true: &Array1<usize>,
        config: ModelTrainConfig,
    ) {
        // default value if batch_size is not set
        let mut train_steps = 1;
        let mut validation_steps = 0;

        if let Some(batch_size) = config.batch_size {
            train_steps = inputs.nrows() / batch_size;
            // if there is a remainder add one more step
            if inputs.nrows() % batch_size != 0 {
                train_steps += 1;
            }

            #[allow(non_snake_case)]
            if let Some((X_val, y_val)) = config.validation_data {
                validation_steps = X_val.nrows() / batch_size;
                // if there is a remainder add one more step
                if X_val.nrows() % batch_size != 0 {
                    validation_steps += 1;
                }
            }
        }

        // Initialize accuracy object
        self.accuracy.init(outputs_true);

        // Main training loop
        for epoch in 1..config.epochs + 1 {
            // Print epoch number
            println!("Epoch: {}", epoch);

            let mut epoch_data_loss = 0.0;
            let mut epoch_regularization_loss = 0.0;
            let mut epoch_accuracy = 0.0;
            let mut num_samples = 0;

            if epoch == 380 {
                // The epoch that crashes
                for (layer_index, layer) in &mut self.layers.iter_mut().enumerate() {
                    if let Some(layer) = layer.is_trainable() {
                        println!("layer({layer_index}): {:?}", layer);
                    }
                }
            }

            // Iterate over steps
            for step in 0..train_steps {
                #[allow(non_snake_case)]
                let (batch_X, batch_y) = if let Some(batch_size) = config.batch_size {
                    let start = step * batch_size;
                    let mut end = start + batch_size;
                    if end > inputs.nrows() {
                        end = inputs.nrows();
                    }
                    (
                        inputs.slice(s![start..end, ..]),
                        outputs_true.slice(s![start..end]),
                    )
                } else {
                    (inputs.view(), outputs_true.view())
                };
                #[allow(non_snake_case)]
                let batch_X = batch_X.to_owned();
                let batch_y = batch_y.to_owned();

                // if step == 80 {
                //     for layer in &mut self.layers {
                //         if let Some(layer) = layer.is_trainable() {
                //             println!("layer: {:?}", layer);
                //         }
                //     }
                // }

                // Perform forward pass
                self.forward(&batch_X);
                let outputs = self.output();

                // Calculate loss
                let data_loss = self.loss.calculate(&outputs, outputs_true);
                let regularization_loss = self.reg_loss();

                let loss = data_loss + regularization_loss;

                // Get predictions and calculate accuracy
                let prediction = self.final_activation.prediction();
                let accuracy = self.accuracy.calculate(&prediction, &batch_y);

                // accumulate values for epoch mean calculation
                epoch_data_loss += data_loss * batch_X.nrows() as f64;
                epoch_regularization_loss += regularization_loss * batch_X.nrows() as f64;
                epoch_accuracy += accuracy * batch_X.nrows() as f64;
                num_samples += batch_X.nrows();

                // Perform backward pass
                self.backward(None, &batch_y);

                // Optimize (update parameters)
                self.optimizer.pre_update_params();
                for trainable_layer in self.layers.iter_mut().map(|layer| layer.is_trainable()) {
                    if let Some(layer) = trainable_layer {
                        self.optimizer.update_params(layer);
                    }
                }
                self.optimizer.post_update_params();

                // Print a summary
                if step == 0
                    || config
                        .print_every
                        .is_some_and(|print_every| step % print_every == 0)
                {
                    println!(
                        "epoch: {}, acc: {:.3}, loss: {:.10} (data_loss: {:.3}, reg_loss: {:.3})",
                        epoch,
                        epoch_accuracy / num_samples as f64,
                        epoch_data_loss / num_samples as f64,
                        epoch_data_loss / num_samples as f64,
                        epoch_regularization_loss / num_samples as f64
                    );
                }
            }
        }

        // If there is the validation data calculate the validation loss and accuracy
        #[allow(non_snake_case)]
        if let Some((X_val, y_val)) = config.validation_data {
            let mut total_data_loss = 0.0;
            let mut total_regularization_loss = 0.0;
            let mut total_accuracy = 0.0;
            let mut num_samples = 0;

            // Iterate over steps
            for step in 0..validation_steps {
                #[allow(non_snake_case)]
                let (batch_X, batch_y) = if let Some(batch_size) = config.batch_size {
                    let start = step * batch_size;
                    let end = start + batch_size;
                    (
                        inputs.slice(s![start..end, ..]), // TODO: check if this works with remainder batches
                        outputs_true.slice(s![start..end]),
                    )
                } else {
                    (inputs.view(), outputs_true.view())
                };
                #[allow(non_snake_case)]
                let batch_X = batch_X.to_owned();
                let batch_y = batch_y.to_owned();

                // Perform forward pass
                self.forward(&batch_X);
                let outputs = self.output();

                let data_loss = self.loss.calculate(outputs, &batch_y);
                let regularization_loss = self.reg_loss();
                let predictions = self.final_activation.prediction();
                let accuracy = self.accuracy.calculate(&predictions, &batch_y);

                total_data_loss += data_loss * batch_X.nrows() as f64;
                total_regularization_loss += regularization_loss * batch_X.nrows() as f64;
                total_accuracy += accuracy * batch_X.nrows() as f64;
                num_samples += batch_X.nrows();
            }

            println!(
                "Validation loss: {:.10}",
                total_data_loss / num_samples as f64
            );
            println!(
                "Validation accuracy: {:.3}",
                total_accuracy / num_samples as f64
            );
            println!(
                "Validation regularization loss: {:.3}",
                total_regularization_loss / num_samples as f64
            );
        }
    }
}

impl<A, L> Model<Array2<f64>, A, L>
where
    A: FinalActivation<Array2<f64>>,
    L: Loss<Array2<f64>>,
{
    // Train the model
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        outputs_true: &Array2<f64>,
        config: ModelTrainConfig,
    ) {
        // default value if batch_size is not set
        let mut train_steps = 1;
        let mut validation_steps = 0;

        if let Some(batch_size) = config.batch_size {
            train_steps = inputs.nrows() / batch_size;
            // if there is a remainder add one more step
            if inputs.nrows() % batch_size != 0 {
                train_steps += 1;
            }

            #[allow(non_snake_case)]
            if let Some((X_val, y_val)) = config.validation_data {
                validation_steps = X_val.nrows() / batch_size;
                // if there is a remainder add one more step
                if X_val.nrows() % batch_size != 0 {
                    validation_steps += 1;
                }
            }
        }

        // Initialize accuracy object
        self.accuracy.init(outputs_true);

        // Main training loop
        for epoch in 1..config.epochs + 1 {
            // Print epoch number
            println!("Epoch: {}", epoch);

            let mut epoch_data_loss = 0.0;
            let mut epoch_regularization_loss = 0.0;
            let mut epoch_accuracy = 0.0;
            let mut num_samples = 0;

            // Iterate over steps
            for step in 0..train_steps {
                #[allow(non_snake_case)]
                let (batch_X, batch_y) = if let Some(batch_size) = config.batch_size {
                    let start = step * batch_size;
                    let end = start + batch_size;
                    (
                        inputs.slice(s![start..end, ..]), // TODO: check if this works with remainder batches
                        outputs_true.slice(s![start..end, ..]),
                    )
                } else {
                    (inputs.view(), outputs_true.view())
                };

                #[allow(non_snake_case)]
                let batch_X = batch_X.to_owned();
                let batch_y = batch_y.to_owned();

                // Perform forward pass
                self.forward(&batch_X);
                let outputs = self.output();

                // Calculate loss
                let data_loss = self.loss.calculate(&outputs, outputs_true);
                let regularization_loss = self.reg_loss();

                let loss = data_loss + regularization_loss;

                // Get predictions and calculate accuracy
                let prediction = self.final_activation.prediction();
                let accuracy = self.accuracy.calculate(&prediction, outputs_true);

                // accumulate values for epoch mean calculation
                epoch_data_loss += data_loss * batch_X.nrows() as f64;
                epoch_regularization_loss += regularization_loss * batch_X.nrows() as f64;
                epoch_accuracy += accuracy * batch_X.nrows() as f64;
                num_samples += batch_X.nrows();

                // Perform backward pass
                self.backward(None, &batch_y);

                // Optimize (update parameters)
                self.optimizer.pre_update_params();
                for trainable_layer in self.layers.iter_mut().map(|layer| layer.is_trainable()) {
                    if let Some(layer) = trainable_layer {
                        self.optimizer.update_params(layer);
                    }
                }
                self.optimizer.post_update_params();

                // Print a summary
                // if step == 0
                //     || config
                //         .print_every
                //         .is_some_and(|print_every| step % print_every == 0)
                // {
                // }
            }
            println!(
                "epoch: {}, acc: {:.3}, loss: {:.10} (data_loss: {:.3}, reg_loss: {:.3})",
                epoch,
                epoch_accuracy / num_samples as f64,
                epoch_data_loss / num_samples as f64,
                epoch_data_loss / num_samples as f64,
                epoch_regularization_loss / num_samples as f64
            );
        }

        // If there is the validation data calculate the validation loss and accuracy

        #[allow(non_snake_case)]
        if let Some((X_val, y_val)) = config.validation_data {
            let mut total_data_loss = 0.0;
            let mut total_regularization_loss = 0.0;
            let mut total_accuracy = 0.0;
            let mut num_samples = 0;

            // Iterate over steps
            for step in 0..validation_steps {
                #[allow(non_snake_case)]
                let (batch_X, batch_y) = if let Some(batch_size) = config.batch_size {
                    let start = step * batch_size;
                    let end = start + batch_size;
                    (
                        inputs.slice(s![start..end, ..]), // TODO: check if this works with remainder batches
                        outputs_true.slice(s![start..end, ..]),
                    )
                } else {
                    (inputs.view(), outputs_true.view())
                };

                #[allow(non_snake_case)]
                let batch_X = batch_X.to_owned();
                let batch_y = batch_y.to_owned();

                // Perform forward pass
                self.forward(&batch_X);
                let outputs = self.output();

                let data_loss = self.loss.calculate(outputs, &batch_y);
                let regularization_loss = self.reg_loss();
                let predictions = self.final_activation.prediction();
                let accuracy = self.accuracy.calculate(&predictions, &batch_y);

                total_data_loss += data_loss * batch_X.nrows() as f64;
                total_regularization_loss += regularization_loss * batch_X.nrows() as f64;
                total_accuracy += accuracy * batch_X.nrows() as f64;
                num_samples += batch_X.nrows();
            }

            println!(
                "Validation loss: {:.10}",
                total_data_loss / num_samples as f64
            );
            println!(
                "Validation accuracy: {:.3}",
                total_accuracy / num_samples as f64
            );
            println!(
                "Validation regularization loss: {:.3}",
                total_regularization_loss / num_samples as f64
            );
        }
    }
}

// impl Model<Array2<f64>, Softmax, SoftmaxCrossEntropy> {
//     pub fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
//         self.forward(inputs);
//         self.final_activation.output().to_owned()
//     }
// }

pub struct ModelTrainConfig<'a> {
    pub epochs: usize,
    pub batch_size: Option<usize>,
    pub validation_data: Option<(&'a Array2<f64>, &'a Array2<f64>)>,
    pub print_every: Option<usize>,
}
impl<'a> Default for ModelTrainConfig<'a> {
    fn default() -> Self {
        Self {
            epochs: 1000,
            batch_size: None,
            validation_data: None,
            print_every: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;
    use crate::accuracy::AccuracyBinary;
    use crate::activation_functions::Sigmoid;
    use crate::loss_functions::BinaryCrossentropy;
    use crate::optimizer::{OptimizerSDG, OptimizerSDGConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_model() {
        let mut model = Model::new();
        model.add(LayerDense::new(2, 1));
        model.add_final_activation(Sigmoid::new());

        model.set(
            BinaryCrossentropy::new(),
            OptimizerSDG::from(OptimizerSDGConfig {
                learning_rate: 1.0,
                // decay_rate: 1e-3,
                ..Default::default()
            }),
            AccuracyBinary,
        );
        let mut model = model.finalize();
        {
            let layer = model.layers[0].is_trainable().unwrap();
            // layer.weights.assign(&array![[-0.04620634764432907], [-0.13326412439346313]]);
            // layer.biases.assign(&array![0.08846387267112732]);
            layer.weights.assign(&array![[0.1], [-0.2]]);
            layer.biases.assign(&array![0.1]);
        }

        let test_inputs = array![[0.5, 0.6]];
        let test_outputs = array![[1.]];

        model.forward(&test_inputs);
        let outputs = model.output();
        assert_abs_diff_eq!(outputs, &array![[0.50749]], epsilon = 0.001);

        let data_loss = model.loss.calculate(outputs, &test_outputs);
        assert_abs_diff_eq!(data_loss, 0.6782, epsilon = 0.0001);

        model.backward(None, &test_outputs);
        assert_abs_diff_eq!(
            model.loss.dinputs(),
            &array![[-1.97044552]],
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            model.final_activation.dinputs(),
            &array![[-0.49250056]],
            epsilon = 0.0001
        );

        let layer = model.layers[0].is_trainable().unwrap();
        assert_abs_diff_eq!(
            layer.dweights.as_ref().unwrap(),
            &array![[-0.24625028], [-0.29550034]],
            epsilon = 0.00001
        );
    }
}
