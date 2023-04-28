use crate::{
    activation_functions::ReLU,
    loss_functions::{Loss, LossCategoricalCrossentropy},
    neurons::LayerDense, analysis_functions::get_accuracy,
};
use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub fn run() {
    // create a dataset
    let (data, labels) = crate::data::vertical_data(100, 3);
    crate::data::plot_scatter(&data, &labels, "vertical.png");

    // create a model
    let mut dense1 = LayerDense::new(2, 3);
    let mut activation1 = ReLU::new();
    let mut dense2 = LayerDense::new(3, 3);
    let mut activation2 = ReLU::new();

    // loss function
    let loss_function = LossCategoricalCrossentropy::new();

    // helper variables
    let mut lowest_loss = f64::INFINITY;
    let mut best_dense1_weights = dense1.weights.clone();
    let mut best_dense1_biases = dense1.biases.clone();
    let mut best_dense2_weights = dense2.weights.clone();
    let mut best_dense2_biases = dense2.biases.clone();

    let training_rate = 0.05;

    // train the model
    for iteration in 0..10000 {
        // generate a new set of weights for iteration
        dense1.weights = Array2::random((2, 3), Normal::new(0., training_rate).unwrap());
        dense1.biases = Array1::random(3, Normal::new(0., training_rate).unwrap());
        dense2.weights = Array2::random((3, 3), Normal::new(0., training_rate).unwrap());
        dense2.biases = Array1::random(3, Normal::new(0., training_rate).unwrap());

        // perform a forward pass of our training data through this layer
        dense1.forward(&data);
        activation1.forward(&dense1.output.as_ref().unwrap());
        dense2.forward(&activation1.output.as_ref().unwrap());
        activation2.forward(&dense2.output.as_ref().unwrap());

        let loss = loss_function.calculate(&activation2.output.as_ref().unwrap(), &labels);

        // calculate accuracy from output of activation2 and targets
        let accuracy = get_accuracy(&activation2.output.as_ref().unwrap(), &labels);

        // if loss is smaller - print and store weights and biases aside
        if loss < lowest_loss {
            println!(
                "New set of weights found, iteration: {}, accuracy: {:.2}, loss: {:.4}",
                iteration, accuracy, loss
            );
            lowest_loss = loss;
            best_dense1_weights = dense1.weights.clone();
            best_dense1_biases = dense1.biases.clone();
            best_dense2_weights = dense2.weights.clone();
            best_dense2_biases = dense2.biases.clone();
        }
    }
}
