use ndarray::prelude::*;

use crate::{
    activation_functions::{ReLU, Softmax},
    loss_functions::Loss,
    neurons::LayerDense,
};

pub fn run() {
    // create dataset
    let (input_position, labels) = crate::data::spiral_data(100, 3);

    // create dense layer with 2 input features and 3 output values
    let mut dense1 = LayerDense::new(2, 3);

    // create ReLU activation (to be used with Dense layer):
    let mut activation1 = ReLU::new();

    // create second dense layer with 3 input features (as we take output
    // of previous layer here) and 3 output values
    let mut dense2 = LayerDense::new(3, 3);

    // create Softmax activation (to be used with Dense layer):
    let mut activation2 = Softmax::new();

    // create loss function
    let loss_function = crate::loss_functions::LossCategoricalCrossentropy::new();

    // perform forward pass through this layer
    dense1.forward(&input_position);

    // perform forward pass through activation function
    // takes in output from previous layer
    activation1.forward(&dense1.output.as_ref().unwrap());

    // perform forward pass through second Dense layer
    dense2.forward(&activation1.output.as_ref().unwrap());

    // perform forward pass through activation function
    activation2.forward(&dense2.output.as_ref().unwrap());

    // lets see output of the first few samples:
    println!(
        "{:?}",
        activation2.output.as_ref().unwrap().slice(s![0..5, ..])
    );

    // perform first pass through activation function
    // it takes in output from the second dense layer and returns loss
    let loss = loss_function.calculate(&activation2.output.as_ref().unwrap(), &labels);

    // print loss value
    println!("loss: {:.3}", loss);

    // calculate accuracy from output of activation2 and targets
    let accuracy = crate::analysis_functions::get_accuracy(
        &activation2.output.as_ref().unwrap(),
        &labels,
    );
    println!("accuracy: {:.3}", accuracy);
}
