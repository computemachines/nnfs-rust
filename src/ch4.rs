use ndarray::prelude::*;

use crate::{activation_functions::{ReLU, Softmax}, neurons::LayerDense};

pub fn run() {
    // create dataset
    let (input_position, _labels) = crate::data::spiral_data(100, 3);

    // create dense layer with 2 input features and 3 output values
    let mut dense1 = LayerDense::new(2, 3);
    let mut dense2 = LayerDense::new(3, 3);

    // create ReLU activation (to be used with Dense layer):
    let mut activation1 = ReLU::new();
    let mut activation2 = Softmax::new();

    // perform forward pass through this layer
    dense1.forward(&input_position);

    // perform forward pass through activation function
    // takes in output from previous layer
    activation1.forward(&dense1.output.as_ref().unwrap());

    dense2.forward(&activation1.output.as_ref().unwrap());

    activation2.forward(&dense2.output.as_ref().unwrap());

    println!("{:?}", activation2.output.as_ref().unwrap().slice(s![0..5, ..]));
}
