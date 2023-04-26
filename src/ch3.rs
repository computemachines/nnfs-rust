use ndarray::prelude::*;

use crate::{data::spiral_data, neurons::LayerDense};

pub fn run1() {
    let inputs = array![
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = array![2.0, 3.0, 0.5];

    let weights2 = array![[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]];
    let biases2 = array![-1.0, 2.0, -0.5];

    let layer1_outputs = inputs.dot(&weights.t()) + &biases;
    let layer2_outputs = layer1_outputs.dot(&weights2.t()) + &biases2;

    println!("{:?}", layer2_outputs);
}

pub fn run2() {
    let (input_position, _labels) = spiral_data(100, 3);

    let mut dense1 = LayerDense::new(2, 3);

    // perform forward pass
    dense1.forward(&input_position);

    println!("{:?}", dense1.output.as_ref().unwrap().slice(s![0..5, ..]));
}
