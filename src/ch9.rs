use std::array;
use std::time::Instant;

use ndarray::{prelude::*, Zip};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::activation_functions::{Softmax, ReLU, };
use crate::data::spiral_data;
use crate::loss_functions::{Loss, LossCategoricalCrossentropy, SoftmaxLossCategoricalCrossentropy};
use crate::model::Layer;
use crate::neurons::LayerDense;

pub fn run() {
    println!("Running ch9");
    // softmax_catagorical_cross_entropy_combined_test();
    full_model_code();
}

pub fn backpropagation_single_relu_neuron() {
    // forward pass
    let x = array![1., -2., 3.];
    let w = array![-3., -1., 2.];
    let b = 1.;

    let xw0 = x[0] * w[0];
    let xw1 = x[1] * w[1];
    let xw2 = x[2] * w[2];
    println!("xw0: {}, xw1: {}, xw2: {}", xw0, xw1, xw2);

    // adding weighted inputs and a bias
    let z = xw0 + xw1 + xw2 + b;
    println!("z: {}", z);

    // ReLU activation function
    let y = f64::max(0., z);
    println!("y: {}", y);

    // backward pass

    // derivative from the next layer
    let dvalue = 1.; // I don't understand why backpropagation starts this way

    // derivative of ReLU and the chain rule
    let drelu_dz = dvalue * if z > 0. { 1. } else { 0. };
    println!("drelu_dz: {}", drelu_dz);

    // derivatives of ReLU w.r.t. the summed multiplication of inputs and weights terms
    // relu(x0w0 + x1w1 + x2w2 + b)
    let dsum_dx0w0 = 1.;
    let drelu_dx0w0 = drelu_dz * dsum_dx0w0;
    println!("drelu_dxw0: {}", drelu_dx0w0);

    let dsum_dx1w1 = 1.;
    let drelu_dx1w1 = drelu_dz * dsum_dx1w1;
    println!("drelu_dxw1: {}", drelu_dx1w1);

    let dsum_dx2w2 = 1.;
    let drelu_dx2w2 = drelu_dz * dsum_dx2w2;
    println!("drelu_dxw2: {}", drelu_dx2w2);

    let dsum_db = 1.;
    let drelu_db = drelu_dz * dsum_db;
    println!("drelu_db: {}", drelu_db);

    // derivatives of the multiplication of inputs and weights w.r.t. the inputs and weights
    let dx0w0_dx0 = w[0];
    let drelu_dx0 = drelu_dx0w0 * dx0w0_dx0;
    println!("drelu_dx0: {}", drelu_dx0);

    let dx1w1_dx1 = w[1];
    let drelu_dx1 = drelu_dx1w1 * dx1w1_dx1;
    println!("drelu_dx1: {}", drelu_dx1);

    let dx2w2_dx2 = w[2];
    let drelu_dx2 = drelu_dx2w2 * dx2w2_dx2;
    println!("drelu_dx2: {}", drelu_dx2);

    let dx0w0_dw0 = x[0];
    let drelu_dw0 = drelu_dx0w0 * dx0w0_dw0;
    println!("drelu_dw0: {}", drelu_dw0);

    let dx1w1_dw1 = x[1];
    let drelu_dw1 = drelu_dx1w1 * dx1w1_dw1;
    println!("drelu_dw1: {}", drelu_dw1);

    let dx2w2_dw2 = x[2];
    let drelu_dw2 = drelu_dx2w2 * dx2w2_dw2;
    println!("drelu_dw2: {}", drelu_dw2);

    // simplified version
    let drelu_dx0 = dvalue * w[0] * (z > 0.) as i32 as f64;
    let drelu_dx1 = dvalue * w[1] * (z > 0.) as i32 as f64;
    let drelu_dx2 = dvalue * w[2] * (z > 0.) as i32 as f64;
    let drelu_dw0 = dvalue * x[0] * (z > 0.) as i32 as f64;
    let drelu_dw1 = dvalue * x[1] * (z > 0.) as i32 as f64;
    let drelu_dw2 = dvalue * x[2] * (z > 0.) as i32 as f64;
    let drelu_db = dvalue * 1. * (z > 0.) as i32 as f64;
    println!("drelu_dx0: {}, drelu_dx1: {}, drelu_dx2: {}, drelu_dw0: {}, drelu_dw1: {}, drelu_dw2: {}, drelu_db: {}", drelu_dx0, drelu_dx1, drelu_dx2, drelu_dw0, drelu_dw1, drelu_dw2, drelu_db);

    // weight gradient
    let dw = array![drelu_dw0, drelu_dw1, drelu_dw2];
    let learning_rate = 0.001;

    // update weights and bias
    let w_updated = w - dw * learning_rate;
    let b_updated = b - drelu_db * learning_rate;

    // new forward pass with updated weights and bias
    let xw0_updated = x[0] * w_updated[0];
    let xw1_updated = x[1] * w_updated[1];
    let xw2_updated = x[2] * w_updated[2];
    let z_updated = xw0_updated + xw1_updated + xw2_updated + b_updated;
    let y_updated = f64::max(0., z_updated);
    println!("y: {}", y);
    println!("y_updated: {}", y_updated);
}

pub fn dinputs_batch() {
    // passed in gradient from the next layer to start the backpropagation (batch)
    let dvalues = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];

    // We have 3 sets of weights - one set for each neuron
    // We have 4 inputs, thus 4 weights
    let mut weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    .reversed_axes();

    // sum weights of given input and multiply by the passed in gradient for this neuron
    let dinputs = dvalues.dot(&weights.t());
}

pub fn dweights_batch() {
    // passed in gradient from the next layer to start the backpropagation (batch)
    let dvalues = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];

    let inputs = array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, 3.3, -0.8]];

    let dweights = inputs.t().dot(&dvalues);
    dbg!(dweights);
}

pub fn dbias_batch() {
    // passed in gradient from the next layer to start the backpropagation (batch)
    let dvalues = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];

    // one bias for each neuron
    // biases are the row vector with a shape (1, neurons)
    let biases = array![[2., 3., 0.5]];

    let dbiases = dvalues.sum_axis(Axis(0)).into_shape((1, 3)).unwrap();
    dbg!(dbiases);
}

pub fn drelu_batch() {
    // Example layer output
    let z = array![[1., 2., -3., -4.], [2., -7., -1., 3.], [-1., 2., 5., -1.]];

    let dvalues = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]];

    // ReLU activation's derivative
    let drelu_dz = z.map(|&x| (x > 0.) as i32 as f64);
    println!("drelu_dz:\n{}", drelu_dz);

    let drelu = drelu_dz * &dvalues;
    println!("drelu:\n{}", drelu);

    let drelu_alt = Zip::from(&dvalues)
        .and(&z)
        .map_collect(|&x, &z| if z > 0. { x } else { 0. });
    println!("drelu_alt:\n{}", drelu_alt);
}

pub fn backpropagation_full_layer_batch() {
    // passed in gradient from the next layer to start the backpropagation (batch)
    let dvalues = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];

    // We have 3 sets of inputs - samples
    let inputs = array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, 3.3, -0.8]];

    // we have 3 sets of weights - one set for each neuron
    // We have 4 inputs, thus 4 weights
    let mut weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    .reversed_axes();

    // one bias for each neuron
    // biases are the row vector with a shape (1, neurons)
    let biases = array![[2., 3., 0.5]];

    // forward pass
    let layer_outputs = &inputs.dot(&weights) + &biases;
    let relu_outputs = layer_outputs.map(|&x| f64::max(0., x));

    // backward pass
    let drelu = Zip::from(&relu_outputs)
        .and(&layer_outputs)
        .map_collect(|&x, &z| if z > 0. { x } else { 0. });

    // dense layer
    // dinputs - multiply by weights
    let dinputs = &drelu.dot(&weights.t());
    // dweights - multiply by inputs
    let dweights = inputs.t().dot(&drelu);
    // dbiases - sum values, do this over samples (first axis), keepdims
    let dbiases = drelu
        .sum_axis(Axis(0))
        .into_shape((1, drelu.shape()[1]))
        .unwrap();

    // update parameters
    let weights = weights - 0.001 * &dweights;
    let biases = biases - 0.001 * &dbiases;

    println!("weights:\n{}", weights);
    println!("biases:\n{}", biases);
}

pub fn softmax_catagorical_cross_entropy_combined_test() {
    let softmax_outputs = array![
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ];

    let class_targets = array![0, 1, 1];

    let start = Instant::now();
    let mut softmax_loss = SoftmaxLossCategoricalCrossentropy::new();
    softmax_loss.backward_labels(&softmax_outputs, &class_targets);
    let dvalues1 = softmax_loss.dinputs.unwrap().clone();
    let elapsed1 = start.elapsed();
    
    let start = Instant::now();
    let mut activation = Softmax::new();
    activation.output = Some(softmax_outputs.clone());
    let mut loss = LossCategoricalCrossentropy::new();
    loss.backward(activation.output.as_ref().unwrap(), &class_targets);
    activation.backward(loss.dinputs.as_ref().unwrap());
    let dvalues2 = activation.dinputs.unwrap();
    let elapsed2 = start.elapsed();

    println!("Gradients combined loss and activation functions: \n{}", dvalues1);
    println!("Gradients separate loss and activation functions: \n{}", dvalues2);

    println!("Combined time[us]: {}\n", elapsed1.as_micros()); // 25us on vm
    println!("Separate time[us]: {}\n", elapsed2.as_micros()); // 249us on vm
}


pub fn full_model_code() {
    // create dataset
    #[allow(non_snake_case)]
    let (X, y) = spiral_data(100, 3);

    // create dense layer with 2 input features and 3 output values
    let mut dense1 = LayerDense::new(2, 3);

    // create ReLU activation (to be used with dense layer)
    let mut activation1 = ReLU::new();

    // create second dense layer with 3 input features (as we take output of
    // previous layers here) and 3 output values
    let mut dense2 = LayerDense::new(3, 3);

    // Create Softmax classifier's combined loss and activation
    let mut loss_activation = SoftmaxLossCategoricalCrossentropy::new();

    // perform forward pass of training data through this layer
    dense1.forward(&X);

    // Perform a forward pass through the activation function that takes the
    // output of the first dense layer here
    activation1.forward(dense1.output.as_ref().unwrap());

    // Perform a forward pass through the second dense layer that takes outputs
    // of activation function of first layer as inputs
    dense2.forward(activation1.output.as_ref().unwrap());

    // Perform a forward pass through the activation+loss functions, take the
    // output of the second dense layer here and returns loss
    let loss = loss_activation.forward_labels(dense2.output.as_ref().unwrap(), &y);

    // Lets see the output of the first few samples
    println!("\n{}", loss_activation.output.as_ref().unwrap().slice(s![0..5, ..]));

    // Print loss value
    println!("Loss: {}", loss);

    // calculate accuracy from output of activation2 and targets
    let accuracy =
        crate::accuracy::get_accuracy(loss_activation.output.as_ref().unwrap(), &y);
    // Print accuracy
    println!("accuracy: {}", accuracy);

    // Backward pass
    let output = loss_activation.output.take().unwrap();
    // dbg!(&output.slice(s![0..5, ..]));
    loss_activation.backward_labels(&output, &y);
    dense2.backward(loss_activation.dinputs.as_ref().unwrap());
    activation1.backward(dense2.dinputs.as_ref().unwrap());
    dense1.backward(activation1.dinputs.as_ref().unwrap());

    println!("dense1 dweights:\n{}", dense1.dweights.unwrap());
    println!("dense1 dbiases:\n{}", dense1.dbiases.unwrap());
    println!("dense2 dweights:\n{}", dense2.dweights.unwrap());
    println!("dense2 dbiases:\n{}", dense2.dbiases.unwrap());
}
