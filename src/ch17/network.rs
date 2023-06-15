use core::num;

use ndarray::prelude::*;
use serde::__private::de;

use crate::{
    activation_functions::{Linear, ReLU, Sigmoid},
    loss_functions,
    loss_functions::{
        BinaryCrossentropy, Loss, MeanSquaredError, SoftmaxLossCategoricalCrossentropy,
    },
    neurons::{LayerDense, LayerDropout},
};

#[derive(Clone)]
pub struct Network {
    pub dense1: LayerDense,
    // pub dropout1: LayerDropout,
    pub dense2: LayerDense,
    pub dense3: LayerDense,
    pub activation1: ReLU,
    pub activation2: ReLU,
    pub activation3: Linear,
    pub loss: MeanSquaredError,
}
pub struct NetworkOutput(pub f64, pub f64, pub Array2<f64>);

impl Network {
    pub fn new(num_neurons_layer: usize) -> Self {
        // create dense layer with 2 input features and 64 output values
        let mut dense1 = LayerDense::new(1, num_neurons_layer);
        // dense1.weight_regularizer_l2 = l2reg;
        // dense1.bias_regularizer_l2 = l2reg;

        // create dropout layer
        // let mut dropout1 = LayerDropout::new(dropout);

        // create ReLU activation (to be used with dense layer)
        let mut activation1 = ReLU::new();

        // create second dense layer with 64 input features (as we take from lower
        // layer) and 64 output values
        let mut dense2 = LayerDense::new(num_neurons_layer, num_neurons_layer);
        // dense2.weight_regularizer_l2 = l2reg;
        // dense2.bias_regularizer_l2 = l2reg;

        // create ReLU activation
        let mut activation2 = ReLU::new();

        // create third dense layer with 64 input features (as we take from lower
        // layer) and 1 output value
        let mut dense3 = LayerDense::new(num_neurons_layer, 1);

        // create linear activation
        let mut activation3 = Linear::new();

        // create loss function
        let mut loss = MeanSquaredError::new();

        Self {
            dense1,
            // dropout1,
            activation1,
            dense2,
            activation2,
            dense3,
            activation3,
            loss,
        }
    }
    pub fn forward<'a>(&'a mut self, input: &Array2<f64>, y_true: &Array2<f64>) -> NetworkOutput {
        self.dense1.forward(input);
        // println!(
        //     "dense1 output: \n{:?}",
        //     self.dense1.output.as_ref().unwrap()
        // );
        self.activation1
            .forward(self.dense1.output.as_ref().unwrap());
        // println!(
        //     "activation1 output: \n{:?}",
        //     self.activation1.output.as_ref().unwrap()
        // );
        // self.dropout1.forward(self.activation1.output.as_ref().unwrap());
        // self.dense2
        //     .forward(self.dropout1.output.as_ref().unwrap());
        self.dense2
            .forward(self.activation1.output.as_ref().unwrap());
        // println!(
        //     "dense2 output: \n{:?}",
        //     self.dense2.output.as_ref().unwrap()
        // );
        self.activation2
            .forward(self.dense2.output.as_ref().unwrap());
        // println!(
        //     "activation2 output: \n{:?}",
        //     self.activation2.output.as_ref().unwrap()
        // );
        self.dense3
            .forward(self.activation2.output.as_ref().unwrap());
        self.activation3
            .forward(self.dense3.output.as_ref().unwrap());
        let data_loss = self
            .loss
            .calculate(self.activation3.output.as_ref().unwrap(), y_true);
        // println!("data_loss: \n{:?}", data_loss);
        let mut regularization_loss: f64 = loss_functions::regularization_loss(&self.dense1)
            + loss_functions::regularization_loss(&self.dense2)
            + loss_functions::regularization_loss(&self.dense3);
        let mut prediction = self.activation3.output.as_ref().unwrap().clone();
        NetworkOutput(data_loss, regularization_loss, prediction)
    }
    pub fn backward(&mut self, y_true: &Array2<f64>) {
        self.loss.backward(
            self.activation3
                .output
                .as_ref()
                .expect("Must call Network#forward(input, y_true) before Network#backward(y_true)"),
            y_true,
        );
        self.activation3
            .backward(self.loss.dinputs.as_ref().unwrap());
        self.dense3
            .backward(self.activation3.dinputs.as_ref().unwrap());
        // println!("loss.dinputs: \n{:?}", self.loss.dinputs.as_ref().unwrap());
        self.activation2
            .backward(self.dense3.dinputs.as_ref().unwrap());
        // println!("activation2.dinputs: \n{:?}", self.activation2.dinputs.as_ref().unwrap());
        self.dense2
            .backward(self.activation2.dinputs.as_ref().unwrap());
        // println!("dense2.dinputs: \n{:?}", self.dense2.dinputs.as_ref().unwrap());
        // self.dropout1.backward(self.dense2.dinputs.as_ref().unwrap());
        self.activation1
            .backward(self.dense2.dinputs.as_ref().unwrap());
        // println!("activation1.dinputs: \n{:?}", self.activation1.dinputs.as_ref().unwrap());
        self.dense1
            .backward(self.activation1.dinputs.as_ref().unwrap());
        // println!("dense1.dinputs: \n{:?}", self.dense1.dinputs.as_ref().unwrap());
    }
}
