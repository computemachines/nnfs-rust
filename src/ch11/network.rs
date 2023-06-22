
use ndarray::prelude::*;

use crate::{neurons::LayerDense, activation_functions::ReLU, loss_functions::SoftmaxLossCategoricalCrossentropy, model::Layer};

#[derive(Clone)]
pub struct Network {
    pub dense1: LayerDense,
    pub dense2: LayerDense,
    pub activation1: ReLU,
    pub loss_activation: SoftmaxLossCategoricalCrossentropy,
}
pub struct NetworkOutput(pub f64, pub Array2<f64>);

impl Network {
    pub fn new(num_labels: usize, num_neurons_layer: usize) -> Self {
        // create dense layer with 2 input features and 64 output values
        let mut dense1 = LayerDense::new(2, num_neurons_layer);

        // create ReLU activation (to be used with dense layer)
        let mut activation1 = ReLU::new();

        // create second dense layer with 64 input features (as we take from lower
        // layer) and 3 output values (output labels)
        let mut dense2 = LayerDense::new(num_neurons_layer, num_labels);

        // create softmax classifier with combined loss and activation
        let mut loss_activation = SoftmaxLossCategoricalCrossentropy::new();

        Self {
            dense1,
            activation1,
            dense2,
            loss_activation,
        }
    }
    pub fn forward(&mut self, input: &Array2<f64>, y_true: &Array1<usize>) -> NetworkOutput {
        self.dense1.forward(input);
        self.activation1
            .forward(self.dense1.output.as_ref().unwrap());
        self.dense2
            .forward(self.activation1.output.as_ref().unwrap());
        let loss = self
            .loss_activation
            .forward_labels(self.dense2.output.as_ref().unwrap(), &y_true);
        let prediction = self.loss_activation.output.take().unwrap();
        NetworkOutput(loss, prediction)
    }
    pub fn backward(&mut self, prediction: &Array2<f64>, y_true: &Array1<usize>) {
        self.loss_activation.backward_labels(prediction, y_true);
        self.dense2
            .backward(self.loss_activation.dinputs.as_ref().unwrap());
        self.activation1
            .backward(self.dense2.dinputs.as_ref().unwrap());
        self.dense1
            .backward(self.activation1.dinputs.as_ref().unwrap());
    }
}
