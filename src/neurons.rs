use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub struct LayerDense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub output: Option<Array2<f64>>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let weights = Array2::random((n_inputs, n_neurons), Normal::new(0., 0.01).unwrap());
        let biases = Array1::zeros(n_neurons);
        Self {
            weights,
            biases,
            output: None,
        }
    }
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.output = Some(inputs.dot(&self.weights) + &self.biases);
    }
}
