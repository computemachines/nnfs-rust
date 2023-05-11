use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

#[derive(Default)]
pub struct LayerDense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dweights: Option<Array2<f64>>,
    pub dbiases: Option<Array1<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let weights = Array2::random((n_inputs, n_neurons), Normal::new(0., 0.01).unwrap());
        let biases = Array1::zeros(n_neurons);
        Self {
            weights,
            biases,
            ..Default::default()
        }
    }
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.output = Some(inputs.dot(&self.weights) + &self.biases);
        self.inputs = Some(inputs.clone());
    }
    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        // gradients on parameters
        self.dweights = Some(self.inputs.as_ref().unwrap().t().dot(dvalues));
        self.dbiases = Some(dvalues.sum_axis(Axis(0)));

        // gradients on values
        self.dinputs = Some(dvalues.dot(&self.weights.t()));
    }
}
