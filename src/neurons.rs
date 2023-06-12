use ndarray::{prelude::*, Zip};
use ndarray_rand::{
    rand_distr::{Binomial, Normal},
    RandomExt,
};
use rand::prelude::Distribution;

#[derive(Default, Clone, Debug)]
pub struct LayerDense {
    pub n_inputs: usize,
    pub n_neurons: usize,

    pub weights: Array2<f64>,
    pub biases: Array1<f64>,

    // Chapter 14
    pub weight_regularizer_l1: f64,
    pub weight_regularizer_l2: f64,
    pub bias_regularizer_l1: f64,
    pub bias_regularizer_l2: f64,

    // following values: Option<_> is used for uninitialized value
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dweights: Option<Array2<f64>>,
    pub dbiases: Option<Array1<f64>>,
    pub dinputs: Option<Array2<f64>>,

    // remaining values: Option<_> is used to avoid allocating optional
    // features. These should be initialized to zeros when needed.
    pub weight_momentums: Option<Array2<f64>>,
    pub bias_momentums: Option<Array1<f64>>,
    pub weight_cache: Option<Array2<f64>>,
    pub bias_cache: Option<Array1<f64>>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let weights = Array2::random((n_inputs, n_neurons), Normal::new(0., 0.01).unwrap());
        // let weights = Array2::ones((n_inputs, n_neurons));
        let biases = Array1::zeros(n_neurons);
        Self {
            n_inputs,
            n_neurons,
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

        // the borrow checker :(
        let weight_regularizer_l1 = self.weight_regularizer_l1;
        let weight_regularizer_l2 = self.weight_regularizer_l2;
        let bias_regularizer_l1 = self.bias_regularizer_l1;
        let bias_regularizer_l2 = self.bias_regularizer_l2;

        // gradients on regularization
        if self.weight_regularizer_l1 > 0. {
            azip!((
                mut dw in self.dweights.as_mut().unwrap(),
                &w in &self.weights) {
                    *dw += if w > 0. { 1. } else { -1. } * weight_regularizer_l1;
            });
        }
        if self.weight_regularizer_l2 > 0. {
            azip!((
                mut dw in self.dweights.as_mut().unwrap(),
                &w in &self.weights) {
                    *dw += w * weight_regularizer_l2 * 2.;
            });
        }
        if self.bias_regularizer_l1 > 0. {
            azip!((
                mut db in self.dbiases.as_mut().unwrap(),
                &b in &self.biases) {
                    *db += if b > 0. { 1. } else { -1. } * bias_regularizer_l1;
            });
        }
        if self.bias_regularizer_l2 > 0. {
            azip!((
                mut db in self.dbiases.as_mut().unwrap(),
                &b in &self.biases) {
                    *db += b * bias_regularizer_l2 * 2.;
            });
        }

        // gradients on values
        self.dinputs = Some(dvalues.dot(&self.weights.t()));
    }
}

#[derive(Default, Clone, Debug)]
pub struct LayerDropout {
    pub rate: f64,
    pub binary_mask: Option<Array2<f64>>,
    pub output: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl LayerDropout {
    pub fn new(rate: f64) -> Self {
        Self {
            rate: 1.0 - rate,
            ..Default::default()
        }
    }
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        // the nnfs book wants to save inputs, but doesn't use it anywhere
        // self.inputs = Some(inputs.clone());

        // generate and save scaled binary mask
        self.binary_mask = Some(Array2::random(
            inputs.raw_dim(),
            Binomial::new(1, self.rate)
                .unwrap()
                .map(|x| x as f64 / self.rate),
        ));
        // apply mask to output values
        self.output = Some(inputs * self.binary_mask.as_ref().unwrap());
    }
    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        self.dinputs = Some(dvalues * self.binary_mask.as_ref().unwrap());
    }
}
