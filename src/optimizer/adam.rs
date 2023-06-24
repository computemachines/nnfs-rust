use ndarray::prelude::*;

use crate::{neurons::LayerDense, util};

use super::Optimizer;

/// RMS Propagation Optimizer
pub struct OptimizerAdam {
    learning_rate: f64,
    pub current_learning_rate: f64,
    decay_rate: f64,
    iterations: usize,
    epsilon: f64,
    // rho: f64,
    beta_1: f64,
    beta_2: f64,
}

#[derive(Clone, Copy)]
pub struct OptimizerAdamConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub epsilon: f64,
    // pub rho: f64
    pub beta_1: f64,
    pub beta_2: f64,
}

impl OptimizerAdam {
    // This is nice because I can add more params without breaking the api
    pub fn from(optimizer: OptimizerAdamConfig) -> Self {
        Self {
            learning_rate: optimizer.learning_rate,
            current_learning_rate: optimizer.learning_rate,
            decay_rate: optimizer.decay_rate,
            iterations: 0,
            epsilon: optimizer.epsilon,
            // rho: optimizer.rho,
            beta_1: optimizer.beta_1,
            beta_2: optimizer.beta_2,
        }
    }
}
impl Optimizer for OptimizerAdam {
    /// Call once before any parameter updates
    fn pre_update_params(&mut self) {
        if self.decay_rate != 0.0 {
            self.current_learning_rate =
                self.learning_rate / (1.0 + self.decay_rate * self.iterations as f64)
        }
    }
    fn update_params(&self, layer: &mut LayerDense) {
        let momentum_correction = 1.0 / (1.0 - self.beta_1.powi(self.iterations as i32 + 1));
        let cache_correction = 1.0 / (1.0 - self.beta_2.powi(self.iterations as i32 + 1));
        azip!((mut weight in &mut layer.weights,
               // ...cache_i represents the singular form of the plural 'cache'
               mut weight_cache_i in layer.weight_cache.get_or_insert_with(|| Array2::zeros((layer.n_inputs, layer.n_neurons))),
               mut weight_momentum in layer.weight_momentums.get_or_insert_with(|| Array2::zeros((layer.n_inputs, layer.n_neurons))),
               &dw in layer.dweights.as_ref().unwrap()) {
            // weight_momentums are a moving average of the gradients
            *weight_momentum = util::weighted_average(self.beta_1, *weight_momentum, dw);
            let weight_momentum_corrected = *weight_momentum * momentum_correction;

            // weigh_cache is a moving average of the squared gradients
            *weight_cache_i = util::weighted_average(self.beta_2, *weight_cache_i, dw * dw);
            let weight_cache_i_corrected = *weight_cache_i * cache_correction;

            *weight +=  -self.current_learning_rate * weight_momentum_corrected / (weight_cache_i_corrected.sqrt() + self.epsilon);
        });

        azip!((mut bias in &mut layer.biases,
               mut bias_cache_i in layer.bias_cache.get_or_insert_with(|| Array1::zeros(layer.n_neurons)),
               mut bias_momentum in layer.bias_momentums.get_or_insert_with(|| Array1::zeros(layer.n_neurons)),
               &db in layer.dbiases.as_ref().unwrap()) {
            // bias_momentums are a moving average of the bias gradients
            *bias_momentum = util::weighted_average(self.beta_1, *bias_momentum, db);
            let bias_momentum_corrected = *bias_momentum * momentum_correction;

            // bias_cache is a moving average of the squared gradients
            *bias_cache_i = util::weighted_average(self.beta_2, *bias_cache_i, db * db);
            let bias_cache_i_corrected = *bias_cache_i * cache_correction;

            *bias += -self.current_learning_rate * bias_momentum_corrected / (bias_cache_i_corrected.sqrt() + self.epsilon);
        });
    }
    /// Call once after any parameter updates
    fn post_update_params(&mut self) {
        self.iterations += 1;
    }

    fn current_learning_rate(&self) -> f64 {
        self.current_learning_rate
    }
}

impl Default for OptimizerAdam {
    fn default() -> Self {
        OptimizerAdam::from(OptimizerAdamConfig::default())
    }
}
impl Default for OptimizerAdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            decay_rate: 0.0,
            epsilon: 1e-7,
            beta_1: 0.9,
            beta_2: 0.999,
        }
    }
}
