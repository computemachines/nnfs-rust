use ndarray::prelude::*;

use crate::neurons::LayerDense;

use super::Optimizer;

/// RMS Propagation Optimizer
pub struct OptimizerRMSProp {
    learning_rate: f64,
    pub current_learning_rate: f64,
    decay_rate: f64,
    iterations: usize,
    epsilon: f64,
    rho: f64,
}

#[derive(Clone, Copy)]
pub struct OptimizerRMSPropConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub epsilon: f64,
    pub rho: f64,
}

impl OptimizerRMSProp {
    // This is nice because I can add more params without breaking the api
    pub fn from(optimizer: OptimizerRMSPropConfig) -> Self {
        Self {
            learning_rate: optimizer.learning_rate,
            current_learning_rate: optimizer.learning_rate,
            decay_rate: optimizer.decay_rate,
            iterations: 0,
            epsilon: optimizer.epsilon,
            rho: optimizer.rho,
        }
    }
}
impl Optimizer for OptimizerRMSProp {
    /// Call once before any parameter updates
    fn pre_update_params(&mut self) {
        if self.decay_rate != 0.0 {
            self.current_learning_rate =
                self.learning_rate / (1.0 + self.decay_rate * self.iterations as f64)
        }
    }
    fn update_params(&self, layer: &mut LayerDense) {
        azip!((mut weight in &mut layer.weights,
               mut weight_cache_i in layer.weight_cache.get_or_insert_with(|| Array2::zeros((layer.n_inputs, layer.n_neurons))),
               &dw in layer.dweights.as_ref().unwrap()) {
            *weight_cache_i = (self.rho * *weight_cache_i) + (1.0 - self.rho) * dw * dw;
            *weight +=  - self.current_learning_rate * dw / (weight_cache_i.sqrt() + self.epsilon);
        });

        azip!((mut bias in &mut layer.biases,
               mut bias_cache_i in layer.bias_cache.get_or_insert_with(|| Array1::zeros(layer.n_neurons)),
               &db in layer.dbiases.as_ref().unwrap()) {
            *bias_cache_i = (self.rho * *bias_cache_i) + (1.0 - self.rho) * db * db;
            *bias += -self.current_learning_rate * db / (bias_cache_i.sqrt() + self.epsilon);
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

impl Default for OptimizerRMSProp {
    fn default() -> Self {
        OptimizerRMSProp::from(OptimizerRMSPropConfig::default())
    }
}
impl Default for OptimizerRMSPropConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            decay_rate: 0.0,
            epsilon: 1e-7,
            rho: 1e-3,
        }
    }
}
