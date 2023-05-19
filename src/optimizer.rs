use ndarray::{prelude::*, Zip};
use ndarray_rand::rand_distr::num_traits::Pow;

use crate::neurons::LayerDense;

pub struct OptimizerSDG {
    learning_rate: f64,
    pub current_learning_rate: f64,
    decay_rate: f64,
    iterations: usize,
    momentum: f64,
}

#[derive(Clone, Copy)]
pub struct OptimizerSDGConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub momentum: f64,
}

impl OptimizerSDG {
    // This is nice because I can add more params without breaking the api
    pub fn from(optimizer: OptimizerSDGConfig) -> Self {
        Self {
            learning_rate: optimizer.learning_rate,
            current_learning_rate: optimizer.learning_rate,
            decay_rate: optimizer.decay_rate,
            iterations: 0,
            momentum: optimizer.momentum,
        }
    }
    /// Call once before any parameter updates
    pub fn pre_update_params(&mut self) {
        if self.decay_rate != 0.0 {
            self.current_learning_rate =
                self.learning_rate / (1.0 + self.decay_rate * self.iterations as f64)
        }
    }
    pub fn update_params(&self, layer: &mut LayerDense) {
        let dweights = layer.dweights.as_ref().unwrap();
        let dbias = layer.dbiases.as_ref().unwrap();
        if self.momentum != 0.0 {
            azip!((weight in &mut layer.weights,
                   weight_momentum in layer.weight_momentums.get_or_insert_with(|| Array2::zeros((layer.n_inputs, layer.n_neurons))),
                   &dw in dweights) {
                let weight_update = self.momentum * *weight_momentum - self.current_learning_rate * dw;
                *weight += weight_update;
                *weight_momentum = weight_update;
            });
            azip!((bias in &mut layer.biases,
                   bias_momentum in layer.bias_momentums.get_or_insert_with(|| Array1::zeros(layer.n_neurons)),
                   &db in dbias) {
                let bias_update = self.momentum * *bias_momentum - self.current_learning_rate * db;
                *bias += -self.current_learning_rate * db;
                *bias_momentum = bias_update;
            });
        }
    }
    /// Call once after any parameter updates
    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

impl Default for OptimizerSDG {
    fn default() -> Self {
        OptimizerSDG::from(OptimizerSDGConfig::default())
    }
}
impl Default for OptimizerSDGConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            decay_rate: 0.0,
            momentum: 0.0,
        }
    }
}

/// Adaptive Gradient Optimizer
pub struct OptimizerAdaGrad {
    learning_rate: f64,
    pub current_learning_rate: f64,
    decay_rate: f64,
    iterations: usize,
    epsilon: f64,
}

#[derive(Clone, Copy)]
pub struct OptimizerAdaGradConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub epsilon: f64,
}

impl OptimizerAdaGrad {
    // This is nice because I can add more params without breaking the api
    pub fn from(optimizer: OptimizerAdaGradConfig) -> Self {
        Self {
            learning_rate: optimizer.learning_rate,
            current_learning_rate: optimizer.learning_rate,
            decay_rate: optimizer.decay_rate,
            iterations: 0,
            epsilon: optimizer.epsilon,
        }
    }
    /// Call once before any parameter updates
    pub fn pre_update_params(&mut self) {
        if self.decay_rate != 0.0 {
            self.current_learning_rate =
                self.learning_rate / (1.0 + self.decay_rate * self.iterations as f64)
        }
    }
    pub fn update_params(&self, layer: &mut LayerDense) {
        azip!((mut weight in &mut layer.weights,
               mut weight_cache_i in layer.weight_cache.get_or_insert_with(|| Array2::zeros((layer.n_inputs, layer.n_neurons))),
               &dw in layer.dweights.as_ref().unwrap()) {
            *weight_cache_i += dw * dw;
            *weight +=  - self.current_learning_rate * dw / (weight_cache_i.sqrt() + self.epsilon);
        });

        azip!((mut bias in &mut layer.biases,
               mut bias_cache_i in layer.bias_cache.get_or_insert_with(|| Array1::zeros(layer.n_neurons)),
               &db in layer.dbiases.as_ref().unwrap()) {
            *bias_cache_i += db * db;
            *bias += -self.current_learning_rate * db / (bias_cache_i.sqrt() + self.epsilon);
        });
    }
    /// Call once after any parameter updates
    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

impl Default for OptimizerAdaGrad {
    fn default() -> Self {
        OptimizerAdaGrad::from(OptimizerAdaGradConfig::default())
    }
}
impl Default for OptimizerAdaGradConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            decay_rate: 0.0,
            epsilon: 1e-7,
        }
    }
}
