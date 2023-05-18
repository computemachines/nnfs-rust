use ndarray::{prelude::*, Zip};

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
        azip!((weight in &mut layer.weights,
               weight_momentum in &mut layer.weight_momentums,
               &dw in dweights) {
            let weight_update = self.momentum * *weight_momentum - self.current_learning_rate * dw;
            *weight += weight_update;
            *weight_momentum = weight_update;
        });
        azip!((bias in &mut layer.biases,
               bias_momentum in &mut layer.bias_momentums,
               &db in dbias) {
            let bias_update = self.momentum * *bias_momentum - self.current_learning_rate * db;
            *bias += -self.current_learning_rate * db;
            *bias_momentum = bias_update;
        });
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
