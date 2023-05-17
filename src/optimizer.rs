use ndarray::{prelude::*, Zip};

use crate::neurons::LayerDense;

pub struct OptimizerSDG {
    learning_rate: f64,
    pub current_learning_rate: f64,
    decay_rate: f64,
    iterations: usize,
}

pub struct OptimizerSDGConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
}

impl OptimizerSDG {
    pub fn new(learning_rate: f64, decay_rate: f64) -> Self {
        Self::from(OptimizerSDGConfig {
            learning_rate,
            decay_rate,
        })
    }
    // This is nice because I can add more params without breaking the api
    pub fn from(optimizer: OptimizerSDGConfig) -> Self {
        Self {
            learning_rate: optimizer.learning_rate,
            current_learning_rate: optimizer.learning_rate,
            decay_rate: optimizer.decay_rate,
            iterations: 0,
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
        // Zip::from(layer.weights);
        azip!((w in &mut layer.weights, &dw in dweights) *w += -self.learning_rate * dw);
        azip!((b in &mut layer.biases, &db in dbias) *b += -self.learning_rate * db);
    }
    /// Call once after any parameter updates
    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

impl Default for OptimizerSDG {
    fn default() -> Self {
        OptimizerSDG::new(1.0, 0.0)
    }
}
impl Default for OptimizerSDGConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            decay_rate: 0.0,
        }
    }
}
