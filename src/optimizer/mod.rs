mod ada_grad;
mod adam;
mod rms_prop;
mod sdg;

pub use ada_grad::{OptimizerAdaGrad, OptimizerAdaGradConfig};
pub use adam::{OptimizerAdam, OptimizerAdamConfig};
pub use rms_prop::{OptimizerRMSProp, OptimizerRMSPropConfig};
pub use sdg::{OptimizerSDG, OptimizerSDGConfig};

use crate::neurons::LayerDense;

pub trait Optimizer {
    /// Call once before any parameter updates
    fn pre_update_params(&mut self);

    /// Call once after all parameters update
    fn post_update_params(&mut self);

    /// Call once per layer per epoch
    fn update_params(&self, layer: &mut LayerDense);

    fn current_learning_rate(&self) -> f64;
}
