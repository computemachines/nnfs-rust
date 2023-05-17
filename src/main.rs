use nnfs::{
    analysis_functions, loss_functions::SoftmaxLossCategoricalCrossEntropy, neurons::LayerDense,
    optimizer::OptimizerSDG, data::{spiral_data, visualize_nn_scatter},
};
use nnfs_rust as nnfs;

use ndarray::prelude::*;
use plotters::prelude::*;

fn main() {
    nnfs::ch10::run();
}