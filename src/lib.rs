#![allow(dead_code, unused)]

pub mod ch2;
pub mod ch3;
pub mod ch4;
pub mod ch5;
pub mod ch6;
pub mod ch9;
pub mod ch10;
pub mod ch11;
pub mod ch14;

pub mod activation_functions;
pub mod analysis_functions;
pub mod data;
pub mod loss_functions;
pub mod neurons;
pub mod optimizer;
pub mod util;

#[macro_use]
use approx::assert_abs_diff_eq;

use ndarray::prelude::*;
pub fn show_data() {
    // nnfs::ch3::run();
    let data = crate::data::spiral_data(50, 5);

    for row in (data.0).axis_iter(Axis(0)) {
        println!("{:?}", row);
    }

    crate::data::plot_scatter(&data.0, &data.1, "spiral.png");
}

#[cfg(test)]
mod tests {
    use crate::{
        activation_functions::ReLU, loss_functions::SoftmaxLossCategoricalCrossEntropy,
        neurons::LayerDense,
    };

    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn ch2_run() {
        ch2::run();
    }

    #[test]
    fn ch3_run() {
        ch3::run1();
        ch3::run2();
    }

    #[test]
    fn ch4_run() {
        ch4::run();
    }

    #[test]
    fn test_ndarray_division_broadcasting() {
        let a = array![[1., 2., 3.], [4., 5., 6.]];
        // check shape
        assert_eq!(a.shape(), &[2, 3]);
        let b = array![1., 2., 3.];
        // check shape
        assert_eq!(b.shape(), &[3]);
        let c = &a / b;
        assert_eq!(c, array![[1., 1., 1.], [4., 2.5, 2.]]);

        let d = array![[1.], [2.]];
        assert_eq!(d.shape(), &[2, 1]);
        let e = a / d;
        assert_eq!(e, array![[1., 2., 3.], [2., 2.5, 3.]]);
    }

    #[test]
    fn test_dense_2x2_forward() {
        let x = array![[-0.5, 3.0], [-0.2, -8.0], [1.2, 2.0],];
        let y = array![0, 0, 1];

        let n = 100000;
        let mut losses = Array1::zeros(n);

        for idx in 0..n {
            let mut dense1 = LayerDense::new(2, 2);
            let mut loss_activation = SoftmaxLossCategoricalCrossEntropy::new();

            // forward
            dense1.forward(&x);
            // dbg!(&dense1.output.as_ref().unwrap());
            let loss = loss_activation.forward_labels(dense1.output.as_ref().unwrap(), &y);
            losses[idx] = loss;
            let predictions = loss_activation.output.take().unwrap();

            let accuracy = analysis_functions::get_accuracy(&predictions, &y);
        }
        println!("{} +- {}", losses.mean().unwrap(), losses.std(1.0));
        assert_abs_diff_eq!(losses.mean().unwrap(), 0.6937, epsilon=0.001);
        assert_abs_diff_eq!(losses.std(1.0), 0.017, epsilon=0.001);
        // assert_abs_diff_eq!(loss, 0.6832295, epsilon = 0.001);
        // assert!(false);
    }

    #[test]
    fn test_dense_2x2_backward() {
        let x = array![[-0.5, 3.0], [-0.2, -8.0], [1.2, 2.0],];
        let y = array![0, 0, 1];

        let n = 100000;
        let mut losses: Array1<f64> = Array1::zeros(n);
    }
}
