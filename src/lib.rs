#![allow(dead_code, unused)]

pub mod ch2;
pub mod ch3;
pub mod ch4;
pub mod ch5;
pub mod ch6;

pub mod data;
pub mod neurons;
pub mod activation_functions;
pub mod loss_functions;
pub mod analysis_functions;

use ndarray::prelude::*;
pub fn show_data() {
    // nnfs::ch3::run();
    let data = crate::data::spiral_data(50, 5);

    for row in (&data.0).axis_iter(Axis(0)) {
        println!("{:?}", row);
    }

    crate::data::plot_scatter(&data.0, &data.1, "spiral.png");
}

#[cfg(test)]
mod tests {
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

        let d = array![[1.],[2.]];
        assert_eq!(d.shape(), &[2, 1]);
        let e = a / d;
        assert_eq!(e, array![[1., 2., 3.], [2., 2.5, 3.]]);
    }
}
