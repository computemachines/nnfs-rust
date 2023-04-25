use nnfs_rust as nnfs;

use ndarray::prelude::*;

fn show_data() {
    // nnfs::ch3::run();
    let data = nnfs::data::spiral_data(50, 5);

    for row in (&data.0).axis_iter(Axis(0)) {
        println!("{:?}", row);
    }

    nnfs::data::plot_scatter(&data.0, &data.1, "spiral.png");
}

fn main() {
    nnfs::ch3::run2();
}