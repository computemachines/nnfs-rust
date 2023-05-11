/// The source of data for the NNFS book.
use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use plotters::{prelude::*, style::full_palette::BLUEGREY};

pub fn spiral_data(n: usize, k: usize) -> (Array2<f64>, Array1<usize>) {
    let d: usize = 2;
    let mut data = Array::zeros((n * k, d));
    let mut labels = Array::zeros(n * k);
    for class in 0..k {
        // indexes chunked by the current class
        let ix = (class * n)..((class + 1) * n);

        let radius: Array1<f64> = Array::linspace(0.0, 1.0, n);
        let theta: Array1<f64> = Array::linspace(class as f64 * 4.0, (class + 1) as f64 * 4.0, n);
        let jitter: Array1<f64> = Array::random(n, Normal::new(0.0, 0.2).unwrap());

        let theta = theta + jitter;

        data.slice_mut(s![ix.clone(), 0])
            .assign(&(&radius * &theta.mapv(f64::cos)));
        data.slice_mut(s![ix.clone(), 1])
            .assign(&(&radius * &theta.mapv(f64::sin)));

        labels.slice_mut(s![ix]).fill(class);
    }
    (data, labels)
}

pub fn vertical_data(n: usize, k: usize) -> (Array2<f64>, Array1<usize>) {
    // n: number of points per class
    // k: number of classes

    let mut data = Array::zeros((n * k, 2));
    let mut labels = Array::zeros(n * k);

    for class in 0..k {
        // indexes chunked by the current class
        let ix = (class * n)..((class + 1) * n);

        let x_center = class as f64 / k as f64;
        let distance_between_x_centers = if k == 1 { 1.0 } else { 1.0 / (k - 1) as f64 };

        let y_center = 0.5;

        let x_values = Array::random(n, Normal::new(x_center, 0.2*distance_between_x_centers).unwrap());
        let y_values = Array::random(n, Normal::new(y_center, 0.2).unwrap());

        data.slice_mut(s![ix.clone(), 0]).assign(&x_values);
        data.slice_mut(s![ix.clone(), 1]).assign(&y_values);
        labels.slice_mut(s![ix]).fill(class);
    }

    (data, labels)
}

pub fn plot_scatter(data: &Array2<f64>, labels: &Array1<usize>, filename: &str) {
    let root_area = BitMapBackend::new(filename, (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut min = [data[[0, 0]], data[[0, 1]]];
    let mut max = [data[[0, 0]], data[[0, 1]]];
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[1] {
            if data[[i, j]] < min[j] {
                min[j] = data[[i, j]];
            }
            if data[[i, j]] > max[j] {
                max[j] = data[[i, j]];
            }
        }
    }
    println!("min: {:?}, max: {:?}", min, max);

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        // .caption("Scatter Plot", ("sans-serif", 40))
        .build_cartesian_2d(min[0]..max[0], min[1]..max[1])
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let colors = vec![BLUEGREY, RED, GREEN, YELLOW, CYAN, MAGENTA, BLACK];

    ctx.draw_series(
        data.outer_iter()
            .zip(labels.into_iter())
            .map(|(point, label)| {
                Circle::new((point[0], point[1]), 3, colors[*label].filled())
            }),
    )
    .unwrap();
}
