use std::ops::Range;

use colorgrad;
/// The source of data for the NNFS book.
use ndarray::{iter::Lanes, s, Array, Array1, Array2};
use ndarray_rand::rand_distr::num_traits::Zero;
use ndarray_rand::{rand_distr::Normal, RandomExt};
use plotpy::{Curve, Plot};
use plotters::{coord::Shift, prelude::*, style::full_palette::BLUEGREY};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::neurons::LayerDense;

const DATA_URL: &str = "https://nnfs.io/datasets/fashion_mnist_images.zip";
const FILE: &str = "fashion_mnist_images.zip";
const FOLDER: &str = "fashion_mnist_images";

use reqwest::blocking::get;
use std::fs::{self, File};
use std::io::copy;
use std::io::Write;
use std::path::Path;

pub fn download_fashion_mnist() -> Result<(), Box<dyn std::error::Error>> {
    // The path where you want to save the file
    let file_path_str = format!("{}/{}", FOLDER, FILE);
    let file_path = Path::new(&file_path_str);

    // Create the directory if it does not exist
    if let Some(dir) = file_path.parent() {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
    }

    // Check if the file already exists
    if file_path.exists() {
        // The file already exists, so we'll abort the operation
        println!("The file already exists, aborting download.");
        return Ok(());
    }

    // If we've reached here, then the file does not exist and we can download it
    let response = get(DATA_URL)?;

    // Ensure the request was successful
    assert!(response.status().is_success());

    // Create or open the file at the specified path
    let mut dest = File::create(&file_path)?;

    // Copy the data from the response to the file
    let content = response.bytes()?;
    dest.write_all(&content)?;
    println!("Downloaded file to {}", file_path_str);

    // the zip crate can't unzip this so I manually unzip using linux p7zip-full

    Ok(())
}

/// Load the Fashion MNIST data from the unzipped folder. Test or train data can be loaded.
/// Shuffle the data if `shuffle` is true.
pub fn load_fashion_mnist_data<Label: Clone + Zero + From<u8>>(
    test: bool,
    shuffle: bool,
) -> (Array2<f64>, Array1<Label>) {
    let mut vec_images = Vec::new();
    let mut vec_labels = Vec::new();

    let num_images = if test { 1000 } else { 6000 };

    for label in 0..=9 {
        for i in 0..num_images {
            let image_filename = format!(
                "./{}/{}/{}/{:04}.png",
                FOLDER,
                if test { "test" } else { "train" },
                label,
                i
            );
            // println!("Loading {}", image_filename);
            // let file = File::open(image_filename.clone()).unwrap();
            let image = image::open(image_filename);
            let image = image.unwrap().to_luma8();
            // let image = image.resize_exact(28, 28, image::imageops::FilterType::Nearest);
            let image = Array1::from_shape_vec(28 * 28, image.into_vec()).unwrap();
            let image = image.mapv(|x| (x as f64 - 127.5) / 127.5);
            vec_images.push(image);
            vec_labels.push(label);
        }
    }

    let mut images = Array2::zeros((vec_images.len(), 28 * 28));
    let mut labels = Array1::zeros(vec_labels.len());
    let mut indices: Vec<usize> = (0..vec_images.len()).collect();
    if shuffle {
        // let mut rng = rand::thread_rng();
        let mut rng = StdRng::seed_from_u64(10);
        indices.shuffle(&mut rng);
    }

    for (index, &shuffled_index) in indices.iter().enumerate() {
        images
            .slice_mut(s![index, ..])
            .assign(&vec_images[shuffled_index]);
        labels[index] = vec_labels[shuffled_index].into();
    }
    (images, labels)
}

pub fn load_image_from_filename(filename: &str) -> Array2<f64> {
    let image = image::open(filename);
    let image = image.unwrap().to_luma8();
    let image = Array1::from_shape_vec(28 * 28, image.into_vec()).unwrap();
    let image = image.mapv(|x| (x as f64 - 127.5) / 127.5);
    image.into_shape((1, 28*28)).unwrap()
}

pub fn spiral_data(samples_per_class: usize, classes: usize) -> (Array2<f64>, Array1<usize>) {
    let d: usize = 2;
    let mut data = Array::zeros((samples_per_class * classes, d));
    let mut labels = Array::zeros(samples_per_class * classes);

    for class_number in 0..classes {
        let ix = (samples_per_class * class_number)..(samples_per_class * (class_number + 1));
        let r = Array::linspace(0.0, 1.0, samples_per_class);
        let t = Array::linspace(
            (class_number as f64) * 4.0,
            ((class_number + 1) as f64) * 4.0,
            samples_per_class,
        ) + &Array::random(samples_per_class, Normal::new(0.0, 0.2).unwrap());

        data.slice_mut(s![ix.clone(), 0])
            .assign(&(r.clone() * &(&t * 2.5).mapv(f64::sin)));
        data.slice_mut(s![ix.clone(), 1])
            .assign(&(r * &(&t * 2.5).mapv(f64::cos)));

        labels.slice_mut(s![ix]).fill(class_number);
    }

    (data, labels.mapv(|x| x as usize))
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

        let x_values = Array::random(
            n,
            Normal::new(x_center, 0.2 * distance_between_x_centers).unwrap(),
        );
        let y_values = Array::random(n, Normal::new(y_center, 0.2).unwrap());

        data.slice_mut(s![ix.clone(), 0]).assign(&x_values);
        data.slice_mut(s![ix.clone(), 1]).assign(&y_values);
        labels.slice_mut(s![ix]).fill(class);
    }

    (data, labels)
}

pub fn sine_data(samples: usize) -> (Array2<f64>, Array2<f64>) {
    let data = Array::linspace(0.0, 1.0, samples)
        .into_shape((samples, 1))
        .unwrap();
    let y = (&data * 2.0 * std::f64::consts::PI).mapv(|v| v.sin());

    (data, y)
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
            .map(|(point, label)| Circle::new((point[0], point[1]), 3, colors[*label].filled())),
    )
    .unwrap();
}

pub fn lin_map(value: f64, from: Range<f64>, to: Range<f64>) -> f64 {
    to.start + (value - from.start) * (to.end - to.start) / (from.end - from.start)
}

pub fn new_root_area(filename: &str, is_gif: bool) -> DrawingArea<BitMapBackend, Shift> {
    if is_gif {
        BitMapBackend::gif(filename, (300, 200), 100)
            .unwrap()
            .into_drawing_area()
    } else {
        BitMapBackend::new(filename, (600, 400)).into_drawing_area()
    }
}

// TODO: fix up this api
pub fn visualize_nn_scatter<'a>(
    data: &Array2<f64>,
    labels: &Array1<usize>,
    max_label: usize,
    mut forward: impl FnMut((f64, f64)) -> RGBAColor,
    root_area: &DrawingArea<BitMapBackend<'a>, Shift>,
) {
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
    // println!("min: {:?}, max: {:?}", min, max);

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        // .caption("Scatter Plot", ("sans-serif", 40))
        .build_cartesian_2d(min[0]..max[0], min[1]..max[1])
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // let colors = vec![
    //     RGBColor(200, 0, 0),
    //     RGBColor(0, 200, 0),
    //     RGBColor(0, 0, 200),
    //     YELLOW,
    //     CYAN,
    //     MAGENTA,
    //     BLACK,
    // ];
    let g = colorgrad::rainbow();
    let colors: Vec<RGBAColor> = (0..max_label)
        .map(|i| {
            let c = g.at(i as f64 / max_label as f64).to_rgba8();
            RGBAColor(c[0], c[1], c[2], c[3] as f64 / 256.0)
        })
        .collect();

    let area = ctx.plotting_area();
    let pixel_range = area.get_pixel_range();
    let pixel_x_range = pixel_range.0.start as f64..pixel_range.0.end as f64;
    let pixel_y_range = pixel_range.1.start as f64..pixel_range.1.end as f64;
    let x_range = area.get_x_range();
    let x_delta = x_range.end - x_range.start;
    let y_range = area.get_y_range();
    let y_delta = y_range.start - y_range.end;
    for x in pixel_range
        .0
        .map(|i| lin_map(i as f64, pixel_x_range.clone(), x_range.clone()))
    {
        for y in pixel_range
            .1
            .clone()
            .map(|j| lin_map(j as f64, pixel_y_range.clone(), y_range.clone()))
        {
            area.draw_pixel((x, y), &forward((x, y)));
        }
    }

    ctx.draw_series(
        data.outer_iter()
            .zip(labels.into_iter())
            .map(|(point, label)| Circle::new((point[0], point[1]), 3, colors[*label].filled())),
    )
    .unwrap();
    root_area.present().unwrap();
}

pub fn plot_regression_data(x: &[f64], y_pred: &[f64], y_true: &[f64], filename: &str) {
    let mut target = Curve::new();
    target.draw(&x, &y_true);
    target.set_line_style("--");
    let mut prediction = Curve::new();
    prediction.draw(&x, &y_pred);
    let mut plot = Plot::new();
    plot.add(&prediction);
    plot.add(&target);
    plot.save(filename).unwrap();
}
