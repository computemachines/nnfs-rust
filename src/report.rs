use ndarray::prelude::*;

pub const FASHION_MNIST_LABELS: [&str; 10] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

/// List categorical inference in descending order with their probabilities and text labels.
pub fn list_categorical_inference(
    prediction: ArrayView1<f64>,
    labels: &[&str],
) {
    let mut prediction = prediction.to_vec();
    let mut labels = labels.to_vec();
    let mut sorted: Vec<(f64, &str)> = prediction.into_iter().zip(labels.into_iter()).collect();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for (prob, label) in sorted {
        println!("{}: {:.2}%", label, prob * 100.);
    }
}