use ndarray::{prelude::*, Zip};

pub fn get_accuracy(predictions: &Array2<f64>, labels: &Array1<usize>) -> f64 {
    let mut correct_predictions = 0;
    for (prediction, label) in predictions.outer_iter().zip(labels) {
        let prediction = prediction.indexed_iter().max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap()).unwrap().0;
        if prediction == *label {
            correct_predictions += 1;
        }
    }
    correct_predictions as f64 / predictions.nrows() as f64
}

pub fn get_accuracy_binary(predictions: &Array2<f64>, tags_true: &Array2<f64>) -> f64 {
    let sum = Zip::from(predictions).and(tags_true).fold(0., |acc, &p, &t| {
        let prediction = if p > 0.5 { 1. } else { 0. };
        if prediction == t {
            acc + 1.0
        } else {
            acc
        }
    });
    // len() is the total number of elements in the array. This is different from python.
    sum / predictions.len() as f64
}