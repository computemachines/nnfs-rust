use ndarray::prelude::*;

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