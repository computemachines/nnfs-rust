use ndarray::{prelude::*, Zip};

pub trait Accuracy<T> {
    fn calculate(&self, predictions: &T, labels: &T) -> f64;
    fn init(&mut self, outputs_true: &T) {}
}

#[derive(Default)]
pub struct AccuracyRegression {
    precision: f64,
}
impl AccuracyRegression {
    pub fn new(precision: f64) -> Self {
        Self { precision }
    }
    pub fn init(&mut self, outputs_true: &Array2<f64>) {
        self.precision = outputs_true.std(0.) / 250.;
    }
}
impl Accuracy<Array2<f64>> for AccuracyRegression {
    fn calculate(&self, predictions: &Array2<f64>, outputs_true: &Array2<f64>) -> f64 {
        Zip::from(predictions)
            .and(outputs_true)
            .fold(0., |acc, &p, &t| {
                if (p - t).abs() < self.precision {
                    acc + 1.0
                } else {
                    acc
                }
            })
            / predictions.len() as f64
    }
}

pub struct AccuracyCategoricalLabels;
impl Accuracy<Array1<usize>> for AccuracyCategoricalLabels {
    fn calculate(&self, predictions: &Array1<usize>, labels: &Array1<usize>) -> f64 {
        Zip::from(predictions)
            .and(labels)
            .fold(0., |acc, &p, &t| if p == t { acc + 1.0 } else { acc })
            / predictions.len() as f64
    }
}

pub fn get_accuracy(predictions: &Array2<f64>, labels: &Array1<usize>) -> f64 {
    let mut correct_predictions = 0;
    for (prediction, label) in predictions.outer_iter().zip(labels) {
        let prediction = prediction
            .indexed_iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .unwrap()
            .0;
        if prediction == *label {
            correct_predictions += 1;
        }
    }
    correct_predictions as f64 / predictions.nrows() as f64
}

pub struct AccuracyBinary;
impl Accuracy<Array2<f64>> for AccuracyBinary {
    fn calculate(&self, predictions: &Array2<f64>, labels: &Array2<f64>) -> f64 {
        Zip::from(predictions)
            .and(labels)
            .fold(0., |acc, &p, &t| {
                let prediction = if p > 0.5 { 1. } else { 0. };
                if prediction == t {
                    acc + 1.0
                } else {
                    acc
                }
            })
            / predictions.len() as f64
    }
}

pub fn get_accuracy_binary(predictions: &Array2<f64>, tags_true: &Array2<f64>) -> f64 {
    let sum = Zip::from(predictions)
        .and(tags_true)
        .fold(0., |acc, &p, &t| {
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
