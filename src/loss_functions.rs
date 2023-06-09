use ndarray::{prelude::*, Zip};
use rand::seq::index::sample;

use crate::{activation_functions::{self}, neurons::LayerDense, model::Layer};

pub trait Loss<T> {
    fn calculate(&self, output: &Array2<f64>, y: &T) -> f64 {
        // calculate sample losses
        let sample_losses: Array1<f64> = self.forward(output, y);

        // calculate mean loss

        // return losses
        sample_losses.mean().unwrap()
    }

    fn forward(&self, dvalues: &Array2<f64>, y_true: &T) -> Array1<f64>;
    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &T);
    fn dinputs(&self) -> &Array2<f64>;
}

pub fn regularization_loss(layer: &LayerDense) -> f64 {
    let mut reg_loss: f64 = 0.;

    // L1 regularization - weights
    if layer.weight_regularizer_l1 > 0. {
        reg_loss +=
            (layer.weight_regularizer_l1) * layer.weights.iter().map(|w| w.abs()).sum::<f64>();
    }
    // L2 regularization - weights
    if layer.weight_regularizer_l2 > 0. {
        reg_loss +=
            layer.weight_regularizer_l2 * layer.weights.iter().map(|w| w.powi(2)).sum::<f64>();
    }

    // L1 regularization - biases
    if layer.bias_regularizer_l1 > 0. {
        reg_loss += (layer.bias_regularizer_l1) * layer.biases.iter().map(|b| b.abs()).sum::<f64>();
    }
    // L2 regularization - biases
    if layer.bias_regularizer_l2 > 0. {
        reg_loss += layer.bias_regularizer_l2 * layer.biases.iter().map(|b| b.powi(2)).sum::<f64>();
    }

    reg_loss
}

#[derive(Default, Clone, Debug)]
pub struct LossCategoricalCrossentropy {
    pub dinputs: Option<Array2<f64>>,
}

impl LossCategoricalCrossentropy {
    pub fn new() -> Self {
        Default::default()
    }
}

// Overloading is an ugly hack, but I am trying to follow the book as closely as possible while keeping the code statically typed
/// Handle the case where y_true is a list of cataegorical labels
impl Loss<Array1<usize>> for LossCategoricalCrossentropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array1<usize>) -> Array1<f64> {
        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1. - 1e-7));

        // y_true are catagorical labels, i.e. indices
        //let correct_confidences = y_pred.slice(s![.., y_true]); // Sadly, this does not work
        let correct_confidences = y_pred_clipped
            .outer_iter()
            .zip(y_true)
            .map(|(row, &index)| row[index]);

        let negative_log_likelihoods = correct_confidences.map(|x| -x.ln());

        Array::from_iter(negative_log_likelihoods)
    }

    /// Build one-hot vectors from y_true and call the other backward function
    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array1<usize>) {
        let y_true_onehot = Array::from_shape_fn((y_true.len(), dvalues.shape()[1]), |(i, j)| {
            if j == y_true[i] {
                1.
            } else {
                0.
            }
        });
        self.backward(dvalues, &y_true_onehot);
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
}

/// Handle the case where y_true is list of one-hot vectors
impl Loss<Array2<f64>> for LossCategoricalCrossentropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
        println!("forward");
        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1. - 1e-7));

        // y_true are one-hot vectors
        let correct_confidences = (y_pred_clipped * y_true).sum_axis(Axis(1));

        // Take negative log likelihoods for each sample
        correct_confidences.mapv(|x| -x.ln())
    }
    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // println!("loss.backward dvalues: {:#?}", dvalues);
        // println!("loss.backward y_true: {:#?}", y_true);
        // number of samples
        let samples = dvalues.shape()[0];

        // number of labels in every sample
        let labels = dvalues.shape()[1];

        // calculate normalized gradient
        self.dinputs = Some(-y_true / (dvalues * samples as f64));
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
}

#[derive(Default, Clone, Debug)]
pub struct BinaryCrossentropy {
    pub dinputs: Option<Array2<f64>>,
}

impl BinaryCrossentropy {
    pub fn new() -> Self {
        Default::default()
    }
}

// Overloading is an ugly hack, but I am trying to follow the book as closely as possible while keeping the code statically typed
/// Handle the case where y_true is a list of cataegorical labels
impl Loss<Array2<f64>> for BinaryCrossentropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1. - 1e-7));

        // Unfortunately, ln() does not broadcast so we have to do this manually
        let sample_losses =
            Zip::from(&y_pred_clipped)
                .and(y_true)
                .map_collect(|y_pred_clipped_el, y_true_el| {
                    -(y_true_el * y_pred_clipped_el.ln()
                        + (1. - y_true_el) * (1. - y_pred_clipped_el).ln())
                });
        sample_losses.sum_axis(Axis(1)) / y_pred.shape()[1] as f64
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // Number of samples
        let samples = dvalues.shape()[0];

        // Number of outputs in every sample
        let outputs = dvalues.shape()[1];

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        let clipped_dvalues = dvalues.mapv(|x| x.max(1e-7).min(1. - 1e-7));

        // Calculate gradient
        self.dinputs = Some(
            -(y_true / &clipped_dvalues - (1. - y_true) / (1. - &clipped_dvalues)) / outputs as f64,
        );

        // Normalize gradient
        self.dinputs.as_mut().unwrap().map_inplace(|x| *x /= samples as f64);
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
}

#[derive(Default, Clone, Debug)]
pub struct MeanSquaredError {
    pub dinputs: Option<Array2<f64>>,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Loss<Array2<f64>> for MeanSquaredError {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
        let sample_losses = (y_true - y_pred).mapv(|x| x.powi(2)).sum_axis(Axis(1));
        sample_losses / y_pred.shape()[1] as f64
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // Number of samples
        let samples = dvalues.shape()[0];

        // Number of outputs in every sample
        let outputs = dvalues.shape()[1];

        // Gradient on values
        self.dinputs = Some(-2. / outputs as f64 * (y_true - dvalues));

        // Normalize gradient
        self.dinputs.as_mut().unwrap().map_inplace(|x| *x /= samples as f64);
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
}

#[derive(Default, Clone, Debug)]
pub struct MeanAbsoluteError {
    pub dinputs: Option<Array2<f64>>,
}

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Loss<Array2<f64>> for MeanAbsoluteError {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
        let sample_losses = (y_true - y_pred).mapv(|x| x.abs()).sum_axis(Axis(1));
        sample_losses / y_pred.shape()[1] as f64
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // Number of samples
        let samples = dvalues.shape()[0];

        // Number of outputs in every sample
        let outputs = dvalues.shape()[1];

        // Gradient on values
        self.dinputs = Some(-1. / outputs as f64 * (y_true - dvalues).mapv(|x| x.signum()));

        // Normalize gradient
        self.dinputs.as_mut().unwrap().map_inplace(|x| *x /= samples as f64);
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
}


/// Softmax classifier - combined Softmax activation and cross-entropy loss for
/// faster backward step
#[derive(Clone)]
pub struct SoftmaxLossCategoricalCrossentropy {
    pub activation: activation_functions::Softmax,
    pub loss: LossCategoricalCrossentropy,
    pub output: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

// I gave up on overloading, it is too complicated. I could have used an enum
// then do runtime checking but that would be a small performance hit just to
// make the code look like the book
impl SoftmaxLossCategoricalCrossentropy {
    /// Creates activation and loss function objects
    pub fn new() -> Self {
        Self {
            activation: activation_functions::Softmax::new(),
            loss: LossCategoricalCrossentropy::new(),
            output: None,
            dinputs: None,
        }
    }

    /// Forward pass for one-hot encoded labels
    pub fn forward_onehot(&mut self, inputs: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        // Output layer's activation function
        self.activation.forward(inputs);

        // Set the output
        self.output = self.activation.output.clone();

        // Calculate and return loss value
        self.loss
            .calculate(self.activation.output.as_ref().unwrap(), y_true)
    }

    /// Forward pass for categorical labels
    pub fn forward_labels(&mut self, inputs: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        // Output layer's activation function
        self.activation.forward(inputs);

        // Set the output
        self.output = self.activation.output.clone();

        // Calculate and return loss value
        self.loss
            .calculate(self.activation.output.as_ref().unwrap(), y_true)
    }

    /// Backward pass for one-hot encoded labels
    pub fn backward_onehot(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // Number of samples
        let samples = dvalues.shape()[0] as f64;

        let y_true_labels: Array1<usize> = y_true
            .outer_iter()
            .map(|row| row.iter().position(|&x| x == 1.).unwrap())
            .collect();

        // Gradient on values
        self.dinputs = Some(dvalues.clone());
        // this is so much simpler using python numpy. Rust ndarray is not as flexible
        azip!((mut row in self.dinputs.as_mut().unwrap().outer_iter_mut(), &label in &y_true_labels) {
            row[label] -= 1.;
            row /= samples; // Normalize gradient
        });
    }

    /// Backward pass for categorical labels
    pub fn backward_labels(&mut self, dvalues: &Array2<f64>, y_true: &Array1<usize>) {
        // Number of samples
        let samples = dvalues.shape()[0] as f64;
        // dont have to convert one-hot to labels

        // Gradient on values
        self.dinputs = Some(dvalues.clone());
        // this is so much simpler using python numpy. Rust ndarray is not as flexible
        azip!((mut row in self.dinputs.as_mut().unwrap().outer_iter_mut(), &label in y_true) {
            row[label] -= 1.;
            row /= samples; // Normalize gradient
        });

        // // TESTING ONLY - REMOVE
        // self.dinputs = Some(Array2::ones([dvalues.shape()[0], dvalues.shape()[1]]));
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_loss_categorical_crossentropy_onehot() {
        let loss = LossCategoricalCrossentropy::new();

        let y_pred = array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]];
        let y_true = array![[1., 0., 0.], [0., 1., 0.], [0., 1., 0.]];
        let y_true_indices = array![0, 1, 1];

        let loss_val1 = loss.calculate(&y_pred, &y_true);
        assert_eq!(loss_val1, 0.38506088005216804);

        let loss_val2 = loss.calculate(&y_pred, &y_true_indices);
        assert_eq!(loss_val2, 0.38506088005216804);
    }

    #[test]
    fn test_backpropagation_crossentropy_onehot() {
        let mut loss = LossCategoricalCrossentropy::new();

        let y_pred = array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]];
        let y_true = array![[1., 0., 0.], [0., 1., 0.], [0., 1., 0.]];
        let y_true_indices = array![0, 1, 1];

        let dvalues = array![[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]];

        let loss_val1 = loss.calculate(&y_pred, &y_true);
        loss.backward(&dvalues, &y_true);
        let dinputs_onehot = loss.dinputs.as_ref().unwrap().clone();

        let loss_val2 = loss.calculate(&y_pred, &y_true_indices);
        loss.backward(&dvalues, &y_true_indices);
        let dinputs_indices = loss.dinputs.as_ref().unwrap().clone();

        assert_eq!(loss_val1, loss_val2);
    }

    #[test]
    fn test_bce_backward() {
        let y = array![[1.0]];
        let y_hat = array![[0.6]];

        let mut loss = BinaryCrossentropy::new();
        let loss_val = loss.calculate(&y_hat, &y);
        assert_abs_diff_eq!(loss_val, 0.51, epsilon = 0.01);

        loss.backward(&y_hat, &y);
        let dvalues = loss.dinputs.as_ref().unwrap();
        assert_abs_diff_eq!(dvalues, &array![[-1.67]], epsilon = 0.01);
    }
}
