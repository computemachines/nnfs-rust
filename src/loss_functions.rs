use ndarray::{prelude::*, Ix};

pub trait Loss<T> {
    fn calculate(&self, output: &Array2<f64>, y: &T) -> f64 {
        // calculate sample losses
        let sample_losses: Array1<f64> = self.forward(output, y);

        // calculate mean loss
        let data_loss = sample_losses.sum_axis(Axis(0)).sum() / sample_losses.len() as f64;

        // return losses
        data_loss
    }

    fn forward(&self, y_pred: &Array2<f64>, y_true: &T) -> Array1<f64>;
}


pub struct LossCategoricalCrossentropy {
    pub output: Option<Array2<f64>>,
}

impl LossCategoricalCrossentropy {
    pub fn new() -> Self {
        Self {
            output: None,
        }
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
        let correct_confidences = y_pred_clipped.outer_iter().zip(y_true).map(|(row, &index)| row[index as usize]);

        let negative_log_likelihoods = correct_confidences.map(|x| -x.ln());

        Array::from_iter(negative_log_likelihoods)
    }
}

/// Handle the case where y_true is list of one-hot vectors
impl Loss<Array2<f64>> for LossCategoricalCrossentropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1. - 1e-7));

        // y_true are one-hot vectors
        let correct_confidences = (y_pred_clipped * y_true).sum_axis(Axis(1));

        // Take negative log likelihoods for each sample
        let negative_log_likelihoods = correct_confidences.mapv(|x| -x.ln());
        negative_log_likelihoods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_categorical_crossentropy_onehot() {
        let loss = LossCategoricalCrossentropy::new();

        let y_pred = array![
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
            [0.02, 0.9, 0.08]
        ];
        let y_true = array![
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.]
        ];
        let y_true_indices = array![0, 1, 1];
        

        let loss_val1 = loss.calculate(&y_pred, &y_true);
        assert_eq!(loss_val1, 0.38506088005216804);

        let loss_val2 = loss.calculate(&y_pred, &y_true_indices);
        assert_eq!(loss_val2, 0.38506088005216804);
    }
}