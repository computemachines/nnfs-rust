use ndarray::prelude::*;

pub struct ReLU {
    pub output: Option<Array2<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.output = Some(inputs.map(|x| x.max(0.)));
    }
}

pub struct Softmax {
    pub output: Option<Array2<f64>>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        let exp_values: Array2<_> = inputs.map(|x| x.exp());
        let norm_per_input = exp_values.sum_axis(Axis(1));
        let norm_per_input = norm_per_input.to_shape([norm_per_input.len(), 1]).unwrap();
        self.output = Some(exp_values / norm_per_input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_abs_diff_eq() {
        let a = array![1., 2., 3.];
        let b = array![1., 2., 3.];
        assert_abs_diff_eq!(a, b);
    }

    #[test]
    fn test_softmax() {
        let mut softmax = Softmax::new();
        let inputs = array![[1., 2., 3.], [4., 5., 6.]];
        softmax.forward(&inputs);

        assert_abs_diff_eq!(
            softmax.output.as_ref().unwrap(),
            &array![
                [0.09003057, 0.24472847, 0.66524096],
                [0.09003057, 0.24472847, 0.66524096]
            ],
            epsilon = 1e-8
        );
    }
}