use ndarray::{prelude::*, Zip};

use crate::model::Layer;

pub trait FinalActivation<T>: Layer {
    fn prediction(&self) -> T;
    fn take_output(&mut self) -> Array2<f64> {
        self.output().clone()
    }
}

#[derive(Default, Clone, Debug)]
pub struct ReLU {
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl Layer for ReLU {
    fn output(&self) -> &Array2<f64> {
        self.output.as_ref().unwrap()
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }

    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64> {
        self.inputs = Some(inputs.clone());
        self.output = Some(inputs.map(|x| x.max(0.)));
        self.output()
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> &Array2<f64> {
        // let mut dvalues = dvalues.clone();
        // let inputs = self.inputs.as_ref().unwrap();
        // dvalues.zip_mut_with(inputs, |dv, i| {
        //     if *i <= 0. {
        //         *dv = 0.;
        //     }
        // });
        let inputs = self.inputs.as_ref().unwrap();
        self.dinputs =
            Some(Zip::from(inputs).and(dvalues).map_collect(
                |i, dv| {
                    if *i <= 0. {
                        0.
                    } else {
                        *dv
                    }
                },
            ));
        self.dinputs()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default, Clone, Debug)]
pub struct Softmax {
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}
impl Layer for Softmax {
    fn output(&self) -> &Array2<f64> {
        self.output.as_ref().unwrap()
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64> {
        self.inputs = Some(inputs.clone());
        let exp_values: Array2<_> = inputs.map(|x| x.exp());
        let norm_per_input = exp_values.sum_axis(Axis(1));
        let norm_per_input = norm_per_input.to_shape([norm_per_input.len(), 1]).unwrap();
        self.output = Some(exp_values / norm_per_input);
        self.output()
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> &Array2<f64> {
        // python: self.dinputs = np.empty_like(dvalues)
        self.dinputs = Some(Array2::zeros(dvalues.raw_dim()));

        // python: for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
        azip!((
            mut dinput_row in self.dinputs.as_mut().unwrap().rows_mut(),
            single_output in self.output.as_ref().unwrap().rows(),
            single_dvalues in dvalues.rows()
        ) {
            // python: single_output = single_output.reshape(-1, 1)
            let single_output = single_output.into_shape((single_output.shape()[0], 1)).unwrap();
            // python: jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            let jacobian =
                Array2::from_diag(&single_output.t().row(0)) - single_output.dot(&single_output.t());
            // python: self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            dinput_row.assign(&jacobian.dot(&single_dvalues));
        });
        self.dinputs()
    }
}
impl FinalActivation<Array1<usize>> for Softmax {
    fn prediction(&self) -> Array1<usize> {
        // this is python argmax
        self.output()
            .outer_iter()
            .map(|row| {
                let row_argmax = row
                    .indexed_iter()
                    .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                    .unwrap()
                    .0;
                row_argmax
            })
            .collect()
    }
}

impl Softmax {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default, Clone, Debug)]
pub struct Sigmoid {
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}
impl Sigmoid {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Layer for Sigmoid {
    fn output(&self) -> &Array2<f64> {
        self.output.as_ref().unwrap()
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64> {
        self.inputs = Some(inputs.clone());
        self.output = Some(inputs.map(|x| 1. / (1. + (-x).exp())));
        self.output()
    }
    fn backward(&mut self, dvalues: &Array2<f64>) -> &Array2<f64> {
        self.dinputs = Some(Array2::zeros(dvalues.raw_dim()));
        azip!((
            mut dinput in self.dinputs.as_mut().unwrap(),
            &dvalue in dvalues,
            &output_value in self.output.as_ref().unwrap(),
        ) {
            *dinput = dvalue * (1. - output_value) * output_value;
        });
        self.dinputs()
    }
}

impl FinalActivation<Array2<usize>> for Sigmoid {
    fn prediction(&self) -> Array2<usize> {
        self.output().mapv(|x| if x > 0.5 { 1 } else { 0 })
    }
}
impl FinalActivation<Array2<f64>> for Sigmoid {
    fn prediction(&self) -> Array2<f64> {
        self.output().mapv(|x| if x > 0.5 { 1. } else { 0. })
    }
}

/// Not in the book, my own addition to try to make a simple perceptron
#[derive(Default, Clone, Debug)]
pub struct Step{
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl Step {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Layer for Step {
    fn output(&self) -> &Array2<f64> {
        self.output.as_ref().unwrap()
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }
    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64> {
        self.inputs = Some(inputs.clone());
        self.output = Some(inputs.map(|x| if *x > 0.0 { 1. } else { 0. }));
        self.output()
    }
    fn backward(&mut self, dvalues: &Array2<f64>) -> &Array2<f64> {
        todo!()
    }
}

impl FinalActivation<Array2<usize>> for Step {
    fn prediction(&self) -> Array2<usize> {
        self.output().mapv(|x| if x > 0.5 { 1 } else { 0 })
    }
}


#[derive(Default, Clone, Debug)]
pub struct Linear {
    pub output: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl Linear {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Layer for Linear {
    fn output(&self) -> &Array2<f64> {
        self.output.as_ref().unwrap()
    }
    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().unwrap()
    }

    fn forward(&mut self, inputs: &Array2<f64>) -> &Array2<f64> {
        self.inputs = Some(inputs.clone());
        self.output = Some(inputs.clone());
        self.output()
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> &Array2<f64> {
        self.dinputs = Some(dvalues.clone());
        self.dinputs()
    }
}

impl FinalActivation<Array2<f64>> for Linear {
    fn prediction(&self) -> Array2<f64> {
        self.output().clone()
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
