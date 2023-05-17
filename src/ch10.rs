use crate::{
    activation_functions::ReLU,
    analysis_functions,
    data::{spiral_data, visualize_nn_scatter, new_root_area},
    loss_functions::SoftmaxLossCategoricalCrossEntropy,
    neurons::LayerDense,
    optimizer::{OptimizerSDG, OptimizerSDGConfig},
};

use ndarray::prelude::*;
use plotters::prelude::*;

// I'm deviating from the book slightly. Using structs and methods is much nicer.
struct NetworkOutput(f64, Array2<f64>);

#[derive(Clone)]
struct Network {
    dense1: LayerDense,
    dense2: LayerDense,
    activation1: ReLU,
    loss_activation: SoftmaxLossCategoricalCrossEntropy,
}
impl Network {
    fn new(num_labels: usize) -> Self {
        // create dense layer with 2 input features and 64 output values
        let mut dense1 = LayerDense::new(2, 64);

        // create ReLU activation (to be used with dense layer)
        let mut activation1 = ReLU::new();

        // create second dense layer with 64 input features (as we take from lower
        // layer) and 3 output values (output labels)
        let mut dense2 = LayerDense::new(64, num_labels);

        // create softmax classifier with combined loss and activation
        let mut loss_activation = SoftmaxLossCategoricalCrossEntropy::new();

        Self {
            dense1,
            activation1,
            dense2,
            loss_activation,
        }
    }
    fn forward(&mut self, input: &Array2<f64>, y_true: &Array1<usize>) -> NetworkOutput {
        self.dense1.forward(input);
        self.activation1
            .forward(self.dense1.output.as_ref().unwrap());
        self.dense2
            .forward(self.activation1.output.as_ref().unwrap());
        let loss = self
            .loss_activation
            .forward_labels(self.dense2.output.as_ref().unwrap(), &y_true);
        let prediction = self.loss_activation.output.take().unwrap();
        NetworkOutput(loss, prediction)
    }
    fn backward(&mut self, prediction: &Array2<f64>, y_true: &Array1<usize>) {
        self.loss_activation.backward_labels(prediction, y_true);
        self.dense2
            .backward(self.loss_activation.dinputs.as_ref().unwrap());
        self.activation1
            .backward(self.dense2.dinputs.as_ref().unwrap());
        self.dense1
            .backward(self.activation1.dinputs.as_ref().unwrap());
    }
}

pub fn run() {
    let num_epochs = 10000;
    let num_labels = 5;
    let mut losses1 = Array1::zeros(num_epochs);
    let mut losses2 = Array1::zeros(num_epochs);

    #[allow(non_snake_case)]
    let (data, labels) = spiral_data(100, num_labels);

    let mut network = Network::new(num_labels);

    // create optimizer object
    let mut optimizer = OptimizerSDG::from(OptimizerSDGConfig {
        learning_rate: 0.5,
        decay_rate: 5e-5,
        ..Default::default()
    });
    // let mut optimizer_other = OptimizerSDG::from(OptimizerSDGConfig {
    //     learning_rate: 4.0,
    //     decay_rate: 1e-2,
    //     ..Default::default()
    // });

    let mut gif = new_root_area("plots/ch10-animated-training.gif", true);

    // train in loop
    for epoch in 0..num_epochs {
        // perform a forward pass
        let NetworkOutput(loss, prediction) = network.forward(&data, &labels);
        // let prediction_other = loss_activation_other.output.take().unwrap();
        let accuracy = analysis_functions::get_accuracy(&prediction, &labels);
        // let accuracy_other = analysis_functions::get_accuracy(&prediction_other, &y);

        losses1[epoch] = loss;
        // losses2[epoch] = loss_other;

        if epoch % 10 == 0 {
            println!("Epoch: {}", epoch);
            println!("Loss: {}", loss);
            println!("Accuracy: {}", accuracy);
            println!("Learning rate: {}", optimizer.current_learning_rate);

            println!("Starting gif frame");
            let mut network_clone = network.clone();
            visualize_nn_scatter(
                &data,
                &labels,
                num_labels,
                |(x, y)| {
                    let NetworkOutput(_, prediction) = network_clone.forward(&array![[x, y]], &labels);
                    let g = colorgrad::rainbow();
    let colors: Vec<RGBAColor> =
        (0..max_label).map(|i| {
            let c = g.at(i as f64 / max_label as f64).to_rgba8();
            RGBAColor(c[0], c[1], c[2], c[3] as f64 / 256.0)}).collect();

                    let r = prediction[[0, 0]] * 256.0;
                    let g = prediction[[0, 1]] * 256.0;
                    let b = prediction[[0, 2]] * 256.0;

                    RGBAColor(r as u8, g as u8, b as u8, 0.5)
                },
                &gif,
            );
            println!("Done gif frame.");
        }

        // perform backward pass
        network.backward(&prediction, &labels);

        // update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(&mut network.dense1);
        optimizer.update_params(&mut network.dense2);
        optimizer.post_update_params();

        // optimizer_other.pre_update_params();
        // optimizer_other.update_params(&mut dense1_other);
        // optimizer_other.update_params(&mut dense2_other);
        // optimizer_other.post_update_params();
    }

    visualize_nn_scatter(
        &data,
        &labels,
        num_labels,
        |(x, y)| {
            let NetworkOutput(_, prediction) = network.forward(&array![[x, y]], &labels);
            let r = prediction[[0, 0]] * 256.0;
            let g = prediction[[0, 1]] * 256.0;
            let b = prediction[[0, 2]] * 256.0;

            RGBAColor(r as u8, g as u8, b as u8, 0.5)
        },
        &new_root_area("plots/ch10_network.png", false),
    );

    let root = BitMapBackend::new("plots/ch10.png", (1024 * 2, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_2d(0..num_epochs, 0.0..2.0)
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            losses1.iter().enumerate().map(|(x, y)| (x, *y)),
            BLACK,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            losses2.iter().enumerate().map(|(x, y)| (x, *y)),
            BLUE,
        ))
        .unwrap();
}
