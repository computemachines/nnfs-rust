use std::fmt::Binary;

use crate::{
    accuracy::{AccuracyBinary, AccuracyCategoricalLabels},
    activation_functions::{FinalActivation, ReLU, Sigmoid, Softmax},
    data::{download_fashion_mnist, load_fashion_mnist_data, load_image_from_filename},
    loss_functions::{BinaryCrossentropy, Loss, LossCategoricalCrossentropy},
    model::{Model, ModelTrainConfig, UninitializedModel},
    neurons::LayerDense,
    optimizer::{OptimizerAdam, OptimizerAdamConfig},
    report::{list_categorical_inference, FASHION_MNIST_LABELS},
};
use clap::{Args, Subcommand, ValueEnum};
use ndarray::prelude::*;

#[derive(Args, Debug, Clone)]
pub struct Ch19Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Clone, Subcommand)]
enum Command {
    #[command(about = "Train the model")]
    Train,
    #[command(about = "Inference on provided image")]
    Inference {
        #[arg(short, long)]
        filename: String,
    },
}

pub fn run(args: Ch19Args) {
    match args.command {
        Command::Train => train(),
        Command::Inference { filename } => inference(filename),
    }
}

fn inference(filename: String) {
    print!("Loading model...");
    let mut model = Model::new();

    // Add layers
    model.add(LayerDense::new(28*28, 128));
    model.add(ReLU::new());
    model.add(LayerDense::new(128, 128));
    model.add(ReLU::new());
    model.add(LayerDense::new(128, 10));
    model.add_final_activation(Softmax::new());

    // TODO: make some type of inference only model. Should not need the loss function or optimizer
    model.set(
        LossCategoricalCrossentropy::new(),
        OptimizerAdam::from(OptimizerAdamConfig::default()),
        AccuracyCategoricalLabels, 
    );

    let mut model = model.finalize();

    model.load_weights_biases("model-weights.pkl", "model-biases.pkl");
    println!("done");
    println!("Loading image...");
    let input = load_image_from_filename(&filename);
    println!("done");
    print!("Running inference...");
    model.forward(&input);
    let inference = model.output();
    println!("done");
    println!("\n\n---- Report ----");
    list_categorical_inference(inference.row(0), &FASHION_MNIST_LABELS);
}

fn train() {
    println!("Load images");
    let (train_images, train_labels) = load_fashion_mnist_data::<usize>(false, true);
    let (test_images, test_labels) = load_fashion_mnist_data::<usize>(true, false);

    // let train_images = array![[1.,0.], [0.,1.], [1.,1.], [0.,0.]];
    // let train_labels = array![[1.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];

    println!("Train images shape: {:?}", train_images.shape());
    println!("Train labels shape: {:?}", train_labels.shape());

    // Instantiate the model
    let mut model = Model::new();

    // Add layers
    model.add(LayerDense::new(train_images.shape()[1], 128));
    model.add(ReLU::new());
    model.add(LayerDense::new(128, 128));
    model.add(ReLU::new());
    model.add(LayerDense::new(128, 10));
    model.add_final_activation(Softmax::new());

    model.set(
        LossCategoricalCrossentropy::new(),
        OptimizerAdam::from(OptimizerAdamConfig {
            // learning_rate: 0.02,
            decay_rate: 1e-3,
            ..Default::default()
        }),
        AccuracyCategoricalLabels, // The book uses binary accuracy here, but that doesn't make sense
    );

    let mut model = model.finalize();

    model.train(
        &train_images,
        &train_labels,
        ModelTrainConfig {
            epochs: 10,
            batch_size: Some(128),
            print_every: Some(100),
            ..Default::default()
        },
    );

    model.save_weights_biases("model-weights.pkl", "model-biases.pkl");
    model.load_weights_biases("model-weights.pkl", "model-biases.pkl");

    // show output from trained model
    let test_image = test_images.slice(s![0..1, ..]).to_owned();
    model.forward(&test_image);
    let inference = model.output();
    println!("Inference: \n{}", inference);

    list_categorical_inference(inference.row(0), &FASHION_MNIST_LABELS);
}
