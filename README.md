# Chapter 19
## Fashion MNIST data
You will need to extract the zip file manually after downloading it. The zip crate can't handle the compression algorithm used.
I took some pictures of my own clothes and some random objects I not included in the training set. 

Using GIMP, I cropped the images to 28x28 and converted them to grayscale, inverted the colors, and did a auto white balance. The backgrounds are not perfectly black like the training images.

Raw image:
![T-shirt](/my_images/2.jpg)

Preprocessed image:
![T-shirt](/my_images/2.png)


```shell
$ cargo run -- ch19 train
...
epoch: 10, acc: 0.915, loss: 10.0595011732 (data_loss: 10.060, reg_loss: 0.000)

$ cargo run -- ch19 inference --filename my_images/2.png 
    Finished dev [unoptimized + debuginfo] target(s) in 0.08s
     Running `target/debug/nnfs-rust ch19 inference --filename my_images/2.png`
Loading model...done
Loading image...
done
Running inference...done


---- Report ----
Bag: 27.11%
Shirt: 21.69%
Coat: 19.01%
Pullover: 15.71%
T-shirt/top: 10.75%
Trouser: 3.24%
Dress: 2.43%
Ankle boot: 0.04%
Sandal: 0.02%
Sneaker: 0.00%
```
The results could definity be inproved by manipulating the training images to make it look superficially more like my images. I get over 90% accuracy on the testing set.
# Chapter 18
## The model api
Example usage:
```rust
let binary_inputs = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
let and_outputs =   array![[0.],     [0.],     [0.],     [1.]];
let xor_outputs =   array![[0.],     [1.],     [1.],     [0.]];

// Instantiate the (uninitialized) model
let mut model = Model::new();

// Add perceptron layers
model.add(LayerDense::new(2, 1));
model.add_final_activation(Sigmoid::new());
    
// Set the loss, optimizer, and accuracy type
model.set(
    BinaryCrossentropy::new(),
    OptimizerSDG::from(OptimizerSDGConfig {
        learning_rate: 1.0,
        ..Default::default()
    }),
    AccuracyBinary,
);

// Finalize the model. This changes the type of the model to permit training.
let mut model = model.finalize();

// Train the model
model.train(
    &binary_inputs,
    &and_outputs,
    ModelTrainConfig {
        epochs: 30,
        print_every: 1,
    },
);


// show output from trained model
model.forward(&binary_inputs);
let inference = model.output();
println!("Inference: \n{}", inference);
```
Rust really shines here, the compiler will catch incompatible uses of Loss, input dimension, and Accuracy types. Python however is much much easier to do exploratory data analysis and visualization with matplotlib. This was the only chapter where I thought that rust was as pleasant to do ML in as python.
```rust
## Example output
```shell
$ cargo run -- ch18
    Finished dev [unoptimized + debuginfo] target(s) in 2.00s
     Running `target/debug/nnfs-rust ch18`
epoch: 5, acc: 0.750, loss: 0.357 (data_loss: 0.357, reg_loss: 0.000)
epoch: 10, acc: 1.000, loss: 0.233 (data_loss: 0.233, reg_loss: 0.000)
epoch: 15, acc: 1.000, loss: 0.177 (data_loss: 0.177, reg_loss: 0.000)
epoch: 20, acc: 1.000, loss: 0.150 (data_loss: 0.150, reg_loss: 0.000)
epoch: 25, acc: 1.000, loss: 0.138 (data_loss: 0.138, reg_loss: 0.000)
epoch: 30, acc: 1.000, loss: 0.132 (data_loss: 0.132, reg_loss: 0.000)
Inference: 
[[0.000015782928969143955],
 [0.0047410583767099794],
 [0.004878318834536076],
 [0.5967060593546276]]
```
## Unfinished work
I did not implement the combined Softmax + Categorical Cross Entropy backward pass with the model but left it open to be done using generics. All I would need to do is write a specialized implementation:
```rust
impl Model<Array1<f64>, Softmax, CategoricalCrossEntropy> {
    ...
    fn backward(&mut self, y_pred: &Array1<f64>, y_true: &Array1<f64>) {
        ...
    }
}
```
Code that used both Softmax and CategoricalCrossEntropy would then get a free speedup without needing to be updated. 

# Chapter 17
Regression
- Requires at least 2 nonlinear activation layers
- Very sensitive to initial weights
- Different activation functions call for different weight initialization strategies
- I struggled to get good training results. 
```shell
$ cargo run -- ch17 adam -l 0.05 -d 5e-3
```
![Prediction vs Training Sine Data](/plots/ch17-sine-prediction-vs-training.png)

# Chapter 16
Binary Logistic Regression
```shell
$ cargo run -- ch16 loss-only adam -l 0.01 -d 5e-7
...
Epoch: 9800,
Data Loss: 0.920, Regularization Loss: 1.082, Accuracy: 0.870,
Test Data Loss: 1.319, Test Accuracy: 0.800

Epoch: 9900,
Data Loss: 0.937, Regularization Loss: 1.132, Accuracy: 0.870,
Test Data Loss: 1.358, Test Accuracy: 0.795
```

# Chapter 15
Dropout
![Training Forward Run Visualization](/plots/ch15-adam-2x64-l0.05-d0.0000005-e0.0000001-b1_0.9-b2_0.999-animation.gif)
```shell
$ cargo run -- ch15 -n 10000 --l2reg 1e-5 --dropout 0.1 loss-only adam
```
# Chapter 14
Regularization
```shell
$ cargo run -- ch14 -n 10000 --l2reg=5e-4 loss-only adam -l 0.05 -d 5e-7 --beta-1 0.9 --beta-2 0.999
    Finished dev [unoptimized + debuginfo] target(s) in 0.05s
     Running `target/debug/nnfs-rust ch14 -n 850 --l2reg 5e-4 animate adam -l 0.05 -d 5e-7 --beta-1 0.9 --beta-2 0.999`

Epoch: 0,
Data Loss: 1.609, Regularization Loss: 0.000, Accuracy: 0.226,
Test Data Loss: 1.609, Test Regularization Loss: 0.000, Test Accuracy: 0.234

Epoch: 1,
Data Loss: 1.601, Regularization Loss: 0.001, Accuracy: 0.276,
Test Data Loss: 1.602, Test Regularization Loss: 0.001, Test Accuracy: 0.268
...

Epoch: 9998,
Data Loss: 2.499, Regularization Loss: 7.817, Accuracy: 0.538,
Test Data Loss: 2.510, Test Regularization Loss: 7.817, Test Accuracy: 0.534

Epoch: 9999,
Data Loss: 2.500, Regularization Loss: 7.820, Accuracy: 0.540,
Test Data Loss: 2.510, Test Regularization Loss: 7.820, Test Accuracy: 0.534
```


# Chapter 13
datasets should be about ~ +-1 or preprocessed
small datasets can be augmented with geometric manipulations of the original data

# Chapter 12
cross-validation

# Chapter 11
Overfitting

# Chapter 10

`cargo run -- ch10 animate adam -l 0.05 -d 5e-7 --beta-1 0.9 --beta-2 0.999`
![Training Forward Run Visualization](/plots/ch10-adam-l0.05-d0.0000005-e0.0000001-b1_0.9-b2_0.999-animation.gif)