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