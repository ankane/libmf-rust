# LIBMF Rust

[LIBMF](https://github.com/cjlin1/libmf) - large-scale sparse matrix factorization - for Rust

[![Build Status](https://github.com/ankane/libmf-rust/workflows/build/badge.svg?branch=master)](https://github.com/ankane/libmf-rust/actions)

## Installation

Add this line to your application’s `Cargo.toml` under `[dependencies]`:

```toml
libmf = "0.1"
```

## Getting Started

Prep your data in the format `row_index, column_index, value`

```rust
let mut data = libmf::Matrix::new();
data.push(0, 0, 5.0);
data.push(0, 2, 3.5);
data.push(1, 1, 4.0);
```

Create a model

```rust
let mut model = libmf::Model::new();
model.fit(&data);
```

Make predictions

```rust
model.predict(row_index, column_index);
```

Get the latent factors (these approximate the training matrix)

```rust
model.p_factors();
model.q_factors();
```

Get the bias (average of all elements in the training matrix)

```rust
model.bias();
```

Save the model to a file

```rust
model.save("model.txt");
```

Load the model from a file

```rust
let model = libmf::Model::load("model.txt");
```

Pass a validation set

```rust
model.fit_eval(&train_set, &eval_set);
```

## Cross-Validation

Perform cross-validation

```rust
model.cv(&data, 5);
```

## Parameters

Set parameters - default values below

```rust
model.loss = 0;                // loss function
model.factors = 8;             // number of latent factors
model.threads = 12;            // number of threads used
model.bins = 25;               // number of bins
model.iterations = 20;         // number of iterations
model.lambda_p1 = 0;           // coefficient of L1-norm regularization on P
model.lambda_p2 = 0.1;         // coefficient of L2-norm regularization on P
model.lambda_q1 = 0;           // coefficient of L1-norm regularization on Q
model.lambda_q2 = 0.1;         // coefficient of L2-norm regularization on Q
model.learning_rate = 0.1;     // learning rate
model.alpha = 0.1;             // importance of negative entries
model.c = 0.0001;              // desired value of negative entries
model.nmf = false;             // perform non-negative MF (NMF)
model.quiet = false;           // no outputs to stdout
```

### Loss Functions

For real-valued matrix factorization

- 0 - squared error (L2-norm)
- 1 - absolute error (L1-norm)
- 2 - generalized KL-divergence

For binary matrix factorization

- 5 - logarithmic error
- 6 - squared hinge loss
- 7 - hinge loss

For one-class matrix factorization

- 10 - row-oriented pair-wise logarithmic loss
- 11 - column-oriented pair-wise logarithmic loss
- 12 - squared error (L2-norm)

## Metrics

Calculate RMSE (for real-valued MF)

```rust
model.rmse(&data);
```

Calculate MAE (for real-valued MF)

```rust
model.mae(&data);
```

Calculate generalized KL-divergence (for non-negative real-valued MF)

```rust
model.gkl(&data);
```

Calculate logarithmic loss (for binary MF)

```rust
model.logloss(&data);
```

Calculate accuracy (for binary MF)

```rust
model.accuracy(&data);
```

Calculate MPR (for one-class MF)

```rust
model.mpr(&data, transpose);
```

Calculate AUC (for one-class MF)

```rust
model.auc(&data, transpose);
```

## Disk-Level Training

Train directly from files

```ruby
model.fit_disk("train.txt")
model.fit_eval_disk("train.txt", "validate.txt")
model.cv_disk("train.txt")
```

Data should be in the format `row_index column_index value`:

```txt
0 0 5.0
0 2 3.5
1 1 4.0
```

## Reference

Specify the initial capacity for a matrix

```rust
let mut data = libmf::Matrix::with_capacity(3);
```

## Resources

- [LIBMF: A Library for Parallel Matrix Factorization in Shared-memory Systems](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_open_source.pdf)

## History

View the [changelog](https://github.com/ankane/libmf-rust/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/libmf-rust/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/libmf-rust/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone --recursive https://github.com/ankane/libmf-rust.git
cd libmf-rust
cargo test
```
