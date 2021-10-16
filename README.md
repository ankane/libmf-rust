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

Fit a model

```rust
let model = libmf::Model::params().fit(&data);
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
let model = libmf::Model::params().fit_eval(&train_set, &eval_set);
```

## Cross-Validation

Perform cross-validation

```rust
libmf::Model::params().cv(&data, 5);
```

## Parameters

Set parameters - default values below

```rust
libmf::Model::params()
    .loss(0)                // loss function
    .factors(8)             // number of latent factors
    .threads(12)            // number of threads used
    .bins(25)               // number of bins
    .iterations(20)         // number of iterations
    .lambda_p1(0.0)         // coefficient of L1-norm regularization on P
    .lambda_p2(0.1)         // coefficient of L2-norm regularization on P
    .lambda_q1(0.0)         // coefficient of L1-norm regularization on Q
    .lambda_q2(0.1)         // coefficient of L2-norm regularization on Q
    .learning_rate(0.1)     // learning rate
    .alpha(0.1)             // importance of negative entries
    .c(0.0001)              // desired value of negative entries
    .nmf(false)             // perform non-negative MF (NMF)
    .quiet(false);          // no outputs to stdout
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

## Reference

Specify the initial capacity for a matrix

```rust
let mut data = libmf::Matrix::with_capacity(3);
```

## Resources

- [LIBMF: A Library for Parallel Matrix Factorization in Shared-memory Systems](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_open_source.pdf)

## Upgrading

### 0.2.0

Use

```rust
let model = libmf::Model::params().factors(20).fit(&data);
```

instead of

```rust
let model = libmf::Model::new();
model.factors = 20;
model.fit(&data);
```

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
