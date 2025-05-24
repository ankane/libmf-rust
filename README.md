# LIBMF Rust

[LIBMF](https://github.com/cjlin1/libmf) - large-scale sparse matrix factorization - for Rust

Check out [Disco](https://github.com/ankane/disco-rust) for higher-level collaborative filtering

[![Build Status](https://github.com/ankane/libmf-rust/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/libmf-rust/actions)

## Installation

Add this line to your application’s `Cargo.toml` under `[dependencies]`:

```toml
libmf = "0.3"
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
let model = libmf::Model::params().fit(&data).unwrap();
```

Make predictions

```rust
model.predict(row_index, column_index);
```

Get the latent factors (these approximate the training matrix)

```rust
model.p(row_index);
model.q(column_index);
// or
model.p_iter();
model.q_iter();
```

Get the bias (average of all elements in the training matrix)

```rust
model.bias();
```

Save the model to a file

```rust
model.save("model.txt").unwrap();
```

Load a model from a file

```rust
let model = libmf::Model::load("model.txt").unwrap();
```

Pass a validation set

```rust
let model = libmf::Model::params().fit_eval(&train_set, &eval_set).unwrap();
```

## Cross-Validation

Perform cross-validation

```rust
let avg_error = libmf::Model::params().cv(&data, 5).unwrap();
```

## Parameters

Set parameters - default values below

```rust
libmf::Model::params()
    .loss(libmf::Loss::RealL2)     // loss function
    .factors(8)                    // number of latent factors
    .threads(12)                   // number of threads
    .bins(25)                      // number of bins
    .iterations(20)                // number of iterations
    .lambda_p1(0.0)                // L1-regularization parameter for P
    .lambda_p2(0.1)                // L2-regularization parameter for P
    .lambda_q1(0.0)                // L1-regularization parameter for Q
    .lambda_q2(0.1)                // L2-regularization parameter for Q
    .learning_rate(0.1)            // learning rate
    .alpha(1.0)                    // importance of negative entries
    .c(0.0001)                     // desired value of negative entries
    .nmf(false)                    // perform non-negative MF (NMF)
    .quiet(false);                 // no outputs to stdout
```

### Loss Functions

For real-valued matrix factorization

- `Loss::RealL2` - squared error (L2-norm)
- `Loss::RealL1` - absolute error (L1-norm)
- `Loss::RealKL` - generalized KL-divergence

For binary matrix factorization

- `Loss::BinaryLog` - logarithmic error
- `Loss::BinaryL2` - squared hinge loss
- `Loss::BinaryL1` - hinge loss

For one-class matrix factorization

- `Loss::OneClassRow` - row-oriented pair-wise logarithmic loss
- `Loss::OneClassCol` - column-oriented pair-wise logarithmic loss
- `Loss::OneClassL2` - squared error (L2-norm)

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

## Example

Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) and use:

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let mut train_set = libmf::Matrix::with_capacity(80000);
    let mut valid_set = libmf::Matrix::with_capacity(20000);

    let file = File::open("path/to/ml-100k/u.data").unwrap();
    let rdr = BufReader::new(file);
    for (i, line) in rdr.lines().enumerate() {
        let line = line.unwrap();
        let mut row = line.split('\t');

        let user_id = row.next().unwrap().parse().unwrap();
        let item_id = row.next().unwrap().parse().unwrap();
        let rating = row.next().unwrap().parse().unwrap();

        let matrix = if i < 80000 { &mut train_set } else { &mut valid_set };
        matrix.push(user_id, item_id, rating);
    }

    let model = libmf::Model::params()
        .factors(20)
        .fit_eval(&train_set, &valid_set)
        .unwrap();
    println!("RMSE: {:?}", model.rmse(&valid_set));
}
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
