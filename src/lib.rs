//! Large-scale sparse matrix factorization for Rust
//!
//! [View the docs](https://github.com/ankane/libmf-rust)

mod bindings;
mod matrix;
mod model;
mod params;

pub use matrix::Matrix;
pub use model::Model;
pub use params::Params;
