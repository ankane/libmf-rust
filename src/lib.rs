//! Large-scale sparse matrix factorization for Rust
//!
//! [View the docs](https://github.com/ankane/libmf-rust)

mod bindings;
mod error;
mod matrix;
mod model;
mod params;

pub use bindings::Loss;
pub use error::Error;
pub use matrix::Matrix;
pub use model::Model;
pub use params::Params;
