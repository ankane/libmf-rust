#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]

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
