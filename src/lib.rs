#![doc = include_str!("../README.md")]
#![no_std]
#![allow(clippy::needless_doctest_main)]

extern crate alloc;

mod bindings;
mod error;
mod matrix;
mod model;
mod params;
mod problem;

pub use bindings::Loss;
pub use error::Error;
pub use matrix::Matrix;
pub use model::Model;
pub use params::Params;
