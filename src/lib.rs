#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]
#![cfg_attr(feature = "no_std", no_std)]

extern crate alloc;

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
