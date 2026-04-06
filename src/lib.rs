#![doc = include_str!("../README.md")]
#![no_std]
#![allow(clippy::needless_doctest_main)]

mod bindings;
mod error;
mod model;
mod params;

pub use bindings::Loss;
pub use bindings::MfNode as Node;
pub use error::Error;
pub use model::Model;
pub use params::Params;
