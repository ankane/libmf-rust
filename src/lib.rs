#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]

mod bindings;
mod error;
mod model;
mod params;
mod problem;

pub use bindings::{Loss, Node};
pub use error::Error;
pub use model::Model;
pub use params::Params;
