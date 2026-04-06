#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod bindings;
mod error;
mod model;
mod params;

pub use bindings::Loss;
pub use bindings::MfNode as Node;
pub use error::Error;
pub use model::Model;
pub use params::Params;
