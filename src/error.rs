use alloc::ffi::NulError;
use core::error;
use core::fmt;

/// An error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    Io,
    Parameter(&'static str),
    Unknown,
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io => f.write_str("cannot open file"),
            Error::Parameter(err) => f.write_str(err),
            Error::Unknown => f.write_str("unknown error"),
        }
    }
}

impl From<NulError> for Error {
    fn from(_err: NulError) -> Error {
        Error::Io
    }
}
