use std::ffi::NulError;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Error {
    Io,
    Parameter(String),
    Unknown
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io => f.write_str("cannot open file"),
            Error::Parameter(ref err) => write!(f, "{}", err.as_str()),
            Error::Unknown => f.write_str("unknown error"),
        }
    }
}

impl From<NulError> for Error {
    fn from(_err: NulError) -> Error {
        Error::Io
    }
}
