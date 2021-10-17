use std::ffi::NulError;

#[derive(Debug)]
pub enum Error {
    Io,
    Parameter(String),
    Unknown
}

impl From<NulError> for Error {
    fn from(_err: NulError) -> Error {
        Error::Io
    }
}
