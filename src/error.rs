#[derive(Debug)]
pub enum Error {
    Io,
    Parameter(String),
    Unknown
}
