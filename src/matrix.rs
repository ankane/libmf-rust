use crate::bindings::MfNode;
use alloc::vec::Vec;

/// A matrix.
pub struct Matrix {
    pub(crate) data: Vec<MfNode>,
}

impl Matrix {
    /// Creates a new matrix.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a new matrix with a minimum capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Returns if the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adds a value to the matrix.
    pub fn push(&mut self, row_index: i32, column_index: i32, value: f32) {
        self.data.push(MfNode {
            u: row_index,
            v: column_index,
            r: value,
        });
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix;

    #[test]
    fn test_new() {
        let mut data = Matrix::new();
        data.push(0, 0, 1.0);
    }

    #[test]
    fn test_with_capacity() {
        let mut data = Matrix::with_capacity(1);
        data.push(0, 0, 1.0);
    }
}
