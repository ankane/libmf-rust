use crate::bindings::{MfNode, MfProblem};

/// A matrix.
pub struct Matrix {
    data: Vec<MfNode>,
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

    /// Adds a value to the matrix.
    pub fn push(&mut self, row_index: i32, column_index: i32, value: f32) {
        assert!(row_index >= 0);
        assert!(column_index >= 0);
        self.data.push(MfNode {
            u: row_index,
            v: column_index,
            r: value,
        });
    }

    pub(crate) fn to_problem(&self) -> MfProblem {
        let data = &self.data;
        let m = data.iter().map(|x| x.u).max().unwrap_or(-1) + 1;
        let n = data.iter().map(|x| x.v).max().unwrap_or(-1) + 1;

        MfProblem {
            m,
            n,
            nnz: data.len() as i64,
            r: data.as_ptr(),
        }
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
