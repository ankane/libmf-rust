use crate::bindings::*;
use crate::{Error, Matrix};
use core::ffi::c_int;

impl TryFrom<&Matrix> for MfProblem {
    type Error = Error;

    fn try_from(data: &Matrix) -> Result<Self, Self::Error> {
        let data = &data.data;
        let mut umax = -1;
        let mut vmax = -1;
        for (i, x) in data.iter().enumerate() {
            if x.u < 0 || x.u == c_int::MAX {
                return Err(Error::Node(i));
            }

            if x.v < 0 || x.v == c_int::MAX {
                return Err(Error::Node(i));
            }

            umax = umax.max(x.u);
            vmax = vmax.max(x.v);
        }

        let nnz = data
            .len()
            .try_into()
            .map_err(|_| Error::Parameter("too much data"))?;

        Ok(MfProblem {
            m: umax + 1,
            n: vmax + 1,
            nnz,
            r: data.as_ptr(),
        })
    }
}
