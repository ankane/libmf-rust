use crate::bindings::*;
use crate::Error;
use core::ffi::c_int;

impl TryFrom<&[Node]> for MfProblem {
    type Error = Error;

    fn try_from(data: &[Node]) -> Result<Self, Self::Error> {
        let mut umax = -1;
        let mut vmax = -1;
        for (i, x) in data.iter().enumerate() {
            if x.0 < 0 || x.0 == c_int::MAX {
                return Err(Error::Node(i));
            }

            if x.1 < 0 || x.1 == c_int::MAX {
                return Err(Error::Node(i));
            }

            umax = umax.max(x.0);
            vmax = vmax.max(x.1);
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
