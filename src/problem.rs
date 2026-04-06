use crate::bindings::*;
use crate::Error;

impl TryFrom<&[MfNode]> for MfProblem {
    type Error = Error;

    fn try_from(data: &[MfNode]) -> Result<Self, Self::Error> {
        let mut umax = -1;
        let mut vmax = -1;
        for (i, x) in data.iter().enumerate() {
            if x.0 < 0 || x.0 == i32::MAX {
                return Err(Error::Node(i));
            }

            if x.1 < 0 || x.1 == i32::MAX {
                return Err(Error::Node(i));
            }

            umax = umax.max(x.0);
            vmax = vmax.max(x.1);
        }

        Ok(MfProblem {
            m: umax + 1,
            n: vmax + 1,
            nnz: data.len().try_into().unwrap(),
            r: data.as_ptr(),
        })
    }
}
