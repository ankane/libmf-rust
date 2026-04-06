use crate::bindings::*;

impl From<&[MfNode]> for MfProblem {
    fn from(data: &[MfNode]) -> Self {
        let mut umax = -1;
        let mut vmax = -1;
        for x in data {
            // TODO return error
            assert!(x.0 >= 0);
            assert!(x.1 >= 0);

            umax = umax.max(x.0);
            vmax = vmax.max(x.1);
        }

        MfProblem {
            m: umax + 1,
            n: vmax + 1,
            nnz: data.len().try_into().unwrap(),
            r: data.as_ptr(),
        }
    }
}
