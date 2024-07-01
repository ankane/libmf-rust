use std::ffi::{c_char, c_double, c_float, c_int, c_longlong};

#[repr(C)]
pub struct MfNode {
    pub u: c_int,
    pub v: c_int,
    pub r: c_float,
}

#[repr(C)]
pub struct MfProblem {
    pub m: c_int,
    pub n: c_int,
    pub nnz: c_longlong,
    pub r: *const MfNode,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MfParameter {
    pub fun: Loss,
    pub k: c_int,
    pub nr_threads: c_int,
    pub nr_bins: c_int,
    pub nr_iters: c_int,
    pub lambda_p1: c_float,
    pub lambda_p2: c_float,
    pub lambda_q1: c_float,
    pub lambda_q2: c_float,
    pub eta: c_float,
    pub alpha: c_float,
    pub c: c_float,
    pub do_nmf: bool,
    pub quiet: bool,
    pub copy_data: bool,
}

#[repr(C)]
pub struct MfModel {
    pub fun: Loss,
    pub m: c_int,
    pub n: c_int,
    pub k: c_int,
    pub b: c_float,
    pub p: *const c_float,
    pub q: *const c_float,
}

/// Loss functions.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum Loss {
    RealL2 = 0,
    RealL1 = 1,
    RealKL = 2,
    BinaryLog = 5,
    BinaryL2 = 6,
    BinaryL1 = 7,
    OneClassRow = 10,
    OneClassCol = 11,
    OneClassL2 = 12,
}

extern "C" {
    pub fn mf_get_default_param() -> MfParameter;
    pub fn mf_save_model(model: *const MfModel, path: *const c_char) -> c_int;
    pub fn mf_load_model(path: *const c_char) -> *mut MfModel;
    pub fn mf_destroy_model(model: *mut *mut MfModel);
    pub fn mf_train(prob: *const MfProblem, param: MfParameter) -> *mut MfModel;
    pub fn mf_train_with_validation(tr: *const MfProblem, va: *const MfProblem, param: MfParameter) -> *mut MfModel;
    pub fn mf_cross_validation(prob: *const MfProblem, nr_folds: c_int, param: MfParameter) -> c_double;
    pub fn mf_predict(model: *const MfModel, u: c_int, v: c_int) -> c_float;
    pub fn calc_rmse(prob: *const MfProblem, model: *const MfModel) -> c_double;
    pub fn calc_mae(prob: *const MfProblem, model: *const MfModel) -> c_double;
    pub fn calc_gkl(prob: *const MfProblem, model: *const MfModel) -> c_double;
    pub fn calc_logloss(prob: *const MfProblem, model: *const MfModel) -> c_double;
    pub fn calc_accuracy(prob: *const MfProblem, model: *const MfModel) -> c_double;
    pub fn calc_mpr(prob: *const MfProblem, model: *const MfModel, transpose: bool) -> c_double;
    pub fn calc_auc(prob: *const MfProblem, model: *const MfModel, transpose: bool) -> c_double;
}
