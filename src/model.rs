use crate::bindings::*;
use crate::Matrix;
use std::ffi::CString;

pub struct Model {
    model: *const MfModel,
    pub loss: i32,
    pub factors: i32,
    pub threads: i32,
    pub bins: i32,
    pub iterations: i32,
    pub lambda_p1: f32,
    pub lambda_p2: f32,
    pub lambda_q1: f32,
    pub lambda_q2: f32,
    pub learning_rate: f32,
    pub alpha: f32,
    pub c: f32,
    pub nmf: bool,
    pub quiet: bool
}

impl Model {
    pub fn new() -> Self {
        Self::with_model(std::ptr::null())
    }

    pub fn load(path: &str) -> Self {
        let cpath = CString::new(path).expect("CString::new failed");
        Self::with_model(unsafe { mf_load_model(cpath.as_ptr()) })
    }

    pub fn fit(&mut self, data: &Matrix) {
        let prob = data.to_problem();
        self.model = unsafe { mf_train(&prob, self.param()) };
    }

    pub fn fit_disk(&mut self, path: &str) {
        let cpath = CString::new(path).expect("CString::new failed");
        self.model = unsafe { mf_train_on_disk(cpath.as_ptr(), self.param()) };
    }

    pub fn fit_eval(&mut self, train_set: &Matrix, eval_set: &Matrix) {
        let tr = train_set.to_problem();
        let va = eval_set.to_problem();
        self.model = unsafe { mf_train_with_validation(&tr, &va, self.param()) };
    }

    pub fn fit_eval_disk(&mut self, train_path: &str, eval_path: &str) {
        let trpath = CString::new(train_path).expect("CString::new failed");
        let vapath = CString::new(eval_path).expect("CString::new failed");
        self.model = unsafe { mf_train_with_validation_on_disk(trpath.as_ptr(), vapath.as_ptr(), self.param()) };
    }

    pub fn cv(&mut self, data: &Matrix, folds: i32) {
        let prob = data.to_problem();
        unsafe { mf_cross_validation(&prob, folds, self.param()); }
    }

    pub fn cv_disk(&mut self, path: &str, folds: i32) {
        let cpath = CString::new(path).expect("CString::new failed");
        unsafe { mf_cross_validation_on_disk(cpath.as_ptr(), folds, self.param()); }
    }

    pub fn predict(&self, row_index: i32, column_index: i32) -> f32 {
        assert!(self.is_fit());
        unsafe { mf_predict(self.model, row_index, column_index) }
    }

    pub fn save(&self, path: &str) {
        assert!(self.is_fit());
        let cpath = CString::new(path).expect("CString::new failed");
        unsafe { mf_save_model(self.model, cpath.as_ptr()); }
    }

    pub fn rows(&self) -> i32 {
        if self.is_fit() {
            unsafe { (*self.model).m }
        } else {
            0
        }
    }

    pub fn columns(&self) -> i32 {
        if self.is_fit() {
            unsafe { (*self.model).n }
        } else {
            0
        }
    }

    pub fn factors(&self) -> i32 {
        if self.is_fit() {
            unsafe { (*self.model).k }
        } else {
            self.factors
        }
    }

    pub fn bias(&self) -> f32 {
        if self.is_fit() {
            unsafe { (*self.model).b }
        } else {
            0.0
        }
    }

    pub fn p_factors(&self) -> &[f32] {
        if self.is_fit() {
            unsafe { std::slice::from_raw_parts((*self.model).p, (self.rows() * self.factors()) as usize) }
        } else {
            &[]
        }
    }

    pub fn q_factors(&self) -> &[f32] {
        if self.is_fit() {
            unsafe { std::slice::from_raw_parts((*self.model).q, (self.columns() * self.factors()) as usize) }
        } else {
            &[]
        }
    }

    pub fn rmse(&self, data: &Matrix) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_rmse(&prob, self.model) }
    }

    pub fn mae(&self, data: &Matrix) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_mae(&prob, self.model) }
    }

    pub fn gkl(&self, data: &Matrix) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_gkl(&prob, self.model) }
    }

    pub fn logloss(&self, data: &Matrix) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_logloss(&prob, self.model) }
    }

    pub fn accuracy(&self, data: &Matrix) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_accuracy(&prob, self.model) }
    }

    pub fn mpr(&self, data: &Matrix, transpose: bool) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_mpr(&prob, self.model, transpose) }
    }

    pub fn auc(&self, data: &Matrix, transpose: bool) -> f64 {
        assert!(self.is_fit());
        let prob = data.to_problem();
        unsafe { calc_auc(&prob, self.model, transpose) }
    }

    fn with_model(model: *const MfModel) -> Self {
        let param = unsafe { mf_get_default_param() };
        Self {
            model: model,
            loss: param.fun,
            factors: param.k,
            threads: param.nr_threads,
            bins: 25, // prevent warning
            iterations: param.nr_iters,
            lambda_p1: param.lambda_p1,
            lambda_p2: param.lambda_p2,
            lambda_q1: param.lambda_q1,
            lambda_q2: param.lambda_q2,
            learning_rate: param.eta,
            alpha: param.alpha,
            c: param.c,
            nmf: param.do_nmf,
            quiet: param.quiet
        }
    }

    fn param(&self) -> MfParameter {
        let mut param = unsafe { mf_get_default_param() };
        param.fun = self.loss;
        param.k = self.factors;
        param.nr_threads = self.threads;
        param.nr_bins = self.bins;
        param.nr_iters = self.iterations;
        param.lambda_p1 = self.lambda_p1;
        param.lambda_p2 = self.lambda_p2;
        param.lambda_q1 = self.lambda_q1;
        param.lambda_q2 = self.lambda_q2;
        param.eta = self.learning_rate;
        param.alpha = self.alpha;
        param.c = self.c;
        param.do_nmf = self.nmf;
        param.quiet = self.quiet;
        param
    }

    fn is_fit(&self) -> bool {
        !self.model.is_null()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Model};

    const TRAIN_PATH: &str = "vendor/libmf/demo/real_matrix.tr.txt";
    const EVAL_PATH: &str = "vendor/libmf/demo/real_matrix.te.txt";

    fn generate_data() -> Matrix {
        let mut data = Matrix::new();
        data.push(0, 0, 1.0);
        data.push(1, 0, 2.0);
        data.push(1, 1, 1.0);
        data
    }

    #[test]
    fn test_fit() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.fit(&data);
        model.predict(0, 1);

        model.p_factors();
        model.q_factors();
        model.bias();
    }

    #[test]
    fn test_fit_disk() {
        let mut model = Model::new();
        model.quiet = true;
        model.fit_disk(TRAIN_PATH);
    }

    #[test]
    fn test_fit_eval() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.fit_eval(&data, &data);
    }

    #[test]
    fn test_fit_eval_disk() {
        let mut model = Model::new();
        model.quiet = true;
        model.fit_eval_disk(TRAIN_PATH, EVAL_PATH);
    }

    #[test]
    fn test_cv() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.cv(&data, 5);
    }

    #[test]
    fn test_cv_disk() {
        let mut model = Model::new();
        model.quiet = true;
        model.cv_disk(TRAIN_PATH, 5);
    }

    #[test]
    fn test_save_load() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.fit(&data);

        model.save("/tmp/model.txt");
        let model = Model::load("/tmp/model.txt");

        model.p_factors();
        model.q_factors();
        model.bias();
    }

    #[test]
    fn test_metrics() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.fit(&data);

        assert!(model.rmse(&data) < 0.15);
        assert!(model.mae(&data) < 0.15);
        assert!(model.gkl(&data) < 0.01);
        assert!(model.logloss(&data) < 0.3);
        assert_eq!(1.0, model.accuracy(&data));
        assert_eq!(0.0, model.mpr(&data, false));
        assert_eq!(1.0, model.auc(&data, false));
    }

    #[test]
    fn test_not_fit() {
        let model = Model::new();
        assert_eq!(0.0, model.bias());
        assert!(model.p_factors().is_empty());
        assert!(model.q_factors().is_empty());
    }

    #[test]
    #[should_panic(expected = "assertion failed: self.is_fit()")]
    fn test_predict_not_fit() {
        let model = Model::new();
        model.predict(0, 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: self.is_fit()")]
    fn test_save_not_fit() {
        let model = Model::new();
        model.save("/tmp/model.txt");
    }

    #[test]
    fn test_predict_out_of_range() {
        let data = generate_data();
        let mut model = Model::new();
        model.quiet = true;
        model.fit(&data);
        assert_eq!(model.bias(), model.predict(1000, 1000));
    }
}
