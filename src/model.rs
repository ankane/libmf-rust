use crate::bindings::*;
use crate::{Error, Matrix, Params};
use std::ffi::CString;

pub struct Model {
    pub(crate) model: *mut MfModel,
}

impl Model {
    pub fn params() -> Params {
        Params::new()
    }

    pub fn load(path: &str) -> Result<Self, Error> {
        let cpath = CString::new(path).expect("CString::new failed");
        let model =  unsafe { mf_load_model(cpath.as_ptr()) };
        if model.is_null() {
            Err(Error("Cannot open model".to_string()))
        } else {
            Ok(Model { model })
        }
    }

    pub fn predict(&self, row_index: i32, column_index: i32) -> f32 {
        unsafe { mf_predict(self.model, row_index, column_index) }
    }

    pub fn save(&self, path: &str) -> Result<(), Error> {
        let cpath = CString::new(path).expect("CString::new failed");
        let status = unsafe { mf_save_model(self.model, cpath.as_ptr()) };
        if status != 0 {
            Err(Error("Cannot save model".to_string()))
        } else {
            Ok(())
        }
    }

    pub fn rows(&self) -> i32 {
        unsafe { (*self.model).m }
    }

    pub fn columns(&self) -> i32 {
        unsafe { (*self.model).n }
    }

    pub fn factors(&self) -> i32 {
        unsafe { (*self.model).k }
    }

    pub fn bias(&self) -> f32 {
        unsafe { (*self.model).b }
    }

    pub fn p_factors(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts((*self.model).p, (self.rows() * self.factors()) as usize) }
    }

    pub fn q_factors(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts((*self.model).q, (self.columns() * self.factors()) as usize) }
    }

    pub fn rmse(&self, data: &Matrix) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_rmse(&prob, self.model) }
    }

    pub fn mae(&self, data: &Matrix) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_mae(&prob, self.model) }
    }

    pub fn gkl(&self, data: &Matrix) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_gkl(&prob, self.model) }
    }

    pub fn logloss(&self, data: &Matrix) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_logloss(&prob, self.model) }
    }

    pub fn accuracy(&self, data: &Matrix) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_accuracy(&prob, self.model) }
    }

    pub fn mpr(&self, data: &Matrix, transpose: bool) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_mpr(&prob, self.model, transpose) }
    }

    pub fn auc(&self, data: &Matrix, transpose: bool) -> f64 {
        let prob = data.to_problem();
        unsafe { calc_auc(&prob, self.model, transpose) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { mf_destroy_model(&mut self.model) };
        assert!(self.model.is_null());
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Model};

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
        let model = Model::params().quiet(true).fit(&data).unwrap();
        model.predict(0, 1);

        model.p_factors();
        model.q_factors();
        model.bias();
    }

    #[test]
    fn test_fit_eval() {
        let data = generate_data();
        Model::params().quiet(true).fit_eval(&data, &data).unwrap();
    }

    #[test]
    fn test_cv() {
        let data = generate_data();
        Model::params().quiet(true).cv(&data, 5);
    }

    #[test]
    fn test_save_load() {
        let data = generate_data();
        let model = Model::params().quiet(true).fit(&data).unwrap();

        model.save("/tmp/model.txt").unwrap();
        let model = Model::load("/tmp/model.txt").unwrap();

        model.p_factors();
        model.q_factors();
        model.bias();
    }

    #[test]
    fn test_save_missing() {
        let data = generate_data();
        let model = Model::params().quiet(true).fit(&data).unwrap();
        let result = model.save("/tmp/missing/model.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_missing() {
        let result = Model::load("/tmp/missing.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics() {
        let data = generate_data();
        let model = Model::params().quiet(true).fit(&data).unwrap();

        assert!(model.rmse(&data) < 0.15);
        assert!(model.mae(&data) < 0.15);
        assert!(model.gkl(&data) < 0.01);
        assert!(model.logloss(&data) < 0.3);
        assert_eq!(1.0, model.accuracy(&data));
        assert_eq!(0.0, model.mpr(&data, false));
        assert_eq!(1.0, model.auc(&data, false));
    }

    #[test]
    fn test_predict_out_of_range() {
        let data = generate_data();
        let model = Model::params().quiet(true).fit(&data).unwrap();
        assert_eq!(model.bias(), model.predict(1000, 1000));
    }

    #[test]
    fn test_fit_empty() {
        let data = Matrix::new();
        let model = Model::params().quiet(true).fit(&data).unwrap();
        assert!(model.p_factors().is_empty());
        assert!(model.q_factors().is_empty());
        assert!(model.bias().is_nan());
    }

    #[test]
    fn test_fit_eval_empty() {
        let data = Matrix::new();
        let model = Model::params().quiet(true).fit_eval(&data, &data).unwrap();
        assert!(model.p_factors().is_empty());
        assert!(model.q_factors().is_empty());
        assert!(model.bias().is_nan());
    }

    #[test]
    fn test_bad_loss() {
        let data = generate_data();
        let result = Model::params().loss(13).fit(&data);
        assert!(result.is_err());
    }
}
