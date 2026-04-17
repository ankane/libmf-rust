use crate::bindings::*;
use crate::{Error, Node, Params};
use core::ffi::CStr;
use core::slice::Chunks;

/// A model.
#[derive(Debug)]
pub struct Model {
    pub(crate) model: *mut MfModel,
}

impl Model {
    /// Returns a new set of parameters.
    pub fn params() -> Params {
        Params::new()
    }

    /// Loads a model from a file.
    pub fn load(path: &CStr) -> Result<Self, Error> {
        let model = unsafe { mf_load_model(path.as_ptr()) };
        if model.is_null() {
            return Err(Error::Io);
        }
        Ok(Model { model })
    }

    /// Returns the predicted value for a row and column.
    pub fn predict(&self, row_index: i32, column_index: i32) -> f32 {
        unsafe { mf_predict(self.model, row_index, column_index) }
    }

    /// Saves the model to a file.
    pub fn save(&self, path: &CStr) -> Result<(), Error> {
        let status = unsafe { mf_save_model(self.model, path.as_ptr()) };
        if status != 0 {
            return Err(Error::Io);
        }
        Ok(())
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> i32 {
        unsafe { (*self.model).m }
    }

    /// Returns the number of columns.
    pub fn columns(&self) -> i32 {
        unsafe { (*self.model).n }
    }

    /// Returns the number of factors.
    pub fn factors(&self) -> i32 {
        unsafe { (*self.model).k }
    }

    /// Returns the bias.
    pub fn bias(&self) -> f32 {
        unsafe { (*self.model).b }
    }

    /// Returns the latent factors for rows.
    pub fn p_factors(&self) -> &[f32] {
        unsafe {
            core::slice::from_raw_parts((*self.model).p, (self.rows() * self.factors()) as usize)
        }
    }

    /// Returns the latent factors for columns.
    pub fn q_factors(&self) -> &[f32] {
        unsafe {
            core::slice::from_raw_parts((*self.model).q, (self.columns() * self.factors()) as usize)
        }
    }

    /// Returns the latent factors for a row.
    pub fn p(&self, row_index: i32) -> Option<&[f32]> {
        if row_index >= 0 && row_index < self.rows() {
            let factors = self.factors();
            let start_index = factors as usize * row_index as usize;
            let end_index = factors as usize * (row_index as usize + 1);
            return Some(&self.p_factors()[start_index..end_index]);
        }
        None
    }

    /// Returns the latent factors for a column.
    pub fn q(&self, column_index: i32) -> Option<&[f32]> {
        if column_index >= 0 && column_index < self.columns() {
            let factors = self.factors();
            let start_index = factors as usize * column_index as usize;
            let end_index = factors as usize * (column_index as usize + 1);
            return Some(&self.q_factors()[start_index..end_index]);
        }
        None
    }

    /// Returns an iterator over the latent factors for rows.
    pub fn p_iter(&self) -> Chunks<'_, f32> {
        self.p_factors().chunks(self.factors() as usize)
    }

    /// Returns an iterator over the latent factors for columns.
    pub fn q_iter(&self) -> Chunks<'_, f32> {
        self.q_factors().chunks(self.factors() as usize)
    }

    /// Calculates RMSE (for real-valued MF).
    pub fn rmse(&self, data: &[Node]) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_rmse(&prob, self.model) })
    }

    /// Calculates MAE (for real-valued MF).
    pub fn mae(&self, data: &[Node]) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_mae(&prob, self.model) })
    }

    /// Calculates generalized KL-divergence (for non-negative real-valued MF).
    pub fn gkl(&self, data: &[Node]) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_gkl(&prob, self.model) })
    }

    /// Calculates logarithmic loss (for binary MF).
    pub fn logloss(&self, data: &[Node]) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_logloss(&prob, self.model) })
    }

    /// Calculates accuracy (for binary MF).
    pub fn accuracy(&self, data: &[Node]) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_accuracy(&prob, self.model) })
    }

    /// Calculates MPR (for one-class MF).
    pub fn mpr(&self, data: &[Node], transpose: bool) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_mpr(&prob, self.model, transpose) })
    }

    /// Calculates AUC (for one-class MF).
    pub fn auc(&self, data: &[Node], transpose: bool) -> Result<f64, Error> {
        let prob = data.try_into()?;
        Ok(unsafe { calc_auc(&prob, self.model, transpose) })
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
    use crate::{Error, Loss, Model, Node};

    fn generate_data() -> [Node; 3] {
        [Node(0, 0, 1.0), Node(1, 0, 2.0), Node(1, 1, 1.0)]
    }

    #[test]
    fn test_fit() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();
        model.predict(0, 1);

        // TODO assert in delta
        assert_eq!(4.0 / 3.0, model.bias());

        let p_factors = model.p_factors();
        let q_factors = model.q_factors();

        assert_eq!(model.p(0), Some(&p_factors[0..8]));
        assert_eq!(model.p(1), Some(&p_factors[8..]));

        assert_eq!(model.q(0), Some(&q_factors[0..8]));
        assert_eq!(model.q(1), Some(&q_factors[8..]));

        let mut p_iter = model.p_iter();
        let mut q_iter = model.q_iter();

        assert_eq!(p_iter.len(), 2);
        assert_eq!(q_iter.len(), 2);

        assert_eq!(model.p(0), p_iter.next());
        assert_eq!(model.p(1), p_iter.next());
        assert_eq!(model.p(2), None);

        assert_eq!(model.q(0), q_iter.next());
        assert_eq!(model.q(1), q_iter.next());
        assert_eq!(model.q(2), None);
    }

    #[test]
    fn test_fit_eval() {
        let data = generate_data();
        Model::params().fit_eval(&data, &data).unwrap();
    }

    #[test]
    fn test_fit_eval_extra() {
        let data = generate_data();
        let model = Model::params()
            .fit_eval(&data, &[Node(1000000, 1000000, 1.0)])
            .unwrap();
        assert_eq!(model.rows(), 2);
        assert_eq!(model.columns(), 2);
    }

    #[test]
    fn test_fit_eval_extra_rows_one_class_l2() {
        let data = generate_data();
        let result = Model::params()
            .loss(Loss::OneClassL2)
            .quiet(false)
            .fit_eval(&data, &[Node(1000000, 1, 1.0)]);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("eval set cannot have extra rows for OneClassL2 loss")
        );
    }

    #[test]
    fn test_fit_eval_extra_columns_one_class_l2() {
        let data = generate_data();
        let result = Model::params()
            .loss(Loss::OneClassL2)
            .quiet(false)
            .fit_eval(&data, &[Node(1, 1000000, 1.0)]);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("eval set cannot have extra columns for OneClassL2 loss")
        );
    }

    #[test]
    fn test_cv() {
        let data = generate_data();
        let avg_error = Model::params().cv(&data, 5).unwrap();
        // not enough data
        assert!(avg_error.is_nan());
    }

    #[test]
    fn test_negative_row_index() {
        let result = Model::params().fit(&[Node(-1, 0, 1.0)]);
        assert_eq!(result.unwrap_err(), Error::Node(0));
    }

    #[test]
    fn test_max_row_index() {
        let result = Model::params().fit(&[Node(i32::MAX, 0, 1.0)]);
        assert_eq!(result.unwrap_err(), Error::Node(0));
    }

    #[test]
    fn test_negative_column_index() {
        let result = Model::params().fit(&[Node(0, -1, 1.0)]);
        assert_eq!(result.unwrap_err(), Error::Node(0));
    }

    #[test]
    fn test_max_column_index() {
        let result = Model::params().fit(&[Node(0, i32::MAX, 1.0)]);
        assert_eq!(result.unwrap_err(), Error::Node(0));
    }

    #[test]
    fn test_loss() {
        let data = generate_data();
        let model = Model::params().loss(Loss::OneClassL2).fit(&data).unwrap();
        assert_eq!(model.bias(), 0.0);
    }

    #[test]
    fn test_loss_real_kl() {
        let data = generate_data();
        assert!(Model::params()
            .loss(Loss::RealKL)
            .nmf(true)
            .fit(&data)
            .is_ok());
    }

    #[test]
    fn test_save_load() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();

        let path = c"target/model.txt";
        model.save(path).unwrap();
        let model = Model::load(path).unwrap();

        model.p_factors();
        model.q_factors();
        model.bias();
    }

    #[test]
    fn test_save_missing() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();
        let result = model.save(c"missing/model.txt");
        assert_eq!(result.unwrap_err(), Error::Io);
    }

    #[test]
    fn test_load_missing() {
        let result = Model::load(c"missing.txt");
        assert_eq!(result.unwrap_err(), Error::Io);
    }

    #[test]
    fn test_metrics() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();

        assert!(model.rmse(&data).unwrap() < 0.15);
        assert!(model.mae(&data).unwrap() < 0.15);
        assert!(model.gkl(&data).unwrap() < 0.01);
        assert!(model.logloss(&data).unwrap() < 0.3);
        assert_eq!(1.0, model.accuracy(&data).unwrap());
        assert_eq!(0.0, model.mpr(&data, false).unwrap());
        assert_eq!(1.0, model.auc(&data, false).unwrap());
    }

    #[test]
    fn test_metrics_empty() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();

        assert_eq!(0.0, model.rmse(&[]).unwrap());
        assert_eq!(0.0, model.mae(&[]).unwrap());
        assert_eq!(0.0, model.gkl(&[]).unwrap());
        assert_eq!(0.0, model.logloss(&[]).unwrap());
        assert_eq!(0.0, model.accuracy(&[]).unwrap());
        assert!(model.mpr(&[], false).unwrap().is_nan());
        assert!(model.auc(&[], false).unwrap().is_nan());
    }

    #[test]
    fn test_predict_out_of_range() {
        let data = generate_data();
        let model = Model::params().fit(&data).unwrap();
        assert_eq!(model.bias(), model.predict(1000, 1000));
    }

    #[test]
    fn test_fit_empty() {
        let result = Model::params().fit(&[]);
        assert_eq!(result.unwrap_err(), Error::Parameter("no data"));
    }

    #[test]
    fn test_fit_eval_empty() {
        let result = Model::params().fit_eval(&[], &[]);
        assert_eq!(result.unwrap_err(), Error::Parameter("no data"));
    }

    #[test]
    fn test_cv_empty() {
        let result = Model::params().cv(&[], 5);
        assert_eq!(result.unwrap_err(), Error::Parameter("no data"));
    }

    #[test]
    fn test_fit_bad_params() {
        let data = generate_data();
        let result = Model::params().factors(0).fit(&data);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("number of factors must be greater than zero")
        );
    }

    #[test]
    fn test_fit_eval_bad_params() {
        let data = generate_data();
        let result = Model::params().factors(0).fit_eval(&data, &data);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("number of factors must be greater than zero")
        );
    }

    #[test]
    fn test_cv_bad_params() {
        let data = generate_data();
        let result = Model::params().factors(0).cv(&data, 5);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("number of factors must be greater than zero")
        );
    }
}
