use crate::bindings::*;
use crate::{Error, Loss, Matrix, Model};

pub struct Params {
    param: MfParameter
}

impl Params {
    pub(crate) fn new() -> Self {
        let mut param = unsafe { mf_get_default_param() };
        param.nr_bins = 25;
        Self {
            param
        }
    }

    pub fn loss(&mut self, value: Loss) -> &mut Self {
        self.param.fun = value;
        self
    }

    pub fn factors(&mut self, value: i32) -> &mut Self {
        self.param.k = value;
        self
    }

    pub fn threads(&mut self, value: i32) -> &mut Self {
        self.param.nr_threads = value;
        self
    }

    pub fn bins(&mut self, value: i32) -> &mut Self {
        self.param.nr_bins = value;
        self
    }

    pub fn iterations(&mut self, value: i32) -> &mut Self {
        self.param.nr_iters = value;
        self
    }

    pub fn lambda_p1(&mut self, value: f32) -> &mut Self {
        self.param.lambda_p1 = value;
        self
    }

    pub fn lambda_p2(&mut self, value: f32) -> &mut Self {
        self.param.lambda_p2 = value;
        self
    }

    pub fn lambda_q1(&mut self, value: f32) -> &mut Self {
        self.param.lambda_q1 = value;
        self
    }

    pub fn lambda_q2(&mut self, value: f32) -> &mut Self {
        self.param.lambda_q2 = value;
        self
    }

    pub fn learning_rate(&mut self, value: f32) -> &mut Self {
        self.param.eta = value;
        self
    }

    pub fn alpha(&mut self, value: f32) -> &mut Self {
        self.param.alpha = value;
        self
    }

    pub fn c(&mut self, value: f32) -> &mut Self {
        self.param.c = value;
        self
    }

    pub fn nmf(&mut self, value: bool) -> &mut Self {
        self.param.do_nmf = value;
        self
    }

    pub fn quiet(&mut self, value: bool) -> &mut Self {
        self.param.quiet = value;
        self
    }

    pub fn fit(&mut self, data: &Matrix) -> Result<Model, Error> {
        let prob = data.to_problem();
        let model = unsafe { mf_train(&prob, self.build_param()?) };
        if model.is_null() {
            Err(Error("fit failed".to_string()))
        } else {
            Ok(Model { model })
        }
    }

    pub fn fit_eval(&mut self, train_set: &Matrix, eval_set: &Matrix) -> Result<Model, Error> {
        let tr = train_set.to_problem();
        let va = eval_set.to_problem();
        let model = unsafe { mf_train_with_validation(&tr, &va, self.build_param()?) };
        if model.is_null() {
            Err(Error("fit_eval failed".to_string()))
        } else {
            Ok(Model { model })
        }
    }

    pub fn cv(&mut self, data: &Matrix, folds: i32) -> Result<f64, Error> {
        let prob = data.to_problem();
        let avg_error = unsafe { mf_cross_validation(&prob, folds, self.build_param()?) };
        // TODO update fork to differentiate between bad parameters and zero error
        if avg_error == 0.0 {
            Err(Error("cv failed".to_string()))
        } else {
            Ok(avg_error)
        }
    }

    // check parameters in Rust for better error message
    fn build_param(&self) -> Result<MfParameter, Error> {
        let param = self.param;

        if param.k < 1 {
            return Err(Error("number of factors must be greater than zero".to_string()));
        }

        if param.nr_threads < 1 {
            return Err(Error("number of threads must be greater than zero".to_string()));
        }

        if param.nr_bins < 1 || param.nr_bins < param.nr_threads {
            return Err(Error("number of bins must be greater than number of threads".to_string()));
        }

        if param.nr_iters < 1 {
            return Err(Error("number of iterations must be greater than zero".to_string()));
        }

        if param.lambda_p1 < 0.0 || param.lambda_p2 < 0.0 || param.lambda_q1 < 0.0 || param.lambda_q2 < 0.0 {
            return Err(Error("regularization coefficient must be non-negative".to_string()));
        }

        if param.eta <= 0.0 {
            return Err(Error("learning rate must be greater than zero".to_string()));
        }

        if matches!(param.fun, Loss::RealKL) && !param.do_nmf {
            return Err(Error("nmf must be set when using generalized KL-divergence".to_string()));
        }

        if param.alpha < 0.0 {
            return Err(Error("alpha must be a non-negative number".to_string()));
        }

        Ok(param)
    }
}
