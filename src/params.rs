use crate::bindings::*;
use crate::{Error, Loss, Model, Node};

/// A set of parameters.
pub struct Params {
    param: MfParameter,
}

impl Params {
    pub(crate) fn new() -> Self {
        let mut param = unsafe { mf_get_default_param() };
        param.nr_bins = 25;
        param.quiet = true;
        Self { param }
    }

    /// Sets the loss function.
    pub fn loss(&mut self, value: Loss) -> &mut Self {
        self.param.fun = value;
        self
    }

    /// Sets the number of latent factors.
    pub fn factors(&mut self, value: i32) -> &mut Self {
        self.param.k = value;
        self
    }

    /// Sets the number of threads.
    pub fn threads(&mut self, value: i32) -> &mut Self {
        self.param.nr_threads = value;
        self
    }

    /// Sets the number of bins.
    pub fn bins(&mut self, value: i32) -> &mut Self {
        self.param.nr_bins = value;
        self
    }

    /// Sets the number of iterations.
    pub fn iterations(&mut self, value: i32) -> &mut Self {
        self.param.nr_iters = value;
        self
    }

    /// Sets the L1-regularization parameter for P.
    pub fn lambda_p1(&mut self, value: f32) -> &mut Self {
        self.param.lambda_p1 = value;
        self
    }

    /// Sets the L2-regularization parameter for P.
    pub fn lambda_p2(&mut self, value: f32) -> &mut Self {
        self.param.lambda_p2 = value;
        self
    }

    /// Sets the L1-regularization parameter for Q.
    pub fn lambda_q1(&mut self, value: f32) -> &mut Self {
        self.param.lambda_q1 = value;
        self
    }

    /// Sets the L2-regularization parameter for Q.
    pub fn lambda_q2(&mut self, value: f32) -> &mut Self {
        self.param.lambda_q2 = value;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(&mut self, value: f32) -> &mut Self {
        self.param.eta = value;
        self
    }

    /// Sets the importance of negative entries.
    pub fn alpha(&mut self, value: f32) -> &mut Self {
        self.param.alpha = value;
        self
    }

    /// Sets the desired value of negative entries.
    pub fn c(&mut self, value: f32) -> &mut Self {
        self.param.c = value;
        self
    }

    /// Sets whether to perform non-negative MF (NMF).
    pub fn nmf(&mut self, value: bool) -> &mut Self {
        self.param.do_nmf = value;
        self
    }

    /// Sets whether to output to stdout.
    pub fn quiet(&mut self, value: bool) -> &mut Self {
        self.param.quiet = value;
        self
    }

    /// Fits a model.
    pub fn fit(&self, data: &[Node]) -> Result<Model, Error> {
        // prevent "posix_memalign() invalid size value: 0" with Valgrind
        if data.is_empty() {
            return Err(Error::Parameter("no data"));
        }

        let prob = data.try_into()?;
        let param = self.build_param()?;
        let model = unsafe { mf_train(&prob, param) };
        if model.is_null() {
            return Err(Error::Unknown);
        }
        Ok(Model { model })
    }

    /// Fits a model and performs cross-validation.
    pub fn fit_eval(&self, train_set: &[Node], eval_set: &[Node]) -> Result<Model, Error> {
        // prevent "posix_memalign() invalid size value: 0" with Valgrind
        if train_set.is_empty() || eval_set.is_empty() {
            return Err(Error::Parameter("no data"));
        }

        let tr = train_set.try_into()?;
        let va = eval_set.try_into()?;
        let param = self.build_param()?;
        let model = unsafe { mf_train_with_validation(&tr, &va, param) };
        if model.is_null() {
            return Err(Error::Unknown);
        }
        Ok(Model { model })
    }

    /// Performs cross-validation.
    pub fn cv(&self, data: &[Node], folds: i32) -> Result<f64, Error> {
        // prevent "posix_memalign() invalid size value: 0" with Valgrind
        if data.is_empty() {
            return Err(Error::Parameter("no data"));
        }

        let prob = data.try_into()?;
        let param = self.build_param()?;
        let avg_error = unsafe { mf_cross_validation(&prob, folds, param) };
        // TODO update fork to differentiate between bad parameters and zero error
        if avg_error == 0.0 {
            return Err(Error::Unknown);
        }
        Ok(avg_error)
    }

    // check parameters in Rust for better error message
    fn build_param(&self) -> Result<MfParameter, Error> {
        let param = self.param;

        if param.k < 1 {
            return Err(Error::Parameter(
                "number of factors must be greater than zero",
            ));
        }

        if param.nr_threads < 1 {
            return Err(Error::Parameter(
                "number of threads must be greater than zero",
            ));
        }

        if param.nr_bins < 1 || param.nr_bins < param.nr_threads {
            return Err(Error::Parameter(
                "number of bins must be greater than number of threads",
            ));
        }

        if param.nr_iters < 1 {
            return Err(Error::Parameter(
                "number of iterations must be greater than zero",
            ));
        }

        if param.lambda_p1 < 0.0
            || param.lambda_p2 < 0.0
            || param.lambda_q1 < 0.0
            || param.lambda_q2 < 0.0
        {
            return Err(Error::Parameter(
                "regularization coefficient must be non-negative",
            ));
        }

        if param.eta <= 0.0 {
            return Err(Error::Parameter("learning rate must be greater than zero"));
        }

        if matches!(param.fun, Loss::RealKL) && !param.do_nmf {
            return Err(Error::Parameter(
                "nmf must be set when using generalized KL-divergence",
            ));
        }

        if param.alpha < 0.0 {
            return Err(Error::Parameter("alpha must be a non-negative number"));
        }

        Ok(param)
    }
}
