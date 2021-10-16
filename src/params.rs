use crate::bindings::*;
use crate::{Matrix, Model};

pub struct Params {
    param: MfParameter,
}

impl Params {
    pub(crate) fn new() -> Self {
        let mut param = unsafe { mf_get_default_param() };
        param.nr_bins = 25;
        Self {
            param,
        }
    }

    // TODO use enum
    pub fn loss(&mut self, value: i32) -> &mut Self {
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

    pub fn fit(&mut self, data: &Matrix) -> Model {
        let prob = data.to_problem();
        Model {
            model: unsafe { mf_train(&prob, self.param) },
        }
    }

    pub fn fit_eval(&mut self, train_set: &Matrix, eval_set: &Matrix) -> Model {
        let tr = train_set.to_problem();
        let va = eval_set.to_problem();
        Model {
            model: unsafe { mf_train_with_validation(&tr, &va, self.param) },
        }
    }

    pub fn cv(&mut self, data: &Matrix, folds: i32) {
        let prob = data.to_problem();
        unsafe { mf_cross_validation(&prob, folds, self.param); }
    }
}
