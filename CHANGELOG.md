## 0.3.0 (unreleased)

- Removed dependency
- Updated Rust edition to 2021

## 0.2.2 (2022-09-23)

- Added `p` and `q`
- Added `p_iter` and `q_iter`

## 0.2.1 (2021-11-15)

- Added `Error` trait to errors
- Added support for paths to `save` and `load`

## 0.2.0 (2021-10-17)

- Added support for Windows

Breaking changes

- Changed pattern for fitting models - use `Model::params()` instead of `Model::new()`
- Changed `fit`, `fit_eval`, `cv`, `save`, and `load` to return `Result`
- Changed `cv` to return average error
- Changed `loss` to use enum

## 0.1.1 (2021-07-27)

- Added more metrics

## 0.1.0 (2021-07-26)

- First release
