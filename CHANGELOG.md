## 0.2.1 (unreleased)

- Added `len` function to `Matrix`

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
