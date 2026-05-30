# Changelog

<!-- towncrier release notes start -->

## 2.0.0

### Added

- CI test matrix across Python 3.10–3.13 with lint and coverage reporting. (ci-matrix)
- CircleCI configuration with lint, test (Python 3.10–3.13), and changelog check jobs. (circleci)
- Test coverage reporting with pytest-cov and Codecov integration. (coverage)
- Dependabot for monthly dependency and GitHub Actions updates. (dependabot)
- Automated Sphinx documentation deployment to GitHub Pages. (docs-deploy)
- `py.typed` marker for PEP 561 downstream type checking support. (py-typed)
- Pyrefly type checker integration with zero errors across the codebase. (pyrefly)
- Auto-release pipeline: prepare-release workflow, tag-triggered PyPI publish via trusted publishing, and GitHub Release with assets. (release-pipeline)
- `SeldonianConfig` dataclass replacing hardcoded module-level globals. (seldonian-config)
- Support for Python 3.14 and free-threaded (no-GIL) Python 3.14t with CI testing via deadsnakes PPA. Uses environment markers to require higher dependency floors on 3.14+ (numpy>=2.3.2, pandas>=2.3.3, scikit-learn>=1.8.0, scipy>=1.16.1, torch>=2.9.0) while preserving existing floors for 3.10–3.13. Added coverage>=7.6.1 to dev extras for free-threaded support. (python314)

### Changed

- Applied ruff formatting and isort across all source and test files. (ruff-formatting)
- Renamed public API to PEP 8: `fHat`→`f_hat`, `expr_tree`→`ExprTree`, `isOperator`→`is_operator`, `isFunc`→`is_func`, `isMod`→`is_mod`, `isConstant`→`is_constant`. Also renamed internal functions: `genFilename`→`gen_filename`, `addMoreResults`→`add_more_results`, `saveToCSV`→`save_to_csv`, `loadAndPlotResults`→`load_and_plot_results`, and camelCase variables (`newFileId`, `newFile`, `nCols`, `resultsQSA`, `resultsLS`, `fileName`) to snake_case. (pep8-naming)

### Fixed

- Eliminated implicit `None` returns in `simple_logistic`, `eval_ghat`, and `ghat` for proper error propagation. (type-safety)


## 1.0.0

### Added

- Pyrefly type checker integration with zero errors across the codebase. (pyrefly)

### Fixed

- Eliminated implicit `None` returns in `simple_logistic`, `eval_ghat`, and `ghat` for proper error propagation. (type-safety)


## 0.1.0

Initial release.

### Added
- Quasi-Seldonian Algorithm (QSA) with candidate selection and safety test
- Five algorithm variants: `base`, `mod`, `const`, `bound`, `opt`
- Expression tree parser for constraint strings in reverse Polish notation
- Interval arithmetic for confidence bound propagation
- Hoeffding and Student's t-test concentration inequalities
- Constant-aware delta allocation
- Union bound optimization for repeated variables
- Decomposed candidate-safety interval estimation
- Logistic regression model with sigmoid predict, log-loss objective
- Synthetic data generation with configurable group ratios
- Parallel experiment runner (Ray)
- Results aggregation and plotting
- Sphinx documentation with mathematical notation and API reference
- CI workflow (GitHub Actions)
- Pre-commit hooks (isort, ruff)
