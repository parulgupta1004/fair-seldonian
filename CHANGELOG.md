# Changelog

<!-- towncrier release notes start -->

## 0.1.0 — 2020-01-01

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
