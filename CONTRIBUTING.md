# Contributing to Fair-Seldonian

Thanks for your interest in contributing! Here's how to get started.

This project has a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold it.

## Setup

```bash
git clone https://github.com/parulgupta1004/fair-seldonian.git
cd fair-seldonian
uv sync --extra dev
pre-commit install               # enable formatting/linting on commit
```

## Code style

Formatting and linting are handled by [pre-commit](https://pre-commit.com/)
hooks (isort, ruff, and ruff-format). Once you've run `pre-commit install`,
they run automatically on each commit. To check everything manually:

```bash
uv run pre-commit run --all-files
```

CI runs the same checks, so running them locally keeps your pull request green.

## Running tests

```bash
uv run pytest
```

## Making changes

1. Fork the repository and create a branch from `master`.
2. Make your changes.
3. Add tests for any new functionality.
4. Run `uv run pytest` and make sure all tests pass.
5. Open a pull request.

## Changelog

This project uses [towncrier](https://towncrier.readthedocs.io/) for changelog management. When your change is user-facing, add a fragment in the `changes/` directory. Name each file `<short-slug>.<type>.md`, where `<type>` is one of `added`, `changed`, `fixed`, or `removed`:

```bash
# For new features
echo "Short description of the change." > changes/my-feature.added.md

# For bug fixes
echo "Short description of the fix." > changes/some-bug.fixed.md

# For other changes: .changed.md, .removed.md
```

(An issue or PR number also works as the slug, e.g. `changes/123.added.md`.)

## Ideas to get you started

All contributions are valued — here are a few ideas, but don't let this list limit you:

- New fairness constraints and inequality types
- Additional model support beyond logistic regression
- Documentation improvements
- Performance optimizations
- Bug reports and test coverage

## Reporting issues

Open an [issue](https://github.com/parulgupta1004/fair-seldonian/issues) with a clear description and, if applicable, a minimal reproducible example.
