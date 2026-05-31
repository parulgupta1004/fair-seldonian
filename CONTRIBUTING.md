# Contributing to Fair-Seldonian

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/parulgupta1004/fair-seldonian.git
cd fair-seldonian
uv sync --extra dev
```

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

This project uses [towncrier](https://towncrier.readthedocs.io/) for changelog management. When your change is user-facing, add a fragment in the `changes/` directory:

```bash
# For new features
echo "Short description of the change." > changes/<issue-or-pr-number>.added

# For bug fixes
echo "Short description of the fix." > changes/<issue-or-pr-number>.fixed

# For other changes
# .changed, .removed
```

## Ideas to get you started

All contributions are valued — here are a few ideas, but don't let this list limit you:

- New fairness constraints and inequality types
- Additional model support beyond logistic regression
- Documentation improvements
- Performance optimizations
- Bug reports and test coverage

## Reporting issues

Open an [issue](https://github.com/parulgupta1004/fair-seldonian/issues) with a clear description and, if applicable, a minimal reproducible example.
