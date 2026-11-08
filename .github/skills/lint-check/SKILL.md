---
name: lint-check
description: "Run all code quality checks: black formatting, flake8 linting, and mypy strict type checking across infinite_maze/ and training/. Use when: finishing a code change, before committing or opening a PR, or checking CI parity for formatting, lint or type errors."
---
Run the following three code quality checks in order against the `infinite_maze/` and `training/` packages (the same targets CI uses in `.github/workflows/quality.yml`). Run them through Poetry so tool versions match CI. Report results for each step before moving to the next, and stop if any step produces errors.

If `training/` does not exist on the current branch, run the checks against `infinite_maze/` only.

## Step 1 — Formatting (black)

```
poetry run black --check infinite_maze/ training/
```

If black reports files that would be reformatted, run:

```
poetry run black infinite_maze/ training/
```

and list the files that were changed. Black is only installed on Python 3.10 or higher (see `pyproject.toml`); if it is missing, say so instead of skipping silently.

## Step 2 — Linting (flake8)

```
poetry run flake8 infinite_maze/ training/
```

List every violation grouped by file. The project config ignores E203 and W503; do not suppress any other codes.

## Step 3 — Type checking (mypy)

```
poetry run mypy infinite_maze/ training/
```

The project uses strict settings (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, etc. — all defined in `pyproject.toml`). List every error with file, line, and error code.

## Summary

After all three steps, produce a table:

| Check | Status | Issues |
|---|---|---|
| black | ✅ / ❌ | n files reformatted / already formatted |
| flake8 | ✅ / ❌ | n violations |
| mypy | ✅ / ❌ | n errors |

If there are any failures, ask whether to fix them before finishing.
