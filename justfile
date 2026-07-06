# diffBloch developer commands.  Run `just --list` to see all.
set shell := ["bash", "-cu"]

# Install dependencies + git hooks (run once per clone)
install:
    uv sync --dev
    uv run pre-commit install

# Fast unit tests (excludes e2e)
test:
    uv run pytest

# End-to-end characterization tests (opt-in; the physics anchors)
test-e2e:
    uv run pytest -m e2e

# Quartz anchors (north star): static + integrated pins over all 99 rotations (~5 min total).
# For a quick sanity subset of the integrated anchor (unrepresentative mean, sanity-only):
#   DIFFBLOCH_ANCHOR_ROTATIONS=5 just anchor
anchor:
    uv run pytest -m e2e -k anchor

# Every test, including e2e
test-all:
    uv run pytest -m ""

# Unit tests with a coverage report
cov:
    uv run pytest --cov=diffBloch --cov-report=term-missing

# Lint + format check (no changes)
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint + format
format:
    uv run ruff check --fix .
    uv run ruff format .

# Static type check (source only)
typecheck:
    uv run mypy src/diffBloch

# Build API docs from docstrings/signatures (fails on warnings)
docs:
    uv run mkdocs build --strict

# Serve docs locally with live reload
docs-serve:
    uv run mkdocs serve

# The pre-push gate: lint, types, unit tests
check: lint typecheck test
