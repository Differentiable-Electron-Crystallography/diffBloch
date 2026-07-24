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

# Checkpointed quartz (seconds, read-only): score the committed frozen coupled checkpoint -> 0.0506.
# Fails FAST if the committed plan.lock is stale for the current recipe/config/code; regenerate via
# `just verify-quartz-full` then `just promote-quartz`. Never mutates the committed fixture.
verify-quartz:
    uv run pytest tests/e2e/test_anchor.py::test_quartz_coupled_anchor -m e2e

# Quartz with full preprocess (~6-16 min): from-scratch coupled fit into the gitignored stash
# tests/fixtures/quartz_anchor/.candidate/ (the committed reference is never touched), streaming
# per-rotation fit progress live; compares the stash run to the committed 0.0506.
verify-quartz-full:
    DIFFBLOCH_ANCHOR_FULL=1 uv run pytest tests/e2e/test_anchor.py::test_quartz_coupled_anchor_full -m e2e -s --log-cli-level=INFO

# Promote the stashed run to be the committed reference checkpoint -- the ONLY path that mutates it.
# Review the resulting git diff, then commit.
promote-quartz:
    @test -f tests/fixtures/quartz_anchor/.candidate/plan.npz -a -f tests/fixtures/quartz_anchor/.candidate/plan.lock || { echo "no stashed run -- run 'just verify-quartz-full' first" >&2; exit 1; }
    cp tests/fixtures/quartz_anchor/.candidate/plan.npz tests/fixtures/quartz_anchor/plan.npz
    cp tests/fixtures/quartz_anchor/.candidate/plan.lock tests/fixtures/quartz_anchor/plan.lock
    git status --short tests/fixtures/quartz_anchor/

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

# Build the docs (Sphinx + furo; fails on warnings)
docs:
    uv run sphinx-build -W -b html docs docs/_build/html

# Serve docs locally with live reload
docs-serve:
    uv run sphinx-autobuild docs docs/_build/html

# The pre-push gate: lint, types, unit tests
check: lint typecheck test
