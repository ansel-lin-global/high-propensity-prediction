# Task runner — run `just --list` to see all recipes.

default:
    @just --list

# --- Environment ---
sync:
    uv sync --all-extras

# --- Quality ---
lint:
    uv run ruff check .

lint-fix:
    uv run ruff check --fix .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

test:
    uv run pytest

# Full pre-push gate
check: lint fmt-check test

# --- Git hooks ---
hooks:
    uv run pre-commit install

hooks-run:
    uv run pre-commit run --all-files

# --- Pipeline recipes ---
compile:
    uv run python scripts/compile_and_package.py

compile-only pipeline:
    uv run python scripts/compile_and_package.py --only {{pipeline}}

submit *args:
    uv run python scripts/submit_pipeline_job.py {{args}}
