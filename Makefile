# DAFCOM Makefile

.PHONY: test test-coverage docs clean install dev-install lint format

# Default target
all: test docs

# Install the package
install:
	pip install -e .

# Install with development dependencies
dev-install:
	pip install -e ".[dev]"

# Run tests
test:
	python run_tests.py

# Run tests with coverage
test-coverage:
	python run_tests.py --pytest

# Build documentation
docs:
	cd docs && bash build_docs.sh

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint the code
lint:
	flake8 src/ tests/

# Format the code
format:
	black src/ tests/
	isort src/ tests/

# Run type checking
type-check:
	mypy src/
