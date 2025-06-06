# DAFCOM Tests

This directory contains unit tests for the DAFCOM package.

## Structure

The test directory structure mirrors the package structure:

```
tests/
├── __init__.py
├── forecast/
│   ├── __init__.py
│   ├── test_xgboost_model.py
│   └── test_two_stage_pipeline.py
└── utils/
    ├── __init__.py
    └── ...
```

## Running Tests

You can run the tests using the `run_tests.py` script in the project root:

```bash
# Run with unittest
python run_tests.py

# Run with pytest and generate coverage reports
python run_tests.py --pytest
```

Alternatively, you can run pytest directly:

```bash
pytest -xvs tests/
```

## Adding New Tests

When adding new functionality to the package, please add corresponding tests. Tests should:

1. Test both normal functionality and error cases
2. Include docstrings explaining what is being tested
3. Be organized in a way that mirrors the package structure

## Code Coverage

Code coverage reports are generated when running tests with the `--pytest` flag. The reports can be found in:

- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`
