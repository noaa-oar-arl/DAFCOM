#!/usr/bin/env python3
"""
Test runner for the DAFCOM package.

This script runs all unit tests for the DAFCOM package and reports the results.
It can be run with pytest or directly as a Python script.
"""

import unittest
import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests_with_unittest():
    """Run tests using the unittest framework."""
    print("\n" + "="*80)
    print("Running DAFCOM unit tests with unittest")
    print("="*80)

    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1


def run_tests_with_pytest():
    """Run tests using pytest with coverage reporting."""
    print("\n" + "="*80)
    print("Running DAFCOM unit tests with pytest and coverage")
    print("="*80)

    # Run pytest with coverage
    args = [
        '-xvs',  # x: stop on first failure, v: verbose, s: don't capture stdout
        '--cov=src/dafcom',  # coverage for dafcom package
        '--cov-report=term',  # terminal report
        '--cov-report=html:./htmlcov',  # HTML report
        '--cov-report=xml:./coverage.xml',  # XML report for CI integration
        'tests'  # test directory
    ]
    return pytest.main(args)


if __name__ == '__main__':
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        sys.exit(run_tests_with_pytest())
    else:
        sys.exit(run_tests_with_unittest())
