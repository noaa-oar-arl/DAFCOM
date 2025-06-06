"""
Data validation utilities for DAFCOM

This module provides utilities for validating input and output data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple


def validate_data(data: Union[pd.DataFrame, np.ndarray],
                 expectations: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a dataset meets expectations.

    Args:
        data: Data to validate (DataFrame or ndarray)
        expectations: Dictionary of validation rules

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for missing values
    if expectations.get('no_missing_values', False):
        if isinstance(data, pd.DataFrame):
            if data.isnull().any().any():
                return False, "Dataset contains missing values"
        elif isinstance(data, np.ndarray):
            if np.isnan(data).any():
                return False, "Dataset contains NaN values"

    # Check shape constraints if specified
    if 'shape' in expectations:
        expected_shape = expectations['shape']
        if isinstance(expected_shape, dict):
            # Can specify constraints like {'min_rows': 100, 'cols': 5}
            if isinstance(data, pd.DataFrame):
                rows, cols = data.shape
                if 'min_rows' in expected_shape and rows < expected_shape['min_rows']:
                    return False, f"DataFrame has {rows} rows, expected at least {expected_shape['min_rows']}"
                if 'max_rows' in expected_shape and rows > expected_shape['max_rows']:
                    return False, f"DataFrame has {rows} rows, expected at most {expected_shape['max_rows']}"
                if 'cols' in expected_shape and cols != expected_shape['cols']:
                    return False, f"DataFrame has {cols} columns, expected {expected_shape['cols']}"
            # Similar checks for numpy arrays
        elif isinstance(expected_shape, (list, tuple)):
            # Direct shape comparison
            if data.shape != tuple(expected_shape):
                return False, f"Data has shape {data.shape}, expected {expected_shape}"

    # Check column existence for DataFrames
    if isinstance(data, pd.DataFrame) and 'required_columns' in expectations:
        missing_cols = [col for col in expectations['required_columns'] if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"

    # All checks passed
    return True, None
