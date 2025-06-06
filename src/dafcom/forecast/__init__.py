"""
DAFCOM Forecast Module

This module provides forecasting capabilities including:
- Unified data processing workflow
- ML model training and inference
- Bias correction for atmospheric models
"""

from .processor import UnifiedTrainingDataProcessor

__all__ = ["UnifiedTrainingDataProcessor"]
