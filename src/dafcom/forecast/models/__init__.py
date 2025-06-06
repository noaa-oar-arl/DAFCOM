"""
DAFCOM ML Models Package

This package provides time series forecasting models:
1. LSTM-based models
2. Transformer-based models
3. XGBoost models for bias correction
4. Two-stage pipeline combining sequential models with XGBoost

These models integrate with the unified training processor to provide
a complete pipeline from data processing to model training.
"""

from .model_factory import create_model, LSTMModel, TransformerModel, TimeSeriesModel
from .time_series_data import TimeSeriesDataPreparation
from .training import train_model, evaluate_model
from .xgboost_model import XGBoostModel, train_xgboost_after_lstm, apply_xgboost_correction
from .two_stage_pipeline import TwoStageForecastPipeline, train_two_stage_model, apply_two_stage_model

__all__ = [
    'create_model',
    'LSTMModel',
    'TransformerModel',
    'TimeSeriesModel',
    'TimeSeriesDataPreparation',
    'train_model',
    'evaluate_model',
    'XGBoostModel',
    'train_xgboost_after_lstm',
    'apply_xgboost_correction',
    'TwoStageForecastPipeline',
    'train_two_stage_model',
    'apply_two_stage_model'
]
