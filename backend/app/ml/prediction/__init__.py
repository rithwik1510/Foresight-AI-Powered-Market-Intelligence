"""
Prediction Module
Contains all ML prediction models for stock forecasting
"""
from app.ml.prediction.base_predictor import (
    BasePredictor,
    PredictionResult,
    ModelMetrics,
    Direction
)
from app.ml.prediction.arima_predictor import ARIMAPredictor, arima_predictor
from app.ml.prediction.prophet_predictor import ProphetPredictor, prophet_predictor
from app.ml.prediction.xgboost_predictor import XGBoostPredictor, xgboost_predictor
from app.ml.prediction.lightgbm_predictor import LightGBMPredictor, lightgbm_predictor
from app.ml.prediction.random_forest_predictor import RandomForestPredictor, random_forest_predictor
from app.ml.prediction.ensemble_predictor import (
    EnsemblePredictor,
    EnsemblePrediction,
    ensemble_predictor
)

__all__ = [
    # Base classes
    "BasePredictor",
    "PredictionResult",
    "ModelMetrics",
    "Direction",
    # Predictor classes
    "ARIMAPredictor",
    "ProphetPredictor",
    "XGBoostPredictor",
    "LightGBMPredictor",
    "RandomForestPredictor",
    # Ensemble
    "EnsemblePredictor",
    "EnsemblePrediction",
    # Global instances
    "arima_predictor",
    "prophet_predictor",
    "xgboost_predictor",
    "lightgbm_predictor",
    "random_forest_predictor",
    "ensemble_predictor",
]
