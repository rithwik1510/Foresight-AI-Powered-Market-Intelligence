"""
Base Predictor - Abstract base class for all ML prediction models
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from app.ml.config import Direction, RiskLevel, PredictionHorizon


@dataclass
class PredictionResult:
    """
    Standardized prediction result from any model

    All models return predictions in this format for easy ensembling
    """
    # Core prediction
    symbol: str
    model_name: str
    prediction_date: datetime
    horizon_days: int

    # Direction prediction
    direction: Direction
    direction_probability: float  # 0.0 to 1.0 for the predicted direction

    # Price prediction
    current_price: float
    predicted_price: float
    predicted_return: float  # Percentage return

    # Confidence
    confidence: float  # 0.0 to 1.0

    # Price range (uncertainty bounds)
    price_lower: Optional[float] = None
    price_upper: Optional[float] = None

    # Additional info
    feature_importance: Optional[Dict[str, float]] = None
    raw_output: Optional[Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "prediction_date": self.prediction_date.isoformat(),
            "horizon_days": self.horizon_days,
            "direction": self.direction.value,
            "direction_probability": round(self.direction_probability, 4),
            "current_price": round(self.current_price, 2),
            "predicted_price": round(self.predicted_price, 2),
            "predicted_return": round(self.predicted_return, 4),
            "confidence": round(self.confidence, 4),
            "price_lower": round(self.price_lower, 2) if self.price_lower else None,
            "price_upper": round(self.price_upper, 2) if self.price_upper else None,
        }


@dataclass
class ModelMetrics:
    """
    Performance metrics for a trained model
    """
    model_name: str
    symbol: Optional[str]  # None for global metrics
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Directional accuracy
    directional_accuracy: float = 0.0

    # Regression metrics
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    mape: float = 0.0  # Mean Absolute Percentage Error

    # Financial metrics
    hit_rate: float = 0.0  # % of correct direction predictions
    profit_factor: float = 0.0  # Gross profit / Gross loss
    sharpe_ratio: float = 0.0

    # Sample info
    n_train_samples: int = 0
    n_test_samples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "train_period": f"{self.train_start.date()} to {self.train_end.date()}",
            "test_period": f"{self.test_start.date()} to {self.test_end.date()}",
            "accuracy": round(self.accuracy, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "hit_rate": round(self.hit_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "n_samples": {"train": self.n_train_samples, "test": self.n_test_samples},
        }


class BasePredictor(ABC):
    """
    Abstract base class for all prediction models

    All prediction models must implement:
    - train(): Train the model on historical data
    - predict(): Generate predictions for new data
    - get_feature_importance(): Return feature importances
    - save() / load(): Persist and load trained models
    """

    def __init__(self, model_name: str):
        """
        Initialize base predictor

        Args:
            model_name: Name of the model (e.g., 'arima', 'prophet', 'xgboost')
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.last_trained: Optional[datetime] = None
        self.metrics: Optional[ModelMetrics] = None
        self.feature_names: List[str] = []

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train the model on historical data

        Args:
            X: Feature DataFrame (rows = samples, columns = features)
            y: Target Series (forward returns or direction)
            validation_split: Fraction of data for validation

        Returns:
            ModelMetrics with training performance
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        current_price: float,
        symbol: str,
        horizon_days: int = 30
    ) -> PredictionResult:
        """
        Generate prediction for new data

        Args:
            X: Feature DataFrame (single row or recent history)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Prediction horizon in days

        Returns:
            PredictionResult with prediction details
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save trained model to disk

        Args:
            path: Path to save model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_dict = {
            "model": self.model,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
        }

        joblib.dump(save_dict, f"{model_path}.joblib")

    def load(self, path: Path) -> None:
        """
        Load trained model from disk

        Args:
            path: Path to model file (without extension)
        """
        model_path = Path(f"{path}.joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        save_dict = joblib.load(model_path)

        self.model = save_dict["model"]
        self.model_name = save_dict["model_name"]
        self.is_trained = save_dict["is_trained"]
        self.last_trained = save_dict["last_trained"]
        self.metrics = save_dict["metrics"]
        self.feature_names = save_dict["feature_names"]

    def _classify_direction(
        self,
        predicted_return: float,
        bullish_threshold: float = 0.02,
        bearish_threshold: float = -0.02
    ) -> Tuple[Direction, float]:
        """
        Classify prediction into direction with probability

        Args:
            predicted_return: Predicted return percentage (as decimal, e.g., 0.05 for 5%)
            bullish_threshold: Return threshold for bullish (default 2%)
            bearish_threshold: Return threshold for bearish (default -2%)

        Returns:
            Tuple of (Direction, probability)
        """
        if predicted_return > bullish_threshold:
            # Calculate probability based on distance from threshold
            prob = min(0.5 + (predicted_return - bullish_threshold) * 5, 0.95)
            return Direction.BULLISH, prob
        elif predicted_return < bearish_threshold:
            prob = min(0.5 + abs(predicted_return - bearish_threshold) * 5, 0.95)
            return Direction.BEARISH, prob
        else:
            # Neutral - probability based on how close to zero
            prob = 0.5 + (0.5 - abs(predicted_return) / max(abs(bullish_threshold), abs(bearish_threshold)) * 0.5)
            return Direction.NEUTRAL, min(prob, 0.7)

    def _calculate_confidence(
        self,
        direction_prob: float,
        model_accuracy: float = 0.5,
        prediction_std: Optional[float] = None
    ) -> float:
        """
        Calculate overall confidence score

        Args:
            direction_prob: Probability for predicted direction
            model_accuracy: Historical model accuracy
            prediction_std: Standard deviation of prediction (if available)

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from direction probability
        base_confidence = direction_prob

        # Adjust for model accuracy
        accuracy_factor = model_accuracy / 0.5  # Normalize around 50%
        accuracy_adjusted = base_confidence * min(accuracy_factor, 1.5)

        # Penalize high uncertainty
        if prediction_std is not None:
            uncertainty_penalty = max(0, 1 - prediction_std * 10)
            accuracy_adjusted *= uncertainty_penalty

        # Clamp between 0.1 and 0.95
        return max(0.1, min(0.95, accuracy_adjusted))

    def _prepare_features(
        self,
        X: pd.DataFrame,
        required_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for prediction

        Args:
            X: Raw feature DataFrame
            required_features: List of features required by model

        Returns:
            Prepared feature DataFrame
        """
        if required_features is None:
            required_features = self.feature_names

        # Select only required features
        available_features = [f for f in required_features if f in X.columns]
        X_prepared = X[available_features].copy()

        # Handle missing values
        X_prepared = X_prepared.ffill().bfill()

        # Replace infinite values
        X_prepared = X_prepared.replace([np.inf, -np.inf], np.nan)
        X_prepared = X_prepared.fillna(0)

        return X_prepared

    def _create_target(
        self,
        prices: pd.Series,
        horizon_days: int = 30,
        target_type: str = "return"
    ) -> pd.Series:
        """
        Create target variable for training

        Args:
            prices: Price series
            horizon_days: Forward look period
            target_type: 'return' for regression, 'direction' for classification

        Returns:
            Target series
        """
        # Forward return
        forward_return = prices.shift(-horizon_days) / prices - 1

        if target_type == "return":
            return forward_return
        elif target_type == "direction":
            # 0 = Bearish, 1 = Neutral, 2 = Bullish
            return pd.Series(
                np.where(
                    forward_return > 0.02, 2,
                    np.where(forward_return < -0.02, 0, 1)
                ),
                index=prices.index
            )
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    def needs_retraining(self, max_age_days: int = 7) -> bool:
        """
        Check if model needs retraining

        Args:
            max_age_days: Maximum age of model in days

        Returns:
            True if model needs retraining
        """
        if not self.is_trained or self.last_trained is None:
            return True

        age = (datetime.now() - self.last_trained).days
        return age >= max_age_days

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name}({status})"
