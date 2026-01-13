"""
Ensemble Predictor - Combines all ML models for robust predictions
Weighted combination with confidence-based adjustment
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import asyncio

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.prediction.arima_predictor import ARIMAPredictor
from app.ml.prediction.prophet_predictor import ProphetPredictor
from app.ml.prediction.xgboost_predictor import XGBoostPredictor
from app.ml.prediction.lightgbm_predictor import LightGBMPredictor
from app.ml.prediction.random_forest_predictor import RandomForestPredictor
from app.ml.config import ml_settings


@dataclass
class EnsemblePrediction:
    """Complete ensemble prediction result"""
    symbol: str
    prediction_date: datetime
    horizon_days: int

    # Direction
    direction: Direction
    direction_probability: float

    # Price predictions
    current_price: float
    predicted_price: float
    predicted_return: float
    price_lower: float
    price_upper: float

    # Confidence
    confidence: float
    model_agreement: float  # How much models agree

    # Individual model predictions
    model_predictions: Dict[str, PredictionResult] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)

    # Feature importance (aggregated)
    top_bullish_factors: List[str] = field(default_factory=list)
    top_bearish_factors: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Risk assessment
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "prediction_date": self.prediction_date.isoformat(),
            "horizon_days": self.horizon_days,
            "direction": self.direction.value,
            "direction_probability": round(self.direction_probability, 4),
            "current_price": round(self.current_price, 2),
            "predicted_price": round(self.predicted_price, 2),
            "predicted_return_pct": round(self.predicted_return * 100, 2),
            "price_range": [round(self.price_lower, 2), round(self.price_upper, 2)],
            "confidence": round(self.confidence, 4),
            "model_agreement": round(self.model_agreement, 4),
            "risk_level": self.risk_level,
            "top_bullish_factors": self.top_bullish_factors,
            "top_bearish_factors": self.top_bearish_factors,
            "model_breakdown": {
                name: {
                    "direction": pred.direction.value,
                    "predicted_return": round(pred.predicted_return * 100, 2),
                    "confidence": round(pred.confidence, 4),
                    "weight": round(self.model_weights.get(name, 0), 4)
                }
                for name, pred in self.model_predictions.items()
            }
        }


class EnsemblePredictor:
    """
    Ensemble Predictor combining multiple ML models

    Models:
    - ARIMA: Statistical baseline (time series)
    - Prophet: Trend + seasonality
    - XGBoost: Direction classification
    - LightGBM: Return magnitude
    - Random Forest: Probability estimation

    Weighting strategy:
    - Base weights from configuration
    - Adjusted by recent model accuracy
    - Boosted by model agreement
    """

    def __init__(self):
        self.config = ml_settings.ensemble

        # Initialize models
        self.models: Dict[str, BasePredictor] = {
            "arima": ARIMAPredictor(),
            "prophet": ProphetPredictor(),
            "xgboost": XGBoostPredictor(),
            "lightgbm": LightGBMPredictor(),
            "random_forest": RandomForestPredictor(),
        }

        # Model weights (can be adjusted based on performance)
        self.base_weights = {
            "arima": 0.15,
            "prophet": 0.20,
            "xgboost": 0.20,
            "lightgbm": 0.25,
            "random_forest": 0.20,
        }

        self.trained_models: List[str] = []
        self.is_trained = False

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_prices: Optional[pd.Series] = None,
        validation_split: float = 0.2
    ) -> Dict[str, ModelMetrics]:
        """
        Train all models in the ensemble

        Args:
            X: Feature DataFrame
            y: Target returns (for ML models)
            y_prices: Price series (for time series models)
            validation_split: Validation split ratio

        Returns:
            Dictionary of model_name -> ModelMetrics
        """
        results = {}
        self.trained_models = []

        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                if name in ["arima", "prophet"]:
                    # Time series models use prices
                    if y_prices is not None:
                        metrics = model.train(X, y_prices, validation_split)
                    else:
                        print(f"Skipping {name}: No price data provided")
                        continue
                else:
                    # ML models use returns/features
                    metrics = model.train(X, y, validation_split)

                results[name] = metrics
                self.trained_models.append(name)
                print(f"  {name} trained: Directional Accuracy = {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else f"  {name} trained")

            except Exception as e:
                print(f"  {name} training error: {e}")
                continue

        self.is_trained = len(self.trained_models) > 0

        # Update weights based on performance
        self._update_weights(results)

        return results

    def _update_weights(self, metrics: Dict[str, ModelMetrics]):
        """Update model weights based on performance"""
        # Calculate performance scores
        scores = {}
        for name, m in metrics.items():
            if m.directional_accuracy:
                scores[name] = m.directional_accuracy
            elif m.accuracy:
                scores[name] = m.accuracy
            else:
                scores[name] = 0.5  # Default

        if not scores:
            return

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            for name in scores:
                # Blend base weight with performance-based weight
                perf_weight = scores[name] / total
                self.base_weights[name] = (
                    0.5 * self.base_weights.get(name, 0.2) +
                    0.5 * perf_weight
                )

        # Re-normalize
        total = sum(self.base_weights.values())
        for name in self.base_weights:
            self.base_weights[name] /= total

    def predict(
        self,
        X: pd.DataFrame,
        current_price: float,
        symbol: str,
        horizon_days: int = 30
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction

        Args:
            X: Feature DataFrame
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Prediction horizon

        Returns:
            EnsemblePrediction with combined results
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        predictions = {}
        model_weights = {}

        # Collect predictions from all trained models
        for name in self.trained_models:
            model = self.models[name]
            try:
                pred = model.predict(X, current_price, symbol, horizon_days)
                predictions[name] = pred
                model_weights[name] = self.base_weights.get(name, 0.2)
            except Exception as e:
                print(f"{name} prediction error: {e}")
                continue

        if not predictions:
            raise ValueError("No models produced predictions")

        # Normalize weights for available models
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

        # Combine predictions
        ensemble_result = self._combine_predictions(
            predictions, model_weights, current_price, symbol, horizon_days
        )

        return ensemble_result

    def _combine_predictions(
        self,
        predictions: Dict[str, PredictionResult],
        weights: Dict[str, float],
        current_price: float,
        symbol: str,
        horizon_days: int
    ) -> EnsemblePrediction:
        """Combine individual predictions into ensemble"""

        # Calculate weighted return
        weighted_return = 0.0
        for name, pred in predictions.items():
            weighted_return += pred.predicted_return * weights[name]

        # Calculate weighted direction probability
        direction_scores = {
            Direction.BULLISH: 0.0,
            Direction.NEUTRAL: 0.0,
            Direction.BEARISH: 0.0,
        }

        for name, pred in predictions.items():
            weight = weights[name]
            direction_scores[pred.direction] += weight * pred.direction_probability

        # Determine ensemble direction
        total_score = sum(direction_scores.values())
        if total_score > 0:
            direction_scores = {k: v / total_score for k, v in direction_scores.items()}

        best_direction = max(direction_scores, key=direction_scores.get)
        direction_probability = direction_scores[best_direction]

        # Calculate model agreement
        direction_counts = {}
        for pred in predictions.values():
            direction_counts[pred.direction] = direction_counts.get(pred.direction, 0) + 1

        most_common = max(direction_counts.values())
        model_agreement = most_common / len(predictions)

        # Calculate price prediction
        predicted_price = current_price * (1 + weighted_return)

        # Calculate price range (average of individual ranges)
        price_lowers = [p.price_lower for p in predictions.values() if p.price_lower]
        price_uppers = [p.price_upper for p in predictions.values() if p.price_upper]

        if price_lowers and price_uppers:
            price_lower = np.mean(price_lowers)
            price_upper = np.mean(price_uppers)
        else:
            # Estimate based on return uncertainty
            std_return = np.std([p.predicted_return for p in predictions.values()])
            price_lower = current_price * (1 + weighted_return - 2 * std_return)
            price_upper = current_price * (1 + weighted_return + 2 * std_return)

        # Calculate ensemble confidence
        # Higher when: high individual confidences + high agreement
        avg_confidence = np.mean([p.confidence for p in predictions.values()])
        confidence = 0.6 * avg_confidence + 0.4 * model_agreement

        # Boost confidence if models agree strongly
        if model_agreement > 0.8:
            confidence = min(1.0, confidence * 1.1)

        # Aggregate feature importance
        all_importance = {}
        for pred in predictions.values():
            if pred.feature_importance:
                for feature, importance in pred.feature_importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)

        feature_importance = {
            k: np.mean(v) for k, v in all_importance.items()
        }

        # Get top factors
        top_bullish, top_bearish = self._extract_top_factors(
            predictions, feature_importance, best_direction
        )

        # Assess risk level
        risk_level = self._assess_risk(
            predictions, model_agreement, weighted_return, direction_probability
        )

        return EnsemblePrediction(
            symbol=symbol,
            prediction_date=datetime.now(),
            horizon_days=horizon_days,
            direction=best_direction,
            direction_probability=direction_probability,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=weighted_return,
            price_lower=price_lower,
            price_upper=price_upper,
            confidence=confidence,
            model_agreement=model_agreement,
            model_predictions=predictions,
            model_weights=weights,
            top_bullish_factors=top_bullish,
            top_bearish_factors=top_bearish,
            feature_importance=feature_importance,
            risk_level=risk_level,
        )

    def _extract_top_factors(
        self,
        predictions: Dict[str, PredictionResult],
        feature_importance: Dict[str, float],
        direction: Direction
    ) -> Tuple[List[str], List[str]]:
        """Extract top bullish and bearish factors"""
        # Define factor categories
        bullish_keywords = ["momentum", "rsi", "macd", "bullish", "volume_sma", "return_5d"]
        bearish_keywords = ["volatility", "bearish", "vix", "risk", "decline"]

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        top_bullish = []
        top_bearish = []

        for feature, importance in sorted_features:
            feature_lower = feature.lower()

            # Categorize based on keywords and values
            if any(kw in feature_lower for kw in bullish_keywords):
                if importance > 0.01 and len(top_bullish) < 3:
                    top_bullish.append(f"{feature} (importance: {importance:.2%})")
            elif any(kw in feature_lower for kw in bearish_keywords):
                if importance > 0.01 and len(top_bearish) < 3:
                    top_bearish.append(f"{feature} (importance: {importance:.2%})")

        # Add default factors based on direction
        if direction == Direction.BULLISH and not top_bullish:
            top_bullish = ["Positive momentum signals", "Model consensus bullish"]
        if direction == Direction.BEARISH and not top_bearish:
            top_bearish = ["Negative momentum signals", "Model consensus bearish"]

        return top_bullish, top_bearish

    def _assess_risk(
        self,
        predictions: Dict[str, PredictionResult],
        agreement: float,
        predicted_return: float,
        direction_prob: float
    ) -> str:
        """Assess risk level of prediction"""
        risk_score = 0

        # Low agreement = higher risk
        if agreement < 0.5:
            risk_score += 3
        elif agreement < 0.7:
            risk_score += 1

        # Low direction probability = higher risk
        if direction_prob < 0.5:
            risk_score += 2
        elif direction_prob < 0.6:
            risk_score += 1

        # High return prediction = higher risk
        if abs(predicted_return) > 0.15:
            risk_score += 2
        elif abs(predicted_return) > 0.10:
            risk_score += 1

        # Wide prediction variance = higher risk
        returns = [p.predicted_return for p in predictions.values()]
        if np.std(returns) > 0.05:
            risk_score += 2
        elif np.std(returns) > 0.03:
            risk_score += 1

        if risk_score <= 2:
            return "LOW"
        elif risk_score <= 5:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            "trained_models": self.trained_models,
            "total_models": len(self.models),
            "weights": self.base_weights,
            "is_trained": self.is_trained,
        }


# Global instance
ensemble_predictor = EnsemblePredictor()
