"""
LightGBM Predictor - Gradient Boosting for return magnitude prediction
Predicts the actual percentage return (regression)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import warnings

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.config import ml_settings


class LightGBMPredictor(BasePredictor):
    """
    LightGBM Regressor for Stock Return Prediction

    Predicts the actual percentage return for the given horizon.

    LightGBM advantages:
    - Faster training than XGBoost
    - Lower memory usage
    - Better accuracy with large datasets
    - Native handling of categorical features
    """

    def __init__(self):
        super().__init__(model_name="lightgbm")
        self.config = ml_settings.lightgbm
        self.scaler = StandardScaler()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train LightGBM regressor

        Args:
            X: Feature DataFrame
            y: Target returns (as decimals, e.g., 0.05 for 5%)
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with training performance
        """
        warnings.filterwarnings('ignore')

        # Store feature names
        self.feature_names = list(X.columns)

        # Prepare features
        X_clean = self._prepare_features(X, self.feature_names)

        # Remove rows with NaN
        valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]

        # Split data (maintain time order)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=validation_split,
            shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        train_start = X.index[0]
        train_end = X.index[int(len(X) * (1 - validation_split)) - 1]
        test_start = X.index[int(len(X) * (1 - validation_split))]
        test_end = X.index[-1]

        try:
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

            # LightGBM parameters
            params = {
                'objective': self.config.objective,
                'metric': self.config.metric,
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': self.config.colsample_bytree,
                'bagging_fraction': self.config.subsample,
                'bagging_freq': 5,
                'max_depth': self.config.max_depth,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }

            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config.n_estimators,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            self.is_trained = True
            self.last_trained = datetime.now()

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            r2 = r2_score(y_test, y_pred)

            # Directional accuracy
            pred_direction = np.sign(y_pred)
            actual_direction = np.sign(y_test)
            directional_accuracy = np.mean(pred_direction == actual_direction)

            # Calculate hit rate (correct direction predictions)
            hit_rate = directional_accuracy

            self.metrics = ModelMetrics(
                model_name=self.model_name,
                symbol=None,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                mae=mae,
                rmse=rmse,
                mape=mape,
                directional_accuracy=directional_accuracy,
                hit_rate=hit_rate,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test)
            )

            return self.metrics

        except Exception as e:
            print(f"LightGBM training error: {e}")
            self.is_trained = False
            return ModelMetrics(
                model_name=self.model_name,
                symbol=None,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

    def predict(
        self,
        X: pd.DataFrame,
        current_price: float,
        symbol: str,
        horizon_days: int = 30
    ) -> PredictionResult:
        """
        Predict return magnitude using LightGBM

        Args:
            X: Feature DataFrame (uses latest row)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Prediction horizon

        Returns:
            PredictionResult with return prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Prepare features
            X_clean = self._prepare_features(X, self.feature_names)

            # Use latest row for prediction
            X_latest = X_clean.iloc[[-1]]
            X_scaled = self.scaler.transform(X_latest)

            # Get prediction
            predicted_return = self.model.predict(X_scaled)[0]

            # Classify direction based on predicted return
            direction, direction_probability = self._classify_direction(predicted_return)

            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)

            # Estimate prediction uncertainty using tree variance
            # LightGBM doesn't have built-in uncertainty, estimate from model
            leaf_indices = self.model.predict(X_scaled, pred_leaf=True)
            # Use standard error estimate based on training RMSE
            prediction_std = self.metrics.rmse if self.metrics else 0.03

            # Price range
            price_lower = current_price * (1 + predicted_return - 2 * prediction_std)
            price_upper = current_price * (1 + predicted_return + 2 * prediction_std)

            # Calculate confidence
            confidence = self._calculate_confidence(
                direction_probability,
                self.metrics.directional_accuracy if self.metrics else 0.5,
                prediction_std
            )

            return PredictionResult(
                symbol=symbol,
                model_name=self.model_name,
                prediction_date=datetime.now(),
                horizon_days=horizon_days,
                direction=direction,
                direction_probability=direction_probability,
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                confidence=confidence,
                price_lower=price_lower,
                price_upper=price_upper,
                feature_importance=self.get_feature_importance(),
                raw_output={
                    "prediction_std": prediction_std,
                    "n_trees_used": self.model.num_trees()
                }
            )

        except Exception as e:
            print(f"LightGBM prediction error: {e}")
            return PredictionResult(
                symbol=symbol,
                model_name=self.model_name,
                prediction_date=datetime.now(),
                horizon_days=horizon_days,
                direction=Direction.NEUTRAL,
                direction_probability=0.5,
                current_price=current_price,
                predicted_price=current_price,
                predicted_return=0.0,
                confidence=0.1
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from LightGBM

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            # Get importance (gain-based)
            importances = self.model.feature_importance(importance_type='gain')

            # Normalize
            total = sum(importances)
            if total > 0:
                importances = importances / total

            feature_importance = dict(zip(self.feature_names, importances))

            # Sort by importance and return top 20
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return dict(sorted_features[:20])

        except Exception:
            return {}

    def get_learning_curve(self) -> Dict[str, List[float]]:
        """
        Get training learning curve

        Returns:
            Dictionary with training and validation loss over iterations
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            # LightGBM stores eval results during training
            return {
                "iterations": list(range(self.model.num_trees())),
                "n_trees": self.model.num_trees()
            }
        except Exception:
            return {}


# Global instance
lightgbm_predictor = LightGBMPredictor()
