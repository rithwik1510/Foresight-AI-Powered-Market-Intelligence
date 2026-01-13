"""
XGBoost Predictor - Gradient Boosting for direction classification
Classifies stock movement into: Bullish, Neutral, Bearish
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.config import ml_settings


class XGBoostPredictor(BasePredictor):
    """
    XGBoost Classifier for Stock Direction Prediction

    Classifies into 3 classes:
    - 0: Bearish (return < -2%)
    - 1: Neutral (-2% <= return <= 2%)
    - 2: Bullish (return > 2%)

    XGBoost excels at:
    - Handling non-linear relationships
    - Feature interactions
    - Robust to overfitting with proper regularization
    """

    def __init__(self):
        super().__init__(model_name="xgboost")
        self.config = ml_settings.xgboost
        self.scaler = StandardScaler()
        self.class_labels = {0: Direction.BEARISH, 1: Direction.NEUTRAL, 2: Direction.BULLISH}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train XGBoost classifier

        Args:
            X: Feature DataFrame
            y: Target (0=Bearish, 1=Neutral, 2=Bullish) or returns to classify
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with training performance
        """
        warnings.filterwarnings('ignore')

        # Store feature names
        self.feature_names = list(X.columns)

        # Prepare features
        X_clean = self._prepare_features(X, self.feature_names)

        # Convert returns to classes if needed
        if y.dtype == float:
            y = self._returns_to_classes(y)

        # Remove rows with NaN
        valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=validation_split,
            shuffle=False  # Keep time order
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        train_start = X.index[0]
        train_end = X.index[int(len(X) * (1 - validation_split)) - 1]
        test_start = X.index[int(len(X) * (1 - validation_split))]
        test_end = X.index[-1]

        try:
            # Initialize XGBoost
            self.model = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_weight=self.config.min_child_weight,
                objective=self.config.objective,
                num_class=self.config.num_class,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )

            # Train with early stopping
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )

            self.is_trained = True
            self.last_trained = datetime.now()

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Directional accuracy (excluding neutral as correct)
            # Convert to binary direction for directional accuracy
            y_test_dir = np.where(y_test == 2, 1, np.where(y_test == 0, -1, 0))
            y_pred_dir = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
            non_neutral_mask = y_test_dir != 0
            if non_neutral_mask.sum() > 0:
                directional_accuracy = np.mean(y_test_dir[non_neutral_mask] == y_pred_dir[non_neutral_mask])
            else:
                directional_accuracy = 0.5

            self.metrics = ModelMetrics(
                model_name=self.model_name,
                symbol=None,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                directional_accuracy=directional_accuracy,
                hit_rate=accuracy,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test)
            )

            return self.metrics

        except Exception as e:
            print(f"XGBoost training error: {e}")
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
        Predict stock direction using XGBoost

        Args:
            X: Feature DataFrame (uses latest row)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Prediction horizon

        Returns:
            PredictionResult with direction prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Prepare features
            X_clean = self._prepare_features(X, self.feature_names)

            # Use latest row for prediction
            X_latest = X_clean.iloc[[-1]]
            X_scaled = self.scaler.transform(X_latest)

            # Get prediction and probabilities
            pred_class = self.model.predict(X_scaled)[0]
            pred_proba = self.model.predict_proba(X_scaled)[0]

            # Map to direction
            direction = self.class_labels[pred_class]
            direction_probability = pred_proba[pred_class]

            # Estimate return based on class and confidence
            if direction == Direction.BULLISH:
                # Estimate return proportional to confidence
                predicted_return = 0.02 + (direction_probability - 0.33) * 0.1
            elif direction == Direction.BEARISH:
                predicted_return = -0.02 - (direction_probability - 0.33) * 0.1
            else:
                predicted_return = (pred_proba[2] - pred_proba[0]) * 0.02

            predicted_price = current_price * (1 + predicted_return)

            # Calculate confidence
            confidence = self._calculate_confidence(
                direction_probability,
                self.metrics.accuracy if self.metrics else 0.5
            )

            # Price range based on confidence
            range_factor = (1 - confidence) * 0.1 + 0.02
            price_lower = current_price * (1 + predicted_return - range_factor)
            price_upper = current_price * (1 + predicted_return + range_factor)

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
                    "class_probabilities": {
                        "bearish": float(pred_proba[0]),
                        "neutral": float(pred_proba[1]),
                        "bullish": float(pred_proba[2])
                    }
                }
            )

        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            return PredictionResult(
                symbol=symbol,
                model_name=self.model_name,
                prediction_date=datetime.now(),
                horizon_days=horizon_days,
                direction=Direction.NEUTRAL,
                direction_probability=0.33,
                current_price=current_price,
                predicted_price=current_price,
                predicted_return=0.0,
                confidence=0.1
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from XGBoost

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            importances = self.model.feature_importances_
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

    def _returns_to_classes(self, returns: pd.Series) -> pd.Series:
        """
        Convert returns to class labels

        Args:
            returns: Return series

        Returns:
            Series with class labels (0=Bearish, 1=Neutral, 2=Bullish)
        """
        return pd.Series(
            np.where(
                returns > 0.02, 2,  # Bullish
                np.where(returns < -0.02, 0, 1)  # Bearish or Neutral
            ),
            index=returns.index
        )


# Global instance
xgboost_predictor = XGBoostPredictor()
