"""
Random Forest Predictor - Ensemble of decision trees for probability estimation
Provides calibrated probability estimates for direction prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.config import ml_settings


class RandomForestPredictor(BasePredictor):
    """
    Random Forest Classifier for Stock Direction Prediction

    Binary classification: Up (return > 0) or Down (return <= 0)
    with calibrated probability estimates.

    Random Forest advantages:
    - Robust to overfitting
    - Natural probability calibration
    - Handles non-linear relationships
    - Built-in feature importance
    - Works well with high-dimensional data
    """

    def __init__(self):
        super().__init__(model_name="random_forest")
        self.config = ml_settings.random_forest
        self.scaler = StandardScaler()
        self.calibrated_model = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train Random Forest classifier with probability calibration

        Args:
            X: Feature DataFrame
            y: Target (binary: 0=Down, 1=Up) or returns to classify
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with training performance
        """
        warnings.filterwarnings('ignore')

        # Store feature names
        self.feature_names = list(X.columns)

        # Prepare features
        X_clean = self._prepare_features(X, self.feature_names)

        # Convert returns to binary if needed
        if y.dtype == float:
            y = (y > 0).astype(int)

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
            # Initialize Random Forest
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                class_weight=self.config.class_weight,
                random_state=42,
                n_jobs=-1,
                oob_score=True  # Out-of-bag score for validation
            )

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Calibrate probabilities using isotonic regression
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv='prefit'
            )
            self.calibrated_model.fit(X_test_scaled, y_test)

            self.is_trained = True
            self.last_trained = datetime.now()

            # Make predictions using calibrated model
            y_pred = self.calibrated_model.predict(X_test_scaled)
            y_pred_proba = self.calibrated_model.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Directional accuracy is the same as accuracy for binary classification
            directional_accuracy = accuracy

            # Out-of-bag score (built-in cross-validation estimate)
            oob_score = self.model.oob_score_

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
            print(f"Random Forest training error: {e}")
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
        Predict stock direction with calibrated probabilities

        Args:
            X: Feature DataFrame (uses latest row)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Prediction horizon

        Returns:
            PredictionResult with probability-based prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Prepare features
            X_clean = self._prepare_features(X, self.feature_names)

            # Use latest row for prediction
            X_latest = X_clean.iloc[[-1]]
            X_scaled = self.scaler.transform(X_latest)

            # Get calibrated probabilities
            probabilities = self.calibrated_model.predict_proba(X_scaled)[0]
            prob_down = probabilities[0]
            prob_up = probabilities[1]

            # Determine direction
            if prob_up > 0.55:
                direction = Direction.BULLISH
                direction_probability = prob_up
            elif prob_down > 0.55:
                direction = Direction.BEARISH
                direction_probability = prob_down
            else:
                direction = Direction.NEUTRAL
                direction_probability = max(prob_up, prob_down)

            # Estimate return magnitude based on probability
            # Higher probability = larger expected move
            if direction == Direction.BULLISH:
                predicted_return = (prob_up - 0.5) * 0.15  # Scale to reasonable return
            elif direction == Direction.BEARISH:
                predicted_return = -(prob_down - 0.5) * 0.15
            else:
                predicted_return = (prob_up - prob_down) * 0.05

            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)

            # Confidence from calibrated probability
            confidence = self._calculate_confidence(
                direction_probability,
                self.metrics.accuracy if self.metrics else 0.5
            )

            # Price range based on prediction uncertainty
            # Use individual tree predictions for uncertainty
            tree_predictions = np.array([
                tree.predict_proba(X_scaled)[0]
                for tree in self.model.estimators_
            ])
            prediction_std = tree_predictions[:, 1].std()

            range_factor = 0.02 + prediction_std * 0.1
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
                    "prob_up": float(prob_up),
                    "prob_down": float(prob_down),
                    "tree_agreement": float(1 - prediction_std),
                    "oob_score": float(self.model.oob_score_) if hasattr(self.model, 'oob_score_') else None
                }
            )

        except Exception as e:
            print(f"Random Forest prediction error: {e}")
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
        Get feature importances from Random Forest

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

    def get_tree_insights(self) -> Dict[str, any]:
        """
        Get insights from the random forest trees

        Returns:
            Dictionary with tree statistics
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            depths = [tree.get_depth() for tree in self.model.estimators_]
            n_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]

            return {
                "n_trees": len(self.model.estimators_),
                "avg_depth": np.mean(depths),
                "max_depth": max(depths),
                "avg_leaves": np.mean(n_leaves),
                "oob_score": self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
            }
        except Exception:
            return {}


# Global instance
random_forest_predictor = RandomForestPredictor()
