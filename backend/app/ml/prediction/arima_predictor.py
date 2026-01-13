"""
ARIMA Predictor - Statistical baseline model for time series forecasting
Uses Auto-ARIMA (pmdarima) to automatically find optimal parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.config import ml_settings


class ARIMAPredictor(BasePredictor):
    """
    ARIMA (AutoRegressive Integrated Moving Average) Predictor

    This is a statistical baseline model that captures:
    - Autoregressive (AR) patterns: dependency on past values
    - Integrated (I): differencing to make series stationary
    - Moving Average (MA): dependency on past forecast errors

    Uses auto_arima to automatically find optimal (p, d, q) parameters.
    """

    def __init__(self):
        super().__init__(model_name="arima")
        self.config = ml_settings.arima
        self.order: Optional[Tuple[int, int, int]] = None
        self.fitted_values: Optional[pd.Series] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train ARIMA model using auto_arima for parameter selection

        Args:
            X: Feature DataFrame (not used directly by ARIMA, uses y only)
            y: Price series or returns series
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with training performance
        """
        warnings.filterwarnings('ignore')

        # Split data
        split_idx = int(len(y) * (1 - validation_split))
        train_data = y.iloc[:split_idx]
        test_data = y.iloc[split_idx:]

        train_start = y.index[0]
        train_end = y.index[split_idx - 1]
        test_start = y.index[split_idx]
        test_end = y.index[-1]

        try:
            # Use auto_arima to find best parameters
            self.model = auto_arima(
                train_data,
                start_p=1,
                start_q=1,
                max_p=self.config.max_p,
                max_d=self.config.max_d,
                max_q=self.config.max_q,
                seasonal=self.config.seasonal,
                m=self.config.m if self.config.seasonal else 1,
                d=None,  # Auto-detect
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_fits=50
            )

            self.order = self.model.order
            self.is_trained = True
            self.last_trained = datetime.now()

            # Make predictions on test set
            predictions = []
            history = list(train_data)

            for i in range(len(test_data)):
                # Forecast one step ahead
                pred = self.model.predict(n_periods=1)[0]
                predictions.append(pred)

                # Update model with actual value (rolling forecast)
                history.append(test_data.iloc[i])
                self.model.update(test_data.iloc[i:i+1])

            predictions = np.array(predictions)
            actuals = test_data.values

            # Calculate metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

            # Directional accuracy (for returns)
            if len(predictions) > 1:
                pred_direction = np.sign(np.diff(predictions))
                actual_direction = np.sign(np.diff(actuals))
                directional_accuracy = np.mean(pred_direction == actual_direction)
            else:
                directional_accuracy = 0.5

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
                hit_rate=directional_accuracy,
                n_train_samples=len(train_data),
                n_test_samples=len(test_data)
            )

            return self.metrics

        except Exception as e:
            print(f"ARIMA training error: {e}")
            # Return default metrics on failure
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
        Generate price forecast using ARIMA

        Args:
            X: Feature DataFrame (contains 'Close' price)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Days to forecast

        Returns:
            PredictionResult with forecast
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Forecast future values
            forecast, conf_int = self.model.predict(
                n_periods=horizon_days,
                return_conf_int=True,
                alpha=0.2  # 80% confidence interval
            )

            # Get the final predicted price
            predicted_price = forecast[-1]
            price_lower = conf_int[-1, 0]
            price_upper = conf_int[-1, 1]

            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price

            # Classify direction
            direction, direction_prob = self._classify_direction(predicted_return)

            # Calculate confidence based on prediction interval width
            interval_width = (price_upper - price_lower) / current_price
            # Narrower interval = higher confidence
            base_confidence = max(0.3, 1 - interval_width * 2)

            confidence = self._calculate_confidence(
                direction_prob,
                self.metrics.directional_accuracy if self.metrics else 0.5,
                interval_width
            )

            return PredictionResult(
                symbol=symbol,
                model_name=self.model_name,
                prediction_date=datetime.now(),
                horizon_days=horizon_days,
                direction=direction,
                direction_probability=direction_prob,
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                confidence=confidence,
                price_lower=price_lower,
                price_upper=price_upper,
                raw_output={"forecast": forecast.tolist(), "order": self.order}
            )

        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            # Return neutral prediction on error
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
        ARIMA doesn't have traditional feature importance.
        Returns model parameters instead.
        """
        if not self.is_trained or self.order is None:
            return {}

        return {
            "AR_order_p": float(self.order[0]),
            "differencing_d": float(self.order[1]),
            "MA_order_q": float(self.order[2]),
        }

    def update(self, new_data: pd.Series) -> None:
        """
        Update model with new data (online learning)

        Args:
            new_data: New price/return observations
        """
        if self.is_trained and self.model is not None:
            self.model.update(new_data)


# Global instance
arima_predictor = ARIMAPredictor()
