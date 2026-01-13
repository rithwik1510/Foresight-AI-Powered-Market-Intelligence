"""
Prophet Predictor - Facebook's time series forecasting model
Excellent for capturing trends, seasonality, and holiday effects
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import warnings

from prophet import Prophet

from app.ml.prediction.base_predictor import (
    BasePredictor, PredictionResult, ModelMetrics, Direction
)
from app.ml.config import ml_settings


class ProphetPredictor(BasePredictor):
    """
    Facebook Prophet Predictor

    Prophet is designed for business time series with:
    - Strong seasonal effects (weekly, yearly)
    - Historical trend changes
    - Missing data and outliers

    It decomposes time series into:
    - Trend: Long-term increase/decrease
    - Seasonality: Weekly, yearly patterns
    - Holidays: Special events (we use market events)
    """

    def __init__(self):
        super().__init__(model_name="prophet")
        self.config = ml_settings.prophet
        self.forecast_df: Optional[pd.DataFrame] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train Prophet model

        Args:
            X: Feature DataFrame (not used, Prophet uses y with dates)
            y: Price series with DatetimeIndex
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with training performance
        """
        warnings.filterwarnings('ignore')

        # Prepare data in Prophet format (ds, y)
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })

        # Split data
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        train_start = df['ds'].iloc[0]
        train_end = df['ds'].iloc[split_idx - 1]
        test_start = df['ds'].iloc[split_idx]
        test_end = df['ds'].iloc[-1]

        try:
            # Initialize Prophet with configuration
            self.model = Prophet(
                yearly_seasonality=self.config.yearly_seasonality,
                weekly_seasonality=self.config.weekly_seasonality,
                daily_seasonality=self.config.daily_seasonality,
                changepoint_prior_scale=self.config.changepoint_prior_scale,
                seasonality_prior_scale=self.config.seasonality_prior_scale,
                interval_width=self.config.interval_width,
            )

            # Add Indian market seasonality (monthly patterns)
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )

            # Fit model
            self.model.fit(train_df)

            self.is_trained = True
            self.last_trained = datetime.now()

            # Make predictions on test set
            future = self.model.make_future_dataframe(periods=len(test_df))
            forecast = self.model.predict(future)

            # Get predictions for test period
            test_predictions = forecast.iloc[split_idx:]['yhat'].values
            test_actuals = test_df['y'].values

            # Calculate metrics
            mae = np.mean(np.abs(test_predictions - test_actuals))
            rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
            mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100

            # Directional accuracy
            if len(test_predictions) > 1:
                pred_direction = np.sign(np.diff(test_predictions))
                actual_direction = np.sign(np.diff(test_actuals))
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
                n_train_samples=len(train_df),
                n_test_samples=len(test_df)
            )

            return self.metrics

        except Exception as e:
            print(f"Prophet training error: {e}")
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
        Generate price forecast using Prophet

        Args:
            X: Feature DataFrame (contains price data with DatetimeIndex)
            current_price: Current stock price
            symbol: Stock symbol
            horizon_days: Days to forecast

        Returns:
            PredictionResult with forecast
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=horizon_days)

            # Generate forecast
            forecast = self.model.predict(future)
            self.forecast_df = forecast

            # Get prediction at horizon
            final_forecast = forecast.iloc[-1]
            predicted_price = final_forecast['yhat']
            price_lower = final_forecast['yhat_lower']
            price_upper = final_forecast['yhat_upper']

            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price

            # Classify direction
            direction, direction_prob = self._classify_direction(predicted_return)

            # Calculate confidence based on uncertainty interval
            interval_width = (price_upper - price_lower) / current_price
            base_confidence = max(0.3, 1 - interval_width)

            confidence = self._calculate_confidence(
                direction_prob,
                self.metrics.directional_accuracy if self.metrics else 0.5,
                interval_width
            )

            # Get trend component for additional insight
            trend_change = forecast['trend'].iloc[-1] - forecast['trend'].iloc[-horizon_days]

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
                raw_output={
                    "trend_change": trend_change,
                    "weekly_seasonality": forecast['weekly'].iloc[-1] if 'weekly' in forecast.columns else 0,
                    "yearly_seasonality": forecast['yearly'].iloc[-1] if 'yearly' in forecast.columns else 0,
                }
            )

        except Exception as e:
            print(f"Prophet prediction error: {e}")
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
        Get component importance from Prophet

        Returns decomposition of the forecast into components
        """
        if not self.is_trained or self.forecast_df is None:
            return {}

        try:
            # Calculate variance contribution of each component
            components = {}

            if 'trend' in self.forecast_df.columns:
                trend_var = self.forecast_df['trend'].var()
                components['trend'] = trend_var

            if 'weekly' in self.forecast_df.columns:
                weekly_var = self.forecast_df['weekly'].var()
                components['weekly_seasonality'] = weekly_var

            if 'yearly' in self.forecast_df.columns:
                yearly_var = self.forecast_df['yearly'].var()
                components['yearly_seasonality'] = yearly_var

            if 'monthly' in self.forecast_df.columns:
                monthly_var = self.forecast_df['monthly'].var()
                components['monthly_seasonality'] = monthly_var

            # Normalize to percentages
            total_var = sum(components.values())
            if total_var > 0:
                components = {k: v / total_var for k, v in components.items()}

            return components

        except Exception:
            return {}

    def get_trend_analysis(self) -> Dict[str, any]:
        """
        Get detailed trend analysis from Prophet

        Returns:
            Dictionary with trend information
        """
        if not self.is_trained or self.forecast_df is None:
            return {}

        try:
            forecast = self.forecast_df
            return {
                "current_trend": forecast['trend'].iloc[-1],
                "trend_slope": forecast['trend'].diff().mean(),
                "changepoints": len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0,
            }
        except Exception:
            return {}


# Global instance
prophet_predictor = ProphetPredictor()
