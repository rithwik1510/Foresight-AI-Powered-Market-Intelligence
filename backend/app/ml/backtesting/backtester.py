"""
Backtester - Walk-forward validation for ML prediction models
Tests historical performance using rolling training windows
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import yfinance as yf

from app.ml.prediction.base_predictor import Direction
from app.ml.prediction.ensemble_predictor import EnsemblePredictor
from app.ml.features.technical_features import technical_feature_generator


@dataclass
class BacktestTrade:
    """Single backtest trade/prediction"""
    date: datetime
    symbol: str
    direction: Direction
    predicted_return: float
    actual_return: float
    confidence: float
    is_correct: bool
    pnl: float  # If we had traded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "predicted_return": round(self.predicted_return * 100, 2),
            "actual_return": round(self.actual_return * 100, 2),
            "confidence": round(self.confidence, 4),
            "is_correct": self.is_correct,
            "pnl": round(self.pnl * 100, 2),
        }


@dataclass
class BacktestResult:
    """Complete backtest results"""
    symbol: str
    start_date: datetime
    end_date: datetime
    horizon_days: int

    # Performance metrics
    total_predictions: int
    correct_predictions: int
    directional_accuracy: float

    # Returns
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float

    # By direction
    bullish_accuracy: float
    bearish_accuracy: float
    neutral_accuracy: float

    # By confidence
    high_confidence_accuracy: float  # confidence > 0.7
    medium_confidence_accuracy: float  # 0.5 < confidence <= 0.7
    low_confidence_accuracy: float  # confidence <= 0.5

    # Individual trades
    trades: List[BacktestTrade] = field(default_factory=list)

    # Model performance
    model_accuracies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "horizon_days": self.horizon_days,
            },
            "summary": {
                "total_predictions": self.total_predictions,
                "correct_predictions": self.correct_predictions,
                "directional_accuracy": round(self.directional_accuracy, 4),
            },
            "returns": {
                "total_return": round(self.total_return * 100, 2),
                "annualized_return": round(self.annualized_return * 100, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "max_drawdown": round(self.max_drawdown * 100, 2),
            },
            "by_direction": {
                "bullish": round(self.bullish_accuracy, 4),
                "bearish": round(self.bearish_accuracy, 4),
                "neutral": round(self.neutral_accuracy, 4),
            },
            "by_confidence": {
                "high": round(self.high_confidence_accuracy, 4),
                "medium": round(self.medium_confidence_accuracy, 4),
                "low": round(self.low_confidence_accuracy, 4),
            },
            "model_accuracies": {
                k: round(v, 4) for k, v in self.model_accuracies.items()
            },
            "trades": [t.to_dict() for t in self.trades[-20:]],  # Last 20 trades
        }


class Backtester:
    """
    Walk-Forward Backtester for ML Predictions

    Uses rolling window approach:
    1. Train on historical data (e.g., 1 year)
    2. Predict next period (e.g., 30 days)
    3. Roll forward and repeat

    This simulates real-world usage where models are
    retrained periodically with new data.
    """

    def __init__(self):
        self.ensemble = EnsemblePredictor()

    def run_backtest(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        train_window_days: int = 365,
        test_window_days: int = 30,
        horizon_days: int = 20,
        retrain_frequency_days: int = 30,
    ) -> BacktestResult:
        """
        Run walk-forward backtest

        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            train_window_days: Days of data for training
            test_window_days: Days to test before retraining
            horizon_days: Prediction horizon
            retrain_frequency_days: How often to retrain

        Returns:
            BacktestResult with performance metrics
        """
        # Fetch all historical data
        total_days_needed = train_window_days + 365  # Extra buffer
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        data_start = start_date - timedelta(days=train_window_days + 100)

        print(f"Fetching {symbol} data from {data_start.strftime('%Y-%m-%d')}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=data_start.strftime('%Y-%m-%d'))

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        print(f"Data shape: {df.shape}")

        # Generate features
        print("Generating features...")
        features = technical_feature_generator.generate_features(df)

        # Create target
        target_returns = df['Close'].pct_change(horizon_days).shift(-horizon_days)

        # Align
        valid_idx = features.index.intersection(target_returns.dropna().index)
        features = features.loc[valid_idx]
        target_returns = target_returns.loc[valid_idx]
        prices = df.loc[valid_idx, 'Close']

        # Filter to backtest period
        backtest_mask = (features.index >= start_date) & (features.index <= end_date)
        backtest_dates = features.index[backtest_mask]

        print(f"Backtest period: {backtest_dates[0]} to {backtest_dates[-1]}")
        print(f"Total test points: {len(backtest_dates)}")

        # Run walk-forward backtest
        trades = []
        model_predictions = {name: [] for name in self.ensemble.models.keys()}
        last_train_date = None

        for i, test_date in enumerate(backtest_dates):
            # Check if we need to retrain
            if last_train_date is None or (test_date - last_train_date).days >= retrain_frequency_days:
                # Get training data
                train_end = test_date - timedelta(days=1)
                train_start = train_end - timedelta(days=train_window_days)

                train_mask = (features.index >= train_start) & (features.index <= train_end)
                X_train = features[train_mask].copy()
                y_train = target_returns[train_mask].copy()
                y_prices_train = prices[train_mask].copy()

                if len(X_train) < 100:
                    continue

                print(f"\nRetraining at {test_date.strftime('%Y-%m-%d')}...")
                try:
                    self.ensemble.train_all(X_train, y_train, y_prices_train)
                    last_train_date = test_date
                except Exception as e:
                    print(f"Training error: {e}")
                    continue

            # Make prediction
            if not self.ensemble.is_trained:
                continue

            try:
                # Get features up to test date
                X_test = features.loc[[test_date]]
                current_price = prices.loc[test_date]

                prediction = self.ensemble.predict(
                    X_test, current_price, symbol, horizon_days
                )

                # Get actual return
                actual_return = target_returns.loc[test_date]

                if pd.isna(actual_return):
                    continue

                # Determine if prediction was correct
                actual_direction = Direction.BULLISH if actual_return > 0.02 else (
                    Direction.BEARISH if actual_return < -0.02 else Direction.NEUTRAL
                )

                # For directional accuracy, count bullish/bearish correctly
                if prediction.direction == Direction.NEUTRAL:
                    is_correct = abs(actual_return) < 0.02
                else:
                    is_correct = (
                        (prediction.direction == Direction.BULLISH and actual_return > 0) or
                        (prediction.direction == Direction.BEARISH and actual_return < 0)
                    )

                # Calculate PnL (if we traded based on prediction)
                if prediction.direction == Direction.BULLISH:
                    pnl = actual_return
                elif prediction.direction == Direction.BEARISH:
                    pnl = -actual_return  # Shorting
                else:
                    pnl = 0  # No trade

                trade = BacktestTrade(
                    date=test_date,
                    symbol=symbol,
                    direction=prediction.direction,
                    predicted_return=prediction.predicted_return,
                    actual_return=actual_return,
                    confidence=prediction.confidence,
                    is_correct=is_correct,
                    pnl=pnl,
                )
                trades.append(trade)

                # Track individual model predictions
                for model_name, model_pred in prediction.model_predictions.items():
                    model_correct = (
                        (model_pred.direction == Direction.BULLISH and actual_return > 0) or
                        (model_pred.direction == Direction.BEARISH and actual_return < 0) or
                        (model_pred.direction == Direction.NEUTRAL and abs(actual_return) < 0.02)
                    )
                    model_predictions[model_name].append(model_correct)

            except Exception as e:
                print(f"Prediction error at {test_date}: {e}")
                continue

            # Progress
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(backtest_dates)} dates")

        # Calculate results
        return self._calculate_results(
            symbol, start_date, end_date, horizon_days, trades, model_predictions
        )

    def _calculate_results(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        horizon_days: int,
        trades: List[BacktestTrade],
        model_predictions: Dict[str, List[bool]]
    ) -> BacktestResult:
        """Calculate backtest metrics"""

        if not trades:
            return BacktestResult(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizon_days=horizon_days,
                total_predictions=0,
                correct_predictions=0,
                directional_accuracy=0.0,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                bullish_accuracy=0.0,
                bearish_accuracy=0.0,
                neutral_accuracy=0.0,
                high_confidence_accuracy=0.0,
                medium_confidence_accuracy=0.0,
                low_confidence_accuracy=0.0,
                trades=[],
            )

        # Basic metrics
        total = len(trades)
        correct = sum(1 for t in trades if t.is_correct)
        directional_accuracy = correct / total

        # Returns
        pnls = [t.pnl for t in trades]
        total_return = sum(pnls)

        # Annualized return
        days = (end_date - start_date).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0.0

        # Sharpe ratio
        if len(pnls) > 1:
            pnl_std = np.std(pnls)
            if pnl_std > 0:
                sharpe_ratio = (np.mean(pnls) / pnl_std) * np.sqrt(252 / horizon_days)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # By direction
        bullish_trades = [t for t in trades if t.direction == Direction.BULLISH]
        bearish_trades = [t for t in trades if t.direction == Direction.BEARISH]
        neutral_trades = [t for t in trades if t.direction == Direction.NEUTRAL]

        bullish_accuracy = sum(1 for t in bullish_trades if t.is_correct) / max(1, len(bullish_trades))
        bearish_accuracy = sum(1 for t in bearish_trades if t.is_correct) / max(1, len(bearish_trades))
        neutral_accuracy = sum(1 for t in neutral_trades if t.is_correct) / max(1, len(neutral_trades))

        # By confidence
        high_conf = [t for t in trades if t.confidence > 0.7]
        med_conf = [t for t in trades if 0.5 < t.confidence <= 0.7]
        low_conf = [t for t in trades if t.confidence <= 0.5]

        high_accuracy = sum(1 for t in high_conf if t.is_correct) / max(1, len(high_conf))
        med_accuracy = sum(1 for t in med_conf if t.is_correct) / max(1, len(med_conf))
        low_accuracy = sum(1 for t in low_conf if t.is_correct) / max(1, len(low_conf))

        # Model accuracies
        model_accuracies = {}
        for model_name, predictions in model_predictions.items():
            if predictions:
                model_accuracies[model_name] = sum(predictions) / len(predictions)

        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
            total_predictions=total,
            correct_predictions=correct,
            directional_accuracy=directional_accuracy,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            bullish_accuracy=bullish_accuracy,
            bearish_accuracy=bearish_accuracy,
            neutral_accuracy=neutral_accuracy,
            high_confidence_accuracy=high_accuracy,
            medium_confidence_accuracy=med_accuracy,
            low_confidence_accuracy=low_accuracy,
            trades=trades,
            model_accuracies=model_accuracies,
        )

    def quick_backtest(
        self,
        symbol: str,
        test_days: int = 60,
        train_days: int = 365,
        horizon_days: int = 20
    ) -> BacktestResult:
        """
        Quick backtest with simplified parameters

        Args:
            symbol: Stock symbol
            test_days: Days to test
            train_days: Days to train on
            horizon_days: Prediction horizon

        Returns:
            BacktestResult
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        return self.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            train_window_days=train_days,
            test_window_days=test_days,
            horizon_days=horizon_days,
            retrain_frequency_days=test_days,  # Only train once
        )


# Global instance
backtester = Backtester()
