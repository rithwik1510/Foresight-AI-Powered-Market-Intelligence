"""
Phase 5 Test Script - Ensemble Prediction & Backtesting
Tests the complete prediction pipeline
"""
import asyncio
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime


async def test_ensemble_pipeline():
    """Test ensemble prediction pipeline"""
    print("=" * 60)
    print("PHASE 5: ENSEMBLE & BACKTESTING TEST")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n[1] Testing imports...")
    try:
        from app.ml.features import feature_pipeline, technical_feature_generator
        from app.ml.prediction import (
            EnsemblePredictor, ensemble_predictor,
            EnsemblePrediction
        )
        from app.ml.backtesting import Backtester, backtester
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Fetch and prepare data
    print("\n[2] Fetching RELIANCE.NS data...")
    try:
        import yfinance as yf

        ticker = yf.Ticker("RELIANCE.NS")
        df = ticker.history(period="2y")

        if df.empty:
            print("   WARNING: No data, using synthetic")
            dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
            df = pd.DataFrame({
                'Open': np.random.uniform(2400, 2600, 500),
                'High': np.random.uniform(2450, 2650, 500),
                'Low': np.random.uniform(2350, 2550, 500),
                'Close': np.random.uniform(2400, 2600, 500),
                'Volume': np.random.uniform(1e7, 5e7, 500)
            }, index=dates)

        print(f"   Data shape: {df.shape}")
        current_price = df['Close'].iloc[-1]
        print(f"   Current price: ₹{current_price:.2f}")
    except Exception as e:
        print(f"   Data fetch error: {e}")
        return False

    # Test 3: Generate features
    print("\n[3] Generating features...")
    try:
        features = technical_feature_generator.generate_features(df)
        target = df['Close'].pct_change(20).shift(-20)

        # Align and clean
        valid_idx = features.index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        prices = df.loc[valid_idx, 'Close']

        features = features.ffill().bfill()

        print(f"   Features shape: {features.shape}")
        print(f"   Target shape: {target.shape}")
    except Exception as e:
        print(f"   Feature generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Train ensemble
    print("\n[4] Training ensemble model...")
    try:
        ensemble = EnsemblePredictor()
        metrics = ensemble.train_all(
            X=features,
            y=target,
            y_prices=prices,
            validation_split=0.2
        )

        print(f"\n   === Training Results ===")
        for model_name, m in metrics.items():
            acc = m.directional_accuracy if m.directional_accuracy else m.accuracy
            print(f"   {model_name}: {acc:.2%}" if acc else f"   {model_name}: N/A")

        print(f"\n   Trained models: {ensemble.trained_models}")
        print(f"   Model weights: {ensemble.base_weights}")
        print("   Ensemble Training: PASSED")
    except Exception as e:
        print(f"   Ensemble training error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Generate ensemble prediction
    print("\n[5] Generating ensemble prediction...")
    try:
        prediction = ensemble.predict(
            X=features,
            current_price=current_price,
            symbol="RELIANCE.NS",
            horizon_days=30
        )

        print(f"\n   === RELIANCE.NS Ensemble Prediction ===")
        print(f"   Direction: {prediction.direction.value}")
        print(f"   Direction Probability: {prediction.direction_probability:.2%}")
        print(f"   Current Price: ₹{prediction.current_price:.2f}")
        print(f"   Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   Predicted Return: {prediction.predicted_return:.2%}")
        print(f"   Price Range: ₹{prediction.price_lower:.2f} - ₹{prediction.price_upper:.2f}")
        print(f"   Confidence: {prediction.confidence:.2%}")
        print(f"   Model Agreement: {prediction.model_agreement:.2%}")
        print(f"   Risk Level: {prediction.risk_level}")

        print(f"\n   Model Breakdown:")
        for model_name, model_pred in prediction.model_predictions.items():
            weight = prediction.model_weights.get(model_name, 0)
            print(f"   - {model_name}: {model_pred.direction.value} ({model_pred.predicted_return:.2%}) [weight: {weight:.1%}]")

        if prediction.top_bullish_factors:
            print(f"\n   Bullish Factors:")
            for factor in prediction.top_bullish_factors[:2]:
                print(f"   + {factor}")

        if prediction.top_bearish_factors:
            print(f"\n   Bearish Factors:")
            for factor in prediction.top_bearish_factors[:2]:
                print(f"   - {factor}")

        print("\n   Ensemble Prediction: PASSED")
    except Exception as e:
        print(f"   Ensemble prediction error: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Quick backtest
    print("\n[6] Running quick backtest (this may take a few minutes)...")
    try:
        result = backtester.quick_backtest(
            symbol="RELIANCE.NS",
            test_days=30,  # Shorter for quick test
            train_days=180,
            horizon_days=10
        )

        print(f"\n   === Backtest Results ===")
        print(f"   Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"   Total Predictions: {result.total_predictions}")
        print(f"   Correct Predictions: {result.correct_predictions}")
        print(f"   Directional Accuracy: {result.directional_accuracy:.2%}")

        print(f"\n   Returns:")
        print(f"   - Total Return: {result.total_return:.2%}")
        print(f"   - Annualized Return: {result.annualized_return:.2%}")
        print(f"   - Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   - Max Drawdown: {result.max_drawdown:.2%}")

        print(f"\n   By Direction:")
        print(f"   - Bullish Accuracy: {result.bullish_accuracy:.2%}")
        print(f"   - Bearish Accuracy: {result.bearish_accuracy:.2%}")

        print(f"\n   By Confidence:")
        print(f"   - High Confidence (>0.7): {result.high_confidence_accuracy:.2%}")
        print(f"   - Medium Confidence: {result.medium_confidence_accuracy:.2%}")
        print(f"   - Low Confidence: {result.low_confidence_accuracy:.2%}")

        if result.model_accuracies:
            print(f"\n   Model Accuracies:")
            for model, acc in result.model_accuracies.items():
                print(f"   - {model}: {acc:.2%}")

        print("\n   Backtesting: PASSED")
    except Exception as e:
        print(f"   Backtest error: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Full feature pipeline (async)
    print("\n[7] Testing full feature pipeline (async)...")
    try:
        # Generate prediction features
        pred_features = await feature_pipeline.generate_prediction_features(
            symbol="RELIANCE.NS",
            company_name="Reliance Industries"
        )

        print(f"   Prediction features shape: {pred_features.shape}")
        print(f"   Feature columns: {len(pred_features.columns)}")
        print(f"   Sample features: {list(pred_features.columns[:5])}")

        print("   Feature Pipeline: PASSED")
    except Exception as e:
        print(f"   Feature pipeline error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PHASE 5 TESTS COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_ensemble_pipeline())
