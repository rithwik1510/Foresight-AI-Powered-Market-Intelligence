"""
Phase 2 Test Script - ML Prediction Models
Tests all 5 prediction models with real RELIANCE.NS data
"""
import asyncio
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_all_models():
    """Test all ML prediction models"""
    print("=" * 60)
    print("PHASE 2: ML PREDICTION MODELS TEST")
    print("=" * 60)

    # Test 1: Import all models
    print("\n[1] Testing imports...")
    try:
        from app.ml.prediction import (
            ARIMAPredictor, arima_predictor,
            ProphetPredictor, prophet_predictor,
            XGBoostPredictor, xgboost_predictor,
            LightGBMPredictor, lightgbm_predictor,
            RandomForestPredictor, random_forest_predictor,
            BasePredictor, PredictionResult, ModelMetrics
        )
        from app.ml.features import TechnicalFeatureGenerator, technical_feature_generator
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        return False

    # Test 2: Fetch real data
    print("\n[2] Fetching RELIANCE.NS data...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("RELIANCE.NS")
        df = ticker.history(period="2y")

        if df.empty:
            print("   WARNING: No data received, using synthetic data")
            # Create synthetic data for testing
            dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
            df = pd.DataFrame({
                'Open': np.random.uniform(2400, 2600, 500),
                'High': np.random.uniform(2450, 2650, 500),
                'Low': np.random.uniform(2350, 2550, 500),
                'Close': np.random.uniform(2400, 2600, 500),
                'Volume': np.random.uniform(1e7, 5e7, 500)
            }, index=dates)
            # Make it more realistic with trend
            df['Close'] = df['Close'].cumsum() / df['Close'].cumsum().max() * 2500 + 100
            df['Open'] = df['Close'] * np.random.uniform(0.99, 1.01, 500)
            df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.02, 500)
            df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.98, 1.0, 500)

        print(f"   Data shape: {df.shape}")
        print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        current_price = df['Close'].iloc[-1]
        print(f"   Current price: ₹{current_price:.2f}")
    except Exception as e:
        print(f"   Data fetch error: {e}")
        return False

    # Test 3: Generate features
    print("\n[3] Generating technical features...")
    try:
        features_df = technical_feature_generator.generate_features(df)
        print(f"   Features generated: {len(features_df.columns)} columns")
        print(f"   Sample features: {list(features_df.columns[:5])}")

        # Create target (forward returns)
        features_df['target'] = df['Close'].pct_change(20).shift(-20)  # 20-day forward return
        features_df = features_df.dropna()
        print(f"   Clean data shape: {features_df.shape}")
    except Exception as e:
        print(f"   Feature generation error: {e}")
        return False

    # Prepare data for models
    X = features_df.drop(columns=['target'])
    y = features_df['target']
    y_prices = df.loc[features_df.index, 'Close']

    # Test 4: ARIMA Model
    print("\n[4] Testing ARIMA Predictor...")
    try:
        arima = ARIMAPredictor()
        metrics = arima.train(X, y_prices, validation_split=0.2)
        print(f"   Training complete!")
        print(f"   - MAE: {metrics.mae:.4f}" if metrics.mae else "   - MAE: N/A")
        print(f"   - RMSE: {metrics.rmse:.4f}" if metrics.rmse else "   - RMSE: N/A")
        print(f"   - Directional Accuracy: {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else "   - Directional Accuracy: N/A")

        prediction = arima.predict(X, current_price, "RELIANCE.NS", horizon_days=30)
        print(f"   Prediction:")
        print(f"   - Direction: {prediction.direction.value}")
        print(f"   - Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   - Predicted Return: {prediction.predicted_return:.2%}")
        print(f"   - Confidence: {prediction.confidence:.2%}")
        print("   ARIMA: PASSED")
    except Exception as e:
        print(f"   ARIMA error: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Prophet Model
    print("\n[5] Testing Prophet Predictor...")
    try:
        prophet = ProphetPredictor()
        metrics = prophet.train(X, y_prices, validation_split=0.2)
        print(f"   Training complete!")
        print(f"   - MAE: {metrics.mae:.4f}" if metrics.mae else "   - MAE: N/A")
        print(f"   - RMSE: {metrics.rmse:.4f}" if metrics.rmse else "   - RMSE: N/A")
        print(f"   - Directional Accuracy: {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else "   - Directional Accuracy: N/A")

        prediction = prophet.predict(X, current_price, "RELIANCE.NS", horizon_days=30)
        print(f"   Prediction:")
        print(f"   - Direction: {prediction.direction.value}")
        print(f"   - Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   - Predicted Return: {prediction.predicted_return:.2%}")
        print(f"   - Confidence: {prediction.confidence:.2%}")
        print("   Prophet: PASSED")
    except Exception as e:
        print(f"   Prophet error: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: XGBoost Model
    print("\n[6] Testing XGBoost Predictor...")
    try:
        xgb = XGBoostPredictor()
        metrics = xgb.train(X, y, validation_split=0.2)
        print(f"   Training complete!")
        print(f"   - Accuracy: {metrics.accuracy:.2%}" if metrics.accuracy else "   - Accuracy: N/A")
        print(f"   - F1 Score: {metrics.f1_score:.4f}" if metrics.f1_score else "   - F1 Score: N/A")
        print(f"   - Directional Accuracy: {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else "   - Directional Accuracy: N/A")

        prediction = xgb.predict(X, current_price, "RELIANCE.NS", horizon_days=30)
        print(f"   Prediction:")
        print(f"   - Direction: {prediction.direction.value}")
        print(f"   - Direction Probability: {prediction.direction_probability:.2%}")
        print(f"   - Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   - Confidence: {prediction.confidence:.2%}")

        # Feature importance
        importance = xgb.get_feature_importance()
        if importance:
            top_features = list(importance.items())[:3]
            print(f"   Top features: {top_features}")
        print("   XGBoost: PASSED")
    except Exception as e:
        print(f"   XGBoost error: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: LightGBM Model
    print("\n[7] Testing LightGBM Predictor...")
    try:
        lgbm = LightGBMPredictor()
        metrics = lgbm.train(X, y, validation_split=0.2)
        print(f"   Training complete!")
        print(f"   - MAE: {metrics.mae:.4f}" if metrics.mae else "   - MAE: N/A")
        print(f"   - RMSE: {metrics.rmse:.4f}" if metrics.rmse else "   - RMSE: N/A")
        print(f"   - Directional Accuracy: {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else "   - Directional Accuracy: N/A")

        prediction = lgbm.predict(X, current_price, "RELIANCE.NS", horizon_days=30)
        print(f"   Prediction:")
        print(f"   - Direction: {prediction.direction.value}")
        print(f"   - Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   - Predicted Return: {prediction.predicted_return:.2%}")
        print(f"   - Confidence: {prediction.confidence:.2%}")

        # Feature importance
        importance = lgbm.get_feature_importance()
        if importance:
            top_features = list(importance.items())[:3]
            print(f"   Top features: {top_features}")
        print("   LightGBM: PASSED")
    except Exception as e:
        print(f"   LightGBM error: {e}")
        import traceback
        traceback.print_exc()

    # Test 8: Random Forest Model
    print("\n[8] Testing Random Forest Predictor...")
    try:
        rf = RandomForestPredictor()
        metrics = rf.train(X, y, validation_split=0.2)
        print(f"   Training complete!")
        print(f"   - Accuracy: {metrics.accuracy:.2%}" if metrics.accuracy else "   - Accuracy: N/A")
        print(f"   - F1 Score: {metrics.f1_score:.4f}" if metrics.f1_score else "   - F1 Score: N/A")
        print(f"   - Directional Accuracy: {metrics.directional_accuracy:.2%}" if metrics.directional_accuracy else "   - Directional Accuracy: N/A")

        prediction = rf.predict(X, current_price, "RELIANCE.NS", horizon_days=30)
        print(f"   Prediction:")
        print(f"   - Direction: {prediction.direction.value}")
        print(f"   - Direction Probability: {prediction.direction_probability:.2%}")
        print(f"   - Predicted Price: ₹{prediction.predicted_price:.2f}")
        print(f"   - Confidence: {prediction.confidence:.2%}")

        # Tree insights
        insights = rf.get_tree_insights()
        if insights:
            print(f"   Tree insights: {insights}")
        print("   Random Forest: PASSED")
    except Exception as e:
        print(f"   Random Forest error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PHASE 2 TESTS COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_all_models()
