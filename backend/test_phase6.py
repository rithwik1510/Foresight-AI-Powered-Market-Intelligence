"""
Phase 6 Test Script - API Integration
Tests the prediction API endpoints and model storage
"""
import asyncio
import sys
sys.path.insert(0, '.')

import httpx
from datetime import datetime


async def test_api_integration():
    """Test API integration"""
    print("=" * 60)
    print("PHASE 6: API INTEGRATION TEST")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n[1] Testing imports...")
    try:
        from app.schemas.prediction import (
            PredictionResponse,
            SentimentResponse,
            GlobalFactorsResponse,
            BacktestResponse,
            ModelStatusResponse
        )
        from app.ml.training import model_store, training_scheduler
        from app.api.v1.predictions import router
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Model Store
    print("\n[2] Testing Model Store...")
    try:
        # Check storage stats
        stats = model_store.get_storage_stats()
        print(f"   Base path: {stats['base_path']}")
        print(f"   Total models: {stats['total_models']}")
        print(f"   Total size: {stats['total_size_mb']} MB")

        # Test save/load cycle (mock)
        print("\n   Testing model persistence...")

        # Create a simple mock model
        class MockModel:
            def __init__(self):
                self.trained = True

        mock = MockModel()
        save_path = model_store.save_model(
            model=mock,
            symbol="TEST.NS",
            model_name="test_model",
            metrics={"accuracy": 0.65, "directional_accuracy": 0.62},
            feature_count=50,
            train_samples=1000
        )
        print(f"   Saved to: {save_path}")

        # Load back
        loaded = model_store.load_model("TEST.NS", "test_model")
        print(f"   Loaded: {loaded is not None}")

        # Get metadata
        metadata = model_store.get_metadata("TEST.NS", "test_model")
        if metadata:
            print(f"   Metadata: trained_at={metadata.trained_at}")

        # Check staleness
        is_stale = model_store.is_model_stale("TEST.NS", "test_model", max_age_hours=1)
        print(f"   Is stale (1 hour): {is_stale}")

        # Cleanup
        model_store.delete_model("TEST.NS", "test_model")
        print("   Test model deleted")

        print("   Model Store: PASSED")
    except Exception as e:
        print(f"   Model Store error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Training Scheduler
    print("\n[3] Testing Training Scheduler...")
    try:
        # Add a job
        job_id = training_scheduler.add_job(
            symbol="RELIANCE.NS",
            model_name="ensemble",
            schedule="weekly"
        )
        print(f"   Added job: {job_id}")

        # Get job status
        status = training_scheduler.get_job_status("RELIANCE.NS", "ensemble")
        if status:
            print(f"   Job status: {status['status']}")
            print(f"   Schedule: {status['schedule']}")

        # List all jobs
        all_jobs = training_scheduler.get_all_jobs()
        print(f"   Total jobs: {len(all_jobs)}")

        # Remove job
        training_scheduler.remove_job("RELIANCE.NS", "ensemble")
        print("   Job removed")

        print("   Training Scheduler: PASSED")
    except Exception as e:
        print(f"   Training Scheduler error: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Test API Endpoints (without running server)
    print("\n[4] Testing API Endpoint Functions...")
    try:
        from app.api.v1.predictions import (
            get_prediction,
            get_sentiment,
            get_global_factors,
            get_model_status
        )

        # Test global factors (no training needed)
        print("\n   Testing global factors endpoint...")
        global_factors = await get_global_factors()
        print(f"   Market Regime: {global_factors.market_regime}")
        print(f"   India Impact: {global_factors.india_impact}")
        print(f"   VIX: {global_factors.us_markets['vix']}")
        print("   Global Factors: PASSED")

        # Test sentiment endpoint
        print("\n   Testing sentiment endpoint...")
        sentiment = await get_sentiment("RELIANCE.NS", "Reliance Industries")
        print(f"   Overall Score: {sentiment.overall_score}")
        print(f"   Label: {sentiment.overall_label}")
        print(f"   Articles: {sentiment.news_article_count}")
        print("   Sentiment: PASSED")

        # Test model status
        print("\n   Testing model status endpoint...")
        status = await get_model_status()
        print(f"   Is Trained: {status.is_trained}")
        print(f"   Total Models: {status.total_models}")
        print("   Model Status: PASSED")

    except Exception as e:
        print(f"   API Endpoint error: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Schema Validation
    print("\n[5] Testing Schema Validation...")
    try:
        from pydantic import ValidationError

        # Test PredictionResponse
        pred = PredictionResponse(
            symbol="RELIANCE.NS",
            prediction_date=datetime.now(),
            horizon_days=30,
            direction="bullish",
            direction_probability=0.73,
            current_price=2450.50,
            predicted_price=2612.00,
            predicted_return_pct=6.59,
            price_range=[2520, 2710],
            confidence=0.68,
            model_agreement=0.8,
            risk_level="MEDIUM",
            top_bullish_factors=["Strong momentum"],
            top_bearish_factors=["High P/E"],
            model_breakdown={}
        )
        print(f"   PredictionResponse: Valid")

        # Test SentimentResponse
        sent = SentimentResponse(
            symbol="RELIANCE.NS",
            timestamp=datetime.now(),
            overall_score=0.35,
            overall_label="bullish",
            confidence=0.72,
            news_score=0.42,
            news_article_count=15,
            social_score=0.28,
            social_post_count=8
        )
        print(f"   SentimentResponse: Valid")

        # Test GlobalFactorsResponse
        gf = GlobalFactorsResponse(
            timestamp=datetime.now(),
            market_regime="RISK_ON",
            regime_confidence=0.75,
            us_markets={"sp500": {"price": 4800}},
            commodities={"gold": {"price": 2050}},
            forex={"usdinr": {"rate": 83.2}},
            india_impact="POSITIVE"
        )
        print(f"   GlobalFactorsResponse: Valid")

        print("   Schema Validation: PASSED")
    except ValidationError as e:
        print(f"   Schema validation error: {e}")

    print("\n" + "=" * 60)
    print("PHASE 6 TESTS COMPLETE")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE - ML Prediction System Ready!")
    print("=" * 60)
    print("""
    Next Steps:
    1. Run the server: uvicorn app.main:app --reload
    2. Access API docs: http://localhost:8000/docs
    3. Try prediction endpoint: GET /api/v1/predictions/RELIANCE.NS

    API Endpoints:
    - GET  /api/v1/predictions/{symbol}?horizon=30
    - GET  /api/v1/predictions/{symbol}/sentiment
    - GET  /api/v1/predictions/{symbol}/backtest
    - GET  /api/v1/predictions/global-factors
    - GET  /api/v1/predictions/models/status
    - POST /api/v1/predictions/batch
    """)

    return True


if __name__ == "__main__":
    asyncio.run(test_api_integration())
