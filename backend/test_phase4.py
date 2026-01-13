"""
Phase 4 Test Script - Global Factors
Tests global market data and economic indicators integration
"""
import asyncio
import sys
sys.path.insert(0, '.')


async def test_global_factors():
    """Test all global factors components"""
    print("=" * 60)
    print("PHASE 4: GLOBAL FACTORS TEST")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n[1] Testing imports...")
    try:
        from app.integrations.global_markets import global_markets_client, GlobalMarketData
        from app.integrations.economic_data import economic_data_client
        from app.ml.features.global_features import global_feature_generator
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        return False

    # Test 2: Global Markets Data
    print("\n[2] Testing Global Markets Client...")
    try:
        print("   Fetching current global market data...")
        data = await global_markets_client.get_current_data()

        print(f"\n   === Global Market Snapshot ===")
        print(f"   US Markets:")
        print(f"   - S&P 500: {data.sp500_price:,.2f} ({data.sp500_change_1d:+.2%} 1D, {data.sp500_change_5d:+.2%} 5D)")
        print(f"   - NASDAQ: {data.nasdaq_price:,.2f} ({data.nasdaq_change_1d:+.2%} 1D)")
        print(f"   - VIX: {data.vix_price:.2f}")

        print(f"\n   Commodities:")
        print(f"   - Gold: ${data.gold_price:,.2f} ({data.gold_change_1d:+.2%})")
        print(f"   - Brent Oil: ${data.oil_price:.2f} ({data.oil_change_1d:+.2%})")

        print(f"\n   Forex:")
        print(f"   - USD/INR: ₹{data.usdinr_rate:.2f} ({data.usdinr_change_1d:+.2%})")
        print(f"   - Dollar Index: {data.dxy_price:.2f} ({data.dxy_change_1d:+.2%})")

        print(f"\n   Bonds:")
        print(f"   - US 10Y Yield: {data.us10y_yield:.2%}")

        print("   Global Markets: PASSED")
    except Exception as e:
        print(f"   Global Markets error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Market Regime
    print("\n[3] Testing Market Regime Detection...")
    try:
        regime = await global_markets_client.get_market_regime()

        print(f"   Current Regime: {regime['regime']}")
        print(f"   Confidence: {regime['confidence']:.2%}")
        print(f"   Risk-On Signals: {regime['signals']['risk_on']}")
        print(f"   Risk-Off Signals: {regime['signals']['risk_off']}")
        print("   Market Regime: PASSED")
    except Exception as e:
        print(f"   Market Regime error: {e}")

    # Test 4: India Impact Analysis
    print("\n[4] Testing India Market Impact...")
    try:
        impact = await global_markets_client.get_indian_market_impact()

        print(f"   Overall Impact: {impact['overall_impact']}")
        print(f"   Net Impact Score: {impact['net_impact_score']:+.2f}")
        print(f"   Market Regime: {impact['market_regime']}")

        if impact['impact_factors']:
            print(f"\n   Impact Factors:")
            for factor in impact['impact_factors'][:3]:
                sign = "+" if factor['impact'] == 'positive' else "-"
                print(f"   {sign} {factor['factor']}: {factor['description'][:50]}...")

        print("   India Impact: PASSED")
    except Exception as e:
        print(f"   India Impact error: {e}")

    # Test 5: Economic Data (FRED)
    print("\n[5] Testing Economic Data Client (FRED)...")
    try:
        if economic_data_client.is_available():
            indicators = await economic_data_client.get_current_indicators()

            if indicators:
                print(f"\n   === US Economic Indicators ===")
                print(f"   Interest Rates:")
                print(f"   - Fed Funds Rate: {indicators.fed_funds_rate:.2%}")

                print(f"\n   Inflation:")
                print(f"   - CPI (YoY): {indicators.cpi_yoy:.2%}")
                print(f"   - PCE (YoY): {indicators.pce_yoy:.2%}")

                print(f"\n   Employment:")
                print(f"   - Unemployment Rate: {indicators.unemployment_rate:.1%}")
                print(f"   - Nonfarm Payrolls Change: {indicators.nonfarm_payrolls_change:+,.0f}K")

                print(f"\n   Other:")
                print(f"   - GDP Growth: {indicators.gdp_growth_rate:.1%}")
                print(f"   - Yield Curve (10Y-2Y): {indicators.yield_curve_spread:.2%}")
                print(f"   - Consumer Sentiment: {indicators.consumer_sentiment:.1f}")

                print("   FRED Economic Data: PASSED")
            else:
                print("   FRED: No data returned")
        else:
            print("   FRED: NOT CONFIGURED (need FRED_API_KEY)")
            print("   FRED: SKIPPED")
    except Exception as e:
        print(f"   FRED error: {e}")

    # Test 6: Economic Regime
    print("\n[6] Testing Economic Regime Detection...")
    try:
        if economic_data_client.is_available():
            econ_regime = await economic_data_client.get_economic_regime()

            print(f"   Economic Regime: {econ_regime['regime']}")
            print(f"   Confidence: {econ_regime['confidence']:.2%}")
            print(f"\n   Signals:")
            for signal, count in econ_regime['signals'].items():
                print(f"   - {signal}: {count}")

            print(f"\n   Implications for India:")
            implications = econ_regime['implications']
            print(f"   - Stocks: {implications['stocks'][:50]}...")
            print(f"   - Emerging Markets: {implications['emerging_markets'][:50]}...")

            print("   Economic Regime: PASSED")
        else:
            print("   Economic Regime: SKIPPED (FRED not configured)")
    except Exception as e:
        print(f"   Economic Regime error: {e}")

    # Test 7: Fed Outlook
    print("\n[7] Testing Fed Outlook Analysis...")
    try:
        if economic_data_client.is_available():
            fed = await economic_data_client.get_fed_outlook()

            print(f"   Fed Outlook: {fed['outlook']}")
            print(f"   Direction: {fed['direction']}")
            print(f"   Hawkish Signals: {fed['hawkish_signals']}")
            print(f"   Dovish Signals: {fed['dovish_signals']}")
            print(f"\n   India Impact: {fed['impact_on_india'][:60]}...")

            print("   Fed Outlook: PASSED")
        else:
            print("   Fed Outlook: SKIPPED (FRED not configured)")
    except Exception as e:
        print(f"   Fed Outlook error: {e}")

    # Test 8: Global Feature Generator
    print("\n[8] Testing Global Feature Generator...")
    try:
        # Current features
        features = await global_feature_generator.generate_features_async(include_economic=False)

        print(f"   Generated {len(features)} current features")
        print(f"   Sample features:")
        for i, (name, value) in enumerate(list(features.items())[:5]):
            print(f"   - {name}: {value:.4f}")

        # Historical features
        print("\n   Generating historical features...")
        hist_features = await global_feature_generator.generate_historical_features(period="3mo")
        print(f"   Historical features shape: {hist_features.shape}")
        print(f"   Date range: {hist_features.index[0]} to {hist_features.index[-1]}")

        print("   Global Feature Generator: PASSED")
    except Exception as e:
        print(f"   Global Feature Generator error: {e}")
        import traceback
        traceback.print_exc()

    # Test 9: Stock Correlation with Global Factors
    print("\n[9] Testing Stock-Global Correlation...")
    try:
        correlations = await global_markets_client.get_correlation_data(
            indian_stock_symbol="RELIANCE.NS",
            period="1y"
        )

        if correlations:
            print(f"   RELIANCE.NS correlations with global factors:")
            for factor, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"   - {factor}: {corr:+.3f}")
            print("   Correlation Analysis: PASSED")
        else:
            print("   No correlation data available")
    except Exception as e:
        print(f"   Correlation error: {e}")

    print("\n" + "=" * 60)
    print("PHASE 4 TESTS COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_global_factors())
