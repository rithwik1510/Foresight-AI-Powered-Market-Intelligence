"""
Technical Feature Generator
Generates 60+ technical indicators from OHLCV data
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from ta import add_all_ta_features
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator,
    CCIIndicator, AroonIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator,
    ROCIndicator, AwesomeOscillatorIndicator
)
from ta.volatility import (
    BollingerBands, AverageTrueRange, KeltnerChannel
)
from ta.volume import (
    OnBalanceVolumeIndicator, AccDistIndexIndicator,
    MFIIndicator, VolumeWeightedAveragePrice
)

from app.ml.config import ml_settings


class TechnicalFeatureGenerator:
    """
    Generates comprehensive technical features for ML models

    Features include:
    - Price-based: Returns, momentum, rate of change
    - Moving Averages: SMA, EMA at multiple periods
    - Momentum: RSI, MACD, Stochastic, Williams %R, CCI
    - Volatility: ATR, Bollinger Bands, Historical volatility
    - Volume: OBV, MFI, Volume ratios
    - Trend: ADX, Aroon
    """

    def __init__(self):
        self.config = ml_settings.features

    def generate_features(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            include_volume: Whether to include volume-based features

        Returns:
            DataFrame with all technical features added
        """
        # Make a copy to avoid modifying original
        data = df.copy()

        # Ensure column names are correct
        data.columns = [col.capitalize() for col in data.columns]

        # 1. Price-based features
        data = self._add_return_features(data)

        # 2. Moving Average features
        data = self._add_moving_average_features(data)

        # 3. Momentum features
        data = self._add_momentum_features(data)

        # 4. Volatility features
        data = self._add_volatility_features(data)

        # 5. Volume features (if available)
        if include_volume and 'Volume' in data.columns:
            data = self._add_volume_features(data)

        # 6. Trend features
        data = self._add_trend_features(data)

        # 7. Pattern features
        data = self._add_pattern_features(data)

        # 8. Price position features
        data = self._add_price_position_features(data)

        return data

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features"""
        close = df['Close']

        # Simple returns at different periods
        for period in self.config.return_periods:
            df[f'return_{period}d'] = close.pct_change(period)
            df[f'log_return_{period}d'] = np.log(close / close.shift(period))

        # Cumulative returns
        df['cum_return_20d'] = close.pct_change(20)
        df['cum_return_60d'] = close.pct_change(60)

        # Return momentum (acceleration)
        df['return_momentum'] = df['return_5d'] - df['return_5d'].shift(5)

        # Return volatility (rolling std of returns)
        df['return_volatility_10d'] = df['return_1d'].rolling(10).std()
        df['return_volatility_20d'] = df['return_1d'].rolling(20).std()

        # Skewness of returns
        df['return_skew_20d'] = df['return_1d'].rolling(20).skew()

        return df

    def _add_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        close = df['Close']

        # Simple Moving Averages
        for period in self.config.sma_periods:
            sma = SMAIndicator(close, window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            # Price distance from SMA (%)
            df[f'price_to_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

        # Exponential Moving Averages
        for period in self.config.ema_periods:
            ema = EMAIndicator(close, window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
            # Price distance from EMA (%)
            df[f'price_to_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

        # Moving Average Crossover signals
        if 50 in self.config.sma_periods and 200 in self.config.sma_periods:
            df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
            df['sma_50_200_diff'] = (df['sma_50'] - df['sma_200']) / df['sma_200'] * 100

        if 12 in self.config.ema_periods and 26 in self.config.ema_periods:
            df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)

        # Moving average slopes (trend direction)
        df['sma_20_slope'] = df['sma_20'].diff(5) / df['sma_20'].shift(5) * 100
        df['sma_50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # RSI at multiple periods
        for period in self.config.rsi_periods:
            rsi = RSIIndicator(close, window=period)
            df[f'rsi_{period}'] = rsi.rsi()

        # MACD
        macd = MACD(
            close,
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high, low, close,
            window=self.config.stoch_k,
            smooth_window=self.config.stoch_d
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)

        # Williams %R
        williams = WilliamsRIndicator(high, low, close, lbp=14)
        df['williams_r'] = williams.williams_r()

        # Rate of Change (ROC)
        roc = ROCIndicator(close, window=12)
        df['roc_12'] = roc.roc()

        roc_20 = ROCIndicator(close, window=20)
        df['roc_20'] = roc_20.roc()

        # CCI (Commodity Channel Index)
        cci = CCIIndicator(high, low, close, window=20)
        df['cci'] = cci.cci()

        # Awesome Oscillator
        ao = AwesomeOscillatorIndicator(high, low)
        df['awesome_oscillator'] = ao.awesome_oscillator()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # Bollinger Bands
        bb = BollingerBands(
            close,
            window=self.config.bb_period,
            window_dev=self.config.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()  # %B indicator

        # Average True Range (ATR)
        atr = AverageTrueRange(high, low, close, window=self.config.atr_period)
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / close * 100  # ATR as % of price

        # Historical Volatility at different windows
        for window in self.config.volatility_windows:
            df[f'volatility_{window}d'] = close.pct_change().rolling(window).std() * np.sqrt(252) * 100

        # Keltner Channel
        kc = KeltnerChannel(high, low, close)
        df['keltner_upper'] = kc.keltner_channel_hband()
        df['keltner_lower'] = kc.keltner_channel_lband()
        df['keltner_width'] = (df['keltner_upper'] - df['keltner_lower']) / close * 100

        # Parkinson volatility estimator (using high-low range)
        df['parkinson_vol'] = np.sqrt(
            (np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2))
        ) * np.sqrt(252) * 100

        # Price range features
        df['daily_range'] = (high - low) / close * 100
        df['daily_range_ma'] = df['daily_range'].rolling(20).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Volume Moving Averages
        for period in self.config.volume_sma_periods:
            df[f'volume_sma_{period}'] = volume.rolling(period).mean()
            df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']

        # On-Balance Volume (OBV)
        obv = OnBalanceVolumeIndicator(close, volume)
        df['obv'] = obv.on_balance_volume()
        df['obv_change'] = df['obv'].pct_change(5)

        # Accumulation/Distribution Index
        adi = AccDistIndexIndicator(high, low, close, volume)
        df['adi'] = adi.acc_dist_index()
        df['adi_change'] = df['adi'].pct_change(5)

        # Money Flow Index (MFI)
        mfi = MFIIndicator(high, low, close, volume, window=14)
        df['mfi'] = mfi.money_flow_index()

        # Volume-Weighted Average Price (VWAP) - daily reset in real trading
        try:
            vwap = VolumeWeightedAveragePrice(high, low, close, volume)
            df['vwap'] = vwap.volume_weighted_average_price()
            df['price_to_vwap'] = (close - df['vwap']) / df['vwap'] * 100
        except Exception:
            # VWAP might fail with certain data
            df['vwap'] = close.rolling(20).mean()
            df['price_to_vwap'] = 0

        # Volume trend
        df['volume_trend'] = (volume.rolling(5).mean() > volume.rolling(20).mean()).astype(int)

        # Price-Volume trend
        df['pv_trend'] = (df['return_1d'] * df['volume_ratio_20']).rolling(5).mean()

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # ADX (Average Directional Index)
        adx = ADXIndicator(high, low, close, window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()  # +DI
        df['adx_neg'] = adx.adx_neg()  # -DI
        df['adx_trend_strength'] = (df['adx_pos'] - df['adx_neg']).abs()

        # Aroon Indicator
        aroon = AroonIndicator(close, window=25)
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']

        # Trend direction based on moving averages
        df['trend_sma'] = np.where(
            close > df['sma_50'],
            np.where(close > df['sma_200'], 2, 1),  # Strong uptrend, Mild uptrend
            np.where(close < df['sma_200'], -2, -1)  # Strong downtrend, Mild downtrend
        )

        # Linear regression slope
        for window in [10, 20, 50]:
            df[f'linreg_slope_{window}'] = self._calculate_linear_regression_slope(close, window)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # Higher highs / Lower lows count
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)
        df['higher_highs_5d'] = df['higher_high'].rolling(5).sum()
        df['lower_lows_5d'] = df['lower_low'].rolling(5).sum()

        # Support/Resistance proximity
        df['recent_high_20d'] = high.rolling(20).max()
        df['recent_low_20d'] = low.rolling(20).min()
        df['distance_from_high'] = (df['recent_high_20d'] - close) / close * 100
        df['distance_from_low'] = (close - df['recent_low_20d']) / close * 100

        # Gap detection
        df['gap_up'] = ((df['Open'] > high.shift(1)) & (df['Open'] > close.shift(1))).astype(int)
        df['gap_down'] = ((df['Open'] < low.shift(1)) & (df['Open'] < close.shift(1))).astype(int)

        # Candlestick patterns (simplified)
        body = close - df['Open']
        upper_shadow = high - pd.concat([close, df['Open']], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, df['Open']], axis=1).min(axis=1) - low

        df['body_size'] = body.abs() / close * 100
        df['upper_shadow_size'] = upper_shadow / close * 100
        df['lower_shadow_size'] = lower_shadow / close * 100

        # Doji detection (small body)
        df['is_doji'] = (df['body_size'] < 0.1).astype(int)

        return df

    def _add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price position features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # Price position in range
        df['price_position_20d'] = (close - low.rolling(20).min()) / (
            high.rolling(20).max() - low.rolling(20).min()
        )
        df['price_position_50d'] = (close - low.rolling(50).min()) / (
            high.rolling(50).max() - low.rolling(50).min()
        )

        # Distance from 52-week high/low
        df['distance_52w_high'] = (high.rolling(252).max() - close) / close * 100
        df['distance_52w_low'] = (close - low.rolling(252).min()) / close * 100

        # Mean reversion indicator
        mean_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df['z_score_20d'] = (close - mean_20) / std_20

        mean_50 = close.rolling(50).mean()
        std_50 = close.rolling(50).std()
        df['z_score_50d'] = (close - mean_50) / std_50

        return df

    @staticmethod
    def _calculate_linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope"""
        def linreg_slope(y):
            if len(y) < window or y.isna().any():
                return np.nan
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope

        return series.rolling(window).apply(linreg_slope, raw=False)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated"""
        features = []

        # Return features
        for period in self.config.return_periods:
            features.extend([f'return_{period}d', f'log_return_{period}d'])
        features.extend([
            'cum_return_20d', 'cum_return_60d', 'return_momentum',
            'return_volatility_10d', 'return_volatility_20d', 'return_skew_20d'
        ])

        # SMA features
        for period in self.config.sma_periods:
            features.extend([f'sma_{period}', f'price_to_sma_{period}'])

        # EMA features
        for period in self.config.ema_periods:
            features.extend([f'ema_{period}', f'price_to_ema_{period}'])

        # MA crossover features
        features.extend([
            'sma_50_200_cross', 'sma_50_200_diff', 'ema_12_26_cross',
            'sma_20_slope', 'sma_50_slope'
        ])

        # RSI features
        for period in self.config.rsi_periods:
            features.append(f'rsi_{period}')

        # Momentum features
        features.extend([
            'macd', 'macd_signal', 'macd_histogram', 'macd_cross',
            'stoch_k', 'stoch_d', 'stoch_cross', 'williams_r',
            'roc_12', 'roc_20', 'cci', 'awesome_oscillator'
        ])

        # Volatility features
        features.extend([
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'atr', 'atr_percent', 'keltner_upper', 'keltner_lower', 'keltner_width',
            'parkinson_vol', 'daily_range', 'daily_range_ma'
        ])
        for window in self.config.volatility_windows:
            features.append(f'volatility_{window}d')

        # Volume features
        for period in self.config.volume_sma_periods:
            features.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
        features.extend([
            'obv', 'obv_change', 'adi', 'adi_change', 'mfi',
            'vwap', 'price_to_vwap', 'volume_trend', 'pv_trend'
        ])

        # Trend features
        features.extend([
            'adx', 'adx_pos', 'adx_neg', 'adx_trend_strength',
            'aroon_up', 'aroon_down', 'aroon_oscillator', 'trend_sma',
            'linreg_slope_10', 'linreg_slope_20', 'linreg_slope_50'
        ])

        # Pattern features
        features.extend([
            'higher_high', 'lower_low', 'higher_highs_5d', 'lower_lows_5d',
            'recent_high_20d', 'recent_low_20d', 'distance_from_high', 'distance_from_low',
            'gap_up', 'gap_down', 'body_size', 'upper_shadow_size', 'lower_shadow_size', 'is_doji'
        ])

        # Price position features
        features.extend([
            'price_position_20d', 'price_position_50d',
            'distance_52w_high', 'distance_52w_low',
            'z_score_20d', 'z_score_50d'
        ])

        return features


# Global instance
technical_feature_generator = TechnicalFeatureGenerator()
