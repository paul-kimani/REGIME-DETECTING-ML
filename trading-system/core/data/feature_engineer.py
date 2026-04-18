"""FeatureEngineer — computes all model features: trend, volatility, oscillators, statistical, MTF."""

import math
import numpy as np
import pandas as pd
import pandas_ta as ta

from core.utils.helpers import atr, z_score, hurst_exponent, rolling_corr, encode_session


class FeatureEngineer:
    """Computes all trading features for model input.

    All computations are strictly look-ahead-free — only past data is used
    at every step (rolling windows, lags, shifts).
    """

    def compute(self, df: pd.DataFrame, mtf_data: dict = None, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Compute all features. Returns DataFrame with all feature columns appended.

        Input df must have columns: timestamp, open, high, low, close, volume
        mtf_data: optional dict with aligned H1/H4 data (from align_timeframes)
        All computations are strictly look-ahead-free (only past data used).
        """
        df = df.copy()

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # ------------------------------------------------------------------
        # 1. Price / returns
        # ------------------------------------------------------------------
        df["log_return"] = np.log(close / close.shift(1))
        df["returns_1"] = close.pct_change(1)
        df["returns_3"] = close.pct_change(3)
        df["returns_6"] = close.pct_change(6)
        df["returns_12"] = close.pct_change(12)
        df["returns_24"] = close.pct_change(24)

        # ------------------------------------------------------------------
        # 2. Trend (pandas_ta)
        # ------------------------------------------------------------------
        adx_df = ta.adx(high, low, close, length=14)
        df["adx_14"] = adx_df["ADX_14"]
        df["plus_di_14"] = adx_df["DMP_14"]
        df["minus_di_14"] = adx_df["DMN_14"]
        df["adx_slope_3"] = df["adx_14"].diff(3)
        df["di_spread"] = df["plus_di_14"] - df["minus_di_14"]

        # ------------------------------------------------------------------
        # 3. EMA structure (normalised by ATR)
        # ------------------------------------------------------------------
        atr_14 = atr(high, low, close, 14)
        df["atr_14"] = atr_14

        df["ema21"] = ta.ema(close, length=21)
        df["ema50"] = ta.ema(close, length=50)
        df["ema200"] = ta.ema(close, length=200)

        df["price_vs_ema21"] = (close - df["ema21"]) / atr_14
        df["price_vs_ema50"] = (close - df["ema50"]) / atr_14
        df["price_vs_ema200"] = (close - df["ema200"]) / atr_14
        df["ema21_vs_ema50"] = (df["ema21"] - df["ema50"]) / atr_14
        df["ema50_vs_ema200"] = (df["ema50"] - df["ema200"]) / atr_14

        # ------------------------------------------------------------------
        # 4. Volatility
        # ------------------------------------------------------------------
        df["atr_50_mean"] = atr_14.rolling(50).mean()
        df["atr_ratio"] = atr_14 / df["atr_50_mean"]

        bb_df = ta.bbands(close, length=20, std=2)
        df["bb_upper"] = bb_df["BBU_20_2.0"]
        df["bb_lower"] = bb_df["BBL_20_2.0"]
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

        df["realised_vol_20"] = (
            close.pct_change().rolling(20).std() * math.sqrt(252 * 96)
        )

        # ------------------------------------------------------------------
        # 5. Oscillators
        # ------------------------------------------------------------------
        df["rsi_14"] = ta.rsi(close, length=14)
        df["rsi_7"] = ta.rsi(close, length=7)
        df["rsi_slope_3"] = df["rsi_14"].diff(3)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        df["macd_line"] = macd_df["MACD_12_26_9"]
        df["macd_signal"] = macd_df["MACDs_12_26_9"]
        df["macd_hist"] = macd_df["MACDh_12_26_9"]
        df["macd_hist_slope"] = df["macd_hist"].diff(3)

        stoch_df = ta.stoch(high, low, close, k=14, d=3)
        df["stoch_k"] = stoch_df["STOCHk_14_3_3"]
        df["stoch_d"] = stoch_df["STOCHd_14_3_3"]

        df["cci_14"] = ta.cci(high, low, close, length=14)

        # ------------------------------------------------------------------
        # 6. Statistical
        # ------------------------------------------------------------------
        df["z_score_50"] = z_score(close, 50)
        df["hurst_100"] = (
            close.rolling(100, min_periods=100)
            .apply(lambda x: hurst_exponent(pd.Series(x)), raw=False)
        )
        df["autocorr_1"] = (
            close.pct_change()
            .rolling(30)
            .apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        )

        # ------------------------------------------------------------------
        # 7. Volume
        # ------------------------------------------------------------------
        df["volume_ratio"] = volume / volume.rolling(20).mean()
        df["volume_slope_3"] = df["volume_ratio"].diff(3)

        # ------------------------------------------------------------------
        # 8. Session / time
        # ------------------------------------------------------------------
        timestamps = df["timestamp"]
        df["session"] = timestamps.apply(encode_session)

        hour = timestamps.dt.hour
        df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * math.pi * hour / 24)
        df["day_of_week"] = timestamps.dt.dayofweek
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # ------------------------------------------------------------------
        # 9. MTF features
        # ------------------------------------------------------------------
        if mtf_data is not None:
            h4 = mtf_data.get("H4")
            h1 = mtf_data.get("H1")

            if h4 is not None and "regime" in h4.columns:
                df["regime_h4"] = h4["regime"].values[: len(df)] if len(h4) >= len(df) else 0
            else:
                df["regime_h4"] = 0

            if h1 is not None and "regime" in h1.columns:
                df["regime_h1"] = h1["regime"].values[: len(df)] if len(h1) >= len(df) else 0
            else:
                df["regime_h1"] = 0

            if h4 is not None and "regime_confidence" in h4.columns:
                df["regime_confidence_h4"] = (
                    h4["regime_confidence"].values[: len(df)] if len(h4) >= len(df) else 0.0
                )
            else:
                df["regime_confidence_h4"] = 0.0

            if h1 is not None and "regime_confidence" in h1.columns:
                df["regime_confidence_h1"] = (
                    h1["regime_confidence"].values[: len(df)] if len(h1) >= len(df) else 0.0
                )
            else:
                df["regime_confidence_h1"] = 0.0

            df["mtf_alignment_score"] = float(mtf_data.get("mtf_alignment_score", 0.0))
        else:
            df["regime_h4"] = 0
            df["regime_h1"] = 0
            df["regime_confidence_h4"] = 0.0
            df["regime_confidence_h1"] = 0.0
            df["mtf_alignment_score"] = 0.0

        # ------------------------------------------------------------------
        # 10. Asset-specific Gold features (only for XAUUSD)
        # ------------------------------------------------------------------
        if symbol == "XAUUSD":
            xauusd_returns = close.pct_change()

            try:
                dxy_close = mtf_data["DXY"]["close"] if (mtf_data and "DXY" in mtf_data) else None
                xagusd_close = mtf_data["XAGUSD"]["close"] if (mtf_data and "XAGUSD" in mtf_data) else None
            except (TypeError, KeyError):
                dxy_close = None
                xagusd_close = None

            if dxy_close is not None:
                dxy_returns = dxy_close.pct_change()
                df["gold_dxy_corr_20"] = rolling_corr(xauusd_returns, dxy_returns, 20)
            else:
                df["gold_dxy_corr_20"] = 0.0

            if xagusd_close is not None:
                xagusd_returns = xagusd_close.pct_change()
                df["gold_silver_corr_20"] = rolling_corr(xauusd_returns, xagusd_returns, 20)
                df["gold_silver_ratio"] = close / xagusd_close.replace(0, np.nan)
            else:
                df["gold_silver_corr_20"] = 0.0
                df["gold_silver_ratio"] = 0.0

            df["gold_silver_ratio_mom_10"] = (
                df["gold_silver_ratio"].pct_change(10)
                if xagusd_close is not None
                else 0.0
            )
            df["gold_dxy_corr_regime"] = df["gold_dxy_corr_20"] - (-0.65)
        else:
            df["gold_dxy_corr_20"] = 0.0
            df["gold_silver_corr_20"] = 0.0
            df["gold_silver_ratio"] = 0.0
            df["gold_silver_ratio_mom_10"] = 0.0
            df["gold_dxy_corr_regime"] = 0.0

        # ------------------------------------------------------------------
        # Drop rows with NaN in the 5-bar warmup minimum essential features
        # ------------------------------------------------------------------
        warmup_cols = ["log_return", "returns_1", "rsi_14", "atr_14"]
        df = df.dropna(subset=warmup_cols).reset_index(drop=True)

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of all feature column names."""
        return [
            # 1. Price / returns
            "log_return",
            "returns_1",
            "returns_3",
            "returns_6",
            "returns_12",
            "returns_24",
            # 2. Trend
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "adx_slope_3",
            "di_spread",
            # 3. EMA structure
            "ema21",
            "ema50",
            "ema200",
            "price_vs_ema21",
            "price_vs_ema50",
            "price_vs_ema200",
            "ema21_vs_ema50",
            "ema50_vs_ema200",
            # 4. Volatility
            "atr_14",
            "atr_50_mean",
            "atr_ratio",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_position",
            "realised_vol_20",
            # 5. Oscillators
            "rsi_14",
            "rsi_7",
            "rsi_slope_3",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "macd_hist_slope",
            "stoch_k",
            "stoch_d",
            "cci_14",
            # 6. Statistical
            "z_score_50",
            "hurst_100",
            "autocorr_1",
            # 7. Volume
            "volume_ratio",
            "volume_slope_3",
            # 8. Session / time
            "session",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "is_monday",
            "is_friday",
            # 9. MTF
            "regime_h4",
            "regime_h1",
            "regime_confidence_h4",
            "regime_confidence_h1",
            "mtf_alignment_score",
            # 10. Gold-specific
            "gold_dxy_corr_20",
            "gold_silver_corr_20",
            "gold_silver_ratio",
            "gold_silver_ratio_mom_10",
            "gold_dxy_corr_regime",
        ]
