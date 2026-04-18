"""Shared utility functions: lot rounding, pip value, ATR, Hurst exponent, z-score, session encoding."""

import math
import numpy as np
import pandas as pd


def round_to_lot_step(volume: float, step: float) -> float:
    """Round volume down to the nearest lot step.

    Uses math.floor to ensure we never over-size a position.

    Args:
        volume: Raw desired volume in lots.
        step: The broker's minimum lot increment (e.g. 0.01).

    Returns:
        Volume floored to the nearest multiple of step.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return math.floor(volume / step) * step


# ---------------------------------------------------------------------------
# Pip-value lookup table
# key   -> (pip_size, contract_size)
# contract_size is in units of the base currency (or oz for metals)
# ---------------------------------------------------------------------------
_PIP_TABLE: dict[str, tuple[float, float]] = {
    # Major forex — standard 100 000 unit contract
    "EURUSD": (0.0001, 100_000),
    "GBPUSD": (0.0001, 100_000),
    "AUDUSD": (0.0001, 100_000),
    "NZDUSD": (0.0001, 100_000),
    "USDCAD": (0.0001, 100_000),
    "USDCHF": (0.0001, 100_000),
    "USDJPY": (0.01,   100_000),
    "EURJPY": (0.01,   100_000),
    "GBPJPY": (0.01,   100_000),
    "AUDJPY": (0.01,   100_000),
    "CHFJPY": (0.01,   100_000),
    "CADJPY": (0.01,   100_000),
    "NZDJPY": (0.01,   100_000),
    "EURGBP": (0.0001, 100_000),
    "EURAUD": (0.0001, 100_000),
    "EURCAD": (0.0001, 100_000),
    "EURNZD": (0.0001, 100_000),
    "EURCHF": (0.0001, 100_000),
    "GBPAUD": (0.0001, 100_000),
    "GBPCAD": (0.0001, 100_000),
    "GBPCHF": (0.0001, 100_000),
    "GBPNZD": (0.0001, 100_000),
    "AUDCAD": (0.0001, 100_000),
    "AUDCHF": (0.0001, 100_000),
    "AUDNZD": (0.0001, 100_000),
    "CADCHF": (0.0001, 100_000),
    "NZDCAD": (0.0001, 100_000),
    "NZDCHF": (0.0001, 100_000),
    # Precious metals
    "XAUUSD": (0.01,   100),   # 100 oz contract
    "XAGUSD": (0.001,  100),   # 100 oz contract
    # Major indices — pip_size * lot_size (contract_size treated as 1)
    "US30":   (1.0,    1),
    "US500":  (0.1,    1),
    "NAS100": (0.1,    1),
    "GER40":  (1.0,    1),
    "UK100":  (1.0,    1),
    "JPN225": (1.0,    1),
    # Crude oil
    "USOIL":  (0.01,   1_000),
    "UKOIL":  (0.01,   1_000),
}

# Approximate USD cross rates used when the account currency is USD but the
# quote currency of the pair is not USD (and vice-versa).
# We keep these as conservative mid estimates; a live feed would override them.
_USD_RATES: dict[str, float] = {
    "EUR": 1.08,
    "GBP": 1.27,
    "AUD": 0.65,
    "NZD": 0.60,
    "CAD": 0.74,
    "CHF": 1.12,
    "JPY": 0.0067,
}


def pip_value(symbol: str, lot_size: float, account_currency: str) -> float:
    """Approximate pip value in account currency for a given symbol and lot size.

    Lookup logic (no live feed required):

    * For pairs where account_currency matches the quote currency the formula is:
        pip_value = pip_size * lot_size * contract_size
    * For XAUUSD / XAGUSD a 100 oz contract is used.
    * For index CFDs (contract_size == 1):
        pip_value = pip_size * lot_size
    * When account_currency does not match the quote currency, a static
      cross-rate table (_USD_RATES) is used to convert.
    * Falls back to a conservative estimate of 10.0 * lot_size if the symbol
      is not in the lookup table.

    Args:
        symbol: Instrument identifier, e.g. "EURUSD", "XAUUSD", "US30".
        lot_size: Number of standard lots.
        account_currency: Three-letter ISO currency code, e.g. "USD".

    Returns:
        Estimated pip value expressed in account_currency.
    """
    symbol_upper = symbol.upper().strip()
    account_currency = account_currency.upper().strip()

    if symbol_upper not in _PIP_TABLE:
        # Conservative fallback: $10 per pip per lot is typical for majors
        return 10.0 * lot_size

    pip_size, contract_size = _PIP_TABLE[symbol_upper]

    # Derive the quote currency from the symbol
    # Forex pairs are exactly 6 characters; metals/indices handled separately
    if symbol_upper in ("XAUUSD", "XAGUSD"):
        quote_currency = "USD"
    elif symbol_upper in ("US30", "US500", "NAS100", "USOIL", "UKOIL"):
        quote_currency = "USD"
    elif symbol_upper in ("GER40", "UK100"):
        quote_currency = "USD"   # most brokers quote in USD; adjust if needed
    elif symbol_upper in ("JPN225",):
        quote_currency = "JPY"
    elif len(symbol_upper) == 6:
        quote_currency = symbol_upper[3:6]
    else:
        # Cannot determine quote currency; use fallback
        return 10.0 * lot_size

    raw_pip_value = pip_size * lot_size * contract_size

    if quote_currency == account_currency:
        return raw_pip_value

    # Convert quote currency -> account currency
    if account_currency == "USD":
        rate = _USD_RATES.get(quote_currency)
        if rate is None:
            return raw_pip_value  # best we can do without a live rate
        return raw_pip_value * rate

    if quote_currency == "USD":
        rate = _USD_RATES.get(account_currency)
        if rate is None:
            return raw_pip_value
        # account_currency/USD rate -> divide by USD rate of account_currency
        return raw_pip_value / rate

    # Cross via USD
    quote_usd = _USD_RATES.get(quote_currency)
    acct_usd = _USD_RATES.get(account_currency)
    if quote_usd is None or acct_usd is None:
        return raw_pip_value
    return raw_pip_value * (quote_usd / acct_usd)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing (EMA with alpha=1/period).

    The true range for each bar is defined as:
        TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Wilder's ATR uses an exponential moving average with alpha = 1/period,
    which is equivalent to com = period - 1 in pandas EWM.

    No look-ahead is introduced: only past closes are used.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Smoothing period (default 14).

    Returns:
        pd.Series of ATR values aligned to the input index.
        The first (period - 1) values will be NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing: EWM with com = period - 1  ->  alpha = 1/period
    return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between two series.

    Args:
        a: First price/return series.
        b: Second price/return series, must share the same index as `a`.
        window: Look-back window in bars.

    Returns:
        pd.Series of rolling correlation values in [-1, 1].
        The first (window - 1) values will be NaN.
    """
    return a.rolling(window).corr(b)


def hurst_exponent(series: pd.Series, window: int = 100) -> float:
    """Estimate Hurst exponent using R/S analysis on the last `window` values.

    The R/S method divides the sub-series of length n into segments, computes
    the rescaled range (R/S) for each segment, and regresses log(R/S) against
    log(n) across multiple lag sizes.  The slope of the regression is H.

    Interpretation:
        H < 0.5  ->  mean-reverting (anti-persistent)
        H = 0.5  ->  random walk
        H > 0.5  ->  trending (persistent)

    Returns 0.5 if there is insufficient data (< 20 observations).

    Args:
        series: Price or return series.
        window: Number of most-recent observations to use (default 100).

    Returns:
        Estimated Hurst exponent as a float in (0, 1).
    """
    data = series.dropna().values[-window:]
    n_obs = len(data)

    if n_obs < 20:
        return 0.5

    candidate_lags = [10, 20, 40, 80, window // 2, window]
    lags = sorted({lag for lag in candidate_lags if 10 <= lag <= n_obs})

    if len(lags) < 2:
        return 0.5

    log_lags: list[float] = []
    log_rs: list[float] = []

    for lag in lags:
        sub = data[:lag]
        mean = sub.mean()
        deviations = np.cumsum(sub - mean)
        r = deviations.max() - deviations.min()
        s = sub.std(ddof=1)
        if s == 0.0:
            continue
        log_lags.append(math.log(lag))
        log_rs.append(math.log(r / s))

    if len(log_lags) < 2:
        return 0.5

    try:
        slope, _ = np.polyfit(log_lags, log_rs, 1)
    except np.linalg.LinAlgError:
        return 0.5

    return float(slope)


def z_score(series: pd.Series, window: int = 50) -> pd.Series:
    """Rolling z-score: (value - rolling_mean) / rolling_std.

    Uses a look-back only window so no future information leaks in.
    The first (window - 1) values will be NaN.

    Args:
        series: Input price or indicator series.
        window: Rolling window length (default 50).

    Returns:
        pd.Series of z-scores aligned to the input index.
    """
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    return (series - mean) / std


def encode_session(timestamp: pd.Timestamp) -> int:
    """Encode the trading session from a UTC timestamp.

    Session boundaries (all times in UTC):
        0 = Asian          00:00 – 08:00
        1 = London         08:00 – 13:00
        3 = London/NY Overlap  13:00 – 17:00
        2 = New York       17:00 – 22:00
        0 = Asian (late)   22:00 – 00:00

    The London-NY overlap (13:00-17:00) is encoded as 3 to allow
    downstream logic to distinguish it from pure London or pure NY
    hours while keeping a compact integer representation.

    Args:
        timestamp: A timezone-aware or naive pd.Timestamp assumed to be UTC.

    Returns:
        Integer session code 0-3.
    """
    hour = timestamp.hour  # 0-23

    if 8 <= hour < 13:
        return 1   # London
    if 13 <= hour < 17:
        return 3   # London/NY Overlap
    if 17 <= hour < 22:
        return 2   # New York
    return 0       # Asian (00:00-08:00 and 22:00-00:00)
