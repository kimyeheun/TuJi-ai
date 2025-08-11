import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_bollinger_bands(close: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower

def compute_mom(close: pd.Series, period: int = 10) -> pd.Series:
    mom = close.diff(period)
    return mom

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_typical_price = typical_price.rolling(window=period).mean()
    def calculate_mean_deviation(series):
        if len(series) < period:
            return np.nan
        sma = series.mean()
        return np.mean(np.abs(series - sma))
    mean_deviation = typical_price.rolling(window=period).apply(calculate_mean_deviation, raw=False)
    denominator = 0.015 * mean_deviation
    cci = (typical_price - sma_typical_price) / denominator
    return cci

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI'] = compute_rsi(df['Close'])
    macd, macd_signal = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    bb_upper, bb_lower = compute_bollinger_bands(df['Close'])
    df['BB_UPPER'] = bb_upper
    df['BB_LOWER'] = bb_lower
    df['MOM'] = compute_mom(df['Close'])
    df['CCI'] = compute_cci(df['High'], df['Low'], df['Close'])
    return df