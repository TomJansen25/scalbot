from typing import Dict, List, Tuple

import pandas as pd


def calc_sma(df: pd.DataFrame, col: str = "close", n: int = 3) -> pd.Series:
    sma = df[col].rolling(window=n, min_periods=n).mean()
    return sma


def calc_ema(df: pd.DataFrame, col: str = "close", alpha: float = 0.25) -> pd.Series:
    ema = df[col].ewm(alpha=alpha, adjust=False).mean()
    return ema


def calc_macd(
    df: pd.DataFrame, col: str, n_long: int = 26, n_short: int = 12, n_macd: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD)
    :param df: pandas DataFrame with column to calculate the MACD based on
    :param col: column in the DataFrame to use for the MACD calculation
    :param n_long:
    :param n_short:
    :param n_macd:
    :return:
    """
    calc_df = df.copy()
    # Calculate short and long Exponential Moving Averages to subtract afterwards
    ema_short = calc_df[col].ewm(span=n_short, min_periods=n_long).mean()
    ema_long = calc_df[col].ewm(span=n_long, min_periods=n_long).mean()
    # Calculate MACD by subtracting the long EMA from the short EMA
    # Then also calculate the signal with the provided span (n) and the diff between MACD and the signal
    macd = ema_short - ema_long
    macd_sign = macd.ewm(span=n_macd, min_periods=n_macd).mean()
    macd_diff = macd - macd_sign
    return macd, macd_sign, macd_diff


def calc_bbands(
    df: pd.DataFrame, col: str, n: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = df[col].rolling(window=n, min_periods=n).mean()
    sma_std = df[col].rolling(window=n, min_periods=n).std()

    bb_upper = sma + (2 * sma_std)
    bb_lower = sma - (2 * sma_std)

    return bb_lower, sma, bb_upper


def calc_rsi(df: pd.DataFrame, col: str, n: int = 14) -> pd.Series:

    delta = df.copy()[col].diff()
    gains = delta * 0
    losses = gains.copy()
    i_gain = delta > 0
    i_loss = delta < 0
    gains[i_gain] = delta[i_gain]
    losses[i_loss] = abs(delta[i_loss])
    rs = pd.Series.ewm(gains, span=n).mean() / pd.Series.ewm(losses, span=n).mean()
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_momentum(
    df: pd.DataFrame, col: str, n: int = 1
) -> Tuple[pd.Series, pd.Series]:
    calc_df = df.copy()
    mom = calc_df[col].diff(periods=n)
    mom_perc = mom / calc_df[col] * 100
    return mom, mom_perc


def calc_rate_of_change(df: pd.DataFrame, col: str, n: int = 5) -> pd.Series:
    rc = df.copy()[col].pct_change(periods=n)
    return rc


def calc_obv(df: pd.DataFrame, n: int = 3) -> pd.Series:
    delta = df.copy().close.diff()
    obv = delta * 0

    i_up = delta > 0
    i_equal = delta == 0
    i_down = delta < 0

    obv[i_up] = df.volume[i_up]
    obv[i_equal] = 0
    obv[i_down] = -df.volume[i_down]

    obv_ma = obv.rolling(window=n, min_periods=n).mean()
    return obv_ma
