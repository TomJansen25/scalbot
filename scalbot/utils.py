from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from loguru import logger

from scalbot.models import Trade


def get_project_dir() -> pathlib.Path:
    """
    Always return the absolute Path of the Pricing folder to ensure all files and code can be found in any terminal
    :return: Absolute path of Pricing folder
    """
    path_dir = pathlib.Path().absolute()
    parts = path_dir.parts[: (path_dir.parts.index("scalbot") + 1)]
    path_dir = pathlib.Path(*parts)
    return path_dir


def setup_logging():
    """
    Setup Loguru Logging with configure()
    """
    logger.configure(
        handlers=[
            dict(sink=sys.stdout),
            dict(
                sink=get_project_dir().joinpath("log.log"),
                enqueue=True,
                serialize=True,
                backtrace=True,
                diagnose=True,
            ),
        ],
        extra={"user": "Tom Jansen"},
    )


def calc_candle_colors(df: pd.DataFrame, shift: int = 10) -> pd.DataFrame:
    """

    :param df:
    :param shift:
    :return:
    """
    df = df.copy()
    df["color"] = np.where(df.close > df.open, "green", "red")

    for shift_index in range(1, shift + 1):
        df[f"prev_{shift_index}"] = df.color.shift(shift_index)

    return df


def is_subdict(small: dict, big: dict) -> bool:
    """
    Checks whether a dictionary "small" is part of another dictionary "big".
    Key-value pairs are considered completely.
    :param small: smaller dictionary that should be part of "big"
    :param big: larger dictionary that should contain "small"
    :return: True if small is part of big, False if not
    """
    return big | small == big


def get_percentage_occurrences(lst: list, value: str | None = None) -> dict | float:
    """
    Returns the percentual occurrence of a value in a list if a value is given, or
    the percentual occurrence of
    every unique value in the list if none provided
    :param lst: list with values to go through
    :param value: value to look for or None if all unique values should be returned
    :return:
    """
    if value:
        perc = lst.count(value) / len(lst)
    else:
        perc = {}

        for val in set(lst):
            perc[val] = lst.count(val) / len(lst)

    return perc


def create_conditional_style(columns: list, pixel_for_char: int):
    """

    :param columns:
    :param pixel_for_char:
    :return:
    """
    style = []
    for col in columns:
        name = col.get("name")
        col_id = col.get("id")
        name_length = len(name)
        pixel = 70 + round(name_length * pixel_for_char)
        pixel = str(pixel) + "px"
        style.append({"if": {"column_id": col_id}, "minWidth": pixel})

    return style


def get_latest_candle(df: pd.DataFrame, col: str = "start") -> dict:
    """
    Retrieve the latest candle of a DataFrame based on the provided column (default = 'start')
    :param df:
    :param col:
    :return: Latest Candle as dictionary
    """
    return df.loc[df[col] == df[col].max()].squeeze().to_dict()


def get_last_trade(symbol: str, broker: str) -> Trade:
    df = pd.read_csv(get_project_dir().joinpath("data", "trades.csv"))
    trade = (
        df.loc[
            (df.symbol == symbol)
            & (df.broker == broker)
            & (df.timestamp == df.timestamp.max())
        ]
        .squeeze()
        .to_dict()
    )
    return Trade.parse_obj(trade)
