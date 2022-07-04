import pathlib
import sys
from typing import Any, Optional, Union

import numpy as np
import yaml
from loguru import logger


def get_project_dir() -> pathlib.Path:
    """
    Returns the absolute path of the main project folder, regardless of running locally or on GCP
    :return: Absolute path of Scalbot folder
    """
    cwd_dir = pathlib.Path().cwd().absolute()
    # If executed in a GCP Cloud Function, the working directory is called "/workspace"
    if cwd_dir == "/workspace":
        return cwd_dir
    # If executed on a Local Machine from within the project folder, return the "/scalbot" folder
    parts = cwd_dir.parts[: (cwd_dir.parts.index("scalbot") + 1)]
    path_dir = pathlib.Path(*parts)
    return path_dir


def get_params() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Return Scalbot Parameters from params.yaml as dictionary
    :return: (dict) parameter settings
    """
    project_dir = get_project_dir()
    params_path = project_dir.joinpath("config", "params.yaml")
    with open(params_path, "r") as params_yaml:
        params: dict[str, dict[str, dict[str, Any]]] = yaml.load(
            params_yaml, Loader=yaml.FullLoader
        )
    return params


def setup_logging(serverless: bool = False):
    """
    Setup Loguru Logging with configure()
    """
    handlers = [dict(sink=sys.stdout, backtrace=False, diagnose=False)]
    if not serverless:
        handlers.append(
            dict(
                sink=get_project_dir().joinpath("log.log"),
                enqueue=True,
                serialize=True,
                backtrace=True,
                diagnose=True,
            )
        )
    logger.configure(
        handlers=handlers,
        extra={"user": "Tom Jansen"},
    )


def is_subdict(small: dict, big: dict) -> bool:
    """
    Checks whether a dictionary "small" is part of another dictionary "big".
    Key-value pairs are considered completely.
    :param small: smaller dictionary that should be part of "big"
    :param big: larger dictionary that should contain "small"
    :return: True if small is part of big, False if not
    """
    return big | small == big


def get_percentage_occurrences(
    lst: list, value: Optional[str] = None
) -> Union[dict, float]:
    """
    Returns the percentual occurrence of a value in a list if a value is given, or
    the percentual occurrence of
    every unique value in the list if none provided
    :param lst: list with values to go through
    :param value: value to look for or None if all unique values should be returned
    :return:
    """
    if value:
        return lst.count(value) / len(lst)
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
        style.append({"if": {"column_id": col_id}, "minWidth": f"{pixel}px"})

    return style


def get_percentual_diff(
    orig_num: float, new_num: float, absolute: bool = False
) -> float:
    diff = (new_num - orig_num) / orig_num
    if absolute:
        diff = abs(diff)
    return diff


def are_prices_equal_enough(
    first_price: Union[int, float], second_price: Union[int, float]
) -> bool:
    equal = False

    if first_price < 10 or second_price < 10:
        if np.round(first_price, 3) == np.round(second_price, 3):
            equal = True

    elif first_price <= 100 or second_price <= 100:
        if np.round(first_price, 2) == np.round(second_price, 2):
            equal = True

    elif first_price <= 1000 or second_price <= 1000:
        if np.round(first_price, 1) == np.round(second_price, 1):
            equal = True

    else:  # first_price > 1000 or second_price > 1000
        first_price = round(float(first_price) * 2) / 2
        second_price = round(float(second_price) * 2) / 2
        if np.round(first_price, 1) == np.round(second_price):
            equal = True

    return equal
