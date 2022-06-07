import pathlib
import sys
from typing import Optional, Union

from loguru import logger


def get_project_dir() -> pathlib.Path:
    """
    Always return the absolute Path of the Pricing folder to ensure all files and code can be found in any terminal
    :return: Absolute path of Pricing folder
    """
    path_dir = pathlib.Path().absolute()
    parts = path_dir.parts[: (path_dir.parts.index("scalbot") + 1)]
    path_dir = pathlib.Path(*parts)
    return path_dir


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
