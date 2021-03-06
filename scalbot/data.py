import os
import tempfile
from abc import ABC
from datetime import timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from loguru import logger
from plotly.graph_objects import Candlestick, Figure
from pydantic import BaseModel, Field

from scalbot.enums import Broker, Symbol
from scalbot.models import Candle
from scalbot.utils import get_project_dir


class Data(ABC, BaseModel):
    """
    Base Data Class
    """

    _data_dir = get_project_dir().joinpath("data")
    candle_frequency: int = Field(default=1, ge=1, le=60)
    symbol: Symbol = Symbol.BTCUSD
    broker: Broker = Broker.BYBIT
    candles: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        candle_frequency: int = 1,
        symbol: Symbol = Symbol.BTCUSD,
        broker: Broker = Broker.BYBIT,
    ):

        super().__init__(
            candle_frequency=candle_frequency,
            symbol=symbol,
            broker=broker,
        )

    def change_symbol(self, symbol: Symbol = Symbol.BTCUSD):
        """

        :param symbol:
        """
        if symbol != self.symbol:
            self.symbol = symbol
            self.candles = None

    def plot_candlesticks(self) -> Figure:
        """

        :return:
        """
        if self.candles is None:
            msg = "No candle history is set yet, data should first be retrieved"
            logger.error(msg)
            raise Warning(msg)

        df = self.candles.loc[self.candles.symbol == self.symbol.value].copy()

        fig = Figure()

        fig.add_trace(
            Candlestick(
                name=self.symbol.value,
                x=df.start,
                open=df.open,
                close=df.close,
                high=df.high,
                low=df.low,
            )
        )

        fig.update_layout(
            title=f"Candle data for {self.symbol.value} from {self.broker.value}",
            xaxis_rangeslider_visible=False,
        )

        return fig

    def resample_candles(self, new_frequency: int = 3):
        """

        :param new_frequency:
        """
        if isinstance(self.candles, pd.DataFrame):
            df = self.candles.set_index(self.candles.start, drop=True).copy()
            ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                ohlc["volume"] = "sum"
            new_df = df.resample(f"{new_frequency}min").apply(ohlc).reset_index()
            new_df["end"] = new_df.start + timedelta(minutes=new_frequency)
            new_df["symbol"] = self.symbol.value
            self.candles = new_df
        else:
            raise Warning("No data in candles object yet, first load a DataFrame!")


class HistoricData(Data):
    """
    Historic Data Class to retrieve full and partial historic data for several Symbols and Brokers
    """

    bybit_data_url = "https://public.bybit.com/spot_index/"

    def get_latest_local_data(self):
        files = list(self._data_dir.glob("*.csv.gz"))
        relevant_files = [file for file in files if self.symbol.value in file.stem]
        latest_file = max(relevant_files, key=os.path.getctime)
        logger.info(
            f"Retrieved latest file with historic data for {self.symbol.value}: {latest_file}"
        )

        df = pd.read_csv(latest_file, compression="gzip")
        df.start = pd.to_datetime(df.start)
        df.end = pd.to_datetime(df.end)

        if not self.candles:
            self.candles = df
        return df

    def is_symbol_data_to_download(self):
        if self.broker == Broker.BYBIT:
            res = requests.get(self.bybit_data_url)
            soup = bs(res.content, "html.parser")
            available_symbols = soup.find_all("a", href=True)

            symbol_link = next(
                filter(lambda x: self.symbol.value in x.text, available_symbols), None
            )
            return bool(symbol_link)
        else:
            logger.warn(f"Only Bybit is implemented so far, other brokers not yet...")
            return False

    def retrieve_historic_data(self) -> pd.DataFrame:
        """
        Retrieve full historic data if not in Class yet, otherwise directly return
        :return: (pd.DataFrame) with full candle history from Symbol
        """
        if not self.candles:
            logger.info(
                f"Retrieving full historical data for {self.symbol.value} "
                f"from {self.broker.value}..."
            )
            if self.is_symbol_data_to_download():
                symbol_url = f"{self.bybit_data_url}{self.symbol.value}/"

                res = requests.get(symbol_url)
                soup = bs(res.content, "html.parser")
                downloads = soup.find_all("a", href=True)

                results = []

                for download in downloads:
                    url = f'{symbol_url}{download["href"]}'
                    logger.info(f"Processing {url} ...")

                    with tempfile.TemporaryDirectory() as tmp_dir_name:
                        tmp_file_name = "tmp_index_prices.csv.gz"
                        tmp_file_path = Path(tmp_dir_name, tmp_file_name)

                        data = requests.get(url).content
                        urlretrieve(url, tmp_file_path)

                        df = pd.read_csv(tmp_file_path, compression="gzip")
                        results.append(df)

                    logger.info(f"Finished retrieving data for {download.text}!")

                full_df = pd.concat(results, ignore_index=True).sort_values(
                    by="start_at", ascending=True
                )
                full_df["start"] = pd.to_datetime(full_df.start_at, unit="s")
                full_df["end"] = full_df.start + timedelta(minutes=1)
                self.candles = full_df
        else:
            logger.info(
                f"Full historical data for {self.symbol.value} from {self.broker.value} is already "
                f"loaded and will be returned directly"
            )

        return self.candles


def calc_candle_colors(df: pd.DataFrame, shift: int = 10) -> pd.DataFrame:
    """

    :param df:
    :param shift:
    :return:
    """
    df = df.copy()
    df["color"] = np.where(df.close > df.open, "green", "red")

    for shift_index in range(1, shift + 1):
        df[f"prev_color_{shift_index}"] = df.color.shift(shift_index)

    df["previous_colors"] = df.apply(
        lambda x: {k: v for k, v in x.squeeze().items() if k.startswith("prev_color_")},
        axis=1,
    )
    return df


def get_latest_candle(df: pd.DataFrame, col: str = "start") -> Candle:
    """
    Retrieve the latest candle of a DataFrame based on the provided column (default = 'start')
    :param df:
    :param col:
    :return: Latest Candle as dictionary
    """
    latest_candle = df.loc[df[col] == df[col].max()].squeeze().to_dict()
    candle = Candle.parse_obj(latest_candle)
    return candle
