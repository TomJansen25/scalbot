"""
Scalbot Class Definition
"""

import json
import time
from abc import ABC
from datetime import datetime, timedelta
from typing import Tuple, Union

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, validator

from scalbot.bigquery import BigQuery
from scalbot.bybit import Bybit
from scalbot.enums import Symbol
from scalbot.technical_indicators import calc_sma
from scalbot.trades import TradingStrategy
from scalbot.utils import (calc_candle_colors, get_last_trade,
                           get_latest_candle, get_percentage_occurrences,
                           get_project_dir, is_subdict)

PROJECT_DIR = get_project_dir()

SUBSET = [
    "start",
    "end",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "turnover",
    "subscription",
]
COLUMNS = [
    "start",
    "end",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "turnover",
    "timestamp",
    "confirm",
    "cross_seq",
    "coin",
    "subscription",
]


class Scalbot(BaseModel, ABC):
    """
    Scalbot Class to start and run bots for several platforms with several strategies
    """

    patterns: list[dict] = Field(default_factory=list)
    trading_strategy: TradingStrategy
    candle_frequency: int = Field(default=1, gt=0)
    bigquery_client: BigQuery

    def __init__(
        self,
        patterns: list[dict],
        trading_strategy: TradingStrategy,
        candle_frequency: int,
    ):
        """

        :param patterns:
        :param trading_strategy:
        :param candle_frequency: minute frequency of candles to be retrieved (1, 3, 5, or 15)
        """
        super().__init__(
            patterns=patterns,
            trading_strategy=trading_strategy,
            candle_frequency=candle_frequency,
            bigquery_client=BigQuery(),
        )

    def update_trading_strategy(self, trading_strategy: TradingStrategy):
        logger.info("Trading strategy updated!")
        self.trading_strategy = trading_strategy

    def run_bybit_bot(self, symbols: Union[Symbol, list[Symbol]] = Symbol.BTCUSD):
        """
        Initializes Bybit HTTP and Websocket connections and starts Bybit Scalbot instance
        :param symbols: string or list of strings indicating symbols to retrieve data of
        """
        logger.info(
            f"Setting up bybit bot with Candle Frequency = {self.candle_frequency} and "
            f"for the following Symbols: {symbols}..."
        )
        bybit = Bybit()

        if not isinstance(symbols, list):
            symbols = [symbols]

        while True:
            for symbol in symbols:
                try:
                    symbol = symbol.value
                    open_position = bybit.get_open_position(symbol)

                    new_orders = (
                        bybit.get_active_orders(symbol=symbol, order_status="New")
                        .get("result")
                        .get("data")
                    )

                    if len(new_orders) > 0:
                        logger.info(
                            f"There currently is an active order waiting to be filled "
                            f"for {symbol} and therefore no further action will be taken"
                        )

                    elif open_position.size > 0:
                        logger.info(
                            f"There currently is an open position for {symbol} of "
                            f"size {open_position.size}"
                        )

                        trade = self.bigquery_client.get_last_trade(
                            symbol=symbol, broker="Bybit"
                        )

                        if open_position.open_sl > 0 or open_position.open_tp > 0:
                            logger.info(
                                "There is still an open quantity to be filled..."
                            )
                            bybit.fill_position_with_defined_trade(
                                symbol=symbol, open_position=open_position, trade=trade
                            )
                        else:
                            logger.info(
                                "The position is completely filled, no action to be taken."
                            )
                    else:
                        logger.info(f"There is no open position for {symbol}...")

                        current_time = datetime.now().time()

                        if current_time.minute % self.candle_frequency == 0:
                            logger.info(
                                f"Modulo of current time and chosen candle frequency = 0, so latest "
                                f"data will be retrieved to check for the defined patterns..."
                            )

                            bybit.setup_http()
                            df = bybit.get_latest_symbol_data_as_df(
                                symbol=symbol, interval=self.candle_frequency
                            )
                            df = calc_candle_colors(df=df)
                            df["sma"] = calc_sma(df=df, col="close", n=9)

                            candle = get_latest_candle(df=df)
                            (
                                make_trade,
                                trade,
                                pattern,
                                trade_candle,
                            ) = self.evaluate_candle(candle=candle, df=df)

                            if make_trade:
                                new_trade = self.trading_strategy.define_trade(
                                    trade=trade, candle=trade_candle, pattern=pattern
                                )
                                logger.info(
                                    f"Pattern found and new trade calculated: {new_trade.dict()}"
                                )

                                order_res = bybit.place_active_order(
                                    symbol=symbol,
                                    order_type="Limit",
                                    side=new_trade.side,
                                    qty=new_trade.quantity_usd,
                                    price=new_trade.price,
                                    stop_loss=new_trade.stop_loss,
                                    order_link_id=str(new_trade.order_link_id),
                                )

                                new_trade.broker = bybit.broker.value
                                new_trade.order_id = order_res.get("result").get(
                                    "order_id"
                                )

                                self.bigquery_client.insert_trade_to_bigquery(
                                    trade=new_trade
                                )

                                df = pd.DataFrame(new_trade.dict(), index=[0])
                                df.to_csv(
                                    PROJECT_DIR.joinpath("data", "trades.csv"),
                                    index=False,
                                    header=False,
                                    mode="a",
                                )
                        else:
                            logger.info(
                                "Modulo of current time is not 0, so no further action will be taken"
                            )
                except Exception as exception:
                    logger.error(f"Error occurred: {exception}")

            time.sleep(60)

    @staticmethod
    def find_candle_pattern(candle: dict, patterns: list) -> Union[dict, None]:
        """
        Compares the color pattern of a candle (as a dict) and checks whether it matches one
        of the specified patterns.
        Returns the matching pattern, and otherwise None
        :param candle: dictionary with pattern of candle
        :param patterns: list of dictionaries of patterns to be compared against
        :return: dictionary with matching pattern or None if no match
        """
        found_pattern = None
        for pattern in patterns:
            if is_subdict(pattern, candle):
                logger.info(f"Found matching pattern: {pattern}")
                found_pattern = pattern
                break
        if not found_pattern:
            logger.info("No matching pattern was found")
        return found_pattern

    def get_turning_point(self, candle: dict, pattern: dict, df: pd.DataFrame) -> dict:
        """

        :param candle:
        :param pattern:
        :param df:
        :return:
        """
        start: datetime = candle.get("start")
        last_key = list(pattern)[-1]
        prev_minutes = int(last_key[-1]) - 1
        turning_point_candle = (
            df.loc[
                df.start
                == start - timedelta(minutes=prev_minutes * self.candle_frequency),
                ["start", "end", "open", "close", "high", "low", "symbol"],
            ]
            .squeeze()
            .to_dict()
        )

        return turning_point_candle

    def evaluate_candle(
        self, candle: dict, df: pd.DataFrame
    ) -> Tuple[bool, str, Union[dict, None], Union[dict, None]]:
        """

        :param candle:
        :param df:
        :return:
        """
        d = {k: v for k, v in candle.items() if k == "color" or k.startswith("prev_")}
        v_pattern = self.find_candle_pattern(d, self.patterns)

        make_trade = False
        trade = ""
        if v_pattern:
            values = list(v_pattern.values())
            color = max(set(values), key=values.count)
            trade = "short" if color == "red" else "long"

            remaining_previous_candles = {
                k: v for k, v in d.items() if k not in v_pattern
            }
            colors = list(remaining_previous_candles.values())

            if trade == "short" and get_percentage_occurrences(colors, "green") > 0.5:
                make_trade = True
            elif trade == "long" and get_percentage_occurrences(colors, "red") > 0.5:
                make_trade = True
            else:
                logger.info("Pattern found but no obvious previous trend")

        turning_point_candle = None
        if make_trade:
            turning_point_candle = self.get_turning_point(candle, v_pattern, df)
            turning_point_candle["trade"] = trade
            turning_point_candle["pattern"] = json.dumps(v_pattern)
            turning_point_candle["source_candle_start"] = candle.get("start")
            logger.info(
                "V Pattern, previous trend, and turning point found and defined!"
            )

        return make_trade, trade, v_pattern, turning_point_candle
