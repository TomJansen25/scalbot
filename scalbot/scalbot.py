"""
Scalbot Class Definition
"""

import json
import time
from abc import ABC
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from scalbot.bigquery import BigQuery
from scalbot.bybit import Bybit
from scalbot.data import calc_candle_colors, get_latest_candle
from scalbot.enums import Symbol
from scalbot.models import Candle, OpenPosition, Trade
from scalbot.technical_indicators import calc_sma
from scalbot.trades import TradingStrategy
from scalbot.utils import get_percentage_occurrences, is_subdict


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

    @logger.catch
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
                # try:
                symbol = symbol.value
                bybit.setup_http()
                open_position = bybit.get_open_position(symbol)

                # CHECK 1: see if there are any open Limit Orders with a Stop Loss, indicating
                # that there is a new active order pending to open a position
                new_limit_orders = bybit.get_active_orders(
                    symbol=symbol, order_status="New", order_type="Limit"
                )
                open_position_orders = [
                    order
                    for order in new_limit_orders
                    if float(order.get("stop_loss")) > 0
                ]

                if len(open_position_orders) > 0:
                    logger.info(
                        f"There currently is an active order waiting to be filled to open a "
                        f"position for {symbol} and therefore no further action will be taken"
                    )

                # CHECK 2: see if there is an open position and check whether the calculated
                # take profits and stop losses are properly set (and if not, set them)

                elif open_position.size > 0:
                    logger.info(
                        f"There currently is an open position for {symbol} of "
                        f"size {open_position.size}"
                    )

                    trade = self.bigquery_client.get_last_trade(
                        symbol=symbol, broker="Bybit"
                    )

                    if open_position.open_sl < open_position.size:
                        bybit.fill_position_with_defined_trade(
                            symbol=symbol,
                            open_position=open_position,
                            trade=trade,
                            fill_take_profit=False,
                        )

                    matched_orders = self.match_open_orders_to_trade(
                        position=open_position, orders=new_limit_orders, trade=trade
                    )

                    logger.info(f"Matched orders: {matched_orders}")

                    for tp_level, level_set in matched_orders.items():
                        if level_set == "order_not_set":
                            side = "Sell" if trade.side == "Buy" else "Buy"
                            order_res = bybit.place_active_order(
                                symbol=trade.symbol,
                                order_type="Limit",
                                side=side,
                                qty=int(
                                    getattr(trade, "quantity_usd")
                                    * getattr(trade, f"{tp_level}_share")
                                ),
                                price=int(getattr(trade, tp_level)),
                                close_on_trigger=True,
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

                        df = bybit.get_latest_symbol_data_as_df(
                            symbol=symbol, interval=self.candle_frequency
                        )
                        df = calc_candle_colors(df=df)
                        df["sma"] = calc_sma(df=df, col="close", n=9)

                        candle = get_latest_candle(df=df)
                        (
                            make_trade,
                            trade_side,
                            pattern,
                            trade_candle,
                        ) = self.evaluate_candle(candle=candle, df=df)

                        if make_trade:
                            new_trade = self.trading_strategy.define_trade(
                                trade_side=trade_side,
                                candle=trade_candle,
                                pattern=pattern,
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
                            new_trade.order_id = order_res.get("order_id")

                            self.bigquery_client.insert_trade_to_bigquery(
                                trade=new_trade
                            )
                    else:
                        logger.info(
                            "Modulo of current time is not 0, so no further action will be taken"
                        )
                # except Exception as exception:
                #   logger.error(f"Error occurred: {exception}")

            time.sleep(60)

    @staticmethod
    def find_open_take_profits(position: OpenPosition, trade: Trade) -> list[str]:

        total_take_profit_size = 0
        required_take_profits = []

        for level in ["take_profit_3", "take_profit_2", "take_profit_1"]:
            tp_size = int(
                getattr(trade, "quantity_usd") * getattr(trade, f"{level}_share")
            )
            if total_take_profit_size >= position.size:
                break
            required_take_profits.append(level)
            total_take_profit_size += tp_size

        return required_take_profits

    @logger.catch
    def match_open_orders_to_trade(
        self, position: OpenPosition, orders: list, trade: Trade
    ) -> dict:

        matched_orders = {}

        filled_position_size = 0
        required_take_profit_levels = self.find_open_take_profits(
            position=position, trade=trade
        )

        for level in required_take_profit_levels:
            tp_price = int(getattr(trade, level))
            tp_size = int(
                getattr(trade, "quantity_usd") * getattr(trade, f"{level}_share")
            )

            if filled_position_size == position.size:
                matched_orders[level] = "order_already_filled"

            next_take_profit = next(
                (
                    order
                    for order in orders
                    if int(float(order.get("price"))) == tp_price
                    and int(float(order.get("qty"))) == tp_size
                ),
                None,
            )
            if next_take_profit:
                filled_position_size += tp_size
                matched_orders[level] = "order_set"
                logger.info(
                    f"Take Profit Level {level[-1]} of price {tp_price} and "
                    f"size {tp_size} is already set!"
                )
            else:
                matched_orders[level] = "order_not_set"
                logger.info(
                    f"Take Profit Level {level[-1]} of price {tp_price} and "
                    f"size {tp_size} cannot be matched yet!"
                )

        return matched_orders

    @staticmethod
    def find_candle_pattern(candle: Candle, patterns: list[dict]) -> Optional[dict]:
        """
        Compares the color pattern of a candle (as a dict) and checks whether it matches one
        of the specified patterns.
        Returns the matching pattern, and otherwise None
        :param candle: dictionary with pattern of candle
        :param patterns: list of dictionaries of patterns to be compared against
        :return: dictionary with matching pattern or None if no match
        """
        found_pattern = None
        candle_pattern = {"color": candle.color} | candle.previous_colors
        for pattern in patterns:
            if is_subdict(pattern, candle_pattern):
                logger.info(f"Found matching pattern: {pattern}")
                found_pattern = pattern
                break
        if not found_pattern:
            logger.info("No matching pattern was found")
        return found_pattern

    @staticmethod
    def define_trade_side(pattern: Optional[dict]) -> Optional[str]:
        """

        :param pattern:
        :return:
        """
        if pattern:
            values = list(pattern.values())
            color = max(set(values), key=values.count)
            trade = "short" if color == "red" else "long"
            logger.info(
                f"The most frequent candle color in the found pattern is {color}, which "
                f"hints towards a {trade} trade"
            )
        else:
            trade = None

        return trade

    @staticmethod
    def get_previous_trend(candle: Candle, v_pattern: Optional[dict]) -> Optional[dict]:
        """

        :param candle:
        :param v_pattern:
        :return:
        """
        candle_colors = {"color": candle.color} | candle.previous_colors
        if v_pattern:
            remaining_previous_candles = {
                k: v for k, v in candle_colors.items() if k not in v_pattern
            }
            colors = list(remaining_previous_candles.values())
            perc_occ = get_percentage_occurrences(colors)
        else:
            perc_occ = None

        return perc_occ

    @staticmethod
    def get_previous_trend_trade_side(
        trend_occurrences: Optional[dict], minimum_trend: float = 0.5
    ) -> Optional[str]:
        """

        :param trend_occurrences:
        :param minimum_trend:
        :return:
        """
        if trend_occurrences:
            max_key = max(trend_occurrences, key=trend_occurrences.get)
            max_value: float = trend_occurrences.get(max_key)
            if max_key == "green" and max_value > minimum_trend:
                trade = "short"
                logger.info(
                    f"In the previous candles, green ones were found the most with "
                    f"{max_value:.2%}, which is above the specified minimum of "
                    f"{minimum_trend:.2%}. This hints towards a short trade."
                )
            elif max_key == "red" and max_value > minimum_trend:
                trade = "long"
                logger.info(
                    f"In the previous candles, red ones were found the most with "
                    f"{max_value:.2%}, which is above the specified minimum of "
                    f"{minimum_trend:.2%}. This hints towards a long trade."
                )
            else:
                trade = None
                logger.info(
                    "In the previous candles, no obvious trend could be detected"
                )
        else:
            trade = None

        return trade

    @staticmethod
    def make_trade(
        trade_side_v_pattern: Optional[str],
        trade_side_sma: Optional[str],
        trade_side_prev_trend: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """

        :param trade_side_v_pattern:
        :param trade_side_sma:
        :param trade_side_prev_trend:
        :return:
        """
        if trade_side_v_pattern == trade_side_sma == trade_side_prev_trend is not None:
            make_trade = True
            trade_side = trade_side_sma
            logger.info(
                f"All calculated trade sides agree, and therefore a {trade_side} trade "
                f"should be made!"
            )

        else:
            make_trade = False
            trade_side = None
            logger.info("Not all calculated trade sides agree, no trade will be made!")

        return make_trade, trade_side

    @logger.catch
    def get_turning_point(
        self, candle: Candle, pattern: dict, df: pd.DataFrame
    ) -> Candle:
        """

        :param candle:
        :param pattern:
        :param df:
        :return:
        """
        last_key = list(pattern)[-1]
        prev_minutes = int(last_key[-1]) - 1
        turning_point = (
            df.loc[
                df.start
                == candle.start
                - timedelta(minutes=prev_minutes * self.candle_frequency),
            ]
            .squeeze()
            .to_dict()
        )
        turning_point_candle = Candle.parse_obj(turning_point)
        return turning_point_candle

    @logger.catch
    def evaluate_candle(
        self, candle: Candle, df: pd.DataFrame
    ) -> Tuple[bool, Optional[str], Optional[dict], Optional[Candle]]:
        """

        :param candle:
        :param df:
        :return:
        """
        logger.info(f"Evaluating the following candle: {candle}")
        # candle_colors = {"color": candle.color} | candle.previous_colors
        v_pattern = self.find_candle_pattern(candle, self.patterns)

        trade_side_v_pattern = self.define_trade_side(v_pattern)

        turning_point_candle = None
        trade_side_sma = None
        if v_pattern:
            turning_point_candle = self.get_turning_point(candle, v_pattern, df)

            if trade_side_v_pattern == "short":
                price = turning_point_candle.low
            else:
                price = turning_point_candle.high

            price_vs_sma = price - turning_point_candle.sma
            trade_side_sma = "short" if price_vs_sma > 0 else "long"

            logger.info(
                f"Trade price would be {price}, and candle SMA 9 = {turning_point_candle.sma}, so "
                f"Price vs. SMA = {price_vs_sma}, which hints towards a {trade_side_sma} trade"
            )

        previous_trend = self.get_previous_trend(candle, v_pattern)
        trade_side_prev_trend = self.get_previous_trend_trade_side(previous_trend)

        make_trade, trade_side = self.make_trade(
            trade_side_v_pattern, trade_side_sma, trade_side_prev_trend
        )

        return make_trade, trade_side, v_pattern, turning_point_candle
