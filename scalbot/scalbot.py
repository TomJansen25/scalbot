"""
Scalbot Class Definition
"""

from abc import ABC
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

import pandas as pd
import pytz
from loguru import logger
from pydantic import BaseModel, Field

from scalbot.bigquery import BigQuery
from scalbot.bybit import Bybit
from scalbot.data import calc_candle_colors, get_latest_candle
from scalbot.enums import Symbol
from scalbot.models import (
    ActiveOrder,
    BaseOrder,
    Candle,
    ConditionalOrder,
    LatestInfo,
    MinifiedOrder,
    OpenPosition,
    Trade,
)
from scalbot.technical_indicators import calc_sma
from scalbot.trades import TradingStrategy
from scalbot.utils import (
    are_prices_equal_enough,
    get_percentage_occurrences,
    is_subdict,
)


class Scalbot(BaseModel, ABC):
    """
    Scalbot Class to initialize and run a scalping bot for several Symbols and Brokers
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
    def run_bybit_bot(self, bybit: Bybit, symbol: Symbol = Symbol.BTCUSD):
        """
        Initializes Bybit HTTP and Websocket connections and starts Bybit Scalbot instance
        :param bybit:
        :param symbol: Symbol enum/string indicating which Symbol to check
        """

        bybit.setup_http()
        open_position = bybit.get_open_position(symbol)

        # CHECK 1: see if there are any open Limit Orders with a Stop Loss, indicating
        # that there is a new active order pending to open a position
        new_limit_orders, open_position_orders = bybit.find_open_position_orders(
            symbol=symbol
        )

        if len(open_position_orders) > 0:
            logger.info(
                f"There currently is an active order waiting to be filled to open a "
                f"position for {symbol.value} and therefore no further action will be taken"
            )

        # CHECK 2: see if there is an open position and check whether the calculated
        # take profits and stop losses are properly set (and if not, set them)

        elif open_position.size > 0:
            logger.info(
                f"There currently is an open position for {symbol} of "
                f"size {open_position.size}"
            )

            trade = self.bigquery_client.get_last_trade(symbol=symbol, broker="Bybit")

            if open_position.open_sl > 0:
                bybit.fill_position_with_defined_trade(
                    symbol=symbol,
                    open_position=open_position,
                    trade=trade,
                    fill_take_profit=False,
                )

            matched_orders, reached_tp_level = self.match_open_orders_to_trade(
                position=open_position, orders=new_limit_orders, trade=trade
            )

            logger.info(f"Matched orders: {matched_orders}")
            logger.info(f"Reached Take Profit Level: {reached_tp_level}")

            if reached_tp_level >= 1:

                _, sls = bybit.get_untriggered_take_profits_and_stop_losses(
                    symbol=symbol
                )
                stop_loss = sls[0]

                if not are_prices_equal_enough(trade.price, float(stop_loss.stop_px)):
                    logger.info(
                        "Take Profit Level 1 has been reached, but Stop Loss is still the "
                        "original. Stop Loss will be cancelled and set to the trade price"
                    )
                    bybit.cancel_conditional_order(
                        symbol=symbol,
                        stop_order_id=stop_loss.stop_order_id,
                    )
                    bybit.add_stop_loss_to_position(
                        symbol=symbol,
                        stop_loss=trade.price,
                        size=open_position.size,
                    )
                else:
                    logger.info(
                        "Take Profit Level 1 has been reached, but Stop Loss is already at "
                        "trade price level. Stop Loss does not need to be adjusted."
                    )

            for tp_level, level_set in matched_orders.items():
                if level_set == "order_not_set":
                    side = "Sell" if trade.side == "Buy" else "Buy"
                    order_res = bybit.place_active_order(
                        symbol=symbol,
                        order_type="Limit",
                        side=side,
                        qty=int(getattr(trade, f"{tp_level}_share")),
                        price=float(getattr(trade, tp_level)),
                        close_on_trigger=True,
                    )
            else:
                logger.info("The position is completely filled, no action to be taken.")

        else:
            logger.info(f"There is no open position for {symbol.value}...")

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

                    self.bigquery_client.insert_trade_to_bigquery(trade=new_trade)
            else:
                logger.info(
                    "Modulo of current time is not 0, so no further action will be taken"
                )

    @staticmethod
    def find_open_take_profits(
        position: OpenPosition, trade: Trade
    ) -> Tuple[list[str], int]:

        total_take_profit_size = 0
        reached_level = 0
        required_take_profits = []

        for level in [3, 2, 1]:
            tp_level = f"take_profit_{level}"  # , "take_profit_2", "take_profit_1"
            tp_size = int(getattr(trade, f"{tp_level}_share"))
            if total_take_profit_size >= position.size:
                reached_level = level
                break
            required_take_profits.append(tp_level)
            total_take_profit_size += tp_size

        return required_take_profits, reached_level

    @logger.catch
    def match_open_orders_to_trade(
        self, position: OpenPosition, orders: list[ActiveOrder], trade: Trade
    ) -> Tuple[dict, int]:

        required_take_profit_levels, reached_level = self.find_open_take_profits(
            position=position, trade=trade
        )
        merged_orders = merge_take_profits_with_same_price(orders)

        order_list = {
            order.order_id: dict(price=order.price, qty=order.qty) for order in orders
        }
        logger.info(f"Retrieved open Take Profit orders {order_list}")

        matched_orders = {}
        filled_position_size = 0

        if position.open_tp == 0:

            for level in required_take_profit_levels:
                tp_price = getattr(trade, level)
                tp_size = getattr(trade, f"{level}_share")

                if filled_position_size == position.size:
                    matched_orders[level] = "order_already_filled"

                next_take_profit = next(
                    (order for order in merged_orders if order.price == tp_price),
                    None,
                )
                if next_take_profit:
                    if next_take_profit.qty == tp_size:
                        logger.info(
                            f"Take Profit Level {level[-1]} of price {tp_price} and "
                            f"size {tp_size} is already set!"
                        )
                    else:
                        logger.info(
                            f"Take Profit Level {level[-1]} of price {tp_price} and is already "
                            f"set but with a different size {next_take_profit.qty}!"
                        )
                    filled_position_size += tp_size
                    matched_orders[level] = "order_set"
                else:
                    matched_orders[level] = "order_not_set"
                    logger.info(
                        f"Take Profit Level {level[-1]} of price {tp_price} and "
                        f"size {tp_size} cannot be matched yet!"
                    )

        else:

            for level in required_take_profit_levels:
                tp_price = getattr(trade, level)
                tp_size = getattr(trade, f"{level}_share")

                if filled_position_size == position.size:
                    matched_orders[level] = "order_already_filled"

                next_take_profit = next(
                    (
                        order
                        for order in merged_orders
                        if order.price == float(tp_price) and order.qty == int(tp_size)
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

        return matched_orders, reached_level

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


def is_order_expired(
    current_order: BaseOrder, max_timedelta: timedelta = timedelta(hours=24)
) -> bool:
    current_date = datetime.now().astimezone(pytz.UTC)
    order_updated_date = current_order.updated_at.astimezone(pytz.UTC)
    diff = current_date - order_updated_date
    if diff > max_timedelta:
        return True
    else:
        return False


def is_order_price_too_far_off_last_price(
    current_order: BaseOrder,
    latest_info: LatestInfo,
    max_rel_diff: float = None,
    max_abs_diff: float = None,
):
    if max_rel_diff is None and max_abs_diff is None:
        raise KeyError(
            f"Either one of both 'max_rel_diff' or 'max_abs_diff' has to be provided, "
            f"otherwise nothing can be calculated!"
        )
    order_price = current_order.price
    current_price = latest_info.last_price
    rel_diff = (order_price - current_price) / current_price
    abs_diff = order_price - current_price

    if max_rel_diff and abs(rel_diff) > max_rel_diff:
        return True
    elif max_abs_diff and abs(abs_diff) > max_abs_diff:
        return True
    else:
        return False


def cancel_invalid_expired_orders(
    bybit: Bybit, symbols: list[Symbol]
) -> list[ActiveOrder]:

    invalid_or_expired_orders: list[ActiveOrder] = []
    cancelled_orders: list[ActiveOrder] = []

    for symbol in symbols:
        open_position = bybit.get_open_position(symbol=symbol)
        _, open_position_orders = bybit.find_open_position_orders(symbol=symbol)
        latest_info = bybit.get_latest_symbol_information(symbol=symbol)

        if open_position.size == 0 and len(open_position_orders) > 0:
            logger.info(
                f"There is currently no open position, but there is an active order "
                f"waiting to open a position for {symbol.value}"
            )

            for order in open_position_orders:
                order_expired = is_order_expired(current_order=order)
                order_invalid = is_order_price_too_far_off_last_price(
                    current_order=order, latest_info=latest_info, max_rel_diff=0.25
                )
                if order_expired or order_invalid:
                    logger.info(
                        f"Order {order.order_id} is expired and/or invalid and will be cancelled..."
                    )
                    invalid_or_expired_orders.append(order)
                else:
                    logger.info(
                        f"Order {order.order_id} is still valid and will not be cancelled!"
                    )

    if len(invalid_or_expired_orders) > 0:
        logger.info(
            f"Found {len(invalid_or_expired_orders)} invalid or expired open orders that "
            f"will be canceled..."
        )

        for order in invalid_or_expired_orders:
            res = bybit.cancel_active_order(
                symbol=Symbol[str(order.symbol)], order_id=order.order_id
            )
            if res:
                cancelled_orders.append(order)
                logger.info(
                    f"Order {order.order_id} for {order.symbol} successfully cancelled!"
                )
    else:
        logger.info(
            f"No orders for {[s.value for s in symbols]} were found to be invalid and/or expired, "
            f"so nothing will be cancelled"
        )

    return cancelled_orders


def merge_take_profits_with_same_price(
    orders: list[ActiveOrder],
) -> list[MinifiedOrder]:
    summed_orders: dict[str, float] = {}
    for order in orders:
        summed_orders[str(order.price)] = (
            summed_orders.get(str(order.price), 0) + order.qty
        )

    result = [
        MinifiedOrder(price=float(price), qty=int(round(qty)))
        for price, qty in summed_orders.items()
    ]
    return result
