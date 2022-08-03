from abc import ABC
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger
from pydantic import BaseModel

from scalbot.data import HistoricData
from scalbot.enums import OrderType, Symbol, TradeResult
from scalbot.models import Candle
from scalbot.scalbot import Scalbot
from scalbot.trades import TradeCalculator
from scalbot.utils import get_params

OPTIMIZE_OPTIONS = ["MINIMIZE_STOP_LOSSES", "MAXIMIZE_PROFIT", ""]

TRADE_RESULT_MAP = {
    TradeResult.STOP_LOSS.value: -1,
    TradeResult.TAKE_PROFIT_1.value: 0,
    TradeResult.TAKE_PROFIT_2.value: 1,
    TradeResult.TAKE_PROFIT_3.value: 2,
}

DEFAULT_SEARCH_SPACE: dict[str, dict[str, float]] = dict(
    risk=dict(low=0.005, high=0.02, step=0.001),
    stop_loss=dict(low=0.001, high=0.01, step=0.0001),
    take_profit_1=dict(low=0.001, high=0.01, step=0.0001),
    take_profit_2=dict(high=0.01, step=0.0001),
    take_profit_3=dict(high=0.01, step=0.0001),
    take_profit_1_share=dict(low=0.25, high=0.75, step=0.01),
    take_profit_2_share=dict(low=0.10, high=0.5, step=0.01),
    take_profit_3_share=dict(low=0.10, high=0.5, step=0.01),
)


class BackTesting(ABC, BaseModel):
    """
    BackTesting Class
    """

    symbol: Symbol
    timespan: relativedelta
    candles: pd.DataFrame
    candle_patterns: list
    optimizer: str
    optimization_search_space: dict

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def __init__(
        self,
        symbol: Symbol,
        timespan: relativedelta = relativedelta(months=6),
        candles: Optional[pd.DataFrame] = None,
        candle_patterns: list[dict] = None,
        optimizer: str = "MAXIMIZE_PROFIT",
        optimization_search_space: dict = None,
    ):
        if optimizer not in OPTIMIZE_OPTIONS:
            raise KeyError(
                f"Provided optimizer ({optimizer}) is not supported. "
                f"Choose one of: {OPTIMIZE_OPTIONS}"
            )

        if candles is None:
            candles = HistoricData(symbol=symbol).get_latest_local_data()
        if not candle_patterns:
            candle_patterns = (
                get_params().get(symbol.value).get("scalbot").get("patterns")
            )
        if not optimization_search_space:
            optimization_search_space = DEFAULT_SEARCH_SPACE

        super().__init__(
            symbol=symbol,
            timespan=timespan,
            candles=candles,
            candle_patterns=candle_patterns,
            optimizer=optimizer,
            optimization_search_space=optimization_search_space,
        )

    def simulate_parameters(
        self,
        df: pd.DataFrame,
        bet_amount: float,
        risk: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_1_share: float,
        take_profit_2: float,
        take_profit_2_share: float,
        take_profit_3: float,
        take_profit_3_share: float,
    ) -> Tuple[pd.DataFrame, float, float]:

        trade_calculator = TradeCalculator(
            bet_amount=bet_amount,
            risk=risk,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_1_share=take_profit_1_share,
            take_profit_2=take_profit_2,
            take_profit_2_share=take_profit_2_share,
            take_profit_3=take_profit_3,
            take_profit_3_share=take_profit_3_share,
        )
        logger.info(
            f"Start simulating following trade calculation strategy: {trade_calculator}..."
        )
        scalbot = Scalbot(
            patterns=self.candle_patterns,
            trade_calculator=trade_calculator,
            candle_frequency=3,
        )

        df = df.copy()

        df[["make_trade", "trade", "pattern", "trade_candle"]] = df.apply(
            lambda x: scalbot.evaluate_candle(
                candle=Candle.parse_obj(x.squeeze().to_dict()), df=df
            ),
            axis=1,
        ).to_list()

        trades = [
            scalbot.trade_calculator.define_trade(
                row.trade, Candle.parse_obj(row.trade_candle), row.pattern
            )
            for row in df.loc[df.make_trade].itertuples()
        ]

        trades_dict = [trade.dict() for trade in trades]
        trades_df = pd.DataFrame(trades_dict)
        trades_df.timestamp = trades_df.source_candle + timedelta(minutes=10)

        trades_df["trade_open_fees"] = trades_df.apply(
            lambda x: calculate_trade_fees(
                quantity=x.quantity_usd, price=x.price, order_type=OrderType.LIMIT
            ),
            axis=1,
        )
        trades_df["stop_loss_fees"] = trades_df.apply(
            lambda x: calculate_trade_fees(
                quantity=x.quantity_usd, price=x.stop_loss, order_type=OrderType.MARKET
            ),
            axis=1,
        )
        trades_df["tp1_fees"] = trades_df.apply(
            lambda x: calculate_trade_fees(
                quantity=x.take_profit_1_share,
                price=x.take_profit_1,
                order_type=OrderType.LIMIT,
            ),
            axis=1,
        )
        trades_df["tp2_fees"] = trades_df.apply(
            lambda x: calculate_trade_fees(
                quantity=x.take_profit_2_share,
                price=x.take_profit_2,
                order_type=OrderType.LIMIT,
            ),
            axis=1,
        )
        trades_df["tp3_fees"] = trades_df.apply(
            lambda x: calculate_trade_fees(
                quantity=x.take_profit_3_share,
                price=x.take_profit_3,
                order_type=OrderType.LIMIT,
            ),
            axis=1,
        )

        df_as_dict = df[["start", "high", "low"]].copy().to_dict("index")

        trades_df["trade_started"] = trades_df.apply(
            lambda x: find_next_price_occurrence_from_dict(
                df_as_dict=df_as_dict, price=x.price, search_from=x.timestamp
            ),
            axis=1,
        )
        trades_df["stop_loss_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence_from_dict(
                df_as_dict=df_as_dict, price=x.stop_loss, search_from=x.trade_started
            ),
            axis=1,
        )
        trades_df["tp1_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence_from_dict(
                df_as_dict=df_as_dict,
                price=x.take_profit_1,
                search_from=x.trade_started,
            ),
            axis=1,
        )
        trades_df["tp2_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence_from_dict(
                df_as_dict=df_as_dict,
                price=x.take_profit_2,
                search_from=x.trade_started,
            ),
            axis=1,
        )
        trades_df["tp3_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence_from_dict(
                df_as_dict=df_as_dict,
                price=x.take_profit_3,
                search_from=x.trade_started,
            ),
            axis=1,
        )

        trades_df["result"] = np.where(
            trades_df.stop_loss_reached < trades_df.tp1_reached,
            TradeResult.STOP_LOSS.value,
            "",
        )
        trades_df.result = np.where(
            (trades_df.stop_loss_reached.notna()) & (trades_df.tp1_reached.isna()),
            TradeResult.STOP_LOSS.value,
            trades_df.result,
        )
        trades_df.result = np.where(
            trades_df.tp1_reached < trades_df.stop_loss_reached,
            TradeResult.TAKE_PROFIT_1.value,
            trades_df.result,
        )
        trades_df.result = np.where(
            trades_df.tp2_reached < trades_df.stop_loss_reached,
            TradeResult.TAKE_PROFIT_2.value,
            trades_df.result,
        )
        trades_df.result = np.where(
            trades_df.tp3_reached < trades_df.stop_loss_reached,
            TradeResult.TAKE_PROFIT_3.value,
            trades_df.result,
        )
        trades_df.result = np.where(
            (trades_df.tp3_reached.notna()) & (trades_df.stop_loss_reached.isna()),
            TradeResult.TAKE_PROFIT_3.value,
            trades_df.result,
        )
        trades_df.result = trades_df.result.fillna("")

        trades_df["status"] = np.where(
            (trades_df.stop_loss_reached.notna()) | (trades_df.tp3_reached.notna()),
            "CLOSED",
            "OPEN",
        )
        trades_df.status = np.where(
            trades_df.trade_started.isna(), "PENDING", trades_df.status
        )
        trades_df.status = np.where(
            (
                trades_df.stop_loss_reached
                > trades_df.trade_started + timedelta(hours=24)
            )
            & (trades_df.tp1_reached > trades_df.trade_started + timedelta(hours=24)),
            "CANCELED",
            trades_df.status,
        )

        trades_df["sl_win"] = -np.abs(
            trades_df.quantity_usd / trades_df.stop_loss - trades_df.position_size
        )
        trades_df["tp1_win"] = np.abs(
            (trades_df.take_profit_1_share / trades_df.take_profit_1)
            - (trades_df.take_profit_1_share / trades_df.price)
        )
        trades_df["tp2_win"] = np.abs(
            (trades_df.take_profit_2_share / trades_df.take_profit_2)
            - (trades_df.take_profit_2_share / trades_df.price)
        )
        trades_df["tp3_win"] = np.abs(
            (trades_df.take_profit_3_share / trades_df.take_profit_3)
            - (trades_df.take_profit_3_share / trades_df.price)
        )

        trades_df["pnl"] = trades_df.apply(lambda x: calculate_pnl(x), axis=1)
        trades_df["pnl_usd"] = trades_df.pnl * trades_df.price
        trades_df = trades_df.loc[
            (trades_df.status == "CLOSED") & (trades_df.result.notna())
        ]
        result_dict = trades_df.result.value_counts().to_dict()
        logger.info(f"Trade results of current simulation: {result_dict}")
        trades_df["result_score"] = trades_df.result.map(TRADE_RESULT_MAP)

        result_score = trades_df.result_score.sum()
        total_pnl = trades_df.pnl_usd.sum()

        return trades_df, result_score, total_pnl

    def optimize_trade_calculation_objective(self, trial: optuna.Trial):
        """
        Defines an Optuna Trial used in function below to optimize Trade Calculation parameters.
        Parameter search space is found and accessed in the "optimization_search_space" Class
        variable
        :param trial: Optuna Trial defined by an Optuna Study Run
        :return: Objective to optimize for the Optuna Study
        """
        df = self.candles.copy()
        max_timestamp = df.start.max()
        df = df.loc[df.start >= max_timestamp - self.timespan]

        # Parameter Grid Setup
        risk = trial.suggest_float(
            name="risk", **self.optimization_search_space.get("risk")
        )
        stop_loss = trial.suggest_float(
            name="stop_loss", **self.optimization_search_space.get("stop_loss")
        )

        take_profit_1 = stop_loss
        take_profit_2 = trial.suggest_float(
            name="take_profit_2",
            low=take_profit_1,
            **self.optimization_search_space.get("take_profit_2"),
        )
        take_profit_3 = trial.suggest_float(
            name="take_profit_3",
            low=take_profit_2,
            **self.optimization_search_space.get("take_profit_3"),
        )
        take_profit_1_share = 0.4
        take_profit_2_share = 0.3
        take_profit_3_share = 0.3

        _, trades_score, trades_pnl = self.simulate_parameters(
            df=df,
            bet_amount=100,
            risk=risk,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_1_share=take_profit_1_share,
            take_profit_2=take_profit_2,
            take_profit_2_share=take_profit_2_share,
            take_profit_3=take_profit_3,
            take_profit_3_share=take_profit_3_share,
        )

        return trades_pnl

    def optimize_trade_calculation(self, n_trials: int = 50) -> optuna.Study:
        """
        Run an Optuna Study to optimize TradeCalculator parameters, maximizing one or several objectives
        :param n_trials:
        :return:
        """
        study = optuna.create_study(
            study_name="optimize_trade_calculator", direction="maximize"
        )
        study.optimize(self.optimize_trade_calculation_objective, n_trials=n_trials)
        logger.info(f"Best found value: {study.best_value}")
        logger.info(f"Best found parameters: {study.best_params}")
        return study


def find_next_price_occurrence(
    df: pd.DataFrame, price: float, search_from: pd.Timestamp
) -> pd.Timestamp:
    df = df.loc[df.start > search_from].copy()
    index = df.loc[(df.low < price) & (df.high > price)].first_valid_index()
    timestamp = df.loc[df.index == index, "start"].squeeze() if index else None
    return timestamp


def find_next_price_occurrence_from_dict(
    df_as_dict, price, search_from
) -> Optional[pd.Timestamp]:
    filtered_dict: dict = {
        k: v
        for k, v in df_as_dict.items()
        if v.get("low") < price < v.get("high") and v.get("start") > search_from
    } | {}
    if len(filtered_dict) > 0:
        first_key = next(iter(filtered_dict))
        first_timestamp = filtered_dict.get(first_key).get("start")
        return first_timestamp
    else:
        return None


def calculate_trade_fees(
    quantity: int, price: float, order_type: OrderType = OrderType.LIMIT
) -> float:
    position = quantity / price
    trade_fees = {OrderType.LIMIT: 0.0001, OrderType.MARKET: 0.0006}
    trade_fee: float = trade_fees.get(order_type, 0)
    return position * trade_fee


def calculate_pnl(s: pd.Series) -> float:
    result = s.result
    if result == TradeResult.STOP_LOSS.value:
        pnl = -s.trade_open_fees - s.stop_loss_fees + s.sl_win
    elif result == TradeResult.TAKE_PROFIT_1.value:
        pnl = -s.trade_open_fees - s.stop_loss_fees + s.tp1_win - s.tp1_fees
    elif result == TradeResult.TAKE_PROFIT_2.value:
        pnl = (
            -s.trade_open_fees
            - s.stop_loss_fees
            + s.tp1_win
            - s.tp1_fees
            + s.tp2_win
            - s.tp2_fees
        )
    elif result == TradeResult.TAKE_PROFIT_3.value:
        pnl = (
            -s.trade_open_fees
            - s.stop_loss_fees
            + s.tp1_win
            - s.tp1_fees
            + s.tp2_win
            - s.tp2_fees
            + s.tp3_win
            - s.tp3_fees
        )
    else:
        pnl = 0
    return pnl
