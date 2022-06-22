from abc import ABC
from datetime import timedelta
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger
from pydantic import BaseModel

from scalbot.enums import Symbol
from scalbot.models import Candle
from scalbot.scalbot import Scalbot
from scalbot.trades import TradingStrategy

V_PATTERNS = [
    {
        "color": "green",
        "prev_color_1": "green",
        "prev_color_2": "green",
        "prev_color_3": "red",
    },
    {
        "color": "red",
        "prev_color_1": "red",
        "prev_color_2": "red",
        "prev_color_3": "green",
    },
]


class BackTesting(ABC, BaseModel):
    """
    BackTesting Class
    """

    symbol: Symbol
    timespan: relativedelta
    candles: Optional[pd.DataFrame] = None
    v_patterns: Optional[list] = None

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def __init__(
        self,
        symbol: Symbol,
        timespan: relativedelta = relativedelta(months=6),
        candles: Optional[pd.DataFrame] = None,
        v_patterns: list[dict] = None,
    ):
        super().__init__(
            symbol=symbol,
            timespan=timespan,
            candles=candles,
            v_patterns=v_patterns,
        )

    def simulate_parameters(
        self,
        df: pd.DataFrame,
        risk: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_1_share: float,
        take_profit_2: float,
        take_profit_2_share: float,
        take_profit_3: float,
        take_profit_3_share: float,
    ):
        trading_strategy = TradingStrategy(
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
        logger.info(trading_strategy)
        scalbot = Scalbot(
            patterns=V_PATTERNS, trading_strategy=trading_strategy, candle_frequency=3
        )

        df[["make_trade", "trade", "pattern", "trade_candle"]] = df.apply(
            lambda x: scalbot.evaluate_candle(
                candle=Candle.parse_obj(x.squeeze().to_dict()), df=df
            ),
            axis=1,
        ).to_list()

        trades = [
            scalbot.trading_strategy.define_trade(
                row.trade, Candle.parse_obj(row.trade_candle), row.pattern
            )
            for row in df.loc[df.make_trade].itertuples()
        ]

        trades_dict = [trade.dict() for trade in trades]
        trades_df = pd.DataFrame(trades_dict)
        trades_df.timestamp = trades_df.source_candle + timedelta(minutes=10)

        trades_df["trade_started"] = trades_df.apply(
            lambda x: find_next_price_occurrence(
                df=df, price=x.price, search_from=x.timestamp
            ),
            axis=1,
        )
        trades_df["stop_loss_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence(
                df=df, price=x.stop_loss, search_from=x.trade_started
            ),
            axis=1,
        )
        trades_df["tp1_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence(
                df=df, price=x.take_profit_1, search_from=x.trade_started
            ),
            axis=1,
        )
        trades_df["tp2_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence(
                df=df, price=x.take_profit_2, search_from=x.trade_started
            ),
            axis=1,
        )
        trades_df["tp3_reached"] = trades_df.apply(
            lambda x: find_next_price_occurrence(
                df=df, price=x.take_profit_3, search_from=x.trade_started
            ),
            axis=1,
        )

        trades_df["result"] = np.where(
            trades_df.stop_loss_reached < trades_df.tp1_reached, "STOP_LOSS", ""
        )
        trades_df.result = np.where(
            (trades_df.stop_loss_reached.notna()) & (trades_df.tp1_reached.isna()),
            "STOP_LOSS",
            trades_df.result,
        )
        trades_df.result = np.where(
            trades_df.tp1_reached < trades_df.stop_loss_reached, "TP1", trades_df.result
        )
        trades_df.result = np.where(
            trades_df.tp2_reached < trades_df.stop_loss_reached, "TP2", trades_df.result
        )
        trades_df.result = np.where(
            trades_df.tp3_reached < trades_df.stop_loss_reached, "TP3", trades_df.result
        )
        trades_df.result = np.where(
            (trades_df.tp3_reached.notna()) & (trades_df.stop_loss_reached.isna()),
            "TP3",
            trades_df.result,
        )
        trades_df.result = trades_df.result.fillna(np.nan)

        trades_df["status"] = np.where(
            (trades_df.stop_loss_reached.notna()) | (trades_df.tp3_reached.notna()),
            "CLOSED",
            "OPEN",
        )
        trades_df.status = np.where(
            trades_df.trade_started.isna(), "PENDING", trades_df.status
        )

        trades_df["stop_loss_result"] = (
            np.abs(trades_df.take_profit_1 - trades_df.price) * trades_df.position_size
        )
        trades_df["tp1_result"] = (
            np.abs(trades_df.take_profit_1 - trades_df.price)
            * trades_df.position_size
            * trades_df.take_profit_1_share
        )
        trades_df["tp2_result"] = (
            np.abs(trades_df.take_profit_2 - trades_df.price)
            * trades_df.position_size
            * trades_df.take_profit_2_share
        )
        trades_df["tp3_result"] = (
            np.abs(trades_df.take_profit_3 - trades_df.price)
            * trades_df.position_size
            * trades_df.take_profit_3_share
        )

        trades_df["profit"] = np.where(
            trades_df.result == "TP3",
            trades_df.tp3_result + trades_df.tp2_result + trades_df.tp1_result,
            0,
        )
        trades_df["profit"] = np.where(
            trades_df.result == "TP2",
            trades_df.tp2_result + trades_df.tp1_result,
            trades_df.profit,
        )
        trades_df["profit"] = np.where(
            trades_df.result == "TP1", trades_df.tp1_result, trades_df.profit
        )
        trades_df["profit"] = np.where(
            trades_df.result == "STOP_LOSS",
            -trades_df.stop_loss_result,
            trades_df.profit,
        )

        trades_df.profit = trades_df.profit.astype(float)
        result = trades_df.profit.sum()
        return result

    def objective(self, trial: optuna.Trial):

        df = self.candles.copy()

        # Parameter Grid Setup
        risk = trial.suggest_float("risk", 0.005, 0.02, step=0.001)
        stop_loss = trial.suggest_float("stop_loss", 0.001, 0.01, step=0.0001)

        # take_profit_1 = trial.suggest_float("take_profit_1", 0.001, 0.01, step = 0.0001)
        take_profit_1 = stop_loss
        take_profit_2 = trial.suggest_float(
            "take_profit_2", take_profit_1, 0.01, step=0.0001
        )
        take_profit_3 = trial.suggest_float(
            "take_profit_3", take_profit_2, 0.01, step=0.0001
        )
        """
        take_profit_1_share = trial.suggest_float("take_profit_1_share", 0.05, 1, step=0.05)

        remaining_share = 1 - take_profit_1_share
        take_profit_2_share = trial.suggest_float("take_profit_2_share", 0, remaining_share)

        remaining_share = 1 - take_profit_2_share - take_profit_1_share
        take_profit_3_share = trial.suggest_float("take_profit_3_share", remaining_share, remaining_share)
        """
        take_profit_1_share = 0.4
        take_profit_2_share = 0.3
        take_profit_3_share = 0.3

        return self.simulate_parameters(
            df=df,
            risk=risk,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_1_share=take_profit_1_share,
            take_profit_2=take_profit_2,
            take_profit_2_share=take_profit_2_share,
            take_profit_3=take_profit_3,
            take_profit_3_share=take_profit_3_share,
        )

    def run_study(self, n_trials: int = 50) -> optuna.Study:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
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
