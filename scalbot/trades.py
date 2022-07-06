import logging
from abc import ABC
from datetime import date, datetime
from typing import Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import pytz
from loguru import logger
from pydantic import BaseModel, Field

from scalbot.bigquery import BigQuery
from scalbot.bybit import Bybit
from scalbot.enums import Broker, Symbol, TradeResult
from scalbot.models import Candle, Trade
from scalbot.utils import setup_logging

setup_logging()


class TradingStrategy(ABC, BaseModel):
    """
    TradingStrategy class to define all kinds of scalping trades based on a particular SL + TP strategy
    """

    bet_amount: float = Field(
        default=0.01,
        gt=0,
        title="Bet Amount",
        description="Indicates the amount to be put in every calculated trade",
    )
    risk: float = Field(
        default=0.01, gt=0, title="Risk", description="Percentual risk to be used"
    )
    stop_loss: float = Field(default=0.0035, ge=0, le=1)
    take_profit_1: float = Field(default=0.0035, ge=0, le=1)
    take_profit_1_share: float = Field(default=0.4, gt=0, le=1)
    take_profit_2: float = Field(default=0.0060, ge=0, le=1)
    take_profit_2_share: float = Field(default=0.3, ge=0, lt=1)
    take_profit_3: float = Field(default=0.0080, ge=0, le=1)
    take_profit_3_share: float = Field(default=0.3, ge=0, lt=1)

    def __init__(
        self,
        *,
        bet_amount: float,
        risk: float,
        stop_loss: float = 0.0035,
        take_profit_1: float = 0.0035,
        take_profit_1_share: float = 0.40,
        take_profit_2: float = 0.0060,
        take_profit_2_share: float = 0.30,
        take_profit_3: float = 0.0080,
        take_profit_3_share: float = 0.30,
    ):
        if take_profit_1_share + take_profit_2_share + take_profit_3_share > 1:
            raise ValueError(
                "Shares of TP Shares cannot exceed 1 (i.e., 100%) together."
            )

        super().__init__(
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

    def get_sample_trade(self, price: int = 35000) -> Trade:

        quantity_usd = int(self.bet_amount * (self.risk / self.stop_loss))
        stop_loss = price + (price * self.stop_loss)
        tp1 = price - (price * self.take_profit_1)
        tp2 = price - (price * self.take_profit_2)
        tp3 = price - (price * self.take_profit_3)

        tp_shares = divide_quantity_over_shares(
            quantity=quantity_usd,
            shares={
                "tp1_share": self.take_profit_1_share,
                "tp2_share": self.take_profit_2_share,
                "tp3_share": self.take_profit_3_share,
            },
        )

        return Trade(
            timestamp=datetime.now(),
            source_candle=datetime.now(),
            pattern={"color": "red", "prev_color_1": "green"},
            side="Buy",
            symbol="BTCUSD",
            price=price,
            quantity_usd=quantity_usd,
            position_size=quantity_usd / price,
            stop_loss=round_trade_price(stop_loss),
            take_profit_1=round_trade_price(tp1),
            take_profit_1_share=tp_shares.get("tp1_share"),
            take_profit_2=round_trade_price(tp2),
            take_profit_2_share=tp_shares.get("tp2_share"),
            take_profit_3=round_trade_price(tp3),
            take_profit_3_share=tp_shares.get("tp3_share"),
            order_link_id=uuid4(),
        )

    def define_trade(self, trade_side: str, candle: Candle, pattern: dict) -> Trade:
        """

        :param trade_side:
        :param candle:
        :param pattern:
        :return:
        """
        if trade_side not in ["short", "long"]:
            raise KeyError(
                f"Provided trade value ({trade_side}) is not supported. Either choose "
                f'"short" or "long"...'
            )

        logging.info(f"Defining trade based on following candle: {candle}")

        if trade_side == "short":
            price = candle.low
            stop_loss = price + (price * self.stop_loss)
            tp1 = price - (price * self.take_profit_1)
            tp2 = price - (price * self.take_profit_2)
            tp3 = price - (price * self.take_profit_3)
        else:
            price = candle.high
            stop_loss = price - (price * self.stop_loss)
            tp1 = price + (price * self.take_profit_1)
            tp2 = price + (price * self.take_profit_2)
            tp3 = price + (price * self.take_profit_3)

        quantity_usd = int(self.bet_amount * (self.risk / self.stop_loss))
        side = "Buy" if trade_side == "long" else "Sell"

        tp_shares = divide_quantity_over_shares(
            quantity=quantity_usd,
            shares={
                "tp1_share": self.take_profit_1_share,
                "tp2_share": self.take_profit_2_share,
                "tp3_share": self.take_profit_3_share,
            },
        )

        return Trade(
            timestamp=datetime.now(),
            source_candle=candle.start,
            pattern=pattern,
            side=side,
            symbol=candle.symbol,
            price=round_trade_price(price),
            quantity_usd=quantity_usd,
            position_size=quantity_usd / price,
            stop_loss=round_trade_price(stop_loss),
            take_profit_1=round_trade_price(tp1),
            take_profit_1_share=tp_shares.get("tp1_share"),
            take_profit_2=round_trade_price(tp2),
            take_profit_2_share=tp_shares.get("tp2_share"),
            take_profit_3=round_trade_price(tp3),
            take_profit_3_share=tp_shares.get("tp3_share"),
            order_link_id=uuid4(),
        )


class TradeSummary(ABC, BaseModel):
    broker: Broker
    symbols: list[Symbol]
    bybit: Bybit
    bigquery_client: BigQuery
    today_datetime: Optional[datetime] = None
    today_timestamp: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, broker: Broker, symbols: list[Symbol], bybit: Bybit):
        """

        :param broker:
        :param symbols
        :param bybit::
        """
        super().__init__(
            broker=broker,
            symbols=symbols,
            bybit=bybit,
            bigquery_client=BigQuery(),
        )

        self.today_datetime, self.today_timestamp = self.get_today_timestamps()

    def get_today_timestamps(self) -> Tuple[datetime, int]:
        if not self.today_datetime or not self.today_timestamp:
            today = date.today()
            today_datetime = datetime(today.year, today.month, today.day).astimezone(
                pytz.UTC
            )
            today_timestamp = int(today_datetime.timestamp())
            self.today_datetime = today_datetime
            self.today_timestamp = today_timestamp
            return today_datetime, today_timestamp
        else:
            return self.today_datetime, self.today_timestamp

    def get_today_pnl(self, symbol: Symbol) -> pd.DataFrame:
        pnls = self.bybit.get_closed_profit_and_loss(
            symbol=symbol, start_time=int(self.today_timestamp)
        )
        if not pnls.get("data"):
            err_msg = (
                "Retrieved PnL data is empty, no profit and losses "
                "for today could be retrieved."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        pnl_df = pd.DataFrame(pnls.get("data"))
        pnl_df["created_at"] = pd.to_datetime(pnl_df.created_at, unit="s")
        pnl_df = pnl_df.drop(
            ["id", "user_id", "exec_type", "fill_count", "leverage"], 1
        )
        pnl_df = pnl_df.rename(columns={"created_at": "pnl_created_at"})
        return pnl_df

    def get_today_orders(self, symbol: Symbol) -> pd.DataFrame:
        orders = self.bybit.get_active_orders(symbol=symbol, limit=50)

        orders_df = pd.DataFrame([order.dict() for order in orders])

        orders_df["created_at"] = pd.to_datetime(orders_df.created_at)
        orders_df["updated_at"] = pd.to_datetime(orders_df.updated_at)

        orders_df = orders_df.loc[orders_df.created_at > self.today_datetime]

        orders_df = orders_df.drop(
            [
                "user_id",
                "position_idx",
                "time_in_force",
                "leaves_qty",
                "leaves_value",
                "tp_trigger_by",
                "sl_trigger_by",
            ],
            axis=1,
        )

        orders_df.qty = orders_df.qty.astype("int64")

        logger.info(
            f"{len(orders_df)} orders from today on {symbol.value} were retrieved"
        )
        return orders_df

    @staticmethod
    def _get_trade_result_from_filled_tps_sls(
        filled_tps_sls: list[str], as_string: bool = False
    ) -> Union[TradeResult, str]:
        if TradeResult.TAKE_PROFIT_3.value in filled_tps_sls:
            res = TradeResult.TAKE_PROFIT_3
        elif TradeResult.TAKE_PROFIT_2.value in filled_tps_sls:
            res = TradeResult.TAKE_PROFIT_2
        elif TradeResult.TAKE_PROFIT_1.value in filled_tps_sls:
            res = TradeResult.TAKE_PROFIT_1
        else:
            res = TradeResult.STOP_LOSS

        if as_string:
            return res.value
        return res

    def _get_symbol_prices_as_df(self, symbols: list[Symbol]) -> pd.DataFrame:
        price_dict = {}

        for symbol in symbols:
            latest_info = self.bybit.get_latest_symbol_information(symbol=symbol)
            price_dict[symbol.value] = latest_info.last_price

        df = pd.DataFrame.from_dict(
            {"symbol": price_dict.keys(), "price_usd": price_dict.values()}
        )
        return df

    def get_trade_summary_as_df(self) -> pd.DataFrame:
        trade_summary_dfs = []

        for symbol in self.symbols:
            pnl_df = self.get_today_pnl(symbol=symbol)
            orders_df = self.get_today_orders(symbol=symbol)
            orders_pnl_df = orders_df.merge(
                pnl_df.drop(columns=["side", "order_type"]),
                how="left",
                on=["symbol", "order_id", "qty"],
            )

            today_trades_df = self.bigquery_client.get_today_trades(symbols=[symbol])
            today_trades_df["next_ts"] = today_trades_df.timestamp.shift(-1)

            full_df = orders_pnl_df.merge(
                today_trades_df[["order_id", "order_link_id", "price"]],
                on="order_id",
                how="left",
                suffixes=("", "_trade"),
            )
            full_df.order_link_id_trade = full_df.order_link_id_trade.fillna(
                method="backfill"
            )

            tp_sl_trades_df = today_trades_df.melt(
                id_vars=["order_link_id"],
                value_vars=["take_profit_1", "take_profit_2", "take_profit_3"],
                var_name="tp_sl_type",
                value_name="calc_price",
            )

            full_df = full_df.merge(
                tp_sl_trades_df,
                left_on=["order_link_id_trade", "price"],
                right_on=["order_link_id", "calc_price"],
                how="left",
            )

            full_df.calc_price = full_df.calc_price.fillna(full_df.price)
            full_df.tp_sl_type = full_df.tp_sl_type.fillna("stop_loss")

            full_df = full_df.loc[
                full_df.closed_pnl.notna(),
                [
                    "order_type",
                    "price",
                    "qty",
                    "created_at",
                    "updated_at",
                    "order_link_id_trade",
                    "order_id",
                    "closed_pnl",
                    "pnl_created_at",
                    "calc_price",
                    "tp_sl_type",
                ],
            ].sort_values("created_at", ascending=False)

            trade_summary_df = (
                full_df.groupby("order_link_id_trade")
                .agg({"closed_pnl": "sum", "tp_sl_type": "unique"})
                .reset_index()
                .rename(columns={"order_link_id_trade": "order_link_id"})
            )

            trade_summary_df["trade_result"] = trade_summary_df.apply(
                lambda x: self._get_trade_result_from_filled_tps_sls(
                    x.tp_sl_type, as_string=True
                ),
                axis=1,
            )

            trade_summary_df = trade_summary_df.drop(columns=["tp_sl_type"]).merge(
                today_trades_df[
                    [
                        "order_link_id",
                        "timestamp",
                        "source_candle",
                        "side",
                        "symbol",
                        "price",
                        "quantity_usd",
                    ]
                ],
                on="order_link_id",
                how="left",
            )

            trade_summary_df["pnl_usd"] = (
                trade_summary_df.closed_pnl * trade_summary_df.price
            )
            trade_summary_dfs.append(trade_summary_df)

        trade_summary_df = pd.concat(trade_summary_dfs, ignore_index=True).sort_values(
            by="timestamp"
        )

        trade_summary_df.timestamp = trade_summary_df.timestamp.dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        trade_summary_df.source_candle = trade_summary_df.source_candle.dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        trade_summary_df.columns = [
            "Order Link ID",
            "Closed PnL",
            "Trade Result",
            "Trade Timestamp",
            "Source Candle Timestamp",
            "Side",
            "Symbol",
            "Price",
            "Quantity (USD)",
            "PnL (USD)",
        ]

        return trade_summary_df


def divide_quantity_over_shares(
    quantity: int, shares: dict[str, float]
) -> dict[str, int]:

    total = 0
    calculated_shares = {}

    for share_id, share in shares.items():
        calc_share = round(quantity * share)
        total += calc_share
        calculated_shares[share_id] = calc_share

    diff = quantity - total
    if diff != 0:
        first_key = next(iter(calculated_shares))
        calculated_shares[first_key] = calculated_shares.get(first_key, 0) + diff

    return calculated_shares


def round_trade_price(price: Union[int, float]):
    if price <= 10:
        price = np.round(price / 0.005) * 0.005
        return np.round(price, 3)
    elif price <= 100:
        return np.round(price, 2)
    elif price <= 1000:
        price = np.round(price / 0.05) * 0.05
        return np.round(price, 2)
    elif price < 10000:
        return np.round(price, 1)
    else:  # price > 10000
        price = np.round(price / 0.5) * 0.5
        return np.round(price, 1)
