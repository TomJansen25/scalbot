import logging
from abc import ABC
from datetime import datetime
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

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

        return Trade(
            timestamp=datetime.now(),
            source_candle=datetime.now(),
            pattern="",
            side="Buy",
            symbol="BTCUSD",
            price=price,
            quantity_usd=quantity_usd,
            position_size=quantity_usd / price,
            stop_loss=np.round(price + (price * self.stop_loss), 2),
            take_profit_1=np.round(price - (price * self.take_profit_1), 2),
            take_profit_1_share=0.4,
            take_profit_2=np.round(price - (price * self.take_profit_2), 2),
            take_profit_2_share=0.3,
            take_profit_3=np.round(price - (price * self.take_profit_3), 2),
            take_profit_3_share=0.3,
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
            price=price,
            quantity_usd=quantity_usd,
            position_size=quantity_usd / price,
            stop_loss=np.round(stop_loss, 2),
            take_profit_1=np.round(tp1, 2),
            take_profit_1_share=tp_shares.get("tp1_share"),
            take_profit_2=np.round(tp2, 2),
            take_profit_2_share=tp_shares.get("tp2_share"),
            take_profit_3=np.round(tp3, 2),
            take_profit_3_share=tp_shares.get("tp3_share"),
            order_link_id=uuid4(),
        )


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
