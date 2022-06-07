from datetime import datetime
from typing import Optional, Union
from uuid import UUID, uuid4

import numpy as np
from pydantic import UUID4, BaseModel, Field, root_validator, validator

from scalbot.enums import Broker, Symbol


def uuid_as_string():
    return str(uuid4())


class Trade(BaseModel):
    timestamp: datetime
    source_candle: datetime
    pattern: Union[str, dict, None] = None
    side: str = Field(default="Buy")
    symbol: Symbol = Field(default=Symbol.BTCUSD)
    price: float = Field()
    quantity_usd: int
    position_size: float
    stop_loss: float
    take_profit_1: float
    take_profit_1_share: float = Field(gt=0)
    take_profit_2: float
    take_profit_2_share: float = Field(ge=0)
    take_profit_3: float
    take_profit_3_share: float = Field(ge=0)
    broker: Optional[Broker] = None
    order_id: Optional[UUID4] = None
    order_link_id: Optional[UUID4] = Field(default_factory=uuid_as_string)

    class Config:
        use_enum_values = True

    @validator("side")
    def valid_side(cls, side):
        if side not in ["Buy", "Sell"]:
            raise ValueError(f"invalid side ({side}) provided")
        return side

    @root_validator
    def valid_take_profit_shares(cls, values):
        tp1_share = values.get("take_profit_1_share")
        tp2_share = values.get("take_profit_2_share")
        tp3_share = values.get("take_profit_3_share")
        total_tp_share = tp1_share + tp2_share + tp3_share
        qty = values.get("quantity_usd")
        if np.round(total_tp_share) != qty:
            raise ValueError(
                f"Sum of all TP Shares does not amount to Quantity of {qty}"
                f", but to {total_tp_share}"
            )
        return values

    @validator("order_link_id")
    def valid_uuid(cls, order_link_id):
        if isinstance(order_link_id, UUID):
            return order_link_id

        UUID(str(order_link_id))
        return order_link_id
