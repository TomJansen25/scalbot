from datetime import datetime

from pydantic import BaseModel, Field

from scalbot.enums import OrderType, Side, Symbol


class Order(BaseModel):
    user_id: int
    position_idx: int
    symbol: Symbol
    side: Side
    order_type: OrderType
    price: float
    qty: float
    time_in_force: str
    order_link_id: str
    created_at: datetime
    updated_at: datetime
    take_profit: float
    stop_loss: float
    tp_trigger_by: str
    sl_trigger_by: str

    class Config:
        use_enum_values = True
        extra = "allow"


class ActiveOrder(Order):
    order_status: str
    order_id: str


class ConditionalOrder(Order):
    stop_order_status: str
    stop_order_type: str
    stop_order_id: str
    stop_px: float
    trigger_by: str
