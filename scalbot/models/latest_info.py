from pydantic import BaseModel

from scalbot.enums import Symbol


class LatestInfo(BaseModel):
    symbol: Symbol
    bid_price: float
    ask_price: float
    last_price: float
    prev_price_24h: float
    price_24h_pcnt: float
    high_price_24h: float
    low_price_24h: float
    prev_price_1h: float
    price_1h_pcnt: float
    mark_price: float
    index_price: float

    class Config:
        use_enum_values = True
        extra = "allow"
