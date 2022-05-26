from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from scalbot.enums import Symbol


class Candle(BaseModel):
    symbol: Symbol
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    turnover: Optional[float]
    color: Optional[str]
    previous_colors: Optional[dict]
    sma: Optional[float]

    class Config:
        use_enum_values = True
