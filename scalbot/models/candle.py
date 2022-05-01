from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
