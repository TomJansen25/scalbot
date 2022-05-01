from pydantic import BaseModel, Field


class OpenPosition(BaseModel):
    size: int = Field(default=0, ge=0)
    open_tp: int = Field(default=0, ge=0)
    open_sl: int = Field(default=0, ge=0)
