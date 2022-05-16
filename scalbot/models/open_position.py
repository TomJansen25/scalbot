from pydantic import BaseModel, Field


class OpenPosition(BaseModel):
    """
    Open Position Model that returns the size/quantity information about the current position of a Symbol
    """

    size: int = Field(
        title="Size", description="Size of current open position", default=0, ge=0
    )
    open_tp: int = Field(
        title="Open TP",
        alias="Open Take Profit Size",
        description="Size (quantity) of current open take profit orders on the position",
        default=0,
        ge=0,
    )
    open_sl: int = Field(
        title="Open SL",
        alias="Open Stop Loss Size",
        description="Size (quantity) of current open stop loss orders on the position",
        default=0,
        ge=0,
    )
