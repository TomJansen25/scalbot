from typing import Optional

from pydantic import BaseModel, Field


class BybitResponse(BaseModel):
    ret_code: int
    ret_msg: str
    ext_code: str
    ext_info: str
    result: Optional[dict]
    time_now: Optional[str]
    rate_limit_status: Optional[int]
    rate_limit_reset_ms: Optional[float]
    rate_limit: Optional[int]
