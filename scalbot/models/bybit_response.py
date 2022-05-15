from pydantic import BaseModel, Field


class BybitResponse(BaseModel):
    ret_msg: str
    ext_code: str
    ext_info: str
    result: dict
    time_now: str
    rate_limit_status: int
    rate_limit_reset_ms: float
    rate_limit: int
