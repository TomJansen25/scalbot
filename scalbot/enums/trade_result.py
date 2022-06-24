from enum import Enum, unique


@unique
class TradeResult(str, Enum):
    """
    Enum Class to define which Brokers can be used
    """

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
