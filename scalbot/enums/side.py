from enum import Enum, unique


@unique
class Side(str, Enum):
    """
    Enum Class to define which Brokers can be used
    """

    BUY = "Buy"
    SELL = "Sell"
