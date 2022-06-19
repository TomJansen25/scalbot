from enum import Enum, unique


@unique
class OrderType(str, Enum):
    """
    Enum Class to define which Brokers can be used
    """

    LIMIT = "Limit"
    MARKET = "Market"
