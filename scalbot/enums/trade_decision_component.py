from enum import Enum, unique


@unique
class TradeDecisionComponent(str, Enum):
    """
    Enum Class to define which Brokers can be used
    """

    CANDLE_PATTERN = "CANDLE_PATTERN"
    SIMPLE_MOVING_AVERAGE = "SIMPLE_MOVING_AVERAGE"
    PREVIOUS_CANDLE_TREND = "PREVIOUS_CANDLE_TREND"
    VOLUME = "VOLUME"
