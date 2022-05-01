from enum import Enum, unique


@unique
class Broker(Enum):
    """
    Enum Class to define which Brokers can be used
    """

    BYBIT = "Bybit"
    BINANCE = "Binance"
    BITVAVO = "Bitvavo"
