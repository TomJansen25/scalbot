from enum import Enum, unique


@unique
class Symbol(Enum):
    """
    Enum Class to define which Symbols can be traded
    """

    ADAUSD = "ADAUSD"
    BITUSD = "BITUSD"
    BTCUSD = "BTCUSD"
    ETHUSD = "ETHUSD"
    XRPUSD = "XRPUSD"
