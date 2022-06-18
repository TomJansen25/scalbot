from enum import Enum, unique


@unique
class Symbol(Enum):
    """
    Enum Class to define which Symbols can be traded
    """

    ADAUSD = "ADAUSD"
    BITUSD = "BITUSD"
    BTCUSD = "BTCUSD"
    DOTUSD = "DOTUSD"
    EOSUSD = "EOSUSD"
    ETHUSD = "ETHUSD"
    LUNAUSD = "LUNAUSD"
    MANAUSD = "MANAUSD"
    XRPUSD = "XRPUSD"
