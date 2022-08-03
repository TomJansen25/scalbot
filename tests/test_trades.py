from datetime import datetime

import pytest

from scalbot.enums import Broker, Symbol
from scalbot.models import Trade
from scalbot.trades import divide_quantity_over_shares, round_trade_price


@pytest.fixture(scope="module")
def sample_trade(
    symbol: Symbol = Symbol.BTCUSD,
    broker: Broker = Broker.BYBIT,
    price: float = 22000.50,
):
    stop_loss = price + (price * 0.01)
    tp1 = price - (price * 0.01)
    tp2 = price - (price * 0.02)
    tp3 = price - (price * 0.03)

    return Trade(
        timestamp=datetime.now(),
        source_candle=datetime.now(),
        side="Sell",
        symbol=symbol,
        price=price,
        quantity_usd=price,
        position_size=1,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_1_share=10,
        take_profit_2=tp2,
        take_profit_2_share=10,
        take_profit_3=tp3,
        take_profit_3_share=10,
        broker=broker,
    )


########## UNIT TESTS ##########


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (6.3221, 6.320),
        (0.9932, 0.995),
        (11, 11.00),
        (54.543, 54.54),
        (123.131, 123.15),
        (832.758, 832.75),
        (5555.92, 5555.9),
        (7210.22, 7210.2),
        (19117.40, 19117.5),
        (21503.88, 21504.0),
    ],
)
def test_round_trade_prices(test_input: float, expected: float):
    assert round_trade_price(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            dict(
                quantity=10,
                shares={"tp1_share": 0.4, "tp2_share": 0.3, "tp3_share": 0.3},
            ),
            {"tp1_share": 4, "tp2_share": 3, "tp3_share": 3},
        ),
        (
            dict(
                quantity=20,
                shares={"tp1_share": 0.5, "tp2_share": 0.3, "tp3_share": 0.2},
            ),
            {"tp1_share": 10, "tp2_share": 6, "tp3_share": 4},
        ),
        (
            dict(
                quantity=25,
                shares={"tp1_share": 0.5, "tp2_share": 0.3, "tp3_share": 0.2},
            ),
            {"tp1_share": 12, "tp2_share": 8, "tp3_share": 5},
        ),
        (
            dict(
                quantity=530,
                shares={"tp1_share": 0.6, "tp2_share": 0.25, "tp3_share": 0.15},
            ),
            {"tp1_share": 318, "tp2_share": 132, "tp3_share": 80},
        ),
    ],
)
def test_divide_quantity_over_shares(test_input, expected):
    assert divide_quantity_over_shares(**test_input) == expected


########## INTEGRATION TESTS ##########
