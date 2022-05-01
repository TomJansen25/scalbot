import pytest

from scalbot.models import OpenPosition


@pytest.fixture()
def full_open_tp_position():
    return OpenPosition(size=100, open_tp=100, open_sl=0)


@pytest.fixture()
def untriggered_take_profits():
    tps = [
        {
            "user_id": 461517,
            "stop_order_status": "Untriggered",
            "symbol": "BTCUSD",
            "side": "Sell",
            "order_type": "Market",
            "stop_order_type": "PartialTakeProfit",
            "price": "0",
            "qty": "12",
            "base_price": "0",
            "order_link_id": "",
            "stop_px": "38684",
            "stop_order_id": "ff6b6ecd-79b5-4e2d-ae70-789c529b6920",
            "trigger_by": "LastPrice",
        },
        {
            "user_id": 461517,
            "stop_order_status": "Untriggered",
            "symbol": "BTCUSD",
            "side": "Sell",
            "order_type": "Market",
            "stop_order_type": "PartialTakeProfit",
            "price": "0",
            "qty": "16",
            "base_price": "0",
            "order_link_id": "",
            "stop_px": "38588",
            "stop_order_id": "4e2bd606-8d9a-4672-8350-268bd843fa54",
            "trigger_by": "LastPrice",
        },
        {
            "user_id": 461517,
            "stop_order_status": "Untriggered",
            "symbol": "BTCUSD",
            "side": "Sell",
            "order_type": "Market",
            "stop_order_type": "PartialTakeProfit",
            "price": "0",
            "qty": "12",
            "base_price": "0",
            "order_link_id": "",
            "stop_px": "38876.5",
            "stop_order_id": "5f06060f-5faa-42a4-91d8-8912b4a235a8",
            "trigger_by": "LastPrice",
        },
    ]
    return tps
