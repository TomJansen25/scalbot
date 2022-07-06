from datetime import datetime, timedelta

import pytest

from scalbot.models import (
    ActiveOrder,
    BaseOrder,
    ConditionalOrder,
    LatestInfo,
    MinifiedOrder,
)
from scalbot.scalbot import (
    is_order_expired,
    is_order_price_too_far_off_last_price,
    merge_take_profits_with_same_price,
)


@pytest.fixture(scope="module")
def base_order_dict() -> dict:
    return dict(
        user_id=1,
        position_idx=1,
        symbol="BTCUSD",
        side="Buy",
        order_type="Limit",
        time_in_force="a",
        order_link_id="a",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        take_profit=0,
        stop_loss=0,
        tp_trigger_by="a",
        sl_trigger_by="a",
        order_status="a",
        order_id="a",
    )


@pytest.fixture(scope="module")
def latest_info_dict() -> dict:
    return dict(
        symbol="BTCUSD",
        bid_price=1,
        ask_price=1,
        prev_price_24h=1,
        price_24h_pcnt=1,
        high_price_24h=1,
        low_price_24h=1,
        prev_price_1h=1,
        price_1h_pcnt=1,
        mark_price=1,
        index_price=1,
    )


@pytest.fixture(scope="module")
def sample_orders_a(base_order_dict) -> list[ActiveOrder]:
    order_a = ActiveOrder(price=10, qty=15, **base_order_dict)
    order_b = ActiveOrder(price=20, qty=20, **base_order_dict)
    order_c = ActiveOrder(price=30, qty=25, **base_order_dict)
    return [order_a, order_b, order_c]


@pytest.fixture(scope="module")
def sample_orders_b(base_order_dict) -> list[ActiveOrder]:
    order_a = ActiveOrder(price=10, qty=15, **base_order_dict)
    order_b = ActiveOrder(price=10, qty=20, **base_order_dict)
    order_c = ActiveOrder(price=20, qty=25, **base_order_dict)
    return [order_a, order_b, order_c]


@pytest.fixture(scope="module")
def sample_orders_c(base_order_dict) -> list[ActiveOrder]:
    order_a = ActiveOrder(price=20, qty=15.6, **base_order_dict)
    order_b = ActiveOrder(price=20, qty=20.1, **base_order_dict)
    order_c = ActiveOrder(price=20, qty=25, **base_order_dict)
    return [order_a, order_b, order_c]


@pytest.mark.parametrize(
    "expected",
    [
        [
            MinifiedOrder(price=10, qty=15),
            MinifiedOrder(price=20, qty=20),
            MinifiedOrder(price=30, qty=25),
        ]
    ],
)
def test_merge_take_profits_with_same_price_a(expected, sample_orders_a):
    assert merge_take_profits_with_same_price(sample_orders_a) == expected


@pytest.mark.parametrize(
    "expected", [[MinifiedOrder(price=10, qty=35), MinifiedOrder(price=20, qty=25)]]
)
def test_merge_take_profits_with_same_price_b(expected, sample_orders_b):
    assert merge_take_profits_with_same_price(sample_orders_b) == expected


@pytest.mark.parametrize("expected", [[MinifiedOrder(price=20, qty=61)]])
def test_merge_take_profits_with_same_price_c(expected, sample_orders_c):
    assert merge_take_profits_with_same_price(sample_orders_c) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            dict(
                updated_at=datetime.now() - timedelta(hours=25),
                max_timedelta=timedelta(hours=24),
            ),
            True,
        ),
        (
            dict(
                updated_at=datetime.now() - timedelta(hours=20),
                max_timedelta=timedelta(hours=24),
            ),
            False,
        ),
        (
            dict(
                updated_at=datetime.now() - timedelta(hours=5),
                max_timedelta=timedelta(hours=4),
            ),
            True,
        ),
    ],
)
def test_is_order_expired(test_input, expected, base_order_dict):
    base_order_dict["updated_at"] = test_input.get("updated_at")
    current_order = BaseOrder(price=10, qty=10, **base_order_dict)
    max_timedelta = test_input.get("max_timedelta")
    assert is_order_expired(current_order, max_timedelta) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            dict(
                current_order_price=100,
                latest_info_price=90,
                max_rel_diff=0.05,
                max_abs_diff=None,
            ),
            True,
        ),
        (
            dict(
                current_order_price=100,
                latest_info_price=90,
                max_rel_diff=0.15,
                max_abs_diff=None,
            ),
            False,
        ),
        (
            dict(
                current_order_price=92.5,
                latest_info_price=90,
                max_rel_diff=None,
                max_abs_diff=2.5,
            ),
            False,
        ),
        (
            dict(
                current_order_price=92.6,
                latest_info_price=90,
                max_rel_diff=None,
                max_abs_diff=2.5,
            ),
            True,
        ),
        (
            dict(
                current_order_price=100,
                latest_info_price=90,
                max_rel_diff=0.10,
                max_abs_diff=12.5,
            ),
            True,
        ),
        (
            dict(
                current_order_price=100,
                latest_info_price=90,
                max_rel_diff=0.20,
                max_abs_diff=2.5,
            ),
            True,
        ),
    ],
)
def test_is_order_price_too_far_off_last_price(
    test_input, expected, base_order_dict, latest_info_dict
):
    base_order_dict["price"] = test_input.get("current_order_price")
    latest_info_dict["last_price"] = test_input.get("latest_info_price")
    assert (
        is_order_price_too_far_off_last_price(
            current_order=BaseOrder(**base_order_dict, qty=10),
            latest_info=LatestInfo(**latest_info_dict),
            max_rel_diff=test_input.get("max_rel_diff"),
            max_abs_diff=test_input.get("max_abs_diff"),
        )
        == expected
    )


@pytest.mark.xfail(raises=KeyError)
def test_is_order_price_too_far_off_last_price_error(base_order_dict, latest_info_dict):
    base_order_dict["price"] = 100
    latest_info_dict["last_price"] = 100
    is_order_price_too_far_off_last_price(
        current_order=BaseOrder(**base_order_dict, qty=10),
        latest_info=LatestInfo(**latest_info_dict),
        max_rel_diff=None,
        max_abs_diff=None,
    )
