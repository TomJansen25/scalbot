import os

from dotenv import load_dotenv
from loguru import logger

from scalbot.bybit import Bybit
from scalbot.enums import Symbol
from scalbot.scalbot import Scalbot, cancel_invalid_expired_orders
from scalbot.trades import TradingStrategy
from scalbot.utils import setup_logging

load_dotenv()
setup_logging(serverless=True)

V_PATTERNS = [
    {
        "color": "green",
        "prev_color_1": "green",
        "prev_color_2": "green",
        "prev_color_3": "red",
    },
    {
        "color": "red",
        "prev_color_1": "red",
        "prev_color_2": "red",
        "prev_color_3": "green",
    },
]

SYMBOLS = [Symbol.BTCUSD]


def run_bybit_bot(event, context):

    logger.info(event)
    logger.info(
        f"FUNCTION CONTEXT: triggered by message with event id {context.event_id} "
        f"published at {context.timestamp} to {context.resource['name']}"
    )

    trading_strategy = TradingStrategy(bet_amount=100, risk=0.01)

    scalbot = Scalbot(
        patterns=V_PATTERNS,
        trading_strategy=trading_strategy,
        candle_frequency=3,
    )
    bybit = Bybit(
        net="test",
        api_key=os.environ.get("BYBIT_GCP_TEST_API_KEY"),
        api_secret=os.environ.get("BYBIT_GCP_TEST_API_SECRET"),
    )

    for symbol in SYMBOLS:
        scalbot.run_bybit_bot(bybit=bybit, symbol=symbol)


def cancel_invalid_or_expired_orders(event, context):
    logger.info(event)
    logger.info(
        f"FUNCTION CONTEXT: triggered by message with event id {context.event_id} "
        f"published at {context.timestamp} to {context.resource['name']}"
    )

    bybit = Bybit(
        net="test",
        api_key=os.environ.get("BYBIT_GCP_TEST_API_KEY"),
        api_secret=os.environ.get("BYBIT_GCP_TEST_API_SECRET"),
    )

    cancel_invalid_expired_orders(bybit=bybit, symbols=SYMBOLS)
