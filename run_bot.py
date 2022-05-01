"""
Run BTC and ETH Bots test script
"""

import argparse

from dotenv import load_dotenv

from scalbot.enums import Broker, Symbol
from scalbot.scalbot import Scalbot
from scalbot.trades import TradingStrategy
from scalbot.utils import setup_logging

setup_logging()
load_dotenv()

V_PATTERNS = [
    {"color": "green", "prev_1": "green", "prev_2": "green", "prev_3": "red"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Define configurations to fetch data")

    parser.add_argument(
        "-cf",
        "--candle_frequency",
        type=int,
        default=1,
        help="Frequency of Candles to be retrieved in minutes (e.g., 1, 3, 5 or 10).",
    )

    parser.add_argument(
        "-b",
        "--broker",
        type=str,
        default="Bybit",
        nargs="?",
        choices=[broker.value for broker in Broker],
        help="Brokers to retrieve data from",
    )

    parser.add_argument(
        "-s",
        "--symbols",
        type=str,
        default="BTCUSD",
        nargs="*",
        choices=[s.value for s in Symbol],
        help="Symbols to be retrieved",
    )

    parser.add_argument(
        "-ba",
        "--bet_amount",
        type=int,
        default=10,
        nargs="?",
        help="Amount of USD to use for every trade",
    )

    parser.add_argument(
        "-r",
        "--risk",
        type=float,
        default=0.01,
        nargs="?",
        help="Risk to be used for every trade",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args().__dict__

    trading_strategy = TradingStrategy(
        bet_amount=args.get("bet_amount"), risk=args.get("risk")
    )

    scalbot = Scalbot(
        patterns=V_PATTERNS,
        trading_strategy=trading_strategy,
        candle_frequency=args.get("candle_frequency"),
    )

    scalbot.run_bybit_bot(symbols=[Symbol.BTCUSD])
