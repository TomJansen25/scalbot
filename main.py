import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from scalbot.bybit import Bybit
from scalbot.enums import Broker, Symbol
from scalbot.mail import Email
from scalbot.scalbot import Scalbot, cancel_invalid_expired_orders
from scalbot.trades import TradeSummary, TradingStrategy
from scalbot.utils import get_params, setup_logging

load_dotenv()
setup_logging(serverless=True)

SYMBOLS = [Symbol.BTCUSD, Symbol.DOTUSD]

BYBIT = Bybit(
    net="test",
    api_key=os.environ.get("BYBIT_GCP_TEST_API_KEY"),
    api_secret=os.environ.get("BYBIT_GCP_TEST_API_SECRET"),
)


def run_bybit_bot(event: dict, context):
    logger.info(event)
    logger.info(
        f"FUNCTION CONTEXT: triggered by message with event id {context.event_id} "
        f"published at {context.timestamp} to {context.resource['name']}"
    )

    params = get_params()
    for symbol in SYMBOLS:
        symbol_params = params[symbol.value]

        trading_strategy = TradingStrategy(**symbol_params.get("trading_strategy"))

        scalbot = Scalbot(
            trading_strategy=trading_strategy, **symbol_params.get("scalbot")
        )

        scalbot.run_bybit_bot(bybit=BYBIT, symbol=symbol)


def cancel_invalid_or_expired_orders(event, context):
    logger.info(event)
    logger.info(
        f"FUNCTION CONTEXT: triggered by message with event id {context.event_id} "
        f"published at {context.timestamp} to {context.resource['name']}"
    )

    cancel_invalid_expired_orders(bybit=BYBIT, symbols=SYMBOLS)


def send_daily_summary(event, context):
    logger.info(event)
    logger.info(
        f"FUNCTION CONTEXT: triggered by message with event id {context.event_id} "
        f"published at {context.timestamp} to {context.resource['name']}"
    )
    trade_summarizer = TradeSummary(broker=Broker.BYBIT, symbols=SYMBOLS, bybit=BYBIT)
    trade_summary_df = trade_summarizer.get_trade_summary_as_df()

    variables = dict(
        num_trades=len(trade_summary_df),
        total_pnl=trade_summary_df["Closed PnL"].sum(),
        table=trade_summary_df.to_html(
            index=False, classes="summary-table", justify="center", decimal=","
        ),
    )

    email = "tomjansen25@gmail.com"
    emailer = Email(email_sender=email, email_receiver=email)
    emailer.set_message_template(template="daily_trade_summary")
    emailer.fill_message_template(variable_substitutes=variables)
    emailer.prepare_message(
        subject=f"Scalbot Summary {datetime.now().strftime('%d-%m-%Y')}",
        message_type="html",
    )
    emailer.send_email()


def list_files(request):
    import os
    import pathlib
    from os import path

    root = path.dirname(path.abspath(__file__))
    print(root)
    cwd = pathlib.Path().cwd()

    print(cwd, cwd.absolute())
    params = cwd.joinpath("config", "params.yaml")
    print(params, params.exists())

    children = os.listdir(root)
    print(children)
    files = [c for c in children if path.isfile(path.join(root, c))]
    print(files)
    return "Files: {}".format(files)
