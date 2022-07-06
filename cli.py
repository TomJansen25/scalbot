import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import typer
from dotenv import load_dotenv

from main import cancel_invalid_or_expired_orders, run_bybit_bot, send_daily_summary
from scalbot.bybit import Bybit
from scalbot.data import HistoricData
from scalbot.enums import Broker, Symbol
from scalbot.mail import Email
from scalbot.scalbot import Scalbot, cancel_invalid_expired_orders
from scalbot.trades import TradeSummary, TradingStrategy
from scalbot.utils import get_params, setup_logging

load_dotenv()
setup_logging()

EMAIL = "tomjansen25@gmail.com"

V_PATTERNS = [
    {
        "color": "green",
        "prev_color_1": "green",
        "prev_color_2": "green",
        "prev_color_3": "red",
    },
    {
        "color": "green",
        "prev_color_1": "red",
        "prev_color_2": "green",
        "prev_color_3": "green",
        "prev_color_4": "red",
    },
    {
        "color": "green",
        "prev_color_1": "green",
        "prev_color_2": "red",
        "prev_color_3": "green",
        "prev_color_4": "red",
    },
    {
        "color": "red",
        "prev_color_1": "red",
        "prev_color_2": "red",
        "prev_color_3": "green",
    },
    {
        "color": "red",
        "prev_color_1": "green",
        "prev_color_2": "red",
        "prev_color_3": "red",
        "prev_color_4": "green",
    },
    {
        "color": "red",
        "prev_color_1": "red",
        "prev_color_2": "green",
        "prev_color_3": "red",
        "prev_color_4": "green",
    },
]

app = typer.Typer(name="scalbot-cli", no_args_is_help=True)


def generate_event_context() -> Tuple[dict, dict]:
    return dict(), dict()


@app.command()
def get_save_historic_data(symbol: Symbol):

    typer.echo(f"Retrieving and saving historical data for {symbol.value}")
    data = HistoricData(symbol=symbol)

    if not data.is_symbol_data_to_download():
        typer.echo(f"Symbol {symbol} can not be downloaded...")
        raise typer.Exit()

    df = data.retrieve_historic_data()

    datestamp = datetime.now().strftime("%Y%m%d")
    save_path = f"data/{symbol.value}_1min_historical_data_{datestamp}.csv.gz"
    df.to_csv(save_path, compression="gzip")
    typer.echo(f"Data retrieved and saved to {save_path}")


@app.command("run-bybit-bot")
def run_bybit_bot_from_cli(
    symbol: Symbol,
    bet_amount: float = 100,
    risk: float = 0.01,
    candle_frequency: int = 1,
):
    trading_strategy = TradingStrategy(bet_amount=bet_amount, risk=risk)

    scalbot = Scalbot(
        patterns=V_PATTERNS,
        trading_strategy=trading_strategy,
        candle_frequency=candle_frequency,
    )

    typer.echo(
        f"Setting up bybit bot with Candle Frequency = {scalbot.candle_frequency} and "
        f"for the following Symbol: {symbol.value}..."
    )

    bybit = Bybit()

    while True:
        scalbot.run_bybit_bot(bybit=bybit, symbol=symbol)
        time.sleep(60)


@app.command("run-bybit-bot-from-params")
def run_bybit_bot_from_params_from_cli(symbols: list[Symbol]):
    params = get_params()
    while True:
        for symbol in symbols:
            symbol_params = params[symbol.value]

            trading_strategy = TradingStrategy(**symbol_params.get("trading_strategy"))

            scalbot = Scalbot(
                trading_strategy=trading_strategy, **symbol_params.get("scalbot")
            )

            typer.echo(
                f"Setting up bybit bot with Candle Frequency = {scalbot.candle_frequency} and "
                f"for the following Symbol: {symbol.value}..."
            )

            bybit = Bybit()
            scalbot.run_bybit_bot(bybit=bybit, symbol=symbol)
        time.sleep(60)


@app.command()
def test_cancel_invalid_expired_orders(symbols: list[Symbol]):
    bybit = Bybit()
    cancel_invalid_expired_orders(bybit=bybit, symbols=symbols)


@app.command()
def test_send_daily_summary(symbol: list[Symbol], broker: Broker = Broker.BYBIT.value):
    bybit = Bybit()

    trade_summarizer = TradeSummary(broker=broker, symbols=symbol, bybit=bybit)
    trade_summary_df = trade_summarizer.get_trade_summary_as_df()

    template = "daily_trade_summary"
    emailer = Email(email_sender=EMAIL, email_receiver=EMAIL)
    emailer.set_message_template(template=template)
    variables = emailer.get_template_variables_from_df(
        template=template, df=trade_summary_df
    )
    emailer.fill_message_template(variable_substitutes=variables)
    emailer.prepare_message(
        subject=f"Scalbot Summary {datetime.now().strftime('%d-%m-%Y')}",
        message_type="html",
    )
    emailer.send_email()


if __name__ == "__main__":
    app()
