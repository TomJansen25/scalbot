from datetime import datetime
from pathlib import Path

import typer

from scalbot.data import HistoricData
from scalbot.enums import Symbol

app = typer.Typer(name="scalbot-cli", no_args_is_help=True)


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


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


if __name__ == "__main__":
    app()
