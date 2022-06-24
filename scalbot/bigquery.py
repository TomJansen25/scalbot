import json
from abc import ABC
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from google.cloud.bigquery import Client
from loguru import logger
from pydantic import BaseModel

from scalbot.enums import Broker, Symbol
from scalbot.models import Trade


class BigQuery(ABC, BaseModel):
    client: Client

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__(client=Client())

    def get_today_trades(
        self,
        symbols: Optional[list[Symbol]] = None,
        brokers: Optional[list[Broker]] = None,
    ) -> pd.DataFrame:

        today = datetime.now().strftime("%Y-%m-%d")

        qry = f"""
        SELECT *
        FROM `scalbot.trades.active_trades`
        WHERE timestamp >= '{today}'
        ORDER BY timestamp asc;
        """

        df = self.client.query(qry).to_dataframe()
        if symbols:
            df = df.loc[df.symbol.isin([symbol.value for symbol in symbols])]
        if brokers:
            df = df.loc[df.broker.isin([broker.value for broker in brokers])]
        return df

    def get_last_trade(
        self, symbol: Union[Symbol, str], broker: Union[Broker, str]
    ) -> Trade:
        """

        :param symbol:
        :param broker:
        :return:
        """

        if isinstance(symbol, Symbol):
            symbol = symbol.value

        if isinstance(broker, Broker):
            broker = broker.value

        qry = f"""
        SELECT *
        FROM `scalbot.trades.active_trades`
        WHERE symbol = '{symbol}' AND broker = '{broker}'
        ORDER BY timestamp desc
        LIMIT 1
        """
        df = self.client.query(qry).to_dataframe()
        trade = Trade.parse_obj(df.squeeze().to_dict())
        return trade

    def insert_trade_to_bigquery(self, trade: Trade):
        """

        :param trade:
        """
        trade.pattern = json.dumps(trade.pattern)
        new_df = pd.DataFrame(trade.dict(), index=[0])
        new_df.order_id = new_df.order_id.astype(str)
        new_df.order_link_id = new_df.order_link_id.astype(str)

        job = self.client.load_table_from_dataframe(
            dataframe=new_df, destination="scalbot.trades.active_trades"
        )
        res = job.result()
        logger.info(f"Job {res.job_id} is {res.state}!")
