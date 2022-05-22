from abc import ABC
from typing import Union

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
        WHERE timestamp = (select max(timestamp) from `scalbot.trades.active_trades`)
            AND symbol = '{symbol}'
            AND broker = '{broker}'
        """
        df = self.client.query(qry).to_dataframe()
        trade = Trade.parse_obj(df.squeeze().to_dict())
        return trade

    def insert_trade_to_bigquery(self, trade: Trade):
        """

        :param trade:
        """
        new_df = pd.DataFrame(trade.dict(), index=[0])
        new_df.order_id = new_df.order_id.astype(str)
        new_df.order_link_id = new_df.order_link_id.astype(str)

        job = self.client.load_table_from_dataframe(
            dataframe=new_df, destination="scalbot.trades.active_trades"
        )
        res = job.result()
        logger.info(f"Job {res.job_id} is {res.state}!")
