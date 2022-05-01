import hashlib
import hmac
from datetime import datetime, timedelta
from os import getenv
from typing import Optional, Tuple, Union
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
import requests
from loguru import logger
from pybit.inverse_perpetual import HTTP, WebSocket

from scalbot.enums import Broker, Symbol
from scalbot.models import OpenPosition, Trade
from scalbot.utils import get_last_trade


class Bybit:
    """
    Bybit Class to connect to Bybit Account and API Endpoints
    """

    broker: Broker = Broker.BYBIT
    net: str = "test"
    domain: str = "bybit"
    url: str = ""
    session: HTTP = None
    ws_url: str = ""
    ws: WebSocket = None
    allowed_symbols: Union[list, None] = None

    def __init__(self, net: str = "test", api_key: str = None, api_secret: str = None):

        if net not in ["test", "main"]:
            raise KeyError(
                f"Provided value of net ({net}) does not exist, "
                f'please choose "test" or "main".'
            )

        self.net = net

        if net == "test":
            self.url = "https://api-testnet.bybit.com/"
            self.ws_url = "wss://stream-testnet.bybit.com/realtime"
        else:
            self.url = "https://api.bybit.com/"
            self.ws_url = "wss://stream.bybit.com/realtime"

        if not api_key:
            api_key_env_var = f"BYBIT_{net.upper()}_API_KEY"
            api_key = getenv(api_key_env_var)
            if not api_key:
                raise KeyError(
                    f"API Key not provided and not set as environment "
                    f"variable {api_key_env_var}"
                )

        self._api_key = api_key

        if not api_secret:
            api_secret_env_var = f"BYBIT_{net.upper()}_API_SECRET"
            api_secret = getenv(api_secret_env_var)
            if not api_secret:
                raise KeyError(
                    f"API Secret not provided and not set as environment "
                    f"variable {api_secret_env_var}"
                )

        self._api_secret = api_secret
        self.setup_http()

    def setup_http(self):
        """
        Set up HTTP connection
        """
        self.session = HTTP(
            endpoint=self.url,
            api_key=self._api_key,
            api_secret=self._api_secret,
        )

    def get_symbols(self) -> list:
        """
        Retrieve all symbols from Bybit that can be tracked and traded with
        :return:
        """
        if not self.allowed_symbols:
            result = self.session.query_symbol()
            symbols = [res.get("name") for res in result.get("result")]
            self.allowed_symbols = symbols
        return self.allowed_symbols

    def get_server_time(self, as_timestamp: bool = True) -> int:
        """

        :param as_timestamp:
        :return:
        """
        server_time: str = self.session.server_time().get("time_now")
        if as_timestamp:
            timestamp = int(float(server_time) * 1000)
        else:
            timestamp = int(float(server_time))
        return timestamp

    def _generate_signature(self, params: dict):
        sign = hmac.new(
            self._api_secret.encode("utf-8"),
            urlencode(params).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sign

    def _generate_request_params(self, params: dict) -> dict:
        params["api_key"] = self._api_key
        params["timestamp"] = self.get_server_time()
        params = dict(sorted(params.items()))

        params["sign"] = self._generate_signature(params)

        return params

    @staticmethod
    def _send_request(method: str, url: str, params: dict) -> dict:

        if method == "GET":
            response = requests.get(url, params)

        elif method == "POST":
            response = requests.post(url, params)

        else:
            raise ValueError(
                f"Incorrect Request method chosen ({method}). "
                f"Either choose 'GET' or 'POST'"
            )

        if not response.ok:
            raise KeyError(
                f"Something didn't go quite right with the Request, "
                f"a {response.status_code} code was returned :("
            )
        result = response.json()
        response.close()

        return result

    def get_wallet_balance(self, symbol: Optional[str] = None) -> dict:
        """

        :param symbol:
        :return:
        """
        params = dict()
        if symbol:
            params["symbol"] = symbol
        params = self._generate_request_params(params=params)

        response = requests.get(self.url + "v2/private/wallet/balance", params)

        return response.json()

    def get_position(self, symbol: Optional[str] = None) -> dict:
        """

        :param symbol:
        :return:
        """
        params = dict()
        if symbol:
            params["symbol"] = symbol
        params = self._generate_request_params(params=params)

        response = requests.get(self.url + "v2/private/position/list", params)

        return response.json()

    def get_active_orders(
        self, symbol: Optional[str] = None, order_status: Optional[str] = None
    ) -> dict:

        logger.info(
            f"Retrieving active orders for {symbol} with order status {order_status}..."
        )
        params = dict()
        if symbol:
            params["symbol"] = symbol
        if order_status:
            params["order_status"] = order_status
        params = self._generate_request_params(params=params)

        response = self._send_request("GET", self.url + "v2/private/order/list", params)
        logger.info(
            f"Successfully retrieved {len(response.get('result').get('data'))} "
            f"{order_status} orders!"
        )

        return response

    def query_active_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> dict:
        params = dict(symbol=symbol)
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id
        params = self._generate_request_params(params=params)

        response = requests.get(self.url + "/v2/private/order", params)
        return response.json()

    def place_active_order(
        self,
        side: str,
        symbol: str,
        order_type: str,
        qty: int,
        price: float,
        time_in_force: str = "GoodTillCancel",
        take_profit: float = None,
        stop_loss: float = None,
        order_link_id: str = None,
    ) -> dict:

        params = {
            "side": side,
            "symbol": symbol,
            "order_type": order_type,
            "qty": qty,
            "price": price,
            "time_in_force": time_in_force,
        }

        if take_profit:
            params["take_profit"] = take_profit
        if stop_loss:
            params["stop_loss"] = stop_loss

        if not order_link_id:
            params["order_link_id"] = str(uuid4())

        params = self._generate_request_params(params=params)
        response = requests.post(self.url + "v2/private/order/create", params)
        response = response.json()

        logger.info(
            f"Order with Order ID {response.get('result').get('order_id')} and Order Link "
            f"ID {response.get('result').get('order_link_id')} successfully created!"
        )

        return response

    def cancel_active_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ):

        if not order_id and not order_link_id:
            raise ValueError(
                "Either order_id or order_link_id is required, "
                "and not both can be left blank"
            )

        params = {"symbol": symbol}
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id

        params = self._generate_request_params(params=params)

        response = requests.post(self.url + "v2/private/order/cancel", params)

        return response.json()

    def get_conditional_orders(
        self, symbol: Optional[str] = None, order_status: Optional[str] = None
    ):

        params = dict()
        if symbol:
            params["symbol"] = symbol
        if order_status:
            params["stop_order_status"] = order_status
        params = self._generate_request_params(params=params)

        response = requests.get(self.url + "/v2/private/stop-order/list", params)

        return response.json()

    def query_conditional_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ):
        params = dict(symbol=symbol)
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id
        params = self._generate_request_params(params=params)

        response = requests.get(self.url + "/v2/private/stop-order", params)
        return response.json()

    def get_tp_sl_mode(self, symbol: str):
        position = self.get_position(symbol=symbol)
        tp_sl_mode = position.get("result").get("tp_sl_mode")
        return tp_sl_mode

    def switch_tp_sl_mode(self, symbol: str, tp_sl_mode: str) -> dict:
        if tp_sl_mode not in ["Full", "Partial"]:
            raise ValueError(
                f"Incorrect TP/SL Mode provided ({tp_sl_mode}). "
                f'Either choose "Full" or "Partial".'
            )
        current_mode = self.get_tp_sl_mode(symbol=symbol)
        if current_mode != tp_sl_mode:
            params = dict(symbol=symbol, tp_sl_mode=tp_sl_mode)
            params = self._generate_request_params(params=params)

            response = requests.post(
                self.url + "/v2/private/tpsl/switch-mode", params=params
            )
            return response.json()
        else:
            raise Warning(
                f"Requested TP/SL Mode for {symbol} was already set, "
                f"nothing was therefore changed"
            )

    def get_open_position(self, symbol: str) -> OpenPosition:

        current_position = self.get_position(symbol=symbol)
        position_size = current_position.get("result").get("size")

        if position_size == 0:
            logger.info(f"No current open position for {symbol}")
            open_pos = OpenPosition(size=0, open_tp=0, open_sl=0)
            return open_pos
        else:
            logger.info(
                f"There currently is a position of {position_size} open for {symbol}"
            )

            tps, sls = self.get_untriggered_take_profits_and_stop_losses(symbol)
            stop_loss_qty = sum([float(sl.get("qty")) for sl in sls])
            take_profit_qty = sum([float(tp.get("qty")) for tp in tps])

            logger.info(f"Untriggered Stop Loss Quantity: {stop_loss_qty}")
            logger.info(f"Untriggered Take Profit Quantity: {take_profit_qty}")

            open_pos = OpenPosition(
                size=position_size,
                open_tp=position_size - take_profit_qty,
                open_sl=position_size - stop_loss_qty,
            )
            return open_pos

    def get_untriggered_take_profits_and_stop_losses(
        self, symbol: str
    ) -> Tuple[list, list]:
        """

        :param symbol:
        :return:
        """
        orders = (
            self.get_conditional_orders(symbol=symbol, order_status="Untriggered")
            .get("result")
            .get("data")
        )
        tps = [
            o
            for o in orders
            if o.get("stop_order_type") in ["PartialTakeProfit", "TakeProfit"]
        ]
        sls = [
            o
            for o in orders
            if o.get("stop_order_type") in ["PartialStopLoss", "StopLoss"]
        ]
        return tps, sls

    def add_take_profit_to_position(self, symbol: str, take_profit: float, size: float):
        params = dict(symbol=symbol, take_profit=take_profit, tp_size=size)
        params = self._generate_request_params(params=params)

        response = requests.post(
            self.url + "/v2/private/position/trading-stop", params=params
        )
        logger.info(
            f"Added a Take Profit with price {take_profit} of size {size} to {symbol}"
        )
        return response.json()

    def add_stop_loss_to_position(self, symbol: str, stop_loss: float, size: float):
        params = dict(symbol=symbol, stop_loss=stop_loss, sl_size=size)
        params = self._generate_request_params(params=params)

        response = requests.post(
            self.url + "/v2/private/position/trading-stop", params=params
        )
        logger.info(
            f"Added a Stop Loss with price {stop_loss} of size {size} to {symbol}"
        )
        return response.json()

    def fill_position_with_defined_trade(
        self, symbol: str, open_position: OpenPosition = None, trade: Trade = None
    ):

        if not open_position:
            open_position = self.get_open_position(symbol=symbol)

        if not trade:
            trade = get_last_trade(symbol=symbol, broker=self.broker.value)

        tps, sls = self.get_untriggered_take_profits_and_stop_losses(symbol=symbol)

        if open_position.open_tp > 0:
            logger.info(
                f"There is still an open Take Profit position of {open_position.open_tp} "
                f"that will tried to be filled..."
            )
            for level in ["take_profit_1", "take_profit_2", "take_profit_3"]:
                next_take_profit = next(
                    (
                        o
                        for o in tps
                        if int(float(o.get("stop_px"))) == int(getattr(trade, level))
                    ),
                    None,
                )
                # If None is returned, it means this calculated TP Level is not on Bybit yet
                if not next_take_profit:
                    self.add_take_profit_to_position(
                        symbol=symbol,
                        take_profit=getattr(trade, level),
                        size=int(open_position.size * getattr(trade, f"{level}_share")),
                    )

        if open_position.open_sl > 0:
            logger.info(
                f"There is still an open Stop Loss position of {open_position.open_sl} "
                f"that will tried to be filled..."
            )
            next_stop_loss = next(
                (
                    s
                    for s in sls
                    if int(float(s.get("stop_px"))) == int(getattr(trade, "stop_loss"))
                ),
                None,
            )
            # If None is returned, it means this calculated SL Level is not on Bybit yet
            if not next_stop_loss:
                self.add_stop_loss_to_position(
                    symbol=symbol,
                    stop_loss=getattr(trade, "stop_loss"),
                    size=open_position.open_sl,
                )

    def get_latest_symbol_data_as_df(
        self, symbol: str, interval: int, from_time: int = None
    ):
        """

        :param symbol:
        :param interval:
        :param from_time:
        :return:
        """

        if not from_time:
            from_time = self.get_server_time(as_timestamp=False) - (interval * 7200)

        logger.info(
            f"Retrieving data for {symbol} with candle frequency = {interval} "
            f"since {datetime.utcfromtimestamp(from_time).strftime('%d.%m.%Y %H:%M:%S')}"
        )

        data = self.session.query_kline(
            symbol=symbol, interval=interval, from_time=from_time
        )
        df = pd.DataFrame(data.get("result"))

        df["start"] = pd.to_datetime(df.open_time, unit="s")
        df["end"] = df.start + timedelta(minutes=interval)
        df["symbol"] = symbol

        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)
        df.volume = df.volume.astype(int)
        df.turnover = df.turnover.astype(float)
        df = df.drop(df.tail(1).index)

        return df

    def setup_websocket(self):
        """
        Setup Websocket connection
        """
        self.ws = WebSocket(
            test=self.net == "test",
            domain=self.domain,
            api_key=self._api_key,
            api_secret=self._api_secret,
        )

    @staticmethod
    def generate_subscriptions(
        candle_frequency: int, symbols: Union[Symbol, list[Symbol]]
    ) -> list[str]:
        """
        Generates Bybit Websocket Subscriptions based on Candle Frequency and Symbol Names
        :param candle_frequency:
        :param symbols:
        :return:
        """
        if not isinstance(symbols, list):
            coins = [symbols]

        subscriptions = []
        for symbol in symbols:

            if isinstance(symbol, str):
                subscriptions.append(f"klineV2.{candle_frequency}.{symbol}")
            elif isinstance(symbol, Symbol):
                subscriptions.append(f"klineV2.{candle_frequency}.{symbol.value}")
            else:
                raise KeyError(f"Provided Symbol has incorrect type: {type(symbol)}")

        return subscriptions
