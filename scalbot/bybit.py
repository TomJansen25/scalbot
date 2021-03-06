import hashlib
import hmac
from datetime import datetime, timedelta
from os import getenv
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
import requests
from loguru import logger
from pybit.inverse_perpetual import HTTP, WebSocket

from scalbot.bigquery import BigQuery
from scalbot.enums import Broker, Symbol
from scalbot.models import (
    ActiveOrder,
    BybitResponse,
    ConditionalOrder,
    LatestInfo,
    OpenPosition,
    Trade,
)


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
                msg = f"API Key not provided and not set as environment variable {api_key_env_var}"
                logger.error(msg)
                raise KeyError(msg)

        self._api_key = api_key

        if not api_secret:
            api_secret_env_var = f"BYBIT_{net.upper()}_API_SECRET"
            api_secret = getenv(api_secret_env_var)
            if not api_secret:
                msg = (
                    f"API Secret not provided and not set as environment "
                    f"variable {api_secret_env_var}"
                )
                logger.error(msg)
                raise KeyError(msg)

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

    def _generate_request_params(
        self, params: Dict[str, Union[str, int, float]]
    ) -> dict:
        params["api_key"] = self._api_key
        params["timestamp"] = self.get_server_time()
        params = dict(sorted(params.items()))

        params["sign"] = self._generate_signature(params)

        return params

    @staticmethod
    def _send_request(method: str, url: str, params: dict) -> BybitResponse:

        if method == "GET":
            response = requests.get(url, params)

        elif method == "POST":
            response = requests.post(url, params)

        else:
            msg = f"Incorrect Request method chosen ({method}). Either choose 'GET' or 'POST'"

            raise ValueError(msg)

        if not response.ok:
            raise KeyError(
                f"Something didn't go quite right with the Request, "
                f"a {response.status_code} code was returned :("
            )
        result = response.json()
        response.close()

        bybit_response = BybitResponse.parse_obj(result)
        if not bybit_response:
            raise KeyError(f"The result of the response is empty... :(")

        return bybit_response

    def get_wallet_balance(self, symbol: Optional[Symbol] = None) -> dict:
        """

        :param symbol:
        :return:
        """
        params: Dict[str, Union[str, int, float]] = dict()
        if symbol:
            params["symbol"] = symbol.value
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "GET", self.url + "v2/private/wallet/balance", params
        )

        return response.result

    def get_position(self, symbol: Optional[Symbol] = None) -> dict:
        """

        :param symbol:
        :return:
        """
        params: Dict[str, Union[str, int, float]] = dict()
        if symbol:
            params["symbol"] = symbol.value
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "GET", self.url + "v2/private/position/list", params
        )

        return response.result

    def get_active_orders(
        self,
        symbol: Optional[Symbol] = None,
        order_status: Optional[str] = None,
        order_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ActiveOrder]:

        params: Dict[str, Union[str, int, float]] = dict()
        if symbol:
            logger.info(
                f"Retrieving active orders for {symbol.value} with order status {order_status}..."
            )
            params["symbol"] = symbol.value
        if order_status:
            params["order_status"] = order_status
        if limit:
            params["limit"] = limit
        params = self._generate_request_params(params=params)

        response = self._send_request("GET", self.url + "v2/private/order/list", params)
        order_list: list = response.result.get("data")
        orders = [ActiveOrder.parse_obj(order) for order in order_list]

        if order_type:
            orders = [order for order in orders if order.order_type == order_type]

        logger.info(f"Successfully retrieved {len(orders)} {order_status} orders!")

        return orders

    def query_active_order(
        self,
        symbol: Symbol,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> ActiveOrder:
        params: Dict[str, Union[str, int, float]] = dict(symbol=symbol.value)
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id
        params = self._generate_request_params(params=params)

        response = self._send_request("GET", self.url + "/v2/private/order", params)
        active_order = ActiveOrder.parse_obj(response.result)
        return active_order

    def place_active_order(
        self,
        side: str,
        symbol: Symbol,
        order_type: str,
        qty: int,
        price: float,
        time_in_force: str = "GoodTillCancel",
        close_on_trigger: bool = False,
        take_profit: float = None,
        stop_loss: float = None,
        order_link_id: str = None,
    ) -> dict:

        params: Dict[str, Union[str, int, float]] = {
            "side": side,
            "symbol": symbol.value,
            "order_type": order_type,
            "qty": qty,
            "price": price,
            "time_in_force": time_in_force,
            "close_on_trigger": close_on_trigger,
        }

        if take_profit:
            params["take_profit"] = take_profit
        if stop_loss:
            params["stop_loss"] = stop_loss

        if not order_link_id:
            params["order_link_id"] = str(uuid4())

        params = self._generate_request_params(params=params)
        response = self._send_request(
            "POST", self.url + "v2/private/order/create", params
        )
        result = response.result

        logger.info(
            f"Order with Order ID {result.get('order_id')} and Order Link "
            f"ID {result.get('order_link_id')} successfully created!"
        )

        return result

    def cancel_active_order(
        self,
        symbol: Symbol,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> dict:

        if not order_id and not order_link_id:
            raise ValueError(
                "Either order_id or order_link_id is required, "
                "and not both can be left blank"
            )

        params: Dict[str, Union[str, int, float]] = dict(symbol=symbol.value)
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id

        params = self._generate_request_params(params=params)

        response = self._send_request(
            "POST", self.url + "v2/private/order/cancel", params
        )

        return response.result

    def get_conditional_orders(
        self, symbol: Optional[Symbol] = None, order_status: Optional[str] = None
    ) -> list[ConditionalOrder]:

        params: Dict[str, Union[str, int, float]] = dict()
        if symbol:
            params["symbol"] = symbol.value
        if order_status:
            params["stop_order_status"] = order_status
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "GET", self.url + "/v2/private/stop-order/list", params
        )
        order_list: list = response.result.get("data")
        orders = [ConditionalOrder.parse_obj(order) for order in order_list]

        return orders

    def query_conditional_order(
        self,
        symbol: Symbol,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> ConditionalOrder:
        params: Dict[str, Union[str, int, float]] = dict(symbol=symbol.value)
        if order_id:
            params["order_id"] = order_id
        if order_link_id:
            params["order_link_id"] = order_link_id
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "GET", self.url + "/v2/private/stop-order", params
        )
        conditional_order = ConditionalOrder.parse_obj(response.result)
        return conditional_order

    def cancel_conditional_order(
        self,
        symbol: Symbol,
        stop_order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> dict:

        if not stop_order_id and not order_link_id:
            raise ValueError(
                "Either order_id or order_link_id is required, "
                "and not both can be left blank"
            )

        params: Dict[str, Union[str, int, float]] = dict(symbol=symbol.value)
        if stop_order_id:
            params["stop_order_id"] = stop_order_id
        if order_link_id:
            params["order_link_id"] = order_link_id

        params = self._generate_request_params(params=params)

        response = self._send_request(
            "POST", self.url + "/v2/private/stop-order/cancel", params
        )

        return response.result

    def get_tp_sl_mode(self, symbol: Symbol) -> str:
        """
        Get currently active TP/SL Mode for a symbol, which is either Full or Partial
        :param symbol:
        :return: "Full" or "Partial"
        """
        position = self.get_position(symbol=symbol)
        tp_sl_mode: str = str(position.get("tp_sl_mode"))
        return tp_sl_mode

    def switch_tp_sl_mode(self, symbol: Symbol, tp_sl_mode: str) -> dict:
        if tp_sl_mode not in ["Full", "Partial"]:
            raise ValueError(
                f"Incorrect TP/SL Mode provided ({tp_sl_mode}). "
                f'Either choose "Full" or "Partial".'
            )
        current_mode = self.get_tp_sl_mode(symbol=symbol)

        if current_mode != tp_sl_mode:
            params: Dict[str, Union[str, int, float]] = dict(
                symbol=symbol.value, tp_sl_mode=tp_sl_mode
            )
            params = self._generate_request_params(params=params)

            response = self._send_request(
                "POST", self.url + "/v2/private/tpsl/switch-mode", params=params
            )
        else:
            raise Warning(
                f"Requested TP/SL Mode for {symbol} was already set, nothing was therefore changed"
            )

        return response.result

    def get_open_position(self, symbol: Symbol) -> OpenPosition:
        """
        Retrieve current open position and set Take Profits and Stop Losses on the position
        :param symbol:
        :return:
        """
        current_position = self.get_position(symbol=symbol)
        position_size = current_position.get("size")

        if position_size == 0:
            logger.info(f"No current open position for {symbol.value}")
            open_pos = OpenPosition(size=0, open_tp=0, open_sl=0)
        else:
            logger.info(
                f"There currently is a position of {position_size} open for {symbol.value}"
            )

            tps, sls = self.get_untriggered_take_profits_and_stop_losses(
                symbol=symbol, order_type="Limit"
            )
            stop_loss_qty = sum([float(sl.qty) for sl in sls])
            take_profit_qty = sum([float(tp.qty) for tp in tps])

            logger.info(f"Untriggered Stop Loss Quantity: {stop_loss_qty}")
            logger.info(f"Untriggered Take Profit Quantity: {take_profit_qty}")

            open_pos = OpenPosition(
                size=position_size,
                open_tp=take_profit_qty,
                open_sl=stop_loss_qty,
            )

        return open_pos

    def get_untriggered_take_profits_and_stop_losses(
        self, symbol: Symbol, order_type: str = "Market"
    ) -> Tuple[list[ConditionalOrder], list[ConditionalOrder]]:
        """

        :param symbol:
        :param order_type
        :return:
        """
        if order_type == "Market" or order_type == "Limit":
            orders = self.get_conditional_orders(
                symbol=symbol, order_status="Untriggered"
            )
            tps = [
                o
                for o in orders
                if o.stop_order_type in ["PartialTakeProfit", "TakeProfit"]
            ]
            sls = [
                o
                for o in orders
                if o.stop_order_type in ["PartialStopLoss", "StopLoss"]
            ]
        else:
            raise ValueError(
                f"Incorrect order type ({order_type}) provided. "
                f'Either choose "Limit" or "Market".'
            )

        if order_type == "Limit":
            tps = self.get_active_orders(
                symbol=symbol, order_status="New", order_type="Limit"
            )

        return tps, sls

    def add_take_profit_to_position(
        self, symbol: Symbol, take_profit: float, size: float
    ) -> dict:
        params: Dict[str, Union[str, int, float]] = dict(
            symbol=symbol.value, take_profit=take_profit, tp_size=size
        )
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "POST", self.url + "/v2/private/position/trading-stop", params=params
        )
        logger.info(
            f"Added a Take Profit with price {take_profit} of size {size} to {symbol.value}"
        )
        return response.result

    def add_stop_loss_to_position(
        self, symbol: Symbol, stop_loss: float, size: float
    ) -> dict:
        params: Dict[str, Union[str, float]] = dict(
            symbol=symbol.value, stop_loss=stop_loss, sl_size=size
        )
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "POST", self.url + "/v2/private/position/trading-stop", params=params
        )
        logger.info(
            f"Added a Stop Loss with price {stop_loss} of size {size} to {symbol.value}"
        )
        return response.result

    def fill_position_with_defined_trade(
        self,
        symbol: Symbol,
        open_position: OpenPosition = None,
        trade: Trade = None,
        fill_stop_loss: bool = True,
        fill_take_profit: bool = True,
    ):
        """

        :param symbol:
        :param open_position:
        :param trade:
        :param fill_stop_loss:
        :param fill_take_profit:
        """
        if not open_position:
            open_position = self.get_open_position(symbol=symbol)

        if not trade:
            bigquery = BigQuery()
            trade = bigquery.get_last_trade(symbol=symbol, broker=self.broker)

        tps, sls = self.get_untriggered_take_profits_and_stop_losses(symbol=symbol)

        if fill_take_profit and open_position.open_tp < open_position.size:
            logger.info(
                f"There is still a quantity of {open_position.size - open_position.open_tp} from "
                f"the current open position that is not covered with Take Profits that will tried "
                f"to be filled..."
            )
            for level in ["take_profit_1", "take_profit_2", "take_profit_3"]:
                next_take_profit = next(
                    (
                        o
                        for o in tps
                        if int(float(o.stop_px)) == int(getattr(trade, level))
                    ),
                    None,
                )
                # If None is returned, it means this calculated TP Level is not on Bybit yet
                if not next_take_profit:
                    self.add_take_profit_to_position(
                        symbol=symbol,
                        take_profit=getattr(trade, level),
                        size=int(getattr(trade, f"{level}_share")),
                    )

        if fill_stop_loss and open_position.open_sl < open_position.size:
            logger.info(
                f"There is still a quantity of {open_position.size - open_position.open_sl} from "
                f"the current open position that is not covered with a Stop Loss that will tried "
                f"to be filled..."
            )
            next_stop_loss = next(
                (
                    s
                    for s in sls
                    if int(float(s.stop_px)) == int(getattr(trade, "stop_loss"))
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
        self, symbol: Symbol, interval: int, from_time: int = None
    ) -> pd.DataFrame:
        """

        :param symbol:
        :param interval:
        :param from_time:
        :return:
        """

        if not from_time:
            from_time = self.get_server_time(as_timestamp=False) - (interval * 7200)

        logger.info(
            f"Retrieving data for {symbol.value} with candle frequency = {interval} "
            f"since {datetime.utcfromtimestamp(from_time).strftime('%d.%m.%Y %H:%M:%S')}"
        )

        data = self.session.query_kline(
            symbol=symbol.value, interval=interval, from_time=from_time
        )
        df = pd.DataFrame(data.get("result"))

        df["start"] = pd.to_datetime(df.open_time, unit="s")
        df["end"] = df.start + timedelta(minutes=interval)
        df["symbol"] = symbol.value

        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)
        df.volume = df.volume.astype(int)
        df.turnover = df.turnover.astype(float)
        df = df.drop(df.tail(1).index)

        return df

    def get_closed_profit_and_loss(
        self,
        symbol: Symbol,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> dict:
        """

        :param symbol:
        :param start_time:
        :param end_time:
        :return:
        """
        params: Dict[str, Union[str, float, int]] = dict()
        params["symbol"] = symbol.value
        params["limit"] = 50
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        params = self._generate_request_params(params=params)

        response = self._send_request(
            "GET", self.url + "/v2/private/trade/closed-pnl/list", params
        )

        return response.result

    def get_latest_symbol_information(self, symbol: Symbol) -> LatestInfo:
        response = self.session.latest_information_for_symbol(symbol=symbol.value)
        # bybit_response = BybitResponse.parse_obj(response)
        latest_info = LatestInfo.parse_obj(response.get("result")[0])
        return latest_info

    def find_open_position_orders(
        self, symbol: Symbol
    ) -> Tuple[list[ActiveOrder], list[ActiveOrder]]:
        new_limit_orders = self.get_active_orders(
            symbol=symbol, order_status="New", order_type="Limit"
        )
        open_position_orders = [
            order for order in new_limit_orders if float(order.stop_loss) > 0
        ]
        return new_limit_orders, open_position_orders
