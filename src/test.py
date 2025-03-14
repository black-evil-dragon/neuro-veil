from datetime import timedelta
import pandas as pd
import logging
import sys
import os
from utils.numbers import parse_quotation
from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService
from services.MarketDataService import MarketDataService
from utils.config import SANDBOX_TOKEN
from utils.time import now

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(
    filename="./neuro.log", format="[%(asctime)s]  %(message)s", filemode="w"
)
logging.getLogger("neuro").addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger("neuro")
log.setLevel(logging.DEBUG)


def main():
    TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)
    instrumentsService = InstrumentsService(TService)
    marketDataService = MarketDataService(TService)

    # Получение данных о сделках
    data = marketDataService.get_last_trades(
        instrumentId="TCS80A107UL4", from_date=now() - timedelta(days=1), to_date=now()
    )
    trades = data["trades"]

    # Преобразование данных о сделках в DataFrame
    trade_data = []
    for trade in trades:
        trade_data.append({
            "time": pd.to_datetime(trade["time"]),
            "direction": trade["direction"],
            "price": parse_quotation(trade["price"]),
            "quantity": int(trade["quantity"])
        })

    df = pd.DataFrame(trade_data)
    df.set_index("time", inplace=True)

    # Группировка по 30 минутам
    half_hourly_trades = df.resample("30min")

    # Результирующий словарь для хранения данных за каждые 30 минут
    result_dict = {}

    for interval, group in half_hourly_trades:
        sell_trades = group[group["direction"] == "TRADE_DIRECTION_SELL"]
        buy_trades = group[group["direction"] == "TRADE_DIRECTION_BUY"]

        total_sell_price = (sell_trades["price"] * sell_trades["quantity"]).sum()
        total_sell_quantity = sell_trades["quantity"].sum()
        avg_sell_price = total_sell_price / total_sell_quantity if total_sell_quantity > 0 else 0

        total_buy_price = (buy_trades["price"] * buy_trades["quantity"]).sum()
        total_buy_quantity = buy_trades["quantity"].sum()
        avg_buy_price = total_buy_price / total_buy_quantity if total_buy_quantity > 0 else 0

        total_trade_price = (group["price"] * group["quantity"]).sum()
        total_trade_quantity = group["quantity"].sum()
        avg_trade_price = total_trade_price / total_trade_quantity if total_trade_quantity > 0 else 0

        result_dict[interval] = {
            "avg_sell_price": avg_sell_price,
            "avg_buy_price": avg_buy_price,
            "avg_trade_price": avg_trade_price,
            "total_sell_quantity": total_sell_quantity,
            "total_buy_quantity": total_buy_quantity,
            "total_trade_quantity": total_trade_quantity
        }

    # Вывод результата
    for interval, stats in result_dict.items():
        print(stats)


if __name__ == "__main__":
    main()