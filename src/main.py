from datetime import datetime, timedelta, timezone

import numpy as np
import pytz

from bot.test import TradingBot
from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService

from data.fetcher import DataFetcher

from models import Model

from utils.config import LOOK_BACK, SANDBOX_TOKEN
from utils.time import now

import logging
import sys


def setup_logger():
    """
    Настройка логгера с кодировкой UTF-8.
    """
    # Создаем логгер
    logger = logging.getLogger("neuro")
    logger.setLevel(logging.DEBUG)

    # Формат логов
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Логирование в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Логирование в файл с кодировкой UTF-8
    log_filename = f"./output/logs/neuro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(
        log_filename, encoding="utf-8"
    )  # Указываем кодировку UTF-8
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


log = setup_logger()


def main():
    log.info("Запуск программы")

    features = [
        "close",

        "open",
        "high",
        "low",
        "volume",
        # "delta"
    ]

    indicators = [
        # "RSI_CLOSE_7"
        "RSI_CLOSE_14",
    
        # "BB_CLOSE_20",

        "SMA_CLOSE_5",
        "SMA_CLOSE_14",
        "SMA_CLOSE_50",
        "SMA_CLOSE_100",

        "EMA_CLOSE_5",
        "EMA_CLOSE_50",


        # "MACD_CLOSE_12_26_9",
    ]


    additive_features = [
        # "IMOEXF_OPEN",
        # "IMOEXF_VOLUME",


        "IMOEXF_CLOSE",
        "USD000UTSTOM_CLOSE",
    ]


    features += indicators + additive_features
    

    TModel = Model()

    TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)
    fetcher = DataFetcher(TService, processor=TModel.processor)
    instrumentsService = InstrumentsService(TService)


    # Сбор данных по инструментам

    # - Tinkoff
    response = instrumentsService.find_instrument("RU000A107UL4")
    tbank_instrument = response.get("instruments")[0]

    print(tbank_instrument)

    log.info(f"Инструмент найден: {tbank_instrument['ticker']}")

    # - Dop: MOEX 
    response = instrumentsService.find_instrument("IMOEXF Индекс МосБиржи")
    moex = response.get("instruments")[0]
    log.info(f"Инструмент найден: {moex['ticker']}")


    # - Dop: Доллар
    response = instrumentsService.find_instrument('BBG0013HGFT4')
    dollar = response.get("instruments")[0]
    log.info(f"Инструмент найден: {dollar['ticker']}")


    data = fetcher.get_data(
        instrument=tbank_instrument,
        from_date=now() - timedelta(days=1500),
        to_date=now(),
        additive_instruments=[
            moex,
            dollar
        ],
        step=50
    )

    log.info(f"Получено {len(data)} записей для обучения")
    TModel.processor.save_to_json(data, "./output/data/tbank-full-test.json")
    exit()

    data = TModel.processor.load_from_json('./output/data/tbank-full-test.json')

    # Получение последней свечи
    last_candle = data[-1]
    last_candle_time = last_candle["time"]

    # Определение даты последнего обновления
    last_update_date = datetime.fromisoformat(last_candle_time).replace(tzinfo=pytz.timezone('Europe/Moscow')).astimezone(pytz.UTC)


    # new_data = fetcher.get_data(
    #     instrument=tbank_instrument,
    #     from_date=last_update_date,
    #     to_date=now(),
    #     additive_instruments=[
    #         moex,
    #         dollar
    #     ]
    # )

    # # # Объединение данных
    # data.extend(new_data)

    # # Сохранение обновленных данных
    TModel.processor.save_to_json(data, "./output/data/tbank-full-test.json")
    exit()
    # TModel.train(data=data[:-250], features=features)


    # TModel.save("tbank", test=True)

    TModel.load(name='tbank', extension='keras', test=True)

    # TModel.predict(
    #     data=data[-500:],
    #     features=features,

    # )


    # Инициализация и запуск бота
    bot = TradingBot(TModel)
    trades, final_balance = bot.run(data=data[:], features=features)

    # Вывод результатов
    print(f"Final Balance: {final_balance}")
    for trade in trades:
        print(trade)


if __name__ == "__main__":
    main()

    log.info("Программа завершена")
