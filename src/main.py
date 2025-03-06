from datetime import datetime, timedelta

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

        # "open",
        # "high",
        # "low",
        # "volume",
        # "delta"
    ]

    indicators = [
        # "RSI_CLOSE_7"
        # "SMA_CLOSE_50",
        # "BB_CLOSE_20",

        "RSI_CLOSE_14",
        "SMA_CLOSE_14",
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
        from_date=now() - timedelta(days=2000),
        to_date=now(),
        additive_instruments=[
            moex,
            dollar
        ]
    )

    log.info(f"Получено {len(data)} записей для обучения")
    TModel.processor.save_to_json(data, "./output/data/tbank_test.json")
    

    data = TModel.processor.load_from_json('./output/data/tbank_test.json')

    TModel.train(data=data[:-250], features=features)


    TModel.save("tbank", test=True)

    TModel.load(name='tbank', extension='keras', test=True)


    TModel.predict(data=data[-250:], features=features)


if __name__ == "__main__":
    main()

    log.info("Программа завершена")
