from datetime import datetime, timedelta

from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService

from data.fetcher import DataFetcher

from models import Model

from utils.config import SANDBOX_TOKEN
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
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')  # Указываем кодировку UTF-8
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger



log = setup_logger()



def main():
    log.info("Запуск программы")

    TModel = Model()

    TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)

    fetcher = DataFetcher(TService, processor=TModel.processor)
    
    instrumentsService = InstrumentsService(TService)


    response = instrumentsService.find_instrument('RU000A107UL4')
    tbank_instrument = response.get('instruments')[0]
    
    log.info(f"Инструмент найден: {tbank_instrument['ticker']}")

    
    response = instrumentsService.find_instrument('RU000A0JP7K5')
    print(response)

    # data = fetcher.get_data(
    #     instrument=tbank_instrument,
    #     from_date=now() - timedelta(days=8000),
    #     to_date=now()
    # )
    
    # log.info(f"Получено {len(data)} записей для обучения")
    
    # TModel.processor.save_to_json(data, './output/data/tbank_data.json')
    # data = TModel.processor.load_from_json('./output/data/tbank_data.json')

    # TModel.train(data, features=['open', 'high', 'low', 'close', 'volume', 'RSI_AVG_14', 'RSI_CLOSE_14', 'SMA_CLOSE_14', 'IMOEXF_CLOSE', 'IMOEXF_OPEN', 'IMOEXF_VOLUME'])
    # TModel.save('tbank', test=True)

if __name__ == "__main__":
    main()

    log.info("Программа завершена")