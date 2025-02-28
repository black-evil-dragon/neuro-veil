from re import S

from matplotlib import pyplot as plt
from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService
from services.MarketDataService import MarketDataService

from config import SANDBOX_TOKEN

from datetime import timedelta
from utils.time import now, prepare_date


from data.analizer import DataAnalyzer
from data.proccessor import DataProcessor


import logging
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


logging.basicConfig(filename="./neuro.log", format='[%(asctime)s]  %(message)s', filemode='w')
logging.getLogger('neuro').addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger('neuro')
log.setLevel(logging.DEBUG)



def generate_trade_signals(current_price, predicted_price, target_profit=0.02):
    """
    Генерация сигналов КУПИТЬ/ПРОДАТЬ на основе предсказания и целевой прибыли.
    
    :param current_price: Текущая цена актива.
    :param predicted_price: Предсказанная цена актива.
    :param target_profit: Целевая прибыль (по умолчанию 2%).
    :return: Сигнал КУПИТЬ, ПРОДАТЬ или НЕЙТРАЛЬНО.
    """
    if predicted_price > current_price * (1 + target_profit):
        return "КУПИТЬ"
    elif predicted_price < current_price * (1 - target_profit):
        return "ПРОДАТЬ"
    else:
        return "НЕЙТРАЛЬНО"



def main():
    processor = DataProcessor()
    analyzer = DataAnalyzer()


    TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)

    instrumentsService = InstrumentsService(TService)
    marketDataService = MarketDataService(TService)

    response = instrumentsService.find_instrument('RU000A107UL4')
    tbank_instrument = response.get('instruments')[0]



    def fetch_data(from_date, to_date):
        data = {}
        from_date = prepare_date(from_date)
        to_date = prepare_date(to_date)

        log.info('\t- Get candles')
        tbank_candles = marketDataService.get_candles(
            from_date=from_date,
            to_date=to_date,
            instrumentId=tbank_instrument.get('uid'),
            interval=marketDataService.CandleInterval.ONE_HOUR,
            limit=5000,
        )
        log.info(f'\t-\tLength: {len(tbank_candles["candles"])}')


        log.info('\t- Get rsi avg 14')
        tbank_RSI_AVG_14 = marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=tbank_instrument.get('uid'),


            interval=marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=marketDataService.IndicatorType.RSI,
            typeOfPrice=marketDataService.TypeOfPrice.AVG,
            length=14
        )
        log.info(f'\t-\tLength: {len(tbank_RSI_AVG_14["technicalIndicators"])}')


        log.info('\t- Get sma close 50')
        tbank_SMA_CLOSE_14 = marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=tbank_instrument.get('uid'),


            interval=marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=marketDataService.IndicatorType.SMA,
            typeOfPrice=marketDataService.TypeOfPrice.CLOSE,
            length=50
        )
        log.info(f'\t-\tLength: {len(tbank_SMA_CLOSE_14["technicalIndicators"])}')


        log.info('\t- Get rsi close 14')
        tbank_RSI_CLOSE_14 = marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=tbank_instrument.get('uid'),


            interval=marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=marketDataService.IndicatorType.RSI,
            typeOfPrice=marketDataService.TypeOfPrice.CLOSE,
            length=14
        )
        log.info(f'\t-\tLength: {len(tbank_RSI_CLOSE_14["technicalIndicators"])}')


        

        # Обработка данных
        processed_candles = processor.process_candles(tbank_candles)
        for candle in processed_candles:
            data[candle['time']] = candle
        
        processed_indicators = processor.process_indicators(tbank_RSI_AVG_14)
        for indicator in processed_indicators:
            data[indicator['time']]['RSI_AVG_14'] = indicator['value']

        processed_indicators = processor.process_indicators(tbank_RSI_CLOSE_14)
        for indicator in processed_indicators:
            data[indicator['time']]['RSI_CLOSE_14'] = indicator['value']

        processed_indicators = processor.process_indicators(tbank_SMA_CLOSE_14)
        for indicator in processed_indicators:
            data[indicator['time']]['SMA_CLOSE_14'] = indicator['value']


        return [data[key] for key in data]


    raw_data = {}

    
    period = 100
    step = 10

    for day_delta in range(period//step):
        from_day = period - step * day_delta
        to_day = period - step * (day_delta + 1)
    
        from_date = now() - timedelta(days=from_day)
        to_date = now() - timedelta(days=to_day)

        log.info(f'FETCH DATA: {from_date} - {to_date}')

        for line in fetch_data(from_date, to_date):
            raw_data[line['time']] = line


    procced_data = [raw_data[key] for key in raw_data]
    

    # Анализ данных
    df = analyzer.create_dataframe(procced_data)

    plt.plot(df.index, df['close'])
    plt.ylabel('Цена')
    plt.xlabel('Дата')
    plt.show()

    features = ['open', 'high', 'low', 'close', 'volume', 'RSI_AVG_14', 'RSI_CLOSE_14', 'SMA_CLOSE_14']
    df = analyzer.normalize_data(df, features)


    data = df[features].values
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    look_back = 60
    X_train, y_train = analyzer.prepare_lstm_data(train, look_back)
    X_test, y_test = analyzer.prepare_lstm_data(test, look_back)


    model = analyzer.build_lstm_model((look_back, len(features)))
    model.fit(X_train, y_train, batch_size=1, epochs=5)

    # Получение предсказаний
    y_pred = model.predict(X_test)



    # Вычисление средней абсолютной ошибки
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Средняя абсолютная ошибка: {mae}")

    plt.plot(y_test, color='blue', label='Actual')
    plt.plot(y_pred, color='red', label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # main()
    import tensorflow as tf

    # Проверка доступности GPU
    print("list_physical_devices:", tf.config.list_physical_devices())
    
    # from_date = prepare_date(now() - timedelta(days=50))
    # to_date = prepare_date(now() - timedelta(days=25))
    # new_candles = marketDataService.get_candles(
    #     from_date=from_date,
    #     to_date=to_date,
    #     instrumentId=tbank_instrument.get('uid'),
    #     interval=marketDataService.CandleInterval.ONE_HOUR,
    #     limit=10000,
    # )

    # # Обработайте новые свечи
    # processed_new_candles = processor.process_candles(new_candles)
    # df_new = analyzer.create_dataframe(processed_new_candles)
    # df_new = analyzer.normalize_data(df_new, features)

    # # Подготовьте данные для модели
    # new_data = df_new[features].values
    # X_new, y_new = analyzer.prepare_lstm_data(new_data, look_back)

    # # Сделайте прогноз
    # y_new_pred = model.predict(X_new)

    # # Сравните предсказания с реальными значениями
    # for i in range(len(y_new)):
    #     current_price = df_new['close'].iloc[i]
    #     predicted_price = y_new_pred[i][0]
    #     signal = generate_trade_signals(current_price, predicted_price, target_profit=0.02)
    #     print(f"Time: {df_new.index[i]}, Signal: {signal}")

    # # Визуализация
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_new, label='Real')
    # plt.plot(y_new_pred, label='Predicted')
    # plt.legend()
    # plt.title('New Data: Real vs Predicted')
    # plt.show()

    # # Получаем последние данные для предсказания
    # last_data = new_data[-look_back:]  # Последние 60 свечей
    # last_data = last_data.reshape((1, look_back, len(features)))  # Подготовка для модели
    # predicted_price = model.predict(last_data)[0][0]  # Предсказанная цена

    # # Текущая цена (последняя известная цена закрытия)
    # current_price = df['close'].iloc[-1]

    # # Генерация сигнала
    # target_profit = 0.02  # Целевая прибыль (2%)
    # signal = generate_trade_signals(current_price, predicted_price, target_profit)

    # # Вывод сигнала
    # print(f"Текущая цена: {current_price}")
    # print(f"Предсказанная цена: {predicted_price}")
    # print(f"Сигнал: {signal}")



