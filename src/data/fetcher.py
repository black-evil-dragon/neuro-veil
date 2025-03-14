from collections import defaultdict
from datetime import datetime, timedelta
import logging

import pandas as pd
import pytz


from utils.numbers import parse_quotation
from utils.time import prepare_date

from services import InstrumentsService, MarketDataService



log = logging.getLogger('neuro')
log.setLevel(logging.DEBUG)


class DataFetcher:
    raw_data = {}
    period = 10
    step = 1

    def __init__(self, TService, processor=None):
        self.instrumentsService = InstrumentsService(TService)
        self.marketDataService = MarketDataService(TService)

        self.processor = processor



    def fetch_data(self, instrument, from_date, to_date, additive_instruments=None):
        """
        Получение данных (свечи и индикаторы) за указанный период.
        """
        data = {}


        # Дневные свечи
        candles = self.marketDataService.get_candles(
            from_date=from_date,
            to_date=to_date,
            instrumentId=instrument.get("uid"),
            interval=self.marketDataService.CandleInterval.THIRTY_MINUTES,
            limit=1200,
        )
        processed_candles = self.processor.process_candles(candles)
        for candle in processed_candles:
            data[candle["time"]] = candle



        # RSI
        # RSI_CLOSE_14
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date - timedelta(days=14),
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.RSI,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=14,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["RSI_CLOSE_14"] = indicator["value"]


        # --------

        # # BB
        # INDICATORS = self.marketDataService.get_tech_analysis(
        #     from_date=from_date,
        #     to_date=to_date,
        #     instrumentUid=instrument.get("uid"),
        #     interval=self.marketDataService.IndicatorIntervalType.ONE_HOUR,
        #     indicatorType=self.marketDataService.IndicatorType.BB,
        #     typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
        #     length=20,
        # )
        # processed_indicators = self.processor.process_indicators(INDICATORS)
        # for indicator in processed_indicators:
        #     data[indicator["time"]]["BB_CLOSE_20"] = indicator["value"]




        # MACD
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date - timedelta(days=14),
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.MACD,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            smoothing={
                "fastLength": 12,
                "slowLength": 26,
                "signalSmoothing": 9,
            }
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["MACD_CLOSE_12_26_9"] = indicator["value"]



        # EMA
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.EMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=5,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["EMA_CLOSE_5"] = indicator["value"]



        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.EMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=50,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["EMA_CLOSE_50"] = indicator["value"]



        # SMA
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.SMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=5,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["SMA_CLOSE_5"] = indicator["value"]
    
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.SMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=14,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["SMA_CLOSE_14"] = indicator["value"]

        
        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.SMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=50,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["SMA_CLOSE_50"] = indicator["value"]



        INDICATORS = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.THIRTY_MINUTES,
            indicatorType=self.marketDataService.IndicatorType.SMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=100,
        )
        processed_indicators = self.processor.process_indicators(INDICATORS)
        for indicator in processed_indicators:
            if indicator["time"] in data:
                data[indicator["time"]]["SMA_CLOSE_100"] = indicator["value"]

    


        # Обработка данных

        if additive_instruments:
            for instrument in additive_instruments:
                log.debug(f'\tДобавляем данные по инструменту: {instrument.get("ticker")}')
                additive_candles = self.marketDataService.get_candles(
                    from_date=from_date,
                    to_date=to_date,
                    instrumentId=instrument.get("uid"),
                    interval=self.marketDataService.CandleInterval.THIRTY_MINUTES,
                    limit=1200,
                )

                processed_candles = self.processor.process_candles(additive_candles)
                log.debug(f'\tВсего {instrument.get("ticker")}: {len(processed_candles) + len(data)}')
                for candle in processed_candles:
                    if candle.get('time') in data:
                        data[candle["time"]][f"{instrument.get('ticker')}_CLOSE"] = candle.get("close")
                        data[candle["time"]][f"{instrument.get('ticker')}_OPEN"] = candle.get("open")
                        data[candle["time"]][f"{instrument.get('ticker')}_VOLUME"] = candle.get("volume")

                

        candles = [data[key] for key in data]

        return candles



    def get_data(self, instrument, from_date, to_date, step=10, additive_instruments=None):
        """
        Получение данных за указанный период с заданным шагом.
        """
        log.info(f"Получение данных с шагом {step} дней для инструмента {instrument['ticker']}")
        raw_data = {}
        is_hours = False

        total_days = (to_date - from_date).days
        log.debug(f"Общее количество дней: {total_days}")

        if total_days < 1:
            log.debug("Общее количество дней меньше 1, работаем с часами")

            total_days = int((to_date - from_date).total_seconds() // 3600)
            is_hours = True

            log.debug(f"Общее количество часов: {total_days}")

        

        for start_day in range(0, total_days, step):
            if not is_hours:
                current_from_date = from_date + timedelta(days=start_day)
                current_to_date = min(from_date + timedelta(days=start_day + step), to_date)
            else:
                current_from_date = from_date + timedelta(hours=start_day + 1)
                current_to_date = min(from_date + timedelta(hours=start_day + step), to_date)
            log.debug(f"Обработка диапазона: {current_from_date} - {current_to_date}")

            try:
                for line in self.fetch_data(instrument, current_from_date, current_to_date, additive_instruments=additive_instruments):
                    utc_time = datetime.fromisoformat(line["time"].rstrip('Z'))
                    msk_time = utc_time.replace(tzinfo=pytz.UTC).astimezone(pytz.timezone('Europe/Moscow'))
                    line["time"] = msk_time.isoformat()
                    raw_data[msk_time.isoformat()] = line
            except Exception as e:
                log.exception(f"Ошибка при обработке диапазона {current_from_date} - {current_to_date}: {e}")
                continue

        sorted_times = sorted(raw_data.keys(), key=lambda x: datetime.fromisoformat(x))
        sorted_data = [raw_data[time] for time in sorted_times]
        for i in range(1, len(sorted_data)):
            sorted_data[i]['delta'] = (sorted_data[i]['close'] - sorted_data[i-1]['close']) / sorted_data[i]['close']
        
        log.info(f"Получено {len(sorted_data)} записей")

        return sorted_data