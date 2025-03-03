from datetime import datetime, timedelta
import logging

import pytz


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


        candles = self.marketDataService.get_candles(
            from_date=from_date,
            to_date=to_date,
            instrumentId=instrument.get("uid"),
            interval=self.marketDataService.CandleInterval.ONE_HOUR,
            limit=5000,
        )


        RSI_AVG_14 = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=self.marketDataService.IndicatorType.RSI,
            typeOfPrice=self.marketDataService.TypeOfPrice.AVG,
            length=14,
        )


        SMA_CLOSE_14 = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=self.marketDataService.IndicatorType.SMA,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=50,
        )


        RSI_CLOSE_14 = self.marketDataService.get_tech_analysis(
            from_date=from_date,
            to_date=to_date,
            instrumentUid=instrument.get("uid"),
            interval=self.marketDataService.IndicatorIntervalType.ONE_HOUR,
            indicatorType=self.marketDataService.IndicatorType.RSI,
            typeOfPrice=self.marketDataService.TypeOfPrice.CLOSE,
            length=14,
        )

    

        # Обработка данных
        processed_candles = self.processor.process_candles(candles)
        for candle in processed_candles:
            data[candle["time"]] = candle

        processed_indicators = self.processor.process_indicators(RSI_AVG_14)
        for indicator in processed_indicators:
            data[indicator["time"]]["RSI_AVG_14"] = indicator["value"]

        processed_indicators = self.processor.process_indicators(RSI_CLOSE_14)
        for indicator in processed_indicators:
            data[indicator["time"]]["RSI_CLOSE_14"] = indicator["value"]

        processed_indicators = self.processor.process_indicators(SMA_CLOSE_14)
        for indicator in processed_indicators:
            data[indicator["time"]]["SMA_CLOSE_14"] = indicator["value"]


        if additive_instruments:
            for instrument in additive_instruments:
                log.debug(f'\tДобавляем данные по инструменту: {instrument.get("uid")}')
                additive_candles = self.marketDataService.get_candles(
                    from_date=from_date,
                    to_date=to_date,
                    instrumentId=instrument.get("uid"),
                    interval=self.marketDataService.CandleInterval.ONE_HOUR,
                    limit=5000,
                )

                processed_candles = self.processor.process_candles(additive_candles)
                log.debug(f'\tВсего: {len(processed_candles)}')
                for candle in processed_candles:
                    if candle.get('time') in data:
                        data[candle["time"]][f"{instrument.get('ticker')}_CLOSE"] = candle.get("close")
                        data[candle["time"]][f"{instrument.get('ticker')}_OPEN"] = candle.get("open")
                        data[candle["time"]][f"{instrument.get('ticker')}_VOLUME"] = candle.get("volume")
                

        return [data[key] for key in data]



    def get_data(self, instrument, from_date, to_date, step=100, additive_instruments=None):
        """
        Получение данных за указанный период с заданным шагом.
        """
        log.info(f"Получение данных с шагом {step} дней для инструмента {instrument['ticker']}")
        raw_data = {}

        total_days = (to_date - from_date).days
        log.debug(f"Общее количество дней: {total_days}")

        for start_day in range(0, total_days, step):
            current_from_date = from_date + timedelta(days=start_day)
            current_to_date = min(from_date + timedelta(days=start_day + step), to_date)
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
        
        log.info(f"Получено {len(sorted_data)} записей")

        return sorted_data