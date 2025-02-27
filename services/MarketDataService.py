import json

import requests

from utils.time import now, now_str


class MarketDataService:
    def __init__(self, service):
        service.name = 'MarketDataService'
        self.manager = service

        self.URL = self.manager.get_url()

    def get_candles(
        self,
        from_date,
        to_date=now_str(),
        interval='CANDLE_INTERVAL_DAY',
        instrumentId='string',
        candleSourceType='CANDLE_SOURCE_UNSPECIFIED',
        limit=0,
    ) -> dict:
        """
        Запросить исторические свечи по инструменту
        https://developer.tbank.ru/invest/api/market-data-service-get-candles

        :param from_date:
        :param to_date:
        :param interval:
        :param instrumentId:
        :param candleSourceType:
        :param limit:
        :return:
        """
        path = '/GetCandles'


        return self.manager.session.post(
            url=self.URL + path,
            data=json.dumps(
                dict(
                    from_=from_date,
                    to=to_date,
                    interval=interval,
                    instrumentId=instrumentId,
                    candleSourceType=candleSourceType,
                    limit=limit
                ),
                default=str,
            ),
        ).json()

